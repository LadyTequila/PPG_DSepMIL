# ============================================================
# mesa_ML.py —— 机器学习基线方法
# ============================================================
#
# 定位：作为第三章"特征工程研究"，与第四章深度学习形成对照。
#
# 任务定义（与 DL 不同）：
#   - DL：从连续信号上滑窗，逐秒预测事件（检测任务）
#   - ML：给定一个候选窗口（事件中心对齐 or 随机非事件段），
#         判断该窗口是否含有呼吸事件（闭集分类任务）
#
# 数据预处理：
#   - 直接复用 prep_mesa_v2.py 中的函数（睡眠时窗 + 事件过滤）
#   - 窗口生成策略改为"事件中心对齐"：
#       * 正样本：每个已确认事件的中心对齐 60s 窗口
#       * 负样本：睡眠时窗内随机采样、与所有事件零重叠的 60s 窗口
#                （数量按 1:1 与正样本平衡）
#
# 特征：
#   - 复用 ppg_extraction 得到 (7, 60) 的形态特征
#   - 对每个通道做统计聚合（mean/std/min/max/median/p25/p75）
#   - 最终每个窗口得到 7×7 = 49 维一维特征向量
#
# 分类器：SVM(RBF) / DecisionTree / RandomForest
#
# 评估：与 DL 保持一致的 5 折 AHI 分层交叉验证，
#       同一批受试者、同一套 fold 划分（random_state=42）。
#
# 用法：
#   python mesa_ML.py
#   python mesa_ML.py --out-dir ./ml_results
# ============================================================

import argparse
import csv
import glob
import json
import os
import pickle
import re
import time
import warnings
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# 复用 v2 的预处理逻辑
from prep_mesa_v2 import (MIN_SPO2, N_FOLDS, AHI_THRESHOLD,
                          get_sleep_window, parse_events_v2,
                          ppg_extraction, compute_ahi)

warnings.filterwarnings("ignore")

DEFAULT_NPZ_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_raw_ppg_spo2_flow"
DEFAULT_XML_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_xml"
DEFAULT_OUT_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/ML_results"

WIN_SIZE         = 60    # 窗口大小（秒），与 DL 一致
NEG_POS_RATIO    = 1.0   # 负样本:正样本比例（1.0 表示平衡）
MAX_NEG_TRIES    = 50    # 每个负样本的最大采样尝试次数
RANDOM_SEED      = 42    # 与 prep_mesa_v2 保持一致


# ── 特征聚合 ──────────────────────────────────────────────────────────────────

def aggregate_features(feat_2d):
    """
    将 (7, RESAMPLE_N) 的形态特征聚合成一维统计向量。

    对每个通道计算 7 个统计量：mean / std / min / max / median / p25 / p75
    最终返回 7 * 7 = 49 维向量。
    """
    n_ch = feat_2d.shape[0]
    stats = np.empty((n_ch, 7), dtype=np.float32)
    for ch in range(n_ch):
        x = feat_2d[ch]
        x = x[np.isfinite(x)] if np.any(~np.isfinite(x)) else x
        if x.size == 0:
            stats[ch] = 0.0
            continue
        stats[ch, 0] = np.mean(x)
        stats[ch, 1] = np.std(x)
        stats[ch, 2] = np.min(x)
        stats[ch, 3] = np.max(x)
        stats[ch, 4] = np.median(x)
        stats[ch, 5] = np.percentile(x, 25)
        stats[ch, 6] = np.percentile(x, 75)
    return stats.flatten()


# ── 窗口生成 ──────────────────────────────────────────────────────────────────

def has_event_overlap(t_start, t_end, events):
    """判断 [t_start, t_end) 是否与任何事件有交集（0 秒也算重叠）。"""
    for (ev_s, ev_d) in events:
        ev_e = ev_s + ev_d
        if ev_s < t_end and ev_e > t_start:
            return True
    return False


def extract_one_window(ppg, spo2, fs_ppg, fs_spo2, hz, t_start, win_size):
    """提取 [t_start, t_start+win_size) 的 PPG 特征，失败返回 None。"""
    spo2_start = int(t_start * fs_spo2)
    spo2_end   = int((t_start + win_size) * fs_spo2)
    spo2_win   = spo2[spo2_start:spo2_end]
    if len(spo2_win) == 0 or np.min(spo2_win) < MIN_SPO2:
        return None

    ppg_start = int(t_start * fs_ppg)
    ppg_end   = int((t_start + win_size) * fs_ppg)
    ppg_win   = ppg[ppg_start:ppg_end]
    if len(ppg_win) < win_size * hz:
        return None

    try:
        feat = ppg_extraction(ppg_win, hz=hz, win_size=win_size)
    except Exception:
        return None

    return feat


def process_subject(npz_path, xml_path, win_size, rng):
    """
    返回 dict（无有效窗口时返回 None）：
        X            : (n_samples, n_features) 聚合特征矩阵
        y            : (n_samples,) 0/1 标签
        meta         : (n_samples, 2) int 矩阵，[[t_start, ev_idx], ...]
                       负样本 ev_idx = -1
        events       : list of (ev_s, ev_d) AASM 确认事件，与 ev_idx 对齐
        sleep_onset  : int
        sleep_offset : int
        ahi          : float
        n_pos / n_neg
    """
    with np.load(npz_path, allow_pickle=True) as z:
        ppg     = z["ppg"].astype("float32")
        spo2    = z["spo2"].astype("float32")
        fs_ppg  = float(z["fs_ppg"][0])
        fs_spo2 = float(z["fs_spo2"][0])

    hz = int(round(fs_ppg))

    sleep_window = get_sleep_window(xml_path)
    if sleep_window is None:
        return None

    sleep_onset, sleep_offset = sleep_window
    sleep_duration_sec = sleep_offset - sleep_onset

    events, event_stats = parse_events_v2(xml_path)
    ahi = compute_ahi(event_stats["total_events"], sleep_duration_sec)

    half = win_size / 2.0

    # ── 正样本：事件中心对齐 ──
    pos_features = []
    pos_meta = []  # [(t_start, ev_idx), ...]
    for ev_idx, (ev_s, ev_d) in enumerate(events):
        ev_center = ev_s + ev_d / 2.0
        t_start = int(round(ev_center - half))
        t_end = t_start + win_size

        # 必须完全落在睡眠时窗内
        if t_start < sleep_onset or t_end > sleep_offset:
            continue

        feat = extract_one_window(ppg, spo2, fs_ppg, fs_spo2, hz, t_start, win_size)
        if feat is None:
            continue

        pos_features.append(aggregate_features(feat))
        pos_meta.append((t_start, ev_idx))

    n_pos = len(pos_features)
    if n_pos == 0:
        return None

    # ── 负样本：随机采样，与事件零重叠 ──
    n_neg_target = int(round(n_pos * NEG_POS_RATIO))
    neg_features = []
    neg_meta = []  # [(t_start, -1), ...]
    t_min = int(sleep_onset)
    t_max = int(sleep_offset) - win_size

    tries = 0
    while len(neg_features) < n_neg_target and tries < n_neg_target * MAX_NEG_TRIES:
        tries += 1
        t_start = int(rng.integers(t_min, t_max + 1))
        t_end = t_start + win_size

        if has_event_overlap(t_start, t_end, events):
            continue

        feat = extract_one_window(ppg, spo2, fs_ppg, fs_spo2, hz, t_start, win_size)
        if feat is None:
            continue

        neg_features.append(aggregate_features(feat))
        neg_meta.append((t_start, -1))

    if len(neg_features) == 0:
        return None

    X_pos = np.stack(pos_features, axis=0)
    X_neg = np.stack(neg_features, axis=0)
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([
        np.ones(len(pos_features), dtype=np.int64),
        np.zeros(len(neg_features), dtype=np.int64),
    ])
    meta = np.array(pos_meta + neg_meta, dtype=np.int64)  # (n_samples, 2)

    return {
        "X": X,
        "y": y,
        "meta": meta,
        "events": [(float(es), float(ed)) for es, ed in events],
        "sleep_onset": int(sleep_onset),
        "sleep_offset": int(sleep_offset),
        "ahi": float(ahi),
        "n_pos": n_pos,
        "n_neg": len(neg_features),
    }


# ── 评估指标 ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_score=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "acc": accuracy_score(y_true, y_pred) * 100.0,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100.0,
        "sens": sens * 100.0,
        "spec": spec * 100.0,
    }
    if y_score is not None:
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_score) * 100.0
        except ValueError:
            metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")
    return metrics


def summarize(metric_list):
    """将 list of dict 汇总为 mean±std 字符串。"""
    keys = list(metric_list[0].keys())
    result = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            result[k] = "N/A"
        else:
            result[k] = f"{vals.mean():.2f}±{vals.std():.2f}"
    return result


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(npz_dir, xml_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== ML 基线方法（事件中心对齐窗口 + 统计特征 + 经典分类器）===")
    print(f"窗口大小: {WIN_SIZE}s | 负:正比例: {NEG_POS_RATIO:.1f} | 5 折 AHI 分层 CV")
    print(f"输出目录: {out_dir}\n")

    npz_files = sorted(glob.glob(os.path.join(npz_dir, "mesa_*_raw_ppg_spo2_flow.npz")))
    if not npz_files:
        raise FileNotFoundError(f"在 {npz_dir} 下未找到任何 NPZ 文件")

    print(f"找到 {len(npz_files)} 个受试者，开始提取特征...\n")

    rng = np.random.default_rng(RANDOM_SEED)

    # 每个受试者一个特征矩阵 + 元信息
    subject_X    = []
    subject_y    = []
    subject_meta = []   # 每个受试者的 (n_win, 2) 矩阵：[t_start, ev_idx]
    subject_events = [] # 每个受试者的 AASM 确认事件列表 [(ev_s, ev_d), ...]
    subject_sleep_window = []  # [(sleep_onset, sleep_offset), ...]
    subject_ahi  = []
    subject_ids  = []

    total_pos = 0
    total_neg = 0

    for npz_path in tqdm(npz_files, desc="处理受试者"):
        match = re.search(r"mesa_(\d{4})_raw", os.path.basename(npz_path))
        if not match:
            continue
        subj_id = match.group(1)

        xml_path = os.path.join(xml_dir, f"mesa-sleep-{subj_id}-nsrr.xml")
        if not os.path.exists(xml_path):
            continue

        result = process_subject(npz_path, xml_path, win_size=WIN_SIZE, rng=rng)
        if result is None or len(result["X"]) == 0:
            tqdm.write(f"  {subj_id}: 无有效窗口，跳过")
            continue

        subject_X.append(result["X"])
        subject_y.append(result["y"])
        subject_meta.append(result["meta"])
        subject_events.append(result["events"])
        subject_sleep_window.append((result["sleep_onset"], result["sleep_offset"]))
        subject_ahi.append(result["ahi"])
        subject_ids.append(subj_id)
        total_pos += result["n_pos"]
        total_neg += result["n_neg"]

    n_subjects = len(subject_ids)
    print(f"\n有效受试者: {n_subjects}")
    print(f"总样本: {total_pos} 正 / {total_neg} 负 = {total_pos + total_neg} 个窗口\n")

    if n_subjects == 0:
        raise RuntimeError("没有任何有效受试者")

    # ── 受试者级元信息持久化（供 case study 脚本读取） ──
    subjects_index_path = os.path.join(out_dir, "subjects_index.json")
    with open(subjects_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "subject_ids": subject_ids,
            "ahi": [float(a) for a in subject_ahi],
            "sleep_windows": [[int(a), int(b)] for a, b in subject_sleep_window],
            "events": [[[float(es), float(ed)] for es, ed in evs]
                       for evs in subject_events],
            "n_pos_per_subject": [int((y == 1).sum()) for y in subject_y],
            "n_neg_per_subject": [int((y == 0).sum()) for y in subject_y],
        }, f, indent=2, ensure_ascii=False)
    print(f"受试者元信息 → {subjects_index_path}")

    # ── 5 折 AHI 分层 CV（和 DL 一致）──
    ahi_arr = np.array(subject_ahi)
    ahi_severe = (ahi_arr > AHI_THRESHOLD).astype(int)
    subject_idx = np.arange(n_subjects)

    skf = StratifiedShuffleSplit(n_splits=N_FOLDS, random_state=RANDOM_SEED, test_size=0.10)

    # ── 分类器定义 ──
    def make_classifiers():
        return {
            "SVM_RBF":      SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                random_state=RANDOM_SEED, class_weight="balanced"),
            "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_SEED,
                                                   class_weight="balanced"),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                                   class_weight="balanced", n_jobs=-1),
        }

    # 每个分类器每折的指标 + 每折的 subject-level AHI 预测用于 pearson r
    fold_metrics = defaultdict(list)   # {clf_name: [metrics_fold0, ...]}
    fold_ahi_pairs = defaultdict(list) # {clf_name: [(true_ahi, pred_ahi), ...]} across subjects

    # 跨折累积的窗口级预测，供最后导出 windows_index.csv
    master_records = []

    fold_id = 0
    for train_idx, test_idx in skf.split(subject_idx, ahi_severe):
        print(f"=== Fold {fold_id} ===")
        print(f"  train subjects: {len(train_idx)}  test subjects: {len(test_idx)}")

        X_train = np.concatenate([subject_X[i] for i in train_idx], axis=0)
        y_train = np.concatenate([subject_y[i] for i in train_idx], axis=0)
        X_test  = np.concatenate([subject_X[i] for i in test_idx],  axis=0)
        y_test  = np.concatenate([subject_y[i] for i in test_idx],  axis=0)

        # 测试集对应的窗口级元信息（subject_id, t_start, ev_idx），按 test_idx 拼接顺序对齐
        test_subject_id_per_win = np.concatenate([
            np.full(len(subject_y[i]), subject_ids[i], dtype=object)
            for i in test_idx
        ])
        test_meta = np.concatenate([subject_meta[i] for i in test_idx], axis=0)
        # test_meta[:, 0] 是 t_start，test_meta[:, 1] 是 ev_idx

        print(f"  train samples: {len(X_train)}  test samples: {len(X_test)}")

        # 标准化（对树模型无影响，但 SVM 必须）
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        classifiers = make_classifiers()
        n_clf = len(classifiers)

        # 收集本折所有分类器的预测，供持久化
        fold_pred = {}    # {clf_name: y_pred}
        fold_score = {}   # {clf_name: y_score}

        for ci, (clf_name, clf) in enumerate(classifiers.items(), start=1):
            print(f"  [{ci}/{n_clf}] {clf_name:13s} 训练中...", end="", flush=True)
            t0 = time.time()
            clf.fit(X_train_s, y_train)
            t_fit = time.time() - t0

            print(f" 完成 fit({t_fit:.1f}s)，预测中...", end="", flush=True)
            t0 = time.time()
            y_pred = clf.predict(X_test_s)

            if hasattr(clf, "predict_proba"):
                y_score = clf.predict_proba(X_test_s)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_score = clf.decision_function(X_test_s)
            else:
                y_score = None
            t_pred = time.time() - t0
            print(f" 完成 predict({t_pred:.1f}s)")

            m = compute_metrics(y_test, y_pred, y_score)
            fold_metrics[clf_name].append(m)

            fold_pred[clf_name] = y_pred.astype(np.int64)
            if y_score is not None:
                fold_score[clf_name] = np.asarray(y_score, dtype=np.float64)

            # 受试者级 AHI 估计（用于 Pearson r）：用测试窗口的阳性率近似
            offset = 0
            for si in test_idx:
                n_win = len(subject_y[si])
                y_pred_subj = y_pred[offset:offset + n_win]
                offset += n_win
                pred_pos_rate = float(np.mean(y_pred_subj))
                true_ahi = float(subject_ahi[si])
                fold_ahi_pairs[clf_name].append((true_ahi, pred_pos_rate))

            print(f"      结果: Acc={m['acc']:.2f}  F1={m['f1']:.2f}  "
                  f"Sens={m['sens']:.2f}  Spec={m['spec']:.2f}  AUROC={m['auroc']:.2f}")

        # ── 本折产物落盘 ──
        fold_dir = os.path.join(out_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        # 模型 + scaler
        models_to_save = {"scaler": scaler}
        for clf_name, clf in classifiers.items():
            models_to_save[clf_name] = clf
        with open(os.path.join(fold_dir, "models.pkl"), "wb") as f:
            pickle.dump(models_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 预测 NPZ：窗口级 (subject_id, t_start, ev_idx, y_true) + 三个分类器的 pred / score
        npz_data = {
            "test_subject_ids": np.asarray(test_subject_id_per_win, dtype=object),
            "test_t_starts": test_meta[:, 0].astype(np.int64),
            "test_ev_idxs": test_meta[:, 1].astype(np.int64),
            "test_y_true": y_test.astype(np.int64),
            "test_subject_idx_in_fold": np.array(test_idx, dtype=np.int64),
        }
        for clf_name, arr in fold_pred.items():
            npz_data[f"y_pred_{clf_name}"] = arr
        for clf_name, arr in fold_score.items():
            npz_data[f"y_score_{clf_name}"] = arr
        np.savez(os.path.join(fold_dir, "predictions.npz"), **npz_data)

        print(f"  → 模型保存至 {fold_dir}/models.pkl")
        print(f"  → 预测保存至 {fold_dir}/predictions.npz")

        # 累积到 master_records 用于最终 CSV
        for i in range(len(y_test)):
            rec = {
                "subject_id": str(test_subject_id_per_win[i]),
                "fold": fold_id,
                "t_start": int(test_meta[i, 0]),
                "ev_idx": int(test_meta[i, 1]),
                "y_true": int(y_test[i]),
            }
            for clf_name in fold_pred:
                rec[f"{clf_name}_pred"] = int(fold_pred[clf_name][i])
            for clf_name in fold_score:
                rec[f"{clf_name}_proba"] = float(fold_score[clf_name][i])
            master_records.append(rec)

        print()
        fold_id += 1

    # ── 汇总报告 ──
    print("=" * 78)
    print("汇总结果（均值±标准差，跨 5 折）")
    print("=" * 78)
    print(f"{'Classifier':<14} {'Acc':>12} {'F1':>12} {'Sens':>12} {'Spec':>12} {'AUROC':>12} {'PearsonR':>10}")
    print("-" * 78)

    summary = {}
    for clf_name, mlist in fold_metrics.items():
        s = summarize(mlist)
        pairs = fold_ahi_pairs[clf_name]
        if len(pairs) >= 3:
            trues = np.array([p[0] for p in pairs])
            preds = np.array([p[1] for p in pairs])
            try:
                r, _ = pearsonr(trues, preds)
            except Exception:
                r = float("nan")
        else:
            r = float("nan")
        pearson_str = f"{r:.4f}" if np.isfinite(r) else "N/A"
        summary[clf_name] = {**s, "pearson_r": pearson_str}
        print(f"{clf_name:<14} {s['acc']:>12} {s['f1']:>12} {s['sens']:>12} "
              f"{s['spec']:>12} {s['auroc']:>12} {pearson_str:>10}")

    # 保存 json
    out_json = os.path.join(out_dir, "ML__results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "win_size": WIN_SIZE,
                "neg_pos_ratio": NEG_POS_RATIO,
                "n_folds": N_FOLDS,
                "random_seed": RANDOM_SEED,
                "n_subjects": n_subjects,
                "n_positive_windows": total_pos,
                "n_negative_windows": total_neg,
            },
            "summary": summary,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n聚合指标 → {out_json}")

    # ── 窗口级总索引 CSV，便于 case study 查询 ──
    csv_path = os.path.join(out_dir, "windows_index.csv")
    if master_records:
        # 字段顺序：基础字段在前，分类器字段按名称排序在后
        base_fields = ["subject_id", "fold", "t_start", "ev_idx", "y_true"]
        clf_fields = sorted(
            k for k in master_records[0].keys() if k not in base_fields
        )
        fieldnames = base_fields + clf_fields
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(master_records)
        print(f"窗口级索引 → {csv_path} （共 {len(master_records)} 行）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML 基线方法：事件中心对齐窗口 + 统计特征 + SVM/决策树/随机森林",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR)
    parser.add_argument("--xml-dir", default=DEFAULT_XML_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    main(npz_dir=args.npz_dir, xml_dir=args.xml_dir, out_dir=args.out_dir)
