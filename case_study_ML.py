# ============================================================
# case_study_ML.py —— 第三章 ML 路线案例分析图与数值导出
# ============================================================
#
# 目的：从 mesa_ML.py 重跑产出的 ML_results/ 中挑选有诊断意义的窗口
#       （RF 高置信正例 / RF 漏检事件），绘制 PPG + SpO2 双子图，
#       并打印关键统计量，供论文 3.5.2 / 3.5.3 节直接引用。
#
# 输入（必须先运行新版 mesa_ML.py 生成）：
#   - ML_results/windows_index.csv      （跨 5 折窗口级总索引）
#   - ML_results/subjects_index.json    （受试者级元信息：AHI / 睡眠时窗 / 事件列表）
#   - shared_subset/mesa_quality7_raw_ppg_spo2_flow/*.npz   （原始信号）
#   - shared_subset/mesa_quality7_xml/*.xml                  （事件类型标注）
#
# 输出：
#   - ML_results/case_studies/pos_*.png      （RF 高置信正例双子图）
#   - ML_results/case_studies/neg_*.png      （RF 漏检事件双子图）
#   - ML_results/case_studies/case_summary.csv  （所有挑出案例的关键数值）
#
# 用法：
#   python case_study_ML.py
#   python case_study_ML.py --n-pos 3 --n-neg 3
#   python case_study_ML.py --clf SVM_RBF
# ============================================================

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_NPZ_DIR     = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_raw_ppg_spo2_flow"
DEFAULT_XML_DIR     = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_xml"
DEFAULT_RESULTS_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/ML_results"
DEFAULT_OUT_DIR     = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/ML_results/case_studies"

WIN_SIZE = 60
CLF_DEFAULT = "RandomForest"


# ── 案例挑选 ──────────────────────────────────────────────────────────────────

def load_csv_records(csv_path):
    """读 windows_index.csv 并把数值列转回原类型。"""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["fold"]    = int(r["fold"])
            r["t_start"] = int(r["t_start"])
            r["ev_idx"]  = int(r["ev_idx"])
            r["y_true"]  = int(r["y_true"])
            for k in list(r.keys()):
                if k.endswith("_pred"):
                    r[k] = int(r[k])
                elif k.endswith("_proba"):
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def pick_cases(records, n_pos, n_neg, clf):
    """挑案例：
       - 正案例（POS）：y_true=1 且 <clf>_proba 最高的若干条
       - 漏检（NEG）  ：y_true=1 且 <clf>_proba 最低的若干条
    """
    proba_key = f"{clf}_proba"
    if not records or proba_key not in records[0]:
        raise KeyError(f"windows_index.csv 中未找到列 {proba_key}，"
                       f"请确认 --clf 名称（如 RandomForest / SVM_RBF / DecisionTree）")

    pos_truth = [r for r in records if r["y_true"] == 1]
    if not pos_truth:
        raise RuntimeError("找不到任何 y_true=1 的窗口")

    pos_cases = sorted(pos_truth, key=lambda r: -r[proba_key])[:n_pos]
    neg_cases = sorted(pos_truth, key=lambda r:  r[proba_key])[:n_neg]
    return pos_cases, neg_cases


# ── XML 事件类型解析（与 prep_mesa_v2.parse_events_v2 同一规则但额外保留 kind） ──

def parse_event_kinds(xml_path):
    """复刻 parse_events_v2 的过滤规则，返回与其 events 列表同序的 kind 标签。

    返回:
        kinds: list of str，元素 ∈ {"ObstructiveApnea", "Hypopnea", "Unsure"}，
               长度与 parse_events_v2 输出的 events 长度一致。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    scored_events = list(root.findall(".//ScoredEvent"))

    kinds = []
    for i, ev in enumerate(scored_events):
        c_lower = (ev.findtext("EventConcept") or "").strip().lower()

        if "obstructive apnea" in c_lower:
            kinds.append("ObstructiveApnea")
            continue

        if "hypopnea" in c_lower or "unsure" in c_lower:
            # 紧邻下一条须为 Arousal / SpO2 desaturation 才确认
            if i + 1 < len(scored_events):
                nxt = (scored_events[i + 1].findtext("EventConcept") or "").strip().lower()
                if "arousal" in nxt or "spo2 desaturation" in nxt:
                    kinds.append("Hypopnea" if "hypopnea" in c_lower else "Unsure")
    return kinds


# ── 信号读取 + 绘图 ──────────────────────────────────────────────────────────

def load_signal(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        ppg     = z["ppg"].astype("float32")
        spo2    = z["spo2"].astype("float32")
        fs_ppg  = float(z["fs_ppg"][0])
        fs_spo2 = float(z["fs_spo2"][0])
    return ppg, spo2, fs_ppg, fs_spo2


def slice_window(signal, fs, t_start, win_size):
    s = int(round(t_start * fs))
    e = int(round((t_start + win_size) * fs))
    return signal[s:e]


def plot_case(case, npz_path, events, event_kinds, out_path, title_prefix, clf):
    """画 PPG + SpO2 双子图，事件区段以阴影高亮。返回该案例的关键统计量 dict。"""
    ppg, spo2, fs_ppg, fs_spo2 = load_signal(npz_path)

    t_start = case["t_start"]
    ppg_win  = slice_window(ppg,  fs_ppg,  t_start, WIN_SIZE)
    spo2_win = slice_window(spo2, fs_spo2, t_start, WIN_SIZE)

    # 解析事件相对于窗口起始的位置
    ev_idx = case["ev_idx"]
    ev_rel_start = ev_rel_end = None
    ev_kind = "Negative"
    ev_duration = None
    if ev_idx >= 0 and ev_idx < len(events):
        ev_s, ev_d = events[ev_idx]
        ev_rel_start = max(0.0, ev_s - t_start)
        ev_rel_end   = min(float(WIN_SIZE), ev_s + ev_d - t_start)
        ev_duration  = float(ev_d)
        ev_kind = event_kinds[ev_idx] if ev_idx < len(event_kinds) else "Unknown"

    # 关键统计
    spo2_min = float(np.min(spo2_win)) if len(spo2_win) else float("nan")
    spo2_max = float(np.max(spo2_win)) if len(spo2_win) else float("nan")
    spo2_drop = spo2_max - spo2_min if np.isfinite(spo2_min) and np.isfinite(spo2_max) else float("nan")

    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    t_ppg = np.arange(len(ppg_win)) / fs_ppg
    axes[0].plot(t_ppg, ppg_win, color="#2271b3", linewidth=0.6)
    axes[0].set_ylabel("PPG amplitude")
    proba_key = f"{clf}_proba"
    proba = case.get(proba_key, float("nan"))
    axes[0].set_title(
        f"{title_prefix}  subj={case['subject_id']}  fold={case['fold']}  "
        f"t_start={t_start}s  event={ev_kind}  "
        f"{clf}_proba={proba:.3f}"
    )
    if ev_rel_start is not None and ev_rel_end > ev_rel_start:
        axes[0].axvspan(ev_rel_start, ev_rel_end, alpha=0.20, color="#d62728",
                        label=f"event ({ev_kind})")
        axes[0].legend(loc="upper right", fontsize=8)

    t_spo2 = np.arange(len(spo2_win)) / fs_spo2
    axes[1].plot(t_spo2, spo2_win, color="#2ca02c", linewidth=1.2)
    axes[1].set_ylabel("SpO$_2$ (%)")
    axes[1].set_xlabel("Time within window (s)")
    if ev_rel_start is not None and ev_rel_end > ev_rel_start:
        axes[1].axvspan(ev_rel_start, ev_rel_end, alpha=0.20, color="#d62728")
    if np.isfinite(spo2_min):
        axes[1].set_ylim(min(80.0, spo2_min - 1.0), 102.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    coverage_pct = ((ev_rel_end - ev_rel_start) / WIN_SIZE * 100.0
                    if ev_rel_start is not None else None)

    return {
        "case_type":            title_prefix,
        "subject_id":           case["subject_id"],
        "fold":                 case["fold"],
        "t_start_s":            t_start,
        "ev_idx":               ev_idx,
        "event_kind":           ev_kind,
        "event_total_dur_s":    ev_duration,
        "event_rel_start_s":    None if ev_rel_start is None else round(float(ev_rel_start), 2),
        "event_rel_end_s":      None if ev_rel_end is None   else round(float(ev_rel_end), 2),
        "event_coverage_pct":   None if coverage_pct is None else round(float(coverage_pct), 2),
        "spo2_min":             round(spo2_min, 2),
        "spo2_max":             round(spo2_max, 2),
        "spo2_drop_pct":        round(float(spo2_drop), 2) if np.isfinite(spo2_drop) else None,
        "y_true":               case["y_true"],
        "RandomForest_proba":   round(case.get("RandomForest_proba", float("nan")), 4),
        "SVM_RBF_proba":        round(case.get("SVM_RBF_proba",      float("nan")), 4),
        "DecisionTree_proba":   round(case.get("DecisionTree_proba", float("nan")), 4),
        "saved_to":             out_path,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(npz_dir, xml_dir, results_dir, out_dir, n_pos, n_neg, clf):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "windows_index.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"未找到 {csv_path} —— 请先运行新版 mesa_ML.py 生成 ML_results/")
    subj_idx_path = os.path.join(results_dir, "subjects_index.json")
    if not os.path.exists(subj_idx_path):
        raise FileNotFoundError(f"未找到 {subj_idx_path}")

    print(f"读取 {csv_path}")
    records = load_csv_records(csv_path)
    print(f"  共 {len(records)} 行 ({sum(1 for r in records if r['y_true']==1)} 正 / "
          f"{sum(1 for r in records if r['y_true']==0)} 负)")

    with open(subj_idx_path, encoding="utf-8") as f:
        subj_idx = json.load(f)
    subj_to_events = dict(zip(subj_idx["subject_ids"], subj_idx["events"]))
    subj_to_ahi    = dict(zip(subj_idx["subject_ids"], subj_idx["ahi"]))

    print(f"\n按 {clf} 的预测概率挑案例：top-{n_pos} 正例 + bottom-{n_neg} 漏检\n")
    pos_cases, neg_cases = pick_cases(records, n_pos=n_pos, n_neg=n_neg, clf=clf)

    summary_rows = []

    def _process(case_list, label):
        for i, c in enumerate(case_list):
            subj = c["subject_id"]
            npz_path = os.path.join(npz_dir, f"mesa_{subj}_raw_ppg_spo2_flow.npz")
            xml_path = os.path.join(xml_dir, f"mesa-sleep-{subj}-nsrr.xml")

            if not os.path.exists(npz_path):
                print(f"  [{label}-{i+1}] subj={subj}: NPZ 缺失，跳过")
                continue

            events = [tuple(e) for e in subj_to_events.get(subj, [])]
            event_kinds = parse_event_kinds(xml_path) if os.path.exists(xml_path) else []

            tag = f"{label}-{i+1}"
            png_name = f"{label.lower()}_{i+1}_subj{subj}_t{c['t_start']}.png"
            out_png = os.path.join(out_dir, png_name)

            info = plot_case(c, npz_path, events, event_kinds, out_png,
                             title_prefix=tag, clf=clf)
            info["subject_ahi"] = round(subj_to_ahi.get(subj, float("nan")), 2)
            summary_rows.append(info)
            print(f"  [{tag}] subj={subj} AHI={info['subject_ahi']} "
                  f"t={c['t_start']}s  {clf}_proba={c.get(f'{clf}_proba'):.3f}  "
                  f"event={info['event_kind']} → {png_name}")

    print("【正案例】")
    _process(pos_cases, "POS")

    print("\n【漏检案例】")
    _process(neg_cases, "NEG")

    # 案例汇总 CSV
    if summary_rows:
        summary_csv = os.path.join(out_dir, "case_summary.csv")
        fieldnames = list(summary_rows[0].keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n汇总 → {summary_csv}")

    # 友好打印（供论文 3.5.2 / 3.5.3 节复制）
    print("\n" + "=" * 78)
    print("关键数值速览（供论文 3.5.2 / 3.5.3 节直接引用）")
    print("=" * 78)
    for row in summary_rows:
        print(f"\n[{row['case_type']}] subj {row['subject_id']}  AHI={row['subject_ahi']}")
        print(f"  事件类型 = {row['event_kind']}  事件总时长 = {row['event_total_dur_s']} s")
        if row['event_rel_start_s'] is not None:
            print(f"  事件相对窗口位置 = [{row['event_rel_start_s']}, {row['event_rel_end_s']}] s "
                  f"({row['event_coverage_pct']}% 覆盖)")
        print(f"  SpO2:  min = {row['spo2_min']}%   max = {row['spo2_max']}%   "
              f"drop = {row['spo2_drop_pct']} %")
        print(f"  各分类器 proba: RF={row['RandomForest_proba']}  "
              f"SVM={row['SVM_RBF_proba']}  DT={row['DecisionTree_proba']}")
        print(f"  图: {row['saved_to']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML 路线案例分析：从 windows_index.csv 挑出 RF 高置信正例 / 漏检事件，绘图并打印关键数值",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz-dir",     default=DEFAULT_NPZ_DIR)
    parser.add_argument("--xml-dir",     default=DEFAULT_XML_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir",     default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-pos",       type=int, default=3,
                        help="正案例数量（按 RF_proba 从高到低）")
    parser.add_argument("--n-neg",       type=int, default=3,
                        help="漏检案例数量（按 RF_proba 从低到高，y_true=1）")
    parser.add_argument("--clf",         default=CLF_DEFAULT,
                        choices=["RandomForest", "SVM_RBF", "DecisionTree"],
                        help="按哪个分类器的预测概率挑案例")
    args = parser.parse_args()

    main(npz_dir=args.npz_dir, xml_dir=args.xml_dir,
         results_dir=args.results_dir, out_dir=args.out_dir,
         n_pos=args.n_pos, n_neg=args.n_neg, clf=args.clf)
