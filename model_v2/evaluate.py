# python evaluate.py 2>&1 | tee logs/evaluate_delay0_win60_soft0_result.log
import csv
import hydra
import json
import logging
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr

from runner import load_dataset, evaluate_onset, evaluate_severity, forward, bland_altman_plot, get_ahis, make_classif_y
from models import MODELS
from utils import config_gpu

log = logging.getLogger(__name__)


def _load_t_starts(dataset_dir, dataset_name, fold):
    """加载 prep_mesa_v2.py 输出的 t_starts pickle（list-of-arrays，与 x_test 同结构）。
    若文件不存在（旧版预处理产物）则返回 None。"""
    path = os.path.join(dataset_dir, f"{dataset_name}_fold{fold}_t_starts_test.pickle")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_subjects_index(dataset_dir):
    """加载 subjects_index.json，没有则返回 None。"""
    path = os.path.join(dataset_dir, "subjects_index.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_windowing_config(dataset_dir):
    """从 ra_studies/MESA_ablation/*/config.json 读取 win_size / stride。

    避免 C7 短窗等非默认配置（如 driven_30s: 30/5）评估时
    AHI 估计公式仍按 stride=30 错算，导致 Pearson r 假性下降。

    Returns
    -------
    (win_size, stride) : tuple[int, int]
        如目录下无 config.json（毕设旧产物 MESA_v2 / MESA_enhance / MESA）
        则返回 v2 默认值 (60, 30)，保持向后兼容。
    """
    path = os.path.join(dataset_dir, "config.json")
    if not os.path.exists(path):
        return 60, 30
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        win_size = int(cfg["components"]["C7_windowing"]["win_size"])
        stride = int(cfg["components"]["C7_windowing"]["stride"])
        return win_size, stride
    except (KeyError, TypeError, json.JSONDecodeError):
        return 60, 30


@hydra.main(version_base="1.2", config_path="config/", config_name="evaluate")
def main(config):
    # 根据 exp_tag 定位对应的子目录
    exp_tag = config.exp_tag
    log_dir    = os.path.join(config.log_dir,    exp_tag)
    weight_dir = os.path.join(config.weight_dir, exp_tag)

    # 将评估日志写入文件（若使用命令行重定向保存日志则注释掉此段，避免内容重叠）
    # log_file = os.path.join(config.log_dir, f"evaluate_result_{exp_tag}.log")
    # fh = logging.FileHandler(log_file, encoding="utf-8")
    # fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    # logging.getLogger().addHandler(fh)

    log.info(f"exp_tag={exp_tag}  log_dir={log_dir}  weight_dir={weight_dir}")
    log.info("Initializing...")

    ### GPU Configuration ###
    device = config_gpu(config)

    # ── 案例分析所需的元信息（subject_id / t_start / 事件列表）──
    subjects_index = _load_subjects_index(config.dataset_dir)
    if subjects_index is not None:
        log.info(f"已加载 subjects_index.json：{len(subjects_index['subject_ids'])} 名受试者")

    # ── 从预处理产物 config.json 读取 windowing 配置 ──
    # 让 AHI 估计公式自动适配 C7 短窗 / 高密度滑窗等非默认配置
    win_size, stride = _load_windowing_config(config.dataset_dir)
    log.info(f"评估窗口配置: win_size={win_size}s, stride={stride}s (AHI 估计使用 stride={stride})")

    # 案例分析逐窗口产物的总目录
    pred_dir = os.path.join(log_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # 跨折累积的窗口级记录，用于最后一并导出 windows_index.csv
    master_records = []

    # Start Running
    log.info("Starting the evaluation...")
    results = []

    all_gt_ahis = []
    all_pred_ahis = []

    for FOLD in range(int(config.num_folds)):
        train_x, train_y, test_x, test_y = load_dataset(
            config.dataset, fold=FOLD, data_dir=config.dataset_dir, features=config.features)

        # 加载本折测试集的窗口元信息
        t_starts_test = _load_t_starts(config.dataset_dir, config.dataset, FOLD)

        if config.save_pred:
            result = np.load(
                f"{config.save_pred}/{FOLD}", allow_pickle=True)
            infer_time = result["infer_time"]
            test_pred = result["pred"]
            test_pred_prob = None
            test_attention = None
        else:
            test_pred, infer_time, test_pred_prob, test_attention = forward(
                train_set=(train_x, train_y),
                test_set=(test_x, test_y),
                model_class=MODELS[config.model],
                model_name=config.model,
                dataset_name=config.dataset,
                outer_fold=FOLD,
                log_dir=log_dir,
                weight_dir=weight_dir,
                device=device,
                wavenet_ch=train_x[0].shape[1]
            )

        if config.mode == "onset":
            result = evaluate_onset(
                test_y, test_pred, test_pred_prob, config, FOLD)

        elif config.mode == "severity":
            result = evaluate_severity(
                test_y, test_pred, config, FOLD, stride=stride)

        result["infer_time"] = infer_time / sum([len(a) for a in test_y])
        results.append(result)

        log.info(f"Fold {FOLD} | Acc: {result['acc']*100:.2f}% | F1: {result['f1']*100:.2f}% | "
                 f"Sensitivity: {result['sens']*100:.2f}% | Specificity: {result['spec']*100:.2f}% | "
                 f"AUROC: {result['auroc']*100:.2f}%")

        gt_ahis, pred_ahis = get_ahis(test_y, test_pred, config, FOLD, stride=stride)
        all_gt_ahis.append(gt_ahis)
        all_pred_ahis.append(pred_ahis)

        # ── 逐窗口产物落盘（供 case_study_DL.py 使用）──
        if subjects_index is not None and t_starts_test is not None:
            test_subjects = subjects_index["fold_assignments"][FOLD]["test_subjects"]

            # 把"按受试者分组"的 list-of-arrays 拍平成"flat 窗口序列"
            flat_subject_ids = []
            flat_t_starts    = []
            for s_idx, subj in enumerate(test_subjects):
                n_win = len(test_y[s_idx])
                flat_subject_ids.extend([subj] * n_win)
                flat_t_starts.extend(t_starts_test[s_idx].tolist())

            flat_subject_ids = np.array(flat_subject_ids, dtype=object)
            flat_t_starts    = np.array(flat_t_starts, dtype=np.int64)

            # 二值真实标签（与 evaluate_onset 内部口径一致）
            stacked_y = np.vstack(test_y)
            if np.any((stacked_y > 0) & (stacked_y < 1)):
                flat_y_true = (stacked_y.mean(axis=1) >= 0.5).astype(np.int64)
            else:
                flat_y_true = make_classif_y(stacked_y).astype(np.int64)

            # 整理一份 NPZ
            npz_data = {
                "subject_ids": flat_subject_ids,
                "t_starts":    flat_t_starts,
                "y_true":      flat_y_true,
                "y_pred":      np.asarray(test_pred,      dtype=np.int64),
                "y_proba":     np.asarray(test_pred_prob, dtype=np.float64) if test_pred_prob is not None else np.array([]),
            }
            if test_attention is not None:
                npz_data["attention"] = test_attention.astype(np.float32)

            npz_path = os.path.join(pred_dir, f"fold_{FOLD}.npz")
            np.savez(npz_path, **npz_data)
            log.info(f"  → 逐窗口预测保存至 {npz_path}"
                     f"（attention {'已收集' if test_attention is not None else '不适用'}）")

            # 累积到 master_records 用于最终 CSV
            n = len(flat_subject_ids)
            for i in range(n):
                rec = {
                    "subject_id": str(flat_subject_ids[i]),
                    "fold":       FOLD,
                    "t_start":    int(flat_t_starts[i]),
                    "y_true":     int(flat_y_true[i]),
                    "y_pred":     int(test_pred[i]),
                }
                if test_pred_prob is not None:
                    rec["y_proba"] = float(test_pred_prob[i])
                master_records.append(rec)
        else:
            log.info(f"  → 跳过逐窗口落盘（缺失 t_starts 或 subjects_index.json，请用新版 prep_mesa_v2 重跑）")

    # Concatenate AHIs
    all_gt_ahis = np.concatenate(all_gt_ahis, axis=None)
    all_pred_ahis = np.concatenate(all_pred_ahis, axis=None)

    # Scatter 图保存到本地
    fig = plt.figure(figsize=(6, 6), dpi=180)
    plt.scatter(all_gt_ahis, all_pred_ahis, s=3, alpha=0.4)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    fig.savefig(os.path.join(log_dir, "scatter_all.png"))
    plt.close(fig)

    # Bland-Altman 图保存到本地
    fig = bland_altman_plot(all_gt_ahis, all_pred_ahis, config)
    fig.savefig(os.path.join(log_dir, "bland_altman_all.png"))
    plt.close(fig)

    pearson = pearsonr(all_gt_ahis, all_pred_ahis)

    results = pd.DataFrame.from_records(results)
    mean = results.mean(axis=0).to_dict()
    std  = results.std(axis=0, ddof=0).to_dict()

    log.info("=" * 50)
    log.info("5-Fold Summary:")
    log.info(f"  Accuracy:    {mean['acc']*100:.2f}% ± {std['acc']*100:.2f}%")
    log.info(f"  F1:          {mean['f1']*100:.2f}% ± {std['f1']*100:.2f}%")
    log.info(f"  Sensitivity: {mean['sens']*100:.2f}% ± {std['sens']*100:.2f}%")
    log.info(f"  Specificity: {mean['spec']*100:.2f}% ± {std['spec']*100:.2f}%")
    log.info(f"  AUROC:       {mean['auroc']*100:.2f}% ± {std['auroc']*100:.2f}%")
    log.info(f"  Pearson r:   {pearson[0]:.4f} (p={pearson[1]:.4f})")
    log.info("=" * 50)

    # ── 跨 5 折的窗口级总索引 CSV（与 ML 路线对称）──
    if master_records:
        csv_path = os.path.join(pred_dir, "windows_index.csv")
        fieldnames = ["subject_id", "fold", "t_start", "y_true", "y_pred"]
        if "y_proba" in master_records[0]:
            fieldnames.append("y_proba")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(master_records)
        log.info(f"窗口级索引 → {csv_path}（共 {len(master_records)} 行）")


if __name__ == "__main__":
    main()
