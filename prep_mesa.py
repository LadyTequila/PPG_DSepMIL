"""
MESA 数据预处理脚本
===================
输入:
    NPZ_DIR  - 每个受试者一个 .npz 文件，含 ppg / spo2 / flow 原始信号
    XML_DIR  - 每个受试者一个 NSRR 标注 .xml 文件

输出 (OUT_DIR):
    mesa_fold{N}_x_train.pickle  -- list of arrays, each shape (n_windows, 7, 60)
    mesa_fold{N}_y_train.pickle  -- list of arrays, each shape (n_windows, 60)
    mesa_fold{N}_x_test.pickle
    mesa_fold{N}_y_test.pickle

用法:
    python prep_mesa.py
    python prep_mesa.py --npz-dir /path/to/npz --xml-dir /path/to/xml --out-dir ./MESA
"""

import argparse
import glob
import json
import os
import pickle
import re
import warnings
import xml.etree.ElementTree as ET

import neurokit2 as nk
import numpy as np
from scipy import signal
from scipy.signal import argrelextrema
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 默认路径 ────────────────────────────────────────────────────────────────
DEFAULT_NPZ_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_raw_ppg_spo2_flow"
DEFAULT_XML_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_xml"
DEFAULT_OUT_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/MESA"

# ── 超参数 ──────────────────────────────────────────────────────────────────
WIN_SIZE      = 60    # 窗口大小（秒）
STRIDE        = 30    # 步长（秒）
RESAMPLE_N    = 60    # 每个窗口重采样后的时间点数
BOUNDARY_SEC  = 1800  # 首尾各去掉 30 分钟
MIN_SPO2      = 60    # SpO2 最低阈值（低于此值的窗口丢弃）
N_FOLDS       = 5     # 交叉验证折数
AHI_THRESHOLD = 15    # AHI > 15 视为 severe（用于分层采样）

# 呼吸暂停相关事件关键词（不区分大小写）
APNEA_KEYWORDS = ["apnea", "hypopnea"]


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def parse_events(xml_path):
    """从 XML 解析呼吸暂停/低通气事件，返回 [(start_sec, duration_sec), ...]。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    events = []
    for ev in root.iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "").lower()
        if any(k in concept for k in APNEA_KEYWORDS):
            start = float(ev.findtext("Start") or 0)
            dur   = float(ev.findtext("Duration") or 0)
            events.append((start, dur))
    return events


def compute_ahi(events, total_seconds):
    """AHI = 事件数 / 总时长（小时）。"""
    hours = total_seconds / 3600.0
    return len(events) / hours if hours > 0 else 0.0


def make_label_array(events, window_start_sec):
    """
    生成 WIN_SIZE 长度的二值标签数组（每秒一个值）。
    某秒内有呼吸暂停/低通气事件则标记为 1，否则为 0。
    """
    labels = np.zeros(WIN_SIZE, dtype=np.float32)
    for (ev_start, ev_dur) in events:
        ev_end = ev_start + ev_dur
        for s in range(WIN_SIZE):
            t = window_start_sec + s
            if ev_start <= t < ev_end:
                labels[s] = 1.0
    return labels


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def ppg_extraction(raw_ppg, hz):
    """
    从一段原始 PPG 信号（长度 = hz * WIN_SIZE）提取 7 个形态特征。
    返回 shape (7, RESAMPLE_N) 的 float32 数组，失败时抛出异常。

    特征顺序:
        0: PWA  - Pulse Wave Amplitude
        1: SPD  - Systolic Phase Duration
        2: DPD  - Diastolic Phase Duration
        3: PA   - Pulse Area
        4: PPI  - PP Interval (RR interval from PPG peaks)
        5: dPWA - Derivative of PWA
        6: dPPI - Derivative of PPI
    """
    # 峰值检测 → RR interval（毫秒）
    ppg_clean = nk.ppg_clean(raw_ppg, sampling_rate=hz)
    info = nk.ppg_findpeaks(ppg_clean, sampling_rate=hz)
    peaks = info["PPG_Peaks"]
    rr_interval = (np.diff(peaks) / hz) * 1000  # ms

    # 高通滤波 + 移动平均 → 局部极值
    sos = signal.cheby2(2, 0.1, 20, "hp", fs=hz, output="sos")
    filtered = signal.sosfilt(sos, raw_ppg)
    filtered_ma = moving_average(filtered, hz // 2)

    local_minima = argrelextrema(filtered_ma, np.less)[0]
    local_maxima = argrelextrema(filtered_ma, np.greater)[0]

    # 去除过近的极小值
    rm_index, rm2_index = [], []
    diffs = np.diff(local_minima)
    for k in range(len(diffs)):
        if diffs[k] < 30:
            rm_index.append(k)
            rm2_index.append(k + 1)
    sel_min = np.delete(local_minima, rm_index)
    sel_max = np.delete(local_maxima, rm2_index)

    pwa_list, systole_list, diastole_list, area_list = [], [], [], []

    count_min = 0
    for mini in sel_min:
        count_min += 1
        count_max = 0
        for maxi in sel_max:
            count_max += 1
            if maxi > mini:
                pwa_list.append(float(filtered_ma[maxi] - filtered_ma[mini]))
                systole_list.append(float(maxi - mini))
                if count_min + 1 < len(sel_min):
                    diastole_list.append(
                        float(sel_min[count_min + 1] - sel_max[count_max - 1])
                    )
                area_list.append(float(np.sum(filtered_ma[mini:maxi])))
                break

    # 重采样到固定长度
    pwa_arr      = signal.resample(pwa_list, RESAMPLE_N)
    systole_arr  = signal.resample(systole_list, RESAMPLE_N)
    diastole_arr = signal.resample(diastole_list, RESAMPLE_N)
    area_arr     = signal.resample(area_list, RESAMPLE_N)
    rr_arr       = signal.resample(rr_interval, RESAMPLE_N)
    diff_rr_arr  = signal.resample(np.diff(rr_interval), RESAMPLE_N)
    diff_pwa_arr = signal.resample(np.diff(pwa_list), RESAMPLE_N)

    return np.array(
        [pwa_arr, systole_arr, diastole_arr, area_arr, rr_arr, diff_pwa_arr, diff_rr_arr],
        dtype=np.float32,
    )


# ── 单个受试者处理 ────────────────────────────────────────────────────────────

def process_subject(npz_path, xml_path):
    """
    处理一个受试者。
    返回:
        features_list: list of np.ndarray (7, RESAMPLE_N)
        labels_list:   list of np.ndarray (WIN_SIZE,)
        ahi:           float
    """
    # 加载信号
    with np.load(npz_path, allow_pickle=True) as z:
        ppg    = z["ppg"].astype("float32")
        spo2   = z["spo2"].astype("float32")
        fs_ppg  = float(z["fs_ppg"][0])
        fs_spo2 = float(z["fs_spo2"][0])

    hz = int(round(fs_ppg))
    total_seconds = int(len(spo2) / fs_spo2)

    # 解析标注
    events = parse_events(xml_path)
    ahi = compute_ahi(events, total_seconds)

    features_list = []
    labels_list   = []

    # 滑窗
    for t_start in range(BOUNDARY_SEC, total_seconds - BOUNDARY_SEC - WIN_SIZE, STRIDE):
        # SpO2 质量过滤
        spo2_start = int(t_start * fs_spo2)
        spo2_end   = int((t_start + WIN_SIZE) * fs_spo2)
        spo2_win   = spo2[spo2_start:spo2_end]
        if len(spo2_win) == 0 or np.min(spo2_win) < MIN_SPO2:
            continue

        # 提取 PPG 窗口
        ppg_start = int(t_start * fs_ppg)
        ppg_end   = int((t_start + WIN_SIZE) * fs_ppg)
        ppg_win   = ppg[ppg_start:ppg_end]
        if len(ppg_win) < WIN_SIZE * hz:
            continue

        # 特征提取
        try:
            feat = ppg_extraction(ppg_win, hz=hz)
        except Exception:
            continue

        # 标签数组（每秒一个值）
        label = make_label_array(events, t_start)

        features_list.append(feat)
        labels_list.append(label)

    return features_list, labels_list, ahi


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(npz_dir, xml_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 收集所有受试者的 NPZ 文件，按受试者编号排序
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "mesa_*_raw_ppg_spo2_flow.npz")))
    if not npz_files:
        raise FileNotFoundError(f"在 {npz_dir} 下未找到任何 NPZ 文件")

    print(f"找到 {len(npz_files)} 个受试者")

    # ── 第一阶段：处理每个受试者，提取特征和标签 ──────────────────────────
    all_features = []   # all_features[i] = list of (7,60) arrays for subject i
    all_labels   = []   # all_labels[i]   = list of (60,) arrays for subject i
    all_ahi      = []   # all_ahi[i]      = float AHI for subject i
    subject_ids  = []   # 受试者编号（用于日志）

    for npz_path in tqdm(npz_files, desc="处理受试者"):
        # 从文件名提取 4 位编号
        match = re.search(r"mesa_(\d{4})_raw", os.path.basename(npz_path))
        if not match:
            print(f"  警告：跳过无法解析编号的文件 {npz_path}")
            continue
        subj_id = match.group(1)

        xml_path = os.path.join(xml_dir, f"mesa-sleep-{subj_id}-nsrr.xml")
        if not os.path.exists(xml_path):
            print(f"  警告：受试者 {subj_id} 缺少 XML 标注文件，跳过")
            continue

        feats, labels, ahi = process_subject(npz_path, xml_path)

        if len(feats) == 0:
            print(f"  警告：受试者 {subj_id} 无有效窗口，跳过")
            continue

        all_features.append(np.array(feats,  dtype=np.float32))  # (n_win, 7, 60)
        all_labels.append(  np.array(labels, dtype=np.float32))  # (n_win, 60)
        all_ahi.append(ahi)
        subject_ids.append(subj_id)

    n_subjects = len(subject_ids)
    print(f"\n有效受试者：{n_subjects} 人")

    if n_subjects == 0:
        raise RuntimeError("没有任何有效受试者，请检查数据路径和文件格式")

    # ── AHI 分层标签（用于 StratifiedShuffleSplit）──────────────────────────
    ahi_arr    = np.array(all_ahi)
    ahi_severe = (ahi_arr > AHI_THRESHOLD).astype(int)  # 1=severe, 0=non-severe
    subject_idx = np.arange(n_subjects)

    print(f"AHI 分布 — severe (>15): {ahi_severe.sum()}  non-severe: {(1-ahi_severe).sum()}")

    # ── 第二阶段：5 折分层划分 ────────────────────────────────────────────────
    skf = StratifiedShuffleSplit(n_splits=N_FOLDS, random_state=42, test_size=0.10)

    log_lines = ["5-Fold Train/Test subject indices\n"]
    fold = 0
    for train_idx, test_idx in skf.split(subject_idx, ahi_severe):
        print(f"\nFold {fold}  train={len(train_idx)} subjects  test={len(test_idx)} subjects")
        log_lines.append(
            f"Fold {fold}\n  TRAIN: {[subject_ids[i] for i in train_idx]}\n"
            f"  TEST:  {[subject_ids[i] for i in test_idx]}\n"
        )

        x_train = [all_features[i] for i in train_idx]
        y_train = [all_labels[i]   for i in train_idx]
        x_test  = [all_features[i] for i in test_idx]
        y_test  = [all_labels[i]   for i in test_idx]

        prefix = os.path.join(out_dir, f"mesa_fold{fold}")
        with open(prefix + "_x_train.pickle", "wb") as f:
            pickle.dump(x_train, f, protocol=4)
        with open(prefix + "_y_train.pickle", "wb") as f:
            pickle.dump(y_train, f, protocol=4)
        with open(prefix + "_x_test.pickle", "wb") as f:
            pickle.dump(x_test, f, protocol=4)
        with open(prefix + "_y_test.pickle", "wb") as f:
            pickle.dump(y_test, f, protocol=4)

        # 简单统计
        total_train_win = sum(x.shape[0] for x in x_train)
        total_test_win  = sum(x.shape[0] for x in x_test)
        print(f"       train windows: {total_train_win}  test windows: {total_test_win}")
        fold += 1

    # 保存折次信息
    log_path = os.path.join(out_dir, "fold_info.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    print(f"\n完成！pickle 文件已保存至 {out_dir}")
    print(f"折次信息已保存至 {log_path}")


# ── 命令行入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MESA 数据预处理")
    parser.add_argument("--npz-dir", default=DEFAULT_NPZ_DIR,
                        help=f"NPZ 文件目录（默认: {DEFAULT_NPZ_DIR}）")
    parser.add_argument("--xml-dir", default=DEFAULT_XML_DIR,
                        help=f"XML 标注目录（默认: {DEFAULT_XML_DIR}）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"输出目录（默认: {DEFAULT_OUT_DIR}）")
    args = parser.parse_args()

    main(
        npz_dir=args.npz_dir,
        xml_dir=args.xml_dir,
        out_dir=args.out_dir,
    )
