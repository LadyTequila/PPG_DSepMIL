# ============================================================
# prep_mesa_enhance.py 相对原始 prep_mesa.py 的改动说明
# ============================================================
#
# 1. 延迟对齐（--delay-sec）
#    - 原始脚本：事件标签与 PPG 信号严格对齐（delay=0）
#    - 改动：make_label_array 新增 delay_sec 参数，将事件时间戳整体向后偏移
#      指定秒数再生成标签，模拟 PPG 信号对呼吸事件的生理性延迟响应
#    - 原始行为等价于 --delay-sec 0
#
# 2. 窗口长度（--win-size）
#    - 原始脚本：WIN_SIZE=60、STRIDE=30 均为硬编码常量
#    - 改动：WIN_SIZE 改为命令行参数，STRIDE 自动设为 win_size // 2（固定 50% 重叠）
#    - 同时 ppg_extraction 接收 win_size 参数，signal.resample 目标长度仍固定为
#      RESAMPLE_N=60，保证模型输入维度不变
#
# 3. 软标签（--soft-label）
#    - 原始脚本：标签为逐秒二值数组（0/1）
#    - 改动：make_label_array 新增 soft_label 参数；启用时，先生成逐秒二值数组，
#      再计算窗口内事件秒数占比（ratio = sum / win_size），并将整个数组填充为
#      该统一浮点值（保持数组长度与原始格式兼容，runner.py 取均值即得软标签）
#    - 原始行为等价于 --soft-label 不传（默认 False）
#
# 4. 输出目录结构
#    - 原始脚本：输出直接放在 OUT_DIR 下
#    - 改动：自动在 OUT_DIR 下创建 delay{D}_win{W}_soft{0/1} 子目录，
#      不同参数组合的数据集相互隔离，不会覆盖
#
# ============================================================

"""
MESA 增强数据预处理脚本
=======================
在 prep_mesa.py 基础上新增三个可配置的改进项：

  1. 延迟对齐（--delay-sec）
        呼吸暂停事件发生后，PPG 信号的响应存在约 10~30 秒的生理性延迟。
        该参数将事件时间戳整体向后偏移指定秒数再生成标签。
        推荐网格搜索：0 / 10 / 15 / 20 / 30

  2. 窗口长度（--win-size）
        原始脚本固定使用 60 秒窗口。此参数支持对比 60/45/30/15 秒配置。
        步长自动设为窗口长度的一半（50% 重叠）。

  3. 软标签（--soft-label）
        将二值标签（0/1）替换为窗口内事件秒数占比（0.0~1.0）。
        每秒仍标记，但最终输出的是连续值而非整数，
        训练时 BCE 损失可直接使用，评估时以 0.5 为阈值二值化。

输出目录由 --out-dir 指定，子目录名自动包含参数组合，例如：
    MESA_enhance/delay0_win60_soft0/mesa_fold0_x_train.pickle

用法示例：
    # 基准对比（与 prep_mesa.py 等价）
    python prep_mesa_enhance.py --delay-sec 0 --win-size 60

    # 延迟对齐实验
    python prep_mesa_enhance.py --delay-sec 15

    # 窗口长度消融
    python prep_mesa_enhance.py --win-size 30

    # 软标签
    python prep_mesa_enhance.py --soft-label

    # 组合
    python prep_mesa_enhance.py --delay-sec 15 --win-size 30 --soft-label
"""

import argparse
import glob
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

# ── 默认路径（与 prep_mesa.py 保持一致）──────────────────────────────────────
DEFAULT_NPZ_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_raw_ppg_spo2_flow"
DEFAULT_XML_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_xml"
DEFAULT_OUT_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/MESA_enhance"

# ── 固定超参数（不参与消融）─────────────────────────────────────────────────
RESAMPLE_N    = 60    # 每个窗口重采样后的时间点数（与模型输入维度一致，不可改）
BOUNDARY_SEC  = 1800  # 首尾各去掉 30 分钟
MIN_SPO2      = 60    # SpO2 最低阈值
N_FOLDS       = 5     # 交叉验证折数
AHI_THRESHOLD = 15    # AHI > 15 视为 severe（用于分层采样）

APNEA_KEYWORDS = ["apnea", "hypopnea"]


# ── 工具函数 ──────────────────────────────────────────────────────────────────

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


def make_label_array(events, window_start_sec, win_size, delay_sec, soft_label):
    """
    生成长度为 win_size 的标签数组（每秒一个值）。

    参数：
        events           : [(start_sec, duration_sec), ...]
        window_start_sec : 当前窗口的起始秒数
        win_size         : 窗口大小（秒）
        delay_sec        : 延迟偏移量（秒）。
                           事件标签向后偏移 delay_sec 秒，
                           即标记为"窗口内的 PPG 响应对应 delay_sec 秒前的事件"。
        soft_label       : 若为 True，标签值为 [0.0, 1.0] 之间的连续值（事件秒占比）；
                           若为 False，标签值为 {0, 1} 二值（与原始脚本一致）。

    返回：
        np.ndarray, shape (win_size,), dtype float32
    """
    labels = np.zeros(win_size, dtype=np.float32)
    for (ev_start, ev_dur) in events:
        # 将事件时间戳向后偏移 delay_sec，模拟 PPG 的生理性延迟响应
        shifted_start = ev_start + delay_sec
        shifted_end   = shifted_start + ev_dur
        for s in range(win_size):
            t = window_start_sec + s
            if shifted_start <= t < shifted_end:
                labels[s] = 1.0

    if soft_label:
        # 返回窗口内事件秒数占比（单一标量，但仍保持 win_size 长度的数组形式
        # 以与原始数据格式兼容；训练时 runner.py 会对每秒标签取均值作为窗口标签）
        ratio = float(labels.sum()) / win_size
        labels = np.full(win_size, ratio, dtype=np.float32)

    return labels


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def ppg_extraction(raw_ppg, hz, win_size):
    """
    从一段原始 PPG 信号提取 7 个形态特征。
    返回 shape (7, RESAMPLE_N) 的 float32 数组，失败时抛出异常。

    特征顺序（与原始脚本完全一致）:
        0: PWA  - Pulse Wave Amplitude
        1: SPD  - Systolic Phase Duration
        2: DPD  - Diastolic Phase Duration
        3: PA   - Pulse Area
        4: PPI  - PP Interval
        5: dPWA - Derivative of PWA
        6: dPPI - Derivative of PPI
    """
    ppg_clean = nk.ppg_clean(raw_ppg, sampling_rate=hz)
    info = nk.ppg_findpeaks(ppg_clean, sampling_rate=hz)
    peaks = info["PPG_Peaks"]
    rr_interval = (np.diff(peaks) / hz) * 1000  # ms

    sos = signal.cheby2(2, 0.1, 20, "hp", fs=hz, output="sos")
    filtered = signal.sosfilt(sos, raw_ppg)
    filtered_ma = moving_average(filtered, hz // 2)

    local_minima = argrelextrema(filtered_ma, np.less)[0]
    local_maxima = argrelextrema(filtered_ma, np.greater)[0]

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


# ── 单个受试者处理 ─────────────────────────────────────────────────────────────

def process_subject(npz_path, xml_path, win_size, delay_sec, soft_label):
    """
    处理一个受试者。
    返回:
        features_list: list of np.ndarray (7, RESAMPLE_N)
        labels_list:   list of np.ndarray (win_size,)
        ahi:           float
    """
    with np.load(npz_path, allow_pickle=True) as z:
        ppg     = z["ppg"].astype("float32")
        spo2    = z["spo2"].astype("float32")
        fs_ppg  = float(z["fs_ppg"][0])
        fs_spo2 = float(z["fs_spo2"][0])

    hz = int(round(fs_ppg))
    total_seconds = int(len(spo2) / fs_spo2)

    events = parse_events(xml_path)
    ahi = compute_ahi(events, total_seconds)

    stride = win_size // 2  # 固定 50% 重叠

    features_list = []
    labels_list   = []

    for t_start in range(BOUNDARY_SEC, total_seconds - BOUNDARY_SEC - win_size, stride):
        # SpO2 质量过滤
        spo2_start = int(t_start * fs_spo2)
        spo2_end   = int((t_start + win_size) * fs_spo2)
        spo2_win   = spo2[spo2_start:spo2_end]
        if len(spo2_win) == 0 or np.min(spo2_win) < MIN_SPO2:
            continue

        # PPG 窗口
        ppg_start = int(t_start * fs_ppg)
        ppg_end   = int((t_start + win_size) * fs_ppg)
        ppg_win   = ppg[ppg_start:ppg_end]
        if len(ppg_win) < win_size * hz:
            continue

        try:
            feat = ppg_extraction(ppg_win, hz=hz, win_size=win_size)
        except Exception:
            continue

        label = make_label_array(events, t_start, win_size, delay_sec, soft_label)

        features_list.append(feat)
        labels_list.append(label)

    return features_list, labels_list, ahi


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(npz_dir, xml_dir, out_dir, win_size, delay_sec, soft_label):
    # 子目录名包含实验参数，方便区分多组实验结果
    exp_tag = f"delay{delay_sec}_win{win_size}_soft{int(soft_label)}"
    exp_dir = os.path.join(out_dir, exp_tag)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"实验配置: 延迟={delay_sec}s | 窗口={win_size}s | 软标签={'是' if soft_label else '否'}")
    print(f"输出目录: {exp_dir}")

    npz_files = sorted(glob.glob(os.path.join(npz_dir, "mesa_*_raw_ppg_spo2_flow.npz")))
    if not npz_files:
        raise FileNotFoundError(f"在 {npz_dir} 下未找到任何 NPZ 文件")

    print(f"找到 {len(npz_files)} 个受试者")

    all_features = []
    all_labels   = []
    all_ahi      = []
    subject_ids  = []

    for npz_path in tqdm(npz_files, desc="处理受试者"):
        match = re.search(r"mesa_(\d{4})_raw", os.path.basename(npz_path))
        if not match:
            print(f"  警告：跳过无法解析编号的文件 {npz_path}")
            continue
        subj_id = match.group(1)

        xml_path = os.path.join(xml_dir, f"mesa-sleep-{subj_id}-nsrr.xml")
        if not os.path.exists(xml_path):
            print(f"  警告：受试者 {subj_id} 缺少 XML 标注文件，跳过")
            continue

        feats, labels, ahi = process_subject(
            npz_path, xml_path,
            win_size=win_size,
            delay_sec=delay_sec,
            soft_label=soft_label,
        )

        if len(feats) == 0:
            print(f"  警告：受试者 {subj_id} 无有效窗口，跳过")
            continue

        all_features.append(np.array(feats,  dtype=np.float32))
        all_labels.append(  np.array(labels, dtype=np.float32))
        all_ahi.append(ahi)
        subject_ids.append(subj_id)

    n_subjects = len(subject_ids)
    print(f"\n有效受试者：{n_subjects} 人")

    if n_subjects == 0:
        raise RuntimeError("没有任何有效受试者，请检查数据路径和文件格式")

    ahi_arr    = np.array(all_ahi)
    ahi_severe = (ahi_arr > AHI_THRESHOLD).astype(int)
    subject_idx = np.arange(n_subjects)

    print(f"AHI 分布 — severe (>15): {ahi_severe.sum()}  non-severe: {(1-ahi_severe).sum()}")

    skf = StratifiedShuffleSplit(n_splits=N_FOLDS, random_state=42, test_size=0.10)

    log_lines = [f"实验参数: delay={delay_sec}s, win_size={win_size}s, soft_label={soft_label}\n",
                 "5-Fold Train/Test subject indices\n"]
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

        prefix = os.path.join(exp_dir, f"mesa_fold{fold}")
        with open(prefix + "_x_train.pickle", "wb") as f:
            pickle.dump(x_train, f, protocol=4)
        with open(prefix + "_y_train.pickle", "wb") as f:
            pickle.dump(y_train, f, protocol=4)
        with open(prefix + "_x_test.pickle", "wb") as f:
            pickle.dump(x_test, f, protocol=4)
        with open(prefix + "_y_test.pickle", "wb") as f:
            pickle.dump(y_test, f, protocol=4)

        total_train_win = sum(x.shape[0] for x in x_train)
        total_test_win  = sum(x.shape[0] for x in x_test)
        print(f"       train windows: {total_train_win}  test windows: {total_test_win}")
        fold += 1

    log_path = os.path.join(exp_dir, "fold_info.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    print(f"\n完成！pickle 文件已保存至 {exp_dir}")
    print(f"折次信息已保存至 {log_path}")


# ── 命令行入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MESA 增强数据预处理（支持延迟对齐、窗口长度对比、软标签）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz-dir",    default=DEFAULT_NPZ_DIR, help="NPZ 文件目录")
    parser.add_argument("--xml-dir",    default=DEFAULT_XML_DIR, help="XML 标注目录")
    parser.add_argument("--out-dir",    default=DEFAULT_OUT_DIR, help="输出根目录（会自动创建子目录）")
    parser.add_argument("--win-size",   type=int,   default=60,    help="窗口大小（秒）：60/45/30/15")
    parser.add_argument("--delay-sec",  type=int,   default=0,     help="标签延迟偏移（秒）：0/10/15/20/30")
    parser.add_argument("--soft-label", action="store_true",       help="启用软标签（事件秒占比，替代二值标签）")
    args = parser.parse_args()

    main(
        npz_dir=args.npz_dir,
        xml_dir=args.xml_dir,
        out_dir=args.out_dir,
        win_size=args.win_size,
        delay_sec=args.delay_sec,
        soft_label=args.soft_label,
    )
