# ============================================================
# prep_mesa_v2.py —— 全新数据预处理脚本
# ============================================================
#
# 相较于 prep_mesa_enhance.py 的主要改动：
#
# 1. 睡眠时窗裁剪（替代固定 BOUNDARY_SEC = 1800）
#    - 原始脚本：首尾各去掉 30 分钟
#    - 改动：从 XML 睡眠分期事件（Stages|Stages）中提取第一个非 Wake
#      分期的起始时刻（sleep_onset_sec）和最后一个非 Wake 分期的结束
#      时刻（sleep_offset_sec），只使用这段区间内的数据
#    - 相关函数：get_sleep_window()
#
# 2. Hypopnea/Unsure 事件过滤
#    - 原始脚本：只要 EventConcept 包含 "apnea" 或 "hypopnea" 就算事件
#    - 改动：
#      a) Obstructive apnea 事件直接保留，不需额外确认
#      b) Hypopnea 和 Unsure 事件统一视为候选事件；只有在 XML 文本
#         中紧跟其后的下一个 ScoredEvent 的 EventConcept 为 Arousal
#         或 SpO2 desaturation 时，才确认为真正的呼吸事件
#    - 相关函数：parse_events_v2()
#
# 3. 保留了 prep_mesa_enhance.py 的所有可配置参数：
#    --delay-sec, --win-size, --soft-label
#
# ============================================================

"""
MESA v2 数据预处理脚本
======================
在 prep_mesa_enhance.py 基础上，按照导师指导做了两项关键改进：

  1. 睡眠时窗裁剪
        只使用"第一次入睡"到"最后一次醒来"之间的数据，
        替代原来首尾各去掉 30 分钟的粗略方法。
        睡眠起止时刻从 XML 中的睡眠分期事件提取。

  2. Hypopnea/Unsure 事件过滤
        Hypopnea 和 Unsure 统一视为候选事件。
        只有在 XML 中紧跟着 Arousal 或 SpO2 desaturation 事件的
        才确认为"真正的 Hypopnea"。
        Obstructive apnea 不受此规则影响，直接保留。

同时保留了延迟对齐、窗口长度、软标签等可配置参数。

用法示例：
    # 基本用法（delay=0, win=60, 二值标签）
    python prep_mesa_v2.py

    # 使用最优参数组合
    python prep_mesa_v2.py --delay-sec 10 --win-size 60 --soft-label

    # 自定义输出目录
    python prep_mesa_v2.py --out-dir ./MESA_v2
"""

import argparse
import glob
import os
import json
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

# ── 默认路径 ─────────────────────────────────────────────────────────────────
DEFAULT_NPZ_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_raw_ppg_spo2_flow"
DEFAULT_XML_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_xml"
DEFAULT_OUT_DIR = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/MESA_v2"

# ── 固定超参数 ────────────────────────────────────────────────────────────────
RESAMPLE_N    = 60    # 每个窗口重采样后的时间点数（与模型输入维度一致）
MIN_SPO2      = 60    # SpO2 最低阈值
N_FOLDS       = 5     # 交叉验证折数
AHI_THRESHOLD = 15    # AHI > 15 视为 severe（用于分层采样）


# ── 睡眠时窗提取 ──────────────────────────────────────────────────────────────

def get_sleep_window(xml_path):
    """
    从 XML 睡眠分期事件中提取第一次入睡和最后一次醒来的时刻。

    遍历所有 EventType 为 "Stages|Stages" 的事件，
    找到第一个非 Wake 分期的 Start（sleep_onset_sec）
    和最后一个非 Wake 分期的 Start + Duration（sleep_offset_sec）。

    返回:
        (sleep_onset_sec, sleep_offset_sec)
        如果没有找到任何非 Wake 分期，返回 None。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sleep_onset = None
    sleep_offset = None

    for ev in root.iter("ScoredEvent"):
        event_type = (ev.findtext("EventType") or "").strip().lower()
        concept = (ev.findtext("EventConcept") or "").strip().lower()

        # 只关注睡眠分期事件
        if "stages" not in event_type:
            continue

        # 跳过 Wake 分期（Wake|0）
        if "wake" in concept or concept.endswith("|0"):
            continue

        # 非 Wake 分期：Stage 1/2/3、REM
        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)
        end = start + dur

        if sleep_onset is None:
            sleep_onset = start
        sleep_offset = end  # 不断更新为最后一个非 Wake 分期的结束时刻

    if sleep_onset is not None and sleep_offset is not None:
        return (sleep_onset, sleep_offset)
    return None


# ── 事件解析（含 Hypopnea/Unsure 过滤）────────────────────────────────────────

def parse_events_v2(xml_path):
    """
    从 XML 解析呼吸事件，应用 Hypopnea/Unsure 过滤规则。

    规则：
      - Obstructive apnea：直接保留
      - Hypopnea 和 Unsure：统一视为候选事件，只有在 XML 中紧跟其后的
        下一个 ScoredEvent 的 EventConcept 为 Arousal 或 SpO2 desaturation
        时才确认为有效事件

    返回:
        events: [(start_sec, duration_sec), ...]  已确认的呼吸事件列表
        stats: dict  统计信息（总数、确认数、拒绝数）
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    scored_events = list(root.findall(".//ScoredEvent"))

    events = []
    n_apnea = 0
    n_candidate = 0
    n_confirmed = 0
    n_rejected = 0

    for i, ev in enumerate(scored_events):
        concept = (ev.findtext("EventConcept") or "").strip()
        c_lower = concept.lower()
        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)

        # ── Obstructive apnea：直接保留 ──
        if "obstructive apnea" in c_lower:
            events.append((start, dur))
            n_apnea += 1
            continue

        # ── Hypopnea 或 Unsure：需要确认 ──
        if "hypopnea" in c_lower or "unsure" in c_lower:
            n_candidate += 1

            # 检查 XML 中的下一个 ScoredEvent
            if i + 1 < len(scored_events):
                nxt = scored_events[i + 1]
                nxt_concept = (nxt.findtext("EventConcept") or "").strip().lower()

                if "arousal" in nxt_concept or "spo2 desaturation" in nxt_concept:
                    events.append((start, dur))
                    n_confirmed += 1
                    continue

            n_rejected += 1

    stats = {
        "obstructive_apnea": n_apnea,
        "candidate_hypopnea_unsure": n_candidate,
        "confirmed": n_confirmed,
        "rejected": n_rejected,
        "total_events": n_apnea + n_confirmed,
    }
    return events, stats


# ── 工具函数（与 prep_mesa_enhance.py 一致）──────────────────────────────────

def compute_ahi(n_events, total_seconds):
    """AHI = 事件数 / 总时长（小时）。"""
    hours = total_seconds / 3600.0
    return n_events / hours if hours > 0 else 0.0


def make_label_array(events, window_start_sec, win_size, delay_sec, soft_label):
    """
    生成长度为 win_size 的标签数组（每秒一个值）。

    参数：
        events           : [(start_sec, duration_sec), ...]
        window_start_sec : 当前窗口的起始秒数
        win_size         : 窗口大小（秒）
        delay_sec        : 延迟偏移量（秒）
        soft_label       : 若为 True，返回事件秒占比；否则返回二值标签

    返回：
        np.ndarray, shape (win_size,), dtype float32
    """
    labels = np.zeros(win_size, dtype=np.float32)
    for (ev_start, ev_dur) in events:
        shifted_start = ev_start + delay_sec
        shifted_end = shifted_start + ev_dur
        for s in range(win_size):
            t = window_start_sec + s
            if shifted_start <= t < shifted_end:
                labels[s] = 1.0

    if soft_label:
        ratio = float(labels.sum()) / win_size
        labels = np.full(win_size, ratio, dtype=np.float32)

    return labels


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def ppg_extraction(raw_ppg, hz, win_size):
    """
    从一段原始 PPG 信号提取 7 个形态特征。
    返回 shape (7, RESAMPLE_N) 的 float32 数组。
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


# ── 单个受试者处理 ────────────────────────────────────────────────────────────

def process_subject(npz_path, xml_path, win_size, delay_sec, soft_label):
    """
    处理一个受试者。

    改动点：
      - 使用 get_sleep_window() 获取睡眠时窗，替代固定 BOUNDARY_SEC
      - 使用 parse_events_v2() 进行事件过滤
      - AHI 基于睡眠时窗的总时长计算（而非整段录制时长）

    返回:
        features_list : list of np.ndarray (7, RESAMPLE_N)
        labels_list   : list of np.ndarray (win_size,)
        t_starts_list : list of int，每个窗口的起始秒（与 features/labels 索引对齐）
        ahi           : float
        event_stats   : dict
        sleep_window  : (onset, offset) or None
        events        : list of (ev_s, ev_d)，AASM 确认后的事件列表
    """
    with np.load(npz_path, allow_pickle=True) as z:
        ppg     = z["ppg"].astype("float32")
        spo2    = z["spo2"].astype("float32")
        fs_ppg  = float(z["fs_ppg"][0])
        fs_spo2 = float(z["fs_spo2"][0])

    hz = int(round(fs_ppg))

    # ── 获取睡眠时窗 ──
    sleep_window = get_sleep_window(xml_path)
    if sleep_window is None:
        return [], [], [], 0.0, {}, None, []

    sleep_onset, sleep_offset = sleep_window
    sleep_duration_sec = sleep_offset - sleep_onset

    # ── 解析并过滤呼吸事件 ──
    events, event_stats = parse_events_v2(xml_path)
    ahi = compute_ahi(event_stats["total_events"], sleep_duration_sec)

    stride = win_size // 2  # 固定 50% 重叠

    # 窗口起止范围：睡眠时窗内
    t_start_min = int(sleep_onset)
    t_start_max = int(sleep_offset) - win_size

    features_list = []
    labels_list   = []
    t_starts_list = []

    for t_start in range(t_start_min, t_start_max, stride):
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
        t_starts_list.append(int(t_start))

    return features_list, labels_list, t_starts_list, ahi, event_stats, sleep_window, events


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(npz_dir, xml_dir, out_dir, win_size, delay_sec, soft_label):
    exp_tag = f"delay{delay_sec}_win{win_size}_soft{int(soft_label)}"
    exp_dir = os.path.join(out_dir, exp_tag)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"实验配置: 延迟={delay_sec}s | 窗口={win_size}s | 软标签={'是' if soft_label else '否'}")
    print(f"输出目录: {exp_dir}")
    print()

    npz_files = sorted(glob.glob(os.path.join(npz_dir, "mesa_*_raw_ppg_spo2_flow.npz")))
    if not npz_files:
        raise FileNotFoundError(f"在 {npz_dir} 下未找到任何 NPZ 文件")

    print(f"找到 {len(npz_files)} 个受试者")

    all_features = []
    all_labels   = []
    all_t_starts = []           # 每受试者的窗口起始秒数组（与 features/labels 索引对齐）
    all_sleep_windows = []      # [(onset, offset), ...] 每受试者一项
    all_events   = []           # 每受试者一个事件列表 [(ev_s, ev_d), ...]
    all_ahi      = []
    subject_ids  = []

    # 汇总统计
    total_apnea = 0
    total_candidate = 0
    total_confirmed = 0
    total_rejected = 0
    skipped_no_sleep = 0

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

        feats, labels, t_starts, ahi, stats, sleep_window, events = process_subject(
            npz_path, xml_path,
            win_size=win_size,
            delay_sec=delay_sec,
            soft_label=soft_label,
        )

        if sleep_window is None:
            print(f"  警告：受试者 {subj_id} 无睡眠分期数据，跳过")
            skipped_no_sleep += 1
            continue

        if len(feats) == 0:
            print(f"  警告：受试者 {subj_id} 无有效窗口，跳过")
            continue

        # 累计统计
        total_apnea     += stats.get("obstructive_apnea", 0)
        total_candidate += stats.get("candidate_hypopnea_unsure", 0)
        total_confirmed += stats.get("confirmed", 0)
        total_rejected  += stats.get("rejected", 0)

        onset, offset = sleep_window
        tqdm.write(
            f"  {subj_id}: 睡眠 {onset:.0f}s~{offset:.0f}s "
            f"({(offset-onset)/3600:.1f}h) | "
            f"事件: apnea={stats['obstructive_apnea']} "
            f"hyp确认={stats['confirmed']}/{stats['candidate_hypopnea_unsure']} | "
            f"窗口={len(feats)} | AHI={ahi:.1f}"
        )

        all_features.append(np.array(feats,  dtype=np.float32))
        all_labels.append(  np.array(labels, dtype=np.float32))
        all_t_starts.append(np.array(t_starts, dtype=np.int64))
        all_sleep_windows.append((int(onset), int(offset)))
        all_events.append([(float(es), float(ed)) for es, ed in events])
        all_ahi.append(ahi)
        subject_ids.append(subj_id)

    n_subjects = len(subject_ids)
    print(f"\n{'='*60}")
    print(f"有效受试者：{n_subjects} 人（{skipped_no_sleep} 人因无睡眠分期被跳过）")
    print(f"事件统计汇总：")
    print(f"  Obstructive apnea:       {total_apnea}")
    print(f"  Hypopnea/Unsure 候选:    {total_candidate}")
    print(f"    -> 确认（紧跟 Arousal/SpO2 desat）: {total_confirmed}")
    print(f"    -> 拒绝:                            {total_rejected}")
    print(f"  总有效事件:              {total_apnea + total_confirmed}")
    print(f"{'='*60}")

    if n_subjects == 0:
        raise RuntimeError("没有任何有效受试者，请检查数据路径和文件格式")

    ahi_arr    = np.array(all_ahi)
    ahi_severe = (ahi_arr > AHI_THRESHOLD).astype(int)
    subject_idx = np.arange(n_subjects)

    print(f"AHI 分布 — severe (>15): {ahi_severe.sum()}  non-severe: {(1-ahi_severe).sum()}")

    skf = StratifiedShuffleSplit(n_splits=N_FOLDS, random_state=42, test_size=0.10)

    log_lines = [
        f"prep_mesa_v2.py 预处理日志\n",
        f"实验参数: delay={delay_sec}s, win_size={win_size}s, soft_label={soft_label}\n",
        f"睡眠时窗: 从 XML 分期事件提取（第一次入睡 ~ 最后一次醒来）\n",
        f"事件过滤: Hypopnea/Unsure 需紧跟 Arousal 或 SpO2 desaturation\n",
        f"\n事件统计:\n",
        f"  Obstructive apnea: {total_apnea}\n",
        f"  Hypopnea/Unsure 确认/候选: {total_confirmed}/{total_candidate}\n",
        f"  Hypopnea/Unsure 拒绝: {total_rejected}\n",
        f"\n5-Fold Train/Test subject indices\n",
    ]
    fold_assignments = []  # 收集每折的 train/test subject 索引，供 subjects_index.json 使用

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

        # 元信息（窗口起始秒）—— 与 features/labels 完全平行的 list-of-arrays
        t_starts_train = [all_t_starts[i] for i in train_idx]
        t_starts_test  = [all_t_starts[i] for i in test_idx]

        prefix = os.path.join(exp_dir, f"mesa_fold{fold}")
        with open(prefix + "_x_train.pickle", "wb") as f:
            pickle.dump(x_train, f, protocol=4)
        with open(prefix + "_y_train.pickle", "wb") as f:
            pickle.dump(y_train, f, protocol=4)
        with open(prefix + "_x_test.pickle", "wb") as f:
            pickle.dump(x_test, f, protocol=4)
        with open(prefix + "_y_test.pickle", "wb") as f:
            pickle.dump(y_test, f, protocol=4)
        # 新增：窗口起始秒元信息 pickle（与 x/y 同结构，case_study 时索引对齐）
        with open(prefix + "_t_starts_train.pickle", "wb") as f:
            pickle.dump(t_starts_train, f, protocol=4)
        with open(prefix + "_t_starts_test.pickle", "wb") as f:
            pickle.dump(t_starts_test, f, protocol=4)

        fold_assignments.append({
            "fold": fold,
            "train_idx": [int(i) for i in train_idx],
            "test_idx":  [int(i) for i in test_idx],
            "train_subjects": [subject_ids[i] for i in train_idx],
            "test_subjects":  [subject_ids[i] for i in test_idx],
        })

        total_train_win = sum(x.shape[0] for x in x_train)
        total_test_win  = sum(x.shape[0] for x in x_test)
        print(f"       train windows: {total_train_win}  test windows: {total_test_win}")
        fold += 1

    log_path = os.path.join(exp_dir, "fold_info.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    # ── 导出受试者级元信息 JSON（供 case_study_DL.py 使用）──
    subjects_index_path = os.path.join(exp_dir, "subjects_index.json")
    with open(subjects_index_path, "w", encoding="utf-8") as f:
        json.dump({
            "subject_ids": subject_ids,
            "ahi": [float(a) for a in all_ahi],
            "sleep_windows": [list(sw) for sw in all_sleep_windows],
            "events": all_events,                  # [[(ev_s, ev_d), ...], ...]
            "n_windows_per_subject": [int(len(t)) for t in all_t_starts],
            "fold_assignments": fold_assignments,
            "config": {
                "win_size":   win_size,
                "delay_sec":  delay_sec,
                "soft_label": bool(soft_label),
                "n_folds":    N_FOLDS,
                "ahi_threshold": AHI_THRESHOLD,
                "random_seed": 42,
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"\n完成！pickle 文件已保存至 {exp_dir}")
    print(f"折次信息已保存至 {log_path}")
    print(f"受试者元信息已保存至 {subjects_index_path}")


# ── 命令行入口 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MESA v2 数据预处理（睡眠时窗裁剪 + Hypopnea/Unsure 过滤）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz-dir",    default=DEFAULT_NPZ_DIR, help="NPZ 文件目录")
    parser.add_argument("--xml-dir",    default=DEFAULT_XML_DIR, help="XML 标注目录")
    parser.add_argument("--out-dir",    default=DEFAULT_OUT_DIR, help="输出根目录（会自动创建子目录）")
    parser.add_argument("--win-size",   type=int,   default=60,  help="窗口大小（秒）")
    parser.add_argument("--delay-sec",  type=int,   default=0,   help="标签延迟偏移（秒）")
    parser.add_argument("--soft-label", action="store_true",     help="启用软标签")
    args = parser.parse_args()

    main(
        npz_dir=args.npz_dir,
        xml_dir=args.xml_dir,
        out_dir=args.out_dir,
        win_size=args.win_size,
        delay_sec=args.delay_sec,
        soft_label=args.soft_label,
    )
