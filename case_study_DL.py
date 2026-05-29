# ============================================================
# case_study_DL.py —— 第四章 DL 路线案例分析图与数值导出
# ============================================================
#
# 目的：从 evaluate.py 重跑产出的 MIL / SIL 逐窗口预测中挑选有诊断
#       意义的案例，绘制 PPG + SpO2 + attention 三通道图，并打印关键
#       统计量，供论文 4.4.4（图 4-6）/ 4.5.2（图 4-8）/ 4.5.3（图 4-9）
#       三处直接引用。
#
# 输入（必须先按新版 evaluate.py 跑完 SIL 与 MIL 两组评估）：
#   - model_v2/logs/delay10_win60_soft1_MIL/predictions/fold_{0..4}.npz
#   - model_v2/logs/delay10_win60_soft1/predictions/fold_{0..4}.npz
#   - MESA_v2/delay10_win60_soft1/subjects_index.json
#   - shared_subset/mesa_quality7_raw_ppg_spo2_flow/*.npz
#   - shared_subset/mesa_quality7_xml/*.xml
#
# 输出（默认）到 model_v2/logs/delay10_win60_soft1_MIL/predictions/case_studies/：
#   - mil_top_attention.png      （图 4-6：5 例 MIL 高置信正案例的 attention 曲线）
#   - sil_miss_mil_correct.png   （图 4-8：1 例 SIL 漏检 + MIL 正确的事件）
#   - mil_miss_hard.png          （图 4-9：1 例 MIL 仍漏检的困难样本）
#   - case_summary.csv           （所有挑出案例的关键数值汇总）
#
# 用法：
#   python case_study_DL.py
#   python case_study_DL.py --top-attention 5 --sil-miss-event-kind Hypopnea
# ============================================================

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

WIN_SIZE = 60

DEFAULT_NPZ_DIR  = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_raw_ppg_spo2_flow"
DEFAULT_XML_DIR  = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_xml"
DEFAULT_MIL_DIR  = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/model_v2/logs/delay10_win60_soft1_MIL/predictions"
DEFAULT_SIL_DIR  = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/model_v2/logs/delay10_win60_soft1/predictions"
DEFAULT_SUBJ_IDX = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/MESA_v2/delay10_win60_soft1/subjects_index.json"
DEFAULT_OUT_DIR  = "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/model_v2/logs/delay10_win60_soft1_MIL/predictions/case_studies"


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_predictions(predictions_dir, n_folds=5):
    """加载 5 折 NPZ 并拼接为 flat 数组。"""
    parts = {
        "subject_ids": [], "t_starts": [], "y_true": [],
        "y_pred": [], "y_proba": [], "attention": [],
    }
    has_attention = False
    for k in range(n_folds):
        npz_path = os.path.join(predictions_dir, f"fold_{k}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        npz = np.load(npz_path, allow_pickle=True)
        for key in list(parts.keys()):
            if key in npz.files:
                parts[key].append(npz[key])
                if key == "attention":
                    has_attention = True
    out = {}
    for k, vlist in parts.items():
        if vlist:
            out[k] = np.concatenate(vlist, axis=0)
        else:
            out[k] = None
    out["_has_attention"] = has_attention
    return out


def parse_event_kinds(xml_path):
    """复刻 parse_events_v2 的过滤规则，返回与其 events 列表同序的 kind 标签。"""
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
            if i + 1 < len(scored_events):
                nxt = (scored_events[i + 1].findtext("EventConcept") or "").strip().lower()
                if "arousal" in nxt or "spo2 desaturation" in nxt:
                    kinds.append("Hypopnea" if "hypopnea" in c_lower else "Unsure")
    return kinds


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


# ── 元信息归集 ────────────────────────────────────────────────────────────────

def find_overlapping_event(t_start, win_size, events):
    """返回与窗口重叠最多的事件 (idx, ev_s, ev_d, overlap_sec)。无重叠返回 None。"""
    t_end = t_start + win_size
    best = None
    best_overlap = 0.0
    for idx, (ev_s, ev_d) in enumerate(events):
        ev_e = ev_s + ev_d
        overlap = max(0.0, min(t_end, ev_e) - max(t_start, ev_s))
        if overlap > best_overlap:
            best_overlap = overlap
            best = (idx, float(ev_s), float(ev_d), float(overlap))
    return best


def collect_case_meta(idx, mil_data, subj_to_events, subj_to_event_kinds, subj_to_ahi):
    """对一个 flat 索引归集案例所有可显示的元信息。"""
    subj = str(mil_data["subject_ids"][idx])
    t_start = int(mil_data["t_starts"][idx])
    events = subj_to_events.get(subj, [])
    event_kinds = subj_to_event_kinds.get(subj, [])

    ev_match = find_overlapping_event(t_start, WIN_SIZE, events)
    if ev_match is None:
        ev_idx = -1
        ev_kind = "None"
        ev_s = ev_d = float("nan")
        coverage_pct = 0.0
        rel_start = rel_end = None
    else:
        ev_idx, ev_s, ev_d, overlap = ev_match
        ev_kind = event_kinds[ev_idx] if ev_idx < len(event_kinds) else "Unknown"
        coverage_pct = overlap / WIN_SIZE * 100.0
        rel_start = max(0.0, ev_s - t_start)
        rel_end   = min(float(WIN_SIZE), ev_s + ev_d - t_start)

    return {
        "subject_id":  subj,
        "ahi":         round(subj_to_ahi.get(subj, float("nan")), 2),
        "t_start_s":   t_start,
        "ev_idx":      int(ev_idx),
        "ev_kind":     ev_kind,
        "ev_total_s":  None if ev_match is None else round(ev_d, 2),
        "ev_rel_start_s": None if rel_start is None else round(rel_start, 2),
        "ev_rel_end_s":   None if rel_end   is None else round(rel_end, 2),
        "ev_coverage_pct": round(coverage_pct, 2),
        "y_true":      int(mil_data["y_true"][idx]),
        "y_pred":      int(mil_data["y_pred"][idx]),
        "y_proba":     round(float(mil_data["y_proba"][idx]), 4),
    }


# ── 案例挑选 ──────────────────────────────────────────────────────────────────

def pick_mil_top_attention(mil, n=5):
    """MIL 高置信正案例：y_true=1 AND y_pred=1，按 y_proba 降序。"""
    mask = (mil["y_true"] == 1) & (mil["y_pred"] == 1)
    idx = np.where(mask)[0]
    return idx[np.argsort(-mil["y_proba"][idx])][:n]


def pick_sil_miss_mil_correct(mil, sil, n=3,
                              mil_min_proba=0.7, sil_max_proba=0.4,
                              event_kind_filter=None,
                              subj_to_events=None,
                              subj_to_event_kinds=None):
    """SIL 漏检但 MIL 正确：MIL_proba>0.7 且 SIL_proba<0.4，按 (MIL−SIL) 差距降序。
    可选 event_kind_filter（如 'Hypopnea'）。"""
    sil_key_to_idx = {
        (str(s), int(t)): i
        for i, (s, t) in enumerate(zip(sil["subject_ids"], sil["t_starts"]))
    }
    cands = []
    for i in range(len(mil["y_true"])):
        if mil["y_true"][i] != 1 or mil["y_pred"][i] != 1:
            continue
        if mil["y_proba"][i] < mil_min_proba:
            continue
        key = (str(mil["subject_ids"][i]), int(mil["t_starts"][i]))
        sil_i = sil_key_to_idx.get(key)
        if sil_i is None:
            continue
        if sil["y_pred"][sil_i] != 0 or sil["y_proba"][sil_i] >= sil_max_proba:
            continue
        # 事件类型过滤
        if event_kind_filter is not None:
            subj = key[0]
            events = subj_to_events.get(subj, [])
            kinds  = subj_to_event_kinds.get(subj, [])
            ev_match = find_overlapping_event(int(mil["t_starts"][i]), WIN_SIZE, events)
            if ev_match is None:
                continue
            ev_idx = ev_match[0]
            ev_kind = kinds[ev_idx] if ev_idx < len(kinds) else "Unknown"
            if ev_kind != event_kind_filter:
                continue
        gap = float(mil["y_proba"][i]) - float(sil["y_proba"][sil_i])
        cands.append((i, sil_i, gap))
    cands.sort(key=lambda x: -x[2])
    return cands[:n]


def pick_mil_miss_hard(mil, n=3,
                       max_proba=0.3,
                       prefer_short_event=True,
                       subj_to_events=None):
    """MIL 持续漏检：y_true=1 AND y_pred=0 AND y_proba<0.3，按 y_proba 升序。
    可选 prefer_short_event：在前 50 个候选中再按事件总时长升序，挑出"短事件"漏检。"""
    mask = (mil["y_true"] == 1) & (mil["y_pred"] == 0) & (mil["y_proba"] < max_proba)
    idx = np.where(mask)[0]
    sorted_idx = idx[np.argsort(mil["y_proba"][idx])]

    if prefer_short_event and subj_to_events is not None:
        # 取前 50 个低置信漏检，按"事件总时长升序"挑前 n 个
        pool = sorted_idx[:50].tolist()
        scored = []
        for i in pool:
            subj = str(mil["subject_ids"][i])
            events = subj_to_events.get(subj, [])
            ev_match = find_overlapping_event(int(mil["t_starts"][i]), WIN_SIZE, events)
            if ev_match is None:
                continue
            scored.append((i, ev_match[2]))  # ev_d
        scored.sort(key=lambda x: x[1])
        return [i for i, _ in scored[:n]]

    return sorted_idx[:n].tolist()


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def _draw_event_shade(ax, rel_start, rel_end, label=None):
    if rel_start is None or rel_end is None or rel_end <= rel_start:
        return
    ax.axvspan(rel_start, rel_end, alpha=0.2, color="#d62728", label=label)


def plot_attention_panels(cases, mil_data, npz_dir, xml_dir, out_path,
                          subj_to_events, subj_to_event_kinds):
    """图 4-6：5 个子图，每子图一例 MIL 高置信正案例的 attention 曲线 + 事件阴影。"""
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.8 * n + 0.5), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, cases):
        meta = collect_case_meta(idx, mil_data, subj_to_events, subj_to_event_kinds,
                                 subj_to_ahi={})
        attn = mil_data["attention"][idx]   # (60,)
        t_axis = np.arange(WIN_SIZE)

        ax.plot(t_axis, attn, color="#1f77b4", linewidth=1.4)
        _draw_event_shade(ax, meta["ev_rel_start_s"], meta["ev_rel_end_s"],
                          label=f"event ({meta['ev_kind']})")
        ax.set_ylabel("attn weight", fontsize=8)
        ax.set_title(
            f"subj {meta['subject_id']}  t={meta['t_start_s']}s  "
            f"event={meta['ev_kind']}({meta['ev_total_s']}s, "
            f"{meta['ev_coverage_pct']:.0f}% cover)  "
            f"MIL_proba={meta['y_proba']}",
            fontsize=9
        )
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time within window (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_case_three_panel(idx, mil_data, sil_data, npz_dir,
                          subj_to_events, subj_to_event_kinds, subj_to_ahi,
                          out_path):
    """图 4-8 / 4-9 三联图：上 PPG, 中 SpO2, 下 attention。"""
    meta = collect_case_meta(idx, mil_data, subj_to_events, subj_to_event_kinds, subj_to_ahi)
    subj = meta["subject_id"]
    t_start = meta["t_start_s"]

    npz_path = os.path.join(npz_dir, f"mesa_{subj}_raw_ppg_spo2_flow.npz")
    ppg, spo2, fs_ppg, fs_spo2 = load_signal(npz_path)
    ppg_win  = slice_window(ppg,  fs_ppg,  t_start, WIN_SIZE)
    spo2_win = slice_window(spo2, fs_spo2, t_start, WIN_SIZE)

    spo2_min  = float(np.min(spo2_win))  if len(spo2_win) else float("nan")
    spo2_max  = float(np.max(spo2_win))  if len(spo2_win) else float("nan")
    spo2_drop = float(spo2_max - spo2_min) if np.isfinite(spo2_min) else float("nan")

    # 配套 SIL 预测（如果 sil_data 提供）
    sil_proba_str = ""
    if sil_data is not None:
        key = (subj, t_start)
        sil_lookup = {
            (str(s), int(t)): i
            for i, (s, t) in enumerate(zip(sil_data["subject_ids"], sil_data["t_starts"]))
        }
        si = sil_lookup.get(key)
        if si is not None:
            sil_proba_str = f"  SIL_proba={float(sil_data['y_proba'][si]):.3f}"

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)

    # 上：PPG
    t_ppg = np.arange(len(ppg_win)) / fs_ppg
    axes[0].plot(t_ppg, ppg_win, color="#2271b3", linewidth=0.6)
    axes[0].set_ylabel("PPG amp.")
    _draw_event_shade(axes[0], meta["ev_rel_start_s"], meta["ev_rel_end_s"],
                      label=f"event ({meta['ev_kind']})")
    # 标题拆为两行：上行身份信息，下行预测信息——避免单行过长溢出
    title_line1 = (
        f"subj {subj}  AHI={meta['ahi']}  t_start={t_start}s  "
        f"event={meta['ev_kind']}({meta['ev_total_s']}s, {meta['ev_coverage_pct']:.0f}% cover)"
    )
    title_line2 = f"MIL_proba={meta['y_proba']}{sil_proba_str}"
    axes[0].set_title(f"{title_line1}\n{title_line2}", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8)

    # 中：SpO2
    t_spo2 = np.arange(len(spo2_win)) / fs_spo2
    axes[1].plot(t_spo2, spo2_win, color="#2ca02c", linewidth=1.2)
    axes[1].set_ylabel("SpO$_2$ (%)")
    _draw_event_shade(axes[1], meta["ev_rel_start_s"], meta["ev_rel_end_s"])
    if np.isfinite(spo2_min):
        axes[1].set_ylim(min(80.0, spo2_min - 1.0), 102.0)

    # 下：attention
    attn = mil_data["attention"][idx]
    t_attn = np.arange(WIN_SIZE)
    axes[2].plot(t_attn, attn, color="#1f77b4", linewidth=1.4)
    axes[2].fill_between(t_attn, 0, attn, alpha=0.15, color="#1f77b4")
    axes[2].set_ylabel("attn weight")
    axes[2].set_xlabel("Time within window (s)")
    _draw_event_shade(axes[2], meta["ev_rel_start_s"], meta["ev_rel_end_s"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        **meta,
        "spo2_min":      round(spo2_min, 2),
        "spo2_max":      round(spo2_max, 2),
        "spo2_drop_pct": round(spo2_drop, 2) if np.isfinite(spo2_drop) else None,
        "saved_to":      out_path,
    }


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    print("加载预测产物 ...")
    mil = load_predictions(args.mil_dir)
    if not mil["_has_attention"]:
        raise RuntimeError(f"{args.mil_dir} 下的 NPZ 不含 attention 字段，无法做 MIL 案例分析")
    sil = load_predictions(args.sil_dir)
    print(f"  MIL: {len(mil['y_true'])} 窗口  /  SIL: {len(sil['y_true'])} 窗口")

    print(f"加载 {args.subjects_index}")
    with open(args.subjects_index, encoding="utf-8") as f:
        subj_idx_json = json.load(f)
    subj_to_events = {
        sid: [tuple(ev) for ev in evs]
        for sid, evs in zip(subj_idx_json["subject_ids"], subj_idx_json["events"])
    }
    subj_to_ahi = dict(zip(subj_idx_json["subject_ids"], subj_idx_json["ahi"]))

    # 为每位受试者解析事件类型（与 events 列表同序）
    print("解析 XML 事件类型 ...")
    subj_to_event_kinds = {}
    for sid in subj_idx_json["subject_ids"]:
        xml_path = os.path.join(args.xml_dir, f"mesa-sleep-{sid}-nsrr.xml")
        if os.path.exists(xml_path):
            subj_to_event_kinds[sid] = parse_event_kinds(xml_path)
        else:
            subj_to_event_kinds[sid] = []

    summary_rows = []

    # ── 图 4-6：5 例 MIL 高置信正案例的 attention 曲线 ──
    print(f"\n[图 4-6] 挑选 MIL 高置信正案例 top-{args.top_attention} ...")
    top_idx = pick_mil_top_attention(mil, n=args.top_attention)
    print(f"  共 {len(top_idx)} 个候选")
    for rank, i in enumerate(top_idx, 1):
        meta = collect_case_meta(i, mil, subj_to_events, subj_to_event_kinds, subj_to_ahi)
        print(f"  TOP-{rank}: subj {meta['subject_id']} t={meta['t_start_s']}s  "
              f"event={meta['ev_kind']}  proba={meta['y_proba']}")
    out_4_6 = os.path.join(args.out_dir, "mil_top_attention.png")
    plot_attention_panels(top_idx.tolist(), mil, args.npz_dir, args.xml_dir, out_4_6,
                          subj_to_events, subj_to_event_kinds)
    print(f"  图 4-6 → {out_4_6}")
    for rank, i in enumerate(top_idx, 1):
        meta = collect_case_meta(i, mil, subj_to_events, subj_to_event_kinds, subj_to_ahi)
        meta["case_type"] = f"FIG_4-6_TOP{rank}_MIL_top_pos"
        meta["saved_to"] = out_4_6
        summary_rows.append(meta)

    # ── 图 4-8：1 例 SIL 漏检 + MIL 正确 ──
    print(f"\n[图 4-8] 挑选 SIL 漏检 + MIL 正确（事件类型 = {args.sil_miss_event_kind}）...")
    sil_miss_cands = pick_sil_miss_mil_correct(
        mil, sil, n=3,
        event_kind_filter=args.sil_miss_event_kind,
        subj_to_events=subj_to_events,
        subj_to_event_kinds=subj_to_event_kinds,
    )
    if not sil_miss_cands:
        print("  ⚠ 没有找到符合条件的案例，尝试不限事件类型...")
        sil_miss_cands = pick_sil_miss_mil_correct(
            mil, sil, n=3, subj_to_events=subj_to_events,
            subj_to_event_kinds=subj_to_event_kinds,
        )

    if sil_miss_cands:
        i_mil, i_sil, gap = sil_miss_cands[0]
        print(f"  最佳: MIL_proba={float(mil['y_proba'][i_mil]):.3f} "
              f"SIL_proba={float(sil['y_proba'][i_sil]):.3f}  gap={gap:.3f}")
        out_4_8 = os.path.join(args.out_dir, "sil_miss_mil_correct.png")
        info = plot_case_three_panel(
            i_mil, mil, sil, args.npz_dir,
            subj_to_events, subj_to_event_kinds, subj_to_ahi,
            out_4_8
        )
        info["case_type"] = "FIG_4-8_SIL_miss_MIL_correct"
        summary_rows.append(info)
        print(f"  图 4-8 → {out_4_8}")
    else:
        print("  ⚠ 仍无候选，跳过图 4-8")

    # ── 图 4-9：1 例 MIL 仍漏检的困难样本 ──
    print(f"\n[图 4-9] 挑选 MIL 仍漏检的困难样本 ...")
    miss_idx = pick_mil_miss_hard(mil, n=3, subj_to_events=subj_to_events)
    if miss_idx:
        i = miss_idx[0]
        print(f"  最佳: subj {mil['subject_ids'][i]} t={int(mil['t_starts'][i])}s  "
              f"MIL_proba={float(mil['y_proba'][i]):.3f}")
        out_4_9 = os.path.join(args.out_dir, "mil_miss_hard.png")
        info = plot_case_three_panel(
            i, mil, sil, args.npz_dir,
            subj_to_events, subj_to_event_kinds, subj_to_ahi,
            out_4_9
        )
        info["case_type"] = "FIG_4-9_MIL_miss_hard"
        summary_rows.append(info)
        print(f"  图 4-9 → {out_4_9}")

    # ── 案例汇总 CSV ──
    if summary_rows:
        csv_path = os.path.join(args.out_dir, "case_summary.csv")
        # 取所有行的 fieldnames 并集，保持稳定顺序
        priority = [
            "case_type", "subject_id", "ahi", "t_start_s", "ev_kind",
            "ev_total_s", "ev_rel_start_s", "ev_rel_end_s", "ev_coverage_pct",
            "spo2_min", "spo2_max", "spo2_drop_pct",
            "y_true", "y_pred", "y_proba", "saved_to",
        ]
        all_keys = set()
        for r in summary_rows:
            all_keys.update(r.keys())
        fieldnames = [k for k in priority if k in all_keys] + \
                     [k for k in sorted(all_keys) if k not in priority]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n汇总 → {csv_path}")

    # ── 友好打印（供论文 4.4.4 / 4.5.2 / 4.5.3 复制）──
    print("\n" + "=" * 80)
    print("关键数值速览（供论文 4.4.4 / 4.5.2 / 4.5.3 直接引用）")
    print("=" * 80)
    for row in summary_rows:
        print(f"\n[{row['case_type']}] subj {row['subject_id']}  AHI={row.get('ahi')}")
        print(f"  t_start = {row['t_start_s']} s  事件类型 = {row['ev_kind']}  "
              f"事件总时长 = {row.get('ev_total_s')} s")
        if row.get("ev_rel_start_s") is not None:
            print(f"  事件相对窗口位置 = [{row['ev_rel_start_s']}, {row['ev_rel_end_s']}] s "
                  f"({row['ev_coverage_pct']}% 覆盖)")
        if "spo2_min" in row:
            print(f"  SpO2: min = {row['spo2_min']}%   max = {row['spo2_max']}%   "
                  f"drop = {row['spo2_drop_pct']}%")
        print(f"  MIL: y_pred = {row['y_pred']}  y_proba = {row['y_proba']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DL 路线案例分析：从 evaluate.py 产出的逐窗口预测中挑案例并绘图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mil-dir",          default=DEFAULT_MIL_DIR)
    parser.add_argument("--sil-dir",          default=DEFAULT_SIL_DIR)
    parser.add_argument("--subjects-index",   default=DEFAULT_SUBJ_IDX)
    parser.add_argument("--npz-dir",          default=DEFAULT_NPZ_DIR)
    parser.add_argument("--xml-dir",          default=DEFAULT_XML_DIR)
    parser.add_argument("--out-dir",          default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-attention",    type=int, default=5,
                        help="图 4-6 中要画几例 MIL 高置信正案例")
    parser.add_argument("--sil-miss-event-kind", default="Hypopnea",
                        choices=["Hypopnea", "ObstructiveApnea", "Unsure", None],
                        help="图 4-8 SIL 漏检案例的事件类型偏好（None 表示不限）")
    args = parser.parse_args()

    main(args)
