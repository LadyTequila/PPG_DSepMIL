# ============================================================
# analyze_event_delay.py —— 呼吸事件响应延迟统计
# ============================================================
#
# 目的：为论文中"delay=10s"这一超参选择提供数据支撑。
#
# 分析思路：
#   呼吸暂停事件发生后，PPG 信号的响应并非瞬时，而是通过两条生理
#   通路显现：(1) SpO2 去饱和（血氧循环时延约 10~30s）；
#          (2) 微觉醒（Arousal）引发的交感兴奋。
#   这两类事件在 MESA 的 XML 标注中被独立标注。
#
#   本脚本对每个已确认呼吸事件（OA / 确认的 Hypopnea/Unsure），
#   在其后的合理时间窗内（默认 90s 以内）查找第一次出现的
#   Arousal 或 SpO2 Desaturation，并统计：
#       - delay_from_start : 响应起始 - 事件起始
#       - delay_from_end   : 响应起始 - 事件结束
#
# 输出：
#   - 终端打印总体 / 按事件类型分组的 mean / median / p25 / p75
#   - 可选：保存 csv 详细记录 + 直方图 png
#
# 用法：
#   python analyze_event_delay.py
#   python analyze_event_delay.py --plot
#   python analyze_event_delay.py --out-dir ./delay_stats
# ============================================================

import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from tqdm import tqdm

DEFAULT_XML_DIR = (
    "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/shared_subset/mesa_quality7_xml"
)
DEFAULT_OUT_DIR = (
    "C:/Users/薛卜元/Desktop/毕业设计/Code/ApSense-main/delay_stats"
)

# 在事件之后多长时间内出现的 Arousal/Desat 视为"相关响应"
MAX_SEARCH_SEC = 90.0


# ── 事件解析 ──────────────────────────────────────────────────────────────────

def classify_event(concept_lower):
    """返回 ('apnea' / 'hypop' / 'arousal' / 'desat' / None)。"""
    if "obstructive apnea" in concept_lower:
        return "apnea"
    if "hypopnea" in concept_lower or "unsure" in concept_lower:
        return "hypop"
    if "arousal" in concept_lower:
        return "arousal"
    if "spo2 desaturation" in concept_lower or "desaturation" in concept_lower:
        return "desat"
    return None


def analyze_one_xml(xml_path, max_search_sec=MAX_SEARCH_SEC):
    """
    返回:
        records: list of dict，每条对应一次 "事件 -> 响应" 配对
        subject_stats: dict 当前受试者的统计
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    scored_events = list(root.findall(".//ScoredEvent"))

    # 预解析为 (kind, start, duration)，便于向前搜索
    parsed = []
    for ev in scored_events:
        concept = (ev.findtext("EventConcept") or "").strip().lower()
        kind = classify_event(concept)
        if kind is None:
            parsed.append(None)
            continue
        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)
        parsed.append((kind, start, dur))

    records = []

    for i, item in enumerate(parsed):
        if item is None:
            continue
        kind, ev_start, ev_dur = item
        if kind not in ("apnea", "hypop"):
            continue

        ev_end = ev_start + ev_dur

        # 搜索紧邻下一个事件，判定该 Hypopnea 是否被"确认"
        immediate_next_kind = None
        for j in range(i + 1, len(parsed)):
            if parsed[j] is None:
                continue
            immediate_next_kind = parsed[j][0]
            break

        # 对齐 parse_events_v2 的确认规则
        if kind == "hypop":
            if immediate_next_kind not in ("arousal", "desat"):
                continue  # 未确认的 Hypop，不纳入延迟统计

        # 向后扫描 Arousal 与 Desat，分别记录首次出现的时间差
        first_arousal = None
        first_desat = None
        for j in range(i + 1, len(parsed)):
            if parsed[j] is None:
                continue
            nxt_kind, nxt_start, _ = parsed[j]
            if nxt_start - ev_start > max_search_sec:
                break
            if nxt_kind == "arousal" and first_arousal is None:
                first_arousal = nxt_start
            elif nxt_kind == "desat" and first_desat is None:
                first_desat = nxt_start
            if first_arousal is not None and first_desat is not None:
                break

        for resp_kind, resp_start in (("arousal", first_arousal),
                                      ("desat", first_desat)):
            if resp_start is None:
                continue
            records.append({
                "event_kind": kind,
                "response_kind": resp_kind,
                "delay_from_start": resp_start - ev_start,
                "delay_from_end": resp_start - ev_end,
                "event_duration": ev_dur,
            })

    return records


# ── 聚合 & 打印 ──────────────────────────────────────────────────────────────

def summarize_array(arr, name):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"{name}: N/A"
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return (
        f"{name:<42} n={arr.size:>6}  "
        f"mean={arr.mean():6.2f}s  std={arr.std():5.2f}  "
        f"median={q50:5.2f}  p25={q25:5.2f}  p75={q75:5.2f}"
    )


def print_summary(all_records):
    # 全体
    print("=" * 100)
    print("总体统计")
    print("=" * 100)

    by_event_response = defaultdict(list)  # {(event_kind, response_kind, delay_metric): [values]}
    by_metric = defaultdict(list)          # {delay_metric: [values]}

    for r in all_records:
        key_start = (r["event_kind"], r["response_kind"], "from_start")
        key_end = (r["event_kind"], r["response_kind"], "from_end")
        by_event_response[key_start].append(r["delay_from_start"])
        by_event_response[key_end].append(r["delay_from_end"])
        by_metric["from_start"].append(r["delay_from_start"])
        by_metric["from_end"].append(r["delay_from_end"])

    print(summarize_array(by_metric["from_start"], "[全部] 响应 - 事件起始"))
    print(summarize_array(by_metric["from_end"],   "[全部] 响应 - 事件结束"))
    print()

    print("按事件×响应类型分组")
    print("-" * 100)
    for ev_kind in ("apnea", "hypop"):
        for resp_kind in ("desat", "arousal"):
            for metric in ("from_start", "from_end"):
                key = (ev_kind, resp_kind, metric)
                if key in by_event_response:
                    label = f"[{ev_kind}->{resp_kind} / {metric}]"
                    print(summarize_array(by_event_response[key], label))
    print()

    # 以起始为参考的总体直方图（便于选 delay 超参）
    print('以「响应起始 - 事件起始」为准的粗分布（秒）')
    print("-" * 100)
    from_start_all = np.asarray(by_metric["from_start"], dtype=np.float64)
    from_start_all = from_start_all[np.isfinite(from_start_all)]
    edges = [0, 5, 10, 15, 20, 25, 30, 40, 60, 90]
    hist, _ = np.histogram(from_start_all, bins=edges)
    total = hist.sum()
    for i in range(len(edges) - 1):
        cnt = hist[i]
        pct = cnt / total * 100.0 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  [{edges[i]:>3}, {edges[i+1]:>3})s : {cnt:>6}  ({pct:5.2f}%)  {bar}")


def save_histogram_png(all_records, out_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图")
        return

    from_start = np.array([r["delay_from_start"] for r in all_records], dtype=np.float64)
    from_start = from_start[np.isfinite(from_start)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(from_start, bins=45, range=(0, 90), color="#3b8686", edgecolor="white")
    ax.axvline(np.median(from_start), color="red", linestyle="--",
               label=f"median = {np.median(from_start):.1f}s")
    ax.axvline(np.mean(from_start), color="orange", linestyle="--",
               label=f"mean = {np.mean(from_start):.1f}s")
    ax.set_xlabel("Response start - Event start  (s)")
    ax.set_ylabel("Count")
    ax.set_title("Respiratory event -> immediate Arousal/SpO2 desaturation latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"直方图已保存至 {out_path}")


def save_csv(all_records, out_path):
    import csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "event_kind", "response_kind",
            "delay_from_start", "delay_from_end", "event_duration",
        ])
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)
    print(f"明细已保存至 {out_path}（共 {len(all_records)} 条记录）")


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(xml_dir, out_dir, plot):
    os.makedirs(out_dir, exist_ok=True)

    xml_files = sorted(glob.glob(os.path.join(xml_dir, "mesa-sleep-*-nsrr.xml")))
    if not xml_files:
        raise FileNotFoundError(f"在 {xml_dir} 下未找到 XML 文件")

    print(f"找到 {len(xml_files)} 份 XML，开始扫描...\n")

    all_records = []
    n_subj_valid = 0

    for xml_path in tqdm(xml_files, desc="扫描 XML"):
        match = re.search(r"mesa-sleep-(\d{4})-nsrr\.xml", os.path.basename(xml_path))
        subj_id = match.group(1) if match else "UNKNOWN"

        try:
            records = analyze_one_xml(xml_path)
        except Exception as e:
            tqdm.write(f"  {subj_id}: 解析失败 ({e})，跳过")
            continue

        for r in records:
            r["subject_id"] = subj_id
        all_records.extend(records)
        if records:
            n_subj_valid += 1

    print(f"\n共获得 {len(all_records)} 条 事件->响应 配对（来自 {n_subj_valid} 名受试者）\n")

    if not all_records:
        print("无有效记录，退出")
        return

    print_summary(all_records)

    csv_path = os.path.join(out_dir, "event_response_delay.csv")
    save_csv(all_records, csv_path)

    if plot:
        png_path = os.path.join(out_dir, "event_response_delay_hist.png")
        save_histogram_png(all_records, png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统计 MESA XML 中呼吸事件到 Arousal/SpO2 去饱和响应的延迟",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml-dir", default=DEFAULT_XML_DIR, help="XML 目录")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--plot", action="store_true", help="同时保存直方图 PNG")
    args = parser.parse_args()

    main(xml_dir=args.xml_dir, out_dir=args.out_dir, plot=args.plot)
