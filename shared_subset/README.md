# MESA quality7 子集使用说明（PPG + SpO2 + Flow + XML）

这份子集用于“分享/复现”，只包含 **overall5==7** 的高质量受试者：
- 原始信号（从 EDF 提取，不重采样/不滤波/不归一化）
- 对应的 NSRR 事件标注 XML（原样拷贝）

> 说明：不同 EDF 的通道命名可能略有差异；脚本默认优先匹配通道名为 `Flow` 的气流通道。

---

## 1. 目录结构

默认会在 `shared_subset/` 下生成两个文件夹：

- `shared_subset/mesa_quality7_raw_ppg_spo2_flow/`
  - `mesa_XXXX_raw_ppg_spo2_flow.npz`（每个受试者一个）
- `shared_subset/mesa_quality7_xml/`
  - `mesa-sleep-XXXX-nsrr.xml`（每个受试者一个）

其中 `XXXX` 为 4 位受试者编号（例如 `0002`）。

---

## 2. 如何生成子集

### 2.1 准备依赖

在当前 Python 环境中安装：

```bash
pip install pyedflib numpy pandas tqdm
```

### 2.2 设置数据集路径

两种方式任选其一：

- 环境变量：

```bash
export MESA_ROOT=/mnt/d/Dataset/mesa
```

- 或在命令行里传参：`--mesa-root /path/to/mesa`

脚本会用到：
- metadata CSV：`$MESA_ROOT/datasets/mesa-sleep-dataset-0.8.0.csv`（用于筛选 overall5==7）
- EDF 目录：`$MESA_ROOT/polysomnography/edfs`
- XML 目录：`$MESA_ROOT/polysomnography/annotations-events-nsrr`

### 2.3 导出三通道信号（PPG + SpO2 + Flow）

```bash
python extract_mesa_raw_ppg_spo2_flow_quality7.py \
  --out-dir shared_subset/mesa_quality7_raw_ppg_spo2_flow \
  --max-workers 6
```

- PPG 通道固定为 `Pleth`
- SpO2 通道固定为 `SpO2`
- Flow（气流/鼻压）通道：
  - 自动模式：若存在通道名精确为 `Flow`（不区分大小写），优先选它；否则尝试关键字匹配
  - 手动模式：强烈推荐在自动识别失败时手动指定

```bash
python extract_mesa_raw_ppg_spo2_flow_quality7.py --flow-channel "Flow"
```

### 2.4 导出标注 XML

```bash
python extract_mesa_quality7_xmls.py \
  --out-dir shared_subset/mesa_quality7_xml \
  --max-workers 8
```

---

## 3. NPZ 文件内容定义

每个 `mesa_XXXX_raw_ppg_spo2_flow.npz` 包含：

- `ppg`: `float32`，形状 `(n_ppg,)`
- `spo2`: `float32`，形状 `(n_spo2,)`
- `flow`: `float32`，形状 `(n_flow,)`
- `fs_ppg`: `float32` 标量（保存为 shape `(1,)` 的数组）
- `fs_spo2`: `float32` 标量（保存为 shape `(1,)` 的数组）
- `fs_flow`: `float32` 标量（保存为 shape `(1,)` 的数组）
- `meta_json`: `str`（保存为 object 数组，shape `(1,)`）

`meta_json` 是一个 JSON 字符串，至少包含：
- `edf_path`: 原 EDF 路径
- `channels`: 实际保存的通道名列表（包含实际的 `flow_channel` 名称）
- `ppg_channel`, `spo2_channel`, `flow_channel`
- `fs_ppg`, `fs_spo2`, `fs_flow`
- `n_ppg`, `n_spo2`, `n_flow`

> 注意：PPG/SpO2/Flow 的采样率可能不一致，这是正常现象。

---

## 4. 读取示例（Python）

```python
import json
import numpy as np

npz_path = "shared_subset/mesa_quality7_raw_ppg_spo2_flow/mesa_0002_raw_ppg_spo2_flow.npz"

with np.load(npz_path, allow_pickle=True) as z:
    ppg = z["ppg"].astype("float32")
    spo2 = z["spo2"].astype("float32")
    flow = z["flow"].astype("float32")

    fs_ppg = float(z["fs_ppg"][0])
    fs_spo2 = float(z["fs_spo2"][0])
    fs_flow = float(z["fs_flow"][0])

    meta = json.loads(z["meta_json"][0])

print(ppg.shape, fs_ppg)
print(spo2.shape, fs_spo2)
print(flow.shape, fs_flow)
print(meta["flow_channel"])
```

---

## 5. XML 文件说明

`shared_subset/mesa_quality7_xml/mesa-sleep-XXXX-nsrr.xml` 为 NSRR 的事件标注文件。

- 脚本只做 **复制**，不改写内容。
- 若个别受试者缺失 XML，脚本会统计为 `missing`。

---

## 6. 常见问题（FAQ）

### Q1：脚本报错“未能自动识别 Flow 通道”怎么办？

- 先查看 EDF 的 `ch_names`，确认气流通道具体名字
- 然后用 `--flow-channel "<实际通道名>"` 手动指定（例如 `Flow`）

### Q2：为什么三路信号长度不同？

因为各通道采样率可能不同，且 EDF 存储/截断方式也可能不完全一致。该子集定位是“原始提取”，不做对齐。

### Q3：怎么验证 XML 与 NPZ 的受试者集合一致？

最简单方式是比较两个目录下的 `XXXX` 列表（文件名中 4 位编号）。

---

## 7. 建议的分享方式

通常做法是把这两个目录一起打包：

```bash
tar -czf mesa_quality7_subset_ppg_spo2_flow_xml.tgz \
  shared_subset/mesa_quality7_raw_ppg_spo2_flow \
  shared_subset/mesa_quality7_xml \
  shared_subset/README.md
```
