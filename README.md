# 基于 PPG 信号的睡眠呼吸暂停事件检测

本项目是华中科技大学本科毕业设计成果代码仓库。在ApSense（Choksatchawathi et al., IEEE IoT-J 2024）端到端PPG-OSA检测框架基础上，从数据清洗、标签构造、深度建模三个层面提出递进式改进方案，最终模型在MESA quality=7子集（276 受试者）上达到事件级AUROC 84.96 ± 1.58、患者级AHI Pearson r 0.8075。

## 研究背景

阻塞性睡眠呼吸暂停（OSA）是常见的睡眠呼吸障碍，临床金标准依赖整夜PSG多导睡眠监测，存在设备复杂、费用高、可及性差等问题。光电容积脉搏波（PPG）作为单一通道、可在指夹式或可穿戴设备上获取的信号，已被多项研究证明对呼吸事件具有可分辨的形态学特征，是低成本筛查方案的代表性候选。

ApSense提出了一套基于PPG形态特征+时空CNN的端到端OSA事件检测框架，在MESA上报告AUROC 77.64%。本研究在复现ApSense的基础上，结合对MESA数据集和误差案例的定量分析，识别出该工作在数据质量、标签粒度、建模范式三个层面的具体不足，并提出对应的改进方案。

## 主要贡献

1. **数据层面**：用NSRR XML标注的睡眠分期信息做精确入睡-觉醒裁剪，替代ApSense的固定30分钟首尾截取，避免清醒期混入；对Hypopnea与Unsure事件按AASM临床定义增加紧邻 Arousal或SpO₂下降的二次确认，提升标签可靠性。
2. **标签层面**：用窗口内事件覆盖比例的软标签替代二值硬标签，缓解事件边界模糊带来的训练噪声；通过定量分析发现PPG对呼吸事件存在约10秒的生理响应延迟，引入延迟对齐机制并实证发现延迟与软标签存在协同效应。
3. **建模层面**：将ApSense的端到端CNN改造为多示例学习（MIL）框架 DSepMIL，引入Smoothed Gated Attention Pooling让模型输出连续的注意力波段，契合呼吸事件的物理本质；同时设计ComboFocalF1Loss（Focal + SoftF1 组合损失）缓解MIL训练中的mode collapse问题。
4. **对照基线**：构建机器学习路径作为深度学习的对照——以事件中心对齐方式构造1:1正负样本，对7通道形态特征提取7种统计量得到49维特征，分别训练 SVM、决策树、随机森林，明确传统特征工程方法在本任务上的性能上限。

## 方法概览

本研究遵循数据-标签-建模三层递进的改进框架：

```
原始PSG→[数据层]XML精确裁剪+AASM紧邻确认→清洗后事件
                                      ↓
                              [标签层]软标签+延迟对齐
                                      ↓
                            [建模层]DSepMIL+ComboFocalF1Loss
                                      ↓
                              事件二分类+AHI估计
```

数据层和标签层的改进体现在三套递进的预处理脚本（`prep_mesa.py` → `prep_mesa_enhance.py` → `prep_mesa_v2.py`）中；建模层的演进体现在两个版本的训练框架（`model_enhance/` → `model_v2/`）中。

## 数据集

本研究使用 [MESA Sleep Study](https://sleepdata.org/datasets/mesa) 数据集（NSRR注册访问），多源人种的中老年队列，包含II型整夜PSG记录。本研究选取其中PPG信号质量评分=7的子集（共276受试者）作为分析对象，每名受试者保留PPG、SpO₂、Nasal Flow三个通道。

**数据访问**：MESA 数据集需通过 [NSRR 网站](https://sleepdata.org/) 注册并提交研究计划申请，本仓库不包含任何原始PSG数据。`shared_subset/` 目录在本地用于存放申请到的子集。

## 项目结构

```
ApSense-main/
├── README.md                            本文件
├── requirements.txt                     Python依赖清单
│
├── prep_mesa.py                         ApSense原版预处理复现
├── prep_mesa_enhance.py                 第一轮改进（延迟对齐 + 软标签）
├── prep_mesa_v2.py                      第二轮改进（XML精确裁剪+AASM紧邻确认）
│
├── mesa_ML.py                           机器学习路径主入口（SVM/DT/RF）
├── analyze_event_delay.py               PPG响应延迟的定量分析脚本
├── case_study_ML.py                     ML路径的逐受试者案例分析
├── case_study_DL.py                     DL路径的逐窗口案例分析（含MIL注意力可视化）
│
├── model_enhance/                       第一轮深度学习框架（多baseline对照）
│   ├── main.py / runner.py / evaluate.py
│   ├── losses.py / utils.py / gen_table.py
│   ├── config/train.yaml / evaluate.yaml   Hydra配置
│   └── models/
│       ├── dsepnet.py / dsepembedder.py    本组改进的DSep系列
│       ├── aiosa.py / aiosa_modify.py      AIOSA baseline（adapted to PPG）
│       ├── rrwavenet.py / mini_rrwavenet.py RRWaveNet baseline
│       ├── lenet5.py / eeg_net_sa.py       LeNet-5 / PPGNetSA baseline
│       └── _dsep_block.py / _better_lstm.py 组件模块
│
├── model_v2/                            最终训练框架（聚焦DSepMIL）
│   ├── main.py / runner.py / evaluate.py
│   ├── losses.py（含 ComboFocalF1Loss）
│   ├── utils.py / gen_table.py
│   ├── config/train.yaml / evaluate.yaml   Hydra配置
│   └── models/
│       ├── dsepnet.py（DSepST15Net_no_branch，端到端基线）
│       ├── dsep_mil.py（DSepMIL，MIL改进版，核心模型）
│       ├── aiosa.py / _dsep_block.py
│       └── __init__.py
│
└── shared_subset/                       MESA数据集子集存放位置（不入库）
    └── README.md                        数据访问说明
```

下列目录用于存放数据与训练产物，已在`.gitignore`中排除：

- `shared_subset/mesa_quality7_*/`：MESA原EDF转换出的npz与XML标注
- `MESA/`, `MESA_v2/`, `MESA_enhance/`：三轮预处理的pickle产物
- `model_enhance/logs/`, `model_enhance/weights/`：第一轮训练的日志和权重
- `model_v2/logs/`, `model_v2/weights/`：最终训练的日志和权重
- `ML_results/`：机器学习路径的输出
- `*.pickle`, `*.npz`, `*.pt`, `*.h5`：各类二进制中间产物

## 环境与依赖

Python 3.10+，CUDA 12.1。详见`requirements.txt`。核心依赖：

- PyTorch ≥ 2.0（带CUDA）
- numpy, pandas, scipy, scikit-learn
- hydra-core（配置管理）
- mlflow（实验跟踪，可选）
- neurokit2（PPG信号清洗与峰值检测）
- pyedflib（EDF文件读取）
- matplotlib, seaborn（可视化）
- imbalanced-learn（类别不平衡处理）

安装：

```bash
# 推荐建一个独立 conda 环境
conda create -n apsense python=3.10 -y
conda activate apsense

# 单独装 GPU 版 PyTorch（按服务器 CUDA 版本选 wheel index）
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 装其余依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

向[NSRR](https://sleepdata.org/)申请MESA数据集访问权限，下载polysomnography子目录中的EDF与XML 文件。本研究使用的 quality=7子集共 276 受试者编号清单见 `shared_subset/README.md`。

下载到本地后，将EDF转为npz格式（仅保留PPG/SpO₂/Flow三个通道）并放置到：

```
shared_subset/mesa_quality7_raw_ppg_spo2_flow/mesa_XXXX_raw_ppg_spo2_flow.npz
shared_subset/mesa_quality7_xml/mesa-sleep-XXXX-nsrr.xml
```

### 2. 预处理

三套预处理脚本对应论文中的递进改进，具体操作请参考对应预处理脚本注释。

```bash
# ApSense原版预处理复现（fixed 30-min trim+不做事件确认）
python prep_mesa.py

# 第一轮改进：加入延迟对齐与软标签（v1→enhance）
python prep_mesa_enhance.py --delay-sec 10 --soft-label

# 第二轮改进：XML精确裁剪+AASM紧邻确认（enhance→v2）
python prep_mesa_v2.py --delay-sec 10 --soft-label
```

每个脚本会按 5 折 AHI 分层 CV 切分，输出 `mesa_fold{k}_x_{split}.pickle` 等格式的30个pickle文件到对应目录。

### 3. 训练

`model_v2/` 用Hydra配置管理。修改 `model_v2/config/train.yaml` 指定 `dataset_dir`（指向上一步的pickle目录）、`exp_tag`（实验标签）、`model`（`DSepST15Net_no_branch`或`DSepMIL`）后：

```bash
cd model_v2
python main.py
# 或者用hydra命令行覆盖：
python main.py model=DSepMIL exp_tag=v2_delay10_MIL_soft1
```

训练采用5×5嵌套交叉验证（外层5 fold×内层KFold 5），输出日志和权重到`logs/{exp_tag}/`和`weights/{exp_tag}/`。

### 4. 评估

```bash
cd model_v2
python evaluate.py exp_tag=v2_delay10_MIL_soft1
```

评估脚本会计算5折的Acc/F1/Sens/Spec/AUROC，以及患者级AHI Pearson r、Bland-Altman一致性分析，并落盘逐窗口预测供案例分析使用。

### 5. 案例分析

```bash
# 机器学习路径的案例分析（事件中心对齐模式）
python case_study_ML.py

# 深度学习路径的案例分析（含 MIL 注意力波段可视化）
python case_study_DL.py
```

## 主要结果

5折交叉验证下的关键实验组：

| 实验配置 | Acc | F1 | Sens | Spec | AUROC | Pearson r |
|---|---|---|---|---|---|---|
| ApSense 原版基线（v1, DSepST15Net）| 67.74 ± 2.61 | 55.95 ± 0.92 | 64.29 ± 5.69 | 68.19 ± 3.64 | 71.92 ± 1.75 | 0.5551 |
| v2 数据层改进（无延迟、硬标签）| 79.57 ± 1.59 | 68.24 ± 0.86 | 61.62 ± 4.40 | 82.94 ± 2.35 | 80.85 ± 0.76 | 0.7668 |
| v2 + 软标签 | 80.35 ± 1.78 | 64.13 ± 1.80 | 70.28 ± 5.61 | 81.31 ± 2.28 | 83.48 ± 2.15 | 0.7964 |
| v2 + 延迟 10s | 76.76 ± 1.99 | 65.79 ± 1.58 | 62.66 ± 4.48 | 79.40 ± 2.98 | 78.94 ± 1.10 | 0.6995 |
| **v2 + DSepMIL + 软标签 + 延迟（最优）** | **81.34 ± 1.26** | **65.22 ± 1.69** | **70.98 ± 4.27** | **82.35 ± 1.45** | **84.96 ± 1.58** | **0.8075** |

机器学习路径作为对照（事件中心对齐 1:1 平衡采样）：

| 模型 | Acc | F1 | Sens | Spec | AUROC | Pearson r |
|---|---|---|---|---|---|---|
| SVM (RBF) | 67.57 ± 1.09 | 68.30 ± 1.11 | 69.90 ± 2.21 | 65.23 ± 2.75 | 73.28 ± 1.65 | 0.1626 |
| 决策树 | 57.72 ± 0.92 | 58.52 ± 0.67 | 59.65 ± 0.96 | 55.78 ± 2.05 | 57.72 ± 0.92 | 0.1251 |
| 随机森林 | 67.09 ± 1.58 | 68.83 ± 1.39 | 72.71 ± 2.60 | 61.48 ± 3.77 | 72.87 ± 1.61 | 0.2009 |

完整实验记录见`TODO.md`。

## 复现说明

本仓库目标是支持论文结果的可复现实验，但**完整复现需要MESA原始数据访问权限**。在获得NSRR授权后，按照流程逐步运行即可重现表中的所有数字。5折划分使用固定随机种子（`random_state=42`，`StratifiedShuffleSplit`），逐受试者分组完全可重复。

## 致谢

感谢葛俊锋老师、桂康老师和储檀学长在课题方向方法学讨论与论文写作过程中的悉心指导。感谢[NSRR](https://sleepdata.org/)与MESA Sleep Study项目组开放高质量的PSG公共数据。本研究的基线方法ApSense由Choksatchawathi等人提出，他们的工作和[开源代码](https://github.com/IoBT-VISTEC/ApSense)是本研究的起点。
