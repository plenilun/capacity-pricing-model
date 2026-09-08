# 面向港口运营状态的集装箱运价预测模型

本仓库整理了一个面向跨太平洋航线的 **集装箱运价预测研究项目**，包含数据处理流程、港口运营指标构造、机器学习与深度学习模型代码，以及各模型预测结果。

本项目对应论文：

> *Research on Container Freight Rate Prediction for Transpacific Routes Based on Multivariate LSTM: An Empirical Analysis Integrating Port Congestion and Capacity Indicators*

项目核心思想是：集装箱运价波动不仅由历史价格和宏观经济变量驱动，也会受到港口拥堵、船舶周转效率和有效运力变化的影响。因此，本项目从港口实际运营状态出发，构建港口拥堵指数和加权 TEU 运力指标，并将其纳入多变量时间序列预测模型中，用于预测中国至美国西海岸航线的 CCFI 运价指数。

## 项目亮点

- 构建 **港口拥堵指数（PCI / IPCI）**，用于刻画船舶等待时间、在泊作业时间和船舶尺度对港口拥堵程度的影响。
- 构建 **加权 TEU（Weighted TEU）** 指标，用实际作业时间修正名义运力，更贴近港口运行中的有效运力状态。
- 整合 CCFI 运价指数、港口到港数据、船舶航速、TEU、Brent 原油价格、VIX 和政策不确定性指数等多源异构数据。
- 实现多类对比模型，包括 VAR、BVAR、Decision Tree、OLS + Random Forest、XGBoost + OLS 和 LSTM。
- 主模型采用 **四周滞后特征 + 四周滑动窗口 + 多变量 LSTM**，用于捕捉运价变化中的非线性和滞后效应。
- 输出各模型预测结果，便于后续复现、对比和论文展示。

## 研究背景

2020 至 2022 年期间，受新冠疫情、供应链扰动和港口拥堵影响，中国至美国西海岸集装箱运输市场出现剧烈波动。洛杉矶港和长滩港的严重拥堵导致船舶等待时间和在泊作业时间显著增加，船舶周转效率下降，实际可用运力受到限制。

传统运价预测模型通常依赖历史价格、宏观经济指标或线性时间序列关系，难以充分刻画港口运营状态对运价形成机制的影响。本项目尝试回答以下问题：

> 港口拥堵和有效运力等运营指标，是否能够提升短期集装箱运价预测能力？

## 方法框架

项目整体流程分为三个阶段：

1. **数据处理与指标构造**
   - 清洗港口到港、船舶、运价、油价、VIX 和政策指数等原始数据。
   - 将日度或事件级数据聚合为周度数据。
   - 构建港口拥堵指数 PCI / IPCI 和加权 TEU 指标。

2. **特征工程**
   - 按周度时间戳对齐所有变量。
   - 处理缺失值，并比较不同变量变换方式。
   - 构造四周滞后特征。
   - 为 LSTM 建立四周滑动窗口输入序列。

3. **模型训练与评估**
   - 训练传统时间序列模型、机器学习模型和深度学习模型。
   - 使用 MSE 和 MAE 评估预测表现。
   - 结合特征重要性和经济含义解释模型结果。

## 核心运营指标

### 港口拥堵指数（PCI / IPCI）

港口拥堵指数用于度量港口侧的运营压力。该指标综合考虑：

- 船舶进港等待时间；
- 船舶在泊作业时间；
- 船长、船宽、型深等船舶尺度因素；
- 不同船舶对港口资源占用的差异。

简化理解如下：

```text
PCI = 加权等待压力 + 加权在泊作业压力
```

PCI 越高，说明港口拥堵越严重，船舶周转效率越低，潜在有效运力约束越强。

### 加权 TEU 运力（Weighted TEU）

加权 TEU 用船舶实际作业时间修正名义 TEU 运力，反映港口运行状态下的有效运力占用情况：

```text
Weighted TEU = sum(TEU_i * operation_hours_i / 24)
```

相比简单统计船舶数量或名义 TEU，加权 TEU 更能反映实际运营过程中的运力供给状态。

## 仓库结构

```text
.
├── README.md
├── data/
│   ├── original data/                 # 原始运价、港口、船舶、油价、VIX 和政策数据
│   ├── processed data/                # 清洗后和合并后的周度数据
│   ├── BVAR 所用数据/                  # BVAR 模型输入数据
│   ├── LSTM模型所用数据/               # LSTM 模型输入数据
│   ├── Codes for processing data/     # 数据处理与指标构造代码
│   └── 数据探索代码/                   # 探索性数据分析代码
├── models/
│   ├── LSTM model/
│   │   ├── LSTM.py                    # 多变量 LSTM 主模型
│   │   ├── Tuning.py                  # Optuna 超参数调优
│   │   └── Compare_transforms.py      # 特征变换对比实验
│   ├── var.py                         # VAR 基准模型
│   ├── bvar.Rmd                       # BVAR 基准模型
│   ├── Decision Tree1.py              # 决策树模型
│   ├── OLS+RF残差拟合模型.py           # OLS + 随机森林残差拟合模型
│   └── XGBoost-OSL Stacking模型.py     # XGBoost + OLS Stacking 模型
└── results/
    ├── ccfi_predictions_LSTM.csv
    ├── ccfi_predictions_BVAR.csv
    ├── CCFI_predictions_VAR.csv
    ├── ccfi_predictions_OLS+RF.csv
    └── ccfi_predictions_XGBoost+OLS.csv
```

## 主要数据文件

| 文件 | 说明 |
|---|---|
| `data/processed data/weekly_with_CCFI_updated.csv` | 周度 CCFI、IPCI、Brent 油价、政策指数和加权油价-VIX 指标 |
| `data/processed data/weekly_with_ccfi_full.csv` | 包含 CCFI、政策指数、加权 TEU、油价-VIX、IPCI 和平均航速的综合周度数据 |
| `data/processed data/weekly_weighted_teu.csv` | 周度加权 TEU 运力指标 |
| `data/processed data/weekly_speed.csv` | 周度船舶航速和航行时间相关特征 |
| `data/LSTM模型所用数据/` | LSTM 模型使用的数据 |
| `data/BVAR 所用数据/` | BVAR 模型使用的数据 |

## 模型说明

| 模型 | 作用 |
|---|---|
| LSTM | 主模型，用于捕捉多变量时间序列中的非线性和滞后关系 |
| VAR | 传统线性时间序列基准模型 |
| BVAR | 贝叶斯 VAR 模型，用于滚动预测对比 |
| Decision Tree | 非线性机器学习基准模型 |
| OLS + RF | 先用 OLS 拟合线性关系，再用随机森林拟合残差 |
| XGBoost + OLS | 基于 XGBoost 和线性回归的 Stacking 类模型 |

## LSTM 模型配置

主模型采用多变量 LSTM 结构，主要设置如下：

- 输入变量使用四周滞后特征；
- 输入序列采用四周滑动窗口；
- 网络结构为两层 LSTM；
- 隐藏层维度为 64；
- 使用 Dropout 防止过拟合；
- 使用 Adam 优化器；
- 使用 early stopping 控制训练过程；
- 使用 MSE 和 MAE 作为评价指标。

论文中最终优化后的主要参数如下：

| 参数 | 取值 |
|---|---:|
| LSTM 层数 | 2 |
| Hidden size | 64 |
| Dropout | 0.191 |
| Learning rate | 0.0146 |
| Maximum epochs | 500 |

## 预测结果

根据仓库中已有预测结果文件重新计算，得到以下指标：

| 结果文件 | 样本数 | MSE | MAE |
|---|---:|---:|---:|
| `ccfi_predictions_LSTM.csv` | 29 | 28,057.03 | 132.35 |
| `ccfi_predictions_BVAR.csv` | 30 | 93,545.47 | 258.32 |
| `ccfi_predictions_OLS+RF.csv` | 26 | 143,778.93 | 323.81 |
| `ccfi_predictions_XGBoost+OLS.csv` | 26 | 145,147.20 | 293.77 |
| `CCFI_predictions_VAR.csv` | 34 | 7,832.33 | 78.08 |

注意：VAR 结果使用了不同的测试窗口，因此不宜直接作为与 LSTM 完全一致口径下的模型对比结果。若进行正式模型比较，应统一训练集、测试集和滚动预测设置。

## 快速开始

安装 Python 依赖：

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn torch optuna xgboost openpyxl
```

BVAR 模型需要使用 R 环境，并安装以下依赖：

```r
install.packages(c("BVAR", "readr", "readxl", "lubridate", "dplyr", "ggplot2"))
```

运行 LSTM 主模型：

```bash
python "models/LSTM model/LSTM.py"
```

运行超参数调优：

```bash
python "models/LSTM model/Tuning.py"
```

运行特征变换对比实验：

```bash
python "models/LSTM model/Compare_transforms.py"
```

## 复现说明

- 部分脚本仍保留原始研究阶段的本地路径，包括 Windows 风格路径。换机运行前需要修改数据路径。
- 文件夹和文件名中包含空格与中文字符，命令行运行时建议使用引号包裹路径。
- 仓库包含原始数据和处理后数据。公开发布前，请确认相关数据是否具备公开共享权限。
- 不同基准模型目前可能使用不同测试窗口或滚动预测设置。若要做严格横向比较，应先统一评估口径。
- 论文中主要建模数据基于 2020 至 2022 年的周度对齐样本，并在缺失值处理后用于模型训练和测试。

## 结果解释

实验结果表明，港口运营指标能够为运价预测提供有价值的信息：

- 滞后 CCFI 体现了运价序列较强的自回归特征；
- PCI 与港口拥堵压力相关，拥堵加剧可能推高运价；
- Weighted TEU 反映有效运力供给，运力增加通常会缓解运价压力；
- 平均航速体现航线运营效率；
- 油价、VIX 和政策不确定性指标代表成本与宏观市场不确定性渠道。

总体来看，将港口运营数据转化为可解释的时间序列特征，并引入深度学习预测模型，有助于提升集装箱运价预测表现，也能够增强模型结果的经济解释力。

## 引用

如果使用本项目代码、数据处理流程或研究思路，请引用对应论文：

```bibtex
@inproceedings{duan2026container,
  title = {Research on Container Freight Rate Prediction for Transpacific Routes Based on Multivariate LSTM: An Empirical Analysis Integrating Port Congestion and Capacity Indicators},
  author = {Duan, Junli and Huang, Guanhao and Liu, Siyu},
  year = {2026},
  note = {ICTLE 2026}
}
```

## License

当前仓库暂未指定开源许可证。如需公开发布，建议在确认数据授权和代码开放范围后补充 LICENSE 文件。
