# Diffusion-based Tabular Data Synthesis

基于扩散模型的表格数据合成系统，支持生成高质量合成表格数据。

## 特性

- 🔥 **多种生成模型**: TabDDPM、CTABGAN、CTABGAN+、TVAE、SMOTE
- 🛠 **灵活的预处理**: 支持多种表格数据预处理策略
- 📊 **完整评估体系**: ML-efficacy + 统计相似性评估
- 🚀 **多数据集支持**: 内置多个基准数据集配置

## 快速开始

### 1. 环境配置

使用 `uv` 包管理器（推荐）：

```bash
# 创建虚拟环境
uv venv --python 3.9
source .venv/bin/activate

# 安装依赖
uv pip install table_evaluator catboost category-encoders dython icecream libzero numpy optuna pandas pyarrow rtdl scikit-learn scipy skorch tomli-w tomli tqdm imbalanced-learn rdt torch
uv pip install "setuptools<70"
uv pip install -e . --no-deps
```

或使用 conda（传统方式）：

```bash
conda env create -f environment.yml
conda activate tabsynth
pip install -e .
```

### 2. 下载数据集

```bash
curl -L -o /tmp/data.tar "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=1"
tar -xf /tmp/data.tar -C src/tabsynth/
```

### 3. 运行 Pipeline

```bash
# 激活环境
source .venv/bin/activate

# 运行完整流程：训练 → 采样 → 评估
python src/tabsynth/scripts/pipeline.py \
    --config src/tabsynth/exp/adult/config.toml \
    --train --sample --eval
```

或使用便捷脚本：

```bash
./run_dataset.sh adult --train --sample --eval
```

## 可用数据集

| 数据集 | 任务类型 | 样本数 | 特征数 | 说明 |
|--------|----------|--------|--------|------|
| adult | 二分类 | 26,048 | 14 | 收入预测 |
| cardio | 二分类 | 44,800 | 11 | 心血管疾病预测 |
| churn2 | 二分类 | 6,400 | 11 | 客户流失预测 |
| california | 回归 | 13,209 | 8 | 加州房价预测 |

运行其他数据集：

```bash
./run_dataset.sh cardio --train --sample --eval
./run_dataset.sh churn2 --train --sample --eval
```

## 项目结构

```
Diffusion-based-Tabular-Data-Synthesis/
├── src/tabsynth/
│   ├── data/                    # 数据目录
│   │   └── <dataset>/           # 各数据集文件夹
│   │       ├── X_num_*.npy      # 数值特征
│   │       ├── X_cat_*.npy      # 类别特征
│   │       ├── y_*.npy          # 标签
│   │       └── info.json        # 数据集元信息
│   ├── exp/                     # 实验配置
│   │   └── <dataset>/
│   │       └── config.toml      # 实验参数配置
│   ├── scripts/                 # 主要脚本
│   │   ├── pipeline.py          # 完整流程
│   │   ├── train.py             # 训练
│   │   ├── sample.py            # 采样
│   │   └── eval_*.py            # 评估
│   ├── tab_ddpm/                # 扩散模型实现
│   ├── tabular_processing/      # 表格数据预处理
│   ├── evaluation/              # 评估模块
│   └── lib/                     # 工具函数
├── outputs/                     # 输出目录（自动生成）
├── docs/                        # 文档
│   └── setup-guide.md           # 详细配置指南
├── run_dataset.sh               # 便捷运行脚本
├── requirements.txt             # 依赖列表
└── setup.cfg                    # 包配置
```

## 添加自定义数据集

### 1. 准备数据文件

将数据放入 `src/tabsynth/data/<your_dataset>/`：

```
X_num_train.npy, X_num_val.npy, X_num_test.npy  # 数值特征 (n_samples, n_num_features)
X_cat_train.npy, X_cat_val.npy, X_cat_test.npy  # 类别特征 (可选)
y_train.npy, y_val.npy, y_test.npy              # 标签 (n_samples,)
info.json                                        # 元信息
```

### 2. 创建 info.json

```json
{
    "name": "Your Dataset",
    "task_type": "binclass",
    "n_num_features": 6,
    "n_cat_features": 4,
    "train_size": 10000,
    "val_size": 2000,
    "test_size": 3000,
    "dataset_config": {
        "cat_columns": ["cat1", "cat2"],
        "int_columns": ["num1", "num2"],
        "target_column": "target"
    }
}
```

### 3. 创建 config.toml

复制 `src/tabsynth/exp/adult/config.toml` 并修改参数。

### 4. 运行

```bash
python src/tabsynth/scripts/pipeline.py \
    --config src/tabsynth/exp/<your_dataset>/config.toml \
    --train --sample --eval
```

## 配置参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `num_timesteps` | 扩散步数 | 1000 |
| `scheduler` | 噪声调度器 | "cosine" |
| `steps` | 训练步数 | 5000-10000 |
| `batch_size` | 批大小 | 4096 |
| `tabular_processor.type` | 预处理策略 | "identity" / "bgm" / "ft" |

详细配置说明请参考 `docs/setup-guide.md`。

## 评估指标

### ML-efficacy

使用合成数据训练分类器，在真实测试集上评估性能：
- Accuracy
- F1-score
- ROC-AUC

### 统计相似性 (TabSynDex)

- `basic_score`: 均值/方差/中位数相似度
- `corr_score`: 特征相关性保持
- `ml_score`: ML-efficacy 一致性
- `sup_score`: 分布覆盖度

## 常见问题

### Q: 依赖安装失败？

部分依赖在 Apple Silicon 上需要特定版本，请参考 `docs/setup-guide.md` 中的详细说明。

### Q: 训练时自动进入 debug 模式？

本地运行时，`pipeline.py` 会自动减少训练步数以加速测试，这是正常行为。正式训练请修改 `RUNS_IN_CLOUD` 变量。

### Q: 如何使用 GPU？

修改 `config.toml` 中的 `device = "cuda:0"`。

## 参考文献

- [TabDDPM: Modelling Tabular Data with Diffusion Models](https://arxiv.org/abs/2209.15421)
- [CTAB-GAN+: Enhancing Tabular Data Synthesis](https://arxiv.org/abs/2204.00401)
- [TabSynDex: A Universal Metric for Robust Evaluation of Synthetic Tabular Data](https://arxiv.org/abs/2207.05295)

## License

MIT License
