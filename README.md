# Diffusion-based Tabular Data Synthesis

基于扩散模型的表格数据合成系统，支持生成高质量合成表格数据。

## 特性

- 🔥 **多种生成模型**: TabDDPM、CTABGAN、CTABGAN+、TVAE、SMOTE
- 🛠 **灵活的预处理**: 支持多种表格数据预处理策略
- 📊 **完整评估体系**: ML-efficacy + 统计相似性评估
- 🚀 **多数据集支持**: 内置多个基准数据集配置
- 🎮 **GPU 加速**: 支持 NVIDIA GPU 全量训练

## 环境要求

- **操作系统**: Linux (Ubuntu 22.04)
- **GPU**: NVIDIA GPU（已验证 RTX 3090, CUDA 11.8）
- **Python**: 3.9.x
- **包管理器**: [uv](https://github.com/astral-sh/uv)

## 快速开始

### 1. 安装 NVIDIA 驱动

如果 `nvidia-smi` 无法运行，需要加载内核模块：

```bash
depmod -a
modprobe nvidia
modprobe nvidia-uvm
modprobe nvidia-modeset
modprobe nvidia-drm
nvidia-smi  # 验证 GPU 可用
```

> 如果驱动未安装，执行 `apt-get install -y nvidia-driver-535`（需绕过代理，见下方说明）

### 2. 环境配置

使用 `uv` 包管理器：

```bash
cd /root/tabular-data-synthesis

# 创建虚拟环境
uv venv --python 3.9
source .venv/bin/activate

# 安装 PyTorch（带 CUDA 支持）
uv pip install torch==1.13.1 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 numpy（必须 <2，兼容 torch 1.x）
uv pip install "numpy<2" --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其余依赖
uv pip install table-evaluator catboost category-encoders dython icecream libzero \
    optuna pandas pyarrow rtdl scikit-learn scipy skorch tomli-w tomli tqdm \
    imbalanced-learn rdt "setuptools<70" \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 以可编辑模式安装项目
uv pip install -e . --no-deps --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

> ⚠️ 所有 `uv pip install` 命令必须加 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple`，因为容器无法访问外网。

### 3. 修复兼容性问题

安装完成后，需要修复以下兼容性问题：

#### 3.1 修复 scikit-plot 的 scipy.interp 兼容性

scipy 1.10+ 移除了 `interp` 函数，scikit-plot 仍在使用。修改两个文件：

**文件 1**: `.venv/lib/python3.9/site-packages/scikitplot/metrics.py`

```python
# 将第 27 行：
from scipy import interp
# 替换为：
try:
    from scipy import interp
except ImportError:
    from numpy import interp
```

**文件 2**: `.venv/lib/python3.9/site-packages/scikitplot/plotters.py`

```python
# 将第 28 行：
from scipy import interp
# 替换为：
try:
    from scipy import interp
except ImportError:
    from numpy import interp
```

### 4. 准备数据集

数据集已存放在项目根目录 `data/` 下，需要创建符号链接到 `src/tabsynth/data/`：

```bash
cd /root/tabular-data-synthesis
rm -rf src/tabsynth/data
ln -s ../../data src/tabsynth/data
```

验证数据可访问：

```bash
ls src/tabsynth/data/adult/  # 应看到 X_num_train.npy, X_cat_train.npy, y_train.npy 等
```

### 5. 运行 Pipeline

```bash
# 激活环境
source .venv/bin/activate

# Debug 模式（默认，CPU 快速验证）
python src/tabsynth/scripts/pipeline.py \
    --config src/tabsynth/exp/adult/config.toml \
    --train --sample --eval

# 全量 GPU 训练模式（推荐）
AZUREML_RUN_ID=1 python src/tabsynth/scripts/pipeline.py \
    --config src/tabsynth/exp/adult/config.toml \
    --train --sample --eval
```

或使用脚本：

```bash
./run_dataset.sh adult --train --sample --eval

# 全量 GPU 模式
AZUREML_RUN_ID=1 ./run_dataset.sh adult --train --sample --eval
```

## 可用数据集

| 数据集 | 任务类型 | 样本数 | 特征数 | 说明 |
|--------|----------|--------|--------|------|
| adult | 二分类 | 26,048 | 14 | 收入预测 |
| cardio | 二分类 | 44,800 | 11 | 心血管疾病预测 |
| churn2 | 二分类 | 6,400 | 11 | 客户流失预测 |
| california | 回归 | 13,209 | 8 | 加州房价预测 |
| diabetes | 二分类 | 491 | 8 | 糖尿病预测 |
| abalone | 回归 | 2,672 | 8 | 鲍鱼年龄预测 |

运行其他数据集：

```bash
AZUREML_RUN_ID=1 ./run_dataset.sh cardio --train --sample --eval
AZUREML_RUN_ID=1 ./run_dataset.sh churn2 --train --sample --eval
AZUREML_RUN_ID=1 ./run_dataset.sh california --train --sample --eval
```

## Debug 模式 vs 完整训练模式

| 模式 | 扩散步数 | 训练步数 | 设备 | 采样数量 | 适用场景 |
|------|----------|----------|------|----------|----------|
| **Debug 模式** | 10 | 2 | CPU | 10000 | 快速验证流程、调试代码 |
| **完整训练模式** | 1000 | 5000-10000 | GPU | 原始数据量 | 正式训练、生产环境 |

### 切换方式

**方式一：环境变量（推荐）**

```bash
# Debug 模式（默认）
python src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/adult/config.toml --train --sample --eval

# 完整训练模式
AZUREML_RUN_ID=1 python src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/adult/config.toml --train --sample --eval
```

**方式二：修改代码**

编辑 `src/tabsynth/lib/variables.py`：

```python
RUNS_IN_CLOUD = True  # 强制启用完整训练模式
```

## 项目结构

```
tabular-data-synthesis/
├── data/                        # 数据目录（实际数据存放位置）
│   └── <dataset>/               # 各数据集文件夹
│       ├── X_num_*.npy          # 数值特征
│       ├── X_cat_*.npy          # 类别特征
│       ├── y_*.npy              # 标签
│       └── info.json            # 数据集元信息
├── src/tabsynth/
│   ├── data/                    # → 符号链接到 ../../data
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
├── run_dataset.sh               # 便捷运行脚本
├── pyproject.toml               # 项目配置和依赖定义
└── requirements.lock            # 精确版本锁定
```

## 配置参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `num_timesteps` | 扩散步数 | 1000 |
| `scheduler` | 噪声调度器 | "cosine" |
| `steps` | 训练步数 | 5000-10000 |
| `batch_size` | 批大小 | 4096 |
| `device` | 训练设备 | "cuda:0" |
| `tabular_processor.type` | 预处理策略 | "identity" / "bgm" / "ft" |

## 添加自定义数据集

### 1. 准备数据文件

将数据放入 `data/<your_dataset>/`：

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
AZUREML_RUN_ID=1 python src/tabsynth/scripts/pipeline.py \
    --config src/tabsynth/exp/<your_dataset>/config.toml \
    --train --sample --eval
```

## 评估指标

### ML-efficacy

使用合成数据训练分类器，在真实测试集上评估性能：
- Accuracy
- F1-score
- ROC-AUC

### 统计相似性

- `basic_score`: 均值/方差/中位数相似度
- `corr_score`: 特征相关性保持
- `ml_score`: ML-efficacy 一致性
- `sup_score`: 分布覆盖度

## 已知问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `ImportError: cannot import name 'interp' from 'scipy'` | scipy 1.10+ 移除了 `interp` | 修改 scikit-plot 源码，使用 `numpy.interp` 替代 |
| `ImportError: cannot import name 'compute_associations'` | dython 新版 API 变更 | 已在项目代码中适配为 `associations()` |
| `InvalidParameterError: 'subsample' must be an int` | sklearn 新版参数类型检查更严格 | 已修复为 `int(1e9)` |
| `TypeError: unexpected keyword argument 'squared'` | sklearn 1.4+ 移除了 `squared` 参数 | 已修复为 `np.sqrt(mean_squared_error(...))` |
| `ModuleNotFoundError: No module named 'pkg_resources'` | setuptools >= 70 不再默认包含 | 安装 `setuptools<70` |
| apt 无法下载包 | 容器代理配置问题 | 运行 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY` 后再执行 apt |

## 已安装的依赖版本

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 1.13.1+cu117 | CUDA 11.7 支持 |
| numpy | 1.26.4 | 必须 <2，兼容 torch |
| scipy | 1.13.1 | |
| catboost | 1.2.10 | |
| scikit-learn | 1.6.1 | |
| pandas | 2.3.3 | |
| dython | 0.5.1 | |
| table-evaluator | 1.6.1 | |
| rtdl | 0.0.13 | |
| setuptools | 69.5.1 | 必须 <70 |

## 参考文献

- [TabDDPM: Modelling Tabular Data with Diffusion Models](https://arxiv.org/abs/2209.15421)
- [CTAB-GAN+: Enhancing Tabular Data Synthesis](https://arxiv.org/abs/2204.00401)
- [TabSynDex: A Universal Metric for Robust Evaluation of Synthetic Tabular Data](https://arxiv.org/abs/2207.05295)
