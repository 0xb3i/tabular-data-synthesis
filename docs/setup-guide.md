# 环境配置与复现指南

本文档记录了使用 `uv` 配置项目环境并运行 minimal 流程的完整步骤，以及遇到的所有问题和解决方案。

## 环境信息

- **操作系统**: macOS (Apple Silicon)
- **Python**: 3.9.6
- **包管理器**: uv 0.11.6+

---

## 快速开始

### 1. 创建虚拟环境

```bash
uv venv --python 3.9
source .venv/bin/activate
```

### 2. 安装依赖

```bash
# 安装主要依赖（放宽版本限制以兼容 Apple Silicon）
uv pip install table_evaluator catboost category-encoders dython icecream libzero numpy optuna pandas pyarrow rtdl scikit-learn scipy skorch tomli-w tomli tqdm imbalanced-learn rdt torch

# 安装 setuptools（旧版，提供 pkg_resources）
uv pip install "setuptools<70"

# 以可编辑模式安装项目
uv pip install -e . --no-deps
```

### 3. 下载数据集

```bash
curl -L -o /tmp/data.tar "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=1"
tar -xf /tmp/data.tar -C src/tabsynth/
```

### 4. 应用代码修复

需要修改以下文件以解决兼容性问题：

#### 4.1 修复 dython 的 scipy.interp 兼容性

文件: `.venv/lib/python3.9/site-packages/dython/model_utils.py`

```python
# 将第 3 行：
from scipy import interp

# 替换为：
try:
    from scipy import interp
except ImportError:
    from numpy import interp
```

#### 4.2 修复 sklearn QuantileTransformer 参数

文件: `src/tabsynth/lib/data.py` 第 221 行

```python
# 将：
subsample=1e9,

# 替换为：
subsample=int(1e9),
```

#### 4.3 修复 sklearn mean_squared_error 参数

文件: `src/tabsynth/evaluation/tabsyndex.py` 第 236-239 行

```python
# 将：
r_rmse = [sk.mean_squared_error(real_y, estimator.predict(real_x), squared=False) for estimator in r_estimators]
r_rmse += [sk.mean_squared_error(fake_y, estimator.predict(fake_x), squared=False) for estimator in r_estimators]
f_rmse = [sk.mean_squared_error(real_y, estimator.predict(real_x), squared=False) for estimator in f_estimators]
f_rmse += [sk.mean_squared_error(fake_y, estimator.predict(fake_x), squared=False) for estimator in f_estimators]

# 替换为：
r_rmse = [np.sqrt(sk.mean_squared_error(real_y, estimator.predict(real_x))) for estimator in r_estimators]
r_rmse += [np.sqrt(sk.mean_squared_error(fake_y, estimator.predict(fake_x))) for estimator in r_estimators]
f_rmse = [np.sqrt(sk.mean_squared_error(real_y, estimator.predict(real_x))) for estimator in f_estimators]
f_rmse += [np.sqrt(sk.mean_squared_error(fake_y, estimator.predict(fake_x))) for estimator in f_estimators]
```

#### 4.4 添加缺失的 dataset_config

文件: `src/tabsynth/data/adult/info.json`

```json
{
    "name": "Adult",
    "id": "adult--default",
    "task_type": "binclass",
    "n_num_features": 6,
    "n_cat_features": 8,
    "test_size": 16281,
    "train_size": 26048,
    "val_size": 6513,
    "dataset_config": {
        "cat_columns": ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"],
        "non_cat_columns": [],
        "log_columns": [],
        "general_columns": ["age"],
        "mixed_columns": {
            "capital-loss": [0.0],
            "capital-gain": [0.0]
        },
        "int_columns": ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
        "problem_type": "binclass",
        "target_column": "income"
    }
}
```

#### 4.5 修改 Python 版本要求

文件: `setup.cfg` 第 28 行

```ini
# 将：
python_requires = >=3.9.7

# 替换为：
python_requires = >=3.9
```

### 5. 运行 Pipeline

```bash
source .venv/bin/activate
python src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/adult/config.toml --train --sample --eval
```

---

## 遇到的问题详解

### 问题 1: catboost 不支持 Apple Silicon

**错误信息**:
```
catboost==1.0.3 has no wheels with a matching platform tag (e.g., `macosx_26_0_arm64`)
```

**原因**: 原始 `requirements.txt` 中指定的 catboost 1.0.3 没有 arm64 的预编译包。

**解决方案**: 不指定版本，让 uv 自动选择兼容的最新版本 (1.2.10)。

---

### 问题 2: scipy.interp 已移除

**错误信息**:
```
ImportError: cannot import name 'interp' from 'scipy'
```

**原因**: `scipy.interp` 在 scipy 1.10+ 中已被移除，但 dython 依赖此函数。

**解决方案**:
- 方案 A: 降级 scipy 到 1.9.x（但在 macOS 上难以安装）
- 方案 B: 修改 dython 源码，使用 `numpy.interp` 替代（推荐）

---

### 问题 3: pkg_resources 模块缺失

**错误信息**:
```
ModuleNotFoundError: No module named 'pkg_resources'
```

**原因**: 新版 setuptools (>=70) 不再默认包含 `pkg_resources`。

**解决方案**: 安装 setuptools < 70 版本。

---

### 问题 4: numpy 2.x 不兼容

**错误信息**:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2
```

**原因**: 部分依赖（如 torch）是用 NumPy 1.x 编译的，不兼容 NumPy 2.x。

**解决方案**: 安装 numpy<2。

---

### 问题 5: sklearn QuantileTransformer 参数类型错误

**错误信息**:
```
InvalidParameterError: The 'subsample' parameter of QuantileTransformer must be an int in the range [1, inf) or None. Got 1000000000.0 instead.
```

**原因**: 新版 sklearn 对参数类型检查更严格，`1e9` 是 float 类型。

**解决方案**: 改为 `int(1e9)`。

---

### 问题 6: sklearn mean_squared_error squared 参数已移除

**错误信息**:
```
TypeError: got an unexpected keyword argument 'squared'
```

**原因**: sklearn 1.4+ 移除了 `squared` 参数。

**解决方案**: 使用 `np.sqrt(sk.mean_squared_error(...))` 替代 `squared=False`。

---

### 问题 7: dataset_config 缺失

**错误信息**:
```
KeyError: 'dataset_config'
```

**原因**: 下载数据集的 `info.json` 缺少 `dataset_config` 字段。

**解决方案**: 手动添加完整的 `dataset_config` 配置。

---

### 问题 8: Python 版本要求过严

**错误信息**:
```
the current Python version (3.9.6) does not satisfy Python>=3.9.7
```

**原因**: `setup.cfg` 中 `python_requires = >=3.9.7` 过于严格。

**解决方案**: 放宽为 `>=3.9`。

---

## 已安装的依赖版本

| 包名 | 版本 |
|------|------|
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| torch | 1.13.1 |
| catboost | 1.2.10 |
| scikit-learn | 1.6.1 |
| pandas | 2.0.3 |
| dython | 0.5.1 |
| table-evaluator | 1.6.1 |
| setuptools | 69.5.1 |

---

## 预期输出

成功运行后，应该看到类似以下输出：

```
Selected tabular processor:  identity
...
Model Loaded Successfully!
Sample timestep    9 Sample timestep    8 ... Sample timestep    0
...
[val]
{'acc': 0.7312, 'f1': 0.5344, 'roc_auc': 0.6327}
[test]
{'acc': 0.7328, 'f1': 0.5327, 'roc_auc': 0.6368}
...
SIMILARITY SCORE
score: 0.4997421633174435
...
Elapsed time: 0:00:13
```

---

## 可用数据集

项目已配置以下数据集，可直接运行：

| 数据集 | 任务类型 | 样本数 | 特征数 | 说明 |
|--------|----------|--------|--------|------|
| adult | 二分类 | 26,048 | 14 | 收入预测 |
| cardio | 二分类 | 44,800 | 11 | 心血管疾病预测 |
| churn2 | 二分类 | 6,400 | 11 | 客户流失预测 |
| california | 回归 | 13,209 | 8 | 加州房价预测 |
| diabetes | 二分类 | 491 | 8 | 糖尿病预测 |
| abalone | 回归 | 2,672 | 8 | 鲍鱼年龄预测 |

### 运行不同数据集

**方法 1: 直接运行**
```bash
# Adult 数据集
python src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/adult/config.toml --train --sample --eval

# Cardio 数据集
python src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/cardio/config.toml --train --sample --eval

# Churn 数据集
python src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/churn2/config.toml --train --sample --eval

# California Housing 数据集
python src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/california/config.toml --train --sample --eval
```

**方法 2: 使用运行脚本**
```bash
./run_dataset.sh adult --train --sample --eval
./run_dataset.sh cardio --train --sample --eval
./run_dataset.sh churn2 --train --sample --eval
```

### 数据集配置文件

每个数据集需要两个配置文件：

1. **info.json**: `src/tabsynth/data/<dataset>/info.json`
   - 包含数据集元信息和 `dataset_config`

2. **config.toml**: `src/tabsynth/exp/<dataset>/config.toml`
   - 包含实验参数（模型、训练、采样配置）

### 添加新数据集

1. 准备数据文件（放入 `src/tabsynth/data/<new_dataset>/`）：
   - `X_num_train.npy`, `X_num_val.npy`, `X_num_test.npy`
   - `X_cat_train.npy`, `X_cat_val.npy`, `X_cat_test.npy`（如果有分类特征）
   - `y_train.npy`, `y_val.npy`, `y_test.npy`

2. 创建 `info.json`（参考现有数据集格式）

3. 创建 `config.toml`（复制现有配置并修改参数）

---

## 注意事项

1. **debug 模式**: 在本地运行时，pipeline.py 会自动启用 debug 模式（减少训练步数），这是预期的行为。

2. **table_evaluator 警告**: 最后的 table_evaluator 可能会有兼容性警告，但不影响核心流程。

3. **数据集**: 确保数据集下载完整，包含 `.npy` 文件和 `info.json`。

4. **虚拟环境**: 每次运行前确保激活虚拟环境 `source .venv/bin/activate`。

5. **无分类特征数据集**: 目前 diabetes 和 california 数据集（无分类特征）需要额外代码修复才能运行。
