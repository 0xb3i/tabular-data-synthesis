"""
展示项目的输入和输出数据格式
"""
import numpy as np
import json
import os

def show_input_data():
    """展示输入数据的格式和内容"""
    print("=" * 80)
    print("【输入数据格式】")
    print("=" * 80)

    data_path = "src/tabsynth/data/adult"

    # 读取 info.json
    with open(f"{data_path}/info.json", 'r') as f:
        info = json.load(f)

    print("\n1. info.json - 数据集元信息:")
    print("-" * 40)
    print(json.dumps(info, indent=2, ensure_ascii=False))

    # 加载 numpy 数据
    X_num_train = np.load(f"{data_path}/X_num_train.npy", allow_pickle=True)
    X_cat_train = np.load(f"{data_path}/X_cat_train.npy", allow_pickle=True)
    y_train = np.load(f"{data_path}/y_train.npy", allow_pickle=True)

    print("\n2. 数值特征 (X_num_train.npy):")
    print("-" * 40)
    print(f"   Shape: {X_num_train.shape}")
    print(f"   Dtype: {X_num_train.dtype}")
    print(f"   前5行样本:")
    print(f"   {X_num_train[:5]}")

    print("\n3. 类别特征 (X_cat_train.npy):")
    print("-" * 40)
    print(f"   Shape: {X_cat_train.shape}")
    print(f"   Dtype: {X_cat_train.dtype}")
    print(f"   前5行样本:")
    print(f"   {X_cat_train[:5]}")

    print("\n4. 标签 (y_train.npy):")
    print("-" * 40)
    print(f"   Shape: {y_train.shape}")
    print(f"   Dtype: {y_train.dtype}")
    print(f"   前10个标签: {y_train[:10]}")
    print(f"   类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # 读取列配置
    with open("src/tabsynth/CTABGAN_Plus/columns.json", 'r') as f:
        columns_config = json.load(f)

    adult_config = columns_config["adult"]

    print("\n5. 列配置 (columns.json):")
    print("-" * 40)
    print(f"   类别列索引: {adult_config['categorical_columns']}")
    print(f"   数值列索引: {adult_config['integer_columns']}")
    print(f"   混合列: {adult_config['mixed_columns']}")
    print(f"   任务类型: {adult_config['problem_type']}")

    print("\n" + "=" * 80)
    print("【Adult 数据集列映射】")
    print("=" * 80)

    # 列名映射 (Adult 数据集的实际列名)
    adult_column_names = {
        '0': 'age',           # 年龄
        '1': 'fnlwgt',        # 权重
        '2': 'education-num', # 教育年限
        '3': 'capital-gain',  # 资本收益
        '4': 'capital-loss',  # 资本损失
        '5': 'hours-per-week',# 每周工作时长
        '6': 'workclass',     # 工作类型
        '7': 'education',     # 教育程度
        '8': 'marital-status',# 婚姻状态
        '9': 'occupation',    # 职业
        '10': 'relationship', # 家庭关系
        '11': 'race',         # 种族
        '12': 'sex',          # 性别
        '13': 'native-country',# 原籍国家
        'y': 'income'         # 收入 (>50K 或 <=50K)
    }

    print("\n数值特征 (X_num) - 6列:")
    for i, idx in enumerate(['0', '1', '2', '3', '4', '5']):
        print(f"   X_num[:, {i}] → {adult_column_names[idx]}")
    print(f"   实际第1行数据: {X_num_train[0]}")

    print("\n类别特征 (X_cat) - 8列:")
    for i, idx in enumerate(['6', '7', '8', '9', '10', '11', '12', '13']):
        print(f"   X_cat[:, {i}] → {adult_column_names[idx]}")
    print(f"   实际第1行数据: {X_cat_train[0]}")

    print("\n标签 (y):")
    print(f"   y → {adult_column_names['y']}")
    print(f"   0: 收入 <=50K, 1: 收入 >50K")

def show_data_structure():
    """展示数据结构图"""
    print("\n")
    print("=" * 80)
    print("【数据结构详解】")
    print("=" * 80)

    structure = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输入数据文件结构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  src/tabsynth/data/[dataset_name]/                                          │
│  ├── info.json              元信息 (任务类型、特征数量、数据大小)              │
│  ├── X_num_train.npy        训练集数值特征  shape: (n_train, n_num)          │
│  ├── X_num_val.npy          验证集数值特征  shape: (n_val, n_num)            │
│  ├── X_num_test.npy         测试集数值特征  shape: (n_test, n_num)           │
│  ├── X_cat_train.npy        训练集类别特征  shape: (n_train, n_cat)          │
│  ├── X_cat_val.npy          验证集类别特征  shape: (n_val, n_cat)            │
│  ├── X_cat_test.npy         测试集类别特征  shape: (n_test, n_cat)           │
│  ├── y_train.npy            训练集标签      shape: (n_train,)               │
│  ├── y_val.npy              验证集标签      shape: (n_val,)                 │
│  ├── y_test.npy             测试集标签      shape: (n_test,)                │
│  └── idx_*.npy              索引文件 (可选)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        Adult 数据集具体示例                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【X_num_train.npy】数值特征矩阵                                            │
│  Shape: (26048, 6)                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ age  fnlwgt   edu-num  capital-gain  capital-loss  hours-per-week │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ 32   228265   9        0             0              30            │   │
│  │ 21   89154    7        0             0              42            │   │
│  │ 33   43716    10       0             0              4             │   │
│  │ ...                                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【X_cat_train.npy】类别特征矩阵                                            │
│  Shape: (26048, 8)                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ workclass  education  marital  occupation  relationship  ...      │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │ Private   HS-grad   Never   Handlers    Own-child    ...          │   │
│  │ Private   11th      Never   Other       Own-child    ...          │   │
│  │ State-gov Some-col  Married Craft       Husband      ...          │   │
│  │ ...                                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  【y_train.npy】标签向量                                                    │
│  Shape: (26048,)                                                            │
│  [0, 0, 0, 0, 1, 0, 0, ...]  (0: <=50K, 1: >50K)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    print(structure)

def show_output_data():
    """展示输出数据的格式和内容"""
    print("\n")
    print("=" * 80)
    print("【输出数据格式】(合成数据)")
    print("=" * 80)

    # 检查是否有输出目录
    output_dirs = []
    if os.path.exists("outputs"):
        for root, dirs, files in os.walk("outputs"):
            if "X_num_train.npy" in files or "X_cat_train.npy" in files:
                output_dirs.append(root)

    if output_dirs:
        output_path = output_dirs[0]
        print(f"\n从 {output_path} 加载合成数据:")

        if os.path.exists(f"{output_path}/X_num_train.npy"):
            X_num_syn = np.load(f"{output_path}/X_num_train.npy", allow_pickle=True)
            print(f"\n1. 合成数值特征 (X_num_train.npy):")
            print("-" * 40)
            print(f"   Shape: {X_num_syn.shape}")
            print(f"   前3行: {X_num_syn[:3]}")

        if os.path.exists(f"{output_path}/X_cat_train.npy"):
            X_cat_syn = np.load(f"{output_path}/X_cat_train.npy", allow_pickle=True)
            print(f"\n2. 合成类别特征 (X_cat_train.npy):")
            print("-" * 40)
            print(f"   Shape: {X_cat_syn.shape}")
            print(f"   前3行: {X_cat_syn[:3]}")

        if os.path.exists(f"{output_path}/y_train.npy"):
            y_syn = np.load(f"{output_path}/y_train.npy", allow_pickle=True)
            print(f"\n3. 合成标签 (y_train.npy):")
            print("-" * 40)
            print(f"   Shape: {y_syn.shape}")
            print(f"   类别分布: {dict(zip(*np.unique(y_syn, return_counts=True)))}")

        # 检查评估结果
        if os.path.exists(f"{output_path}/results_similarity.json"):
            with open(f"{output_path}/results_similarity.json", 'r') as f:
                results = json.load(f)
            print(f"\n4. 评估结果 (results_similarity.json):")
            print("-" * 40)
            if 'sim_score' in results:
                for k, v in results['sim_score'].items():
                    print(f"   {k}: {v}")
    else:
        print("\n暂无生成的合成数据输出。")
        print("运行以下命令生成合成数据:")
        print("  uv run python src/tabsynth/scripts/pipeline.py --config src/tabsynth/exp/adult/config.toml --train --sample --eval")

def show_output_structure():
    """展示输出结构图"""
    output_structure = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输出数据文件结构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  outputs/[experiment_name]/                                                  │
│  ├── config.toml            实验配置文件副本                                  │
│  ├── info.json              数据集元信息副本                                  │
│  ├── model.pt               训练好的扩散模型                                  │
│  ├── model_ema.pt           EMA版本的模型                                    │
│  ├── loss.csv               训练损失记录                                      │
│  │                                                                         │
│  ├── 【合成数据】                                                            │
│  │   ├── X_num_train.npy    合成数值特征  shape: (n_samples, n_num)          │
│  │   ├── X_cat_train.npy    合成类别特征  shape: (n_samples, n_cat)          │
│  │   └── y_train.npy        合成标签      shape: (n_samples,)               │
│  │                                                                         │
│  ├── 【中间数据】                                                            │
│  │   ├── X_num_unnorm.npy   标准化逆变换前的数值数据                          │
│  │   └── X_cat_unnorm.npy   编码逆变换前的类别数据                            │
│  │                                                                         │
│  └── 【评估结果】                                                            │
│      ├── results_similarity.json  TabSynDex 相似度评估结果                   │
│      ├── train_synthetic.csv     合成训练数据 DataFrame                      │
│      ├── test_real.csv           真实测试数据 DataFrame                      │
│      └── plots/                  可视化图表目录                               │
│          ├── column_comparison.png                                         │
│          └── ...                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    print(output_structure)

if __name__ == '__main__':
    show_input_data()
    show_data_structure()
    show_output_data()
    show_output_structure()
