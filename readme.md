# Air Quality Classification And Prediction

基于空气污染物指标的空气质量分类项目，使用 `PyTorch` 实现了两种监督学习模型，并结合 `PCA + KMeans` 完成了数据预处理与聚类分析。

项目结果：

- 基于空气质量相关指标完成空气质量等级分类
- 对比 `MLP` 与 `1D CNN` 两种模型的分类效果，`MLP`模型以更高的准确率（98%）和更快的计算速度优于`1D CNN`模型。

    这说明对于由6种污染物浓度构成的低维表格特征，简单的MLP模型较CNN模型更符合数据结构特点，能够在不引入局部卷积假设的情况下实现高精度分类。
    
- 使用 `PCA + KMeans` 对原始数据进行降维、聚类与标签辅助构建，ARI、NMI均在0.8以上，生成可视化混淆矩阵图。

## 项目简介

本项目以空气质量数据为基础，输入特征包括：

- `PM2.5`
- `PM10`
- `SO2`
- `NO2`
- `CO`
- `O3`

项目中使用了两类标签：

- `fcm`：原始数据中的空气质量类别标签
- `class`：通过 `KMeans` 聚类后生成的聚类标签

其中，`weather.csv` 为原始数据文件，`weather_new_test.csv` 为经过聚类处理后生成的新数据文件，后续 `MLP` 和 `CNN` 模型均基于该文件进行训练与测试。

## 项目特点

- 使用 `PCA` 将原始特征降维到二维，便于可视化分析
- 使用 `KMeans` 对空气质量样本进行 6 类聚类
- 使用匈牙利算法对聚类标签与真实标签进行对齐
- 使用 `MLP` 和 `1D CNN` 两种网络结构进行分类建模
- 项目中已包含训练完成的模型权重，便于直接测试和复现

## 数据说明

### 1. 原始数据

文件路径：

`data/weather.csv`

字段说明：

| 字段名 | 含义 |
| --- | --- |
| `PM2.5` | 细颗粒物浓度 |
| `PM10` | 可吸入颗粒物浓度 |
| `So2` | 二氧化硫浓度 |
| `No2` | 二氧化氮浓度 |
| `Co` | 一氧化碳浓度 |
| `O3` | 臭氧浓度 |
| `fcm` | 原始类别标签 |

### 2. 聚类后的数据

文件路径：

`data/weather_new_test.csv`

字段说明：

| 字段名 | 含义 |
| --- | --- |
| 前 6 列 | 空气质量特征 |
| `fcm` | 原始标签 |
| `class` | `KMeans` 聚类标签 |

当前数据集共 `1816` 条样本，代码中的划分方式为：

- 训练集：前 `1499` 条样本
- 测试集：后 `317` 条样本

## 模型说明

### 1. PCA + KMeans

文件路径：

`models/pca+k_means/k_means`

主要流程：

1. 读取 `weather.csv`
2. 提取前 6 个污染物指标作为特征
3. 使用 `PCA(n_components=2)` 进行降维
4. 使用 `KMeans(n_clusters=6)` 进行聚类
5. 将聚类结果写入 `weather_new_test.csv`
6. 计算并输出聚类评估指标：
   - `ARI`
   - `NMI`
   - 混淆矩阵
7. 生成可视化图像

输出结果包括：

- `models/pca+k_means/Kmeans_original_X.png`
- `models/pca+k_means/confusion matrix.png`

### 2. MLP 分类模型

文件路径：

`models/mlp/test_mlp.py`

网络结构：

- 输入层：6 维特征
- 隐藏层 1：`Linear(6, 64) + ReLU`
- 隐藏层 2：`Linear(64, 128) + ReLU`
- 输出层：`Linear(128, 6)`

训练设置：

- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam`
- 学习率：`1e-3`
- 训练轮数：`100`
- 批大小：`50`

模型保存路径：

`models/model_mlp.pth`

### 3. 1D CNN 分类模型

文件路径：

`models/cnn/test_con.py`

网络结构：

- 输入形状：`[batch_size, 1, 6]`
- 卷积层 1：`Conv1d(1, 64, kernel_size=3, padding=1) + ReLU`
- 卷积层 2：`Conv1d(64, 128, kernel_size=3, padding=1) + ReLU`
- 全连接层：`Linear(128 * 6 * 1, 128) + ReLU + Linear(128, 6)`

训练设置：

- 损失函数：`CrossEntropyLoss`
- 优化器：`Adam`
- 学习率：`1e-3`
- 训练轮数：`100`
- 批大小：`50`

模型保存路径：

`models/model.pth`

## 项目结构

```text
air_quality_classification_and_prediction/
├─ data/
│  ├─ weather.csv
│  └─ weather_new_test.csv
├─ models/
│  ├─ cnn/
│  │  └─ test_con.py
│  ├─ mlp/
│  │  └─ test_mlp.py
│  ├─ pca+k_means/
│  │  ├─ k_means
│  │  ├─ Kmeans_original_X.png
│  │  └─ confusion matrix.png
│  ├─ model.pth
│  └─ model_mlp.pth
└─ readme.md
```

## 环境依赖

建议使用 `Python 3.9+`，主要依赖如下：

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy
```

## 运行方式

请在项目根目录下运行以下脚本，否则相对路径可能无法正确读取 `data` 目录中的文件。

### 1. 执行 PCA + KMeans 聚类

```bash
python ./models/pca+k_means/k_means
```

运行后将：

- 生成聚类标签
- 输出 `ARI`、`NMI` 等评估指标
- 更新 `data/weather_new_test.csv`
- 显示聚类散点图和混淆矩阵图

### 2. 训练并测试 MLP 模型

```bash
python ./models/mlp/test_mlp.py
```

运行后将：

- 读取 `weather_new_test.csv`
- 对特征进行标准化
- 完成模型训练
- 保存模型到 `models/model_mlp.pth`
- 输出测试准确率和测试耗时

### 3. 训练并测试 CNN 模型

```bash
python ./models/cnn/test_con.py
```

运行后将：

- 读取 `weather_new_test.csv`
- 对特征进行标准化
- 完成模型训练
- 保存模型到 `models/model.pth`
- 输出测试集逐样本预测结果与总体准确率

## 实现流程

整个项目的大致流程如下：

1. 准备原始空气质量数据 `weather.csv`
2. 使用 `PCA + KMeans` 对数据进行聚类分析
3. 生成带聚类标签的新数据文件 `weather_new_test.csv`
4. 使用新数据分别训练 `MLP` 和 `CNN` 模型
5. 在测试集上评估模型分类效果

## 可视化结果

项目中已经提供了部分实验图像：

- `Kmeans_original_X.png`：PCA 降维后的聚类分布图
- `confusion matrix.png`：聚类标签与原始标签对齐后的混淆矩阵

## 注意事项

- 项目脚本使用相对路径读取数据，运行时请确保当前工作目录为项目根目录
- `weather_new_test.csv` 由聚类脚本生成，若重新运行聚类脚本，文件内容可能被覆盖
- 当前仓库中已包含训练好的模型权重，可直接用于加载和测试
- 如果需要进一步优化结果，可以尝试调整网络结构、训练轮数、学习率或数据划分方式

## 后续可改进方向

- 增加更规范的训练、验证、测试集划分
- 将训练代码与测试代码拆分为独立模块
- 增加模型评估指标，如精确率、召回率、F1-score
- 增加参数配置文件，提升实验可复现性
- 为项目补充 `requirements.txt` 和实验结果记录

## 总结

本项目围绕空气质量分类任务，完成了从数据聚类分析到深度学习分类建模的完整流程。通过 `PCA + KMeans`、`MLP` 和 `1D CNN` 三部分内容，可以较完整地展示空气质量数据处理、聚类分析以及分类预测的基本实践过程。
对于由6种污染物浓度构成的低维表格特征，简单的MLP模型较CNN模型更符合数据结构特点，能够在不引入局部卷积假设的情况下实现高精度分类。
