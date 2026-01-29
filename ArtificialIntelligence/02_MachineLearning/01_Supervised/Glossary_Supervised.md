# Supervised Learning Glossary (监督学习词汇表)

## English-Chinese Technical Terms by Category (按分类整理的英中专业术语)

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Artificial Intelligence (AI)** | 人工智能 | 使机器具有智能行为的技术 |
| **Machine Learning (ML)** | 机器学习 | 让计算机从数据中学习的技术 |
| **Supervised Learning** | 监督学习 | 使用标注数据训练模型的机器学习方法 |
| **Unsupervised Learning** | 无监督学习 | 使用无标注数据训练模型的机器学习方法 |
| **Reinforcement Learning** | 强化学习 | 通过奖励信号学习的机器学习方法 |
| **Deep Learning** | 深度学习 | 使用多层神经网络的机器学习 |
| **Algorithm** | 算法 | 解决问题的步骤和方法 |
| **Model** | 模型 | 学习到的函数或算法 |
| **Heuristic** | 启发式 | 基于经验的问题解决方法 |

---

## 2. 数据相关 (Data-Related Terms)

| English | 中文 | 说明 |
|---------|------|------|
| **Dataset** | 数据集 | 用于训练和测试的数据集合 |
| **Sample** | 样本 | 数据集中的单个数据点 |
| **Label** | 标签 | 监督学习中的目标变量 |
| **Labeled Data** | 标注数据 | 带有正确答案的数据 |
| **Ground Truth** | 真实标签 | 数据的正确答案 |
| **Target Variable** | 目标变量 | 要预测的变量 |
| **Training Set** | 训练集 | 用于训练模型的数据 |
| **Validation Set** | 验证集 | 用于调优超参数和模型选择的数据 |
| **Test Set** | 测试集 | 用于最终评估模型的数据 |
| **Holdout Method** | 留出法 | 将数据分为训练集和测试集 |
| **Imbalanced Data** | 不平衡数据 | 类别分布不均的数据 |
| **Class Imbalance** | 类别不平衡 | 不同类别的样本数量差异很大 |
| **Noise** | 噪声 | 数据中的随机误差 |
| **Outlier** | 异常值 | 与其他数据显著不同的数据点 |
| **Sampling** | 采样 | 从总体中选择样本的过程 |
| **Data Augmentation** | 数据增强 | 通过变换生成更多训练数据 |

---

## 3. 特征工程 (Feature Engineering)

| English | 中文 | 说明 |
|---------|------|------|
| **Feature** | 特征 | 描述样本的属性或变量 |
| **Feature Engineering** | 特征工程 | 创建和选择有用特征的过程 |
| **Feature Selection** | 特征选择 | 选择最相关特征的过程 |
| **Feature Extraction** | 特征提取 | 从原始数据中提取特征 |
| **Feature Scaling** | 特征缩放 | 将特征调整到相似的尺度 |
| **Normalization** | 归一化 | 将数据缩放到特定范围 |
| **One-Hot Encoding** | 独热编码 | 将类别变量转换为二进制向量 |
| **Dimensionality Reduction** | 降维 | 减少特征数量的技术 |
| **Principal Component Analysis (PCA)** | 主成分分析 | 降维技术 |
| **Curse of Dimensionality** | 维度灾难 | 高维空间中数据稀疏导致的问题 |
| **Multicollinearity** | 多重共线性 | 特征之间高度相关的现象 |
| **Preprocessing** | 预处理 | 训练前对数据进行的处理 |

---

## 4. 回归算法 (Regression Algorithms)

| English | 中文 | 说明 |
|---------|------|------|
| **Regression** | 回归 | 预测连续值的监督学习任务 |
| **Linear Regression** | 线性回归 | 拟合线性关系的回归算法 |
| **Polynomial Regression** | 多项式回归 | 使用多项式特征的回归 |
| **Ridge Regression** | 岭回归 | L2正则化的线性回归 |
| **Lasso Regression** | Lasso回归 | L1正则化的线性回归 |
| **Residual** | 残差 | 实际值与预测值的差 |

---

## 5. 分类算法 (Classification Algorithms)

| English | 中文 | 说明 |
|---------|------|------|
| **Classification** | 分类 | 预测离散类别的监督学习任务 |
| **Classifier** | 分类器 | 执行分类任务的模型 |
| **Binary Classification** | 二分类 | 只有两个类别的分类问题 |
| **Multi-class Classification** | 多分类 | 有三个或更多类别的分类 |
| **Logistic Regression** | 逻辑回归 | 用于分类的线性模型 |
| **Decision Tree** | 决策树 | 基于树形结构的分类/回归算法 |
| **Decision Boundary** | 决策边界 | 分类器划分不同类别的边界 |
| **K-Nearest Neighbors (KNN)** | K近邻算法 | 基于距离的分类/回归算法 |
| **Support Vector Machine (SVM)** | 支持向量机 | 寻找最优分类超平面的算法 |
| **Kernel** | 核函数 | 将数据映射到高维空间的函数 |
| **Naive Bayes** | 朴素贝叶斯 | 基于贝叶斯定理的分类算法 |
| **Bayes' Theorem** | 贝叶斯定理 | 描述条件概率关系的定理 |
| **Prior Probability** | 先验概率 | 观察数据前的概率 |
| **Probability** | 概率 | 事件发生的可能性 |
| **Threshold** | 阈值 | 决策的临界值 |
| **Clustering** | 聚类 | 无监督学习中将相似样本分组的任务 |

---

## 6. 神经网络 (Neural Networks)

| English | 中文 | 说明 |
|---------|------|------|
| **Neural Network** | 神经网络 | 模仿生物神经元的计算模型 |
| **Neuron** | 神经元 | 神经网络的基本计算单元 |
| **Perceptron** | 感知机 | 最简单的神经网络模型 |
| **Input Layer** | 输入层 | 神经网络接收数据的层 |
| **Hidden Layer** | 隐藏层 | 神经网络中输入和输出之间的层 |
| **Output Layer** | 输出层 | 神经网络产生预测的层 |
| **Activation Function** | 激活函数 | 神经网络中引入非线性的函数 |
| **Sigmoid Function** | Sigmoid函数 | S形激活函数，输出范围(0,1) |
| **Softmax** | Softmax函数 | 将向量转换为概率分布 |
| **Feedforward** | 前馈 | 神经网络中信息向前传播 |
| **Backpropagation** | 反向传播 | 神经网络训练中计算梯度的算法 |
| **Weight** | 权重 | 模型中的可学习参数 |
| **Bias** | 偏差/偏置 | 模型的系统性误差；或神经网络中的偏置项 |

---

## 7. 模型训练 (Model Training)

| English | 中文 | 说明 |
|---------|------|------|
| **Training** | 训练 | 让模型从数据中学习的过程 |
| **Fitting** | 拟合 | 训练模型以匹配数据的过程 |
| **Inference** | 推理 | 使用训练好的模型进行预测 |
| **Prediction** | 预测 | 模型对新数据的输出 |
| **Parameter** | 参数 | 模型从数据中学习的变量 |
| **Batch** | 批次 | 一次训练迭代使用的样本集合 |
| **Batch Size** | 批次大小 | 每个批次包含的样本数量 |
| **Mini-Batch** | 小批量 | 介于单样本和全批量之间的批次 |
| **Epoch** | 轮次 | 完整遍历一次训练集 |
| **Iteration** | 迭代 | 训练过程中的一次参数更新 |
| **Convergence** | 收敛 | 训练过程中损失函数趋于稳定 |
| **Learning Rate** | 学习率 | 梯度下降中的步长参数 |
| **Gradient** | 梯度 | 函数变化率的向量 |
| **Gradient Descent** | 梯度下降 | 通过梯度优化参数的算法 |
| **Stochastic Gradient Descent (SGD)** | 随机梯度下降 | 使用单个样本更新参数的优化算法 |
| **Optimization** | 优化 | 寻找最佳参数的过程 |
| **Loss Function** | 损失函数 | 衡量预测误差的函数 |
| **Cost Function** | 成本函数 | 同损失函数 |
| **Objective Function** | 目标函数 | 优化过程中要最大化或最小化的函数 |
| **Cross-Entropy** | 交叉熵 | 分类问题常用的损失函数 |
| **Mean Squared Error (MSE)** | 均方误差 | 回归评估指标 |
| **Mean Absolute Error (MAE)** | 平均绝对误差 | 回归评估指标 |
| **Root Mean Squared Error (RMSE)** | 均方根误差 | MSE的平方根 |
| **Vanishing Gradient** | 梯度消失 | 训练中梯度变得过小的问题 |
| **Exploding Gradient** | 梯度爆炸 | 训练中梯度变得过大的问题 |

---

## 8. 模型评估 (Model Evaluation)

| English | 中文 | 说明 |
|---------|------|------|
| **Evaluation Metrics** | 评估指标 | 衡量模型性能的标准 |
| **Metric** | 指标 | 衡量性能的标准 |
| **Accuracy** | 准确率 | 正确预测的样本数占总样本数的比例 |
| **Precision** | 精确率 | 预测为正的样本中真正为正的比例 |
| **Recall** | 召回率 | 实际为正的样本中被正确预测的比例 |
| **Specificity** | 特异度 | 真负例率 |
| **F1-Score** | F1分数 | 精确率和召回率的调和平均 |
| **Confusion Matrix** | 混淆矩阵 | 展示分类结果的表格 |
| **True Positive (TP)** | 真正例 | 实际为正且预测为正 |
| **True Negative (TN)** | 真负例 | 实际为负且预测为负 |
| **False Positive (FP)** | 假正例 | 实际为负但预测为正 |
| **False Negative (FN)** | 假负例 | 实际为正但预测为负 |
| **Error Rate** | 错误率 | 预测错误的样本比例 |
| **R-Squared (R²)** | R平方/决定系数 | 回归模型解释方差的比例 |
| **Cross-Validation** | 交叉验证 | 评估模型泛化能力的技术 |
| **K-Fold Cross-Validation** | K折交叉验证 | 将数据分成K份进行交叉验证 |
| **Learning Curve** | 学习曲线 | 展示训练过程中性能变化的曲线 |
| **Baseline** | 基线模型 | 用于比较的简单模型 |

---

## 9. 过拟合与正则化 (Overfitting & Regularization)

| English | 中文 | 说明 |
|---------|------|------|
| **Overfitting** | 过拟合 | 模型在训练集上表现好但泛化能力差 |
| **Underfitting** | 欠拟合 | 模型过于简单，无法捕捉数据规律 |
| **Generalization** | 泛化 | 模型在新数据上的表现能力 |
| **Bias-Variance Tradeoff** | 偏差-方差权衡 | 模型复杂度与泛化能力之间的平衡 |
| **Variance** | 方差 | 模型对训练数据变化的敏感度 |
| **Regularization** | 正则化 | 防止过拟合的技术 |
| **Weight Decay** | 权重衰减 | L2正则化的另一种称呼 |
| **Dropout** | 丢弃法 | 防止过拟合的正则化技术 |
| **Early Stopping** | 早停 | 在验证误差开始上升时停止训练 |

---

## 10. 超参数与调优 (Hyperparameters & Tuning)

| English | 中文 | 说明 |
|---------|------|------|
| **Hyperparameter** | 超参数 | 训练前设置的参数 |
| **Hyperparameter Tuning** | 超参数调优 | 寻找最佳超参数的过程 |
| **Grid Search** | 网格搜索 | 遍历超参数组合的调优方法 |
| **Model Selection** | 模型选择 | 选择最佳模型的过程 |

---

## 11. 集成学习 (Ensemble Learning)

| English | 中文 | 说明 |
|---------|------|------|
| **Ensemble Learning** | 集成学习 | 组合多个模型以提高性能 |
| **Bagging** | 装袋法 | 集成学习方法，通过自助采样训练多个模型 |
| **Boosting** | 提升法 | 集成学习方法，顺序训练弱学习器 |
| **Random Forest** | 随机森林 | 基于决策树的集成学习算法 |

---

## 12. 其他重要概念 (Other Important Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Transfer Learning** | 迁移学习 | 将一个任务学到的知识应用到另一个任务 |
| **Entropy** | 熵 | 衡量数据不确定性的指标 |
| **Information Gain** | 信息增益 | 决策树中选择分裂特征的标准 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **AI** | Artificial Intelligence | 人工智能 |
| **ML** | Machine Learning | 机器学习 |
| **DL** | Deep Learning | 深度学习 |
| **NN** | Neural Network | 神经网络 |
| **CNN** | Convolutional Neural Network | 卷积神经网络 |
| **RNN** | Recurrent Neural Network | 循环神经网络 |
| **KNN** | K-Nearest Neighbors | K近邻算法 |
| **SVM** | Support Vector Machine | 支持向量机 |
| **PCA** | Principal Component Analysis | 主成分分析 |
| **SGD** | Stochastic Gradient Descent | 随机梯度下降 |
| **MSE** | Mean Squared Error | 均方误差 |
| **MAE** | Mean Absolute Error | 平均绝对误差 |
| **RMSE** | Root Mean Squared Error | 均方根误差 |
| **TP** | True Positive | 真正例 |
| **TN** | True Negative | 真负例 |
| **FP** | False Positive | 假正例 |
| **FN** | False Negative | 假负例 |

---

## 使用说明 (Usage Guide)

### 如何使用本词汇表

1. **主题学习**: 按分类组织，便于系统学习相关概念
2. **双语对照**: 提供英文原文和中文翻译，适合中英文切换学习
3. **简明解释**: 每个术语都有简短说明，帮助理解核心概念
4. **学习建议**: 建议按分类顺序学习，从基础概念到高级技术

### 学习路径建议

1. **初学者**: 从"基础概念"和"数据相关"开始
2. **进阶者**: 重点学习"回归算法"、"分类算法"和"模型评估"
3. **高级者**: 深入"神经网络"、"集成学习"和"超参数调优"

### 相关资源

- [Supervised Learning Tutorial (English)](./Supervised.md)
- [监督学习教程（中文）](./01_Supervised.md)
- [Regression Algorithms](./Regression/README.md)
- [Classification Algorithms](./Classification/README.md)
- [Evaluation Metrics](./Evaluation/README.md)

---

**最后更新**: 2026-01-29
