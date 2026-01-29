# 监督学习（Supervised Learning）

## 📚 目录概览

监督学习是机器学习的核心分支，通过标注数据学习输入到输出的映射关系。本模块包含完整的监督学习算法实现和教程。

### 📁 模块结构

```
01_Supervised/
├── Regression/          # 回归算法
├── Classification/      # 分类算法
└── Evaluation/         # 模型评估
```

## 🎯 学习路径

| 阶段 | 时间 | 学习主题 | 核心内容 | 实践项目 |
|------|------|----------|----------|----------|
| **入门阶段** | 1-2周 | **理解监督学习基本概念** | • 什么是监督学习<br>• 训练集、测试集、验证集<br>• 过拟合与欠拟合 | - |
| | | **从回归开始** | • 线性回归（最简单的算法）<br>• 理解损失函数和梯度下降 | 房价预测 |
| | | **学习分类算法** | • 逻辑回归（二分类入门）<br>• K近邻（直观易懂） | 鸢尾花分类 |
| **进阶阶段** | 2-4周 | **深入算法原理** | • 决策树的构建过程<br>• 朴素贝叶斯的概率推导<br>• 正则化技术（Ridge、Lasso） | - |
| | | **模型评估与选择** | • 交叉验证<br>• 评估指标的选择<br>• 混淆矩阵分析 | - |
| | | **特征工程基础** | • 特征缩放<br>• 多项式特征<br>• 特征选择 | - |
| **精通阶段** | 4-8周 | **算法调优** | • 超参数优化<br>• 学习曲线分析<br>• 模型集成 | - |
| | | **实战项目** | • 信用评分系统<br>• 客户流失预测<br>• 房价预测系统 | 完整项目实战 |
| | | **生产部署** | • 模型持久化<br>• API服务化<br>• 性能优化 | - |

## 📖 子模块详细教程

### [Regression - 回归算法](./Regression/README.md)
- 线性回归（Linear Regression）
- 岭回归（Ridge Regression）
- Lasso回归（Lasso Regression）
- 多项式回归（Polynomial Regression）

### [Classification - 分类算法](./Classification/README.md)
- 逻辑回归（Logistic Regression）
- K近邻（K-Nearest Neighbors）
- 决策树（Decision Tree）
- 朴素贝叶斯（Naive Bayes）

### [Evaluation - 模型评估](./Evaluation/README.md)
- 回归评估指标
- 分类评估指标
- 混淆矩阵

## 🚀 快速开始

### 1. 回归示例：预测房价

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;

// 准备数据：房屋面积 -> 价格
double[,] X = new double[,] {
    { 50 },   // 50平米
    { 80 },   // 80平米
    { 120 },  // 120平米
    { 150 }   // 150平米
};

double[] y = new double[] { 150, 240, 360, 450 }; // 价格（万元）

// 训练模型
var model = new LinearRegression();
model.Fit(X, y);

// 预测
double[,] XTest = new double[,] { { 100 } }; // 100平米
double[] predictions = model.Predict(XTest);

Console.WriteLine($"预测价格: {predictions[0]}万元");
```

### 2. 分类示例：鸢尾花分类

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;

// 准备数据：花瓣长度、花瓣宽度 -> 类别
double[,] X = new double[,] {
    { 1.4, 0.2 },  // 类别0
    { 1.3, 0.2 },  // 类别0
    { 4.5, 1.5 },  // 类别1
    { 4.7, 1.6 }   // 类别1
};

int[] y = new int[] { 0, 0, 1, 1 };

// 训练KNN分类器
var model = new KNearestNeighbors(k: 3);
model.Fit(X, y);

// 预测
double[,] XTest = new double[,] { { 4.0, 1.3 } };
int[] predictions = model.Predict(XTest);

Console.WriteLine($"预测类别: {predictions[0]}");
```

## 📊 核心概念

### 监督学习的工作流程

```
1. 数据收集 → 2. 数据预处理 → 3. 特征工程
                    ↓
4. 模型训练 ← 5. 模型选择 ← 6. 数据分割
                    ↓
7. 模型评估 → 8. 超参数调优 → 9. 模型部署
```

### 回归 vs 分类

| 特征 | 回归 | 分类 |
|------|------|------|
| **输出类型** | 连续值 | 离散类别 |
| **示例问题** | 房价预测、温度预测 | 垃圾邮件识别、图像分类 |
| **评估指标** | MSE, RMSE, R² | Accuracy, Precision, Recall |
| **典型算法** | 线性回归、岭回归 | 逻辑回归、决策树 |

## 🔑 关键术语

- **特征（Features）**: 输入变量，用于预测目标
- **标签（Labels）**: 输出变量，我们要预测的目标
- **训练集（Training Set）**: 用于训练模型的数据
- **测试集（Test Set）**: 用于评估模型性能的数据
- **过拟合（Overfitting）**: 模型在训练集上表现好，但在测试集上表现差
- **欠拟合（Underfitting）**: 模型过于简单，无法捕捉数据的模式
- **正则化（Regularization）**: 防止过拟合的技术

## 📚 推荐学习资源

### 书籍
- 《统计学习方法》- 李航
- 《机器学习》- 周志华
- 《Pattern Recognition and Machine Learning》- Christopher Bishop

### 在线课程
- Andrew Ng的机器学习课程（Coursera）
- 吴恩达深度学习专项课程
- Fast.ai实用机器学习课程

### 实践平台
- Kaggle竞赛平台
- UCI机器学习数据集库
- Scikit-learn官方教程

## 💡 学习建议

1. **理论与实践结合**：每学习一个算法，立即用代码实现
2. **从简单到复杂**：先掌握线性回归和逻辑回归
3. **重视数学基础**：线性代数、概率论、微积分
4. **多做项目**：参加Kaggle竞赛，积累实战经验
5. **阅读源码**：理解算法的实现细节

## 🎓 进阶方向

掌握监督学习后，可以继续学习：
- **02_Unsupervised**: 无监督学习（聚类、降维）
- **03_FeatureEngineering**: 特征工程技术
- **04_Ensemble**: 集成学习方法
- **深度学习**: 神经网络、CNN、RNN

## 📞 获取帮助

- 查看各子目录的详细教程
- 阅读代码注释和文档字符串
- 参考示例代码和测试用例

---

**开始学习之旅吧！** 🚀
