# 回归算法详解（Regression）

## 📚 目录

- [什么是回归](#什么是回归)
- [算法列表](#算法列表)
- [入门教程](#入门教程)
- [进阶教程](#进阶教程)
- [精通教程](#精通教程)
- [实战案例](#实战案例)

## 什么是回归

回归分析是预测**连续数值**的监督学习任务。给定输入特征，预测一个实数输出。

### 典型应用场景
- 📈 房价预测：根据面积、位置等预测价格
- 🌡️ 温度预测：根据历史数据预测未来温度
- 💰 销售预测：根据广告投入预测销售额
- 📊 股票价格预测：根据历史数据预测未来价格

### 回归 vs 分类
| 回归 | 分类 |
|------|------|
| 输出连续值（如23.5） | 输出离散类别（如"猫"或"狗"） |
| 预测"多少" | 预测"是什么" |

## 算法列表

本模块包含4个核心回归算法：

| 算法 | 难度 | 适用场景 | 关键特点 |
|------|------|----------|----------|
| **LinearRegression** | ⭐ | 线性关系数据 | 最简单，易解释 |
| **RidgeRegression** | ⭐⭐ | 特征相关性高 | L2正则化，防止过拟合 |
| **LassoRegression** | ⭐⭐⭐ | 需要特征选择 | L1正则化，产生稀疏解 |
| **PolynomialRegression** | ⭐⭐ | 非线性关系 | 拟合曲线关系 |

---

## 入门教程

### 第1课：线性回归基础

#### 理论基础

线性回归假设输入和输出之间存在线性关系：

```
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

- `y`: 预测值
- `w₀`: 截距（bias）
- `w₁, w₂, ..., wₙ`: 权重（weights）
- `x₁, x₂, ..., xₙ`: 特征

**目标**：找到最佳的权重，使预测值与真实值的误差最小。

#### 代码示例：简单线性回归

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

// 示例：根据学习时间预测考试成绩
double[,] X = new double[,] {
    { 1 },   // 学习1小时
    { 2 },   // 学习2小时
    { 3 },   // 学习3小时
    { 4 },   // 学习4小时
    { 5 }    // 学习5小时
};

double[] y = new double[] { 50, 60, 70, 80, 90 }; // 对应成绩

// 1. 创建模型
var model = new LinearRegression();

// 2. 训练模型
model.Fit(X, y);

// 3. 预测
double[,] XTest = new double[,] { { 3.5 } }; // 学习3.5小时
double[] predictions = model.Predict(XTest);

Console.WriteLine($"预测成绩: {predictions[0]}分");

// 4. 评估模型
double[] yPred = model.Predict(X);
double mse = RegressionMetrics.MeanSquaredError(y, yPred);
double r2 = RegressionMetrics.RSquared(y, yPred);

Console.WriteLine($"MSE: {mse:F2}");
Console.WriteLine($"R²: {r2:F2}");
```

#### 练习题

1. **基础练习**：预测房价
   - 输入：房屋面积（平方米）
   - 输出：价格（万元）
   - 数据：{50→150, 80→240, 120→360, 150→450}

2. **进阶练习**：多特征回归
   - 输入：面积、房间数、楼层
   - 输出：价格
   - 尝试分析每个特征的重要性

### 第2课：理解损失函数

#### 均方误差（MSE）

最常用的回归损失函数：

```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

- `yᵢ`: 真实值
- `ŷᵢ`: 预测值
- `n`: 样本数量

**特点**：
- 对大误差惩罚更重（平方项）
- 可微分，便于优化
- 单位是目标变量单位的平方

#### 代码示例：计算MSE

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

double[] yTrue = new double[] { 100, 200, 300 };
double[] yPred = new double[] { 110, 190, 310 };

double mse = RegressionMetrics.MeanSquaredError(yTrue, yPred);
double rmse = RegressionMetrics.RootMeanSquaredError(yTrue, yPred);
double mae = RegressionMetrics.MeanAbsoluteError(yTrue, yPred);

Console.WriteLine($"MSE: {mse:F2}");   // 均方误差
Console.WriteLine($"RMSE: {rmse:F2}"); // 均方根误差
Console.WriteLine($"MAE: {mae:F2}");   // 平均绝对误差
```

---

## 进阶教程

### 第3课：正则化技术

#### 为什么需要正则化？

**过拟合问题**：模型在训练集上表现很好，但在测试集上表现差。

**解决方案**：在损失函数中添加惩罚项，限制权重的大小。

#### Ridge回归（L2正则化）

损失函数：
```
Loss = MSE + α * Σwᵢ²
```

**特点**：
- 权重趋向于较小的值
- 不会将权重压缩到0
- 适合特征间存在多重共线性的情况

**代码示例**：

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;

// 准备数据
double[,] X = new double[,] {
    { 1, 2 },
    { 2, 4 },
    { 3, 6 },
    { 4, 8 }
};
double[] y = new double[] { 3, 5, 7, 9 };

// Ridge回归，alpha控制正则化强度
var model = new RidgeRegression(alpha: 1.0);
model.Fit(X, y);

// 预测
double[,] XTest = new double[,] { { 5, 10 } };
double[] predictions = model.Predict(XTest);

Console.WriteLine($"预测值: {predictions[0]:F2}");
```

**参数调优**：
- `alpha = 0`: 等同于普通线性回归
- `alpha` 很小: 轻微正则化
- `alpha` 很大: 强正则化，可能欠拟合

#### Lasso回归（L1正则化）

损失函数：
```
Loss = MSE + α * Σ|wᵢ|
```

**特点**：
- 可以将某些权重压缩到0
- 自动进行特征选择
- 产生稀疏模型

**代码示例**：

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;

var model = new LassoRegression(alpha: 0.5);
model.Fit(X, y);

// 查看有多少特征被选中
int nonZeroCount = model.GetNonZeroWeightsCount();
Console.WriteLine($"选中的特征数: {nonZeroCount}");
```

**Ridge vs Lasso**：

| 特性 | Ridge | Lasso |
|------|-------|-------|
| 正则化类型 | L2 | L1 |
| 特征选择 | ❌ | ✅ |
| 权重分布 | 均匀较小 | 稀疏（部分为0） |
| 适用场景 | 所有特征都重要 | 需要特征选择 |

### 第4课：多项式回归

#### 处理非线性关系

当数据呈现曲线关系时，线性回归效果不佳。多项式回归通过添加高次项来拟合曲线。

**原理**：
```
y = w₀ + w₁x + w₂x² + w₃x³ + ...
```

**代码示例**：

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;

// 非线性数据：y = x²
double[,] X = new double[,] {
    { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
};
double[] y = new double[] { 1, 4, 9, 16, 25 };

// 使用2阶多项式
var model = new PolynomialRegression(degree: 2);
model.Fit(X, y);

// 预测
double[,] XTest = new double[,] { { 6 } };
double[] predictions = model.Predict(XTest);

Console.WriteLine($"预测值: {predictions[0]:F2}"); // 应该接近36
```

**注意事项**：
- 阶数太低：欠拟合
- 阶数太高：过拟合
- 通常使用2-4阶

---

## 精通教程

### 第5课：模型评估与选择

#### 评估指标详解

**1. R²（决定系数）**
```
R² = 1 - (SS_res / SS_tot)
```
- 范围：(-∞, 1]
- R² = 1: 完美拟合
- R² = 0: 模型等同于预测均值
- R² < 0: 模型比预测均值还差

**2. RMSE vs MAE**
- RMSE：对大误差更敏感
- MAE：对异常值更鲁棒

**代码示例**：

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

double[] yTrue = new double[] { 100, 200, 300, 400 };
double[] yPred = new double[] { 110, 190, 310, 380 };

// 计算所有指标
double mse = RegressionMetrics.MeanSquaredError(yTrue, yPred);
double rmse = RegressionMetrics.RootMeanSquaredError(yTrue, yPred);
double mae = RegressionMetrics.MeanAbsoluteError(yTrue, yPred);
double r2 = RegressionMetrics.RSquared(yTrue, yPred);
double mape = RegressionMetrics.MeanAbsolutePercentageError(yTrue, yPred);

Console.WriteLine($"MSE:  {mse:F2}");
Console.WriteLine($"RMSE: {rmse:F2}");
Console.WriteLine($"MAE:  {mae:F2}");
Console.WriteLine($"R²:   {r2:F4}");
Console.WriteLine($"MAPE: {mape:F2}%");
```

### 第6课：交叉验证

#### K折交叉验证

将数据分成K份，轮流使用其中一份作为测试集，其余作为训练集。

**代码示例**：

```csharp
// 简单的K折交叉验证实现
public static double CrossValidate(double[,] X, double[] y, int k = 5)
{
    int n = y.Length;
    int foldSize = n / k;
    double totalR2 = 0;

    for (int i = 0; i < k; i++)
    {
        // 分割数据
        var (XTrain, yTrain, XTest, yTest) = SplitData(X, y, i, foldSize);

        // 训练和评估
        var model = new LinearRegression();
        model.Fit(XTrain, yTrain);
        double[] yPred = model.Predict(XTest);

        double r2 = RegressionMetrics.RSquared(yTest, yPred);
        totalR2 += r2;
    }

    return totalR2 / k; // 平均R²
}
```

---

## 实战案例

### 案例1：房价预测系统

**问题描述**：根据房屋特征预测价格

**数据特征**：
- 面积（平方米）
- 房间数
- 楼层
- 建造年份
- 距离市中心距离（公里）

**完整代码**：

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class HousePricePrediction
{
    public static void Main()
    {
        // 1. 准备数据
        double[,] X = new double[,] {
            // 面积, 房间数, 楼层, 年份, 距离
            { 50,  2, 3, 2010, 5 },
            { 80,  3, 5, 2015, 3 },
            { 120, 4, 8, 2018, 2 },
            { 150, 5, 10, 2020, 1 },
            { 60,  2, 4, 2012, 4 }
        };

        double[] y = new double[] { 150, 280, 450, 600, 200 }; // 价格（万元）

        // 2. 数据分割（80%训练，20%测试）
        int trainSize = (int)(X.GetLength(0) * 0.8);
        var (XTrain, yTrain, XTest, yTest) = SplitData(X, y, trainSize);

        // 3. 尝试不同模型
        Console.WriteLine("=== 线性回归 ===");
        TestModel(new LinearRegression(), XTrain, yTrain, XTest, yTest);

        Console.WriteLine("\n=== 岭回归 ===");
        TestModel(new RidgeRegression(alpha: 1.0), XTrain, yTrain, XTest, yTest);

        Console.WriteLine("\n=== Lasso回归 ===");
        TestModel(new LassoRegression(alpha: 0.5), XTrain, yTrain, XTest, yTest);

        // 4. 使用最佳模型进行预测
        var bestModel = new RidgeRegression(alpha: 1.0);
        bestModel.Fit(XTrain, yTrain);

        // 预测新房价格
        double[,] newHouse = new double[,] { { 100, 3, 6, 2019, 2.5 } };
        double[] prediction = bestModel.Predict(newHouse);

        Console.WriteLine($"\n新房预测价格: {prediction[0]:F2}万元");
    }

    static void TestModel(dynamic model, double[,] XTrain, double[] yTrain,
                         double[,] XTest, double[] yTest)
    {
        model.Fit(XTrain, yTrain);
        double[] yPred = model.Predict(XTest);

        double rmse = RegressionMetrics.RootMeanSquaredError(yTest, yPred);
        double r2 = RegressionMetrics.RSquared(yTest, yPred);

        Console.WriteLine($"RMSE: {rmse:F2}");
        Console.WriteLine($"R²: {r2:F4}");
    }
}
```

### 案例2：销售预测

**问题**：根据广告投入预测销售额

**特征**：
- 电视广告费用
- 网络广告费用
- 报纸广告费用

**建议使用**：多项式回归（捕捉非线性关系）

---

## 📊 算法选择指南

```
开始
  ↓
数据是线性关系？
  ├─ 是 → 特征数量多？
  │        ├─ 是 → 特征相关性高？
  │        │        ├─ 是 → Ridge回归
  │        │        └─ 否 → 需要特征选择？
  │        │                 ├─ 是 → Lasso回归
  │        │                 └─ 否 → 线性回归
  │        └─ 否 → 线性回归
  └─ 否 → 多项式回归
```

## 🎯 学习检查清单

### 入门级
- [ ] 理解回归的基本概念
- [ ] 能够使用LinearRegression进行简单预测
- [ ] 理解MSE和R²指标
- [ ] 完成房价预测练习

### 进阶级
- [ ] 理解正则化的作用
- [ ] 能够选择合适的alpha参数
- [ ] 掌握Ridge和Lasso的区别
- [ ] 能够处理非线性数据

### 精通级
- [ ] 能够实现交叉验证
- [ ] 理解所有评估指标的含义
- [ ] 能够诊断过拟合/欠拟合
- [ ] 完成完整的实战项目

## 📚 延伸阅读

- 《统计学习方法》第1-2章
- Scikit-learn回归文档
- Andrew Ng机器学习课程Week 1-2

---

**下一步**：学习[分类算法](../Classification/README.md)
