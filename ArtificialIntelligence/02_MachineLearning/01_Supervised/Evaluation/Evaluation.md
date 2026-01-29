# 模型评估详解（Evaluation）

## 📚 目录

- [为什么需要模型评估](#为什么需要模型评估)
- [评估工具列表](#评估工具列表)
- [入门教程](#入门教程)
- [进阶教程](#进阶教程)
- [精通教程](#精通教程)
- [实战案例](#实战案例)

## 为什么需要模型评估

模型评估是机器学习流程中的关键环节，用于：
- ✅ 衡量模型性能
- ✅ 比较不同模型
- ✅ 发现过拟合/欠拟合
- ✅ 指导模型优化

### 评估的黄金法则

> **永远不要在训练集上评估模型！**

必须使用独立的测试集来评估模型的泛化能力。

## 评估工具列表

本模块包含3个核心评估工具：

| 工具 | 适用场景 | 关键指标 |
|------|----------|----------|
| **RegressionMetrics** | 回归问题 | MSE, RMSE, MAE, R², MAPE |
| **ClassificationMetrics** | 分类问题 | Accuracy, Precision, Recall, F1 |
| **ConfusionMatrix** | 分类问题 | 混淆矩阵可视化 |

---

## 入门教程

### 第1课：回归评估基础

#### 核心指标

**1. 均方误差（MSE - Mean Squared Error）**

```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**特点**：
- 对大误差惩罚更重
- 单位是目标变量单位的平方
- 值越小越好

**代码示例**：

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

double[] yTrue = new double[] { 100, 200, 300, 400 };
double[] yPred = new double[] { 110, 190, 310, 380 };

double mse = RegressionMetrics.MeanSquaredError(yTrue, yPred);
Console.WriteLine($"MSE: {mse:F2}");
// 输出: MSE: 150.00
```

**2. 均方根误差（RMSE - Root Mean Squared Error）**

```
RMSE = √MSE
```

**特点**：
- 与目标变量同单位
- 更直观易懂
- 值越小越好

**代码示例**：

```csharp
double rmse = RegressionMetrics.RootMeanSquaredError(yTrue, yPred);
Console.WriteLine($"RMSE: {rmse:F2}");
// 输出: RMSE: 12.25（与价格单位相同）
```

**3. 平均绝对误差（MAE - Mean Absolute Error）**

```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```

**特点**：
- 对异常值不敏感
- 易于理解
- 值越小越好

**代码示例**：

```csharp
double mae = RegressionMetrics.MeanAbsoluteError(yTrue, yPred);
Console.WriteLine($"MAE: {mae:F2}");
// 输出: MAE: 10.00
```

#### 完整示例：房价预测评估

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

// 1. 训练模型
double[,] XTrain = new double[,] {
    { 50 }, { 80 }, { 120 }, { 150 }
};
double[] yTrain = new double[] { 150, 240, 360, 450 };

var model = new LinearRegression();
model.Fit(XTrain, yTrain);

// 2. 在测试集上预测
double[,] XTest = new double[,] {
    { 60 }, { 100 }, { 140 }
};
double[] yTest = new double[] { 180, 300, 420 };
double[] yPred = model.Predict(XTest);

// 3. 计算所有评估指标
double mse = RegressionMetrics.MeanSquaredError(yTest, yPred);
double rmse = RegressionMetrics.RootMeanSquaredError(yTest, yPred);
double mae = RegressionMetrics.MeanAbsoluteError(yTest, yPred);
double r2 = RegressionMetrics.RSquared(yTest, yPred);

// 4. 输出评估报告
Console.WriteLine("=== 回归模型评估报告 ===");
Console.WriteLine($"MSE:  {mse:F2}");
Console.WriteLine($"RMSE: {rmse:F2}");
Console.WriteLine($"MAE:  {mae:F2}");
Console.WriteLine($"R²:   {r2:F4}");
```

### 第2课：分类评估基础

#### 核心指标

**1. 准确率（Accuracy）**

```
Accuracy = 正确预测数 / 总样本数
```

**代码示例**：

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

int[] yTrue = new int[] { 0, 1, 1, 0, 1, 0, 1, 1 };
int[] yPred = new int[] { 0, 1, 0, 0, 1, 1, 1, 1 };

double accuracy = ClassificationMetrics.Accuracy(yTrue, yPred);
Console.WriteLine($"准确率: {accuracy:P2}");
// 输出: 准确率: 75.00%
```

**适用场景**：
- ✅ 类别平衡的数据
- ❌ 类别不平衡的数据

**2. 精确率（Precision）**

```
Precision = TP / (TP + FP)
```

**含义**：预测为正的样本中，真正为正的比例

**代码示例**：

```csharp
double precision = ClassificationMetrics.Precision(yTrue, yPred, positiveClass: 1);
Console.WriteLine($"精确率: {precision:P2}");
```

**3. 召回率（Recall）**

```
Recall = TP / (TP + FN)
```

**含义**：实际为正的样本中，被正确预测的比例

**代码示例**：

```csharp
double recall = ClassificationMetrics.Recall(yTrue, yPred, positiveClass: 1);
Console.WriteLine($"召回率: {recall:P2}");
```

**4. F1分数**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**含义**：精确率和召回率的调和平均

**代码示例**：

```csharp
double f1 = ClassificationMetrics.F1Score(yTrue, yPred, positiveClass: 1);
Console.WriteLine($"F1分数: {f1:P2}");
```

#### 完整示例：垃圾邮件分类评估

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

// 1. 训练模型
double[,] XTrain = new double[,] {
    { 0, 0 }, { 0, 1 }, { 5, 3 }, { 8, 5 }
};
int[] yTrain = new int[] { 0, 0, 1, 1 };

var model = new LogisticRegression();
model.Fit(XTrain, yTrain);

// 2. 在测试集上预测
double[,] XTest = new double[,] {
    { 1, 0 }, { 6, 4 }, { 0, 0 }, { 10, 8 }
};
int[] yTest = new int[] { 0, 1, 0, 1 };
int[] yPred = model.Predict(XTest);

// 3. 计算所有评估指标
double accuracy = ClassificationMetrics.Accuracy(yTest, yPred);
double precision = ClassificationMetrics.Precision(yTest, yPred, 1);
double recall = ClassificationMetrics.Recall(yTest, yPred, 1);
double f1 = ClassificationMetrics.F1Score(yTest, yPred, 1);

// 4. 输出评估报告
Console.WriteLine("=== 分类模型评估报告 ===");
Console.WriteLine($"准确率: {accuracy:P2}");
Console.WriteLine($"精确率: {precision:P2}");
Console.WriteLine($"召回率: {recall:P2}");
Console.WriteLine($"F1分数: {f1:P2}");
```

---

## 进阶教程

### 第3课：R²决定系数深入理解

#### 理论基础

R²（R-squared）衡量模型解释的方差比例：

```
R² = 1 - (SS_res / SS_tot)

其中：
SS_res = Σ(yᵢ - ŷᵢ)²  （残差平方和）
SS_tot = Σ(yᵢ - ȳ)²   （总平方和）
```

#### R²的含义

| R²值 | 含义 | 模型质量 |
|------|------|----------|
| 1.0 | 完美拟合 | 理想状态 |
| 0.9-1.0 | 非常好 | 优秀 |
| 0.7-0.9 | 较好 | 良好 |
| 0.5-0.7 | 一般 | 可接受 |
| < 0.5 | 较差 | 需要改进 |
| < 0 | 很差 | 比预测均值还差 |

**代码示例**：

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

double[] yTrue = new double[] { 100, 200, 300, 400, 500 };
double[] yPred = new double[] { 110, 190, 310, 390, 510 };

double r2 = RegressionMetrics.RSquared(yTrue, yPred);
Console.WriteLine($"R²: {r2:F4}");
Console.WriteLine($"模型解释了 {r2:P2} 的方差");
```

### 第4课：混淆矩阵详解

#### 理论基础

混淆矩阵是分类问题评估的核心工具：

```
                预测
              正类  负类
实  正类      TP    FN
际  负类      FP    TN
```

**术语解释**：
- **TP（True Positive）**：真正例 - 正确预测为正
- **TN（True Negative）**：真负例 - 正确预测为负
- **FP（False Positive）**：假正例 - 错误预测为正（第一类错误）
- **FN（False Negative）**：假负例 - 错误预测为负（第二类错误）

#### 代码示例：创建和分析混淆矩阵

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

int[] yTrue = new int[] { 0, 1, 1, 0, 1, 0, 1, 1, 0, 0 };
int[] yPred = new int[] { 0, 1, 0, 0, 1, 1, 1, 1, 0, 1 };

// 创建混淆矩阵
var cm = new ConfusionMatrix(yTrue, yPred);

// 打印混淆矩阵
Console.WriteLine(cm.ToString());

// 获取各项统计
int tp = cm.GetTruePositives(1);
int tn = cm.GetTrueNegatives(1);
int fp = cm.GetFalsePositives(1);
int fn = cm.GetFalseNegatives(1);

Console.WriteLine($"\n统计信息:");
Console.WriteLine($"真正例(TP): {tp}");
Console.WriteLine($"真负例(TN): {tn}");
Console.WriteLine($"假正例(FP): {fp}");
Console.WriteLine($"假负例(FN): {fn}");

// 手动计算指标
double precision = (double)tp / (tp + fp);
double recall = (double)tp / (tp + fn);
double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);

Console.WriteLine($"\n基于混淆矩阵的指标:");
Console.WriteLine($"精确率: {precision:P2}");
Console.WriteLine($"召回率: {recall:P2}");
Console.WriteLine($"准确率: {accuracy:P2}");
```

### 第5课：Precision-Recall权衡

#### 理论基础

精确率和召回率通常存在权衡关系：
- 提高阈值 → 精确率↑，召回率↓
- 降低阈值 → 精确率↓，召回率↑

#### 不同场景的选择

**1. 重视精确率的场景**
- 垃圾邮件过滤：避免误判正常邮件
- 推荐系统：确保推荐的都是用户喜欢的
- 广告投放：避免浪费广告费

**2. 重视召回率的场景**
- 疾病诊断：避免漏诊
- 欺诈检测：不能放过任何欺诈
- 安全检测：宁可误报，不可漏报

**3. 平衡两者的场景**
- 客户流失预测
- 信用评分
- 一般分类任务

**代码示例：调整阈值**

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;

var model = new LogisticRegression();
model.Fit(XTrain, yTrain);

// 获取概率预测
double[] probabilities = model.PredictProba(XTest);

// 尝试不同阈值
double[] thresholds = new double[] { 0.3, 0.5, 0.7 };

foreach (var threshold in thresholds)
{
    // 根据阈值转换为类别
    int[] predictions = probabilities.Select(p => p >= threshold ? 1 : 0).ToArray();

    double precision = ClassificationMetrics.Precision(yTest, predictions, 1);
    double recall = ClassificationMetrics.Recall(yTest, predictions, 1);

    Console.WriteLine($"阈值 {threshold:F1}:");
    Console.WriteLine($"  精确率: {precision:P2}");
    Console.WriteLine($"  召回率: {recall:P2}");
}
```

---

## 精通教程

### 第6课：交叉验证

#### 理论基础

交叉验证是评估模型泛化能力的重要技术，避免单次分割的偶然性。

**K折交叉验证流程**：
1. 将数据分成K份
2. 轮流使用其中一份作为测试集
3. 其余K-1份作为训练集
4. 计算K次评估指标的平均值

#### 代码示例：实现K折交叉验证

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class CrossValidation
{
    public static double KFoldCV(double[,] X, double[] y, int k = 5)
    {
        int n = y.Length;
        int foldSize = n / k;
        double totalR2 = 0;

        for (int fold = 0; fold < k; fold++)
        {
            // 分割数据
            int testStart = fold * foldSize;
            int testEnd = (fold == k - 1) ? n : testStart + foldSize;

            var (XTrain, yTrain, XTest, yTest) = SplitFold(X, y, testStart, testEnd);

            // 训练和评估
            var model = new LinearRegression();
            model.Fit(XTrain, yTrain);
            double[] yPred = model.Predict(XTest);

            double r2 = RegressionMetrics.RSquared(yTest, yPred);
            totalR2 += r2;

            Console.WriteLine($"Fold {fold + 1}: R² = {r2:F4}");
        }

        double avgR2 = totalR2 / k;
        Console.WriteLine($"\n平均 R²: {avgR2:F4}");

        return avgR2;
    }

    private static (double[,], double[], double[,], double[]) SplitFold(
        double[,] X, double[] y, int testStart, int testEnd)
    {
        int n = y.Length;
        int m = X.GetLength(1);
        int testSize = testEnd - testStart;
        int trainSize = n - testSize;

        double[,] XTrain = new double[trainSize, m];
        double[] yTrain = new double[trainSize];
        double[,] XTest = new double[testSize, m];
        double[] yTest = new double[testSize];

        int trainIdx = 0;
        for (int i = 0; i < n; i++)
        {
            if (i >= testStart && i < testEnd)
            {
                // 测试集
                int testIdx = i - testStart;
                for (int j = 0; j < m; j++)
                    XTest[testIdx, j] = X[i, j];
                yTest[testIdx] = y[i];
            }
            else
            {
                // 训练集
                for (int j = 0; j < m; j++)
                    XTrain[trainIdx, j] = X[i, j];
                yTrain[trainIdx] = y[i];
                trainIdx++;
            }
        }

        return (XTrain, yTrain, XTest, yTest);
    }
}
```

### 第7课：学习曲线分析

#### 理论基础

学习曲线展示训练集大小与模型性能的关系，用于诊断：
- **过拟合**：训练误差低，验证误差高
- **欠拟合**：训练误差和验证误差都高
- **良好拟合**：训练误差和验证误差都低且接近

#### 代码示例：绘制学习曲线数据

```csharp
public class LearningCurve
{
    public static void PlotLearningCurve(double[,] X, double[] y)
    {
        int n = y.Length;
        int[] trainSizes = new int[] {
            n / 10, n / 5, n / 3, n / 2, (int)(n * 0.7), (int)(n * 0.9)
        };

        Console.WriteLine("训练集大小\t训练R²\t验证R²");
        Console.WriteLine("----------------------------------------");

        foreach (var size in trainSizes)
        {
            // 使用前size个样本训练
            var (XTrain, yTrain, XVal, yVal) = SplitData(X, y, size);

            var model = new LinearRegression();
            model.Fit(XTrain, yTrain);

            // 计算训练集和验证集的R²
            double[] yTrainPred = model.Predict(XTrain);
            double[] yValPred = model.Predict(XVal);

            double trainR2 = RegressionMetrics.RSquared(yTrain, yTrainPred);
            double valR2 = RegressionMetrics.RSquared(yVal, yValPred);

            Console.WriteLine($"{size}\t\t{trainR2:F4}\t{valR2:F4}");
        }
    }
}
```

---

## 实战案例

### 案例1：完整的模型评估流程

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class CompleteEvaluationPipeline
{
    public static void Main()
    {
        // 1. 准备数据
        double[,] X = LoadData();
        double[] y = LoadLabels();

        // 2. 数据分割（70%训练，15%验证，15%测试）
        var (XTrain, yTrain, XVal, yVal, XTest, yTest) =
            SplitTrainValTest(X, y, 0.7, 0.15, 0.15);

        // 3. 训练模型
        var model = new LinearRegression();
        model.Fit(XTrain, yTrain);

        // 4. 在验证集上评估（用于调参）
        Console.WriteLine("=== 验证集评估 ===");
        EvaluateRegression(model, XVal, yVal);

        // 5. 在测试集上最终评估
        Console.WriteLine("\n=== 测试集评估（最终性能） ===");
        EvaluateRegression(model, XTest, yTest);

        // 6. 交叉验证评估
        Console.WriteLine("\n=== 5折交叉验证 ===");
        double cvScore = CrossValidation.KFoldCV(XTrain, yTrain, k: 5);

        // 7. 学习曲线分析
        Console.WriteLine("\n=== 学习曲线 ===");
        LearningCurve.PlotLearningCurve(XTrain, yTrain);
    }

    static void EvaluateRegression(LinearRegression model, double[,] X, double[] y)
    {
        double[] yPred = model.Predict(X);

        double mse = RegressionMetrics.MeanSquaredError(y, yPred);
        double rmse = RegressionMetrics.RootMeanSquaredError(y, yPred);
        double mae = RegressionMetrics.MeanAbsoluteError(y, yPred);
        double r2 = RegressionMetrics.RSquared(y, yPred);
        double mape = RegressionMetrics.MeanAbsolutePercentageError(y, yPred);

        Console.WriteLine($"MSE:  {mse:F2}");
        Console.WriteLine($"RMSE: {rmse:F2}");
        Console.WriteLine($"MAE:  {mae:F2}");
        Console.WriteLine($"R²:   {r2:F4}");
        Console.WriteLine($"MAPE: {mape:F2}%");
    }
}
```

### 案例2：分类模型完整评估

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class ClassificationEvaluation
{
    public static void Main()
    {
        // 1. 准备数据
        double[,] X = LoadData();
        int[] y = LoadLabels();

        // 2. 数据分割
        var (XTrain, yTrain, XTest, yTest) = SplitData(X, y, 0.8);

        // 3. 训练模型
        var model = new LogisticRegression();
        model.Fit(XTrain, yTrain);

        // 4. 预测
        int[] yPred = model.Predict(XTest);

        // 5. 计算所有指标
        Console.WriteLine("=== 分类评估报告 ===\n");

        double accuracy = ClassificationMetrics.Accuracy(yTest, yPred);
        double precision = ClassificationMetrics.Precision(yTest, yPred, 1);
        double recall = ClassificationMetrics.Recall(yTest, yPred, 1);
        double f1 = ClassificationMetrics.F1Score(yTest, yPred, 1);
        double specificity = ClassificationMetrics.Specificity(yTest, yPred, 1);

        Console.WriteLine($"准确率:   {accuracy:P2}");
        Console.WriteLine($"精确率:   {precision:P2}");
        Console.WriteLine($"召回率:   {recall:P2}");
        Console.WriteLine($"F1分数:   {f1:P2}");
        Console.WriteLine($"特异度:   {specificity:P2}");

        // 6. 混淆矩阵
        Console.WriteLine("\n=== 混淆矩阵 ===");
        var cm = new ConfusionMatrix(yTest, yPred);
        Console.WriteLine(cm.ToString());

        // 7. 详细分析
        Console.WriteLine("\n=== 详细分析 ===");
        int tp = cm.GetTruePositives(1);
        int tn = cm.GetTrueNegatives(1);
        int fp = cm.GetFalsePositives(1);
        int fn = cm.GetFalseNegatives(1);

        Console.WriteLine($"真正例: {tp}");
        Console.WriteLine($"真负例: {tn}");
        Console.WriteLine($"假正例: {fp} (误报)");
        Console.WriteLine($"假负例: {fn} (漏报)");
    }
}
```

---

## 📊 评估指标选择指南

### 回归问题

```
开始
  ↓
关心预测误差的单位？
  ├─ 是 → RMSE或MAE
  │        ├─ 对异常值敏感？
  │        │   ├─ 是 → RMSE
  │        │   └─ 否 → MAE
  └─ 否 → 关心解释方差？
           ├─ 是 → R²
           └─ 否 → 关心百分比误差？
                    └─ 是 → MAPE
```

### 分类问题

```
开始
  ↓
类别平衡？
  ├─ 是 → Accuracy
  └─ 否 → 关注什么？
           ├─ 避免误报 → Precision
           ├─ 避免漏报 → Recall
           └─ 平衡两者 → F1-Score
```

## 🎯 学习检查清单

### 入门级
- [ ] 理解MSE、RMSE、MAE的含义
- [ ] 理解Accuracy、Precision、Recall
- [ ] 能够计算基本评估指标
- [ ] 理解训练集/测试集分割的重要性

### 进阶级
- [ ] 理解R²的含义和应用
- [ ] 能够创建和分析混淆矩阵
- [ ] 理解Precision-Recall权衡
- [ ] 能够选择合适的评估指标

### 精通级
- [ ] 能够实现交叉验证
- [ ] 能够绘制和分析学习曲线
- [ ] 能够诊断过拟合/欠拟合
- [ ] 能够进行完整的模型评估流程

## 📚 延伸阅读

- 《统计学习方法》第8章
- Scikit-learn模型评估文档
- "Precision and Recall" - Wikipedia

---

**恭喜！** 你已经完成了监督学习的全部教程。继续探索其他模块吧！🎉
