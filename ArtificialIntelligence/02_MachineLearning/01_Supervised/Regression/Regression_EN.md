# Regression: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: What is Regression?](#chapter-1-what-is-regression)
  - [Chapter 2: Linear Regression Fundamentals](#chapter-2-linear-regression-fundamentals)
  - [Chapter 3: Loss Functions and Optimization](#chapter-3-loss-functions-and-optimization)
  - [Chapter 4: Multiple Linear Regression](#chapter-4-multiple-linear-regression)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 5: Regularization - Ridge Regression](#chapter-5-regularization---ridge-regression)
  - [Chapter 6: Regularization - Lasso Regression](#chapter-6-regularization---lasso-regression)
  - [Chapter 7: Polynomial Regression](#chapter-7-polynomial-regression)
  - [Chapter 8: Feature Engineering for Regression](#chapter-8-feature-engineering-for-regression)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 9: Advanced Evaluation Techniques](#chapter-9-advanced-evaluation-techniques)
  - [Chapter 10: Handling Real-World Challenges](#chapter-10-handling-real-world-challenges)
  - [Chapter 11: Model Selection and Comparison](#chapter-11-model-selection-and-comparison)
  - [Chapter 12: Production Deployment](#chapter-12-production-deployment)

---

## Introduction

Welcome to the comprehensive guide on **Regression**! This tutorial will take you from understanding basic concepts to building production-ready regression systems.

### What You'll Learn

| Level | Duration | Topics Covered | Skills Acquired |
|-------|----------|----------------|-----------------|
| **Beginner** | 2-3 weeks | Linear regression, Loss functions, Multiple regression | Build simple predictive models |
| **Intermediate** | 3-5 weeks | Regularization, Polynomial regression, Feature engineering | Handle overfitting, non-linear relationships |
| **Advanced** | 4-6 weeks | Advanced evaluation, Real-world challenges, Deployment | Build production systems |

### Prerequisites

- Basic understanding of supervised learning
- Familiarity with C# programming
- Basic algebra and statistics
- Understanding of functions and derivatives (helpful)

### Regression vs Classification

**Key Difference**:

```
Regression: Predicts continuous values
Example: House price = $350,000 (any real number)

Classification: Predicts discrete categories
Example: Email type = "Spam" (fixed categories)
```

| Aspect | Regression | Classification |
|--------|-----------|----------------|
| **Output** | Continuous numbers | Discrete categories |
| **Question** | "How much?" | "Which one?" |
| **Examples** | Price, temperature, sales | Spam/not spam, cat/dog |
| **Evaluation** | MSE, RMSE, R² | Accuracy, precision, recall |

---

## Part I: Beginner Level

### Chapter 1: What is Regression?

#### 1.1 The Big Picture

**Regression** is a supervised learning task where we predict continuous numerical values based on input features.

**Real-World Analogy**:
Think of a real estate appraiser:
- **Input**: House features (size, location, age)
- **Process**: Analyze similar houses and their prices
- **Output**: Estimated price ($350,000)

In regression:
- **Input features** = House characteristics
- **Model** = Learned relationship
- **Output** = Predicted continuous value

#### 1.2 Types of Regression Problems

**1. Simple Linear Regression**
- One input feature
- Example: Predict salary based on years of experience

**2. Multiple Linear Regression**
- Multiple input features
- Example: Predict house price based on size, location, age

**3. Polynomial Regression**
- Non-linear relationships
- Example: Predict crop yield based on fertilizer amount (diminishing returns)

**4. Regularized Regression**
- Prevent overfitting
- Examples: Ridge, Lasso

#### 1.3 Real-World Applications

**Business Applications**:
- 📈 **Sales Forecasting**: Predict future sales based on historical data
- 💰 **Price Optimization**: Determine optimal pricing
- 📊 **Demand Prediction**: Forecast product demand
- 💵 **Revenue Projection**: Estimate future revenue

**Finance Applications**:
- 📉 **Stock Price Prediction**: Forecast stock prices
- 💳 **Credit Scoring**: Predict loan default amounts
- 💱 **Risk Assessment**: Estimate financial risk

**Science & Engineering**:
- 🌡️ **Temperature Forecasting**: Predict future temperatures
- ⚡ **Energy Consumption**: Forecast energy usage
- 🏗️ **Load Prediction**: Estimate structural loads

#### 1.4 The Regression Workflow

```
┌─────────────────┐
│  1. Collect     │
│     Data        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Explore &   │
│     Visualize   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Prepare     │
│     Features    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Split Data  │
│  (Train/Test)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Train       │
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. Evaluate    │
│     Performance │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  7. Tune &      │
│     Optimize    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  8. Deploy      │
│     Model       │
└─────────────────┘
```

#### 1.5 Quick Start Example

Let's see regression in action with a simple example:

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

// Problem: Predict exam score based on study hours
double[,] studyHours = new double[,] {
    { 1 },  // 1 hour
    { 2 },  // 2 hours
    { 3 },  // 3 hours
    { 4 },  // 4 hours
    { 5 }   // 5 hours
};

double[] examScores = new double[] { 50, 60, 70, 80, 90 };

// Create and train model
var model = new LinearRegression();
model.Fit(studyHours, examScores);

// Predict score for 3.5 hours of study
double[,] newStudent = new double[,] { { 3.5 } };
double[] prediction = model.Predict(newStudent);

Console.WriteLine($"Predicted score: {prediction[0]:F1}");
// Output: Predicted score: 75.0

// Evaluate model
double[] predictions = model.Predict(studyHours);
double r2 = RegressionMetrics.RSquared(examScores, predictions);
Console.WriteLine($"R²: {r2:F4}");
// Output: R²: 1.0000 (perfect fit for this simple example)
```

**What just happened?**
1. We provided training data (study hours → exam scores)
2. The model learned the linear relationship
3. We predicted a score for a new input
4. We evaluated how well the model fits the data

---

### Chapter 2: Linear Regression Fundamentals

#### 2.1 The Linear Model

**Mathematical Form**:

```
Simple Linear Regression (one feature):
y = w₀ + w₁x

Multiple Linear Regression (many features):
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

Or in vector form:
y = w₀ + w^T x
```

**Components**:
- **y**: Predicted value (output)
- **x**: Input features
- **w₀**: Intercept (bias) - value when all features are 0
- **w₁, w₂, ..., wₙ**: Weights (slopes) - how much each feature affects output

#### 2.2 Geometric Interpretation

**2D Example** (one feature):
```
Price ↑
      │        ●
      │      ╱
      │    ●╱
      │  ╱●
      │╱●
      └──────────→ Size

● = Data points
╱ = Best fit line (y = w₀ + w₁x)
```

**Goal**: Find the line that best fits the data points.

#### 2.3 The Normal Equation

**Closed-form solution** for linear regression:

```
w = (X^T X)^(-1) X^T y

Where:
- X = feature matrix
- y = target values
- w = optimal weights
```

**Advantages**:
- ✅ Exact solution (no iterations needed)
- ✅ No hyperparameters to tune
- ✅ Fast for small to medium datasets

**Disadvantages**:
- ❌ Slow for large datasets (matrix inversion is O(n³))
- ❌ Requires matrix to be invertible
- ❌ Memory intensive for many features

#### 2.4 Complete Implementation Example

**Problem**: Predict house prices based on size

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class HousePriceExample
{
    public static void Main()
    {
        // Step 1: Prepare training data
        // Features: [size_sqft]
        double[,] XTrain = new double[,] {
            { 1000 },
            { 1500 },
            { 2000 },
            { 2500 },
            { 3000 }
        };

        // Prices in thousands
        double[] yTrain = new double[] { 200, 300, 400, 500, 600 };

        // Step 2: Create and train model
        var model = new LinearRegression();
        model.Fit(XTrain, yTrain);

        // Step 3: Make predictions
        double[,] XTest = new double[,] {
            { 1800 },
            { 2200 }
        };

        double[] predictions = model.Predict(XTest);

        Console.WriteLine("Predictions:");
        Console.WriteLine($"1800 sqft house: ${predictions[0]:F1}k");
        Console.WriteLine($"2200 sqft house: ${predictions[1]:F1}k");

        // Step 4: Evaluate model
        double[] yTrainPred = model.Predict(XTrain);
        double mse = RegressionMetrics.MeanSquaredError(yTrain, yTrainPred);
        double rmse = RegressionMetrics.RootMeanSquaredError(yTrain, yTrainPred);
        double r2 = RegressionMetrics.RSquared(yTrain, yTrainPred);

        Console.WriteLine($"\nModel Performance:");
        Console.WriteLine($"MSE:  {mse:F2}");
        Console.WriteLine($"RMSE: {rmse:F2}k");
        Console.WriteLine($"R²:   {r2:F4}");
    }
}
```

#### 2.5 Interpreting the Model

**Understanding Weights**:

```csharp
// After training, you can interpret the weights
// Example: y = 50 + 0.2x

// Interpretation:
// - Intercept (50): Base price when size = 0
// - Slope (0.2): Each additional sqft adds $200 to price
```

**Example Interpretation**:
```
Model: Price = 50 + 0.2 × Size

For a 2000 sqft house:
Price = 50 + 0.2 × 2000 = 50 + 400 = $450k

Meaning:
- Base price: $50k
- Size contribution: $400k (2000 × $0.2k per sqft)
```

#### 2.6 Assumptions of Linear Regression

**Important Assumptions**:

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Features are not highly correlated

**Checking Assumptions**:
```csharp
// Check linearity: Plot predictions vs actual
// Check homoscedasticity: Plot residuals vs predictions
// Check normality: Plot histogram of residuals
```

#### 2.7 Practice Exercise

**Exercise**: Predict student grades based on study time

```csharp
// Study hours per week
double[,] studyHours = new double[,] {
    { 5 }, { 10 }, { 15 }, { 20 }, { 25 }, { 30 }
};

// Final grades (0-100)
double[] grades = new double[] { 60, 70, 75, 85, 90, 95 };

// TODO:
// 1. Split data 80/20
// 2. Train linear regression model
// 3. Evaluate with RMSE and R²
// 4. Predict grade for 18 hours of study
// 5. Interpret the slope (what does it mean?)
```

**Expected Output**:
```
RMSE: ~3-5 points
R²: ~0.95-0.98
Prediction for 18 hours: ~80-82
Interpretation: Each additional hour adds ~1.5 points to grade
```

---

### Chapter 3: Loss Functions and Optimization

#### 3.1 What is a Loss Function?

**Definition**: A loss function measures how far our predictions are from the actual values.

**Goal**: Minimize the loss function to find the best model parameters.

```
Loss = f(predictions, actual_values)

Lower loss = Better model
```

#### 3.2 Mean Squared Error (MSE)

**Most common loss function for regression**:

```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²

Where:
- n = number of samples
- yᵢ = actual value
- ŷᵢ = predicted value
```

**Why Square the Errors?**

1. **Always positive**: Negative and positive errors don't cancel out
2. **Penalizes large errors**: Squaring makes large errors much worse
3. **Mathematically convenient**: Differentiable, easy to optimize

**Example**:
```
Actual: [100, 200, 300]
Predicted: [110, 190, 310]

Errors: [10, -10, 10]
Squared errors: [100, 100, 100]
MSE = (100 + 100 + 100) / 3 = 100
```

**Code Example**:

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

double[] yTrue = new double[] { 100, 200, 300, 400 };
double[] yPred = new double[] { 110, 190, 310, 380 };

double mse = RegressionMetrics.MeanSquaredError(yTrue, yPred);
Console.WriteLine($"MSE: {mse:F2}");
// Output: MSE: 150.00

// Interpretation: Average squared error is 150
```

#### 3.3 Root Mean Squared Error (RMSE)

**Definition**: Square root of MSE

```
RMSE = √MSE
```

**Advantages over MSE**:
- Same units as target variable (more interpretable)
- Easier to communicate to stakeholders

**Example**:
```
MSE = 150 (dollars²) ← Hard to interpret
RMSE = 12.25 dollars ← Easy to interpret!

Meaning: On average, predictions are off by $12.25
```

**Code Example**:

```csharp
double rmse = RegressionMetrics.RootMeanSquaredError(yTrue, yPred);
Console.WriteLine($"RMSE: {rmse:F2}");
// Output: RMSE: 12.25
```

#### 3.4 Mean Absolute Error (MAE)

**Definition**: Average of absolute errors

```
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```

**Characteristics**:
- More robust to outliers than MSE
- Treats all errors equally (no squaring)
- Same units as target variable

**MSE vs MAE**:

| Aspect | MSE/RMSE | MAE |
|--------|----------|-----|
| **Outlier sensitivity** | High | Low |
| **Large error penalty** | Quadratic | Linear |
| **Use when** | Outliers are important | Outliers are noise |
| **Interpretation** | Weighted average error | Simple average error |

**Example**:
```
Errors: [10, 10, 10, 10, 100]

MAE = (10+10+10+10+100)/5 = 28
RMSE = √[(100+100+100+100+10000)/5] = 45.6

RMSE is much higher due to the outlier!
```

#### 3.5 Gradient Descent (Alternative to Normal Equation)

**When to use**: Large datasets where normal equation is too slow

**Concept**: Iteratively adjust weights to minimize loss

```
Repeat until convergence:
  1. Calculate gradient of loss function
  2. Update weights: w := w - α × gradient

Where α = learning rate (step size)
```

**Visual Representation**:
```
Loss ↑
     │    ╱╲
     │   ╱  ╲
     │  ╱    ╲
     │ ╱      ╲
     │╱    ●   ╲  ← Start here
     └──────────→ w
         ↓
     Move downhill
         ↓
     │╱        ╲
     │    ●     ╲  ← After updates
     └──────────→ w
         ↓
     │╱          ╲
     │      ●     ╲  ← Converged!
     └──────────→ w
```

**Pseudocode**:

```csharp
// Simplified gradient descent
public void GradientDescent(double[,] X, double[] y, double learningRate, int maxIterations)
{
    // Initialize weights randomly
    double[] weights = InitializeWeights(X.GetLength(1));

    for (int iter = 0; iter < maxIterations; iter++)
    {
        // Calculate predictions
        double[] predictions = Predict(X, weights);

        // Calculate gradient
        double[] gradient = CalculateGradient(X, y, predictions);

        // Update weights
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] -= learningRate * gradient[i];
        }

        // Check convergence
        double loss = CalculateMSE(y, predictions);
        if (loss < tolerance)
            break;
    }
}
```

#### 3.6 Learning Rate Selection

**Critical hyperparameter**: Controls step size

```
Learning Rate Too Small:
- Slow convergence
- Many iterations needed
- May not reach minimum in time

Learning Rate Too Large:
- Overshoots minimum
- May diverge (loss increases)
- Unstable training

Learning Rate Just Right:
- Fast convergence
- Stable training
- Reaches minimum efficiently
```

**Typical Values**: 0.001 to 0.1

---

### Chapter 4: Multiple Linear Regression

#### 4.1 From One to Many Features

**Simple Linear Regression**:
```
y = w₀ + w₁x
One feature → One weight
```

**Multiple Linear Regression**:
```
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
Many features → Many weights
```

**Example - House Price Prediction**:
```
Price = w₀ + w₁×Size + w₂×Bedrooms + w₃×Age + w₄×Location

Features:
- x₁ = Size (sqft)
- x₂ = Number of bedrooms
- x₃ = Age (years)
- x₄ = Distance to city center (miles)
```

#### 4.2 Matrix Representation

**Compact notation using matrices**:

```
y = Xw

Where:
- y = [y₁, y₂, ..., yₙ]ᵀ (n × 1)
- X = feature matrix (n × m)
- w = [w₀, w₁, ..., wₘ]ᵀ (m × 1)

Example:
┌     ┐   ┌           ┐   ┌    ┐
│ 200 │   │ 1 1000 2  │   │ w₀ │
│ 300 │ = │ 1 1500 3  │ × │ w₁ │
│ 400 │   │ 1 2000 3  │   │ w₂ │
└     ┘   └           ┘   └    ┘
  y            X            w
```

#### 4.3 Complete Implementation Example

**Problem**: Predict house prices using multiple features

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class MultipleRegressionExample
{
    public static void Main()
    {
        // Step 1: Prepare training data
        // Features: [size_sqft, bedrooms, age_years, distance_miles]
        double[,] XTrain = new double[,] {
            { 1000, 2, 10, 5 },
            { 1500, 3, 5, 3 },
            { 2000, 3, 2, 2 },
            { 2500, 4, 1, 1 },
            { 3000, 4, 0, 1 }
        };

        // Prices in thousands
        double[] yTrain = new double[] { 200, 300, 400, 500, 600 };

        // Step 2: Train model
        var model = new LinearRegression();
        model.Fit(XTrain, yTrain);

        // Step 3: Prepare test data
        double[,] XTest = new double[,] {
            { 1800, 3, 4, 2.5 },
            { 2200, 3, 3, 2 }
        };

        double[] yTest = new double[] { 360, 440 };

        // Step 4: Make predictions
        double[] predictions = model.Predict(XTest);

        Console.WriteLine("Predictions:");
        for (int i = 0; i < predictions.Length; i++)
        {
            Console.WriteLine($"House {i + 1}: ${predictions[i]:F1}k (Actual: ${yTest[i]}k)");
        }

        // Step 5: Evaluate model
        double mse = RegressionMetrics.MeanSquaredError(yTest, predictions);
        double rmse = RegressionMetrics.RootMeanSquaredError(yTest, predictions);
        double mae = RegressionMetrics.MeanAbsoluteError(yTest, predictions);
        double r2 = RegressionMetrics.RSquared(yTest, predictions);

        Console.WriteLine($"\nModel Performance:");
        Console.WriteLine($"MSE:  {mse:F2}");
        Console.WriteLine($"RMSE: {rmse:F2}k");
        Console.WriteLine($"MAE:  {mae:F2}k");
        Console.WriteLine($"R²:   {r2:F4}");

        // Step 6: Interpret feature importance
        Console.WriteLine($"\nFeature Interpretation:");
        Console.WriteLine("Each additional sqft adds $X to price");
        Console.WriteLine("Each additional bedroom adds $Y to price");
        Console.WriteLine("Each year of age reduces price by $Z");
        Console.WriteLine("Each mile from city reduces price by $W");
    }
}
```

#### 4.4 Feature Scaling

**Problem**: Features with different scales can dominate the model

```
Example:
Feature 1: Size (1000-3000 sqft)
Feature 2: Bedrooms (1-5)
Feature 3: Age (0-50 years)

Size has much larger values → Dominates the model!
```

**Solution**: Scale features to similar ranges

**Standardization (Z-score normalization)**:
```
x_scaled = (x - mean) / std

Result: mean = 0, std = 1
```

**Code Example**:

```csharp
public static double[,] StandardizeFeatures(double[,] X)
{
    int n = X.GetLength(0);
    int m = X.GetLength(1);
    double[,] XScaled = new double[n, m];

    for (int j = 0; j < m; j++)
    {
        // Calculate mean
        double mean = 0;
        for (int i = 0; i < n; i++)
            mean += X[i, j];
        mean /= n;

        // Calculate standard deviation
        double std = 0;
        for (int i = 0; i < n; i++)
            std += Math.Pow(X[i, j] - mean, 2);
        std = Math.Sqrt(std / n);

        // Standardize
        for (int i = 0; i < n; i++)
            XScaled[i, j] = (X[i, j] - mean) / (std > 0 ? std : 1);
    }

    return XScaled;
}

// Usage
double[,] XScaled = StandardizeFeatures(XTrain);
model.Fit(XScaled, yTrain);
```

#### 4.5 Multicollinearity

**Problem**: When features are highly correlated with each other

**Example**:
```
Feature 1: House size in sqft
Feature 2: House size in square meters

These are perfectly correlated!
Causes numerical instability
```

**Detection**:
- Calculate correlation matrix
- Look for correlations > 0.9

**Solutions**:
1. Remove one of the correlated features
2. Combine correlated features
3. Use regularization (Ridge regression)

#### 4.6 Practice Exercise

**Exercise**: Predict car prices

```csharp
// Features: [age_years, mileage_k, engine_size_L, horsepower]
double[,] cars = new double[,] {
    { 1, 10, 2.0, 150 },   // $25k
    { 3, 30, 2.0, 150 },   // $20k
    { 5, 50, 1.8, 130 },   // $15k
    { 7, 70, 1.6, 110 },   // $10k
    { 10, 100, 1.6, 110 }  // $5k
};

double[] prices = new double[] { 25, 20, 15, 10, 5 };

// TODO:
// 1. Split data 80/20
// 2. Standardize features
// 3. Train multiple linear regression
// 4. Evaluate with RMSE and R²
// 5. Predict price for: [4 years, 40k miles, 1.8L, 140hp]
// 6. Which feature has the strongest effect on price?
```

**Expected Output**:
```
RMSE: ~$1-2k
R²: ~0.95-0.99
Prediction for [4, 40, 1.8, 140]: ~$17-18k
Most important feature: Age (strongest negative correlation)
```

---

## Part II: Intermediate Level

### Chapter 5: Regularization - Ridge Regression

#### 5.1 The Overfitting Problem

**Scenario**: Model with many features

```
Training R²: 0.99 ✅
Test R²: 0.65 ❌

Problem: Model memorized training data!
```

**Causes**:
- Too many features relative to samples
- Features are highly correlated
- Model is too complex

**Solution**: Regularization!

#### 5.2 Ridge Regression (L2 Regularization)

**Concept**: Add penalty for large weights

**Loss Function**:
```
Loss = MSE + α × Σwᵢ²
       ↑         ↑
    Fit data  Penalty

Where α = regularization strength
```

**Effect**:
- Shrinks weights toward zero
- Prevents any single feature from dominating
- Reduces model complexity

**How it Works**:
```
Without regularization:
Weights can be very large: [100, -50, 200, -150]

With regularization:
Weights are smaller: [10, -5, 20, -15]
```

#### 5.3 Choosing Alpha (α)

**Alpha controls regularization strength**:

```
α = 0:
- No regularization
- Same as linear regression
- May overfit

α small (0.01-0.1):
- Light regularization
- Slight reduction in overfitting

α medium (1-10):
- Moderate regularization
- Good balance

α large (100+):
- Strong regularization
- May underfit
- Weights very small
```

#### 5.4 Complete Implementation Example

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class RidgeRegressionExample
{
    public static void Main()
    {
        // Data with correlated features
        double[,] X = new double[,] {
            { 1, 2, 1.5 },
            { 2, 4, 3.0 },
            { 3, 6, 4.5 },
            { 4, 8, 6.0 },
            { 5, 10, 7.5 }
        };
        double[] y = new double[] { 3, 5, 7, 9, 11 };

        // Split data
        var (XTrain, yTrain, XTest, yTest) = TrainTestSplit(X, y, 0.4);

        // Try different alpha values
        double[] alphas = { 0.01, 0.1, 1.0, 10.0, 100.0 };

        Console.WriteLine("Alpha\t\tTrain R²\tTest R²\t\tGap");
        Console.WriteLine("------------------------------------------------");

        foreach (var alpha in alphas)
        {
            var model = new RidgeRegression(alpha: alpha);
            model.Fit(XTrain, yTrain);

            double[] yTrainPred = model.Predict(XTrain);
            double[] yTestPred = model.Predict(XTest);

            double trainR2 = RegressionMetrics.RSquared(yTrain, yTrainPred);
            double testR2 = RegressionMetrics.RSquared(yTest, yTestPred);
            double gap = trainR2 - testR2;

            Console.WriteLine($"{alpha}\t\t{trainR2:F4}\t\t{testR2:F4}\t\t{gap:F4}");
        }

        // Select best alpha (smallest gap)
        var bestModel = new RidgeRegression(alpha: 1.0);
        bestModel.Fit(XTrain, yTrain);
    }
}
```

#### 5.5 When to Use Ridge Regression

**Use Ridge when**:
- Many features (high-dimensional data)
- Features are correlated (multicollinearity)
- Model is overfitting
- All features are potentially useful

**Don't use Ridge when**:
- Few features relative to samples
- Need feature selection (use Lasso instead)
- Features are already well-behaved

---

### Chapter 6: Regularization - Lasso Regression

#### 6.1 Lasso vs Ridge

**Key Difference**: Lasso can set weights to exactly zero!

```
Ridge (L2): Shrinks weights toward zero
Weights: [0.5, 0.3, 0.2, 0.1]

Lasso (L1): Sets some weights to exactly zero
Weights: [0.8, 0.0, 0.4, 0.0]
```

**Loss Function**:
```
Loss = MSE + α × Σ|wᵢ|
              ↑
         L1 penalty (absolute value)
```

#### 6.2 Automatic Feature Selection

**Lasso's Superpower**: Performs feature selection automatically

**Example**:
```
10 features → Lasso → 3 features with non-zero weights

Result: Simpler, more interpretable model
```

**Why This Happens**:
- L1 penalty creates sparse solutions
- Forces less important features to exactly zero
- Keeps only the most important features

#### 6.3 Complete Implementation Example

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class LassoRegressionExample
{
    public static void Main()
    {
        // Data with many features (some irrelevant)
        double[,] X = new double[,] {
            { 1, 2, 3, 4, 5 },
            { 2, 4, 6, 8, 10 },
            { 3, 6, 9, 12, 15 },
            { 4, 8, 12, 16, 20 },
            { 5, 10, 15, 20, 25 }
        };
        // Only first 2 features are relevant
        double[] y = new double[] { 3, 5, 7, 9, 11 };

        // Try different alpha values
        double[] alphas = { 0.1, 0.5, 1.0, 5.0 };

        Console.WriteLine("Alpha\tNon-zero\tR²");
        Console.WriteLine("--------------------------------");

        foreach (var alpha in alphas)
        {
            var model = new LassoRegression(alpha: alpha);
            model.Fit(X, y);

            double[] predictions = model.Predict(X);
            double r2 = RegressionMetrics.RSquared(y, predictions);
            int nonZero = model.GetNonZeroWeightsCount();

            Console.WriteLine($"{alpha}\t{nonZero}\t\t{r2:F4}");
        }

        // Best model: Good R² with few features
        var bestModel = new LassoRegression(alpha: 0.5);
        bestModel.Fit(X, y);

        Console.WriteLine($"\nSelected {bestModel.GetNonZeroWeightsCount()} out of {X.GetLength(1)} features");
    }
}
```

#### 6.4 Ridge vs Lasso Comparison

| Aspect | Ridge (L2) | Lasso (L1) |
|--------|-----------|-----------|
| **Penalty** | Σwᵢ² | Σ\|wᵢ\| |
| **Weight shrinkage** | Toward zero | To exactly zero |
| **Feature selection** | No | Yes |
| **Sparse solution** | No | Yes |
| **Interpretability** | Moderate | High |
| **Use when** | All features useful | Some features irrelevant |
| **Multicollinearity** | Handles well | May arbitrarily select one |

#### 6.5 When to Use Lasso

**Use Lasso when**:
- Many features, only some are important
- Need interpretable model
- Want automatic feature selection
- Sparse solutions are desired

**Example Scenarios**:
- Gene expression data (thousands of genes, few relevant)
- Text classification (many words, few important)
- Sensor data (many sensors, some redundant)

---

### Chapter 7: Polynomial Regression

#### 7.1 Handling Non-Linear Relationships

**Problem**: Real-world relationships are often curved, not straight

**Example**:
```
Fertilizer vs Crop Yield:
- Too little: Low yield
- Optimal amount: High yield
- Too much: Yield decreases (diminishing returns)

This is a curve, not a line!
```

#### 7.2 Polynomial Features

**Concept**: Transform features to capture non-linearity

```
Original feature: x
Polynomial features: x, x², x³, ...

Example:
x = 2
Degree 2: [2, 4]
Degree 3: [2, 4, 8]
```

**Mathematical Form**:
```
Degree 1 (Linear): y = w₀ + w₁x
Degree 2: y = w₀ + w₁x + w₂x²
Degree 3: y = w₀ + w₁x + w₂x² + w₃x³
```

#### 7.3 Complete Implementation Example

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Regression;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class PolynomialRegressionExample
{
    public static void Main()
    {
        // Non-linear data: y = x²
        double[,] X = new double[,] {
            { 1 }, { 2 }, { 3 }, { 4 }, { 5 }
        };
        double[] y = new double[] { 1, 4, 9, 16, 25 };

        // Try different polynomial degrees
        int[] degrees = { 1, 2, 3, 4 };

        Console.WriteLine("Degree\tR²\t\tInterpretation");
        Console.WriteLine("------------------------------------------------");

        foreach (var degree in degrees)
        {
            var model = new PolynomialRegression(degree: degree);
            model.Fit(X, y);

            double[] predictions = model.Predict(X);
            double r2 = RegressionMetrics.RSquared(y, predictions);

            string interpretation = degree switch
            {
                1 => "Underfitting (linear for quadratic data)",
                2 => "Perfect fit (matches true relationship)",
                3 => "Starting to overfit",
                4 => "Overfitting (too complex)",
                _ => ""
            };

            Console.WriteLine($"{degree}\t{r2:F4}\t\t{interpretation}");
        }

        // Best model: Degree 2 (matches true relationship)
        var bestModel = new PolynomialRegression(degree: 2);
        bestModel.Fit(X, y);

        // Predict for new value
        double[,] XNew = new double[,] { { 6 } };
        double[] prediction = bestModel.Predict(XNew);
        Console.WriteLine($"\nPrediction for x=6: {prediction[0]:F1} (Expected: 36)");
    }
}
```

#### 7.4 Choosing the Right Degree

**Guidelines**:

```
Degree 1 (Linear):
- Straight line
- Use when relationship is linear
- Simplest, least prone to overfitting

Degree 2 (Quadratic):
- One curve (U-shape or inverted U)
- Most common for real-world data
- Good balance

Degree 3 (Cubic):
- S-shaped curves
- Use when data has inflection points
- More complex

Degree 4+:
- Multiple curves
- Rarely needed
- High risk of overfitting
```

**Visual Guide**:
```
Degree 1:  ──────  (Straight line)
Degree 2:  ╲_╱    (One curve)
Degree 3:  ╱‾╲_   (S-curve)
Degree 4:  ╲_╱‾╲  (Multiple curves)
```

#### 7.5 Avoiding Overfitting

**Problem**: High-degree polynomials can overfit

**Example**:
```
5 data points, degree 4 polynomial:
- Perfect fit on training data (R² = 1.0)
- Terrible on test data (R² < 0)
- Model memorized noise!
```

**Solutions**:
1. **Use cross-validation** to select degree
2. **Limit degree** to 2-3 for most problems
3. **Add regularization** (Ridge/Lasso with polynomial features)
4. **Get more data** to support higher degrees

---

### Chapter 8: Feature Engineering for Regression

#### 8.1 Why Feature Engineering Matters

**Impact**: Good features can improve model performance more than algorithm choice!

**Example**:
```
Bad features: Raw sensor readings
Model R²: 0.65

Good features: Engineered features (ratios, interactions)
Model R²: 0.92

Same algorithm, better features!
```

#### 8.2 Feature Transformations

**Common Transformations**:

**1. Log Transform**:
```csharp
// For skewed data (e.g., income, prices)
double[] logFeature = feature.Select(x => Math.Log(x + 1)).ToArray();

// Why: Reduces impact of outliers, makes distribution more normal
```

**2. Square Root Transform**:
```csharp
// For count data
double[] sqrtFeature = feature.Select(x => Math.Sqrt(x)).ToArray();
```

**3. Binning**:
```csharp
// Convert continuous to categorical
// Age → Age groups: [0-18, 19-35, 36-60, 60+]
```

#### 8.3 Feature Interactions

**Concept**: Combine features to capture relationships

**Example - House Prices**:
```csharp
// Original features
double size = 2000;  // sqft
double quality = 8;  // 1-10 scale

// Interaction feature
double sizeQuality = size * quality;  // 16000

// Interpretation: Large, high-quality house is worth more
// than sum of individual effects
```

**Code Example**:
```csharp
public static double[,] AddInteractionFeatures(double[,] X)
{
    int n = X.GetLength(0);
    int m = X.GetLength(1);

    // Add all pairwise interactions
    int newFeatures = m + (m * (m - 1)) / 2;
    double[,] XNew = new double[n, newFeatures];

    for (int i = 0; i < n; i++)
    {
        int idx = 0;

        // Original features
        for (int j = 0; j < m; j++)
            XNew[i, idx++] = X[i, j];

        // Interaction features
        for (int j = 0; j < m; j++)
            for (int k = j + 1; k < m; k++)
                XNew[i, idx++] = X[i, j] * X[i, k];
    }

    return XNew;
}
```

#### 8.4 Domain-Specific Features

**Real Estate Example**:
```csharp
// Raw features
double size = 2000;
double bedrooms = 3;
double bathrooms = 2;

// Engineered features
double pricePerSqft = price / size;
double roomsPerBathroom = (bedrooms + 1) / bathrooms;
double isLuxury = (size > 3000 && bathrooms > 3) ? 1 : 0;
```

**Time Series Example**:
```csharp
// Raw: Daily sales
// Engineered:
double movingAverage7Day = CalculateMA(sales, 7);
double dayOfWeek = DateTime.Now.DayOfWeek;
double isWeekend = (dayOfWeek == 0 || dayOfWeek == 6) ? 1 : 0;
double monthOfYear = DateTime.Now.Month;
```

#### 8.5 Feature Selection

**Goal**: Keep only useful features

**Methods**:

**1. Correlation Analysis**:
```csharp
// Remove features with low correlation to target
double correlation = CalculateCorrelation(feature, target);
if (Math.Abs(correlation) < 0.1)
    RemoveFeature(feature);
```

**2. Lasso Regression**:
```csharp
// Automatic feature selection
var lasso = new LassoRegression(alpha: 0.5);
lasso.Fit(X, y);
// Features with zero weights are removed
```

**3. Forward Selection**:
```
1. Start with no features
2. Add feature that improves model most
3. Repeat until no improvement
```

**4. Backward Elimination**:
```
1. Start with all features
2. Remove feature that hurts model least
3. Repeat until removing any feature hurts performance
```

---

## Part III: Advanced Level

### Chapter 9: Advanced Evaluation Techniques

#### 9.1 Beyond R² and RMSE

**Additional Metrics**:

**1. Mean Absolute Percentage Error (MAPE)**:
```
MAPE = (100/n) × Σ|yᵢ - ŷᵢ| / |yᵢ|

Interpretation: Average percentage error
Example: MAPE = 5% means predictions are off by 5% on average
```

**2. Adjusted R²**:
```
Adjusted R² = 1 - [(1-R²) × (n-1) / (n-p-1)]

Where:
- n = number of samples
- p = number of features

Penalizes adding unnecessary features
```

**Code Example**:
```csharp
public static double CalculateMAPE(double[] yTrue, double[] yPred)
{
    double sum = 0;
    int count = 0;

    for (int i = 0; i < yTrue.Length; i++)
    {
        if (Math.Abs(yTrue[i]) > 1e-10)  // Avoid division by zero
        {
            sum += Math.Abs((yTrue[i] - yPred[i]) / yTrue[i]);
            count++;
        }
    }

    return (sum / count) * 100;
}
```

#### 9.2 Residual Analysis

**Residuals**: Differences between actual and predicted values

```
Residual = yᵢ - ŷᵢ
```

**What to Check**:

**1. Residual Plot**:
```
Good pattern:
Residuals ↑
         │  ●  ●
         │ ● ● ●
    0────┼────────  (Random scatter around 0)
         │ ● ● ●
         │  ●  ●
         └──────────→ Predictions

Bad pattern:
Residuals ↑
         │    ●●●
         │  ●●
    0────┼●●────────  (Curved pattern = non-linearity)
         │
         │
         └──────────→ Predictions
```

**2. Normality of Residuals**:
```csharp
// Check if residuals are normally distributed
double[] residuals = CalculateResiduals(yTrue, yPred);
// Plot histogram or Q-Q plot
```

**3. Homoscedasticity**:
```
Check if variance of residuals is constant
If variance increases with predictions → Problem!
```

#### 9.3 Cross-Validation for Regression

**K-Fold Cross-Validation**:

```csharp
public static double CrossValidateRegression(double[,] X, double[] y, int k = 5)
{
    int n = y.Length;
    int foldSize = n / k;
    double[] scores = new double[k];

    for (int fold = 0; fold < k; fold++)
    {
        // Create fold
        var (XTrain, yTrain, XTest, yTest) = CreateFold(X, y, fold, foldSize);

        // Train and evaluate
        var model = new LinearRegression();
        model.Fit(XTrain, yTrain);
        double[] yPred = model.Predict(XTest);

        scores[fold] = RegressionMetrics.RSquared(yTest, yPred);
    }

    double mean = scores.Average();
    double std = CalculateStdDev(scores);

    Console.WriteLine($"Cross-Validation Results:");
    Console.WriteLine($"Mean R²: {mean:F4} ± {std:F4}");

    return mean;
}
```

---

### Chapter 10: Handling Real-World Challenges

#### 10.1 Missing Data

**Strategies**:

**1. Remove Rows**:
```csharp
// Simple but loses data
var cleanData = data.Where(row => !HasMissingValues(row));
```

**2. Mean/Median Imputation**:
```csharp
public static void ImputeMissing(double[,] X, int featureIndex)
{
    // Calculate mean of non-missing values
    double sum = 0;
    int count = 0;

    for (int i = 0; i < X.GetLength(0); i++)
    {
        if (!double.IsNaN(X[i, featureIndex]))
        {
            sum += X[i, featureIndex];
            count++;
        }
    }

    double mean = sum / count;

    // Fill missing values with mean
    for (int i = 0; i < X.GetLength(0); i++)
    {
        if (double.IsNaN(X[i, featureIndex]))
            X[i, featureIndex] = mean;
    }
}
```

**3. Predictive Imputation**:
```csharp
// Use other features to predict missing values
// Train a model to predict the missing feature
```

#### 10.2 Outliers

**Detection**:

**1. Z-Score Method**:
```csharp
public static bool IsOutlier(double value, double mean, double std)
{
    double zScore = Math.Abs((value - mean) / std);
    return zScore > 3;  // 3 standard deviations
}
```

**2. IQR Method**:
```csharp
public static bool IsOutlierIQR(double value, double q1, double q3)
{
    double iqr = q3 - q1;
    double lowerBound = q1 - 1.5 * iqr;
    double upperBound = q3 + 1.5 * iqr;
    return value < lowerBound || value > upperBound;
}
```

**Handling**:
- Remove if data error
- Cap at threshold (winsorization)
- Use robust regression methods
- Transform data (log transform)

#### 10.3 Heteroscedasticity

**Problem**: Variance of errors is not constant

**Detection**:
```
Plot residuals vs predictions
If spread increases → Heteroscedasticity
```

**Solutions**:
1. **Transform target variable** (log, sqrt)
2. **Weighted least squares**
3. **Use robust standard errors**

---

### Chapter 11: Model Selection and Comparison

#### 11.1 Comparing Multiple Models

```csharp
public static void CompareRegressionModels(double[,] X, double[] y)
{
    var models = new Dictionary<string, dynamic>
    {
        { "Linear Regression", new LinearRegression() },
        { "Ridge (α=1.0)", new RidgeRegression(alpha: 1.0) },
        { "Lasso (α=0.5)", new LassoRegression(alpha: 0.5) },
        { "Polynomial (degree=2)", new PolynomialRegression(degree: 2) }
    };

    var (XTrain, yTrain, XTest, yTest) = TrainTestSplit(X, y, 0.2);

    Console.WriteLine($"{"Model",-30} {"RMSE",-10} {"R²",-10} {"Time (ms)",-12}");
    Console.WriteLine(new string('-', 62));

    foreach (var (name, model) in models)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();

        model.Fit(XTrain, yTrain);
        double[] yPred = model.Predict(XTest);

        sw.Stop();

        double rmse = RegressionMetrics.RootMeanSquaredError(yTest, yPred);
        double r2 = RegressionMetrics.RSquared(yTest, yPred);

        Console.WriteLine($"{name,-30} {rmse,-10:F2} {r2,-10:F4} {sw.ElapsedMilliseconds,-12}");
    }
}
```

#### 11.2 Hyperparameter Tuning

**Grid Search for Alpha**:

```csharp
public static double FindBestAlpha(double[,] X, double[] y)
{
    double[] alphas = { 0.001, 0.01, 0.1, 1.0, 10.0, 100.0 };
    double bestAlpha = 0;
    double bestScore = double.MinValue;

    foreach (var alpha in alphas)
    {
        // Use cross-validation
        double score = CrossValidateWithAlpha(X, y, alpha);

        if (score > bestScore)
        {
            bestScore = score;
            bestAlpha = alpha;
        }
    }

    Console.WriteLine($"Best alpha: {bestAlpha} with R² = {bestScore:F4}");
    return bestAlpha;
}
```

---

### Chapter 12: Production Deployment

#### 12.1 Model Serialization

```csharp
// Save model
public static void SaveModel(LinearRegression model, string filepath)
{
    // Serialize model parameters
    var modelData = new
    {
        Weights = model.GetWeights(),
        Intercept = model.GetIntercept()
    };

    string json = JsonSerializer.Serialize(modelData);
    File.WriteAllText(filepath, json);
}

// Load model
public static LinearRegression LoadModel(string filepath)
{
    string json = File.ReadAllText(filepath);
    var modelData = JsonSerializer.Deserialize<ModelData>(json);

    var model = new LinearRegression();
    model.SetWeights(modelData.Weights);
    model.SetIntercept(modelData.Intercept);

    return model;
}
```

#### 12.2 API Deployment

```csharp
[ApiController]
[Route("api/[controller]")]
public class PredictionController : ControllerBase
{
    private static LinearRegression _model;

    static PredictionController()
    {
        _model = LoadModel("model.json");
    }

    [HttpPost("predict")]
    public IActionResult Predict([FromBody] PredictionRequest request)
    {
        double[,] features = request.ToFeatureArray();
        double[] prediction = _model.Predict(features);

        return Ok(new { prediction = prediction[0] });
    }
}
```

#### 12.3 Monitoring

```csharp
public class ModelMonitor
{
    private Queue<double> _errors = new Queue<double>();

    public void LogPrediction(double prediction, double actual)
    {
        double error = Math.Abs(prediction - actual);
        _errors.Enqueue(error);

        if (_errors.Count > 1000)
            _errors.Dequeue();

        double avgError = _errors.Average();

        if (avgError > threshold)
        {
            Console.WriteLine("⚠️ Model performance degraded!");
            // Trigger retraining
        }
    }
}
```

---

## Conclusion

Congratulations! You've mastered regression from beginner to expert level.

### Key Takeaways

1. **Start Simple**: Begin with linear regression
2. **Regularize**: Use Ridge/Lasso to prevent overfitting
3. **Engineer Features**: Good features matter more than algorithms
4. **Evaluate Properly**: Use multiple metrics and cross-validation
5. **Monitor Production**: Track performance over time

### Algorithm Selection Guide

| Scenario | Recommended Algorithm |
|----------|----------------------|
| **Linear relationship, few features** | Linear Regression |
| **Many correlated features** | Ridge Regression |
| **Need feature selection** | Lasso Regression |
| **Non-linear relationship** | Polynomial Regression |
| **High-dimensional data** | Ridge or Lasso |

### Resources for .NET/C# Machine Learning

- ML.NET Documentation
- Accord.NET Framework
- Math.NET Numerics
- "Programming ML.NET" by Matt R. Cole

---

## Alternative Technology Stack

**Note**: While this tutorial uses **.NET C#**, the concepts apply to any language.

**Python Alternative**: Python offers excellent regression tools:
- **scikit-learn**: Comprehensive regression algorithms
- **statsmodels**: Statistical modeling
- **NumPy/Pandas**: Data manipulation

Both ecosystems are mature and production-ready.

---

**Happy Regressing!** 📈

*Last Updated: 2026-01-29*

