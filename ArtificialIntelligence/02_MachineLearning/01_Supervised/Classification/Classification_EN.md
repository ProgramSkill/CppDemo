# Classification: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: What is Classification?](#chapter-1-what-is-classification)
  - [Chapter 2: Understanding Classification Problems](#chapter-2-understanding-classification-problems)
  - [Chapter 3: Logistic Regression - Your First Classifier](#chapter-3-logistic-regression---your-first-classifier)
  - [Chapter 4: K-Nearest Neighbors (KNN)](#chapter-4-k-nearest-neighbors-knn)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 5: Decision Trees](#chapter-5-decision-trees)
  - [Chapter 6: Naive Bayes Classifier](#chapter-6-naive-bayes-classifier)
  - [Chapter 7: Evaluation Metrics Deep Dive](#chapter-7-evaluation-metrics-deep-dive)
  - [Chapter 8: The Confusion Matrix](#chapter-8-the-confusion-matrix)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 9: Handling Imbalanced Datasets](#chapter-9-handling-imbalanced-datasets)
  - [Chapter 10: Multi-class Classification](#chapter-10-multi-class-classification)
  - [Chapter 11: Model Selection and Comparison](#chapter-11-model-selection-and-comparison)
  - [Chapter 12: Production Deployment](#chapter-12-production-deployment)

---

## Introduction

Welcome to the comprehensive guide on **Classification**! This tutorial will take you from understanding basic concepts to building production-ready classification systems.

### What You'll Learn

| Level | Duration | Topics Covered | Skills Acquired |
|-------|----------|----------------|-----------------|
| **Beginner** | 2-3 weeks | Classification basics, Logistic Regression, KNN | Build simple classifiers, understand evaluation |
| **Intermediate** | 3-5 weeks | Decision Trees, Naive Bayes, Advanced metrics | Choose appropriate algorithms, handle real data |
| **Advanced** | 4-6 weeks | Imbalanced data, Multi-class problems, Deployment | Build production systems, optimize performance |

### Prerequisites

- Basic understanding of supervised learning
- Familiarity with C# programming
- Basic statistics knowledge
- Understanding of linear algebra (helpful but not required)

### How to Use This Guide

1. **Sequential Learning**: Follow chapters in order for structured learning
2. **Hands-On Practice**: Complete all code examples and exercises
3. **Real Projects**: Apply concepts through case studies
4. **Reference**: Use as a quick reference when needed

---

## Part I: Beginner Level

### Chapter 1: What is Classification?

#### 1.1 The Big Picture

**Classification** is a supervised learning task where we predict **discrete categories** (classes) rather than continuous values.

**Real-World Analogy**:
Think of a mail sorting facility:
- Workers look at each letter (input features)
- They decide which bin it goes into (class prediction)
- "Local", "National", or "International" (discrete categories)

In classification:
- **Letters** = Input samples
- **Address, stamps, size** = Features
- **Sorting decision** = Classification
- **Bins** = Classes/Categories

#### 1.2 Classification vs Regression

**Key Difference**:

```
Regression: Predicts numbers
Example: House price = $350,000 (continuous)

Classification: Predicts categories
Example: Email type = "Spam" (discrete)
```

**Comparison Table**:

| Aspect | Regression | Classification |
|--------|-----------|----------------|
| **Output** | Continuous values | Discrete categories |
| **Example** | Predict temperature: 72.5°F | Predict weather: "Sunny" |
| **Evaluation** | MSE, RMSE, R² | Accuracy, Precision, Recall |
| **Algorithms** | Linear Regression, Ridge | Logistic Regression, Decision Trees |

#### 1.3 Types of Classification Problems

**1. Binary Classification**
- Only 2 classes
- Examples:
  - Email: Spam or Not Spam
  - Medical: Disease or Healthy
  - Credit: Approved or Denied

**2. Multi-class Classification**
- 3 or more mutually exclusive classes
- Examples:
  - Iris species: Setosa, Versicolor, or Virginica
  - Handwritten digits: 0, 1, 2, ..., 9
  - Product categories: Electronics, Clothing, Books

**3. Multi-label Classification**
- Multiple classes can be assigned to one sample
- Examples:
  - News article tags: [Politics, Economy, International]
  - Movie genres: [Action, Comedy, Drama]
  - Image tags: [Cat, Outdoor, Sunny]

#### 1.4 Real-World Applications

**Business Applications**:
- 📧 **Spam Detection**: Filter unwanted emails
- 💳 **Credit Scoring**: Assess loan risk
- 🛒 **Customer Segmentation**: Group customers by behavior
- 📞 **Churn Prediction**: Identify customers likely to leave

**Healthcare Applications**:
- 🏥 **Disease Diagnosis**: Detect diseases from symptoms
- 🔬 **Medical Image Analysis**: Identify tumors in scans
- 💊 **Drug Response**: Predict treatment effectiveness

**Technology Applications**:
- 🖼️ **Image Recognition**: Classify objects in photos
- 📝 **Sentiment Analysis**: Determine opinion polarity
- 🗣️ **Speech Recognition**: Convert speech to text
- 🤖 **Fraud Detection**: Identify suspicious transactions

#### 1.5 The Classification Workflow

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
│  (Train/Val/Test)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Choose      │
│     Classifier  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. Train       │
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  7. Evaluate    │
│     Performance │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  8. Tune &      │
│     Optimize    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  9. Deploy      │
│     Model       │
└─────────────────┘
```

#### 1.6 Quick Start Example

Let's see classification in action with a simple spam detection example:

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

// Problem: Classify emails as spam or not spam
// Features: [word_count_free, word_count_win, exclamation_marks]
double[,] emails = new double[,] {
    { 0, 0, 0 },  // Normal email
    { 1, 0, 1 },  // Normal email
    { 5, 3, 5 },  // Spam
    { 8, 5, 8 },  // Spam
    { 0, 1, 0 },  // Normal email
    { 10, 8, 10 } // Spam
};

int[] labels = new int[] { 0, 0, 1, 1, 0, 1 }; // 0=Normal, 1=Spam

// Create and train classifier
var classifier = new LogisticRegression(learningRate: 0.1, maxIterations: 1000);
classifier.Fit(emails, labels);

// Classify new email
double[,] newEmail = new double[,] { { 6, 4, 6 } };
int[] prediction = classifier.Predict(newEmail);
double[] probability = classifier.PredictProba(newEmail);

Console.WriteLine($"Prediction: {(prediction[0] == 1 ? "Spam" : "Normal")}");
Console.WriteLine($"Spam probability: {probability[0]:P2}");

// Evaluate model
int[] predictions = classifier.Predict(emails);
double accuracy = ClassificationMetrics.Accuracy(labels, predictions);
Console.WriteLine($"Accuracy: {accuracy:P2}");
```

**What just happened?**
1. We provided training data (email features → spam/normal labels)
2. The classifier learned patterns distinguishing spam from normal emails
3. We classified a new email based on learned patterns
4. We evaluated the classifier's accuracy

---

### Chapter 2: Understanding Classification Problems

#### 2.1 The Decision Boundary

**Concept**: The decision boundary is the line (or surface) that separates different classes in feature space.

**2D Example** (2 features):
```
Feature 2 ↑
          │
    ●     │     ○
  ●   ●   │   ○   ○
    ●     │     ○
──────────┼──────────→ Feature 1
    ●     │     ○
  ●   ●   │   ○   ○
    ●     │     ○

● = Class 0
○ = Class 1
│ = Decision Boundary
```

**Linear vs Non-linear Boundaries**:

```
Linear Boundary:          Non-linear Boundary:
    ●   │   ○                 ●     ○
  ●   ● │ ○   ○             ●   ╱─╲   ○
    ●   │   ○               ● ╱     ╲ ○
────────┼────────          ●│   ●   │○
    ●   │   ○               ● ╲     ╱ ○
  ●   ● │ ○   ○             ●   ╲─╱   ○
    ●   │   ○                 ●     ○
```

#### 2.2 Features and Labels

**Features (X)**:
- Input variables used for prediction
- Can be numerical or categorical
- Quality matters more than quantity

**Example - Email Classification**:
```csharp
// Features for email classification
double[,] features = new double[,] {
    // [length, num_links, num_images, has_attachments, spam_words]
    { 100,  0,  1,  0,  0 },  // Normal email
    { 50,   10, 5,  1,  15 }, // Spam email
    { 200,  2,  3,  1,  1 },  // Normal email
};
```

**Labels (y)**:
- Output variable we want to predict
- Must be discrete categories
- Encoded as integers (0, 1, 2, ...)

```csharp
int[] labels = new int[] { 0, 1, 0 }; // 0=Normal, 1=Spam
```

#### 2.3 Training vs Testing

**Why Split Data?**

Imagine studying for an exam:
- **Training set** = Practice problems you study from
- **Test set** = Actual exam questions (never seen before)

If you memorize practice problems without understanding concepts, you'll fail the real exam. Similarly, models need to **generalize** to new data.

**Typical Split**:
```
Total Data: 100 samples
├─ Training: 70 samples (70%)
├─ Validation: 15 samples (15%)
└─ Test: 15 samples (15%)
```

**Code Example**:

```csharp
public static (double[,], int[], double[,], int[])
    TrainTestSplit(double[,] X, int[] y, double testSize = 0.2)
{
    int n = y.Length;
    int m = X.GetLength(1);
    int testCount = (int)(n * testSize);
    int trainCount = n - testCount;

    // Shuffle indices
    var indices = Enumerable.Range(0, n).OrderBy(x => Random.Shared.Next()).ToArray();

    // Allocate arrays
    double[,] XTrain = new double[trainCount, m];
    int[] yTrain = new int[trainCount];
    double[,] XTest = new double[testCount, m];
    int[] yTest = new int[testCount];

    // Fill training set
    for (int i = 0; i < trainCount; i++)
    {
        int idx = indices[i];
        for (int j = 0; j < m; j++)
            XTrain[i, j] = X[idx, j];
        yTrain[i] = y[idx];
    }

    // Fill test set
    for (int i = 0; i < testCount; i++)
    {
        int idx = indices[trainCount + i];
        for (int j = 0; j < m; j++)
            XTest[i, j] = X[idx, j];
        yTest[i] = y[idx];
    }

    return (XTrain, yTrain, XTest, yTest);
}
```

#### 2.4 Overfitting in Classification

**Definition**: Model performs well on training data but poorly on new data.

**Visual Example**:

```
Training Data:        Overfit Model:        Good Model:
  ●     ○               ●─────○               ●     ○
●   ●     ○           ●─╱ ╲─╱  ○           ●   ●  │  ○
  ●   ○   ○             ╱   ╲   ○           ●   ○ │ ○
                      ●─╱     ╲─○                 │
```

**Signs of Overfitting**:
```
Training Accuracy: 99% ✅
Test Accuracy: 65% ❌

Gap = 34% (Too large!)
```

**Solutions**:
1. **More training data**: Provides more examples to learn from
2. **Simpler model**: Reduce model complexity
3. **Regularization**: Add penalty for complexity
4. **Cross-validation**: Better estimate of generalization
5. **Early stopping**: Stop training before overfitting occurs

#### 2.5 Class Balance

**Balanced Dataset**:
```
Class 0: 500 samples (50%)
Class 1: 500 samples (50%)
✅ Balanced - Easy to learn
```

**Imbalanced Dataset**:
```
Class 0: 950 samples (95%)
Class 1: 50 samples (5%)
❌ Imbalanced - Model biased toward majority class
```

**Why It Matters**:
A naive model that always predicts the majority class achieves 95% accuracy but is useless!

**Example**:
```csharp
// Fraud detection dataset
// 99% legitimate transactions, 1% fraud

// Naive model: Always predict "legitimate"
// Accuracy: 99% ✅
// But catches 0% of fraud! ❌
```

**We'll cover solutions in Chapter 9.**

#### 2.6 Feature Scaling for Classification

**Why Scale?**

Some algorithms (like KNN) are sensitive to feature scales:

```
Feature 1: Age (20-80)
Feature 2: Income ($20,000-$200,000)

Without scaling:
Distance dominated by income!
```

**Standardization** (Z-score normalization):
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
```

**When to Scale**:
- ✅ KNN, SVM, Neural Networks
- ❌ Decision Trees, Naive Bayes (not needed)

---

### Chapter 3: Logistic Regression - Your First Classifier

#### 3.1 Why "Regression" for Classification?

**Confusing Name Alert!** 🚨

Despite its name, **Logistic Regression is a classification algorithm**, not regression!

**History**: Named "regression" because it uses regression techniques, but outputs probabilities for classification.

#### 3.2 The Sigmoid Function

**Problem**: Linear regression outputs any value (-∞ to +∞), but we need probabilities (0 to 1).

**Solution**: Sigmoid function!

```
σ(z) = 1 / (1 + e^(-z))

Where: z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

**Properties**:
- Input: Any real number
- Output: Between 0 and 1
- S-shaped curve
- σ(0) = 0.5

**Visualization**:
```
σ(z) ↑
1.0  │         ╱────
     │       ╱
0.5  │     ╱
     │   ╱
0.0  │─╱─────────────→ z
    -∞  -5  0  5  ∞
```

#### 3.3 From Probability to Prediction

**Sigmoid Output**: Probability that sample belongs to class 1

```
P(y=1|x) = σ(w^T x)
```

**Decision Rule**:
```
If P(y=1|x) ≥ 0.5 → Predict Class 1
If P(y=1|x) < 0.5 → Predict Class 0
```

**Example**:
```csharp
// Email features: [spam_words, links]
// Learned weights: w₀=-2, w₁=0.5, w₂=0.3

// New email: [spam_words=6, links=4]
z = -2 + 0.5*6 + 0.3*4 = -2 + 3 + 1.2 = 2.2
P(spam) = 1 / (1 + e^(-2.2)) = 0.90

// Since 0.90 > 0.5 → Predict Spam
```

#### 3.4 Training Logistic Regression

**Loss Function**: Cross-Entropy (Log Loss)

```
Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

Where:
- y = actual label (0 or 1)
- ŷ = predicted probability
```

**Why This Loss?**
- Penalizes confident wrong predictions heavily
- Rewards confident correct predictions
- Convex (has single global minimum)

**Optimization**: Gradient Descent

```
Repeat until convergence:
  w := w - α * ∇Loss

Where:
- α = learning rate
- ∇Loss = gradient of loss function
```

#### 3.5 Complete Implementation Example

**Problem**: Credit Card Fraud Detection

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class FraudDetectionExample
{
    public static void Main()
    {
        // Step 1: Prepare data
        // Features: [transaction_amount, time_since_last, num_transactions_today]
        double[,] XTrain = new double[,] {
            { 50,   120,  2 },  // Legitimate
            { 100,  60,   3 },  // Legitimate
            { 5000, 5,    10 }, // Fraud
            { 75,   90,   2 },  // Legitimate
            { 8000, 2,    15 }, // Fraud
            { 200,  45,   4 },  // Legitimate
            { 10000, 1,   20 }, // Fraud
            { 150,  75,   3 }   // Legitimate
        };

        int[] yTrain = new int[] { 0, 0, 1, 0, 1, 0, 1, 0 }; // 0=Legit, 1=Fraud

        // Step 2: Feature scaling (important for gradient descent!)
        double[,] XTrainScaled = StandardizeFeatures(XTrain);

        // Step 3: Create and train model
        var model = new LogisticRegression(
            learningRate: 0.1,
            maxIterations: 1000
        );

        model.Fit(XTrainScaled, yTrain);

        // Step 4: Make predictions
        double[,] XTest = new double[,] {
            { 6000, 3, 12 }  // Suspicious transaction
        };
        double[,] XTestScaled = StandardizeFeatures(XTest);

        int[] prediction = model.Predict(XTestScaled);
        double[] probability = model.PredictProba(XTestScaled);

        Console.WriteLine($"Prediction: {(prediction[0] == 1 ? "Fraud" : "Legitimate")}");
        Console.WriteLine($"Fraud probability: {probability[0]:P2}");

        // Step 5: Evaluate on training set
        int[] yPred = model.Predict(XTrainScaled);

        double accuracy = ClassificationMetrics.Accuracy(yTrain, yPred);
        double precision = ClassificationMetrics.Precision(yTrain, yPred, positiveClass: 1);
        double recall = ClassificationMetrics.Recall(yTrain, yPred, positiveClass: 1);
        double f1 = ClassificationMetrics.F1Score(yTrain, yPred, positiveClass: 1);

        Console.WriteLine($"\nModel Performance:");
        Console.WriteLine($"Accuracy:  {accuracy:P2}");
        Console.WriteLine($"Precision: {precision:P2}");
        Console.WriteLine($"Recall:    {recall:P2}");
        Console.WriteLine($"F1-Score:  {f1:P2}");
    }

    static double[,] StandardizeFeatures(double[,] X)
    {
        // Implementation from Chapter 2.6
        int n = X.GetLength(0);
        int m = X.GetLength(1);
        double[,] XScaled = new double[n, m];

        for (int j = 0; j < m; j++)
        {
            double mean = 0, std = 0;
            for (int i = 0; i < n; i++)
                mean += X[i, j];
            mean /= n;

            for (int i = 0; i < n; i++)
                std += Math.Pow(X[i, j] - mean, 2);
            std = Math.Sqrt(std / n);

            for (int i = 0; i < n; i++)
                XScaled[i, j] = (X[i, j] - mean) / (std > 0 ? std : 1);
        }

        return XScaled;
    }
}
```

#### 3.6 Hyperparameters

**Learning Rate (α)**:
- Too small: Slow convergence
- Too large: May overshoot minimum
- Typical values: 0.001 to 0.1

**Max Iterations**:
- Number of training epochs
- Stop when loss converges
- Typical values: 100 to 10,000

**Convergence Check**:
```csharp
// Stop if loss change is very small
if (Math.Abs(currentLoss - previousLoss) < 1e-6)
    break;
```

#### 3.7 Advantages and Limitations

**Advantages**:
- ✅ **Probabilistic output**: Get confidence scores
- ✅ **Interpretable**: Can analyze feature importance
- ✅ **Fast training**: Efficient for large datasets
- ✅ **No hyperparameters**: Only learning rate and iterations
- ✅ **Works well**: Good baseline for binary classification

**Limitations**:
- ❌ **Linear decision boundary**: Can't handle complex patterns
- ❌ **Binary only**: Need modifications for multi-class
- ❌ **Feature scaling**: Sensitive to feature scales
- ❌ **Outliers**: Can be affected by extreme values

#### 3.8 When to Use Logistic Regression

**Good For**:
- Binary classification problems
- When you need probability estimates
- When interpretability is important
- Large datasets with many features
- Baseline model before trying complex algorithms

**Not Good For**:
- Non-linear decision boundaries
- When features have complex interactions
- Very small datasets (may overfit)

#### 3.9 Practice Exercises

**Exercise 1: Student Pass/Fail Prediction**

```csharp
// Predict if student will pass based on study hours and attendance
// Features: [study_hours_per_week, attendance_percentage]
double[,] students = new double[,] {
    { 2,  60 },  // Fail
    { 5,  80 },  // Pass
    { 8,  90 },  // Pass
    { 1,  50 },  // Fail
    { 10, 95 },  // Pass
    { 3,  65 }   // Fail
};

int[] passed = new int[] { 0, 1, 1, 0, 1, 0 }; // 0=Fail, 1=Pass

// TODO:
// 1. Split data 80/20
// 2. Train logistic regression
// 3. Evaluate accuracy
// 4. Predict for new student: [6 hours, 85% attendance]
```

**Expected Output**:
```
Training Accuracy: ~85%
Prediction for [6, 85]: Pass (probability ~0.75)
```

**Exercise 2: Email Spam Detection**

Build a spam classifier using these features:
- Number of spam words
- Number of links
- Email length
- Number of exclamation marks

Test with at least 10 training examples and evaluate using precision and recall.

---

### Chapter 4: K-Nearest Neighbors (KNN)

#### 4.1 The Intuition

**Core Idea**: "You are the average of your K nearest neighbors"

**Real-World Analogy**:
Imagine moving to a new neighborhood and wanting to know if it's safe:
- Look at the K closest houses
- If most are safe → Neighborhood is probably safe
- If most have security issues → Neighborhood might be risky

**In Machine Learning**:
- Find K training samples closest to the test point
- Take a majority vote among these K neighbors
- Assign the most common class

#### 4.2 How KNN Works

**Step-by-Step Process**:

1. **Calculate Distance**: Measure distance from test point to all training points
2. **Find K Nearest**: Select K closest training points
3. **Vote**: Count class labels among K neighbors
4. **Predict**: Assign majority class

**Visual Example** (K=3):

```
Training Data:          Find 3 Nearest:        Prediction:
  ●     ○                 ●     ○
●   ●     ○             ●  ●?     ○            ? → Class ●
  ●   ○   ○               ●  ╱○   ○            (2 ● vs 1 ○)
                            ╱
                          ●

● = Class 0
○ = Class 1
? = Test point
```

#### 4.3 Distance Metrics

**Euclidean Distance** (most common):
```
d(p, q) = √[(p₁-q₁)² + (p₂-q₂)² + ... + (pₙ-qₙ)²]
```

**Example**:
```csharp
public static double EuclideanDistance(double[] point1, double[] point2)
{
    double sum = 0;
    for (int i = 0; i < point1.Length; i++)
    {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return Math.Sqrt(sum);
}
```

**Other Distance Metrics**:

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Manhattan** | Σ\|pᵢ-qᵢ\| | Grid-like paths, high dimensions |
| **Minkowski** | (Σ\|pᵢ-qᵢ\|ᵖ)^(1/p) | Generalization of Euclidean/Manhattan |
| **Cosine** | 1 - (p·q)/(‖p‖‖q‖) | Text classification, high dimensions |

#### 4.4 Choosing K

**The K Dilemma**:

```
K = 1:                    K = 5:                    K = 15:
  ●     ○                   ●     ○                   ●     ○
●  ?●     ○               ●  ?●     ○               ●  ?●     ○
  ●   ○   ○                 ●   ○   ○                 ●   ○   ○

Overfitting              Good Balance              Underfitting
(Too sensitive)          (Just right)              (Too smooth)
```

**Guidelines**:

| K Value | Effect | When to Use |
|---------|--------|-------------|
| **K = 1** | Very flexible, noisy | Never (too sensitive) |
| **K = 3-5** | Flexible | Small datasets |
| **K = 10-20** | Balanced | Medium datasets |
| **K = √n** | Rule of thumb | General guideline |
| **K = n** | Always predicts majority class | Never (useless) |

**Important**: Always use **odd K** to avoid ties in binary classification!

#### 4.5 Complete Implementation Example

**Problem**: Iris Flower Classification

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class IrisClassificationExample
{
    public static void Main()
    {
        // Step 1: Prepare Iris dataset
        // Features: [sepal_length, sepal_width, petal_length, petal_width]
        double[,] XTrain = new double[,] {
            // Setosa (Class 0)
            { 5.1, 3.5, 1.4, 0.2 },
            { 4.9, 3.0, 1.4, 0.2 },
            { 4.7, 3.2, 1.3, 0.2 },
            // Versicolor (Class 1)
            { 7.0, 3.2, 4.7, 1.4 },
            { 6.4, 3.2, 4.5, 1.5 },
            { 6.9, 3.1, 4.9, 1.5 },
            // Virginica (Class 2)
            { 6.3, 3.3, 6.0, 2.5 },
            { 5.8, 2.7, 5.1, 1.9 },
            { 7.1, 3.0, 5.9, 2.1 }
        };

        int[] yTrain = new int[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 };

        // Step 2: Feature scaling (IMPORTANT for KNN!)
        double[,] XTrainScaled = StandardizeFeatures(XTrain);

        // Step 3: Try different K values
        int[] kValues = { 1, 3, 5, 7 };

        Console.WriteLine("Testing different K values:\n");

        foreach (int k in kValues)
        {
            var model = new KNearestNeighbors(k: k);
            model.Fit(XTrainScaled, yTrain);

            // Evaluate on training set
            int[] yPred = model.Predict(XTrainScaled);
            double accuracy = ClassificationMetrics.Accuracy(yTrain, yPred);

            Console.WriteLine($"K={k}: Accuracy = {accuracy:P2}");
        }

        // Step 4: Use best K value
        var bestModel = new KNearestNeighbors(k: 3);
        bestModel.Fit(XTrainScaled, yTrain);

        // Step 5: Classify new flower
        double[,] XTest = new double[,] {
            { 5.9, 3.0, 5.1, 1.8 }  // Unknown iris
        };
        double[,] XTestScaled = StandardizeFeatures(XTest);

        int[] prediction = bestModel.Predict(XTestScaled);

        string[] species = { "Setosa", "Versicolor", "Virginica" };
        Console.WriteLine($"\nPredicted species: {species[prediction[0]]}");
    }

    static double[,] StandardizeFeatures(double[,] X)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);
        double[,] XScaled = new double[n, m];

        for (int j = 0; j < m; j++)
        {
            double mean = 0, std = 0;
            for (int i = 0; i < n; i++)
                mean += X[i, j];
            mean /= n;

            for (int i = 0; i < n; i++)
                std += Math.Pow(X[i, j] - mean, 2);
            std = Math.Sqrt(std / n);

            for (int i = 0; i < n; i++)
                XScaled[i, j] = (X[i, j] - mean) / (std > 0 ? std : 1);
        }

        return XScaled;
    }
}
```

#### 4.6 KNN for Multi-class Classification

**Advantage**: KNN naturally handles multi-class problems!

**Voting Process**:
```
K = 5 neighbors:
- 3 votes for Class A
- 1 vote for Class B
- 1 vote for Class C

Winner: Class A (majority)
```

**Weighted Voting** (optional):
```
Give closer neighbors more weight:

weight = 1 / distance

Closer neighbors have more influence
```

#### 4.7 Computational Complexity

**Training Time**: O(1)
- KNN is a "lazy learner"
- No training phase!
- Just stores the data

**Prediction Time**: O(n × m)
- n = number of training samples
- m = number of features
- Must calculate distance to ALL training points

**Problem**: Slow for large datasets!

**Solutions**:
1. **KD-Trees**: Organize data for faster search
2. **Ball Trees**: Alternative data structure
3. **Approximate KNN**: Trade accuracy for speed
4. **Dimensionality Reduction**: Reduce feature count

#### 4.8 Advantages and Limitations

**Advantages**:
- ✅ **Simple**: Easy to understand and implement
- ✅ **No training**: Instant "training" time
- ✅ **Multi-class**: Naturally handles multiple classes
- ✅ **Non-linear**: Can learn complex decision boundaries
- ✅ **Adaptable**: Automatically adapts to new data

**Limitations**:
- ❌ **Slow prediction**: Must compute all distances
- ❌ **Memory intensive**: Stores all training data
- ❌ **Curse of dimensionality**: Poor performance in high dimensions
- ❌ **Sensitive to scale**: Requires feature scaling
- ❌ **Sensitive to noise**: Outliers affect predictions
- ❌ **Imbalanced data**: Biased toward majority class

#### 4.9 When to Use KNN

**Good For**:
- Small to medium datasets (< 10,000 samples)
- Low-dimensional data (< 20 features)
- Non-linear decision boundaries
- Multi-class problems
- When you need a quick baseline

**Not Good For**:
- Large datasets (slow predictions)
- High-dimensional data (curse of dimensionality)
- Real-time predictions (too slow)
- When interpretability is needed

#### 4.10 Improving KNN Performance

**1. Feature Scaling** (Critical!):
```csharp
// Always scale features before using KNN
XScaled = StandardizeFeatures(X);
```

**2. Feature Selection**:
```csharp
// Remove irrelevant features
// Keep only features with high correlation to target
```

**3. Dimensionality Reduction**:
```csharp
// Use PCA to reduce dimensions
// Helps with curse of dimensionality
```

**4. Distance Metric Selection**:
```csharp
// Try different distance metrics
// Euclidean, Manhattan, Cosine, etc.
```

**5. Optimal K Selection**:
```csharp
// Use cross-validation to find best K
for (int k = 1; k <= 20; k += 2)
{
    double score = CrossValidate(k);
    // Select K with highest score
}
```

#### 4.11 Practice Exercises

**Exercise 1: Handwritten Digit Recognition**

```csharp
// Simplified digit recognition (0 vs 1)
// Features: [pixel_intensity_1, pixel_intensity_2, ..., pixel_intensity_16]
// 4x4 pixel grid

double[,] digits = new double[,] {
    // Digit 0 (circular pattern)
    { 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 },
    { 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 },
    // Digit 1 (vertical line)
    { 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
    { 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 }
};

int[] labels = new int[] { 0, 0, 1, 1 };

// TODO:
// 1. Add more training examples
// 2. Try K = 1, 3, 5
// 3. Test with new digit patterns
// 4. Calculate accuracy
```

**Exercise 2: Customer Segmentation**

Build a customer classifier using:
- Age
- Annual income
- Purchase frequency
- Average purchase amount

Classify customers into: "Budget", "Regular", "Premium"

---

## Part II: Intermediate Level

### Chapter 5: Decision Trees

#### 5.1 The Tree Metaphor

**Decision Tree**: A flowchart-like structure for making decisions.

**Real-World Analogy**:
```
Should I go outside?
├─ Is it raining?
│  ├─ Yes → Stay inside
│  └─ No → Is it hot?
│     ├─ Yes → Go to beach
│     └─ No → Go for walk
```

**In Classification**:
```
Classify Email
├─ Spam words > 5?
│  ├─ Yes → SPAM
│  └─ No → Has links?
│     ├─ Yes → Check sender
│     │  ├─ Known → NOT SPAM
│     │  └─ Unknown → SPAM
│     └─ No → NOT SPAM
```

#### 5.2 Tree Components

**1. Root Node**: Top of tree, first decision
**2. Internal Nodes**: Decision points
**3. Branches**: Outcomes of decisions
**4. Leaf Nodes**: Final predictions

```
        [Root]
       /      \
   [Node]    [Node]
   /    \    /    \
[Leaf][Leaf][Leaf][Leaf]
```

#### 5.3 How Trees Make Decisions

**Splitting Process**:

1. **Start with all data** at root
2. **Find best feature** to split on
3. **Split data** based on feature value
4. **Repeat recursively** for each subset
5. **Stop** when reaching stopping criteria

**Stopping Criteria**:
- All samples in node have same class
- Maximum depth reached
- Minimum samples per node reached
- No information gain from splitting

#### 5.4 Information Gain and Entropy

**Entropy**: Measure of impurity/disorder in a set

```
Entropy(S) = -Σ pᵢ × log₂(pᵢ)

Where pᵢ = proportion of class i
```

**Examples**:

```
Pure node (all same class):
Class A: 100%, Class B: 0%
Entropy = -(1×log₂(1) + 0×log₂(0)) = 0
✅ Perfect purity

Mixed node (50/50 split):
Class A: 50%, Class B: 50%
Entropy = -(0.5×log₂(0.5) + 0.5×log₂(0.5)) = 1
❌ Maximum impurity

Mostly pure (80/20 split):
Class A: 80%, Class B: 20%
Entropy = -(0.8×log₂(0.8) + 0.2×log₂(0.2)) ≈ 0.72
```

**Information Gain**: Reduction in entropy after split

```
IG = Entropy(parent) - Weighted_Average_Entropy(children)
```

**Goal**: Choose split that maximizes information gain!

#### 5.5 Complete Implementation Example

**Problem**: Credit Risk Assessment

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class CreditRiskExample
{
    public static void Main()
    {
        // Features: [age, income_k, debt_ratio, credit_history_years]
        double[,] XTrain = new double[,] {
            { 25, 30,  0.8, 2 },   // High risk
            { 35, 80,  0.3, 10 },  // Low risk
            { 45, 120, 0.2, 15 },  // Low risk
            { 22, 20,  0.9, 1 },   // High risk
            { 50, 150, 0.1, 20 },  // Low risk
            { 28, 40,  0.7, 3 },   // High risk
            { 40, 100, 0.25, 12 }, // Low risk
            { 30, 50,  0.6, 5 }    // Medium risk → High
        };

        int[] yTrain = new int[] { 1, 0, 0, 1, 0, 1, 0, 1 }; // 0=Low, 1=High

        // Create decision tree with hyperparameters
        var model = new DecisionTreeClassifier(
            maxDepth: 5,           // Prevent overfitting
            minSamplesSplit: 2     // Minimum samples to split
        );

        model.Fit(XTrain, yTrain);

        // Predict new applicant
        double[,] XTest = new double[,] {
            { 32, 65, 0.5, 6 }  // New applicant
        };

        int[] prediction = model.Predict(XTest);
        Console.WriteLine($"Risk Level: {(prediction[0] == 0 ? "Low Risk" : "High Risk")}");

        // Evaluate
        int[] yPred = model.Predict(XTrain);
        double accuracy = ClassificationMetrics.Accuracy(yTrain, yPred);
        Console.WriteLine($"Training Accuracy: {accuracy:P2}");
    }
}
```

#### 5.6 Hyperparameters

**1. Max Depth**:
```
maxDepth = 1: Very simple (underfitting)
maxDepth = 5: Balanced
maxDepth = 20: Very complex (overfitting)
```

**2. Min Samples Split**:
```
minSamplesSplit = 2: Can split any node with 2+ samples
minSamplesSplit = 10: Only split nodes with 10+ samples
```

**3. Min Samples Leaf**:
```
Minimum samples required in leaf node
Prevents tiny leaves
```

#### 5.7 Advantages and Limitations

**Advantages**:
- ✅ **Interpretable**: Easy to visualize and explain
- ✅ **No scaling needed**: Works with raw features
- ✅ **Handles mixed data**: Numerical and categorical
- ✅ **Non-linear**: Captures complex patterns
- ✅ **Feature importance**: Shows which features matter

**Limitations**:
- ❌ **Overfitting**: Easily creates overly complex trees
- ❌ **Unstable**: Small data changes → different tree
- ❌ **Biased**: Favors features with many values
- ❌ **Not optimal**: Greedy algorithm (local optimum)

---

### Chapter 6: Naive Bayes Classifier

#### 6.1 Bayes' Theorem

**Foundation**: Probability theory

```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

Posterior = Likelihood × Prior / Evidence
```

**Example**:
```
P(Spam|"free money") = P("free money"|Spam) × P(Spam) / P("free money")
```

#### 6.2 The "Naive" Assumption

**Assumption**: Features are independent given the class

```
P(x₁,x₂,...,xₙ|Class) = P(x₁|Class) × P(x₂|Class) × ... × P(xₙ|Class)
```

**Why "Naive"?**
- Features are rarely truly independent
- But algorithm works well anyway!
- Simplifies computation dramatically

#### 6.3 Gaussian Naive Bayes

**Assumption**: Features follow normal distribution

```
P(xᵢ|Class) = (1/√(2πσ²)) × exp(-(xᵢ-μ)²/(2σ²))

Where:
- μ = mean of feature for class
- σ² = variance of feature for class
```

#### 6.4 Complete Implementation Example

**Problem**: Sentiment Analysis

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Classification;

public class SentimentAnalysisExample
{
    public static void Main()
    {
        // Features: [positive_words, negative_words, exclamations, word_count]
        double[,] XTrain = new double[,] {
            { 5, 0, 2, 50 },   // Positive
            { 6, 1, 3, 60 },   // Positive
            { 7, 0, 4, 55 },   // Positive
            { 0, 5, 1, 45 },   // Negative
            { 1, 6, 0, 50 },   // Negative
            { 0, 7, 2, 40 },   // Negative
            { 4, 1, 2, 52 },   // Positive
            { 2, 5, 1, 48 }    // Negative
        };

        int[] yTrain = new int[] { 1, 1, 1, 0, 0, 0, 1, 0 }; // 0=Neg, 1=Pos

        // Create and train Naive Bayes
        var model = new NaiveBayesClassifier();
        model.Fit(XTrain, yTrain);

        // Classify new review
        double[,] XTest = new double[,] {
            { 3, 2, 1, 48 }  // Mixed sentiment
        };

        int[] prediction = model.Predict(XTest);
        Console.WriteLine($"Sentiment: {(prediction[0] == 1 ? "Positive" : "Negative")}");
    }
}
```

#### 6.5 Advantages and Limitations

**Advantages**:
- ✅ **Fast**: Training and prediction are very fast
- ✅ **Scalable**: Works well with high-dimensional data
- ✅ **Small data**: Performs well with limited training data
- ✅ **Probabilistic**: Provides probability estimates
- ✅ **Multi-class**: Naturally handles multiple classes

**Limitations**:
- ❌ **Independence assumption**: Rarely holds in practice
- ❌ **Zero frequency**: Problems with unseen feature values
- ❌ **Continuous features**: Assumes normal distribution

**When to Use**:
- Text classification (spam, sentiment)
- Document categorization
- Real-time prediction (fast)
- High-dimensional data
- Baseline model

---

### Chapter 7: Evaluation Metrics Deep Dive

#### 7.1 Why Accuracy Isn't Enough

**The Accuracy Trap**:

```
Dataset: 95% Normal emails, 5% Spam

Model that always predicts "Normal":
Accuracy = 95% ✅

But catches 0% of spam! ❌
Completely useless!
```

**Lesson**: Accuracy can be misleading, especially with imbalanced data.

#### 7.2 The Four Outcomes

**Binary Classification Outcomes**:

```
                Predicted
              Positive  Negative
Actual  Pos      TP        FN
        Neg      FP        TN
```

**Definitions**:
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I Error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II Error)

**Example - Medical Test**:
```
TP: Test says "disease" and patient has disease ✅
TN: Test says "healthy" and patient is healthy ✅
FP: Test says "disease" but patient is healthy ❌ (False alarm)
FN: Test says "healthy" but patient has disease ❌ (Missed diagnosis)
```

#### 7.3 Precision

**Definition**: Of all positive predictions, how many were correct?

```
Precision = TP / (TP + FP)
```

**Interpretation**:
- High precision = Few false alarms
- Low precision = Many false alarms

**When to Prioritize**:
- Spam filter (don't want to block legitimate emails)
- Product recommendations (don't want to annoy users)
- Medical procedures (avoid unnecessary treatments)

**Example**:
```
Email classifier:
TP = 40 (correctly identified spam)
FP = 10 (legitimate emails marked as spam)

Precision = 40 / (40 + 10) = 0.80 = 80%

Meaning: 80% of emails marked as spam are actually spam
```

#### 7.4 Recall (Sensitivity)

**Definition**: Of all actual positives, how many did we catch?

```
Recall = TP / (TP + FN)
```

**Interpretation**:
- High recall = Catch most positives
- Low recall = Miss many positives

**When to Prioritize**:
- Disease diagnosis (don't want to miss sick patients)
- Fraud detection (catch as many frauds as possible)
- Security systems (detect all threats)

**Example**:
```
Disease screening:
TP = 40 (correctly identified sick patients)
FN = 10 (missed sick patients)

Recall = 40 / (40 + 10) = 0.80 = 80%

Meaning: We catch 80% of sick patients
```

#### 7.5 The Precision-Recall Tradeoff

**The Dilemma**: Can't maximize both simultaneously!

```
High Precision, Low Recall:
- Very conservative
- Only predict positive when very confident
- Miss many positives

High Recall, Low Precision:
- Very aggressive
- Predict positive liberally
- Many false alarms
```

**Visual Example**:
```
Threshold = 0.9 (High Precision):
Only predict spam if 90%+ confident
→ Few false positives, but miss some spam

Threshold = 0.3 (High Recall):
Predict spam if 30%+ confident
→ Catch most spam, but many false positives
```

#### 7.6 F1-Score

**Definition**: Harmonic mean of precision and recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why Harmonic Mean?**
- Penalizes extreme values
- Both precision and recall must be high for good F1

**Example**:
```
Scenario 1:
Precision = 0.9, Recall = 0.9
F1 = 2 × (0.9 × 0.9) / (0.9 + 0.9) = 0.90 ✅

Scenario 2:
Precision = 0.9, Recall = 0.1
F1 = 2 × (0.9 × 0.1) / (0.9 + 0.1) = 0.18 ❌

Scenario 3:
Precision = 0.5, Recall = 0.5
F1 = 2 × (0.5 × 0.5) / (0.5 + 0.5) = 0.50
```

**When to Use**: When you need balance between precision and recall

#### 7.7 Specificity

**Definition**: Of all actual negatives, how many did we correctly identify?

```
Specificity = TN / (TN + FP)
```

**Also called**: True Negative Rate

**Use Case**: Medical tests (correctly identifying healthy patients)

#### 7.8 Code Example - All Metrics

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class MetricsExample
{
    public static void Main()
    {
        // Predictions vs actual labels
        int[] yTrue = new int[] { 1, 0, 1, 1, 0, 1, 0, 0, 1, 1 };
        int[] yPred = new int[] { 1, 0, 1, 0, 0, 1, 1, 0, 1, 1 };

        // Calculate all metrics
        double accuracy = ClassificationMetrics.Accuracy(yTrue, yPred);
        double precision = ClassificationMetrics.Precision(yTrue, yPred, positiveClass: 1);
        double recall = ClassificationMetrics.Recall(yTrue, yPred, positiveClass: 1);
        double f1 = ClassificationMetrics.F1Score(yTrue, yPred, positiveClass: 1);
        double specificity = ClassificationMetrics.Specificity(yTrue, yPred, positiveClass: 1);

        Console.WriteLine("Classification Metrics:");
        Console.WriteLine($"Accuracy:    {accuracy:P2}");
        Console.WriteLine($"Precision:   {precision:P2}");
        Console.WriteLine($"Recall:      {recall:P2}");
        Console.WriteLine($"F1-Score:    {f1:P2}");
        Console.WriteLine($"Specificity: {specificity:P2}");

        // Interpret results
        Console.WriteLine("\nInterpretation:");
        if (precision > 0.8)
            Console.WriteLine("✅ High precision - Few false alarms");
        if (recall > 0.8)
            Console.WriteLine("✅ High recall - Catching most positives");
        if (f1 > 0.8)
            Console.WriteLine("✅ Good balance between precision and recall");
    }
}
```

#### 7.9 Metric Selection Guide

| Scenario | Primary Metric | Reason |
|----------|---------------|--------|
| **Balanced classes** | Accuracy | Simple and intuitive |
| **Imbalanced classes** | F1-Score | Balances precision and recall |
| **Spam detection** | Precision | Avoid blocking legitimate emails |
| **Disease screening** | Recall | Don't miss sick patients |
| **Fraud detection** | Recall | Catch all fraudulent transactions |
| **Recommendation systems** | Precision@K | Quality of top K recommendations |
| **Search engines** | Recall | Find all relevant documents |

---

### Chapter 8: The Confusion Matrix

#### 8.1 Understanding the Matrix

**Structure**:
```
                    Predicted
                 Class 0  Class 1  Class 2
Actual  Class 0    50       5        2
        Class 1     3      45        4
        Class 2     1       2       48
```

**Reading the Matrix**:
- **Diagonal**: Correct predictions
- **Off-diagonal**: Errors
- **Row**: Actual class distribution
- **Column**: Predicted class distribution

#### 8.2 Binary Confusion Matrix

```
                Predicted
              Negative  Positive
Actual  Neg      TN        FP
        Pos      FN        TP
```

**Example - Email Classification**:
```
                Predicted
              Normal    Spam
Actual  Normal  850      50      (900 normal emails)
        Spam     10      90      (100 spam emails)

TP = 90  (correctly identified spam)
TN = 850 (correctly identified normal)
FP = 50  (normal marked as spam)
FN = 10  (spam marked as normal)
```

#### 8.3 Code Example

```csharp
using ArtificialIntelligence.MachineLearning.Supervised.Evaluation;

public class ConfusionMatrixExample
{
    public static void Main()
    {
        // Example predictions
        int[] yTrue = new int[] { 0, 1, 1, 0, 1, 0, 1, 1, 0, 0 };
        int[] yPred = new int[] { 0, 1, 0, 0, 1, 1, 1, 1, 0, 0 };

        // Create confusion matrix
        var cm = new ConfusionMatrix(yTrue, yPred);

        // Display matrix
        Console.WriteLine("Confusion Matrix:");
        Console.WriteLine(cm.ToString());

        // Get individual values
        int tp = cm.GetTruePositives(1);
        int tn = cm.GetTrueNegatives(1);
        int fp = cm.GetFalsePositives(1);
        int fn = cm.GetFalseNegatives(1);

        Console.WriteLine($"\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}");

        // Calculate metrics from confusion matrix
        double precision = (double)tp / (tp + fp);
        double recall = (double)tp / (tp + fn);
        double accuracy = (double)(tp + tn) / (tp + tn + fp + fn);

        Console.WriteLine($"\nPrecision: {precision:P2}");
        Console.WriteLine($"Recall:    {recall:P2}");
        Console.WriteLine($"Accuracy:  {accuracy:P2}");
    }
}
```

#### 8.4 Multi-class Confusion Matrix

**Example - Iris Classification**:
```
                    Predicted
                 Setosa  Versi  Virgin
Actual  Setosa     50      0      0
        Versi       0     47      3
        Virgin      0      2     48

Perfect classification of Setosa
Some confusion between Versicolor and Virginica
```

#### 8.5 Analyzing Confusion Patterns

**Common Patterns**:

**1. Perfect Classification**:
```
        Pred 0  Pred 1
Act 0    100      0
Act 1      0    100

All predictions on diagonal ✅
```

**2. Systematic Bias**:
```
        Pred 0  Pred 1
Act 0     90     10
Act 1     40     60

Model biased toward Class 0 ❌
```

**3. Random Guessing**:
```
        Pred 0  Pred 1
Act 0     50     50
Act 1     50     50

No pattern, like flipping a coin ❌
```

#### 8.6 Using Confusion Matrix for Improvement

**Identify Weaknesses**:
```
                Predicted
              Cat   Dog   Bird
Actual  Cat    45    3     2
        Dog     5   40     5
        Bird    1    4    45

Issue: Dogs often confused with Birds
Solution: Add more distinguishing features
```

**Class-Specific Performance**:
```csharp
// Calculate per-class metrics
for (int classLabel = 0; classLabel < numClasses; classLabel++)
{
    int tp = cm.GetTruePositives(classLabel);
    int fp = cm.GetFalsePositives(classLabel);
    int fn = cm.GetFalseNegatives(classLabel);

    double precision = (double)tp / (tp + fp);
    double recall = (double)tp / (tp + fn);

    Console.WriteLine($"Class {classLabel}: Precision={precision:P2}, Recall={recall:P2}");
}
```

---

## Part III: Advanced Level

### Chapter 9: Handling Imbalanced Datasets

#### 9.1 The Imbalance Problem

**Definition**: One class has significantly more samples than others

**Common Examples**:
```
Fraud Detection:    99.5% legitimate, 0.5% fraud
Disease Diagnosis:  95% healthy, 5% sick
Spam Detection:     90% normal, 10% spam
Defect Detection:   99% good products, 1% defective
```

**Why It's a Problem**:
```
Naive model: Always predict majority class
Accuracy: 99.5% ✅
But useless for detecting fraud! ❌
```

#### 9.2 Evaluation Metrics for Imbalanced Data

**Don't Use**: Accuracy

**Do Use**:
- **F1-Score**: Balances precision and recall
- **Precision-Recall Curve**: Shows tradeoff
- **ROC-AUC**: Area under ROC curve
- **Balanced Accuracy**: Average of per-class accuracies

**Example**:
```csharp
// For imbalanced data, use F1-Score
double f1 = ClassificationMetrics.F1Score(yTrue, yPred, positiveClass: 1);

// Or balanced accuracy
double balancedAcc = (recall_class0 + recall_class1) / 2;
```

#### 9.3 Resampling Techniques

**1. Oversampling (Increase Minority Class)**:
```csharp
public static (double[,], int[]) Oversample(double[,] X, int[] y, int minorityClass)
{
    // Count samples per class
    int majorityCount = y.Count(label => label != minorityClass);
    int minorityCount = y.Count(label => label == minorityClass);

    // Duplicate minority samples to match majority
    int duplicates = majorityCount / minorityCount;

    // Create new dataset with duplicated minority samples
    // ... implementation ...

    return (XResampled, yResampled);
}
```

**2. Undersampling (Decrease Majority Class)**:
```csharp
public static (double[,], int[]) Undersample(double[,] X, int[] y, int majorityClass)
{
    // Randomly remove majority class samples
    // Until balanced with minority class
    // ... implementation ...

    return (XResampled, yResampled);
}
```

**Pros and Cons**:

| Method | Pros | Cons |
|--------|------|------|
| **Oversampling** | No data loss | May overfit, larger dataset |
| **Undersampling** | Faster training | Loses information |

#### 9.4 Class Weights

**Concept**: Penalize misclassifying minority class more heavily

```
Loss = Σ weight[class] × error

Where weight[minority] > weight[majority]
```

**Example**:
```csharp
// Fraud detection: 99% legitimate, 1% fraud
// Give fraud class 99x more weight

double[] classWeights = new double[] { 1.0, 99.0 };

// In loss function:
loss = classWeights[actualClass] * error;
```

#### 9.5 Threshold Adjustment

**Default Threshold**: 0.5
```
If P(positive) ≥ 0.5 → Predict positive
```

**Adjusted Threshold**: Lower for minority class
```
If P(fraud) ≥ 0.2 → Predict fraud

Increases recall (catch more fraud)
Decreases precision (more false alarms)
```

**Code Example**:
```csharp
public static int[] PredictWithThreshold(double[] probabilities, double threshold)
{
    return probabilities.Select(p => p >= threshold ? 1 : 0).ToArray();
}

// Try different thresholds
double[] thresholds = { 0.3, 0.5, 0.7 };
foreach (var threshold in thresholds)
{
    int[] predictions = PredictWithThreshold(probabilities, threshold);
    double recall = ClassificationMetrics.Recall(yTrue, predictions, 1);
    Console.WriteLine($"Threshold {threshold}: Recall = {recall:P2}");
}
```

#### 9.6 Ensemble Methods

**SMOTE (Synthetic Minority Over-sampling Technique)**:
- Generate synthetic samples
- Interpolate between minority class samples
- More sophisticated than simple duplication

**Balanced Random Forest**:
- Each tree trained on balanced subset
- Combines predictions from all trees

---

### Chapter 10: Multi-class Classification

#### 10.1 Multi-class vs Binary

**Binary**: 2 classes (Spam/Not Spam)
**Multi-class**: 3+ classes (Cat/Dog/Bird)

**Key Difference**: Decision boundary becomes more complex

#### 10.2 One-vs-Rest (OvR)

**Strategy**: Train N binary classifiers (one per class)

```
Class 0 vs Rest: Is it Class 0? Yes/No
Class 1 vs Rest: Is it Class 1? Yes/No
Class 2 vs Rest: Is it Class 2? Yes/No

Prediction: Class with highest confidence
```

**Example**:
```csharp
public class OneVsRestClassifier
{
    private List<LogisticRegression> classifiers;

    public void Fit(double[,] X, int[] y)
    {
        int numClasses = y.Max() + 1;
        classifiers = new List<LogisticRegression>();

        for (int c = 0; c < numClasses; c++)
        {
            // Create binary labels: class c vs rest
            int[] binaryY = y.Select(label => label == c ? 1 : 0).ToArray();

            var clf = new LogisticRegression();
            clf.Fit(X, binaryY);
            classifiers.Add(clf);
        }
    }

    public int[] Predict(double[,] X)
    {
        int n = X.GetLength(0);
        int[] predictions = new int[n];

        for (int i = 0; i < n; i++)
        {
            double maxProb = -1;
            int bestClass = 0;

            for (int c = 0; c < classifiers.Count; c++)
            {
                double[,] sample = GetRow(X, i);
                double[] prob = classifiers[c].PredictProba(sample);

                if (prob[0] > maxProb)
                {
                    maxProb = prob[0];
                    bestClass = c;
                }
            }

            predictions[i] = bestClass;
        }

        return predictions;
    }
}
```

#### 10.3 One-vs-One (OvO)

**Strategy**: Train N×(N-1)/2 binary classifiers (one for each pair)

```
For 3 classes:
- Class 0 vs Class 1
- Class 0 vs Class 2
- Class 1 vs Class 2

Prediction: Majority vote
```

**Comparison**:

| Method | Classifiers | Training Time | Prediction Time |
|--------|-------------|---------------|-----------------|
| **OvR** | N | Faster | Faster |
| **OvO** | N×(N-1)/2 | Slower | Slower |

#### 10.4 Native Multi-class Algorithms

**Algorithms that naturally handle multi-class**:
- Decision Trees ✅
- Naive Bayes ✅
- KNN ✅

**Algorithms that need adaptation**:
- Logistic Regression (use OvR or Softmax)
- SVM (use OvR or OvO)

#### 10.5 Softmax Regression

**Extension of Logistic Regression for multi-class**:

```
P(class=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)

Properties:
- Outputs sum to 1
- Each output is a probability
- Generalizes sigmoid to multiple classes
```

---

### Chapter 11: Model Selection and Comparison

#### 11.1 Algorithm Comparison Framework

```csharp
public class ModelComparison
{
    public static void CompareModels(double[,] X, int[] y)
    {
        // Split data
        var (XTrain, yTrain, XTest, yTest) = TrainTestSplit(X, y, 0.2);

        // Define models to compare
        var models = new Dictionary<string, dynamic>
        {
            { "Logistic Regression", new LogisticRegression() },
            { "KNN (k=3)", new KNearestNeighbors(k: 3) },
            { "KNN (k=5)", new KNearestNeighbors(k: 5) },
            { "Decision Tree", new DecisionTreeClassifier(maxDepth: 5) },
            { "Naive Bayes", new NaiveBayesClassifier() }
        };

        Console.WriteLine("Model Comparison Results:\n");
        Console.WriteLine($"{"Model",-25} {"Accuracy",-12} {"Precision",-12} {"Recall",-12} {"F1-Score",-12}");
        Console.WriteLine(new string('-', 73));

        foreach (var (name, model) in models)
        {
            // Train
            model.Fit(XTrain, yTrain);

            // Predict
            int[] yPred = model.Predict(XTest);

            // Evaluate
            double accuracy = ClassificationMetrics.Accuracy(yTest, yPred);
            double precision = ClassificationMetrics.Precision(yTest, yPred, 1);
            double recall = ClassificationMetrics.Recall(yTest, yPred, 1);
            double f1 = ClassificationMetrics.F1Score(yTest, yPred, 1);

            Console.WriteLine($"{name,-25} {accuracy,-12:P2} {precision,-12:P2} {recall,-12:P2} {f1,-12:P2}");
        }
    }
}
```

#### 11.2 Cross-Validation

**K-Fold Cross-Validation**:

```csharp
public static double CrossValidate(dynamic model, double[,] X, int[] y, int k = 5)
{
    int n = y.Length;
    int foldSize = n / k;
    double totalScore = 0;

    for (int fold = 0; fold < k; fold++)
    {
        // Create train/test split for this fold
        var (XTrain, yTrain, XTest, yTest) = GetFold(X, y, fold, foldSize);

        // Train and evaluate
        model.Fit(XTrain, yTrain);
        int[] yPred = model.Predict(XTest);
        double score = ClassificationMetrics.Accuracy(yTest, yPred);

        totalScore += score;
        Console.WriteLine($"Fold {fold + 1}: {score:P2}");
    }

    double avgScore = totalScore / k;
    Console.WriteLine($"Average: {avgScore:P2}");
    return avgScore;
}
```

#### 11.3 Hyperparameter Tuning

**Grid Search**:

```csharp
public static void GridSearchKNN(double[,] X, int[] y)
{
    int[] kValues = { 1, 3, 5, 7, 9, 11, 15, 21 };
    double bestScore = 0;
    int bestK = 0;

    Console.WriteLine("Grid Search for KNN:\n");

    foreach (int k in kValues)
    {
        var model = new KNearestNeighbors(k: k);
        double score = CrossValidate(model, X, y, k: 5);

        Console.WriteLine($"K={k}: Score={score:P2}\n");

        if (score > bestScore)
        {
            bestScore = score;
            bestK = k;
        }
    }

    Console.WriteLine($"Best K: {bestK} with score {bestScore:P2}");
}
```

#### 11.4 Algorithm Selection Guide

**Decision Tree**:

```
Start
  ↓
Need probability estimates?
  ├─ Yes → Logistic Regression or Naive Bayes
  └─ No → Continue
       ↓
       Need interpretability?
       ├─ Yes → Decision Tree
       └─ No → Continue
            ↓
            Small dataset?
            ├─ Yes → KNN
            └─ No → Continue
                 ↓
                 High-dimensional?
                 ├─ Yes → Naive Bayes
                 └─ No → Try all, compare
```

**Quick Reference**:

| Scenario | Recommended Algorithm |
|----------|----------------------|
| **Small data, simple** | KNN |
| **Need probabilities** | Logistic Regression, Naive Bayes |
| **Need interpretability** | Decision Tree |
| **Text classification** | Naive Bayes |
| **Non-linear patterns** | Decision Tree, KNN |
| **Large dataset** | Logistic Regression, Naive Bayes |
| **Real-time prediction** | Naive Bayes (fastest) |

---

### Chapter 12: Production Deployment

#### 12.1 Model Serialization

**Save trained model**:

```csharp
using System.Text.Json;

public class ModelSerializer
{
    public static void SaveModel(LogisticRegression model, string filepath)
    {
        // Serialize model parameters
        var modelData = new
        {
            Weights = model.GetWeights(),
            Bias = model.GetBias(),
            ModelType = "LogisticRegression"
        };

        string json = JsonSerializer.Serialize(modelData);
        File.WriteAllText(filepath, json);
    }

    public static LogisticRegression LoadModel(string filepath)
    {
        string json = File.ReadAllText(filepath);
        var modelData = JsonSerializer.Deserialize<dynamic>(json);

        var model = new LogisticRegression();
        model.SetWeights(modelData.Weights);
        model.SetBias(modelData.Bias);

        return model;
    }
}
```

#### 12.2 REST API Deployment

**Create prediction endpoint**:

```csharp
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("api/[controller]")]
public class PredictionController : ControllerBase
{
    private static LogisticRegression _model;

    static PredictionController()
    {
        // Load model on startup
        _model = ModelSerializer.LoadModel("model.json");
    }

    [HttpPost("predict")]
    public IActionResult Predict([FromBody] PredictionRequest request)
    {
        try
        {
            // Convert request to feature array
            double[,] features = new double[,] {
                {
                    request.Feature1,
                    request.Feature2,
                    request.Feature3
                }
            };

            // Make prediction
            int[] prediction = _model.Predict(features);
            double[] probability = _model.PredictProba(features);

            return Ok(new
            {
                prediction = prediction[0],
                probability = probability[0],
                timestamp = DateTime.UtcNow
            });
        }
        catch (Exception ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }
}

public class PredictionRequest
{
    public double Feature1 { get; set; }
    public double Feature2 { get; set; }
    public double Feature3 { get; set; }
}
```

#### 12.3 Model Monitoring

**Track model performance in production**:

```csharp
public class ModelMonitor
{
    private List<double> _accuracyHistory = new List<double>();

    public void LogPrediction(int prediction, int actual)
    {
        // Store prediction and actual label
        // Calculate rolling accuracy

        bool correct = prediction == actual;
        _accuracyHistory.Add(correct ? 1.0 : 0.0);

        // Keep last 1000 predictions
        if (_accuracyHistory.Count > 1000)
            _accuracyHistory.RemoveAt(0);

        // Check if performance degraded
        double recentAccuracy = _accuracyHistory.Average();
        if (recentAccuracy < 0.7)
        {
            Console.WriteLine("⚠️ WARNING: Model performance degraded!");
            Console.WriteLine($"Recent accuracy: {recentAccuracy:P2}");
            // Trigger retraining or alert
        }
    }
}
```

#### 12.4 A/B Testing

**Compare model versions**:

```csharp
public class ABTestingController
{
    private LogisticRegression _modelA;
    private LogisticRegression _modelB;
    private Random _random = new Random();

    [HttpPost("predict")]
    public IActionResult Predict([FromBody] PredictionRequest request)
    {
        // Randomly assign to model A or B (50/50 split)
        bool useModelA = _random.NextDouble() < 0.5;

        var model = useModelA ? _modelA : _modelB;
        string modelVersion = useModelA ? "A" : "B";

        // Make prediction
        int[] prediction = model.Predict(features);

        // Log for analysis
        LogABTest(modelVersion, prediction[0], request);

        return Ok(new
        {
            prediction = prediction[0],
            modelVersion = modelVersion
        });
    }
}
```

#### 12.5 Best Practices

**1. Version Control**:
- Track model versions
- Store training data snapshots
- Document hyperparameters

**2. Input Validation**:
```csharp
public bool ValidateInput(double[] features)
{
    // Check for null
    if (features == null) return false;

    // Check dimensions
    if (features.Length != expectedFeatureCount) return false;

    // Check for invalid values
    if (features.Any(f => double.IsNaN(f) || double.IsInfinity(f)))
        return false;

    // Check ranges
    for (int i = 0; i < features.Length; i++)
    {
        if (features[i] < minValues[i] || features[i] > maxValues[i])
            return false;
    }

    return true;
}
```

**3. Error Handling**:
```csharp
try
{
    int[] prediction = model.Predict(features);
    return Ok(prediction);
}
catch (ArgumentException ex)
{
    return BadRequest("Invalid input features");
}
catch (Exception ex)
{
    LogError(ex);
    return StatusCode(500, "Prediction failed");
}
```

**4. Performance Optimization**:
- Cache frequently used predictions
- Batch predictions when possible
- Use async operations for I/O

**5. Monitoring and Alerts**:
- Track prediction latency
- Monitor error rates
- Alert on performance degradation
- Log prediction distributions

---

## Conclusion

Congratulations! You've completed the comprehensive guide to Classification in Machine Learning.

### Key Takeaways

**1. Algorithm Selection**:
- Start simple (Logistic Regression, KNN)
- Consider data characteristics (size, dimensions, balance)
- Compare multiple algorithms
- Choose based on requirements (speed, interpretability, accuracy)

**2. Evaluation**:
- Don't rely solely on accuracy
- Use appropriate metrics (Precision, Recall, F1)
- Always use confusion matrix
- Consider business context

**3. Real-World Challenges**:
- Handle imbalanced data carefully
- Scale features appropriately
- Validate on unseen data
- Monitor production performance

**4. Best Practices**:
- Cross-validation for robust evaluation
- Hyperparameter tuning for optimization
- Feature engineering for better performance
- Regular model retraining

### Algorithm Comparison Summary

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Logistic Regression** | Fast, interpretable, probabilistic | Linear only | Binary classification, baseline |
| **KNN** | Simple, non-linear, no training | Slow prediction, memory intensive | Small datasets, multi-class |
| **Decision Tree** | Interpretable, handles mixed data | Overfits easily, unstable | Need interpretability |
| **Naive Bayes** | Very fast, works with small data | Independence assumption | Text classification, high-dim |

### Next Steps

**Continue Learning**:
1. **Ensemble Methods**: Random Forests, Gradient Boosting
2. **Deep Learning**: Neural Networks for classification
3. **Advanced Techniques**: Feature engineering, AutoML
4. **Specialized Domains**: Computer Vision, NLP

**Practice Projects**:
1. Build a spam filter from scratch
2. Create a sentiment analysis system
3. Develop a fraud detection model
4. Implement a medical diagnosis classifier

**Resources for .NET/C# Machine Learning**:
- ML.NET Documentation (Microsoft's ML framework)
- Accord.NET Framework (comprehensive ML library)
- Math.NET Numerics (numerical computing)
- "Programming ML.NET" by Matt R. Cole
- Microsoft Learn: Machine Learning with .NET

---

## Alternative Technology Stack

**Note**: While this tutorial uses the **.NET C# technology stack** for all implementations, the concepts, algorithms, and techniques covered are language-agnostic and can be implemented in other programming languages.

**Python Alternative**: Python is another popular choice for machine learning with excellent libraries:
- **scikit-learn**: Comprehensive classification algorithms
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **TensorFlow/PyTorch**: Deep learning

The choice between C# and Python depends on your project requirements:
- **Choose C#/.NET** for: Enterprise applications, Windows integration, strong typing, performance-critical applications, existing .NET ecosystem
- **Choose Python** for: Data science workflows, rapid prototyping, extensive ML ecosystem, research, Jupyter notebooks

Both ecosystems are mature and capable of handling production machine learning workloads.

---

**Happy Classifying!** 🎯

*Last Updated: 2026-01-29*

