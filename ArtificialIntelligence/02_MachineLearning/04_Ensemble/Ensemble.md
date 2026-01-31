# Ensemble Learning: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: What is Ensemble Learning?](#chapter-1-what-is-ensemble-learning)
  - [Chapter 2: Voting Methods](#chapter-2-voting-methods)
  - [Chapter 3: Introduction to Bagging](#chapter-3-introduction-to-bagging)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: Random Forests](#chapter-4-random-forests)
  - [Chapter 5: Boosting Fundamentals](#chapter-5-boosting-fundamentals)
  - [Chapter 6: AdaBoost](#chapter-6-adaboost)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: Gradient Boosting](#chapter-7-gradient-boosting)
  - [Chapter 8: XGBoost and LightGBM](#chapter-8-xgboost-and-lightgbm)
  - [Chapter 9: Stacking and Blending](#chapter-9-stacking-and-blending)

---

## Introduction

**Ensemble Learning** combines multiple models to produce better predictions than any single model alone.

### Why Ensembles Work

| Concept | Explanation |
|---------|-------------|
| **Wisdom of Crowds** | Aggregating diverse opinions often beats individuals |
| **Error Reduction** | Different models make different errors |
| **Bias-Variance** | Can reduce both bias and variance |

### Ensemble Methods Overview

| Method | Strategy | Example |
|--------|----------|---------|
| **Bagging** | Parallel training, averaging | Random Forest |
| **Boosting** | Sequential training, error correction | XGBoost |
| **Stacking** | Meta-learning from base models | Stacked Generalization |

---

## Part I: Beginner Level

### Chapter 1: What is Ensemble Learning?

#### 1.1 The Concept

**Ensemble Learning**: Training multiple models and combining their predictions.

```
Model 1 ──┐
Model 2 ──┼──► Combine ──► Final Prediction
Model 3 ──┘
```

#### 1.2 Types of Combination

**For Classification**:
- **Hard Voting**: Majority vote
- **Soft Voting**: Average probabilities

**For Regression**:
- **Averaging**: Mean of predictions
- **Weighted Averaging**: Weighted mean

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Create base models
model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC(probability=True)

# Voting ensemble
voting_clf = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2), ('svc', model3)],
    voting='soft'  # or 'hard'
)

voting_clf.fit(X_train, y_train)
print(f"Accuracy: {voting_clf.score(X_test, y_test):.4f}")
```

---

### Chapter 2: Voting Methods

#### 2.1 Hard Voting

Each model votes, majority wins:

```python
import numpy as np

def hard_voting(predictions):
    """
    predictions: array of shape (n_models, n_samples)
    """
    # Count votes for each class
    return np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=predictions
    )

# Example
pred1 = [0, 1, 1, 0, 1]
pred2 = [0, 0, 1, 1, 1]
pred3 = [1, 1, 1, 0, 0]

predictions = np.array([pred1, pred2, pred3])
final_pred = hard_voting(predictions)
print(f"Final predictions: {final_pred}")  # [0, 1, 1, 0, 1]
```

#### 2.2 Soft Voting

Average predicted probabilities:

```python
def soft_voting(probabilities):
    """
    probabilities: array of shape (n_models, n_samples, n_classes)
    """
    avg_proba = np.mean(probabilities, axis=0)
    return np.argmax(avg_proba, axis=1)

# Example
proba1 = [[0.9, 0.1], [0.4, 0.6], [0.3, 0.7]]
proba2 = [[0.8, 0.2], [0.6, 0.4], [0.2, 0.8]]
proba3 = [[0.7, 0.3], [0.5, 0.5], [0.4, 0.6]]

probabilities = np.array([proba1, proba2, proba3])
final_pred = soft_voting(probabilities)
print(f"Final predictions: {final_pred}")
```

---

### Chapter 3: Introduction to Bagging

#### 3.1 Bootstrap Aggregating (Bagging)

**Idea**: Train models on different bootstrap samples, then average.

**Steps**:
1. Create N bootstrap samples (random sampling with replacement)
2. Train a model on each sample
3. Aggregate predictions (vote/average)

```
Original Data
     │
     ├──► Bootstrap 1 ──► Model 1 ──┐
     ├──► Bootstrap 2 ──► Model 2 ──┼──► Average
     └──► Bootstrap 3 ──► Model 3 ──┘
```

#### 3.2 Implementation

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging with decision trees
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,      # 80% of data per bootstrap
    bootstrap=True,        # With replacement
    n_jobs=-1,            # Use all cores
    random_state=42
)

bagging_clf.fit(X_train, y_train)
print(f"Accuracy: {bagging_clf.score(X_test, y_test):.4f}")
```

#### 3.3 Out-of-Bag (OOB) Evaluation

About 37% of samples are not used in each bootstrap (OOB samples).

```python
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    oob_score=True,  # Enable OOB evaluation
    random_state=42
)

bagging_clf.fit(X_train, y_train)
print(f"OOB Score: {bagging_clf.oob_score_:.4f}")
```

---

## Part II: Intermediate Level

### Chapter 4: Random Forests

#### 4.1 What is Random Forest?

**Random Forest** = Bagging + Random Feature Selection

**Key Innovation**: At each split, consider only a random subset of features.

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth
    max_features='sqrt',   # Features per split
    min_samples_split=2,   # Minimum samples to split
    min_samples_leaf=1,    # Minimum samples in leaf
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

rf_clf.fit(X_train, y_train)
print(f"Accuracy: {rf_clf.score(X_test, y_test):.4f}")
print(f"OOB Score: {rf_clf.oob_score_:.4f}")
```

#### 4.2 Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importance
importance = rf_clf.feature_importances_
feature_names = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]

# Sort and plot
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance[indices])
plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
```

#### 4.3 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

### Chapter 5: Boosting Fundamentals

#### 5.1 Boosting Concept

**Boosting**: Train models sequentially, each focusing on previous errors.

```
Data ──► Model 1 ──► Errors ──► Model 2 ──► Errors ──► Model 3 ──► ...
                 (weighted)            (weighted)
```

**Key Differences from Bagging**:
| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Sample weights | Uniform | Adjusted |
| Error focus | None | High on errors |
| Overfitting risk | Lower | Higher |

#### 5.2 Weak Learners

Boosting uses **weak learners** (slightly better than random):
- Decision stumps (depth=1 trees)
- Shallow trees

---

### Chapter 6: AdaBoost

#### 6.1 Algorithm

**Adaptive Boosting**:
1. Initialize equal weights for all samples
2. Train weak learner
3. Increase weights on misclassified samples
4. Repeat
5. Combine with weighted vote

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Stump
    n_estimators=100,
    learning_rate=0.5,  # Shrinkage
    algorithm='SAMME.R',
    random_state=42
)

ada_clf.fit(X_train, y_train)
print(f"Accuracy: {ada_clf.score(X_test, y_test):.4f}")
```

#### 6.2 Mathematics

**Weight Update**:
```
wᵢ⁽ᵗ⁺¹⁾ = wᵢ⁽ᵗ⁾ × exp(αₜ × I(yᵢ ≠ hₜ(xᵢ)))

Where:
- αₜ = 0.5 × log((1 - εₜ) / εₜ)  # Learner weight
- εₜ = weighted error rate
```

---

## Part III: Advanced Level

### Chapter 7: Gradient Boosting

#### 7.1 Concept

**Gradient Boosting**: Boosting using gradient descent on loss function.

Each new model fits the **negative gradient** (pseudo-residuals) of the loss.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    subsample=0.8,  # Stochastic GB
    random_state=42
)

gb_clf.fit(X_train, y_train)
print(f"Accuracy: {gb_clf.score(X_test, y_test):.4f}")
```

#### 7.2 Algorithm Steps

```
1. Initialize F₀(x) = argmin_γ Σ L(yᵢ, γ)
2. For m = 1 to M:
   a. Compute pseudo-residuals: rᵢₘ = -∂L/∂F(xᵢ)
   b. Fit tree hₘ to residuals
   c. Update: Fₘ(x) = Fₘ₋₁(x) + ν × hₘ(x)
```

---

### Chapter 8: XGBoost and LightGBM

#### 8.1 XGBoost

**Extreme Gradient Boosting** - Optimized implementation with:
- Regularization (L1, L2)
- Parallel processing
- Tree pruning
- Built-in cross-validation

```python
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,      # L1 regularization
    reg_lambda=1,     # L2 regularization
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_clf.fit(X_train, y_train)
print(f"Accuracy: {xgb_clf.score(X_test, y_test):.4f}")
```

#### 8.2 LightGBM

**Light Gradient Boosting Machine** - Faster with:
- Leaf-wise growth (vs level-wise)
- Gradient-based one-side sampling
- Exclusive feature bundling

```python
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=-1,       # No limit
    num_leaves=31,      # Max leaves per tree
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_clf.fit(X_train, y_train)
print(f"Accuracy: {lgb_clf.score(X_test, y_test):.4f}")
```

#### 8.3 Comparison

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Tree growth | Level-wise | Leaf-wise | Symmetric |
| Speed | Fast | Faster | Fast |
| Categorical | Manual | Built-in | Best |
| Memory | More | Less | Medium |

---

### Chapter 9: Stacking and Blending

#### 9.1 Stacking

**Stacked Generalization**: Use predictions of base models as features for a meta-model.

```
Level 0 (Base Models):
X ──► Model 1 ──► Pred 1 ──┐
X ──► Model 2 ──► Pred 2 ──┼──► [Pred 1, Pred 2, Pred 3] ──► Meta Model ──► Final
X ──► Model 3 ──► Pred 3 ──┘
```

```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5,  # Cross-validation for meta-features
    stack_method='auto'
)

stacking_clf.fit(X_train, y_train)
print(f"Accuracy: {stacking_clf.score(X_test, y_test):.4f}")
```

#### 9.2 Blending

Simpler than stacking - uses holdout set instead of CV:

```python
def blending(X_train, y_train, X_test, base_models, meta_model, holdout_ratio=0.2):
    # Split training data
    split = int(len(X_train) * (1 - holdout_ratio))
    X_train_base, X_blend = X_train[:split], X_train[split:]
    y_train_base, y_blend = y_train[:split], y_train[split:]
    
    # Train base models and get blend predictions
    blend_preds = np.zeros((len(X_blend), len(base_models)))
    test_preds = np.zeros((len(X_test), len(base_models)))
    
    for i, model in enumerate(base_models):
        model.fit(X_train_base, y_train_base)
        blend_preds[:, i] = model.predict_proba(X_blend)[:, 1]
        test_preds[:, i] = model.predict_proba(X_test)[:, 1]
    
    # Train meta model
    meta_model.fit(blend_preds, y_blend)
    
    # Final predictions
    return meta_model.predict(test_preds)
```

---

## Summary

| Method | Training | Variance | Bias | Best For |
|--------|----------|----------|------|----------|
| Bagging | Parallel | Reduces | Same | High-variance models |
| Random Forest | Parallel | Reduces | Same | Most problems |
| AdaBoost | Sequential | May increase | Reduces | Weak learners |
| Gradient Boosting | Sequential | May increase | Reduces | Structured data |
| XGBoost/LightGBM | Sequential | Controlled | Reduces | Competitions |
| Stacking | Two-level | Reduces | Reduces | Max performance |

---

**Last Updated**: 2024-01-29
