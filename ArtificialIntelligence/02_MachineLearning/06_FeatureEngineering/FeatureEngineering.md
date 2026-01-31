# Feature Engineering: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: What is Feature Engineering?](#chapter-1-what-is-feature-engineering)
  - [Chapter 2: Handling Missing Data](#chapter-2-handling-missing-data)
  - [Chapter 3: Encoding Categorical Variables](#chapter-3-encoding-categorical-variables)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: Feature Scaling](#chapter-4-feature-scaling)
  - [Chapter 5: Feature Creation](#chapter-5-feature-creation)
  - [Chapter 6: Feature Selection](#chapter-6-feature-selection)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: Automated Feature Engineering](#chapter-7-automated-feature-engineering)
  - [Chapter 8: Feature Engineering for Different Data Types](#chapter-8-feature-engineering-for-different-data-types)
  - [Chapter 9: Best Practices](#chapter-9-best-practices)

---

## Introduction

**Feature Engineering** is the process of using domain knowledge to create features that make machine learning algorithms work better.

> "Coming up with features is difficult, time-consuming, requires expert knowledge. Applied machine learning is basically feature engineering." — Andrew Ng

### Why Feature Engineering Matters

| Aspect | Impact |
|--------|--------|
| **Model Performance** | Good features > Complex models |
| **Training Speed** | Better features = Faster convergence |
| **Interpretability** | Meaningful features = Explainable models |

---

## Part I: Beginner Level

### Chapter 1: What is Feature Engineering?

#### 1.1 Definition

**Feature Engineering**: Transforming raw data into features that better represent the underlying problem.

**Pipeline**:
```
Raw Data → Feature Engineering → Model-Ready Features → ML Model
```

#### 1.2 Types of Feature Engineering

| Type | Description | Example |
|------|-------------|---------|
| **Transformation** | Change existing features | Log transform, Scaling |
| **Creation** | Create new features | Ratios, Aggregations |
| **Selection** | Choose relevant features | Filter, Wrapper methods |
| **Extraction** | Extract from complex data | Text → TF-IDF, PCA |

---

### Chapter 2: Handling Missing Data

#### 2.1 Detecting Missing Values

```python
import pandas as pd
import numpy as np

# Load data
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

# Check missing values
print("Missing values per column:")
print(df.isnull().sum())

print("\nMissing percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))
```

#### 2.2 Imputation Strategies

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Mean imputation
mean_imputer = SimpleImputer(strategy='mean')
df_mean = pd.DataFrame(mean_imputer.fit_transform(df), columns=df.columns)

# Median imputation (robust to outliers)
median_imputer = SimpleImputer(strategy='median')
df_median = pd.DataFrame(median_imputer.fit_transform(df), columns=df.columns)

# Mode imputation (for categorical)
mode_imputer = SimpleImputer(strategy='most_frequent')

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=3)
df_knn = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
```

#### 2.3 Missing Indicator

```python
# Add indicator for missing values
df['B_missing'] = df['B'].isnull().astype(int)
```

---

### Chapter 3: Encoding Categorical Variables

#### 3.1 Label Encoding

For ordinal categories:

```python
from sklearn.preprocessing import LabelEncoder

# Ordinal data
education = ['High School', 'Bachelor', 'Master', 'PhD']
le = LabelEncoder()
encoded = le.fit_transform(education)
print(f"Encoded: {encoded}")  # [0, 1, 2, 3]
```

#### 3.2 One-Hot Encoding

For nominal categories:

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

colors = pd.DataFrame({'color': ['red', 'blue', 'green', 'red']})

# Using pandas
one_hot = pd.get_dummies(colors, prefix='color')
print(one_hot)

# Using sklearn
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(colors)
```

#### 3.3 Target Encoding

```python
def target_encode(df, column, target):
    """Encode categorical with target mean"""
    means = df.groupby(column)[target].mean()
    return df[column].map(means)

# Example
df['category_encoded'] = target_encode(df, 'category', 'target')
```

#### 3.4 Frequency Encoding

```python
def frequency_encode(df, column):
    """Encode by frequency"""
    freq = df[column].value_counts(normalize=True)
    return df[column].map(freq)
```

---

## Part II: Intermediate Level

### Chapter 4: Feature Scaling

#### 4.1 Standardization (Z-score)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Formula: z = (x - μ) / σ
# Result: mean=0, std=1
```

#### 4.2 Min-Max Normalization

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Formula: x_scaled = (x - min) / (max - min)
# Result: values in [0, 1]
```

#### 4.3 Robust Scaling

```python
from sklearn.preprocessing import RobustScaler

# Robust to outliers (uses median and IQR)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Formula: x_scaled = (x - median) / IQR
```

#### 4.4 When to Use Which

| Scaler | Use When |
|--------|----------|
| StandardScaler | Normal distribution, no outliers |
| MinMaxScaler | Bounded range needed, no outliers |
| RobustScaler | Outliers present |
| None | Tree-based models |

---

### Chapter 5: Feature Creation

#### 5.1 Mathematical Transformations

```python
import numpy as np

# Log transform (for skewed data)
df['log_income'] = np.log1p(df['income'])

# Square root
df['sqrt_area'] = np.sqrt(df['area'])

# Power transform
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['transformed'] = pt.fit_transform(df[['skewed_feature']])
```

#### 5.2 Interaction Features

```python
# Multiplication
df['area'] = df['length'] * df['width']

# Ratio
df['price_per_sqft'] = df['price'] / df['area']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

#### 5.3 Date/Time Features

```python
# Extract components from datetime
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

#### 5.4 Aggregation Features

```python
# Group statistics
df['user_avg_purchase'] = df.groupby('user_id')['amount'].transform('mean')
df['user_total_orders'] = df.groupby('user_id')['order_id'].transform('count')
df['user_max_purchase'] = df.groupby('user_id')['amount'].transform('max')
```

---

### Chapter 6: Feature Selection

#### 6.1 Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Univariate selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]

# Correlation-based selection
corr_matrix = df.corr()
# Remove highly correlated features
high_corr = np.where(np.abs(corr_matrix) > 0.9)
```

#### 6.2 Wrapper Methods

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
rfe = RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=10)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
feature_ranking = rfe.ranking_
```

#### 6.3 Embedded Methods (Feature Importance)

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# L1 regularization (Lasso)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
selected = X.columns[lasso.coef_ != 0]
```

---

## Part III: Advanced Level

### Chapter 7: Automated Feature Engineering

#### 7.1 Featuretools

```python
import featuretools as ft

# Create entity set
es = ft.EntitySet(id='data')

es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions_df,
    index='transaction_id',
    time_index='transaction_time'
)

es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers_df,
    index='customer_id'
)

# Define relationship
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    agg_primitives=['mean', 'sum', 'count', 'max', 'min', 'std'],
    trans_primitives=['month', 'weekday', 'is_weekend']
)
```

---

### Chapter 8: Feature Engineering for Different Data Types

#### 8.1 Text Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
text_features = tfidf.fit_transform(df['text'])

# Text statistics
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]))
```

#### 8.2 Image Features

```python
# Using pre-trained models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()
```

#### 8.3 Geospatial Features

```python
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points"""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

# Cluster-based features
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
df['location_cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
```

---

### Chapter 9: Best Practices

#### 9.1 Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

#### 9.2 Avoiding Data Leakage

```python
# WRONG: Fit on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test data statistics!

# RIGHT: Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use train statistics
```

#### 9.3 Feature Documentation

```python
feature_documentation = {
    'user_avg_purchase': {
        'description': 'Average purchase amount per user',
        'formula': 'mean(amount) GROUP BY user_id',
        'source': 'transactions table',
        'missing_handling': 'Fill with global mean'
    }
}
```

---

## Summary

| Stage | Techniques |
|-------|------------|
| **Missing Data** | Imputation, Indicators |
| **Categorical** | One-hot, Target, Frequency encoding |
| **Numeric** | Scaling, Transformations |
| **Creation** | Interactions, Aggregations, Time features |
| **Selection** | Filter, Wrapper, Embedded methods |

---

**Last Updated**: 2024-01-29
