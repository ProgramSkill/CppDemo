# Unsupervised Learning: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: What is Unsupervised Learning?](#chapter-1-what-is-unsupervised-learning)
  - [Chapter 2: K-Means Clustering](#chapter-2-k-means-clustering)
  - [Chapter 3: Hierarchical Clustering](#chapter-3-hierarchical-clustering)
  - [Chapter 4: Introduction to Dimensionality Reduction](#chapter-4-introduction-to-dimensionality-reduction)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 5: Advanced Clustering Methods](#chapter-5-advanced-clustering-methods)
  - [Chapter 6: Principal Component Analysis](#chapter-6-principal-component-analysis)
  - [Chapter 7: t-SNE and UMAP](#chapter-7-t-sne-and-umap)
  - [Chapter 8: Association Rule Learning](#chapter-8-association-rule-learning)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 9: Gaussian Mixture Models](#chapter-9-gaussian-mixture-models)
  - [Chapter 10: Autoencoders](#chapter-10-autoencoders)
  - [Chapter 11: Anomaly Detection](#chapter-11-anomaly-detection)
  - [Chapter 12: Real-World Applications](#chapter-12-real-world-applications)

---

## Introduction

Welcome to the comprehensive guide on **Unsupervised Learning**! Unlike supervised learning where we have labeled data, unsupervised learning discovers hidden patterns in data without predefined labels.

### What You'll Learn

| Level | Duration | Topics Covered | Skills Acquired |
|-------|----------|----------------|-----------------|
| **Beginner** | 2-4 weeks | K-Means, Hierarchical Clustering, Basic PCA | Understand clustering concepts |
| **Intermediate** | 4-6 weeks | DBSCAN, Advanced PCA, t-SNE, Association Rules | Apply techniques to real data |
| **Advanced** | 4-6 weeks | GMM, Autoencoders, Anomaly Detection | Solve complex unsupervised problems |

### Supervised vs Unsupervised Learning

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| **Labels** | Required | Not required |
| **Goal** | Predict labels | Find patterns |
| **Examples** | Classification, Regression | Clustering, Dimensionality Reduction |
| **Evaluation** | Accuracy, MSE | Silhouette, Reconstruction error |

---

## Part I: Beginner Level

### Chapter 1: What is Unsupervised Learning?

#### 1.1 Definition

**Unsupervised Learning** is a type of machine learning where the algorithm learns patterns from unlabeled data.

**Key Insight**: Without labels, the algorithm must find structure on its own.

#### 1.2 Types of Unsupervised Learning

| Type | Goal | Examples |
|------|------|----------|
| **Clustering** | Group similar data points | Customer segmentation |
| **Dimensionality Reduction** | Reduce number of features | Data visualization, Compression |
| **Association** | Find relationships between items | Market basket analysis |
| **Anomaly Detection** | Find unusual data points | Fraud detection |

#### 1.3 When to Use Unsupervised Learning

- **Exploratory data analysis**: Understanding data structure
- **No labeled data available**: Labels are expensive or impossible to obtain
- **Feature learning**: Learning representations for downstream tasks
- **Data preprocessing**: Reducing noise or dimensions

---

### Chapter 2: K-Means Clustering

#### 2.1 Introduction

**K-Means** is the most popular clustering algorithm. It partitions data into K distinct clusters.

**Algorithm**:
```
1. Initialize K cluster centroids randomly
2. Repeat until convergence:
   a. Assign each point to nearest centroid
   b. Update centroids as mean of assigned points
```

#### 2.2 Mathematical Formulation

**Objective**: Minimize within-cluster sum of squares (WCSS)
```
J = Σᵢ₌₁ᴷ Σₓ∈Cᵢ ||x - μᵢ||²

Where:
- K = number of clusters
- Cᵢ = set of points in cluster i
- μᵢ = centroid of cluster i
```

#### 2.3 Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Visualize results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
plt.title('Original Data')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
```

#### 2.4 Choosing K: The Elbow Method

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal K')
plt.show()
```

#### 2.5 Silhouette Score

**Silhouette Score**: Measures how similar a point is to its own cluster compared to other clusters.

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

Where:
- a(i) = average distance to points in same cluster
- b(i) = average distance to points in nearest cluster
- Range: [-1, 1], higher is better
```

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, y_pred)
print(f"Silhouette Score: {score:.4f}")
```

#### 2.6 Limitations of K-Means

| Limitation | Description | Solution |
|------------|-------------|----------|
| Requires K | Must specify number of clusters | Elbow method, Silhouette |
| Spherical clusters | Assumes circular cluster shapes | DBSCAN, Spectral clustering |
| Sensitive to outliers | Outliers affect centroids | K-Medoids, DBSCAN |
| Random initialization | Different runs give different results | K-Means++, multiple runs |

---

### Chapter 3: Hierarchical Clustering

#### 3.1 Introduction

**Hierarchical Clustering** builds a tree of clusters (dendrogram). Two approaches:
- **Agglomerative** (bottom-up): Start with each point as cluster, merge
- **Divisive** (top-down): Start with one cluster, split

#### 3.2 Linkage Methods

| Method | Distance Between Clusters |
|--------|--------------------------|
| **Single** | Minimum distance between points |
| **Complete** | Maximum distance between points |
| **Average** | Average distance between all pairs |
| **Ward** | Minimizes variance increase |

#### 3.3 Implementation

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

# Generate data
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# Create linkage matrix
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=12)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Cut dendrogram to get clusters
n_clusters = 3
clusters = fcluster(Z, n_clusters, criterion='maxclust')

# Using sklearn
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_pred = hc.fit_predict(X)
```

#### 3.4 Advantages and Disadvantages

**Advantages**:
- No need to specify K beforehand
- Produces interpretable dendrogram
- Can choose any number of clusters from dendrogram

**Disadvantages**:
- Computationally expensive: O(n³) for naive, O(n² log n) for optimized
- Cannot undo merge/split decisions

---

### Chapter 4: Introduction to Dimensionality Reduction

#### 4.1 Why Reduce Dimensions?

- **Curse of Dimensionality**: High-dimensional data is sparse
- **Visualization**: Can only visualize 2D/3D data
- **Noise Reduction**: Remove irrelevant features
- **Computation**: Faster algorithms with fewer features

#### 4.2 Feature Selection vs Feature Extraction

| Approach | Method | Example |
|----------|--------|---------|
| **Feature Selection** | Select subset of original features | Filter, Wrapper methods |
| **Feature Extraction** | Create new features from original | PCA, Autoencoders |

#### 4.3 Basic Example

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load high-dimensional data
iris = load_iris()
X = iris.data  # 4 features

# Reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=iris.target, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Iris Dataset - PCA Projection')
plt.colorbar(label='Species')
plt.show()
```

---

## Part II: Intermediate Level

### Chapter 5: Advanced Clustering Methods

#### 5.1 DBSCAN (Density-Based Spatial Clustering)

**Key Concepts**:
- **Core point**: Has at least `min_samples` points within `eps` radius
- **Border point**: Within `eps` of a core point
- **Noise point**: Neither core nor border

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Data with non-spherical clusters
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

# K-Means fails on this data
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# DBSCAN works well
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# Compare
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
axes[0].set_title('K-Means (fails)')

axes[1].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis')
axes[1].set_title('DBSCAN (succeeds)')
plt.show()
```

**Advantages**:
- No need to specify K
- Can find arbitrarily shaped clusters
- Robust to outliers (identifies noise)

**Disadvantages**:
- Sensitive to eps and min_samples
- Struggles with varying density clusters

#### 5.2 Spectral Clustering

Uses eigenvalues of similarity matrix to reduce dimensions before clustering.

```python
from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
y_spectral = spectral.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_spectral, cmap='viridis')
plt.title('Spectral Clustering')
plt.show()
```

---

### Chapter 6: Principal Component Analysis

#### 6.1 Mathematical Foundation

**Goal**: Find directions (principal components) that maximize variance.

**Steps**:
1. Center data: X_centered = X - mean(X)
2. Compute covariance matrix: C = (1/n) × X_centered^T × X_centered
3. Find eigenvalues and eigenvectors of C
4. Sort by eigenvalue (largest first)
5. Project data onto top k eigenvectors

```
Principal Components = Eigenvectors of Covariance Matrix
Variance Explained = Eigenvalue / Sum(Eigenvalues)
```

#### 6.2 Implementation from Scratch

```python
import numpy as np

def pca_from_scratch(X, n_components):
    # Center data
    X_centered = X - X.mean(axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # Select top k components
    components = eigenvectors[:, :n_components]
    
    # Project data
    X_transformed = X_centered @ components
    
    # Explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / eigenvalues.sum()
    
    return X_transformed, explained_variance_ratio

# Example
X, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
X_pca, var_ratio = pca_from_scratch(X, n_components=2)
print(f"Variance explained: {var_ratio}")
```

#### 6.3 Choosing Number of Components

```python
from sklearn.decomposition import PCA

# Fit PCA with all components
pca_full = PCA()
pca_full.fit(X)

# Plot cumulative explained variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumsum)+1), cumsum, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance vs Components')
plt.legend()
plt.show()

# Find number for 95% variance
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components for 95% variance: {n_components_95}")
```

---

### Chapter 7: t-SNE and UMAP

#### 7.1 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Purpose**: Non-linear dimensionality reduction for visualization.

**Key Parameters**:
- **perplexity**: Balance between local and global structure (typically 5-50)
- **learning_rate**: Step size (typically 10-1000)

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('t-SNE Visualization of Digits Dataset')
plt.show()
```

#### 7.2 UMAP (Uniform Manifold Approximation and Projection)

**Advantages over t-SNE**:
- Faster
- Better preserves global structure
- Can be used for new data (transform)

```python
# pip install umap-learn
import umap

# Apply UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.title('UMAP Visualization of Digits Dataset')
plt.show()
```

#### 7.3 Comparison

| Aspect | PCA | t-SNE | UMAP |
|--------|-----|-------|------|
| Type | Linear | Non-linear | Non-linear |
| Speed | Fast | Slow | Medium |
| Global structure | Yes | Limited | Yes |
| New data | Yes | No | Yes |
| Interpretable | Yes | No | No |

---

### Chapter 8: Association Rule Learning

#### 8.1 Introduction

**Goal**: Find interesting relationships between variables in large datasets.

**Example**: Market Basket Analysis
- "Customers who buy bread often buy butter"

#### 8.2 Key Metrics

**Support**: How frequently an itemset appears
```
Support(A) = Count(A) / Total Transactions
```

**Confidence**: How often B appears when A appears
```
Confidence(A → B) = Support(A ∪ B) / Support(A)
```

**Lift**: How much more likely B is given A
```
Lift(A → B) = Confidence(A → B) / Support(B)
Lift > 1: Positive correlation
Lift = 1: No correlation
Lift < 1: Negative correlation
```

#### 8.3 Apriori Algorithm

```python
# pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# Sample transactions
transactions = [
    ['bread', 'milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'butter', 'eggs'],
    ['bread', 'milk', 'butter'],
    ['bread', 'milk', 'butter', 'eggs']
]

# Encode transactions
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

---

## Part III: Advanced Level

### Chapter 9: Gaussian Mixture Models

#### 9.1 Introduction

**GMM** assumes data comes from a mixture of Gaussian distributions.

**Model**:
```
P(x) = Σₖ πₖ × N(x | μₖ, Σₖ)

Where:
- πₖ = mixing coefficient (weight) for component k
- μₖ = mean of component k
- Σₖ = covariance of component k
```

#### 9.2 Expectation-Maximization (EM) Algorithm

**E-Step**: Compute responsibility of each component for each point
```
γₖ(xᵢ) = P(zᵢ = k | xᵢ)
```

**M-Step**: Update parameters using responsibilities
```
μₖ = Σᵢ γₖ(xᵢ) × xᵢ / Σᵢ γₖ(xᵢ)
```

#### 9.3 Implementation

```python
from sklearn.mixture import GaussianMixture

# Generate data
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
y_gmm = gmm.fit_predict(X)

# Get probabilities
proba = gmm.predict_proba(X)
print(f"Sample probabilities:\n{proba[:3]}")

# Model parameters
print(f"\nMeans:\n{gmm.means_}")
print(f"\nWeights: {gmm.weights_}")

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis', alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=200)
plt.title('Gaussian Mixture Model')
plt.show()
```

#### 9.4 GMM vs K-Means

| Aspect | K-Means | GMM |
|--------|---------|-----|
| Cluster shape | Spherical | Elliptical |
| Assignment | Hard | Soft (probabilities) |
| Covariance | Identity | Full, diagonal, etc. |
| Objective | Minimize distance | Maximize likelihood |

---

### Chapter 10: Autoencoders

#### 10.1 Introduction

**Autoencoder** is a neural network that learns to compress and reconstruct data.

```
Input → Encoder → Latent Space → Decoder → Reconstruction
  x        f(x)        z           g(z)        x̂
```

**Loss**: Reconstruction error
```
L = ||x - x̂||²
```

#### 10.2 Architecture

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(latent_dim, activation='relu')
        ])
        
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(784, activation='sigmoid')  # For MNIST
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Create and train
autoencoder = Autoencoder(latent_dim=32)
autoencoder.compile(optimizer='adam', loss='mse')
```

#### 10.3 Variational Autoencoder (VAE)

**Key Difference**: Latent space is a probability distribution.

```
Encoder outputs: μ (mean) and σ (std)
Sampling: z = μ + σ × ε, where ε ~ N(0, 1)
Loss = Reconstruction Loss + KL Divergence
```

---

### Chapter 11: Anomaly Detection

#### 11.1 Methods

| Method | Approach | Use Case |
|--------|----------|----------|
| **Statistical** | Z-score, IQR | Simple datasets |
| **Distance-based** | LOF, k-NN | Local anomalies |
| **Density-based** | DBSCAN | Cluster-based |
| **Isolation Forest** | Random trees | High-dimensional |
| **Autoencoder** | Reconstruction error | Complex patterns |

#### 11.2 Isolation Forest

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Generate data with anomalies
np.random.seed(42)
X_normal = np.random.randn(200, 2)
X_anomaly = np.random.uniform(-4, 4, (20, 2))
X = np.vstack([X_normal, X_anomaly])

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso_forest.fit_predict(X)

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='blue', label='Normal')
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', label='Anomaly')
plt.title('Isolation Forest - Anomaly Detection')
plt.legend()
plt.show()
```

#### 11.3 Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(X)

# LOF scores (negative = outlier)
scores = lof.negative_outlier_factor_
print(f"Outlier scores range: {scores.min():.2f} to {scores.max():.2f}")
```

---

### Chapter 12: Real-World Applications

#### 12.1 Customer Segmentation

```python
# Cluster customers based on purchasing behavior
from sklearn.preprocessing import StandardScaler

# Features: recency, frequency, monetary
customer_data = np.array([
    [10, 5, 500],   # Recent, frequent, high value
    [100, 1, 50],   # Old, rare, low value
    [5, 10, 1000],  # Very recent, very frequent, very high
    # ... more customers
])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_data)

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
segments = kmeans.fit_predict(X_scaled)
```

#### 12.2 Image Compression with PCA

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X = digits.data  # 64 dimensions (8x8 images)

# Compress to different dimensions
for n in [10, 20, 40]:
    pca = PCA(n_components=n)
    X_compressed = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_compressed)
    
    mse = ((X - X_reconstructed) ** 2).mean()
    print(f"Components: {n}, MSE: {mse:.4f}, Variance: {pca.explained_variance_ratio_.sum():.2%}")
```

#### 12.3 Recommendation via Matrix Factorization

```python
from sklearn.decomposition import NMF

# User-item rating matrix
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4]
])

# Non-negative Matrix Factorization
nmf = NMF(n_components=2, random_state=42)
W = nmf.fit_transform(ratings)  # User features
H = nmf.components_             # Item features

# Reconstruct (predict missing ratings)
predicted_ratings = W @ H
print("Predicted ratings:")
print(predicted_ratings.round(1))
```

---

## Summary

| Topic | Key Algorithm | Use Case |
|-------|--------------|----------|
| **Clustering** | K-Means, DBSCAN, GMM | Customer segmentation |
| **Dim Reduction** | PCA, t-SNE, UMAP | Visualization, Compression |
| **Association** | Apriori, FP-Growth | Market basket analysis |
| **Anomaly** | Isolation Forest, LOF | Fraud detection |

---

**Last Updated**: 2024-01-29
