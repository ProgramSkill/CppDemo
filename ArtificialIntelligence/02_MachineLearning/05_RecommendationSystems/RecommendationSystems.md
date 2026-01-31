# Recommendation Systems: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: What are Recommendation Systems?](#chapter-1-what-are-recommendation-systems)
  - [Chapter 2: Content-Based Filtering](#chapter-2-content-based-filtering)
  - [Chapter 3: Collaborative Filtering Basics](#chapter-3-collaborative-filtering-basics)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: Matrix Factorization](#chapter-4-matrix-factorization)
  - [Chapter 5: Evaluation Metrics](#chapter-5-evaluation-metrics)
  - [Chapter 6: Hybrid Systems](#chapter-6-hybrid-systems)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: Deep Learning for Recommendations](#chapter-7-deep-learning-for-recommendations)
  - [Chapter 8: Sequential Recommendations](#chapter-8-sequential-recommendations)
  - [Chapter 9: Production Systems](#chapter-9-production-systems)

---

## Introduction

**Recommendation Systems** predict user preferences and suggest relevant items.

### Applications

| Domain | Examples |
|--------|----------|
| E-commerce | Product recommendations (Amazon) |
| Streaming | Movie/Music suggestions (Netflix, Spotify) |
| Social Media | Friend/Content suggestions |
| News | Article recommendations |

### Types Overview

| Type | Approach | Pros | Cons |
|------|----------|------|------|
| **Content-Based** | Item features | No cold start for items | Limited diversity |
| **Collaborative** | User behavior | Discovers new patterns | Cold start problem |
| **Hybrid** | Combined | Best of both | More complex |

---

## Part I: Beginner Level

### Chapter 1: What are Recommendation Systems?

#### 1.1 The Problem

Given:
- Users U = {u₁, u₂, ..., uₙ}
- Items I = {i₁, i₂, ..., iₘ}
- Interactions (ratings, clicks, purchases)

Goal: Predict which items a user will like

#### 1.2 User-Item Matrix

```
        Item1  Item2  Item3  Item4
User1    5      3      ?      1
User2    4      ?      ?      1
User3    1      1      ?      5
User4    ?      ?      5      4
```

**Task**: Fill in the missing values (?)

```python
import numpy as np
import pandas as pd

# Create user-item matrix
ratings = pd.DataFrame({
    'User1': [5, 3, np.nan, 1],
    'User2': [4, np.nan, np.nan, 1],
    'User3': [1, 1, np.nan, 5],
    'User4': [np.nan, np.nan, 5, 4]
}, index=['Item1', 'Item2', 'Item3', 'Item4']).T

print(ratings)
```

---

### Chapter 2: Content-Based Filtering

#### 2.1 Concept

Recommend items similar to what user has liked before.

**Steps**:
1. Create item feature vectors
2. Build user profile from liked items
3. Recommend items similar to profile

#### 2.2 Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Item descriptions
items = {
    'Movie1': 'action adventure superhero',
    'Movie2': 'romantic comedy love',
    'Movie3': 'action thriller crime',
    'Movie4': 'romantic drama love',
    'Movie5': 'action superhero sci-fi'
}

# Create TF-IDF features
tfidf = TfidfVectorizer()
item_features = tfidf.fit_transform(items.values())

# User profile: liked Movie1 and Movie3
user_liked = [0, 2]  # indices
user_profile = item_features[user_liked].mean(axis=0)

# Calculate similarities
similarities = cosine_similarity(user_profile, item_features)[0]

# Recommend (excluding already liked)
for idx, (name, sim) in enumerate(zip(items.keys(), similarities)):
    if idx not in user_liked:
        print(f"{name}: {sim:.3f}")
```

#### 2.3 Pros and Cons

| Pros | Cons |
|------|------|
| No cold start for new items | Limited to existing interests |
| Transparent recommendations | Feature engineering required |
| User-independent | Can't discover new interests |

---

### Chapter 3: Collaborative Filtering Basics

#### 3.1 User-Based CF

**Idea**: Similar users like similar items

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# User-item matrix (0 = not rated)
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4]
])

# Calculate user similarity
user_similarity = cosine_similarity(ratings)
print("User Similarity Matrix:")
print(user_similarity)

def predict_rating(user_idx, item_idx, ratings, similarity):
    # Find users who rated this item
    rated_users = ratings[:, item_idx] > 0
    rated_users[user_idx] = False  # Exclude target user
    
    if not any(rated_users):
        return 0
    
    # Weighted average by similarity
    weights = similarity[user_idx, rated_users]
    values = ratings[rated_users, item_idx]
    
    return np.dot(weights, values) / (np.sum(np.abs(weights)) + 1e-10)

# Predict User0's rating for Item2
pred = predict_rating(0, 2, ratings, user_similarity)
print(f"\nPredicted rating for User0, Item2: {pred:.2f}")
```

#### 3.2 Item-Based CF

**Idea**: Similar items get similar ratings

```python
# Calculate item similarity
item_similarity = cosine_similarity(ratings.T)

def predict_rating_item_based(user_idx, item_idx, ratings, item_sim):
    # Find items user has rated
    rated_items = ratings[user_idx, :] > 0
    rated_items[item_idx] = False
    
    if not any(rated_items):
        return 0
    
    weights = item_sim[item_idx, rated_items]
    values = ratings[user_idx, rated_items]
    
    return np.dot(weights, values) / (np.sum(np.abs(weights)) + 1e-10)
```

---

## Part II: Intermediate Level

### Chapter 4: Matrix Factorization

#### 4.1 Concept

Decompose user-item matrix into latent factors:
```
R ≈ P × Qᵀ

Where:
- R: m×n rating matrix
- P: m×k user factor matrix
- Q: n×k item factor matrix
- k: number of latent factors
```

#### 4.2 SVD Implementation

```python
from scipy.sparse.linalg import svds
import numpy as np

# Fill missing values with mean
ratings_filled = ratings.copy().astype(float)
ratings_filled[ratings_filled == 0] = np.nan
mean_rating = np.nanmean(ratings_filled)
ratings_filled = np.nan_to_num(ratings_filled, nan=mean_rating)

# Perform SVD
k = 2  # Number of latent factors
U, sigma, Vt = svds(ratings_filled, k=k)

# Reconstruct matrix
sigma_diag = np.diag(sigma)
predicted_ratings = U @ sigma_diag @ Vt

print("Predicted Ratings:")
print(predicted_ratings.round(2))
```

#### 4.3 Alternating Least Squares (ALS)

```python
def als(R, k=10, lambda_reg=0.1, n_iterations=20):
    m, n = R.shape
    
    # Initialize factors randomly
    P = np.random.rand(m, k)
    Q = np.random.rand(n, k)
    
    # Mask for observed ratings
    mask = R > 0
    
    for iteration in range(n_iterations):
        # Fix Q, solve for P
        for i in range(m):
            rated = mask[i, :]
            if not any(rated):
                continue
            Q_rated = Q[rated, :]
            R_rated = R[i, rated]
            P[i, :] = np.linalg.solve(
                Q_rated.T @ Q_rated + lambda_reg * np.eye(k),
                Q_rated.T @ R_rated
            )
        
        # Fix P, solve for Q
        for j in range(n):
            rated = mask[:, j]
            if not any(rated):
                continue
            P_rated = P[rated, :]
            R_rated = R[rated, j]
            Q[j, :] = np.linalg.solve(
                P_rated.T @ P_rated + lambda_reg * np.eye(k),
                P_rated.T @ R_rated
            )
    
    return P, Q

P, Q = als(ratings, k=2)
predicted = P @ Q.T
print("ALS Predictions:")
print(predicted.round(2))
```

---

### Chapter 5: Evaluation Metrics

#### 5.1 Rating Prediction Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Example
true_ratings = [5, 3, 4, 2, 1]
pred_ratings = [4.5, 3.2, 3.8, 2.5, 1.2]

print(f"RMSE: {rmse(true_ratings, pred_ratings):.4f}")
print(f"MAE: {mae(true_ratings, pred_ratings):.4f}")
```

#### 5.2 Ranking Metrics

```python
def precision_at_k(recommended, relevant, k):
    """Precision@K: relevant items in top-k / k"""
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    """Recall@K: relevant items in top-k / total relevant"""
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    """Normalized Discounted Cumulative Gain"""
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) 
              if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

# Example
recommended = ['A', 'B', 'C', 'D', 'E']
relevant = ['A', 'C', 'F']

print(f"Precision@3: {precision_at_k(recommended, relevant, 3):.4f}")
print(f"Recall@3: {recall_at_k(recommended, relevant, 3):.4f}")
print(f"NDCG@3: {ndcg_at_k(recommended, relevant, 3):.4f}")
```

---

### Chapter 6: Hybrid Systems

#### 6.1 Combination Strategies

| Strategy | Description |
|----------|-------------|
| **Weighted** | Combine scores with weights |
| **Switching** | Switch based on conditions |
| **Feature Combination** | Combine features from both |
| **Cascade** | Use one to refine another |

#### 6.2 Weighted Hybrid

```python
def hybrid_recommend(user_id, content_scores, collab_scores, alpha=0.5):
    """
    alpha: weight for content-based (1-alpha for collaborative)
    """
    hybrid_scores = alpha * content_scores + (1 - alpha) * collab_scores
    return np.argsort(hybrid_scores)[::-1]

# Example
content = np.array([0.8, 0.2, 0.6, 0.9, 0.3])
collab = np.array([0.3, 0.7, 0.5, 0.4, 0.9])

recommendations = hybrid_recommend(0, content, collab, alpha=0.6)
print(f"Recommended items (sorted): {recommendations}")
```

---

## Part III: Advanced Level

### Chapter 7: Deep Learning for Recommendations

#### 7.1 Neural Collaborative Filtering

```python
import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32, hidden_dims=[64, 32]):
        super().__init__()
        
        # Embeddings
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        
        # MLP layers
        layers = []
        input_dim = embed_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embed(user_ids)
        item_embed = self.item_embed(item_ids)
        
        concat = torch.cat([user_embed, item_embed], dim=-1)
        return self.mlp(concat).squeeze()
```

#### 7.2 Two-Tower Model

```python
class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64):
        super().__init__()
        
        # User tower
        self.user_tower = nn.Sequential(
            nn.Embedding(n_users, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        
        # Item tower
        self.item_tower = nn.Sequential(
            nn.Embedding(n_items, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
    
    def forward(self, user_ids, item_ids):
        user_embed = self.user_tower[0](user_ids)
        user_embed = self.user_tower[1](user_embed)
        user_embed = self.user_tower[2](user_embed)
        
        item_embed = self.item_tower[0](item_ids)
        item_embed = self.item_tower[1](item_embed)
        item_embed = self.item_tower[2](item_embed)
        
        # Dot product similarity
        return (user_embed * item_embed).sum(dim=-1)
```

---

### Chapter 8: Sequential Recommendations

#### 8.1 Session-Based Recommendations

Using RNN/Transformer to model user sessions:

```python
class SessionRec(nn.Module):
    def __init__(self, n_items, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_items)
    
    def forward(self, session_items):
        # session_items: (batch, seq_len)
        embeds = self.item_embed(session_items)
        _, hidden = self.gru(embeds)
        logits = self.fc(hidden.squeeze(0))
        return logits
```

---

### Chapter 9: Production Systems

#### 9.1 Two-Stage Architecture

```
Stage 1: Candidate Generation (fast, simple)
    - ANN search on embeddings
    - Multiple retrievers (CF, content, popular)
    
Stage 2: Ranking (slower, complex)
    - Deep neural network
    - More features
    - Personalization
```

#### 9.2 Key Considerations

| Aspect | Considerations |
|--------|----------------|
| **Latency** | Real-time vs batch, caching |
| **Scalability** | Millions of users/items |
| **Freshness** | How often to retrain |
| **Cold Start** | New users/items handling |
| **Diversity** | Avoid filter bubbles |

---

## Summary

| Method | Best For | Cold Start | Scalability |
|--------|----------|------------|-------------|
| Content-Based | New items | Good | Good |
| User-Based CF | Small datasets | Poor | Poor |
| Item-Based CF | E-commerce | Moderate | Good |
| Matrix Factorization | Medium datasets | Poor | Good |
| Deep Learning | Large datasets | Can handle | Excellent |

---

**Last Updated**: 2024-01-29
