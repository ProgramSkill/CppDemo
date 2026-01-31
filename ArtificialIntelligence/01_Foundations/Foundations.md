# Mathematical Foundations for AI: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: Linear Algebra Basics](#chapter-1-linear-algebra-basics)
  - [Chapter 2: Calculus Fundamentals](#chapter-2-calculus-fundamentals)
  - [Chapter 3: Probability Basics](#chapter-3-probability-basics)
  - [Chapter 4: Basic Statistics](#chapter-4-basic-statistics)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 5: Advanced Linear Algebra](#chapter-5-advanced-linear-algebra)
  - [Chapter 6: Multivariable Calculus](#chapter-6-multivariable-calculus)
  - [Chapter 7: Probability Distributions](#chapter-7-probability-distributions)
  - [Chapter 8: Statistical Inference](#chapter-8-statistical-inference)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 9: Optimization Theory](#chapter-9-optimization-theory)
  - [Chapter 10: Information Theory](#chapter-10-information-theory)
  - [Chapter 11: Algorithm Complexity](#chapter-11-algorithm-complexity)
  - [Chapter 12: Applied Mathematics in ML](#chapter-12-applied-mathematics-in-ml)

---

## Introduction

Welcome to the comprehensive guide on **Mathematical Foundations for AI**! Mathematics is the language of machine learning. This tutorial will take you from basic concepts to advanced techniques essential for understanding and implementing AI algorithms.

### What You'll Learn

| Level | Duration | Topics Covered | Skills Acquired |
|-------|----------|----------------|-----------------|
| **Beginner** | 3-4 weeks | Vectors, Matrices, Derivatives, Basic Probability | Understand fundamental concepts |
| **Intermediate** | 4-6 weeks | Eigenvalues, Gradients, Distributions, Inference | Apply math to ML problems |
| **Advanced** | 4-6 weeks | Optimization, Information Theory, Complexity | Deep understanding of ML theory |

### Why Math for AI?

| Area | Application in AI |
|------|------------------|
| **Linear Algebra** | Neural networks, PCA, Embeddings |
| **Calculus** | Gradient descent, Backpropagation |
| **Probability** | Bayesian learning, Generative models |
| **Statistics** | Hypothesis testing, Model evaluation |
| **Optimization** | Training algorithms, Loss minimization |

---

## Part I: Beginner Level

### Chapter 1: Linear Algebra Basics

#### 1.1 Scalars, Vectors, and Matrices

**Scalar**: A single number
```
a = 5
```

**Vector**: An ordered list of numbers
```
v = [1, 2, 3]  (row vector)

    ⎡1⎤
v = ⎢2⎥  (column vector)
    ⎣3⎦
```

**Matrix**: A 2D array of numbers
```
    ⎡1 2 3⎤
A = ⎢4 5 6⎥
    ⎣7 8 9⎦
```

**Tensor**: An n-dimensional array (generalization)

```python
import numpy as np

# Scalar
scalar = 5

# Vector
vector = np.array([1, 2, 3])

# Matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 3D Tensor
tensor = np.random.randn(3, 4, 5)  # Shape: (3, 4, 5)
```

#### 1.2 Vector Operations

**Addition and Subtraction**:
```
u = [1, 2, 3]
v = [4, 5, 6]

u + v = [5, 7, 9]
u - v = [-3, -3, -3]
```

**Scalar Multiplication**:
```
2 × [1, 2, 3] = [2, 4, 6]
```

**Dot Product** (Inner Product):
```
u · v = u₁v₁ + u₂v₂ + u₃v₃
      = 1×4 + 2×5 + 3×6
      = 32
```

**Vector Norm** (Length/Magnitude):
```
||v|| = √(v₁² + v₂² + ... + vₙ²)

||[3, 4]|| = √(9 + 16) = 5
```

```python
import numpy as np

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Operations
print(u + v)           # [5, 7, 9]
print(u - v)           # [-3, -3, -3]
print(2 * u)           # [2, 4, 6]
print(np.dot(u, v))    # 32
print(np.linalg.norm(u))  # 3.74...
```

#### 1.3 Matrix Operations

**Matrix Addition**:
```
⎡1 2⎤   ⎡5 6⎤   ⎡6  8⎤
⎣3 4⎦ + ⎣7 8⎦ = ⎣10 12⎦
```

**Matrix Multiplication**:
```
(A × B)ᵢⱼ = Σₖ Aᵢₖ × Bₖⱼ

⎡1 2⎤   ⎡5 6⎤   ⎡1×5+2×7  1×6+2×8⎤   ⎡19 22⎤
⎣3 4⎦ × ⎣7 8⎦ = ⎣3×5+4×7  3×6+4×8⎦ = ⎣43 50⎦
```

**Transpose**:
```
    ⎡1 2 3⎤         ⎡1 4⎤
A = ⎣4 5 6⎦   Aᵀ = ⎢2 5⎥
                    ⎣3 6⎦
```

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Operations
print(A + B)       # Matrix addition
print(A @ B)       # Matrix multiplication
print(A.T)         # Transpose
print(np.linalg.det(A))  # Determinant: -2
print(np.linalg.inv(A))  # Inverse
```

#### 1.4 Special Matrices

| Matrix Type | Definition | Example |
|-------------|------------|---------|
| **Identity** | Diagonal of 1s, rest 0s | I₃ = diag(1,1,1) |
| **Diagonal** | Non-zero only on diagonal | diag(2,3,4) |
| **Symmetric** | A = Aᵀ | Covariance matrices |
| **Orthogonal** | AᵀA = I | Rotation matrices |

```python
import numpy as np

# Identity matrix
I = np.eye(3)

# Diagonal matrix
D = np.diag([1, 2, 3])

# Check symmetry
A = np.array([[1, 2], [2, 4]])
is_symmetric = np.allclose(A, A.T)
```

---

### Chapter 2: Calculus Fundamentals

#### 2.1 Derivatives

**Definition**: Rate of change of a function
```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**Common Derivatives**:
| Function | Derivative |
|----------|------------|
| xⁿ | n·xⁿ⁻¹ |
| eˣ | eˣ |
| ln(x) | 1/x |
| sin(x) | cos(x) |
| cos(x) | -sin(x) |

**Rules**:
- **Sum Rule**: (f + g)' = f' + g'
- **Product Rule**: (fg)' = f'g + fg'
- **Chain Rule**: (f(g(x)))' = f'(g(x)) · g'(x)

```python
import sympy as sp

x = sp.Symbol('x')

# Derivatives
f = x**3 + 2*x**2 - 5*x + 3
f_prime = sp.diff(f, x)
print(f_prime)  # 3*x**2 + 4*x - 5

# Chain rule example
g = sp.exp(x**2)
g_prime = sp.diff(g, x)
print(g_prime)  # 2*x*exp(x**2)
```

#### 2.2 Partial Derivatives

For functions of multiple variables:
```
f(x, y) = x² + xy + y²

∂f/∂x = 2x + y
∂f/∂y = x + 2y
```

**Gradient**: Vector of all partial derivatives
```
∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z, ...]
```

```python
import numpy as np

def f(x, y):
    return x**2 + x*y + y**2

def gradient_f(x, y):
    df_dx = 2*x + y
    df_dy = x + 2*y
    return np.array([df_dx, df_dy])

# Example
print(gradient_f(1, 2))  # [4, 5]
```

#### 2.3 Gradient Descent

**Idea**: Move in the direction of steepest descent to minimize a function

**Update Rule**:
```
θ_new = θ_old - α × ∇f(θ_old)

Where:
- θ: parameters
- α: learning rate
- ∇f: gradient
```

```python
import numpy as np

def gradient_descent(f, grad_f, x0, learning_rate=0.1, n_iterations=100):
    x = x0
    history = [x.copy()]
    
    for _ in range(n_iterations):
        gradient = grad_f(x)
        x = x - learning_rate * gradient
        history.append(x.copy())
    
    return x, history

# Example: minimize f(x,y) = x² + y²
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x_min, history = gradient_descent(f, grad_f, np.array([5.0, 5.0]))
print(f"Minimum at: {x_min}")  # Close to [0, 0]
```

---

### Chapter 3: Probability Basics

#### 3.1 Fundamental Concepts

**Probability**: Measure of likelihood (0 to 1)
```
P(A) ∈ [0, 1]
P(Ω) = 1  (certain event)
P(∅) = 0  (impossible event)
```

**Conditional Probability**:
```
P(A|B) = P(A ∩ B) / P(B)
```

**Independence**:
```
P(A ∩ B) = P(A) × P(B)  iff A and B are independent
```

#### 3.2 Bayes' Theorem

```
P(A|B) = P(B|A) × P(A) / P(B)
```

**Components**:
- P(A|B): Posterior probability
- P(B|A): Likelihood
- P(A): Prior probability
- P(B): Evidence

**Example**: Medical diagnosis
```
Disease prevalence: P(D) = 0.01
Test sensitivity: P(+|D) = 0.95
Test false positive: P(+|¬D) = 0.05

P(D|+) = P(+|D) × P(D) / P(+)
       = 0.95 × 0.01 / (0.95×0.01 + 0.05×0.99)
       ≈ 0.16
```

```python
def bayes_theorem(prior, likelihood, evidence):
    return (likelihood * prior) / evidence

# Medical diagnosis example
p_disease = 0.01
p_positive_given_disease = 0.95
p_positive_given_no_disease = 0.05

p_positive = p_positive_given_disease * p_disease + \
             p_positive_given_no_disease * (1 - p_disease)

p_disease_given_positive = bayes_theorem(
    p_disease, p_positive_given_disease, p_positive
)
print(f"P(Disease|Positive) = {p_disease_given_positive:.4f}")  # 0.1610
```

#### 3.3 Random Variables

**Discrete Random Variable**: Takes countable values
```
X ∈ {x₁, x₂, ..., xₙ}
P(X = xᵢ) = pᵢ
Σ pᵢ = 1
```

**Continuous Random Variable**: Takes any value in a range
```
P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx

Where f(x) is the probability density function (PDF)
```

**Expected Value** (Mean):
```
E[X] = Σ xᵢ × P(X = xᵢ)     (discrete)
E[X] = ∫ x × f(x) dx        (continuous)
```

**Variance**:
```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
```

```python
import numpy as np

# Discrete random variable
values = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6] * 6)  # Fair die

expected_value = np.sum(values * probabilities)
variance = np.sum((values - expected_value)**2 * probabilities)

print(f"E[X] = {expected_value}")    # 3.5
print(f"Var(X) = {variance}")        # 2.917
```

---

### Chapter 4: Basic Statistics

#### 4.1 Descriptive Statistics

**Measures of Central Tendency**:
```
Mean:   μ = (1/n) × Σxᵢ
Median: Middle value when sorted
Mode:   Most frequent value
```

**Measures of Spread**:
```
Variance:     σ² = (1/n) × Σ(xᵢ - μ)²
Std Dev:      σ = √σ²
Range:        max - min
IQR:          Q3 - Q1
```

```python
import numpy as np
from scipy import stats

data = np.array([2, 4, 4, 4, 5, 5, 7, 9])

print(f"Mean: {np.mean(data)}")          # 5.0
print(f"Median: {np.median(data)}")      # 4.5
print(f"Mode: {stats.mode(data)[0]}")    # 4
print(f"Std Dev: {np.std(data)}")        # 2.0
print(f"Variance: {np.var(data)}")       # 4.0
```

#### 4.2 Correlation and Covariance

**Covariance**: Measure of joint variability
```
Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]
```

**Correlation**: Normalized covariance (-1 to 1)
```
ρ(X, Y) = Cov(X, Y) / (σₓ × σᵧ)
```

```python
import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

covariance = np.cov(X, Y)[0, 1]
correlation = np.corrcoef(X, Y)[0, 1]

print(f"Covariance: {covariance}")    # 1.5
print(f"Correlation: {correlation}")   # 0.82
```

---

## Part II: Intermediate Level

### Chapter 5: Advanced Linear Algebra

#### 5.1 Eigenvalues and Eigenvectors

**Definition**: For a matrix A, if:
```
Av = λv

Then:
- v is an eigenvector
- λ is the corresponding eigenvalue
```

**Significance in ML**:
- PCA uses eigenvectors of covariance matrix
- PageRank uses dominant eigenvector

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)    # [5, 2]
print("Eigenvectors:\n", eigenvectors)

# Verify: Av = λv
v = eigenvectors[:, 0]
lambda_0 = eigenvalues[0]
print("Av =", A @ v)
print("λv =", lambda_0 * v)
```

#### 5.2 Singular Value Decomposition (SVD)

**Decomposition**: Any matrix A can be written as:
```
A = U × Σ × Vᵀ

Where:
- U: Left singular vectors (m×m orthogonal)
- Σ: Singular values (m×n diagonal)
- V: Right singular vectors (n×n orthogonal)
```

**Applications**:
- Dimensionality reduction
- Matrix completion
- Image compression

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])

U, S, Vt = np.linalg.svd(A)

print("U:\n", U)
print("Singular values:", S)
print("V^T:\n", Vt)

# Reconstruct
Sigma = np.zeros((3, 2))
Sigma[:2, :2] = np.diag(S)
A_reconstructed = U @ Sigma @ Vt
print("Reconstructed:\n", A_reconstructed)
```

#### 5.3 Matrix Decompositions

| Decomposition | Form | Use Case |
|---------------|------|----------|
| **LU** | A = LU | Solving linear systems |
| **QR** | A = QR | Least squares |
| **Cholesky** | A = LLᵀ | Positive definite matrices |
| **Eigendecomposition** | A = QΛQ⁻¹ | Symmetric matrices |

---

### Chapter 6: Multivariable Calculus

#### 6.1 Gradients and Jacobians

**Gradient** (for scalar function f: Rⁿ → R):
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Jacobian** (for vector function f: Rⁿ → Rᵐ):
```
        ⎡∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ⎤
J(f) =  ⎢∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ⎥
        ⎣  ...      ...    ...    ...  ⎦
```

#### 6.2 Hessian Matrix

**Second-order partial derivatives**:
```
        ⎡∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ...⎤
H(f) =  ⎢∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ...⎥
        ⎣    ...        ...      ...⎦
```

**Use in Optimization**:
- Positive definite Hessian → local minimum
- Negative definite Hessian → local maximum
- Indefinite Hessian → saddle point

```python
import numpy as np

def hessian_numerical(f, x, epsilon=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += epsilon; x_pp[j] += epsilon
            x_pm = x.copy(); x_pm[i] += epsilon; x_pm[j] -= epsilon
            x_mp = x.copy(); x_mp[i] -= epsilon; x_mp[j] += epsilon
            x_mm = x.copy(); x_mm[i] -= epsilon; x_mm[j] -= epsilon
            
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * epsilon**2)
    
    return H
```

#### 6.3 Chain Rule for Backpropagation

**Multivariate Chain Rule**:
```
∂L/∂x = Σᵢ (∂L/∂yᵢ) × (∂yᵢ/∂x)
```

**Neural Network Example**:
```
Layer: y = σ(Wx + b)
Loss: L

∂L/∂W = ∂L/∂y × ∂y/∂(Wx+b) × ∂(Wx+b)/∂W
      = δ × σ'(Wx+b) × xᵀ
```

---

### Chapter 7: Probability Distributions

#### 7.1 Discrete Distributions

**Bernoulli**: Single binary outcome
```
P(X = 1) = p
P(X = 0) = 1 - p
E[X] = p, Var(X) = p(1-p)
```

**Binomial**: Number of successes in n trials
```
P(X = k) = C(n,k) × pᵏ × (1-p)ⁿ⁻ᵏ
E[X] = np, Var(X) = np(1-p)
```

**Poisson**: Count of rare events
```
P(X = k) = (λᵏ × e⁻λ) / k!
E[X] = λ, Var(X) = λ
```

```python
from scipy import stats
import numpy as np

# Binomial: 10 coin flips, p=0.5
binomial = stats.binom(n=10, p=0.5)
print(f"P(X=5) = {binomial.pmf(5):.4f}")  # 0.2461

# Poisson: average 3 events
poisson = stats.poisson(mu=3)
print(f"P(X=2) = {poisson.pmf(2):.4f}")   # 0.2240
```

#### 7.2 Continuous Distributions

**Normal (Gaussian)**:
```
f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
E[X] = μ, Var(X) = σ²
```

**Exponential**:
```
f(x) = λ × e⁻λx  for x ≥ 0
E[X] = 1/λ, Var(X) = 1/λ²
```

**Uniform**:
```
f(x) = 1/(b-a)  for a ≤ x ≤ b
E[X] = (a+b)/2, Var(X) = (b-a)²/12
```

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Normal distribution
normal = stats.norm(loc=0, scale=1)  # μ=0, σ=1
x = np.linspace(-4, 4, 100)
plt.plot(x, normal.pdf(x))
plt.title('Standard Normal Distribution')

# Sample from distribution
samples = normal.rvs(size=1000)
print(f"Sample mean: {samples.mean():.4f}")
print(f"Sample std: {samples.std():.4f}")
```

#### 7.3 Maximum Likelihood Estimation

**Idea**: Find parameters that maximize the probability of observed data

**Likelihood**:
```
L(θ|x₁,...,xₙ) = Π P(xᵢ|θ)
```

**Log-Likelihood** (easier to work with):
```
ℓ(θ) = Σ log P(xᵢ|θ)
```

**MLE**: Find θ that maximizes ℓ(θ)

```python
import numpy as np
from scipy import stats
from scipy.optimize import minimize

# Generate data from normal distribution
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, 100)

# Negative log-likelihood for normal distribution
def neg_log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

# MLE
result = minimize(neg_log_likelihood, x0=[0, 1], args=(data,))
mle_mu, mle_sigma = result.x

print(f"True: μ={true_mu}, σ={true_sigma}")
print(f"MLE:  μ={mle_mu:.2f}, σ={mle_sigma:.2f}")
```

---

### Chapter 8: Statistical Inference

#### 8.1 Hypothesis Testing

**Steps**:
1. State null hypothesis H₀ and alternative H₁
2. Choose significance level α (usually 0.05)
3. Calculate test statistic
4. Compare to critical value or compute p-value
5. Reject or fail to reject H₀

**Example: t-test**
```python
from scipy import stats
import numpy as np

# Two groups
group1 = np.random.normal(100, 15, 30)
group2 = np.random.normal(105, 15, 30)

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Groups are significantly different")
else:
    print("Fail to reject H₀: No significant difference")
```

#### 8.2 Confidence Intervals

**Definition**: Range of values that contains the true parameter with probability (1-α)

**For mean with known σ**:
```
CI = x̄ ± z_{α/2} × (σ/√n)
```

**For mean with unknown σ**:
```
CI = x̄ ± t_{α/2,n-1} × (s/√n)
```

```python
import numpy as np
from scipy import stats

data = np.random.normal(100, 15, 50)

mean = np.mean(data)
sem = stats.sem(data)  # Standard error of mean
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)

print(f"Sample mean: {mean:.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")
```

---

## Part III: Advanced Level

### Chapter 9: Optimization Theory

#### 9.1 Convex Optimization

**Convex Function**: f is convex if:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
```

**Properties**:
- Local minimum = Global minimum
- Gradient descent converges to global optimum

**Common Convex Functions in ML**:
- Mean Squared Error
- Cross-entropy loss
- L1 and L2 regularization

#### 9.2 Gradient Descent Variants

**Batch Gradient Descent**:
```
θ = θ - α × (1/n) × Σ∇L(θ, xᵢ, yᵢ)
```

**Stochastic Gradient Descent (SGD)**:
```
θ = θ - α × ∇L(θ, xᵢ, yᵢ)
```

**Mini-batch Gradient Descent**:
```
θ = θ - α × (1/m) × Σ∇L(θ, xⱼ, yⱼ)  for batch of size m
```

**Momentum**:
```
v = β × v + α × ∇L(θ)
θ = θ - v
```

**Adam** (Adaptive Moment Estimation):
```
m = β₁ × m + (1-β₁) × ∇L(θ)
v = β₂ × v + (1-β₂) × (∇L(θ))²
θ = θ - α × m / (√v + ε)
```

```python
import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params
```

#### 9.3 Constrained Optimization

**Lagrangian Method**:
```
Minimize f(x) subject to g(x) = 0

L(x, λ) = f(x) + λ × g(x)

∇L = 0 gives optimal solution
```

**KKT Conditions** (for inequality constraints):
```
Minimize f(x) subject to g(x) ≤ 0

1. ∇f(x*) + Σλᵢ∇gᵢ(x*) = 0
2. gᵢ(x*) ≤ 0
3. λᵢ ≥ 0
4. λᵢ × gᵢ(x*) = 0
```

---

### Chapter 10: Information Theory

#### 10.1 Entropy

**Shannon Entropy**: Measure of uncertainty
```
H(X) = -Σ P(x) × log₂P(x)
```

**Properties**:
- H(X) ≥ 0
- H(X) = 0 iff X is deterministic
- Maximum when all outcomes equally likely

```python
import numpy as np

def entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

# Fair coin: maximum entropy
print(f"Fair coin: {entropy([0.5, 0.5]):.4f}")  # 1.0

# Biased coin
print(f"Biased coin: {entropy([0.9, 0.1]):.4f}")  # 0.469
```

#### 10.2 Cross-Entropy and KL Divergence

**Cross-Entropy**:
```
H(P, Q) = -Σ P(x) × log Q(x)
```

**KL Divergence** (Relative Entropy):
```
D_KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
             = H(P, Q) - H(P)
```

**Properties**:
- D_KL ≥ 0
- D_KL = 0 iff P = Q
- Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P)

```python
import numpy as np

def cross_entropy(p, q):
    return -np.sum(p * np.log(q + 1e-10))

def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.5, 0.3, 0.2])

print(f"Cross-entropy: {cross_entropy(p, q):.4f}")
print(f"KL divergence: {kl_divergence(p, q):.4f}")
```

#### 10.3 Mutual Information

**Definition**: Information shared between two variables
```
I(X; Y) = H(X) + H(Y) - H(X, Y)
        = Σ P(x,y) × log(P(x,y) / (P(x)P(y)))
```

**Applications in ML**:
- Feature selection
- Information bottleneck
- Variational autoencoders

---

### Chapter 11: Algorithm Complexity

#### 11.1 Big-O Notation

**Common Complexities**:

| Notation | Name | Example |
|----------|------|---------|
| O(1) | Constant | Array access |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Linear search |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Bubble sort |
| O(2ⁿ) | Exponential | Subset enumeration |

#### 11.2 Space Complexity

**Memory Usage Analysis**:
```python
# O(1) space
def sum_array(arr):
    total = 0
    for x in arr:
        total += x
    return total

# O(n) space
def copy_array(arr):
    return [x for x in arr]

# O(n²) space
def create_matrix(n):
    return [[0] * n for _ in range(n)]
```

#### 11.3 ML Algorithm Complexities

| Algorithm | Training | Prediction | Space |
|-----------|----------|------------|-------|
| Linear Regression | O(nd² + d³) | O(d) | O(d²) |
| k-NN | O(1) | O(nd) | O(nd) |
| Decision Tree | O(nd log n) | O(log n) | O(nodes) |
| SVM | O(n²) to O(n³) | O(sv × d) | O(n²) |
| Neural Network | O(epochs × n × params) | O(params) | O(params) |

---

### Chapter 12: Applied Mathematics in ML

#### 12.1 Regularization Theory

**L1 Regularization** (Lasso):
```
Loss = MSE + λ × Σ|wᵢ|
```
- Promotes sparsity
- Feature selection

**L2 Regularization** (Ridge):
```
Loss = MSE + λ × Σwᵢ²
```
- Prevents large weights
- Numerical stability

**Elastic Net**:
```
Loss = MSE + λ₁ × Σ|wᵢ| + λ₂ × Σwᵢ²
```

#### 12.2 Kernel Methods

**Kernel Trick**: Map data to higher dimensions implicitly
```
K(x, y) = φ(x)ᵀφ(y)
```

**Common Kernels**:
| Kernel | Formula |
|--------|---------|
| Linear | xᵀy |
| Polynomial | (xᵀy + c)ᵈ |
| RBF (Gaussian) | exp(-γ||x-y||²) |

#### 12.3 Matrix Calculus for Deep Learning

**Key Derivatives**:
```
∂(Wx)/∂W = xᵀ
∂(Wx)/∂x = Wᵀ
∂(xᵀAx)/∂x = (A + Aᵀ)x
```

**Softmax Gradient**:
```
∂softmax(z)ᵢ/∂zⱼ = softmax(z)ᵢ × (δᵢⱼ - softmax(z)ⱼ)
```

---

## Summary

This guide covered the essential mathematical foundations for AI:

| Topic | Key Concepts | ML Applications |
|-------|--------------|-----------------|
| **Linear Algebra** | Vectors, Matrices, Eigenvalues | Neural networks, PCA |
| **Calculus** | Derivatives, Gradients | Backpropagation, Optimization |
| **Probability** | Distributions, Bayes | Bayesian learning, Generative models |
| **Statistics** | Inference, Hypothesis testing | Model evaluation |
| **Optimization** | Gradient descent, Convexity | Training algorithms |
| **Information Theory** | Entropy, KL divergence | Loss functions, VAEs |

---

**Last Updated**: 2024-01-29
