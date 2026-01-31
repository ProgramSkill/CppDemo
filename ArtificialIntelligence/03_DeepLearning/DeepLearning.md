# Deep Learning: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: Neural Network Fundamentals](#chapter-1-neural-network-fundamentals)
  - [Chapter 2: Activation Functions](#chapter-2-activation-functions)
  - [Chapter 3: Backpropagation](#chapter-3-backpropagation)
  - [Chapter 4: Training Neural Networks](#chapter-4-training-neural-networks)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 5: Convolutional Neural Networks](#chapter-5-convolutional-neural-networks)
  - [Chapter 6: Recurrent Neural Networks](#chapter-6-recurrent-neural-networks)
  - [Chapter 7: Regularization Techniques](#chapter-7-regularization-techniques)
  - [Chapter 8: Optimization Algorithms](#chapter-8-optimization-algorithms)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 9: Attention and Transformers](#chapter-9-attention-and-transformers)
  - [Chapter 10: Generative Models](#chapter-10-generative-models)
  - [Chapter 11: Advanced Architectures](#chapter-11-advanced-architectures)
  - [Chapter 12: Practical Deep Learning](#chapter-12-practical-deep-learning)

---

## Introduction

**Deep Learning** is a subset of machine learning using neural networks with multiple layers to learn hierarchical representations of data.

### Why Deep Learning?

| Advantage | Description |
|-----------|-------------|
| **Automatic Feature Learning** | No manual feature engineering |
| **Scalability** | Performance improves with data |
| **State-of-the-Art** | Best results in vision, NLP, speech |
| **Transfer Learning** | Reuse pretrained models |

### Deep Learning vs Traditional ML

| Aspect | Traditional ML | Deep Learning |
|--------|---------------|---------------|
| Features | Manual engineering | Automatic learning |
| Data requirement | Small to medium | Large |
| Compute | CPU sufficient | GPU/TPU needed |
| Interpretability | Often clear | Often opaque |

---

## Part I: Beginner Level

### Chapter 1: Neural Network Fundamentals

#### 1.1 The Perceptron

```
Inputs: x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ
Bias: b

Output: y = activation(Σᵢ wᵢxᵢ + b)
```

```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
    
    def forward(self, x):
        return 1 if np.dot(self.weights, x) + self.bias > 0 else 0
```

#### 1.2 Multi-Layer Perceptron (MLP)

```
Input Layer → Hidden Layer(s) → Output Layer

Each connection has a weight
Each neuron has a bias
```

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# Example
model = MLP(784, 256, 10)  # For MNIST
```

#### 1.3 Forward Propagation

```python
def forward_pass(X, weights, biases):
    """Simple forward pass through layers"""
    activations = [X]
    
    for W, b in zip(weights, biases):
        z = activations[-1] @ W + b  # Linear transformation
        a = relu(z)  # Activation
        activations.append(a)
    
    return activations
```

---

### Chapter 2: Activation Functions

#### 2.1 Common Activation Functions

| Function | Formula | Range | Use Case |
|----------|---------|-------|----------|
| **Sigmoid** | 1/(1+e⁻ˣ) | (0, 1) | Binary classification output |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Hidden layers |
| **ReLU** | max(0, x) | [0, ∞) | Most hidden layers |
| **Leaky ReLU** | max(αx, x) | (-∞, ∞) | Avoiding dead neurons |
| **Softmax** | eˣⁱ/Σeˣʲ | (0, 1) | Multi-class output |

```python
import torch.nn.functional as F

# Implementations
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / exp_x.sum()

# In PyTorch
x = torch.randn(10)
F.relu(x)
F.sigmoid(x)
F.softmax(x, dim=0)
```

#### 2.2 Choosing Activation Functions

```
Hidden Layers: ReLU (default), Leaky ReLU, GELU
Output Layer:
  - Binary classification: Sigmoid
  - Multi-class: Softmax
  - Regression: Linear (no activation)
```

---

### Chapter 3: Backpropagation

#### 3.1 The Chain Rule

```
∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w

Where:
- L: Loss
- y: Output
- z: Pre-activation (weighted sum)
- w: Weight
```

#### 3.2 Gradient Computation

```python
def backward_pass(activations, weights, y_true):
    """Compute gradients via backpropagation"""
    gradients = []
    
    # Output layer gradient
    delta = activations[-1] - y_true  # For MSE loss
    
    for i in range(len(weights) - 1, -1, -1):
        # Gradient for weights
        grad_W = activations[i].T @ delta
        gradients.insert(0, grad_W)
        
        # Propagate error backwards
        if i > 0:
            delta = (delta @ weights[i].T) * relu_derivative(activations[i])
    
    return gradients
```

#### 3.3 Automatic Differentiation in PyTorch

```python
import torch

# Forward pass
x = torch.randn(10, requires_grad=True)
y = x ** 2 + 2 * x
loss = y.sum()

# Backward pass (automatic!)
loss.backward()

# Gradients
print(x.grad)  # dy/dx = 2x + 2
```

---

### Chapter 4: Training Neural Networks

#### 4.1 Loss Functions

```python
import torch.nn as nn

# Regression
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()

# Classification
bce_loss = nn.BCELoss()  # Binary
ce_loss = nn.CrossEntropyLoss()  # Multi-class

# Example
predictions = model(X)
loss = ce_loss(predictions, labels)
```

#### 4.2 Training Loop

```python
def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

#### 4.3 Evaluation

```python
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x)
            _, predicted = torch.max(predictions, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return correct / total
```

---

## Part II: Intermediate Level

### Chapter 5: Convolutional Neural Networks

#### 5.1 Convolution Operation

```
Input: Image (H × W × C)
Kernel: Filter (k × k × C)
Output: Feature Map

output[i,j] = Σ input[i+m, j+n] × kernel[m, n]
```

#### 5.2 CNN Architecture

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
```

#### 5.3 Key CNN Components

| Component | Purpose |
|-----------|---------|
| **Convolution** | Feature extraction |
| **Pooling** | Downsampling, translation invariance |
| **Batch Norm** | Stabilize training |
| **Dropout** | Regularization |

---

### Chapter 6: Recurrent Neural Networks

#### 6.1 RNN Basics

```
hₜ = tanh(Wₓₕ × xₜ + Wₕₕ × hₜ₋₁ + b)
yₜ = Wₕᵧ × hₜ

Hidden state carries information across time steps
```

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, hidden = self.rnn(x)
        # Use last hidden state
        return self.fc(hidden.squeeze(0))
```

#### 6.2 LSTM (Long Short-Term Memory)

```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                           num_layers=2, 
                           dropout=0.5,
                           batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)
```

#### 6.3 GRU (Gated Recurrent Unit)

```python
self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
```

---

### Chapter 7: Regularization Techniques

#### 7.1 Dropout

```python
class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        return self.fc2(x)
```

#### 7.2 Batch Normalization

```python
self.bn = nn.BatchNorm1d(256)  # For 1D
self.bn2d = nn.BatchNorm2d(64)  # For 2D (CNNs)

# Usage
x = self.bn(F.relu(self.fc1(x)))
```

#### 7.3 Weight Decay (L2 Regularization)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

#### 7.4 Early Stopping

```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

---

### Chapter 8: Optimization Algorithms

#### 8.1 Optimizer Comparison

| Optimizer | Formula | Use Case |
|-----------|---------|----------|
| **SGD** | θ = θ - α∇L | Simple, interpretable |
| **Momentum** | v = βv + α∇L, θ = θ - v | Faster convergence |
| **Adam** | Adaptive learning rate | Default choice |
| **AdamW** | Adam + decoupled weight decay | Best for Transformers |

#### 8.2 Learning Rate Scheduling

```python
# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# One cycle
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, epochs=10, steps_per_epoch=len(train_loader)
)

# Training loop
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

---

## Part III: Advanced Level

### Chapter 9: Attention and Transformers

#### 9.1 Self-Attention

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention = F.softmax(scores, dim=-1)
        
        return torch.matmul(attention, V)
```

#### 9.2 Transformer Architecture

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
```

---

### Chapter 10: Generative Models

#### 10.1 Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
```

#### 10.2 GAN (Generative Adversarial Network)

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
```

---

### Chapter 11: Advanced Architectures

#### 11.1 ResNet (Residual Networks)

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        return F.relu(out)
```

#### 11.2 U-Net (for Segmentation)

```python
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        # Decoder (upsampling)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, 1)
    
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # Decoder with skip connections
        d1 = self.up(e2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)
```

---

### Chapter 12: Practical Deep Learning

#### 12.1 Transfer Learning

```python
from torchvision import models

# Load pretrained model
model = models.resnet50(pretrained=True)

# Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

#### 12.2 Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 12.3 Model Deployment

```python
# Save model
torch.save(model.state_dict(), 'model.pt')

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, 'model.onnx')

# TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

---

## Summary

| Architecture | Best For | Key Innovation |
|--------------|----------|----------------|
| **MLP** | Tabular data | Universal approximator |
| **CNN** | Images | Local feature learning |
| **RNN/LSTM** | Sequences | Memory over time |
| **Transformer** | All modalities | Self-attention |
| **ResNet** | Deep networks | Skip connections |
| **GAN** | Generation | Adversarial training |

---

**Last Updated**: 2024-01-29
