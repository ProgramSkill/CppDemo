# Model Interpretability: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: Why Interpretability?](#chapter-1-why-interpretability)
  - [Chapter 2: Intrinsic Interpretability](#chapter-2-intrinsic-interpretability)
  - [Chapter 3: Feature Importance](#chapter-3-feature-importance)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: SHAP Values](#chapter-4-shap-values)
  - [Chapter 5: LIME](#chapter-5-lime)
  - [Chapter 6: Partial Dependence Plots](#chapter-6-partial-dependence-plots)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: Attention Visualization](#chapter-7-attention-visualization)
  - [Chapter 8: Concept-Based Explanations](#chapter-8-concept-based-explanations)
  - [Chapter 9: Responsible AI](#chapter-9-responsible-ai)

---

## Introduction

**Model Interpretability** is the ability to explain or present model decisions in understandable terms to humans.

### Why It Matters

| Reason | Description |
|--------|-------------|
| **Trust** | Users need to trust predictions |
| **Debugging** | Find and fix model errors |
| **Compliance** | Regulations require explanations (GDPR) |
| **Fairness** | Detect and mitigate bias |
| **Improvement** | Understand what model learns |

### Interpretability vs Accuracy Trade-off

```
Interpretability ◄──────────────────────► Accuracy
    Linear       Decision    Random     Neural
   Regression    Trees       Forest     Networks
```

---

## Part I: Beginner Level

### Chapter 1: Why Interpretability?

#### 1.1 Types of Explanations

| Type | Question Answered |
|------|-------------------|
| **Global** | How does the model work overall? |
| **Local** | Why this prediction for this instance? |

#### 1.2 Model-Specific vs Model-Agnostic

| Approach | Description | Examples |
|----------|-------------|----------|
| **Model-Specific** | Built into model | Tree rules, Coefficients |
| **Model-Agnostic** | Works with any model | SHAP, LIME |

---

### Chapter 2: Intrinsic Interpretability

#### 2.1 Linear Models

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Train model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Interpret coefficients
coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print("Feature importance (by coefficient magnitude):")
print(coefficients)

# For classification:
# - Positive coefficient → increases probability of class 1
# - Larger magnitude → stronger influence
```

#### 2.2 Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Train
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)

# Visualize
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# Extract rules
from sklearn.tree import export_text
rules = export_text(tree, feature_names=feature_names)
print(rules)
```

#### 2.3 Rule-Based Models

```python
# Extract rules from decision tree
def get_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    rules = []
    
    def recurse(node, rule):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], rule + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], rule + [f"{name} > {threshold:.2f}"])
        else:
            rules.append(" AND ".join(rule))
    
    recurse(0, [])
    return rules
```

---

### Chapter 3: Feature Importance

#### 3.1 Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Calculate importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Plot
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance_mean'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
```

#### 3.2 Tree-Based Importance

```python
from sklearn.ensemble import RandomForestClassifier

# Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Built-in importance (Gini/MDI)
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

---

## Part II: Intermediate Level

### Chapter 4: SHAP Values

#### 4.1 Concept

**SHAP (SHapley Additive exPlanations)**: Uses game theory to assign importance to each feature.

```
Prediction = Base Value + Σ SHAP values

Each SHAP value = contribution of that feature
```

#### 4.2 Implementation

```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot (global)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Force plot (local - single prediction)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Dependence plot (feature interaction)
shap.dependence_plot("feature_name", shap_values, X_test)

# Waterfall plot (single prediction breakdown)
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=feature_names
))
```

#### 4.3 SHAP for Different Models

```python
# Tree models
explainer = shap.TreeExplainer(tree_model)

# Linear models
explainer = shap.LinearExplainer(linear_model, X_train)

# Deep learning
explainer = shap.DeepExplainer(deep_model, X_train[:100])

# Any model (slower)
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
```

---

### Chapter 5: LIME

#### 5.1 Concept

**LIME (Local Interpretable Model-agnostic Explanations)**: Explains individual predictions by fitting a simple model locally.

**How it works**:
1. Generate perturbed samples around instance
2. Get predictions for perturbed samples
3. Fit simple model (e.g., linear) weighted by proximity
4. Interpret simple model

#### 5.2 Implementation

```python
from lime import lime_tabular

# Create explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

# Explain single prediction
exp = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=10
)

# Show explanation
exp.show_in_notebook()
# or
exp.as_list()  # [(feature, contribution), ...]
```

#### 5.3 LIME for Text

```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['negative', 'positive'])

exp = explainer.explain_instance(
    text_instance,
    classifier.predict_proba,
    num_features=10
)
exp.show_in_notebook()
```

---

### Chapter 6: Partial Dependence Plots

#### 6.1 Concept

**PDP**: Shows the average effect of a feature on predictions, marginalizing over other features.

```python
from sklearn.inspection import PartialDependenceDisplay

# Single feature
PartialDependenceDisplay.from_estimator(model, X_train, ['feature_name'])
plt.show()

# Multiple features
PartialDependenceDisplay.from_estimator(
    model, X_train, 
    ['feature1', 'feature2', ('feature1', 'feature2')]  # Including interaction
)
plt.show()
```

#### 6.2 Individual Conditional Expectation (ICE)

```python
# ICE plots - one line per instance
PartialDependenceDisplay.from_estimator(
    model, X_train, ['feature_name'],
    kind='both'  # Shows both PDP and ICE
)
plt.show()
```

---

## Part III: Advanced Level

### Chapter 7: Attention Visualization

#### 7.1 Attention Weights in Transformers

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Get attention weights
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

attention = outputs.attentions  # Tuple of (batch, heads, seq, seq)

# Visualize attention
import seaborn as sns

tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
attention_matrix = attention[-1][0].mean(dim=0).detach().numpy()  # Last layer, avg heads

plt.figure(figsize=(10, 10))
sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens)
plt.title('Attention Weights')
plt.show()
```

#### 7.2 Grad-CAM for CNNs

```python
import torch
import torch.nn.functional as F

def grad_cam(model, image, target_layer):
    # Forward pass
    features = []
    gradients = []
    
    def forward_hook(module, input, output):
        features.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    # Register hooks
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    
    # Forward and backward
    output = model(image)
    output[0, output.argmax()].backward()
    
    # Compute Grad-CAM
    pooled_gradients = gradients[0].mean(dim=[0, 2, 3])
    activation = features[0].squeeze()
    
    for i in range(len(pooled_gradients)):
        activation[i] *= pooled_gradients[i]
    
    heatmap = activation.mean(dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0) / heatmap.max()
    
    return heatmap
```

---

### Chapter 8: Concept-Based Explanations

#### 8.1 Testing with Concept Activation Vectors (TCAV)

Explain predictions in terms of human-understandable concepts:

```python
# Conceptual example
# Instead of "pixel 123 contributes X to prediction"
# We get "stripedness contributes to tiger classification"

# Define concept examples
striped_examples = [images_of_striped_things]
not_striped_examples = [images_without_stripes]

# Train concept classifier
concept_model.fit(striped_examples + not_striped_examples, 
                  [1]*len(striped_examples) + [0]*len(not_striped_examples))

# Compute directional derivatives
# to see how much "stripedness" affects "tiger" classification
```

---

### Chapter 9: Responsible AI

#### 9.1 Fairness Metrics

```python
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate

# Compute metrics by group
metric_frame = MetricFrame(
    metrics={
        'selection_rate': selection_rate,
        'accuracy': accuracy_score,
        'fpr': false_positive_rate
    },
    y_true=y_test,
    y_pred=predictions,
    sensitive_features=sensitive_attribute
)

print(metric_frame.by_group)
print(f"Disparity: {metric_frame.difference()}")
```

#### 9.2 Model Cards

Document your model:

```markdown
## Model Card

### Model Details
- Developer: [Name]
- Model type: Random Forest Classifier
- Training data: [Description]

### Intended Use
- Primary use: [Use case]
- Out-of-scope: [What not to use for]

### Metrics
- Accuracy: 0.85
- False Positive Rate: 0.10

### Limitations
- [Known limitations]

### Ethical Considerations
- [Bias analysis results]
```

---

## Summary

| Method | Scope | Model-Agnostic | Complexity |
|--------|-------|----------------|------------|
| Coefficients | Global | No | Low |
| Tree Rules | Global/Local | No | Low |
| Permutation | Global | Yes | Medium |
| SHAP | Both | Yes | Medium |
| LIME | Local | Yes | Medium |
| PDP/ICE | Global | Yes | Low |
| Attention | Local | No | High |

---

**Last Updated**: 2024-01-29
