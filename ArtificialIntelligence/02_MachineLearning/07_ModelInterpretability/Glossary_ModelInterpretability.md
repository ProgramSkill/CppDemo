# Model Interpretability Glossary (模型可解释性词汇表)

## English-Chinese Technical Terms

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Interpretability** | 可解释性 | 理解模型决策的能力 |
| **Explainability** | 可解释性 | 同Interpretability |
| **Black Box** | 黑箱 | 不透明的模型 |
| **White Box** | 白箱 | 透明可理解的模型 |
| **Transparency** | 透明度 | 模型的清晰程度 |
| **Explanation** | 解释 | 对预测的说明 |

---

## 2. 解释类型 (Types of Explanations)

| English | 中文 | 说明 |
|---------|------|------|
| **Global Explanation** | 全局解释 | 整体模型行为 |
| **Local Explanation** | 局部解释 | 单个预测的解释 |
| **Model-Specific** | 模型特定 | 针对特定模型 |
| **Model-Agnostic** | 模型无关 | 适用于任何模型 |
| **Post-hoc** | 事后解释 | 训练后的解释 |
| **Intrinsic** | 内在的 | 模型本身可解释 |

---

## 3. 解释方法 (Explanation Methods)

| English | 中文 | 说明 |
|---------|------|------|
| **SHAP** | SHAP值 | Shapley加性解释 |
| **LIME** | LIME | 局部可解释模型 |
| **Feature Importance** | 特征重要性 | 特征的重要程度 |
| **Permutation Importance** | 置换重要性 | 打乱特征后的影响 |
| **Partial Dependence** | 部分依赖 | 特征与预测的关系 |
| **ICE Plot** | ICE图 | 个体条件期望 |
| **Attention** | 注意力 | 注意力权重可视化 |
| **Grad-CAM** | Grad-CAM | 梯度类激活映射 |
| **Saliency Map** | 显著图 | 重要区域可视化 |

---

## 4. 可解释模型 (Interpretable Models)

| English | 中文 | 说明 |
|---------|------|------|
| **Linear Model** | 线性模型 | 系数可解释 |
| **Logistic Regression** | 逻辑回归 | 分类的线性模型 |
| **Decision Tree** | 决策树 | 规则可解释 |
| **Rule-Based Model** | 基于规则的模型 | if-then规则 |
| **GAM** | 广义加性模型 | 可加的非线性模型 |

---

## 5. SHAP相关 (SHAP Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Shapley Value** | Shapley值 | 博弈论中的贡献值 |
| **Base Value** | 基础值 | 平均预测值 |
| **SHAP Value** | SHAP值 | 特征的贡献 |
| **Summary Plot** | 汇总图 | 全局SHAP可视化 |
| **Force Plot** | 力图 | 单个预测的SHAP |
| **Waterfall Plot** | 瀑布图 | 特征贡献分解 |
| **Dependence Plot** | 依赖图 | 特征交互可视化 |

---

## 6. 公平性与偏见 (Fairness and Bias)

| English | 中文 | 说明 |
|---------|------|------|
| **Fairness** | 公平性 | 模型对不同群体的公平 |
| **Bias** | 偏见 | 模型的不公平倾向 |
| **Discrimination** | 歧视 | 基于敏感属性的差异 |
| **Protected Attribute** | 受保护属性 | 敏感特征如性别、种族 |
| **Demographic Parity** | 人口统计均等 | 各组预测率相等 |
| **Equalized Odds** | 均等化概率 | 各组TPR/FPR相等 |
| **Disparate Impact** | 差异影响 | 不同群体的影响差异 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **SHAP** | SHapley Additive exPlanations | Shapley加性解释 |
| **LIME** | Local Interpretable Model-agnostic Explanations | 局部可解释模型无关解释 |
| **PDP** | Partial Dependence Plot | 部分依赖图 |
| **ICE** | Individual Conditional Expectation | 个体条件期望 |
| **TCAV** | Testing with Concept Activation Vectors | 概念激活向量测试 |
| **XAI** | Explainable AI | 可解释人工智能 |

---

**最后更新**: 2024-01-29
