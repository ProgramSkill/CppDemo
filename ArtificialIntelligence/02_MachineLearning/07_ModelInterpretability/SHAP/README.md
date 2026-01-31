# SHAP (SHapley Additive exPlanations)

基于博弈论的模型解释方法。

## 📚 内容概览

| 主题 | 描述 | 难度 |
|------|------|------|
| Shapley值 | 博弈论基础 | ⭐⭐ |
| TreeSHAP | 树模型专用 | ⭐⭐ |
| DeepSHAP | 深度学习 | ⭐⭐⭐ |
| KernelSHAP | 模型无关 | ⭐⭐⭐ |

## 💡 核心思想

```
预测值 = 基础值 + Σ SHAP值

每个SHAP值表示该特征对预测的贡献
正值 → 增加预测值
负值 → 降低预测值
```
