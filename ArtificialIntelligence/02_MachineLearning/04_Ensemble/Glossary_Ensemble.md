# Ensemble Learning Glossary (集成学习词汇表)

## English-Chinese Technical Terms

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Ensemble Learning** | 集成学习 | 组合多个模型 |
| **Base Learner** | 基学习器 | 集成中的单个模型 |
| **Weak Learner** | 弱学习器 | 略好于随机的模型 |
| **Strong Learner** | 强学习器 | 高性能模型 |
| **Model Combination** | 模型组合 | 合并多个模型 |
| **Diversity** | 多样性 | 模型间的差异 |

---

## 2. Bagging相关 (Bagging Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Bagging** | 装袋法 | Bootstrap Aggregating |
| **Bootstrap Sampling** | 自助采样 | 有放回抽样 |
| **Aggregation** | 聚合 | 合并预测结果 |
| **Random Forest** | 随机森林 | Bagging + 随机特征 |
| **Out-of-Bag (OOB)** | 袋外数据 | 未被采样的数据 |
| **OOB Error** | 袋外误差 | OOB数据上的误差 |
| **Feature Bagging** | 特征装袋 | 随机选择特征子集 |

---

## 3. Boosting相关 (Boosting Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Boosting** | 提升法 | 顺序训练，关注错误 |
| **AdaBoost** | 自适应提升 | Adaptive Boosting |
| **Gradient Boosting** | 梯度提升 | 基于梯度的提升 |
| **XGBoost** | XGBoost | 极端梯度提升 |
| **LightGBM** | LightGBM | 轻量梯度提升机 |
| **CatBoost** | CatBoost | 类别特征优化的提升 |
| **Sample Weight** | 样本权重 | 样本的重要性 |
| **Learning Rate** | 学习率 | 步长/收缩率 |
| **Shrinkage** | 收缩 | 控制每棵树的贡献 |
| **Residual** | 残差 | 预测与真实的差 |
| **Pseudo-Residual** | 伪残差 | 损失函数的负梯度 |

---

## 4. Stacking相关 (Stacking Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Stacking** | 堆叠法 | 模型堆叠 |
| **Stacked Generalization** | 堆叠泛化 | 分层组合模型 |
| **Meta-Learner** | 元学习器 | 学习基学习器输出的模型 |
| **Meta-Features** | 元特征 | 基学习器的预测 |
| **Blending** | 混合法 | 简化的堆叠方法 |
| **Level-0 Models** | 0层模型 | 基础模型 |
| **Level-1 Model** | 1层模型 | 元模型 |

---

## 5. 投票方法 (Voting Methods)

| English | 中文 | 说明 |
|---------|------|------|
| **Voting** | 投票 | 组合分类器 |
| **Hard Voting** | 硬投票 | 多数表决 |
| **Soft Voting** | 软投票 | 概率平均 |
| **Weighted Voting** | 加权投票 | 按权重投票 |
| **Averaging** | 平均法 | 预测值平均 |
| **Weighted Average** | 加权平均 | 加权预测平均 |

---

## 6. 树相关 (Tree Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Decision Tree** | 决策树 | 树形分类器 |
| **Decision Stump** | 决策桩 | 深度为1的树 |
| **Tree Depth** | 树深度 | 树的最大层数 |
| **Leaf Node** | 叶节点 | 终端节点 |
| **Split** | 分裂 | 节点划分 |
| **Pruning** | 剪枝 | 减少树的复杂度 |
| **Level-wise Growth** | 层级生长 | 按层生长树 |
| **Leaf-wise Growth** | 叶子生长 | 按叶子生长树 |

---

## 7. 正则化 (Regularization)

| English | 中文 | 说明 |
|---------|------|------|
| **Regularization** | 正则化 | 防止过拟合 |
| **L1 Regularization** | L1正则化 | 绝对值惩罚 |
| **L2 Regularization** | L2正则化 | 平方惩罚 |
| **Early Stopping** | 早停 | 提前停止训练 |
| **Subsampling** | 子采样 | 使用数据子集 |
| **Column Subsampling** | 列子采样 | 使用特征子集 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **RF** | Random Forest | 随机森林 |
| **GBDT** | Gradient Boosting Decision Tree | 梯度提升决策树 |
| **GBM** | Gradient Boosting Machine | 梯度提升机 |
| **XGB** | XGBoost | 极端梯度提升 |
| **LGB** | LightGBM | 轻量梯度提升机 |
| **OOB** | Out-of-Bag | 袋外 |

---

**最后更新**: 2024-01-29
