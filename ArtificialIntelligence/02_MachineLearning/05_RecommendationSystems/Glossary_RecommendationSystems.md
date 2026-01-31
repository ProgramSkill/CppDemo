# Recommendation Systems Glossary (推荐系统词汇表)

## English-Chinese Technical Terms

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Recommendation System** | 推荐系统 | 预测用户偏好的系统 |
| **User** | 用户 | 系统的使用者 |
| **Item** | 物品 | 被推荐的对象 |
| **Rating** | 评分 | 用户对物品的评价 |
| **Interaction** | 交互 | 用户与物品的互动 |
| **User-Item Matrix** | 用户-物品矩阵 | 评分/交互矩阵 |
| **Sparsity** | 稀疏性 | 矩阵中缺失值的比例 |

---

## 2. 过滤方法 (Filtering Methods)

| English | 中文 | 说明 |
|---------|------|------|
| **Content-Based Filtering** | 基于内容的过滤 | 基于物品特征推荐 |
| **Collaborative Filtering** | 协同过滤 | 基于用户行为推荐 |
| **Hybrid Filtering** | 混合过滤 | 结合多种方法 |
| **User-Based CF** | 基于用户的CF | 相似用户推荐 |
| **Item-Based CF** | 基于物品的CF | 相似物品推荐 |
| **Memory-Based** | 基于内存的 | 使用全部数据 |
| **Model-Based** | 基于模型的 | 学习预测模型 |

---

## 3. 矩阵分解 (Matrix Factorization)

| English | 中文 | 说明 |
|---------|------|------|
| **Matrix Factorization** | 矩阵分解 | 分解评分矩阵 |
| **Latent Factor** | 潜在因子 | 隐藏的特征维度 |
| **SVD** | 奇异值分解 | 矩阵分解方法 |
| **ALS** | 交替最小二乘 | 优化算法 |
| **NMF** | 非负矩阵分解 | 非负约束的分解 |
| **Embedding** | 嵌入 | 低维向量表示 |
| **User Embedding** | 用户嵌入 | 用户的向量表示 |
| **Item Embedding** | 物品嵌入 | 物品的向量表示 |

---

## 4. 评估指标 (Evaluation Metrics)

| English | 中文 | 说明 |
|---------|------|------|
| **RMSE** | 均方根误差 | 预测评分误差 |
| **MAE** | 平均绝对误差 | 预测误差 |
| **Precision@K** | K处精确率 | top-K相关比例 |
| **Recall@K** | K处召回率 | 相关中被推荐的比例 |
| **NDCG** | 归一化折损累积增益 | 排序质量 |
| **MAP** | 平均精确率均值 | 平均排序质量 |
| **Hit Rate** | 命中率 | 推荐命中比例 |
| **Coverage** | 覆盖率 | 被推荐物品的比例 |
| **Diversity** | 多样性 | 推荐的多样程度 |

---

## 5. 深度学习方法 (Deep Learning Methods)

| English | 中文 | 说明 |
|---------|------|------|
| **Neural Collaborative Filtering** | 神经协同过滤 | 深度学习CF |
| **NCF** | NCF | 神经协同过滤缩写 |
| **Deep Matrix Factorization** | 深度矩阵分解 | 深度学习MF |
| **Two-Tower Model** | 双塔模型 | 分别编码用户和物品 |
| **Wide & Deep** | 宽深模型 | 结合记忆和泛化 |
| **AutoRec** | AutoRec | 自编码器推荐 |
| **Attention** | 注意力 | 注意力机制 |
| **Transformer** | Transformer | 自注意力架构 |

---

## 6. 序列推荐 (Sequential Recommendation)

| English | 中文 | 说明 |
|---------|------|------|
| **Sequential Recommendation** | 序列推荐 | 考虑时序的推荐 |
| **Session-Based** | 基于会话的 | 单次会话内推荐 |
| **Next-Item Prediction** | 下一物品预测 | 预测下一个交互 |
| **User History** | 用户历史 | 用户的行为序列 |
| **Temporal Dynamics** | 时序动态 | 随时间变化的模式 |

---

## 7. 系统问题 (System Issues)

| English | 中文 | 说明 |
|---------|------|------|
| **Cold Start** | 冷启动 | 新用户/物品问题 |
| **Data Sparsity** | 数据稀疏 | 评分数据稀少 |
| **Scalability** | 可扩展性 | 大规模数据处理 |
| **Filter Bubble** | 过滤气泡 | 信息茧房 |
| **Popularity Bias** | 流行偏差 | 偏向热门物品 |
| **Candidate Generation** | 候选生成 | 第一阶段检索 |
| **Ranking** | 排序 | 第二阶段精排 |
| **Real-Time** | 实时 | 实时推荐 |
| **Batch** | 批处理 | 离线批量计算 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **CF** | Collaborative Filtering | 协同过滤 |
| **CB** | Content-Based | 基于内容 |
| **MF** | Matrix Factorization | 矩阵分解 |
| **SVD** | Singular Value Decomposition | 奇异值分解 |
| **ALS** | Alternating Least Squares | 交替最小二乘 |
| **NCF** | Neural Collaborative Filtering | 神经协同过滤 |
| **NDCG** | Normalized Discounted Cumulative Gain | 归一化折损累积增益 |

---

**最后更新**: 2024-01-29
