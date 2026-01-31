# Unsupervised Learning Glossary (无监督学习词汇表)

## English-Chinese Technical Terms by Category

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Unsupervised Learning** | 无监督学习 | 从无标签数据学习 |
| **Unlabeled Data** | 无标签数据 | 没有目标变量的数据 |
| **Pattern Discovery** | 模式发现 | 发现数据中的规律 |
| **Clustering** | 聚类 | 将相似数据分组 |
| **Dimensionality Reduction** | 降维 | 减少特征数量 |
| **Association Rule** | 关联规则 | 项目间的关系 |
| **Anomaly Detection** | 异常检测 | 发现异常数据点 |

---

## 2. 聚类算法 (Clustering Algorithms)

| English | 中文 | 说明 |
|---------|------|------|
| **Cluster** | 簇/聚类 | 相似数据点的组 |
| **Centroid** | 质心 | 簇的中心点 |
| **K-Means** | K均值 | 基于距离的聚类算法 |
| **K-Medoids** | K中心点 | 使用实际数据点作为中心 |
| **Hierarchical Clustering** | 层次聚类 | 构建聚类树 |
| **Agglomerative** | 凝聚型 | 自底向上合并 |
| **Divisive** | 分裂型 | 自顶向下分割 |
| **Dendrogram** | 树状图 | 层次聚类的可视化 |
| **Linkage** | 链接方法 | 簇间距离的计算方式 |
| **Single Linkage** | 单链接 | 最小距离 |
| **Complete Linkage** | 完全链接 | 最大距离 |
| **Average Linkage** | 平均链接 | 平均距离 |
| **Ward's Method** | Ward方法 | 最小化方差增量 |
| **DBSCAN** | 密度聚类 | 基于密度的聚类 |
| **Core Point** | 核心点 | 邻域内有足够点 |
| **Border Point** | 边界点 | 核心点邻域内的点 |
| **Noise Point** | 噪声点 | 不属于任何簇的点 |
| **Epsilon (eps)** | 邻域半径 | DBSCAN的距离参数 |
| **Min Samples** | 最小样本数 | 核心点的阈值 |
| **Spectral Clustering** | 谱聚类 | 基于图的聚类 |
| **Gaussian Mixture Model** | 高斯混合模型 | 概率聚类模型 |
| **Soft Clustering** | 软聚类 | 概率性簇分配 |
| **Hard Clustering** | 硬聚类 | 确定性簇分配 |

---

## 3. 聚类评估 (Clustering Evaluation)

| English | 中文 | 说明 |
|---------|------|------|
| **Inertia** | 惯性 | 簇内平方和 |
| **WCSS** | 簇内平方和 | Within-Cluster Sum of Squares |
| **Silhouette Score** | 轮廓系数 | 聚类质量度量 |
| **Elbow Method** | 肘部法则 | 选择K值的方法 |
| **Davies-Bouldin Index** | DB指数 | 簇间分离度 |
| **Calinski-Harabasz Index** | CH指数 | 方差比准则 |
| **Adjusted Rand Index** | 调整兰德指数 | 与真实标签比较 |
| **Normalized Mutual Information** | 标准化互信息 | 聚类一致性度量 |

---

## 4. 降维方法 (Dimensionality Reduction)

| English | 中文 | 说明 |
|---------|------|------|
| **Dimensionality Reduction** | 降维 | 减少特征维度 |
| **Feature Extraction** | 特征提取 | 创建新特征 |
| **Feature Selection** | 特征选择 | 选择原始特征子集 |
| **Principal Component Analysis** | 主成分分析 | 线性降维方法 |
| **PCA** | PCA | 主成分分析缩写 |
| **Principal Component** | 主成分 | 最大方差方向 |
| **Explained Variance** | 解释方差 | 保留的信息量 |
| **Variance Ratio** | 方差比 | 各成分解释的方差比例 |
| **Covariance Matrix** | 协方差矩阵 | 特征间关系矩阵 |
| **t-SNE** | t-SNE | 非线性降维可视化 |
| **Perplexity** | 困惑度 | t-SNE的参数 |
| **UMAP** | UMAP | 统一流形近似 |
| **Manifold Learning** | 流形学习 | 非线性降维 |
| **Linear Discriminant Analysis** | 线性判别分析 | 监督降维方法 |
| **LDA** | LDA | 线性判别分析缩写 |
| **Kernel PCA** | 核PCA | 非线性PCA |
| **Autoencoder** | 自编码器 | 神经网络降维 |
| **Latent Space** | 潜在空间 | 压缩表示空间 |
| **Reconstruction Error** | 重构误差 | 原始与重构的差异 |

---

## 5. 关联规则 (Association Rules)

| English | 中文 | 说明 |
|---------|------|------|
| **Association Rule** | 关联规则 | 项目间的关系 |
| **Itemset** | 项集 | 项目的集合 |
| **Frequent Itemset** | 频繁项集 | 出现频率高的项集 |
| **Support** | 支持度 | 项集出现的频率 |
| **Confidence** | 置信度 | 规则的可靠性 |
| **Lift** | 提升度 | 规则的相关性 |
| **Apriori Algorithm** | Apriori算法 | 频繁项集挖掘算法 |
| **FP-Growth** | FP增长 | 高效频繁模式挖掘 |
| **Market Basket Analysis** | 购物篮分析 | 关联规则应用 |
| **Antecedent** | 前件 | 规则的条件部分 |
| **Consequent** | 后件 | 规则的结论部分 |

---

## 6. 异常检测 (Anomaly Detection)

| English | 中文 | 说明 |
|---------|------|------|
| **Anomaly** | 异常 | 异常数据点 |
| **Outlier** | 离群点 | 偏离正常的点 |
| **Novelty Detection** | 新奇检测 | 检测新类型数据 |
| **Isolation Forest** | 孤立森林 | 基于随机树的异常检测 |
| **Local Outlier Factor** | 局部离群因子 | 基于密度的异常检测 |
| **One-Class SVM** | 单类SVM | 单类分类方法 |
| **Contamination** | 污染比例 | 异常点的预期比例 |
| **Anomaly Score** | 异常分数 | 数据点的异常程度 |

---

## 7. 高斯混合模型 (Gaussian Mixture Models)

| English | 中文 | 说明 |
|---------|------|------|
| **Gaussian Mixture Model** | 高斯混合模型 | 概率生成模型 |
| **GMM** | GMM | 高斯混合模型缩写 |
| **Mixture Component** | 混合成分 | 单个高斯分布 |
| **Mixing Coefficient** | 混合系数 | 成分的权重 |
| **Expectation-Maximization** | 期望最大化 | GMM的训练算法 |
| **EM Algorithm** | EM算法 | 期望最大化算法 |
| **E-Step** | E步 | 计算期望 |
| **M-Step** | M步 | 最大化参数 |
| **Responsibility** | 责任度 | 成分对数据点的贡献 |
| **Covariance Type** | 协方差类型 | full, tied, diag, spherical |
| **BIC** | 贝叶斯信息准则 | 模型选择准则 |
| **AIC** | 赤池信息准则 | 模型选择准则 |

---

## 8. 自编码器 (Autoencoders)

| English | 中文 | 说明 |
|---------|------|------|
| **Autoencoder** | 自编码器 | 无监督神经网络 |
| **Encoder** | 编码器 | 压缩输入 |
| **Decoder** | 解码器 | 重构输入 |
| **Bottleneck** | 瓶颈层 | 最小维度层 |
| **Latent Representation** | 潜在表示 | 压缩后的表示 |
| **Variational Autoencoder** | 变分自编码器 | 生成式自编码器 |
| **VAE** | VAE | 变分自编码器缩写 |
| **Denoising Autoencoder** | 去噪自编码器 | 学习去除噪声 |
| **Sparse Autoencoder** | 稀疏自编码器 | 稀疏表示学习 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **PCA** | Principal Component Analysis | 主成分分析 |
| **t-SNE** | t-Distributed Stochastic Neighbor Embedding | t分布随机邻域嵌入 |
| **UMAP** | Uniform Manifold Approximation and Projection | 统一流形近似与投影 |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise | 基于密度的噪声应用空间聚类 |
| **GMM** | Gaussian Mixture Model | 高斯混合模型 |
| **EM** | Expectation-Maximization | 期望最大化 |
| **LOF** | Local Outlier Factor | 局部离群因子 |
| **VAE** | Variational Autoencoder | 变分自编码器 |
| **NMF** | Non-negative Matrix Factorization | 非负矩阵分解 |
| **ICA** | Independent Component Analysis | 独立成分分析 |

---

## 使用说明 (Usage Guide)

### 学习路径建议

1. **初学者**: 从"聚类算法"基础开始，掌握K-Means
2. **进阶者**: 学习"降维方法"和"聚类评估"
3. **高级者**: 深入"高斯混合模型"和"自编码器"

### 相关资源

- [Unsupervised Learning Tutorial](./Unsupervised.md)
- [Clustering Methods](./Clustering/README.md)
- [Dimensionality Reduction](./DimensionReduction/README.md)

---

**最后更新**: 2024-01-29
