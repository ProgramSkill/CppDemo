# Feature Engineering Glossary (特征工程词汇表)

## English-Chinese Technical Terms

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Feature Engineering** | 特征工程 | 创建和转换特征的过程 |
| **Feature** | 特征 | 模型的输入变量 |
| **Raw Data** | 原始数据 | 未处理的数据 |
| **Feature Vector** | 特征向量 | 特征的数值表示 |
| **Domain Knowledge** | 领域知识 | 专业领域的知识 |

---

## 2. 缺失值处理 (Missing Data Handling)

| English | 中文 | 说明 |
|---------|------|------|
| **Missing Value** | 缺失值 | 数据中的空值 |
| **Imputation** | 填充 | 填补缺失值 |
| **Mean Imputation** | 均值填充 | 用均值填补 |
| **Median Imputation** | 中位数填充 | 用中位数填补 |
| **Mode Imputation** | 众数填充 | 用众数填补 |
| **KNN Imputation** | KNN填充 | 用近邻值填补 |
| **Missing Indicator** | 缺失指示器 | 标记是否缺失 |

---

## 3. 类别编码 (Categorical Encoding)

| English | 中文 | 说明 |
|---------|------|------|
| **Categorical Variable** | 类别变量 | 非数值变量 |
| **Label Encoding** | 标签编码 | 整数编码 |
| **One-Hot Encoding** | 独热编码 | 二进制向量编码 |
| **Ordinal Encoding** | 序数编码 | 有序类别编码 |
| **Target Encoding** | 目标编码 | 用目标均值编码 |
| **Frequency Encoding** | 频率编码 | 用出现频率编码 |
| **Binary Encoding** | 二进制编码 | 二进制表示 |
| **Hash Encoding** | 哈希编码 | 哈希函数编码 |

---

## 4. 特征缩放 (Feature Scaling)

| English | 中文 | 说明 |
|---------|------|------|
| **Feature Scaling** | 特征缩放 | 调整特征范围 |
| **Standardization** | 标准化 | Z-score标准化 |
| **Normalization** | 归一化 | 缩放到指定范围 |
| **Min-Max Scaling** | 最小最大缩放 | 缩放到[0,1] |
| **Robust Scaling** | 稳健缩放 | 基于中位数和IQR |
| **Z-score** | Z分数 | (x-μ)/σ |

---

## 5. 特征变换 (Feature Transformation)

| English | 中文 | 说明 |
|---------|------|------|
| **Log Transform** | 对数变换 | 取对数 |
| **Square Root** | 平方根 | 取平方根 |
| **Power Transform** | 幂变换 | Box-Cox, Yeo-Johnson |
| **Binning** | 分箱 | 离散化连续变量 |
| **Discretization** | 离散化 | 转为类别变量 |
| **Polynomial Features** | 多项式特征 | 高次项特征 |

---

## 6. 特征创建 (Feature Creation)

| English | 中文 | 说明 |
|---------|------|------|
| **Feature Creation** | 特征创建 | 创建新特征 |
| **Interaction Feature** | 交互特征 | 特征间的组合 |
| **Aggregation** | 聚合 | 统计聚合特征 |
| **Ratio** | 比率 | 特征间的比值 |
| **Difference** | 差值 | 特征间的差 |
| **Cross Feature** | 交叉特征 | 特征交叉组合 |

---

## 7. 特征选择 (Feature Selection)

| English | 中文 | 说明 |
|---------|------|------|
| **Feature Selection** | 特征选择 | 选择重要特征 |
| **Filter Method** | 过滤法 | 基于统计的选择 |
| **Wrapper Method** | 包装法 | 基于模型的选择 |
| **Embedded Method** | 嵌入法 | 模型内部选择 |
| **Feature Importance** | 特征重要性 | 特征的重要程度 |
| **Correlation** | 相关性 | 特征间的关联 |
| **Mutual Information** | 互信息 | 信息相关度 |
| **RFE** | 递归特征消除 | 逐步移除特征 |

---

## 8. 时间特征 (Temporal Features)

| English | 中文 | 说明 |
|---------|------|------|
| **Date Feature** | 日期特征 | 从日期提取 |
| **Time Feature** | 时间特征 | 从时间提取 |
| **Lag Feature** | 滞后特征 | 历史值特征 |
| **Rolling Feature** | 滚动特征 | 滑动窗口统计 |
| **Cyclical Encoding** | 周期编码 | 正弦余弦编码 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **FE** | Feature Engineering | 特征工程 |
| **OHE** | One-Hot Encoding | 独热编码 |
| **RFE** | Recursive Feature Elimination | 递归特征消除 |
| **PCA** | Principal Component Analysis | 主成分分析 |
| **TF-IDF** | Term Frequency-Inverse Document Frequency | 词频-逆文档频率 |

---

**最后更新**: 2024-01-29
