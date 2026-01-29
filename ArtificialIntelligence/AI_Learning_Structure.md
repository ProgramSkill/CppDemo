# AI学习目录结构

## 📚 目录组织说明

本文档定义了人工智能学习内容的目录结构，帮助系统化地组织和学习AI相关知识。

---

## 🗂️ 目录结构

### 01_Foundations - 基础知识
AI学习的数学和算法基础

```
01_Foundations/
├── Math/
│   ├── LinearAlgebra/          # 线性代数：矩阵、向量运算
│   ├── Calculus/               # 微积分：导数、梯度、优化
│   └── Probability/            # 概率论：概率分布、贝叶斯定理
├── Statistics/
│   ├── DescriptiveStats/       # 描述性统计
│   ├── InferentialStats/       # 推断统计
│   └── Hypothesis/             # 假设检验
└── Algorithms/
    ├── DataStructures/         # 数据结构
    ├── Complexity/             # 算法复杂度
    └── Optimization/           # 优化算法
```

**学习重点：**
- 线性代数：矩阵运算、特征值、特征向量
- 微积分：梯度下降、反向传播的数学基础
- 概率统计：理解模型的不确定性和评估

---

### 02_MachineLearning - 机器学习
传统机器学习算法和方法

```
02_MachineLearning/
├── Supervised/
│   ├── Regression/             # 回归：线性回归、逻辑回归
│   ├── Classification/         # 分类：决策树、SVM、KNN
│   └── Evaluation/             # 模型评估：交叉验证、指标
├── Unsupervised/
│   ├── Clustering/             # 聚类：K-means、层次聚类
│   ├── DimensionReduction/     # 降维：PCA、t-SNE
│   └── Association/            # 关联规则
├── Reinforcement/
│   ├── MDPs/                   # 马尔可夫决策过程
│   ├── QLearning/              # Q学习
│   └── PolicyGradient/         # 策略梯度
└── Ensemble/
    ├── Bagging/                # Bagging：随机森林
    ├── Boosting/               # Boosting：AdaBoost、XGBoost
    └── Stacking/               # 模型堆叠
```

**学习重点：**
- 监督学习：从标注数据中学习
- 无监督学习：发现数据中的模式
- 特征工程：数据预处理和特征选择

---

### 03_DeepLearning - 深度学习
神经网络和深度学习架构

```
03_DeepLearning/
├── NeuralNetworks/
│   ├── Perceptron/             # 感知机
│   ├── MLP/                    # 多层感知机
│   ├── Activation/             # 激活函数
│   └── Backpropagation/        # 反向传播
├── CNN/
│   ├── Convolution/            # 卷积操作
│   ├── Pooling/                # 池化层
│   ├── Architectures/          # 经典架构：LeNet、VGG、ResNet
│   └── Applications/           # 应用场景
├── RNN/
│   ├── BasicRNN/               # 基础RNN
│   ├── LSTM/                   # 长短期记忆网络
│   ├── GRU/                    # 门控循环单元
│   └── Seq2Seq/                # 序列到序列模型
└── Transformers/
    ├── Attention/              # 注意力机制
    ├── SelfAttention/          # 自注意力
    ├── BERT/                   # BERT模型
    └── GPT/                    # GPT系列模型
```

**学习重点：**
- 神经网络基础：前向传播、反向传播
- CNN：图像处理的核心技术
- Transformer：现代NLP的基础架构

---

### 04_NLP - 自然语言处理
文本和语言理解技术

```
04_NLP/
├── TextProcessing/
│   ├── Tokenization/           # 分词
│   ├── Embedding/              # 词嵌入：Word2Vec、GloVe
│   └── Preprocessing/          # 文本预处理
├── LanguageModels/
│   ├── NGrams/                 # N-gram模型
│   ├── PretrainedModels/       # 预训练模型：BERT、GPT
│   └── FineTuning/             # 微调技术
└── Applications/
    ├── TextClassification/     # 文本分类
    ├── NER/                    # 命名实体识别
    ├── MachineTranslation/     # 机器翻译
    ├── QA/                     # 问答系统
    └── Summarization/          # 文本摘要
```

**学习重点：**
- 词嵌入：将文本转换为向量表示
- 预训练模型：利用大规模语料库的知识
- 实际应用：分类、翻译、问答等任务

---

### 05_ComputerVision - 计算机视觉
图像和视频处理技术

```
05_ComputerVision/
├── ImageProcessing/
│   ├── Filtering/              # 图像滤波
│   ├── EdgeDetection/          # 边缘检测
│   └── Transformation/         # 图像变换
├── ObjectDetection/
│   ├── RCNN/                   # R-CNN系列
│   ├── YOLO/                   # YOLO系列
│   └── SSD/                    # SSD算法
├── ImageSegmentation/
│   ├── SemanticSegmentation/   # 语义分割
│   └── InstanceSegmentation/   # 实例分割
└── ImageGeneration/
    ├── GAN/                    # 生成对抗网络
    ├── VAE/                    # 变分自编码器
    └── Diffusion/              # 扩散模型
```

**学习重点：**
- 图像处理基础：滤波、特征提取
- 目标检测：定位和识别图像中的对象
- 图像生成：GAN和扩散模型的应用

---

### 06_Projects - 实战项目
动手实践和项目开发

```
06_Projects/
├── BeginnerProjects/
│   ├── IrisClassification/     # 鸢尾花分类
│   ├── HousePricePredict/      # 房价预测
│   └── DigitRecognition/       # 手写数字识别
├── IntermediateProjects/
│   ├── SentimentAnalysis/      # 情感分析
│   ├── ImageClassifier/        # 图像分类器
│   └── RecommendSystem/        # 推荐系统
└── AdvancedProjects/
    ├── ChatBot/                # 聊天机器人
    ├── ObjectDetector/         # 目标检测系统
    └── ImageGenerator/         # 图像生成器
```

**学习重点：**
- 从简单项目开始，逐步提升难度
- 完整的项目流程：数据处理、模型训练、评估、部署
- 实际问题解决能力

---

### 07_Resources - 学习资源
参考资料和学习材料

```
07_Resources/
├── Papers/
│   ├── Classic/                # 经典论文
│   ├── Recent/                 # 最新研究
│   └── ReadingNotes/           # 论文笔记
├── Books/
│   ├── MachineLearning/        # 机器学习书籍
│   ├── DeepLearning/           # 深度学习书籍
│   └── Notes/                  # 读书笔记
├── Tutorials/
│   ├── OnlineCourses/          # 在线课程
│   ├── VideoLectures/          # 视频讲座
│   └── Blogs/                  # 技术博客
└── Tools/
    ├── Frameworks/             # 框架：TensorFlow、PyTorch
    ├── Libraries/              # 库：NumPy、Pandas
    └── Datasets/               # 数据集资源
```

**推荐资源：**
- 书籍：《机器学习》周志华、《深度学习》Goodfellow
- 课程：Andrew Ng的机器学习课程、Stanford CS231n
- 框架：PyTorch、TensorFlow、scikit-learn

---

## 🎯 建议学习路径

### 阶段一：基础准备（1-2个月）
1. 数学基础：线性代数、微积分、概率统计
2. 编程基础：Python、NumPy、Pandas
3. 算法基础：数据结构、基础算法

### 阶段二：机器学习入门（2-3个月）
1. 监督学习：回归、分类算法
2. 无监督学习：聚类、降维
3. 模型评估：交叉验证、性能指标
4. 实战项目：鸢尾花分类、房价预测

### 阶段三：深度学习基础（2-3个月）
1. 神经网络：感知机、MLP、反向传播
2. CNN：卷积神经网络及图像应用
3. RNN/LSTM：序列数据处理
4. 实战项目：手写数字识别、图像分类

### 阶段四：专业方向深入（3-6个月）
根据兴趣选择方向：
- **NLP方向**：Transformer、BERT、GPT、文本应用
- **CV方向**：目标检测、图像分割、图像生成
- **强化学习**：Q-Learning、策略梯度、游戏AI

### 阶段五：实战与进阶（持续）
1. 完成综合项目
2. 阅读前沿论文
3. 参与开源项目
4. 关注最新技术动态

---

## 📝 使用建议

1. **循序渐进**：按照目录顺序学习，打好基础
2. **理论实践结合**：每学一个概念就动手实现
3. **做好笔记**：在对应目录下记录学习笔记和代码
4. **定期复习**：定期回顾之前的内容，巩固知识
5. **项目驱动**：通过实际项目来检验学习效果

---

## 🔄 更新日志

- 2026-01-29：创建初始目录结构文档
