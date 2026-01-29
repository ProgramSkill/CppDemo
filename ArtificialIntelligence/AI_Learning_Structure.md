# AI学习目录结构

## 📚 完整目录树

```
ArtificialIntelligence/
│
├── 00_Programming/                          # 编程基础
│   ├── Python/                              # Python基础：语法、数据类型、控制流
│   ├── OOP/                                 # 面向对象编程：类、继承、多态
│   ├── Debugging/                           # 调试技巧：断点、日志、性能分析
│   ├── NumPy/                               # NumPy：数组操作、广播、线性代数
│   ├── Pandas/                              # Pandas：数据处理、DataFrame、数据清洗
│   └── Matplotlib/                          # Matplotlib：数据可视化、图表绘制
│
├── 01_Foundations/                          # 数学与算法基础
│   ├── Math/                                # 数学基础
│   │   ├── LinearAlgebra/                   # 线性代数：矩阵、向量运算、特征值、特征向量
│   │   ├── Calculus/                        # 微积分：导数、梯度、链式法则、泰勒展开
│   │   ├── Probability/                     # 概率论：概率分布、贝叶斯定理、统计推断
│   │   ├── NumericalOptimization/           # 数值优化：凸优化、梯度下降变体、L-BFGS、牛顿法
│   │   └── InformationTheory/               # 信息论：熵、KL散度、互信息、交叉熵
│   ├── Statistics/                          # 统计学
│   │   ├── DescriptiveStats/                # 描述性统计：均值、方差、分布、可视化
│   │   └── InferentialStats/                # 推断统计：置信区间、参数估计
│   │       └── HypothesisTesting/           # 假设检验：t检验、卡方检验、方差分析
│   └── Algorithms/                          # 基础算法
│       ├── DataStructures/                  # 数据结构：数组、链表、树、图、哈希表
│       └── Complexity/                      # 算法复杂度：时间复杂度、空间复杂度、大O表示
│
├── 02_MachineLearning/                      # 机器学习
│   ├── Supervised/                          # 监督学习
│   │   ├── Regression/                      # 回归：线性回归、多项式回归、逻辑回归
│   │   ├── Classification/                  # 分类：决策树、SVM、KNN、朴素贝叶斯
│   │   └── Evaluation/                      # 模型评估：交叉验证、混淆矩阵、ROC曲线
│   ├── Unsupervised/                        # 无监督学习
│   │   ├── Clustering/                      # 聚类：K-means、DBSCAN、层次聚类
│   │   ├── DimensionReduction/              # 降维：PCA、t-SNE、LDA
│   │   └── Association/                     # 关联规则：Apriori、FP-Growth
│   ├── Reinforcement/                       # 强化学习
│   │   ├── MDPs/                            # 马尔可夫决策过程：状态、动作、奖励
│   │   ├── QLearning/                       # Q学习：Q表、Q网络
│   │   └── PolicyGradient/                  # 策略梯度：REINFORCE、Actor-Critic
│   ├── Ensemble/                            # 集成学习
│   │   ├── Bagging/                         # Bagging：随机森林、Bootstrap
│   │   ├── Boosting/                        # Boosting：AdaBoost、XGBoost、LightGBM
│   │   └── Stacking/                        # 模型堆叠：元学习器、多层集成
│   ├── FeatureEngineering/                  # 特征工程
│   │   ├── Scaling/                         # 特征缩放：标准化、归一化、鲁棒缩放
│   │   ├── Encoding/                        # 特征编码：独热编码、标签编码、目标编码
│   │   └── Selection/                       # 特征选择：过滤法、包装法、嵌入法
│   └── ModelInterpretability/               # 模型可解释性
│       ├── FeatureImportance/               # 特征重要性：排列重要性、SHAP值
│       ├── LIME/                            # LIME：局部可解释模型
│       └── PartialDependence/               # 部分依赖图、个体条件期望
│
├── 03_DeepLearning/                         # 深度学习
│   ├── NeuralNetworks/                      # 神经网络基础
│   │   ├── Perceptron/                      # 感知机：单层感知机、多层感知机
│   │   ├── MLP/                             # 多层感知机：全连接层、隐藏层
│   │   ├── Activation/                      # 激活函数：ReLU、Sigmoid、Tanh、Softmax
│   │   └── Backpropagation/                 # 反向传播：链式法则、梯度计算
│   ├── CNN/                                 # 卷积神经网络
│   │   ├── Convolution/                     # 卷积操作：卷积核、特征图、步长、填充
│   │   ├── Pooling/                         # 池化层：最大池化、平均池化
│   │   ├── Architectures/                   # 经典架构：LeNet、AlexNet、VGG、ResNet、Inception
│   │   └── Applications/                    # 应用场景：图像分类、目标检测、图像分割
│   ├── RNN/                                 # 循环神经网络
│   │   ├── BasicRNN/                        # 基础RNN：循环单元、时间序列
│   │   ├── LSTM/                            # 长短期记忆网络：遗忘门、输入门、输出门
│   │   ├── GRU/                             # 门控循环单元：更新门、重置门
│   │   └── Seq2Seq/                         # 序列到序列模型：编码器-解码器架构
│   ├── Transformers/                        # Transformer架构
│   │   ├── Attention/                       # 注意力机制：Query、Key、Value
│   │   ├── SelfAttention/                   # 自注意力：多头注意力、位置编码
│   │   ├── BERT/                            # BERT模型：预训练、掩码语言模型
│   │   └── GPT/                             # GPT系列模型：自回归、生成式预训练
│   ├── Regularization_Optimization/         # 正则化与优化
│   │   ├── Regularization/                  # 正则化：Dropout、L1/L2正则、Early Stopping
│   │   ├── Normalization/                   # 归一化：BatchNorm、LayerNorm、GroupNorm
│   │   ├── Optimizers/                      # 优化器：SGD、Adam、AdamW、RMSprop
│   │   └── LearningRateScheduling/          # 学习率调度：余弦退火、warmup、衰减策略
│   └── LargeScaleTraining/                  # 大规模训练
│       ├── DistributedTraining/             # 分布式训练：数据并行、模型并行、流水线并行
│       ├── MixedPrecision/                  # 混合精度训练：FP16、BF16、自动混合精度
│       └── MemoryOptimization/              # 内存优化：梯度累积、梯度检查点、ZeRO
│
├── 04_NLP/                                  # 自然语言处理
│   ├── TextProcessing/                      # 文本处理
│   │   ├── Tokenization/                    # 分词：中文分词、英文分词、子词分词
│   │   ├── Embedding/                       # 词嵌入：Word2Vec、GloVe、FastText
│   │   └── Preprocessing/                   # 文本预处理：清洗、标准化、停用词
│   ├── LanguageModels/                      # 语言模型
│   │   ├── NGrams/                          # N-gram模型：统计语言模型
│   │   ├── PretrainedModels/                # 预训练模型：BERT、GPT、T5、RoBERTa
│   │   └── FineTuning/                      # 微调技术：迁移学习、领域适应
│   └── Applications/                        # NLP应用
│       ├── TextClassification/              # 文本分类：情感分析、主题分类
│       ├── NER/                             # 命名实体识别：人名、地名、机构名
│       ├── MachineTranslation/              # 机器翻译：神经机器翻译、注意力机制
│       ├── QA/                              # 问答系统：阅读理解、知识问答
│       └── Summarization/                   # 文本摘要：抽取式摘要、生成式摘要
│
├── 05_ComputerVision/                       # 计算机视觉
│   ├── ImageProcessing/                     # 图像处理
│   │   ├── Filtering/                       # 图像滤波：高斯滤波、中值滤波、双边滤波
│   │   ├── EdgeDetection/                   # 边缘检测：Canny、Sobel、Laplacian
│   │   └── Transformation/                  # 图像变换：旋转、缩放、仿射变换
│   ├── ObjectDetection/                     # 目标检测
│   │   ├── RCNN/                            # R-CNN系列：R-CNN、Fast R-CNN、Faster R-CNN
│   │   ├── YOLO/                            # YOLO系列：YOLOv3、YOLOv4、YOLOv5、YOLOv8
│   │   └── SSD/                             # SSD算法：单次检测、多尺度特征
│   ├── ImageSegmentation/                   # 图像分割
│   │   ├── SemanticSegmentation/            # 语义分割：FCN、U-Net、DeepLab
│   │   └── InstanceSegmentation/            # 实例分割：Mask R-CNN、YOLACT
│   ├── ImageGeneration/                     # 图像生成
│   │   ├── GAN/                             # 生成对抗网络：DCGAN、StyleGAN、CycleGAN
│   │   ├── VAE/                             # 变分自编码器：编码器、解码器、潜在空间
│   │   └── Diffusion/                       # 扩散模型：DDPM、Stable Diffusion
│   └── FoundationModels/                    # 基础模型（前沿）
│       ├── VisionTransformer/               # Vision Transformer：ViT、DeiT、Swin Transformer
│       ├── MultiModal/                      # 多模态模型：CLIP、ALIGN、Florence
│       └── UniversalSegmentation/           # 通用分割：Segment Anything (SAM)、Mask2Former
│
├── 06_Projects/                             # 实战项目
│   ├── BeginnerProjects/                    # 初级项目
│   │   ├── IrisClassification/              # 鸢尾花分类：经典入门项目
│   │   ├── HousePricePredict/               # 房价预测：回归问题实践
│   │   └── DigitRecognition/                # 手写数字识别：MNIST数据集
│   ├── IntermediateProjects/                # 中级项目
│   │   ├── SentimentAnalysis/               # 情感分析：文本分类应用
│   │   ├── ImageClassifier/                 # 图像分类器：CNN实战
│   │   └── RecommendSystem/                 # 推荐系统：协同过滤、内容推荐
│   └── AdvancedProjects/                    # 高级项目
│       ├── ChatBot/                         # 聊天机器人：对话系统、意图识别
│       ├── ObjectDetector/                  # 目标检测系统：实时检测应用
│       └── ImageGenerator/                  # 图像生成器：GAN或扩散模型应用
│
├── 07_Engineering/                          # AI工程实践
│   ├── ModelDeployment/                     # 模型部署
│   │   ├── ServingFrameworks/               # 服务框架：TensorFlow Serving、TorchServe、ONNX Runtime
│   │   ├── APIDesign/                       # API设计：RESTful API、gRPC、FastAPI
│   │   └── Containerization/                # 容器化：Docker、Kubernetes、微服务架构
│   ├── MLOps/                               # MLOps实践
│   │   ├── CICD/                            # CI/CD：自动化测试、持续集成、持续部署
│   │   ├── ModelVersioning/                 # 模型版本管理：MLflow、DVC、Git LFS
│   │   ├── Monitoring/                      # 模型监控：性能监控、数据漂移检测、A/B测试
│   │   └── ExperimentTracking/              # 实验跟踪：Weights & Biases、TensorBoard、Neptune
│   ├── ModelOptimization/                   # 模型优化
│   │   ├── Compression/                     # 模型压缩：剪枝、知识蒸馏、低秩分解
│   │   ├── Quantization/                    # 量化：INT8量化、动态量化、量化感知训练
│   │   └── Acceleration/                    # 加速推理：TensorRT、OpenVINO、ONNX优化
│   ├── EdgeDeployment/                      # 边缘计算部署
│   │   ├── MobileDeployment/                # 移动端部署：TensorFlow Lite、Core ML、NCNN
│   │   ├── EmbeddedSystems/                 # 嵌入式系统：树莓派、Jetson Nano、边缘TPU
│   │   └── WebDeployment/                   # Web部署：TensorFlow.js、ONNX.js、WebAssembly
│   └── DataManagement/                      # 数据管理
│       ├── DataAnnotation/                  # 数据标注：标注工具、质量控制、众包平台
│       ├── DataPipeline/                    # 数据流水线：ETL、数据清洗、特征存储
│       └── DataVersioning/                  # 数据版本控制：DVC、数据血缘、数据治理
│
└── 08_Resources/                            # 学习资源
    ├── Papers/                              # 论文资源
    │   ├── Classic/                         # 经典论文：AlexNet、ResNet、Transformer等
    │   ├── Recent/                          # 最新研究：前沿技术、新方法
    │   └── ReadingNotes/                    # 论文笔记：理解与总结
    ├── Books/                               # 书籍资源
    │   ├── MachineLearning/                 # 机器学习书籍：周志华《机器学习》等
    │   ├── DeepLearning/                    # 深度学习书籍：Goodfellow《深度学习》等
    │   └── Notes/                           # 读书笔记：章节总结、知识点整理
    ├── Tutorials/                           # 教程资源
    │   ├── OnlineCourses/                   # 在线课程：Coursera、edX、Udacity
    │   ├── VideoLectures/                   # 视频讲座：Stanford CS231n、CS224n
    │   └── Blogs/                           # 技术博客：Medium、知乎、CSDN
    └── Tools/                               # 工具资源
        ├── Frameworks/                      # 深度学习框架：TensorFlow、PyTorch、Keras
        ├── Libraries/                       # 常用库：NumPy、Pandas、Scikit-learn、Matplotlib
        └── Datasets/                        # 数据集资源：Kaggle、UCI、ImageNet、COCO

09_LLMs/                                     # 大语言模型（专题深入）
├── Foundations/                             # LLM理论基础
│   ├── Tokenization/                        # 分词技术：BPE、SentencePiece、tiktoken
│   ├── PretrainingObjectives/               # 预训练目标：MLM、CLM、Seq2Seq、Instruction Tuning
│   ├── ScalingLaws/                         # 缩放法则：Scaling Law、Chinchilla定律
│   └── SafetyAlignment/                     # 安全对齐：RLHF、RLAIF、Constitutional AI、偏见与安全
├── ModelFamilies/                           # 模型家族
│   ├── GPTSeries/                           # GPT系列：GPT-2/3/4、架构演化与能力涌现
│   ├── LLaMA_Family/                        # LLaMA家族：LLaMA、LLaMA 2/3及衍生模型
│   ├── OpenSourceLLMs/                      # 开源LLM：Mistral、Qwen、GLM、Gemma
│   └── SpecializedLLMs/                     # 专用LLM：Code LLM、多模态LLM、领域LLM
├── PromptEngineering/                       # 提示工程
│   ├── BasicPatterns/                       # 基础模式：Few-shot、Chain-of-Thought、ReAct、自一致性
│   ├── ToolUse_Agents/                      # 工具使用与智能体：Toolformer、ReWOO、Planner-Worker
│   └── Evaluation/                          # 提示评估：鲁棒性测试、提示优化
├── RAG_and_Tools/                           # 检索增强与工具调用
│   ├── RAG/                                 # RAG框架：索引构建、检索策略、重排序、上下文压缩
│   ├── VectorDB/                            # 向量数据库：Faiss、Milvus、Chroma、Pinecone、pgvector
│   └── LLMApps/                             # LLM应用：Chatbot、知识库问答、文档助手、代码助手
├── FineTuning_Training/                     # 训练与微调
│   ├── FullFineTuning/                      # 全量微调：从头预训练、继续预训练、指令微调
│   ├── ParameterEfficient/                  # 参数高效微调：LoRA、QLoRA、Adapter、Prefix Tuning
│   ├── PreferenceOptimization/              # 偏好优化：RLHF、DPO、KTO、IPO
│   └── Efficiency/                          # 效率优化：KV Cache、FlashAttention、量化推理、投机采样
└── Systems_Engineering/                     # LLM系统工程
    ├── Serving_Inference/                   # 服务与推理：vLLM、TensorRT-LLM、Text Generation Inference
    ├── LLMOps/                              # LLMOps：监控、评估、A/B测试、成本优化
    └── Security_Privacy/                    # 安全与隐私：提示注入防御、数据泄露防护、内容审核
```

## 🎯 学习路径建议

| 阶段 | 时间 | 学习内容 | 实战项目 |
|------|------|----------|----------|
| **阶段一：基础准备** | 1-2个月 | **编程基础**：Python、OOP、NumPy、Pandas、Matplotlib<br>**数学基础**：线性代数、微积分、概率论、信息论<br>**算法基础**：数据结构、算法复杂度 | - |
| **阶段二：机器学习入门** | 2-3个月 | 监督学习（回归、分类算法）<br>无监督学习（聚类、降维）<br>模型评估（交叉验证、性能指标） | 鸢尾花分类<br>房价预测 |
| **阶段三：深度学习基础** | 2-3个月 | 神经网络（感知机、MLP、反向传播）<br>CNN（卷积神经网络及图像应用）<br>RNN/LSTM（序列数据处理） | 手写数字识别<br>图像分类 |
| **阶段四：专业方向深入** | 3-6个月 | **NLP方向**：Transformer、BERT、GPT、文本应用<br>**CV方向**：目标检测、图像分割、图像生成<br>**LLM方向**：提示工程、RAG、模型微调、LLM应用开发<br>**强化学习**：Q-Learning、策略梯度、游戏AI | 情感分析<br>目标检测系统<br>RAG问答系统<br>聊天机器人 |
| **阶段五：实战与进阶** | 持续 | **AI工程实践**：模型部署、MLOps、模型优化、边缘部署<br>完成综合项目<br>阅读前沿论文<br>参与开源项目<br>关注最新技术动态 | 生产级AI系统<br>开源贡献 |

## 📝 使用说明

**目录组织原则：**
- 按照学习难度从基础到高级递进
- 每个主题包含理论、实践和应用三个层次
- 项目驱动学习，理论与实践相结合

**学习建议：**
1. **循序渐进**：按照00-09的顺序学习，从编程基础到LLM专题，打好基础
2. **理论实践结合**：每学一个概念就动手实现
3. **做好笔记**：在对应目录下记录学习笔记和代码
4. **定期复习**：定期回顾之前的内容，巩固知识
5. **项目驱动**：通过实际项目来检验学习效果
6. **工程导向**：重视模型部署和MLOps实践，培养工程化思维
7. **LLM专题**：在掌握Transformer基础后，可深入09_LLMs/学习提示工程、RAG和微调技术

**推荐资源：**
- **书籍**：《机器学习》周志华、《深度学习》Goodfellow、《统计学习方法》李航
- **课程**：Andrew Ng的机器学习课程、Stanford CS231n（计算机视觉）、CS224n（NLP）
- **框架**：PyTorch（推荐）、TensorFlow、Scikit-learn
- **社区**：GitHub、Kaggle、Papers with Code

---

**更新日志：** 2026-01-29 创建完整树形目录结构
