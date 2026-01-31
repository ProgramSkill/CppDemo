# Deep Learning Glossary (深度学习词汇表)

## English-Chinese Technical Terms

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Deep Learning** | 深度学习 | 多层神经网络的学习 |
| **Neural Network** | 神经网络 | 模拟神经元的计算模型 |
| **Neuron** | 神经元 | 网络的基本单元 |
| **Layer** | 层 | 神经元的集合 |
| **Input Layer** | 输入层 | 接收数据的层 |
| **Hidden Layer** | 隐藏层 | 中间计算层 |
| **Output Layer** | 输出层 | 产生结果的层 |
| **Weight** | 权重 | 连接的强度 |
| **Bias** | 偏置 | 激活阈值调整 |
| **Parameter** | 参数 | 可学习的变量 |

---

## 2. 激活函数 (Activation Functions)

| English | 中文 | 说明 |
|---------|------|------|
| **Activation Function** | 激活函数 | 非线性变换 |
| **Sigmoid** | Sigmoid | S形函数 (0,1) |
| **Tanh** | 双曲正切 | (-1,1) |
| **ReLU** | ReLU | max(0,x) |
| **Leaky ReLU** | 泄漏ReLU | 允许负值小梯度 |
| **GELU** | GELU | Transformer常用 |
| **Softmax** | Softmax | 多分类输出 |
| **Swish** | Swish | x×sigmoid(x) |

---

## 3. 训练相关 (Training Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Forward Propagation** | 前向传播 | 计算输出 |
| **Backpropagation** | 反向传播 | 计算梯度 |
| **Gradient** | 梯度 | 损失对参数的导数 |
| **Loss Function** | 损失函数 | 衡量预测误差 |
| **Optimizer** | 优化器 | 更新参数的算法 |
| **Learning Rate** | 学习率 | 参数更新步长 |
| **Epoch** | 轮次 | 完整数据集遍历 |
| **Batch** | 批次 | 数据子集 |
| **Batch Size** | 批大小 | 批次中的样本数 |
| **Iteration** | 迭代 | 一次参数更新 |
| **Convergence** | 收敛 | 训练稳定 |

---

## 4. 优化器 (Optimizers)

| English | 中文 | 说明 |
|---------|------|------|
| **SGD** | 随机梯度下降 | 基础优化器 |
| **Momentum** | 动量 | 加速收敛 |
| **Adam** | Adam | 自适应学习率 |
| **AdamW** | AdamW | Adam+权重衰减 |
| **RMSprop** | RMSprop | 自适应学习率 |
| **Learning Rate Scheduler** | 学习率调度器 | 动态调整学习率 |

---

## 5. 正则化 (Regularization)

| English | 中文 | 说明 |
|---------|------|------|
| **Regularization** | 正则化 | 防止过拟合 |
| **Dropout** | Dropout | 随机丢弃神经元 |
| **Batch Normalization** | 批归一化 | 标准化层输入 |
| **Layer Normalization** | 层归一化 | 对单样本标准化 |
| **Weight Decay** | 权重衰减 | L2正则化 |
| **Early Stopping** | 早停 | 验证损失上升时停止 |
| **Data Augmentation** | 数据增强 | 扩充训练数据 |

---

## 6. CNN相关 (CNN Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Convolutional Neural Network** | 卷积神经网络 | 处理网格数据 |
| **Convolution** | 卷积 | 特征提取操作 |
| **Kernel/Filter** | 卷积核/滤波器 | 卷积的权重矩阵 |
| **Feature Map** | 特征图 | 卷积输出 |
| **Pooling** | 池化 | 下采样操作 |
| **Max Pooling** | 最大池化 | 取最大值 |
| **Average Pooling** | 平均池化 | 取平均值 |
| **Stride** | 步长 | 卷积移动距离 |
| **Padding** | 填充 | 边界补零 |
| **Channel** | 通道 | 特征图的深度 |

---

## 7. RNN相关 (RNN Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Recurrent Neural Network** | 循环神经网络 | 处理序列数据 |
| **Hidden State** | 隐藏状态 | 时间步间的信息 |
| **LSTM** | 长短期记忆 | 解决长期依赖 |
| **GRU** | 门控循环单元 | LSTM的简化版 |
| **Cell State** | 细胞状态 | LSTM的长期记忆 |
| **Gate** | 门 | 控制信息流动 |
| **Forget Gate** | 遗忘门 | 决定丢弃信息 |
| **Input Gate** | 输入门 | 决定存储信息 |
| **Output Gate** | 输出门 | 决定输出信息 |
| **Sequence-to-Sequence** | 序列到序列 | Seq2Seq模型 |
| **Bidirectional** | 双向 | 正向和反向 |

---

## 8. Transformer相关 (Transformer Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Transformer** | Transformer | 注意力架构 |
| **Attention** | 注意力 | 加权聚合机制 |
| **Self-Attention** | 自注意力 | 序列内部注意力 |
| **Multi-Head Attention** | 多头注意力 | 多个注意力头 |
| **Query** | 查询 | 注意力的Q |
| **Key** | 键 | 注意力的K |
| **Value** | 值 | 注意力的V |
| **Positional Encoding** | 位置编码 | 序列位置信息 |
| **Encoder** | 编码器 | 编码输入 |
| **Decoder** | 解码器 | 生成输出 |

---

## 9. 生成模型 (Generative Models)

| English | 中文 | 说明 |
|---------|------|------|
| **Generative Model** | 生成模型 | 生成新数据 |
| **Autoencoder** | 自编码器 | 压缩重建模型 |
| **VAE** | 变分自编码器 | 概率生成模型 |
| **GAN** | 生成对抗网络 | 对抗训练 |
| **Generator** | 生成器 | 生成假样本 |
| **Discriminator** | 判别器 | 区分真假 |
| **Latent Space** | 潜在空间 | 压缩表示空间 |
| **Diffusion Model** | 扩散模型 | 去噪生成 |

---

## 10. 常见架构 (Common Architectures)

| English | 中文 | 说明 |
|---------|------|------|
| **ResNet** | 残差网络 | 跳跃连接 |
| **Skip Connection** | 跳跃连接 | 残差连接 |
| **U-Net** | U-Net | 分割网络 |
| **VGG** | VGG | 深层CNN |
| **Inception** | Inception | 多尺度卷积 |
| **EfficientNet** | EfficientNet | 高效缩放网络 |
| **ViT** | Vision Transformer | 图像Transformer |
| **BERT** | BERT | 双向Transformer |
| **GPT** | GPT | 生成式预训练 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **DL** | Deep Learning | 深度学习 |
| **NN** | Neural Network | 神经网络 |
| **CNN** | Convolutional Neural Network | 卷积神经网络 |
| **RNN** | Recurrent Neural Network | 循环神经网络 |
| **LSTM** | Long Short-Term Memory | 长短期记忆 |
| **GRU** | Gated Recurrent Unit | 门控循环单元 |
| **GAN** | Generative Adversarial Network | 生成对抗网络 |
| **VAE** | Variational Autoencoder | 变分自编码器 |
| **MLP** | Multi-Layer Perceptron | 多层感知机 |
| **BN** | Batch Normalization | 批归一化 |
| **LN** | Layer Normalization | 层归一化 |

---

**最后更新**: 2024-01-29
