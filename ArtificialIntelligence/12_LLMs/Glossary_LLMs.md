# LLM Glossary (大语言模型词汇表)

## English-Chinese Technical Terms

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Large Language Model** | 大语言模型 | 大规模语言模型 |
| **LLM** | LLM | 大语言模型缩写 |
| **Foundation Model** | 基础模型 | 通用预训练模型 |
| **Pretrained Model** | 预训练模型 | 大规模预训练的模型 |
| **Token** | 词元 | 文本的基本单位 |
| **Context Window** | 上下文窗口 | 可处理的最大长度 |
| **Parameter** | 参数 | 模型的可学习变量 |

---

## 2. 提示工程 (Prompt Engineering)

| English | 中文 | 说明 |
|---------|------|------|
| **Prompt** | 提示 | 输入给模型的文本 |
| **Prompt Engineering** | 提示工程 | 设计有效提示 |
| **System Prompt** | 系统提示 | 定义模型角色 |
| **Zero-Shot** | 零样本 | 无示例推理 |
| **Few-Shot** | 少样本 | 少量示例推理 |
| **Chain of Thought** | 思维链 | 逐步推理 |
| **In-Context Learning** | 上下文学习 | 从示例中学习 |

---

## 3. 微调相关 (Fine-tuning)

| English | 中文 | 说明 |
|---------|------|------|
| **Fine-tuning** | 微调 | 任务特定训练 |
| **Full Fine-tuning** | 全量微调 | 更新所有参数 |
| **PEFT** | 参数高效微调 | 只更新部分参数 |
| **LoRA** | 低秩适应 | 低秩分解微调 |
| **QLoRA** | 量化LoRA | 量化+LoRA |
| **Adapter** | 适配器 | 插入的小型网络 |
| **SFT** | 监督微调 | 指令微调 |

---

## 4. 对齐与安全 (Alignment & Safety)

| English | 中文 | 说明 |
|---------|------|------|
| **Alignment** | 对齐 | 符合人类意图 |
| **RLHF** | 人类反馈强化学习 | 使用人类反馈训练 |
| **Reward Model** | 奖励模型 | 评估输出质量 |
| **DPO** | 直接偏好优化 | 简化的对齐方法 |
| **Hallucination** | 幻觉 | 生成虚假信息 |
| **Guardrail** | 护栏 | 安全限制 |
| **Red Teaming** | 红队测试 | 安全测试 |

---

## 5. RAG相关 (RAG Related)

| English | 中文 | 说明 |
|---------|------|------|
| **RAG** | 检索增强生成 | 结合检索和生成 |
| **Vector Database** | 向量数据库 | 存储嵌入向量 |
| **Embedding** | 嵌入 | 向量表示 |
| **Retrieval** | 检索 | 获取相关文档 |
| **Chunking** | 分块 | 文档切分 |
| **Semantic Search** | 语义搜索 | 基于语义的搜索 |

---

## 6. 代理相关 (Agent Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Agent** | 代理 | 自主执行任务 |
| **Tool Use** | 工具使用 | 调用外部工具 |
| **Function Calling** | 函数调用 | 调用API函数 |
| **ReAct** | ReAct | 推理+行动框架 |
| **Planning** | 规划 | 任务分解和规划 |
| **Memory** | 记忆 | 存储历史信息 |

---

## 7. 推理优化 (Inference Optimization)

| English | 中文 | 说明 |
|---------|------|------|
| **Quantization** | 量化 | 降低精度 |
| **INT8** | INT8量化 | 8位整数量化 |
| **INT4** | INT4量化 | 4位整数量化 |
| **KV Cache** | KV缓存 | 键值缓存 |
| **Speculative Decoding** | 推测解码 | 加速生成 |
| **Batching** | 批处理 | 并行处理请求 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **LLM** | Large Language Model | 大语言模型 |
| **RAG** | Retrieval Augmented Generation | 检索增强生成 |
| **RLHF** | Reinforcement Learning from Human Feedback | 人类反馈强化学习 |
| **SFT** | Supervised Fine-Tuning | 监督微调 |
| **PEFT** | Parameter-Efficient Fine-Tuning | 参数高效微调 |
| **LoRA** | Low-Rank Adaptation | 低秩适应 |
| **CoT** | Chain of Thought | 思维链 |
| **DPO** | Direct Preference Optimization | 直接偏好优化 |

---

**最后更新**: 2024-01-29
