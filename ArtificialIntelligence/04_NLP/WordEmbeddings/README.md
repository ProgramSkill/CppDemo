# Word Embeddings (词嵌入)

将词转换为稠密向量表示。

## 📚 内容概览

| 主题 | 描述 | 难度 |
|------|------|------|
| Word2Vec | CBOW, Skip-gram | ⭐⭐ |
| GloVe | 全局向量 | ⭐⭐ |
| FastText | 子词嵌入 | ⭐⭐ |
| 预训练嵌入 | 使用现有模型 | ⭐ |

## 💡 Word2Vec思想

```
Skip-gram: 给定中心词，预测上下文
CBOW: 给定上下文，预测中心词

king - man + woman ≈ queen
```
