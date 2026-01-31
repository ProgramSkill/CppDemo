# Multimodal Learning: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: What is Multimodal Learning?](#chapter-1-what-is-multimodal-learning)
  - [Chapter 2: Multimodal Data](#chapter-2-multimodal-data)
  - [Chapter 3: Basic Fusion Techniques](#chapter-3-basic-fusion-techniques)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: Vision-Language Models](#chapter-4-vision-language-models)
  - [Chapter 5: Cross-Modal Learning](#chapter-5-cross-modal-learning)
  - [Chapter 6: Multimodal Transformers](#chapter-6-multimodal-transformers)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: CLIP and Contrastive Learning](#chapter-7-clip-and-contrastive-learning)
  - [Chapter 8: Large Multimodal Models](#chapter-8-large-multimodal-models)
  - [Chapter 9: Applications](#chapter-9-applications)

---

## Introduction

**Multimodal Learning** combines information from multiple modalities (text, image, audio, video) for richer understanding.

### Modalities

| Modality | Examples |
|----------|----------|
| **Text** | Documents, captions |
| **Image** | Photos, diagrams |
| **Audio** | Speech, music |
| **Video** | Movies, recordings |

---

## Part I: Beginner Level

### Chapter 1: What is Multimodal Learning?

#### 1.1 Definition

Multimodal learning processes and relates information from multiple modalities.

#### 1.2 Why Multimodal?

- **Complementary information**: Different modalities provide different views
- **Robustness**: One modality can compensate for another
- **Human-like**: Humans integrate multiple senses

---

### Chapter 2: Multimodal Data

#### 2.1 Data Types

```python
# Image + Caption dataset
from datasets import load_dataset

dataset = load_dataset("coco_captions")
sample = dataset['train'][0]
image = sample['image']
caption = sample['caption']
```

#### 2.2 Alignment

Ensuring different modalities refer to the same concept.

---

### Chapter 3: Basic Fusion Techniques

#### 3.1 Early Fusion

```python
# Concatenate features early
image_features = image_encoder(image)  # (batch, 512)
text_features = text_encoder(text)     # (batch, 256)

combined = torch.cat([image_features, text_features], dim=1)  # (batch, 768)
output = classifier(combined)
```

#### 3.2 Late Fusion

```python
# Separate predictions, then combine
image_pred = image_model(image)
text_pred = text_model(text)

final_pred = (image_pred + text_pred) / 2  # Average
```

#### 3.3 Attention-based Fusion

```python
class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
    
    def forward(self, query, key_value):
        # Query from one modality, K/V from another
        output, _ = self.attention(query, key_value, key_value)
        return output
```

---

## Part II: Intermediate Level

### Chapter 4: Vision-Language Models

#### 4.1 Image Captioning

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Generate caption
inputs = processor(images=image, return_tensors="pt")
output_ids = model.generate(**inputs)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

#### 4.2 Visual Question Answering (VQA)

```python
from transformers import ViltProcessor, ViltForQuestionAnswering

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Answer question about image
encoding = processor(image, question, return_tensors="pt")
outputs = model(**encoding)
answer = model.config.id2label[outputs.logits.argmax(-1).item()]
```

---

### Chapter 5: Cross-Modal Learning

#### 5.1 Cross-Modal Retrieval

```python
# Find images given text query, or text given image query
def cross_modal_retrieval(query_embedding, gallery_embeddings):
    similarities = cosine_similarity(query_embedding, gallery_embeddings)
    ranked_indices = similarities.argsort()[::-1]
    return ranked_indices
```

#### 5.2 Shared Embedding Space

```python
class SharedEmbeddingModel(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
    
    def forward(self, image, text):
        image_embed = self.image_encoder(image)
        text_embed = self.text_encoder(text)
        return image_embed, text_embed
```

---

### Chapter 6: Multimodal Transformers

#### 6.1 Architecture

```python
class MultimodalTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.image_embed = nn.Linear(image_dim, d_model)
        self.text_embed = nn.Embedding(vocab_size, d_model)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead),
            num_layers
        )
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, image, text):
        # Embed both modalities
        img_tokens = self.image_embed(image)
        txt_tokens = self.text_embed(text)
        
        # Concatenate and process
        combined = torch.cat([img_tokens, txt_tokens], dim=1)
        output = self.transformer(combined)
        
        return self.classifier(output[:, 0])  # CLS token
```

---

## Part III: Advanced Level

### Chapter 7: CLIP and Contrastive Learning

#### 7.1 CLIP (Contrastive Language-Image Pre-training)

```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Zero-shot classification
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)
```

#### 7.2 Contrastive Loss

```python
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    # Normalize
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Compute similarities
    logits = (image_embeds @ text_embeds.T) / temperature
    
    # Labels: diagonal elements are positives
    labels = torch.arange(len(logits)).to(logits.device)
    
    # Bidirectional loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2
```

---

### Chapter 8: Large Multimodal Models

#### 8.1 GPT-4V Style Models

```python
# Using multimodal LLM
from transformers import LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Process image and text together
inputs = processor(images=image, text="Describe this image:", return_tensors="pt")
output = model.generate(**inputs)
```

#### 8.2 Architecture Pattern

```
Image → Vision Encoder → Projection → LLM → Text
Text  ────────────────────────────────────→
```

---

### Chapter 9: Applications

| Application | Modalities | Task |
|-------------|------------|------|
| Image Captioning | Image + Text | Generate description |
| VQA | Image + Text | Answer questions |
| Text-to-Image | Text → Image | Generate images |
| Video Understanding | Video + Text | Describe/answer |
| Multimodal Search | Any | Retrieve across modalities |

---

## Summary

| Era | Approach | Examples |
|-----|----------|----------|
| Early | Feature concatenation | Early/Late fusion |
| Neural | Joint embeddings | ViLBERT |
| Pretrained | Contrastive learning | CLIP |
| LMM | Large multimodal models | GPT-4V, LLaVA |

---

**Last Updated**: 2024-01-29
