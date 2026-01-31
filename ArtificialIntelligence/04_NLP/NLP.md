# Natural Language Processing: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: Text Preprocessing](#chapter-1-text-preprocessing)
  - [Chapter 2: Text Representation](#chapter-2-text-representation)
  - [Chapter 3: Basic NLP Tasks](#chapter-3-basic-nlp-tasks)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: Word Embeddings](#chapter-4-word-embeddings)
  - [Chapter 5: Sequence Models](#chapter-5-sequence-models)
  - [Chapter 6: Named Entity Recognition](#chapter-6-named-entity-recognition)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: Transformer Models](#chapter-7-transformer-models)
  - [Chapter 8: Large Language Models](#chapter-8-large-language-models)
  - [Chapter 9: Advanced Applications](#chapter-9-advanced-applications)

---

## Introduction

**Natural Language Processing (NLP)** enables computers to understand, interpret, and generate human language.

### NLP Applications

| Application | Description |
|-------------|-------------|
| **Text Classification** | Sentiment, spam, topic |
| **Named Entity Recognition** | Extract entities |
| **Machine Translation** | Language translation |
| **Question Answering** | Answer questions |
| **Text Generation** | Generate text |
| **Summarization** | Condense text |

---

## Part I: Beginner Level

### Chapter 1: Text Preprocessing

#### 1.1 Basic Steps

```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The quick brown foxes are jumping over the lazy dogs!"

# Lowercasing
text_lower = text.lower()

# Remove punctuation
text_clean = re.sub(r'[^\w\s]', '', text_lower)

# Tokenization
tokens = word_tokenize(text_clean)

# Remove stopwords
stop_words = set(stopwords.words('english'))
tokens_filtered = [t for t in tokens if t not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(t) for t in tokens_filtered]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(t) for t in tokens_filtered]

print(f"Original: {text}")
print(f"Tokens: {tokens}")
print(f"Filtered: {tokens_filtered}")
print(f"Stemmed: {stemmed}")
print(f"Lemmatized: {lemmatized}")
```

#### 1.2 Preprocessing Pipeline

```python
import spacy

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc 
              if not token.is_stop and not token.is_punct and token.is_alpha]
    return tokens

result = preprocess("The quick brown foxes are jumping!")
```

---

### Chapter 2: Text Representation

#### 2.1 Bag of Words

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "I love deep learning",
    "Machine learning is great"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("Matrix:\n", X.toarray())
```

#### 2.2 TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)

print("TF-IDF Matrix:\n", X_tfidf.toarray())
```

#### 2.3 N-grams

```python
# Bigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
X_bigram = bigram_vectorizer.fit_transform(corpus)

print("Bigrams:", bigram_vectorizer.get_feature_names_out())
```

---

### Chapter 3: Basic NLP Tasks

#### 3.1 Text Classification

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Create pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train
text_clf.fit(X_train, y_train)

# Predict
predictions = text_clf.predict(X_test)
```

#### 3.2 Sentiment Analysis

```python
from transformers import pipeline

# Using pretrained model
sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]
```

---

## Part II: Intermediate Level

### Chapter 4: Word Embeddings

#### 4.1 Word2Vec

```python
from gensim.models import Word2Vec

# Train Word2Vec
sentences = [["I", "love", "machine", "learning"],
             ["deep", "learning", "is", "great"]]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Get word vector
vector = model.wv['learning']

# Find similar words
similar = model.wv.most_similar('learning', topn=5)
```

#### 4.2 Pre-trained Embeddings

```python
import gensim.downloader as api

# Load pre-trained embeddings
glove = api.load('glove-wiki-gigaword-100')

# Word arithmetic
result = glove.most_similar(positive=['king', 'woman'], negative=['man'])
# Result: 'queen'
```

#### 4.3 Embedding Layer in PyTorch

```python
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)  # Average pooling
        return self.fc(pooled)
```

---

### Chapter 5: Sequence Models

#### 5.1 LSTM for Text

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(hidden)
```

#### 5.2 Seq2Seq

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return self.fc(output), hidden, cell
```

---

### Chapter 6: Named Entity Recognition

```python
import spacy

nlp = spacy.load('en_core_web_sm')

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")
# Apple Inc.: ORG
# Steve Jobs: PERSON
# Cupertino: GPE
# California: GPE
```

---

## Part III: Advanced Level

### Chapter 7: Transformer Models

#### 7.1 BERT

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# CLS token embedding
cls_embedding = outputs.last_hidden_state[:, 0, :]
```

#### 7.2 Fine-tuning BERT

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

---

### Chapter 8: Large Language Models

#### 8.1 Using GPT

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "Artificial intelligence is"
inputs = tokenizer(prompt, return_tensors='pt')

outputs = model.generate(
    inputs['input_ids'],
    max_length=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0])
```

#### 8.2 Prompt Engineering

```python
# Zero-shot
prompt = """Classify the sentiment of the following text as positive or negative.
Text: I love this movie!
Sentiment:"""

# Few-shot
prompt = """Classify sentiment:
Text: Great product! → positive
Text: Terrible service. → negative
Text: Amazing experience! →"""
```

---

### Chapter 9: Advanced Applications

#### 9.1 Question Answering

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = "BERT was developed by Google and released in 2018."
question = "Who developed BERT?"

result = qa_pipeline(question=question, context=context)
print(result['answer'])  # Google
```

#### 9.2 Text Summarization

```python
summarizer = pipeline("summarization")

text = """Long article text here..."""
summary = summarizer(text, max_length=100, min_length=30)
print(summary[0]['summary_text'])
```

#### 9.3 Machine Translation

```python
translator = pipeline("translation_en_to_de")
result = translator("Hello, how are you?")
print(result[0]['translation_text'])
```

---

## Summary

| Era | Approach | Examples |
|-----|----------|----------|
| Traditional | Rule-based, Statistical | Regex, HMM |
| Neural | RNN/LSTM | Seq2Seq |
| Transformer | Attention | BERT, GPT |
| LLM | Large-scale | ChatGPT, Claude |

---

**Last Updated**: 2024-01-29
