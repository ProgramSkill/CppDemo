# Speech Processing: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: Audio Fundamentals](#chapter-1-audio-fundamentals)
  - [Chapter 2: Feature Extraction](#chapter-2-feature-extraction)
  - [Chapter 3: Speech Recognition Basics](#chapter-3-speech-recognition-basics)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: Deep Learning for Speech](#chapter-4-deep-learning-for-speech)
  - [Chapter 5: Text-to-Speech](#chapter-5-text-to-speech)
  - [Chapter 6: Speaker Recognition](#chapter-6-speaker-recognition)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: End-to-End Models](#chapter-7-end-to-end-models)
  - [Chapter 8: Whisper and Modern ASR](#chapter-8-whisper-and-modern-asr)
  - [Chapter 9: Voice Assistants](#chapter-9-voice-assistants)

---

## Introduction

**Speech Processing** enables machines to understand, generate, and manipulate human speech.

### Speech Tasks

| Task | Description |
|------|-------------|
| **ASR** | Automatic Speech Recognition |
| **TTS** | Text-to-Speech Synthesis |
| **Speaker Recognition** | Identify/verify speakers |
| **Speech Enhancement** | Noise reduction |
| **Voice Conversion** | Change voice characteristics |

---

## Part I: Beginner Level

### Chapter 1: Audio Fundamentals

#### 1.1 Digital Audio

```python
import librosa
import numpy as np

# Load audio
audio, sr = librosa.load('speech.wav', sr=16000)

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f} seconds")
print(f"Shape: {audio.shape}")
```

#### 1.2 Audio Visualization

```python
import librosa.display
import matplotlib.pyplot as plt

# Waveform
plt.figure(figsize=(12, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title('Waveform')

# Spectrogram
D = librosa.stft(audio)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
```

---

### Chapter 2: Feature Extraction

#### 2.1 Mel-Frequency Cepstral Coefficients (MFCCs)

```python
# Extract MFCCs
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
print(f"MFCC shape: {mfccs.shape}")  # (n_mfcc, time_frames)

# Visualize
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
```

#### 2.2 Mel Spectrogram

```python
# Mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
```

---

### Chapter 3: Speech Recognition Basics

#### 3.1 Traditional Pipeline

```
Audio → Feature Extraction → Acoustic Model → Language Model → Text
        (MFCC)              (HMM/DNN)        (N-gram)
```

#### 3.2 Using Pretrained Models

```python
import speech_recognition as sr

recognizer = sr.Recognizer()

with sr.AudioFile('speech.wav') as source:
    audio_data = recognizer.record(source)

# Using Google Speech Recognition
text = recognizer.recognize_google(audio_data)
print(f"Recognized: {text}")
```

---

## Part II: Intermediate Level

### Chapter 4: Deep Learning for Speech

#### 4.1 CNN for Audio

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

#### 4.2 RNN for Sequence Modeling

```python
class SpeechRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 
                           num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, time, features)
        output, _ = self.lstm(x)
        return self.fc(output)
```

---

### Chapter 5: Text-to-Speech

#### 5.1 TTS Pipeline

```
Text → Text Analysis → Acoustic Model → Vocoder → Audio
       (G2P)          (Tacotron)       (WaveNet)
```

#### 5.2 Using Modern TTS

```python
from TTS.api import TTS

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech
tts.tts_to_file(text="Hello, how are you?", file_path="output.wav")
```

---

### Chapter 6: Speaker Recognition

#### 6.1 Speaker Embedding

```python
from speechbrain.pretrained import EncoderClassifier

# Load pretrained model
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Get speaker embedding
embedding = classifier.encode_batch(waveform)
```

#### 6.2 Speaker Verification

```python
def verify_speaker(embedding1, embedding2, threshold=0.7):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity > threshold
```

---

## Part III: Advanced Level

### Chapter 7: End-to-End Models

#### 7.1 CTC (Connectionist Temporal Classification)

```python
import torch.nn as nn

class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, 
                              num_layers=3, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(self, x):
        output, _ = self.encoder(x)
        return self.fc(output)

# CTC Loss
ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
```

#### 7.2 Attention-based Models

```python
# Transformer-based ASR
class TransformerASR(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, vocab_size):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x)
```

---

### Chapter 8: Whisper and Modern ASR

```python
import whisper

# Load model
model = whisper.load_model("base")

# Transcribe
result = model.transcribe("audio.mp3")
print(result["text"])

# With language detection
result = model.transcribe("audio.mp3", language=None)
print(f"Detected language: {result['language']}")
```

---

### Chapter 9: Voice Assistants

#### Architecture

```
Wake Word Detection → ASR → NLU → Dialog Management → TTS → Response
```

---

## Summary

| Task | Traditional | Modern |
|------|-------------|--------|
| ASR | HMM + N-gram | Transformer (Whisper) |
| TTS | Concatenative | Neural (Tacotron, VITS) |
| Speaker | i-vector | ECAPA-TDNN |

---

**Last Updated**: 2024-01-29
