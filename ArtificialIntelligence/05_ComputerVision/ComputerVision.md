# Computer Vision: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: Image Fundamentals](#chapter-1-image-fundamentals)
  - [Chapter 2: Image Processing](#chapter-2-image-processing)
  - [Chapter 3: Image Classification](#chapter-3-image-classification)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: Object Detection](#chapter-4-object-detection)
  - [Chapter 5: Image Segmentation](#chapter-5-image-segmentation)
  - [Chapter 6: Transfer Learning](#chapter-6-transfer-learning)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: Advanced Architectures](#chapter-7-advanced-architectures)
  - [Chapter 8: Generative Models for Vision](#chapter-8-generative-models-for-vision)
  - [Chapter 9: Vision Transformers](#chapter-9-vision-transformers)

---

## Introduction

**Computer Vision** enables machines to interpret and understand visual information from the world.

### CV Tasks Overview

| Task | Description | Output |
|------|-------------|--------|
| **Classification** | What is in the image? | Class label |
| **Detection** | Where are objects? | Bounding boxes |
| **Segmentation** | Pixel-level labels | Mask |
| **Generation** | Create new images | Image |

---

## Part I: Beginner Level

### Chapter 1: Image Fundamentals

#### 1.1 Digital Images

```python
import cv2
import numpy as np

# Load image
img = cv2.imread('image.jpg')  # BGR format
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Shape: {img.shape}")  # (height, width, channels)
print(f"Dtype: {img.dtype}")  # uint8 (0-255)

# Access pixel
pixel = img[100, 100]  # [B, G, R]

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

#### 1.2 Image Transformations

```python
# Resize
resized = cv2.resize(img, (224, 224))

# Rotate
M = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)
rotated = cv2.warpAffine(img, M, (width, height))

# Flip
flipped_h = cv2.flip(img, 1)  # Horizontal
flipped_v = cv2.flip(img, 0)  # Vertical
```

---

### Chapter 2: Image Processing

#### 2.1 Filters and Convolutions

```python
# Blur
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# Edge detection
edges = cv2.Canny(gray, 100, 200)

# Custom kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])
filtered = cv2.filter2D(gray, -1, kernel)
```

#### 2.2 Data Augmentation

```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

### Chapter 3: Image Classification

#### 3.1 CNN for Classification

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

#### 3.2 Using Pretrained Models

```python
from torchvision import models

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

---

## Part II: Intermediate Level

### Chapter 4: Object Detection

#### 4.1 YOLO

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Inference
results = model('image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = box.cls[0]
```

#### 4.2 Faster R-CNN

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inference
predictions = model([image])
# predictions contains boxes, labels, scores
```

---

### Chapter 5: Image Segmentation

#### 5.1 Semantic Segmentation

```python
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(pretrained=True)
model.eval()

output = model(image)['out']
predictions = output.argmax(1)  # Per-pixel class
```

#### 5.2 Instance Segmentation

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn

model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

predictions = model([image])
# predictions contains boxes, labels, scores, masks
```

---

### Chapter 6: Transfer Learning

```python
# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final layers
for param in model.fc.parameters():
    param.requires_grad = True

# Fine-tune with lower learning rate
optimizer = torch.optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

---

## Part III: Advanced Level

### Chapter 7: Advanced Architectures

#### 7.1 ResNet

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
```

#### 7.2 U-Net for Segmentation

```python
class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2)
        self.dec1 = self.conv_block(192, 64)
        self.final = nn.Conv2d(64, n_classes, 1)
```

---

### Chapter 8: Generative Models for Vision

#### 8.1 Image Generation with Diffusion

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

image = pipe("a photo of an astronaut riding a horse").images[0]
```

---

### Chapter 9: Vision Transformers

#### 9.1 ViT Architecture

```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Patch embedding: image → patches → linear projection
# Position embedding added
# Transformer encoder processes
```

---

## Summary

| Task | Classic | Modern |
|------|---------|--------|
| Classification | VGG, ResNet | ViT, EfficientNet |
| Detection | R-CNN | YOLO, DETR |
| Segmentation | FCN | U-Net, Mask R-CNN |
| Generation | GAN | Diffusion |

---

**Last Updated**: 2024-01-29
