# Time Series Analysis: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: Time Series Fundamentals](#chapter-1-time-series-fundamentals)
  - [Chapter 2: Exploratory Analysis](#chapter-2-exploratory-analysis)
  - [Chapter 3: Basic Forecasting](#chapter-3-basic-forecasting)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: ARIMA Models](#chapter-4-arima-models)
  - [Chapter 5: Seasonal Decomposition](#chapter-5-seasonal-decomposition)
  - [Chapter 6: Feature Engineering for Time Series](#chapter-6-feature-engineering-for-time-series)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: Deep Learning for Time Series](#chapter-7-deep-learning-for-time-series)
  - [Chapter 8: Transformers for Time Series](#chapter-8-transformers-for-time-series)
  - [Chapter 9: Anomaly Detection](#chapter-9-anomaly-detection)

---

## Introduction

**Time Series Analysis** deals with data points collected over time, enabling forecasting and pattern detection.

### Applications

| Domain | Application |
|--------|-------------|
| Finance | Stock prediction, Risk |
| Weather | Forecasting |
| IoT | Sensor analysis |
| Business | Demand forecasting |

---

## Part I: Beginner Level

### Chapter 1: Time Series Fundamentals

#### 1.1 Components

```
Time Series = Trend + Seasonality + Residual
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create time series
dates = pd.date_range('2020-01-01', periods=365, freq='D')
ts = pd.Series(np.random.randn(365).cumsum(), index=dates)

ts.plot(figsize=(12, 4))
plt.title('Time Series')
plt.show()
```

#### 1.2 Stationarity

```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Stationary' if result[1] < 0.05 else 'Non-stationary')
```

---

### Chapter 2: Exploratory Analysis

#### 2.1 Autocorrelation

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, ax=axes[0])
plot_pacf(ts, ax=axes[1])
plt.show()
```

#### 2.2 Rolling Statistics

```python
# Moving average
rolling_mean = ts.rolling(window=30).mean()
rolling_std = ts.rolling(window=30).std()

plt.figure(figsize=(12, 4))
plt.plot(ts, label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.legend()
```

---

### Chapter 3: Basic Forecasting

#### 3.1 Naive Methods

```python
# Last value
naive_forecast = ts.shift(1)

# Seasonal naive
seasonal_naive = ts.shift(365)  # Same day last year

# Moving average
ma_forecast = ts.rolling(window=7).mean()
```

#### 3.2 Exponential Smoothing

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=7)
fitted = model.fit()
forecast = fitted.forecast(30)
```

---

## Part II: Intermediate Level

### Chapter 4: ARIMA Models

#### 4.1 ARIMA Components

- **AR(p)**: Autoregressive - past values
- **I(d)**: Integrated - differencing
- **MA(q)**: Moving Average - past errors

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(ts, order=(1, 1, 1))
fitted = model.fit()
print(fitted.summary())

# Forecast
forecast = fitted.forecast(steps=30)
```

#### 4.2 Auto ARIMA

```python
from pmdarima import auto_arima

model = auto_arima(ts, seasonal=True, m=7, 
                   suppress_warnings=True, stepwise=True)
print(model.summary())
```

---

### Chapter 5: Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts, model='additive', period=7)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
```

---

### Chapter 6: Feature Engineering for Time Series

#### 6.1 Lag Features

```python
def create_lag_features(df, column, lags):
    for lag in lags:
        df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

df = create_lag_features(df, 'value', [1, 7, 14, 30])
```

#### 6.2 Rolling Features

```python
def create_rolling_features(df, column, windows):
    for window in windows:
        df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window).mean()
        df[f'{column}_rolling_std_{window}'] = df[column].rolling(window).std()
    return df
```

#### 6.3 Date Features

```python
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
```

---

## Part III: Advanced Level

### Chapter 7: Deep Learning for Time Series

#### 7.1 LSTM

```python
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

#### 7.2 Temporal Convolutional Networks

```python
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                   dilation=dilation, padding=(kernel_size-1)*dilation))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
```

---

### Chapter 8: Transformers for Time Series

```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])
```

---

### Chapter 9: Anomaly Detection

```python
from sklearn.ensemble import IsolationForest

# Detect anomalies
clf = IsolationForest(contamination=0.01)
predictions = clf.fit_predict(ts.values.reshape(-1, 1))
anomalies = ts[predictions == -1]
```

---

## Summary

| Method | Type | Best For |
|--------|------|----------|
| ARIMA | Statistical | Univariate, linear |
| Exponential Smoothing | Statistical | Trend, seasonality |
| LSTM | Deep Learning | Long sequences |
| Transformer | Deep Learning | Complex patterns |
| Prophet | Hybrid | Business forecasting |

---

**Last Updated**: 2024-01-29
