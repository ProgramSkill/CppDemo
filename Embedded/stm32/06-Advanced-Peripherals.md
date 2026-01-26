# Chapter 6: Advanced Peripherals

## 6.1 ADC (Analog-to-Digital Converter)

### Overview

The ADC converts analog voltage signals into digital values that the microcontroller can process.

**Key Features:**
- 12-bit resolution (0-4095)
- Multiple channels (up to 16 external + 2 internal)
- Conversion modes: single, continuous, scan, discontinuous
- DMA support
- Internal temperature sensor and VREFINT

**Voltage Range:**
- 0V to VREF+ (typically 3.3V)
- Digital value = (Vin / VREF+) × 4095

### ADC Configuration

**Example: Single Channel ADC (PA0)**

```c
ADC_HandleTypeDef hadc1;

void ADC1_Init(void)
{
    ADC_ChannelConfTypeDef sConfig = {0};
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable clocks
    __HAL_RCC_ADC1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure GPIO pin as analog
    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Configure ADC
    hadc1.Instance = ADC1;
    hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
    hadc1.Init.ContinuousConvMode = DISABLE;
    hadc1.Init.DiscontinuousConvMode = DISABLE;
    hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
    hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
    hadc1.Init.NbrOfConversion = 1;

    HAL_ADC_Init(&hadc1);

    // Configure channel
    sConfig.Channel = ADC_CHANNEL_0;
    sConfig.Rank = ADC_REGULAR_RANK_1;
    sConfig.SamplingTime = ADC_SAMPLETIME_55CYCLES_5;

    HAL_ADC_ConfigChannel(&hadc1, &sConfig);
}
```

### ADC Reading

**Single Conversion:**
```c
uint16_t ADC_Read(void)
{
    uint16_t value;

    // Start conversion
    HAL_ADC_Start(&hadc1);

    // Wait for conversion to complete
    HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);

    // Read value
    value = HAL_ADC_GetValue(&hadc1);

    return value;
}

// Convert to voltage
float ADC_ToVoltage(uint16_t adc_value)
{
    return (adc_value * 3.3f) / 4095.0f;
}
```

**Continuous Mode with DMA:**
```c
#define ADC_BUFFER_SIZE 100
uint16_t adc_buffer[ADC_BUFFER_SIZE];

void ADC_DMA_Init(void)
{
    // Configure ADC for continuous mode
    hadc1.Init.ContinuousConvMode = ENABLE;
    HAL_ADC_Init(&hadc1);

    // Start ADC with DMA
    HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buffer, ADC_BUFFER_SIZE);
}

// DMA callback
void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef* hadc)
{
    // Buffer filled, process data
    uint32_t sum = 0;
    for (int i = 0; i < ADC_BUFFER_SIZE; i++) {
        sum += adc_buffer[i];
    }
    uint16_t average = sum / ADC_BUFFER_SIZE;
}
```

### Multi-Channel ADC (Scan Mode)

```c
void ADC_MultiChannel_Init(void)
{
    ADC_ChannelConfTypeDef sConfig = {0};

    hadc1.Init.ScanConvMode = ADC_SCAN_ENABLE;
    hadc1.Init.NbrOfConversion = 3;  // 3 channels
    HAL_ADC_Init(&hadc1);

    // Channel 0 (PA0)
    sConfig.Channel = ADC_CHANNEL_0;
    sConfig.Rank = ADC_REGULAR_RANK_1;
    sConfig.SamplingTime = ADC_SAMPLETIME_55CYCLES_5;
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);

    // Channel 1 (PA1)
    sConfig.Channel = ADC_CHANNEL_1;
    sConfig.Rank = ADC_REGULAR_RANK_2;
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);

    // Channel 2 (PA2)
    sConfig.Channel = ADC_CHANNEL_2;
    sConfig.Rank = ADC_REGULAR_RANK_3;
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);
}
```

### Internal Temperature Sensor

```c
float ADC_ReadTemperature(void)
{
    ADC_ChannelConfTypeDef sConfig = {0};

    // Configure temperature sensor channel
    sConfig.Channel = ADC_CHANNEL_TEMPSENSOR;
    sConfig.Rank = ADC_REGULAR_RANK_1;
    sConfig.SamplingTime = ADC_SAMPLETIME_239CYCLES_5;  // Longer sampling time
    HAL_ADC_ConfigChannel(&hadc1, &sConfig);

    // Read ADC value
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
    uint16_t adc_value = HAL_ADC_GetValue(&hadc1);

    // Convert to temperature (formula from datasheet)
    float voltage = (adc_value * 3.3f) / 4095.0f;
    float temperature = ((voltage - 0.76f) / 0.0025f) + 25.0f;

    return temperature;
}
```

## 6.2 DAC (Digital-to-Analog Converter)

### Overview

The DAC converts digital values to analog voltage output.

**Key Features:**
- 12-bit resolution
- 2 output channels (PA4, PA5)
- Output range: 0V to VREF+ (typically 3.3V)
- DMA support for waveform generation

### DAC Configuration

```c
DAC_HandleTypeDef hdac;

void DAC_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable clocks
    __HAL_RCC_DAC_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure PA4 as analog
    GPIO_InitStruct.Pin = GPIO_PIN_4;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Configure DAC
    hdac.Instance = DAC;
    HAL_DAC_Init(&hdac);

    DAC_ChannelConfTypeDef sConfig = {0};
    sConfig.DAC_Trigger = DAC_TRIGGER_NONE;
    sConfig.DAC_OutputBuffer = DAC_OUTPUTBUFFER_ENABLE;

    HAL_DAC_ConfigChannel(&hdac, &sConfig, DAC_CHANNEL_1);
    HAL_DAC_Start(&hdac, DAC_CHANNEL_1);
}
```

### DAC Output

```c
void DAC_SetVoltage(float voltage)
{
    // Convert voltage to 12-bit value
    uint16_t dac_value = (uint16_t)((voltage / 3.3f) * 4095.0f);

    // Set DAC output
    HAL_DAC_SetValue(&hdac, DAC_CHANNEL_1, DAC_ALIGN_12B_R, dac_value);
}

// Example: Output 1.65V (half of 3.3V)
DAC_SetVoltage(1.65f);
```

### Waveform Generation

**Sine Wave:**
```c
#define SINE_SAMPLES 100
uint16_t sine_wave[SINE_SAMPLES];

void Generate_SineWave(void)
{
    for (int i = 0; i < SINE_SAMPLES; i++) {
        float angle = (2.0f * 3.14159f * i) / SINE_SAMPLES;
        float value = (sinf(angle) + 1.0f) / 2.0f;  // Normalize to 0-1
        sine_wave[i] = (uint16_t)(value * 4095.0f);
    }

    // Output with timer trigger and DMA
    HAL_DAC_Start_DMA(&hdac, DAC_CHANNEL_1, (uint32_t*)sine_wave,
                      SINE_SAMPLES, DAC_ALIGN_12B_R);
}
```

## 6.3 DMA (Direct Memory Access)

### Overview

DMA allows peripherals to transfer data to/from memory without CPU intervention.

**Benefits:**
- Reduces CPU load
- Enables background data transfers
- Improves system performance

**Common Uses:**
- ADC data acquisition
- UART/SPI/I2C data transfer
- Memory-to-memory transfers
- DAC waveform generation

### DMA Configuration

**Example: UART TX with DMA**

```c
DMA_HandleTypeDef hdma_usart1_tx;

void UART_DMA_Init(void)
{
    // Enable DMA clock
    __HAL_RCC_DMA1_CLK_ENABLE();

    // Configure DMA
    hdma_usart1_tx.Instance = DMA1_Channel4;
    hdma_usart1_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_usart1_tx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart1_tx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart1_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart1_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart1_tx.Init.Mode = DMA_NORMAL;
    hdma_usart1_tx.Init.Priority = DMA_PRIORITY_LOW;

    HAL_DMA_Init(&hdma_usart1_tx);

    // Link DMA to UART
    __HAL_LINKDMA(&huart1, hdmatx, hdma_usart1_tx);

    // Enable DMA interrupt
    HAL_NVIC_SetPriority(DMA1_Channel4_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Channel4_IRQn);
}

// DMA interrupt handler
void DMA1_Channel4_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&hdma_usart1_tx);
}

// Usage
uint8_t tx_buffer[100] = "Hello DMA\r\n";
HAL_UART_Transmit_DMA(&huart1, tx_buffer, strlen((char*)tx_buffer));
```

### Memory-to-Memory Transfer

```c
void DMA_MemCopy(uint32_t *src, uint32_t *dst, uint32_t size)
{
    DMA_HandleTypeDef hdma;

    hdma.Instance = DMA1_Channel1;
    hdma.Init.Direction = DMA_MEMORY_TO_MEMORY;
    hdma.Init.PeriphInc = DMA_PINC_ENABLE;
    hdma.Init.MemInc = DMA_MINC_ENABLE;
    hdma.Init.PeriphDataAlignment = DMA_PDATAALIGN_WORD;
    hdma.Init.MemDataAlignment = DMA_MDATAALIGN_WORD;
    hdma.Init.Mode = DMA_NORMAL;
    hdma.Init.Priority = DMA_PRIORITY_HIGH;

    HAL_DMA_Init(&hdma);
    HAL_DMA_Start(&hdma, (uint32_t)src, (uint32_t)dst, size);
    HAL_DMA_PollForTransfer(&hdma, HAL_DMA_FULL_TRANSFER, HAL_MAX_DELAY);
}
```

## 6.4 RTC (Real-Time Clock)

### Overview

The RTC provides calendar and time-keeping functions.

**Features:**
- Calendar with date and time
- Alarm functionality
- Backup domain (powered by VBAT)
- Wakeup from low-power modes

### RTC Configuration

```c
RTC_HandleTypeDef hrtc;

void RTC_Init(void)
{
    // Enable PWR and BKP clocks
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_RCC_BKP_CLK_ENABLE();

    // Enable access to backup domain
    HAL_PWR_EnableBkUpAccess();

    // Configure RTC
    hrtc.Instance = RTC;
    hrtc.Init.AsynchPrediv = 127;
    hrtc.Init.OutPut = RTC_OUTPUTSOURCE_NONE;

    HAL_RTC_Init(&hrtc);
}

void RTC_SetTime(uint8_t hours, uint8_t minutes, uint8_t seconds)
{
    RTC_TimeTypeDef sTime = {0};

    sTime.Hours = hours;
    sTime.Minutes = minutes;
    sTime.Seconds = seconds;

    HAL_RTC_SetTime(&hrtc, &sTime, RTC_FORMAT_BIN);
}

void RTC_SetDate(uint8_t year, uint8_t month, uint8_t date, uint8_t weekday)
{
    RTC_DateTypeDef sDate = {0};

    sDate.Year = year;
    sDate.Month = month;
    sDate.Date = date;
    sDate.WeekDay = weekday;

    HAL_RTC_SetDate(&hrtc, &sDate, RTC_FORMAT_BIN);
}

void RTC_GetDateTime(void)
{
    RTC_TimeTypeDef sTime;
    RTC_DateTypeDef sDate;

    HAL_RTC_GetTime(&hrtc, &sTime, RTC_FORMAT_BIN);
    HAL_RTC_GetDate(&hrtc, &sDate, RTC_FORMAT_BIN);

    printf("%02d/%02d/%02d %02d:%02d:%02d\r\n",
           sDate.Year, sDate.Month, sDate.Date,
           sTime.Hours, sTime.Minutes, sTime.Seconds);
}
```

## 6.5 Watchdog Timers

### Independent Watchdog (IWDG)

The IWDG is used to detect and recover from software failures.

```c
IWDG_HandleTypeDef hiwdg;

void IWDG_Init(void)
{
    hiwdg.Instance = IWDG;
    hiwdg.Init.Prescaler = IWDG_PRESCALER_64;
    hiwdg.Init.Reload = 4095;  // Maximum value

    HAL_IWDG_Init(&hiwdg);
}

// Refresh watchdog (must be called periodically)
void IWDG_Refresh(void)
{
    HAL_IWDG_Refresh(&hiwdg);
}
```

### Window Watchdog (WWDG)

```c
WWDG_HandleTypeDef hwwdg;

void WWDG_Init(void)
{
    hwwdg.Instance = WWDG;
    hwwdg.Init.Prescaler = WWDG_PRESCALER_8;
    hwwdg.Init.Window = 80;
    hwwdg.Init.Counter = 127;
    hwwdg.Init.EWIMode = WWDG_EWI_ENABLE;

    HAL_WWDG_Init(&hwwdg);
}
```

## 6.6 Low Power Modes

### Power Modes

**Sleep Mode:**
- CPU stopped
- Peripherals running
- Wakeup: Any interrupt

**Stop Mode:**
- All clocks stopped except LSI/LSE
- SRAM and registers retained
- Wakeup: EXTI, RTC

**Standby Mode:**
- Lowest power consumption
- Only backup domain active
- Wakeup: WKUP pin, RTC alarm, NRST

### Entering Low Power Modes

```c
// Enter Sleep Mode
HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI);

// Enter Stop Mode
HAL_PWR_EnterSTOPMode(PWR_LOWPOWERREGULATOR_ON, PWR_STOPENTRY_WFI);

// Enter Standby Mode
HAL_PWR_EnterSTANDBYMode();
```

### Wakeup from Standby

```c
void Standby_Wakeup_Init(void)
{
    // Enable WKUP pin
    HAL_PWR_EnableWakeUpPin(PWR_WAKEUP_PIN1);

    // Check if woken from standby
    if (__HAL_PWR_GET_FLAG(PWR_FLAG_SB) != RESET) {
        __HAL_PWR_CLEAR_FLAG(PWR_FLAG_SB);
        // System woke from standby
    }
}
```

## Next Steps

Proceed to [Chapter 7: RTOS Development](07-RTOS-Development.md) to learn about real-time operating systems on STM32.
