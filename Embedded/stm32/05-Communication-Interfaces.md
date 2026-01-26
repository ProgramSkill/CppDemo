# Chapter 5: Communication Interfaces

## 5.1 UART/USART

### Overview

UART (Universal Asynchronous Receiver/Transmitter) is one of the most common serial communication protocols.

**Key Features:**
- Asynchronous (no clock signal)
- Full-duplex communication
- Configurable baud rate
- Common baud rates: 9600, 115200, 921600

**USART vs UART:**
- USART: Synchronous + Asynchronous
- UART: Asynchronous only
- STM32 typically has USART peripherals

### UART Frame Format

```
Start  Data Bits    Parity  Stop
Bit    (5-9 bits)   (opt)   Bit(s)
 │         │          │       │
 ▼         ▼          ▼       ▼
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│0│D0│D1│D2│D3│D4│D5│D6│D7│P│1│
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
```

**Common Configuration: 8N1**
- 8 data bits
- No parity
- 1 stop bit

### UART Configuration with HAL

**Example: USART1 on PA9 (TX) and PA10 (RX)**

```c
UART_HandleTypeDef huart1;

void UART1_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable clocks
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure GPIO pins
    // PA9: USART1_TX
    // PA10: USART1_RX
    GPIO_InitStruct.Pin = GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Configure UART
    huart1.Instance = USART1;
    huart1.Init.BaudRate = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HWFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;

    HAL_UART_Init(&huart1);
}
```

### UART Transmission

**Blocking Mode:**
```c
uint8_t data[] = "Hello World\r\n";
HAL_UART_Transmit(&huart1, data, sizeof(data)-1, HAL_MAX_DELAY);
```

**Interrupt Mode:**
```c
uint8_t tx_data[] = "Hello\r\n";
HAL_UART_Transmit_IT(&huart1, tx_data, sizeof(tx_data)-1);

// Enable interrupt
HAL_NVIC_SetPriority(USART1_IRQn, 0, 0);
HAL_NVIC_EnableIRQ(USART1_IRQn);

// Interrupt handler
void USART1_IRQHandler(void)
{
    HAL_UART_IRQHandler(&huart1);
}

// Callback
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        // Transmission complete
    }
}
```

**DMA Mode:**
```c
uint8_t tx_buffer[100];
HAL_UART_Transmit_DMA(&huart1, tx_buffer, 100);
```

### UART Reception

**Blocking Mode:**
```c
uint8_t rx_data[10];
HAL_UART_Receive(&huart1, rx_data, 10, 1000);  // Timeout 1000ms
```

**Interrupt Mode:**
```c
uint8_t rx_buffer[1];

// Start reception
HAL_UART_Receive_IT(&huart1, rx_buffer, 1);

// Callback
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        // Process received byte
        uint8_t received = rx_buffer[0];

        // Echo back
        HAL_UART_Transmit(&huart1, &received, 1, 100);

        // Restart reception
        HAL_UART_Receive_IT(&huart1, rx_buffer, 1);
    }
}
```

### Printf Redirection

To use printf() with UART:

```c
#include <stdio.h>

// Retarget printf to UART
int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}

// Usage
printf("Counter: %d\r\n", counter);
```

## 5.2 SPI (Serial Peripheral Interface)

### Overview

SPI is a synchronous serial communication protocol for short-distance communication.

**Key Features:**
- Synchronous (clock signal)
- Full-duplex
- Master-slave architecture
- High speed (up to tens of MHz)

**SPI Signals:**
- **MOSI**: Master Out Slave In (data from master)
- **MISO**: Master In Slave Out (data from slave)
- **SCK**: Serial Clock
- **NSS/CS**: Chip Select (active low)

### SPI Modes

| Mode | CPOL | CPHA | Clock Polarity | Clock Phase |
|------|------|------|----------------|-------------|
| 0    | 0    | 0    | Idle Low       | Sample on rising edge |
| 1    | 0    | 1    | Idle Low       | Sample on falling edge |
| 2    | 1    | 0    | Idle High      | Sample on falling edge |
| 3    | 1    | 1    | Idle High      | Sample on rising edge |

### SPI Configuration

**Example: SPI1 Master Mode**

```c
SPI_HandleTypeDef hspi1;

void SPI1_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable clocks
    __HAL_RCC_SPI1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure GPIO pins
    // PA5: SPI1_SCK
    // PA6: SPI1_MISO
    // PA7: SPI1_MOSI
    GPIO_InitStruct.Pin = GPIO_PIN_5 | GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Configure CS pin (PA4) as GPIO output
    GPIO_InitStruct.Pin = GPIO_PIN_4;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);  // CS high (inactive)

    // Configure SPI
    hspi1.Instance = SPI1;
    hspi1.Init.Mode = SPI_MODE_MASTER;
    hspi1.Init.Direction = SPI_DIRECTION_2LINES;
    hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
    hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
    hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
    hspi1.Init.NSS = SPI_NSS_SOFT;
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;  // 72MHz/16 = 4.5MHz
    hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;

    HAL_SPI_Init(&hspi1);
}
```

### SPI Communication

**Transmit Data:**
```c
uint8_t tx_data[] = {0x01, 0x02, 0x03};

// Select slave (CS low)
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);

// Transmit
HAL_SPI_Transmit(&hspi1, tx_data, 3, HAL_MAX_DELAY);

// Deselect slave (CS high)
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
```

**Receive Data:**
```c
uint8_t rx_data[3];

HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);
HAL_SPI_Receive(&hspi1, rx_data, 3, HAL_MAX_DELAY);
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
```

**Transmit and Receive (Full-Duplex):**
```c
uint8_t tx_data[] = {0x01, 0x02, 0x03};
uint8_t rx_data[3];

HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);
HAL_SPI_TransmitReceive(&hspi1, tx_data, rx_data, 3, HAL_MAX_DELAY);
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
```

### Example: Reading from SPI Flash

```c
#define CMD_READ_ID     0x9F
#define CMD_READ_DATA   0x03

void SPI_Flash_ReadID(uint8_t *id)
{
    uint8_t cmd = CMD_READ_ID;

    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_RESET);
    HAL_SPI_Transmit(&hspi1, &cmd, 1, HAL_MAX_DELAY);
    HAL_SPI_Receive(&hspi1, id, 3, HAL_MAX_DELAY);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_4, GPIO_PIN_SET);
}
```

## 5.3 I2C (Inter-Integrated Circuit)

### Overview

I2C is a multi-master, multi-slave serial communication protocol.

**Key Features:**
- Synchronous (clock signal)
- Two wires: SDA (data), SCL (clock)
- Multiple devices on same bus
- 7-bit or 10-bit addressing
- Standard mode: 100 kHz, Fast mode: 400 kHz

**I2C Signals:**
- **SDA**: Serial Data (bidirectional)
- **SCL**: Serial Clock
- Both lines require pull-up resistors (typically 4.7kΩ)

### I2C Transaction Format

```
START | ADDR | R/W | ACK | DATA | ACK | ... | STOP
  S     7-bit   1    1     8-bit  1          P
```

**START Condition**: SDA falls while SCL is high
**STOP Condition**: SDA rises while SCL is high

### I2C Configuration

**Example: I2C1 on PB6 (SCL) and PB7 (SDA)**

```c
I2C_HandleTypeDef hi2c1;

void I2C1_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable clocks
    __HAL_RCC_I2C1_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    // Configure GPIO pins
    // PB6: I2C1_SCL
    // PB7: I2C1_SDA
    GPIO_InitStruct.Pin = GPIO_PIN_6 | GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;  // Open-drain
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    // Configure I2C
    hi2c1.Instance = I2C1;
    hi2c1.Init.ClockSpeed = 100000;  // 100 kHz
    hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
    hi2c1.Init.OwnAddress1 = 0;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;

    HAL_I2C_Init(&hi2c1);
}
```

### I2C Communication

**Write Data:**
```c
uint8_t device_addr = 0x50 << 1;  // 7-bit address shifted left
uint8_t data[] = {0x00, 0x01, 0x02};

HAL_I2C_Master_Transmit(&hi2c1, device_addr, data, 3, HAL_MAX_DELAY);
```

**Read Data:**
```c
uint8_t device_addr = 0x50 << 1;
uint8_t rx_data[3];

HAL_I2C_Master_Receive(&hi2c1, device_addr, rx_data, 3, HAL_MAX_DELAY);
```

**Write Register:**
```c
void I2C_WriteRegister(uint8_t dev_addr, uint8_t reg_addr, uint8_t value)
{
    uint8_t data[2] = {reg_addr, value};
    HAL_I2C_Master_Transmit(&hi2c1, dev_addr << 1, data, 2, HAL_MAX_DELAY);
}
```

**Read Register:**
```c
uint8_t I2C_ReadRegister(uint8_t dev_addr, uint8_t reg_addr)
{
    uint8_t value;

    // Write register address
    HAL_I2C_Master_Transmit(&hi2c1, dev_addr << 1, &reg_addr, 1, HAL_MAX_DELAY);

    // Read register value
    HAL_I2C_Master_Receive(&hi2c1, dev_addr << 1, &value, 1, HAL_MAX_DELAY);

    return value;
}
```

### Example: Reading from MPU6050 (Accelerometer/Gyroscope)

```c
#define MPU6050_ADDR    0x68
#define WHO_AM_I_REG    0x75
#define ACCEL_XOUT_H    0x3B

void MPU6050_Init(void)
{
    uint8_t data;

    // Wake up MPU6050
    data = 0x00;
    I2C_WriteRegister(MPU6050_ADDR, 0x6B, data);

    // Verify device ID
    data = I2C_ReadRegister(MPU6050_ADDR, WHO_AM_I_REG);
    if (data == 0x68) {
        // MPU6050 detected
    }
}

void MPU6050_ReadAccel(int16_t *ax, int16_t *ay, int16_t *az)
{
    uint8_t data[6];
    uint8_t reg = ACCEL_XOUT_H;

    // Read 6 bytes starting from ACCEL_XOUT_H
    HAL_I2C_Master_Transmit(&hi2c1, MPU6050_ADDR << 1, &reg, 1, HAL_MAX_DELAY);
    HAL_I2C_Master_Receive(&hi2c1, MPU6050_ADDR << 1, data, 6, HAL_MAX_DELAY);

    // Combine high and low bytes
    *ax = (int16_t)((data[0] << 8) | data[1]);
    *ay = (int16_t)((data[2] << 8) | data[3]);
    *az = (int16_t)((data[4] << 8) | data[5]);
}
```

## 5.4 CAN (Controller Area Network)

### Overview

CAN is a robust vehicle bus standard designed for microcontrollers and devices to communicate without a host computer.

**Key Features:**
- Multi-master protocol
- Message-based communication
- Built-in error detection
- Priority-based arbitration
- Differential signaling (CAN_H, CAN_L)

**CAN Speeds:**
- Low speed: 125 kbps
- High speed: 500 kbps, 1 Mbps

### CAN Frame Format

```
SOF | ID | RTR | Control | Data | CRC | ACK | EOF
```

**Standard Frame**: 11-bit identifier
**Extended Frame**: 29-bit identifier

### CAN Configuration

```c
CAN_HandleTypeDef hcan;

void CAN_Init(void)
{
    // Enable clock
    __HAL_RCC_CAN1_CLK_ENABLE();

    // Configure CAN
    hcan.Instance = CAN1;
    hcan.Init.Prescaler = 9;
    hcan.Init.Mode = CAN_MODE_NORMAL;
    hcan.Init.SyncJumpWidth = CAN_SJW_1TQ;
    hcan.Init.TimeSeg1 = CAN_BS1_6TQ;
    hcan.Init.TimeSeg2 = CAN_BS2_1TQ;
    hcan.Init.TimeTriggeredMode = DISABLE;
    hcan.Init.AutoBusOff = DISABLE;
    hcan.Init.AutoWakeUp = DISABLE;
    hcan.Init.AutoRetransmission = ENABLE;
    hcan.Init.ReceiveFifoLocked = DISABLE;
    hcan.Init.TransmitFifoPriority = DISABLE;

    HAL_CAN_Init(&hcan);

    // Configure filter
    CAN_FilterTypeDef filter;
    filter.FilterBank = 0;
    filter.FilterMode = CAN_FILTERMODE_IDMASK;
    filter.FilterScale = CAN_FILTERSCALE_32BIT;
    filter.FilterIdHigh = 0x0000;
    filter.FilterIdLow = 0x0000;
    filter.FilterMaskIdHigh = 0x0000;
    filter.FilterMaskIdLow = 0x0000;
    filter.FilterFIFOAssignment = CAN_RX_FIFO0;
    filter.FilterActivation = ENABLE;

    HAL_CAN_ConfigFilter(&hcan, &filter);
    HAL_CAN_Start(&hcan);
}
```

### CAN Transmission

```c
void CAN_Transmit(uint32_t id, uint8_t *data, uint8_t len)
{
    CAN_TxHeaderTypeDef TxHeader;
    uint32_t TxMailbox;

    TxHeader.StdId = id;
    TxHeader.ExtId = 0;
    TxHeader.RTR = CAN_RTR_DATA;
    TxHeader.IDE = CAN_ID_STD;
    TxHeader.DLC = len;
    TxHeader.TransmitGlobalTime = DISABLE;

    HAL_CAN_AddTxMessage(&hcan, &TxHeader, data, &TxMailbox);
}
```

## Next Steps

Proceed to [Chapter 6: Advanced Peripherals](06-Advanced-Peripherals.md) to learn about ADC, DAC, DMA, and more.
