# GPIO - General Purpose Input/Output
# GPIO - 通用输入输出

## 1. Overview / 概述

### What is GPIO? / 什么是GPIO？

GPIO (General Purpose Input/Output) is the most fundamental peripheral in STM32 microcontrollers. It allows the MCU to interact with external devices by reading digital signals (input) or controlling external circuits (output).

GPIO（通用输入输出）是STM32微控制器中最基础的外设。它允许MCU通过读取数字信号（输入）或控制外部电路（输出）与外部设备交互。

### Key Features / 主要特性

- **Configurable I/O modes**: Input, Output, Alternate Function, Analog
- **可配置的I/O模式**：输入、输出、复用功能、模拟

- **Multiple output types**: Push-pull, Open-drain
- **多种输出类型**：推挽、开漏

- **Programmable pull-up/pull-down resistors**
- **可编程上拉/下拉电阻**

- **High output drive capability**: Up to 25mA per pin (typical)
- **高输出驱动能力**：每个引脚最高25mA（典型值）

- **Fast switching speed**: Up to 50MHz (depends on configuration)
- **快速开关速度**：最高50MHz（取决于配置）

- **Interrupt capability**: External interrupt on any GPIO pin
- **中断能力**：任何GPIO引脚都可配置外部中断

### Application Scenarios / 应用场景

- LED control / LED控制
- Button/switch input / 按键/开关输入
- Relay control / 继电器控制
- Digital sensor interface / 数字传感器接口
- Communication protocol bit-banging / 通信协议位操作
- External device control / 外部设备控制

---

## 2. Hardware Architecture / 硬件架构

### GPIO Port Structure / GPIO端口结构

STM32 GPIO pins are organized into ports (GPIOA, GPIOB, GPIOC, etc.), with each port containing up to 16 pins (Pin 0 to Pin 15).

STM32的GPIO引脚按端口组织（GPIOA、GPIOB、GPIOC等），每个端口包含最多16个引脚（Pin 0到Pin 15）。

**Example / 示例:**
- GPIOA: PA0, PA1, PA2, ..., PA15
- GPIOB: PB0, PB1, PB2, ..., PB15
- GPIOC: PC0, PC1, PC2, ..., PC15

### GPIO Pin Modes / GPIO引脚模式

#### 1. Input Mode / 输入模式

**Input Floating (浮空输入):**
- No pull-up or pull-down resistor
- Pin voltage determined by external circuit
- Use case: Reading external signals with defined logic levels

**Input Pull-up (上拉输入):**
- Internal pull-up resistor enabled (~40kΩ)
- Pin reads HIGH when not connected
- Use case: Button input (active-low)

**Input Pull-down (下拉输入):**
- Internal pull-down resistor enabled (~40kΩ)
- Pin reads LOW when not connected
- Use case: Button input (active-high)

#### 2. Output Mode / 输出模式

**Output Push-Pull (推挽输出):**
- Can actively drive both HIGH and LOW
- Strong output drive capability
- Use case: LED control, relay control

**Output Open-Drain (开漏输出):**
- Can actively drive LOW, HIGH is floating
- Requires external pull-up resistor
- Use case: I2C bus, multi-device communication

#### 3. Alternate Function Mode / 复用功能模式

- GPIO pin used by peripheral (UART, SPI, I2C, Timer, etc.)
- Pin function determined by peripheral configuration
- Use case: Serial communication, PWM output

#### 4. Analog Mode / 模拟模式

- Pin connected to ADC/DAC
- Digital input/output disabled
- Use case: Analog signal acquisition, analog output

### GPIO Speed Configuration / GPIO速度配置

| Speed | Frequency | Use Case |
|-------|-----------|----------|
| Low | 2 MHz | Low-speed I/O, reduces EMI |
| Medium | 25 MHz | General-purpose I/O |
| High | 50 MHz | High-speed communication |
| Very High | 100 MHz | Very high-speed applications (STM32F4/F7) |

| 速度 | 频率 | 使用场景 |
|------|------|----------|
| 低速 | 2 MHz | 低速I/O，减少EMI干扰 |
| 中速 | 25 MHz | 通用I/O |
| 高速 | 50 MHz | 高速通信 |
| 超高速 | 100 MHz | 超高速应用（STM32F4/F7） |

---

## 3. Register Description / 寄存器说明

### Key GPIO Registers / 主要GPIO寄存器

| Register | Full Name | Description |
|----------|-----------|-------------|
| GPIOx_MODER | Mode Register | Configure pin mode (Input/Output/AF/Analog) |
| GPIOx_OTYPER | Output Type Register | Configure output type (Push-pull/Open-drain) |
| GPIOx_OSPEEDR | Output Speed Register | Configure output speed |
| GPIOx_PUPDR | Pull-up/Pull-down Register | Configure pull-up/pull-down resistors |
| GPIOx_IDR | Input Data Register | Read input pin state |
| GPIOx_ODR | Output Data Register | Write output pin state |
| GPIOx_BSRR | Bit Set/Reset Register | Atomic set/reset individual pins |
| GPIOx_LCKR | Lock Register | Lock pin configuration |
| GPIOx_AFRL/AFRH | Alternate Function Register | Configure alternate function |

### Register Configuration Examples / 寄存器配置示例

**Configure PA5 as Output Push-Pull:**
```c
// Enable GPIOA clock
RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;

// Set PA5 as output mode (01)
GPIOA->MODER &= ~(0x3 << (5 * 2));  // Clear bits
GPIOA->MODER |= (0x1 << (5 * 2));   // Set as output

// Set PA5 as push-pull output (0)
GPIOA->OTYPER &= ~(0x1 << 5);

// Set PA5 speed to medium (01)
GPIOA->OSPEEDR &= ~(0x3 << (5 * 2));
GPIOA->OSPEEDR |= (0x1 << (5 * 2));

// No pull-up/pull-down (00)
GPIOA->PUPDR &= ~(0x3 << (5 * 2));
```

---

## 4. HAL Library Configuration / HAL库配置

### GPIO Initialization Structure / GPIO初始化结构体

```c
typedef struct {
    uint32_t Pin;       // GPIO pin number (GPIO_PIN_0 to GPIO_PIN_15)
    uint32_t Mode;      // GPIO mode (Input/Output/AF/Analog)
    uint32_t Pull;      // Pull-up/Pull-down configuration
    uint32_t Speed;     // Output speed
    uint32_t Alternate; // Alternate function selection (for AF mode)
} GPIO_InitTypeDef;
```

### Common HAL Functions / 常用HAL函数

```c
// Initialize GPIO pin
void HAL_GPIO_Init(GPIO_TypeDef *GPIOx, GPIO_InitTypeDef *GPIO_Init);

// Read input pin state
GPIO_PinState HAL_GPIO_ReadPin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);

// Write output pin state
void HAL_GPIO_WritePin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin, GPIO_PinState PinState);

// Toggle output pin
void HAL_GPIO_TogglePin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);

// Lock pin configuration
HAL_StatusTypeDef HAL_GPIO_LockPin(GPIO_TypeDef *GPIOx, uint16_t GPIO_Pin);
```

---

## 5. Programming Examples / 编程示例

### Example 1: LED Blink (Output) / 示例1：LED闪烁（输出）

**Hardware Connection / 硬件连接:**
- LED connected to PA5 (with current-limiting resistor)
- LED正极连接PA5（串联限流电阻），负极接地

**Code / 代码:**

```c
#include "stm32f4xx_hal.h"

void GPIO_LED_Init(void) {
    // Enable GPIOA clock
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure PA5 as output
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;  // Push-pull output
    GPIO_InitStruct.Pull = GPIO_NOPULL;          // No pull-up/pull-down
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW; // Low speed
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    GPIO_LED_Init();

    while (1) {
        HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);  // Toggle LED
        HAL_Delay(500);                          // 500ms delay
    }
}
```

### Example 2: Button Input (Input with Pull-up) / 示例2：按键输入（上拉输入）

**Hardware Connection / 硬件连接:**
- Button connected between PC13 and GND
- 按键一端连接PC13，另一端接地

**Code / 代码:**

```c
#include "stm32f4xx_hal.h"

void GPIO_Button_Init(void) {
    // Enable GPIOC clock
    __HAL_RCC_GPIOC_CLK_ENABLE();

    // Configure PC13 as input with pull-up
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;      // Input mode
    GPIO_InitStruct.Pull = GPIO_PULLUP;          // Pull-up enabled
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    GPIO_LED_Init();    // Initialize LED on PA5
    GPIO_Button_Init(); // Initialize button on PC13

    while (1) {
        // Read button state (active-low)
        if (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET) {
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);  // LED ON
        } else {
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET); // LED OFF
        }
    }
}
```

---

## 6. Common Pitfalls and Solutions / 常见问题与解决方案

### Pitfall 1: Forgot to Enable GPIO Clock / 忘记使能GPIO时钟

**Problem / 问题:**
```c
// Wrong: GPIO clock not enabled
GPIO_InitTypeDef GPIO_InitStruct = {0};
GPIO_InitStruct.Pin = GPIO_PIN_5;
HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);  // Won't work!
```

**Solution / 解决方案:**
```c
// Correct: Enable clock first
__HAL_RCC_GPIOA_CLK_ENABLE();  // Must enable clock before configuration
GPIO_InitTypeDef GPIO_InitStruct = {0};
GPIO_InitStruct.Pin = GPIO_PIN_5;
HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
```

### Pitfall 2: Incorrect Pull-up/Pull-down Configuration / 上拉/下拉配置错误

**Problem / 问题:**
- Button connected to GND, but configured with pull-down
- Button reads unstable values

**Solution / 解决方案:**
- Button to GND → Use pull-up (GPIO_PULLUP)
- Button to VCC → Use pull-down (GPIO_PULLDOWN)

### Pitfall 3: Output Drive Capability Exceeded / 输出驱动能力超限

**Problem / 问题:**
- Connecting high-current load directly to GPIO pin
- Pin may be damaged or output voltage drops

**Solution / 解决方案:**
- Use transistor or MOSFET for high-current loads
- Maximum current per pin: ~25mA (typical)
- Maximum total current per port: ~120mA (typical)

---

## 7. Best Practices / 最佳实践

1. **Always enable GPIO clock before configuration**
   **配置前务必使能GPIO时钟**

2. **Use appropriate pull-up/pull-down for inputs**
   **为输入引脚配置合适的上拉/下拉**

3. **Select appropriate output speed to reduce EMI**
   **选择合适的输出速度以减少EMI干扰**

4. **Use BSRR register for atomic bit operations**
   **使用BSRR寄存器进行原子位操作**

5. **Add debouncing for button inputs**
   **为按键输入添加消抖处理**

6. **Use external drivers for high-current loads**
   **高电流负载使用外部驱动器**

---

## Next Steps / 下一步

- Learn about **EXTI (External Interrupts)** for interrupt-driven GPIO input
- 学习**EXTI（外部中断）**实现中断驱动的GPIO输入

- Explore **Alternate Function** mode for peripheral usage
- 探索**复用功能**模式用于外设

- Study **DMA** for efficient GPIO operations
- 学习**DMA**实现高效的GPIO操作

---

**Document Version**: 1.0
**Last Updated**: 2026-01-29
**Related Topics**: EXTI, Timer, DMA
