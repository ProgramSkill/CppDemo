# Chapter 4: Basic Programming

## 4.1 STM32 Programming Fundamentals

### Memory Map

STM32 uses memory-mapped I/O, where peripherals are accessed through specific memory addresses.

**Key Memory Regions:**
```
0x0800 0000 - 0x080X XXXX : Flash Memory (program storage)
0x2000 0000 - 0x2000 XXXX : SRAM (data storage)
0x4000 0000 - 0x5FFF FFFF : Peripherals
0xE000 0000 - 0xE00F FFFF : Cortex-M core peripherals
```

### Register Access

Peripherals are controlled by writing to registers at specific addresses.

**Example: GPIO Register Structure**
```c
typedef struct {
    volatile uint32_t CRL;      // Port configuration register low
    volatile uint32_t CRH;      // Port configuration register high
    volatile uint32_t IDR;      // Input data register
    volatile uint32_t ODR;      // Output data register
    volatile uint32_t BSRR;     // Bit set/reset register
    volatile uint32_t BRR;      // Bit reset register
    volatile uint32_t LCKR;     // Port configuration lock register
} GPIO_TypeDef;

#define GPIOC_BASE  0x40011000
#define GPIOC       ((GPIO_TypeDef *) GPIOC_BASE)
```

## 4.2 Clock System

### Clock Tree Overview

STM32 has multiple clock sources and a complex clock tree for flexibility and power optimization.

**Clock Sources:**
- **HSI**: High-Speed Internal (8 MHz RC oscillator)
- **HSE**: High-Speed External (crystal/resonator)
- **LSI**: Low-Speed Internal (40 kHz for watchdog)
- **LSE**: Low-Speed External (32.768 kHz for RTC)
- **PLL**: Phase-Locked Loop (frequency multiplication)

**STM32F1 Clock Tree (Simplified):**
```
HSE (8 MHz) ──┬──> AHB Prescaler ──> HCLK (72 MHz max)
              │                       │
              └──> PLL (×9) ─────────┤
                                      ├──> APB1 Prescaler ──> PCLK1 (36 MHz max)
                                      │
                                      └──> APB2 Prescaler ──> PCLK2 (72 MHz max)
```

### Clock Configuration with HAL

**Example: Configure System Clock to 72 MHz**
```c
void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    // Configure HSE oscillator
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;  // 8 MHz × 9 = 72 MHz

    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    // Configure system clocks
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;   // HCLK = 72 MHz
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;    // PCLK1 = 36 MHz
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;    // PCLK2 = 72 MHz

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) {
        Error_Handler();
    }
}
```

### Peripheral Clock Enable

Before using any peripheral, enable its clock:

```c
// Enable GPIO port C clock
__HAL_RCC_GPIOC_CLK_ENABLE();

// Enable USART1 clock
__HAL_RCC_USART1_CLK_ENABLE();

// Enable TIM2 clock
__HAL_RCC_TIM2_CLK_ENABLE();
```

## 4.3 GPIO Programming

### GPIO Modes

**Output Modes:**
- Push-Pull: Can drive high and low
- Open-Drain: Can only pull low (needs external pull-up)

**Input Modes:**
- Floating: High impedance
- Pull-Up: Internal pull-up resistor
- Pull-Down: Internal pull-down resistor
- Analog: For ADC/DAC

**Alternate Function:**
- Used for peripheral functions (UART, SPI, etc.)

### GPIO Configuration with HAL

**Example: Configure PC13 as Output (LED)**
```c
void LED_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable GPIOC clock
    __HAL_RCC_GPIOC_CLK_ENABLE();

    // Configure PC13 as output
    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;  // Push-pull output
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}
```

**Example: Configure PA0 as Input (Button)**
```c
void Button_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable GPIOA clock
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure PA0 as input with pull-up
    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_PULLUP;

    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}
```

### GPIO Operations

**Write to GPIO:**
```c
// Set pin high
HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET);

// Set pin low
HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET);

// Toggle pin
HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
```

**Read from GPIO:**
```c
// Read pin state
GPIO_PinState state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0);

if (state == GPIO_PIN_SET) {
    // Pin is high
} else {
    // Pin is low
}
```

### Complete LED Blink Example

```c
#include "main.h"

void SystemClock_Config(void);
void LED_Init(void);

int main(void)
{
    // Initialize HAL library
    HAL_Init();

    // Configure system clock
    SystemClock_Config();

    // Initialize LED
    LED_Init();

    while (1)
    {
        // Toggle LED
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);

        // Delay 500ms
        HAL_Delay(500);
    }
}

void LED_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOC_CLK_ENABLE();

    GPIO_InitStruct.Pin = GPIO_PIN_13;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);
}
```

## 4.4 Interrupts and NVIC

### Interrupt System

STM32 uses the Nested Vectored Interrupt Controller (NVIC) for interrupt management.

**Key Concepts:**
- **Interrupt Vector**: Address of interrupt handler function
- **Priority**: Determines which interrupt executes first
- **Preemption**: Higher priority can interrupt lower priority
- **Nesting**: Interrupts can be nested based on priority

### Interrupt Priority

**STM32F1 Priority Levels:**
- 4-bit priority: 16 levels (0-15)
- Lower number = higher priority
- Priority 0 is highest

**Priority Grouping:**
```c
// Set priority grouping
HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_4);

// 4 bits for preemption priority, 0 bits for sub-priority
```

### GPIO Interrupt (EXTI)

**Example: Button Interrupt on PA0**
```c
void Button_Interrupt_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable GPIOA clock
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure PA0 as interrupt input
    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;  // Interrupt on falling edge
    GPIO_InitStruct.Pull = GPIO_PULLUP;

    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Configure NVIC
    HAL_NVIC_SetPriority(EXTI0_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(EXTI0_IRQn);
}

// Interrupt handler
void EXTI0_IRQHandler(void)
{
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);
}

// Callback function
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == GPIO_PIN_0) {
        // Button pressed - toggle LED
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
    }
}
```

### Interrupt Best Practices

1. **Keep ISRs Short**
   - Minimize code in interrupt handlers
   - Set flags and process in main loop

2. **Avoid Blocking Operations**
   - No delays in ISRs
   - No printf or long operations

3. **Use Volatile Variables**
   ```c
   volatile uint8_t flag = 0;

   void ISR_Handler(void) {
       flag = 1;  // Set flag
   }

   int main(void) {
       while (1) {
           if (flag) {
               flag = 0;
               // Process event
           }
       }
   }
   ```

4. **Disable Interrupts When Needed**
   ```c
   __disable_irq();  // Disable all interrupts
   // Critical section
   __enable_irq();   // Enable interrupts
   ```

## 4.5 Timers

### Timer Types

**Basic Timers (TIM6, TIM7)**
- Simple up-counting
- Used for time base generation

**General-Purpose Timers (TIM2-TIM5)**
- Up/down/up-down counting
- Input capture, output compare
- PWM generation

**Advanced Timers (TIM1, TIM8)**
- All general-purpose features
- Complementary outputs
- Dead-time insertion
- Motor control applications

### Timer Configuration

**Example: TIM2 for 1ms Interrupt**
```c
TIM_HandleTypeDef htim2;

void TIM2_Init(void)
{
    // Enable TIM2 clock
    __HAL_RCC_TIM2_CLK_ENABLE();

    // Configure timer
    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 7200 - 1;     // 72 MHz / 7200 = 10 kHz
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = 10 - 1;          // 10 kHz / 10 = 1 kHz (1ms)
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;

    HAL_TIM_Base_Init(&htim2);

    // Enable interrupt
    HAL_NVIC_SetPriority(TIM2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(TIM2_IRQn);

    // Start timer
    HAL_TIM_Base_Start_IT(&htim2);
}

// Interrupt handler
void TIM2_IRQHandler(void)
{
    HAL_TIM_IRQHandler(&htim2);
}

// Callback function
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM2) {
        // Called every 1ms
        // Toggle LED or increment counter
    }
}
```

### PWM Generation

**Example: PWM on TIM3 Channel 1 (PA6)**
```c
TIM_HandleTypeDef htim3;

void PWM_Init(void)
{
    TIM_OC_InitTypeDef sConfigOC = {0};
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Enable clocks
    __HAL_RCC_TIM3_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Configure PA6 as alternate function (TIM3_CH1)
    GPIO_InitStruct.Pin = GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Configure timer for PWM
    htim3.Instance = TIM3;
    htim3.Init.Prescaler = 72 - 1;       // 72 MHz / 72 = 1 MHz
    htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim3.Init.Period = 1000 - 1;        // 1 MHz / 1000 = 1 kHz PWM
    htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;

    HAL_TIM_PWM_Init(&htim3);

    // Configure PWM channel
    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = 500;               // 50% duty cycle
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

    HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1);

    // Start PWM
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
}

// Change duty cycle
void PWM_SetDutyCycle(uint16_t duty)
{
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, duty);
}
```

## 4.6 Delay Functions

### HAL_Delay()

```c
HAL_Delay(1000);  // Delay 1000ms (1 second)
```

**Characteristics:**
- Uses SysTick timer
- Blocking (CPU waits)
- 1ms resolution
- Simple to use

**Limitations:**
- Blocks execution
- Not suitable for precise timing
- Doesn't work in interrupts

### Custom Delay with Timers

For non-blocking delays, use timer flags:

```c
uint32_t timestamp = HAL_GetTick();

while (1) {
    if (HAL_GetTick() - timestamp >= 1000) {
        timestamp = HAL_GetTick();
        // Do something every 1 second
    }

    // Other code can run here
}
```

## Next Steps

Proceed to [Chapter 5: Communication Interfaces](05-Communication-Interfaces.md) to learn about UART, SPI, and I2C.
