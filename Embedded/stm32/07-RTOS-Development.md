# Chapter 7: RTOS Development

## 7.1 Introduction to RTOS

### What is an RTOS?

A Real-Time Operating System (RTOS) is a specialized operating system designed for embedded systems that require deterministic timing and task management.

**Key Features:**
- Task scheduling and management
- Inter-task communication
- Synchronization mechanisms
- Memory management
- Deterministic behavior

**When to Use RTOS:**
- Multiple concurrent tasks
- Complex timing requirements
- Need for task prioritization
- Modular code organization
- Scalable applications

### FreeRTOS Overview

FreeRTOS is the most popular RTOS for embedded systems.

**Advantages:**
- Free and open source
- Small footprint
- Well documented
- Large community
- Officially supported by ST

## 7.2 FreeRTOS Setup

### Adding FreeRTOS to Project

**Using STM32CubeMX:**
1. Open your project in CubeMX
2. Middleware → FREERTOS → Enable
3. Configure RTOS settings
4. Generate code

**Manual Configuration:**
- Download FreeRTOS from freertos.org
- Add source files to project
- Configure FreeRTOSConfig.h

### FreeRTOSConfig.h Key Settings

```c
#define configUSE_PREEMPTION              1
#define configUSE_IDLE_HOOK               0
#define configUSE_TICK_HOOK               0
#define configCPU_CLOCK_HZ                72000000
#define configTICK_RATE_HZ                1000
#define configMAX_PRIORITIES              5
#define configMINIMAL_STACK_SIZE          128
#define configTOTAL_HEAP_SIZE             15360
#define configMAX_TASK_NAME_LEN           16
```

## 7.3 Tasks

### Creating Tasks

```c
#include "FreeRTOS.h"
#include "task.h"

// Task function
void vTaskLED(void *pvParameters)
{
    while (1)
    {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        vTaskDelay(pdMS_TO_TICKS(500));  // Delay 500ms
    }
}

// Create task in main()
int main(void)
{
    HAL_Init();
    SystemClock_Config();
    LED_Init();

    // Create task
    xTaskCreate(vTaskLED,           // Task function
                "LED Task",         // Task name
                128,                // Stack size (words)
                NULL,               // Parameters
                1,                  // Priority
                NULL);              // Task handle

    // Start scheduler
    vTaskStartScheduler();

    // Should never reach here
    while (1);
}
```

### Task Priorities

- Higher number = higher priority
- Priority 0 = lowest (idle task)
- Same priority tasks use round-robin scheduling

```c
xTaskCreate(vTaskHigh, "High", 128, NULL, 3, NULL);    // Highest
xTaskCreate(vTaskMed, "Medium", 128, NULL, 2, NULL);   // Medium
xTaskCreate(vTaskLow, "Low", 128, NULL, 1, NULL);      // Lowest
```

### Task States

```
┌─────────┐  Create   ┌─────────┐
│ Dormant │─────────→│ Ready   │
└─────────┘           └────┬────┘
                           │ ↑
                  Schedule │ │ Preempt/Yield
                           ↓ │
                      ┌────┴────┐
                      │ Running │
                      └────┬────┘
                           │
                    Block  │
                           ↓
                      ┌─────────┐
                      │ Blocked │
                      └─────────┘
```

### Task Management Functions

```c
// Delay task
vTaskDelay(pdMS_TO_TICKS(1000));  // Delay 1 second

// Delay until specific time
TickType_t xLastWakeTime = xTaskGetTickCount();
vTaskDelayUntil(&xLastWakeTime, pdMS_TO_TICKS(100));  // Periodic 100ms

// Suspend task
vTaskSuspend(xTaskHandle);

// Resume task
vTaskResume(xTaskHandle);

// Delete task
vTaskDelete(xTaskHandle);
```

## 7.4 Queues

Queues provide inter-task communication.

### Creating and Using Queues

```c
#include "queue.h"

QueueHandle_t xQueue;

// Create queue
xQueue = xQueueCreate(10, sizeof(uint32_t));  // 10 items, 4 bytes each

// Send to queue (from task)
uint32_t data = 123;
xQueueSend(xQueue, &data, portMAX_DELAY);

// Receive from queue
uint32_t received;
if (xQueueReceive(xQueue, &received, pdMS_TO_TICKS(1000)) == pdPASS) {
    // Data received
    printf("Received: %lu\r\n", received);
}
```

### Queue Example: Producer-Consumer

```c
QueueHandle_t xDataQueue;

void vProducerTask(void *pvParameters)
{
    uint32_t counter = 0;

    while (1)
    {
        counter++;
        xQueueSend(xDataQueue, &counter, portMAX_DELAY);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void vConsumerTask(void *pvParameters)
{
    uint32_t data;

    while (1)
    {
        if (xQueueReceive(xDataQueue, &data, portMAX_DELAY) == pdPASS) {
            printf("Consumed: %lu\r\n", data);
        }
    }
}

int main(void)
{
    // Initialize hardware
    HAL_Init();
    SystemClock_Config();

    // Create queue
    xDataQueue = xQueueCreate(5, sizeof(uint32_t));

    // Create tasks
    xTaskCreate(vProducerTask, "Producer", 128, NULL, 2, NULL);
    xTaskCreate(vConsumerTask, "Consumer", 128, NULL, 1, NULL);

    // Start scheduler
    vTaskStartScheduler();

    while (1);
}
```

## 7.5 Semaphores

### Binary Semaphore

Used for synchronization.

```c
#include "semphr.h"

SemaphoreHandle_t xBinarySemaphore;

// Create semaphore
xBinarySemaphore = xSemaphoreCreateBinary();

// Give semaphore (from ISR)
void EXTI0_IRQHandler(void)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;

    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_0);

    xSemaphoreGiveFromISR(xBinarySemaphore, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

// Take semaphore (in task)
void vTaskButton(void *pvParameters)
{
    while (1)
    {
        if (xSemaphoreTake(xBinarySemaphore, portMAX_DELAY) == pdPASS) {
            // Button pressed
            printf("Button event\r\n");
        }
    }
}
```

### Counting Semaphore

Used for resource management.

```c
SemaphoreHandle_t xCountingSemaphore;

// Create counting semaphore (max 5, initial 5)
xCountingSemaphore = xSemaphoreCreateCounting(5, 5);

// Take resource
xSemaphoreTake(xCountingSemaphore, portMAX_DELAY);
// Use resource
xSemaphoreGive(xCountingSemaphore);
```

### Mutex

Used for mutual exclusion (protecting shared resources).

```c
SemaphoreHandle_t xMutex;

// Create mutex
xMutex = xSemaphoreCreateMutex();

// Protect critical section
void vTaskShared(void *pvParameters)
{
    while (1)
    {
        if (xSemaphoreTake(xMutex, portMAX_DELAY) == pdPASS) {
            // Critical section - access shared resource
            shared_variable++;

            xSemaphoreGive(xMutex);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
```

## 7.6 Software Timers

Software timers execute callback functions after a specified time.

```c
#include "timers.h"

TimerHandle_t xTimer;

// Timer callback
void vTimerCallback(TimerHandle_t xTimer)
{
    // Called when timer expires
    HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
}

// Create timer
xTimer = xTimerCreate("Timer",                  // Name
                      pdMS_TO_TICKS(1000),      // Period (1 second)
                      pdTRUE,                   // Auto-reload
                      0,                        // Timer ID
                      vTimerCallback);          // Callback

// Start timer
xTimerStart(xTimer, 0);

// Stop timer
xTimerStop(xTimer, 0);

// Reset timer
xTimerReset(xTimer, 0);
```

## 7.7 Event Groups

Event groups allow tasks to wait for multiple events.

```c
#include "event_groups.h"

EventGroupHandle_t xEventGroup;

#define BIT_0 (1 << 0)
#define BIT_1 (1 << 1)
#define BIT_2 (1 << 2)

// Create event group
xEventGroup = xEventGroupCreate();

// Set bits (from task or ISR)
xEventGroupSetBits(xEventGroup, BIT_0 | BIT_1);

// Wait for bits
EventBits_t uxBits = xEventGroupWaitBits(
    xEventGroup,
    BIT_0 | BIT_1,      // Bits to wait for
    pdTRUE,             // Clear on exit
    pdTRUE,             // Wait for all bits
    portMAX_DELAY);     // Timeout
```

## 7.8 Memory Management

### Dynamic Allocation

```c
// Allocate memory
void *ptr = pvPortMalloc(100);

// Free memory
vPortFree(ptr);
```

### Heap Schemes

FreeRTOS provides 5 heap implementations:
- **heap_1**: Simple, no free
- **heap_2**: Best fit, allows free
- **heap_3**: Wraps malloc/free
- **heap_4**: First fit, coalescence
- **heap_5**: Multiple memory regions

## 7.9 Complete RTOS Example

```c
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"

QueueHandle_t xADCQueue;
SemaphoreHandle_t xUARTMutex;

// ADC Task - reads sensor every 100ms
void vTaskADC(void *pvParameters)
{
    uint16_t adc_value;

    while (1)
    {
        adc_value = ADC_Read();
        xQueueSend(xADCQueue, &adc_value, 0);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// Processing Task - processes ADC data
void vTaskProcess(void *pvParameters)
{
    uint16_t adc_value;
    float voltage;

    while (1)
    {
        if (xQueueReceive(xADCQueue, &adc_value, portMAX_DELAY) == pdPASS) {
            voltage = ADC_ToVoltage(adc_value);

            // Print with mutex protection
            if (xSemaphoreTake(xUARTMutex, portMAX_DELAY) == pdPASS) {
                printf("Voltage: %.2f V\r\n", voltage);
                xSemaphoreGive(xUARTMutex);
            }
        }
    }
}

// LED Task - blinks LED
void vTaskLED(void *pvParameters)
{
    while (1)
    {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_13);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

int main(void)
{
    // Initialize hardware
    HAL_Init();
    SystemClock_Config();
    LED_Init();
    ADC1_Init();
    UART1_Init();

    // Create queue and mutex
    xADCQueue = xQueueCreate(10, sizeof(uint16_t));
    xUARTMutex = xSemaphoreCreateMutex();

    // Create tasks
    xTaskCreate(vTaskADC, "ADC", 128, NULL, 2, NULL);
    xTaskCreate(vTaskProcess, "Process", 256, NULL, 2, NULL);
    xTaskCreate(vTaskLED, "LED", 128, NULL, 1, NULL);

    // Start scheduler
    vTaskStartScheduler();

    while (1);
}
```

## 7.10 Best Practices

1. **Task Design**
   - Keep tasks focused on single responsibility
   - Use appropriate stack sizes
   - Set correct priorities

2. **Resource Sharing**
   - Always use mutexes for shared resources
   - Keep critical sections short
   - Avoid priority inversion

3. **Timing**
   - Use vTaskDelayUntil() for periodic tasks
   - Don't use HAL_Delay() in RTOS tasks
   - Consider task execution time

4. **Debugging**
   - Enable stack overflow checking
   - Monitor heap usage
   - Use task statistics

5. **Interrupts**
   - Keep ISRs short
   - Use FromISR() functions
   - Check return values

## Next Steps

Proceed to [Chapter 8: Project Examples](08-Project-Examples.md) for complete project implementations.
