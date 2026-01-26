# Chapter 8: Project Examples

## 8.1 Project 1: Temperature Monitoring System

### Overview

A complete temperature monitoring system using:
- DS18B20 temperature sensor (1-Wire protocol)
- OLED display (I2C)
- UART logging
- LED indicators

### Hardware Connections

```
STM32F103C8T6:
- PA9: UART TX (logging)
- PA10: UART RX
- PB6: I2C1 SCL (OLED)
- PB7: I2C1 SDA (OLED)
- PA0: DS18B20 Data (1-Wire)
- PC13: LED (status indicator)
```

### Main Code Structure

```c
#include "main.h"
#include "ds18b20.h"
#include "ssd1306.h"

UART_HandleTypeDef huart1;
I2C_HandleTypeDef hi2c1;

void SystemClock_Config(void);
void Peripherals_Init(void);

int main(void)
{
    float temperature;
    char buffer[32];

    // Initialize
    HAL_Init();
    SystemClock_Config();
    Peripherals_Init();

    // Initialize OLED
    SSD1306_Init();
    SSD1306_Clear();
    SSD1306_GotoXY(0, 0);
    SSD1306_Puts("Temp Monitor", &Font_11x18, 1);
    SSD1306_UpdateScreen();

    while (1)
    {
        // Read temperature
        if (DS18B20_ReadTemperature(&temperature)) {
            // Display on OLED
            SSD1306_GotoXY(0, 20);
            sprintf(buffer, "%.2f C", temperature);
            SSD1306_Puts(buffer, &Font_16x26, 1);
            SSD1306_UpdateScreen();

            // Log via UART
            sprintf(buffer, "Temperature: %.2f C\r\n", temperature);
            HAL_UART_Transmit(&huart1, (uint8_t*)buffer, strlen(buffer), HAL_MAX_DELAY);

            // LED indicator
            if (temperature > 30.0f) {
                HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET);  // LED on
            } else {
                HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET);    // LED off
            }
        }

        HAL_Delay(1000);  // Update every second
    }
}
```

### DS18B20 Driver (Simplified)

```c
// ds18b20.h
#ifndef DS18B20_H
#define DS18B20_H

#include "stm32f1xx_hal.h"
#include <stdbool.h>

bool DS18B20_Init(void);
bool DS18B20_ReadTemperature(float *temperature);

#endif

// ds18b20.c
#include "ds18b20.h"
#include "onewire.h"

#define DS18B20_CMD_CONVERT     0x44
#define DS18B20_CMD_READ_SCRATCHPAD 0xBE

bool DS18B20_ReadTemperature(float *temperature)
{
    uint8_t data[9];

    // Start conversion
    OneWire_Reset();
    OneWire_WriteByte(0xCC);  // Skip ROM
    OneWire_WriteByte(DS18B20_CMD_CONVERT);
    HAL_Delay(750);  // Wait for conversion

    // Read scratchpad
    OneWire_Reset();
    OneWire_WriteByte(0xCC);
    OneWire_WriteByte(DS18B20_CMD_READ_SCRATCHPAD);

    for (int i = 0; i < 9; i++) {
        data[i] = OneWire_ReadByte();
    }

    // Calculate temperature
    int16_t raw = (data[1] << 8) | data[0];
    *temperature = (float)raw / 16.0f;

    return true;
}
```

## 8.2 Project 2: Motor Control with PWM

### Overview

DC motor speed control using:
- PWM for speed control
- Encoder for feedback
- PID controller
- UART for commands

### Hardware Setup

```
STM32F103C8T6:
- PA6: TIM3_CH1 (PWM output to motor driver)
- PA7: Motor direction control
- PB6: TIM4_CH1 (Encoder A)
- PB7: TIM4_CH2 (Encoder B)
- PA9/PA10: UART (commands)
```

### PWM Motor Control

```c
#include "main.h"

TIM_HandleTypeDef htim3;  // PWM
TIM_HandleTypeDef htim4;  // Encoder

typedef struct {
    float Kp;
    float Ki;
    float Kd;
    float setpoint;
    float integral;
    float prev_error;
} PID_Controller;

PID_Controller motor_pid = {
    .Kp = 2.0f,
    .Ki = 0.5f,
    .Kd = 0.1f,
    .setpoint = 0,
    .integral = 0,
    .prev_error = 0
};

void Motor_SetSpeed(int16_t speed)
{
    if (speed >= 0) {
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_7, GPIO_PIN_SET);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, speed);
    } else {
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_7, GPIO_PIN_RESET);
        __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, -speed);
    }
}

int16_t Encoder_GetCount(void)
{
    return (int16_t)__HAL_TIM_GET_COUNTER(&htim4);
}

void Encoder_Reset(void)
{
    __HAL_TIM_SET_COUNTER(&htim4, 0);
}

float PID_Calculate(PID_Controller *pid, float measured)
{
    float error = pid->setpoint - measured;
    pid->integral += error;
    float derivative = error - pid->prev_error;

    float output = (pid->Kp * error) + (pid->Ki * pid->integral) + (pid->Kd * derivative);

    pid->prev_error = error;

    // Limit output
    if (output > 1000) output = 1000;
    if (output < -1000) output = -1000;

    return output;
}

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    Motor_Init();
    Encoder_Init();
    UART_Init();

    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
    HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL);

    motor_pid.setpoint = 500;  // Target speed

    while (1)
    {
        // Read encoder
        int16_t encoder_count = Encoder_GetCount();
        Encoder_Reset();

        // Calculate PID
        float control = PID_Calculate(&motor_pid, encoder_count);

        // Set motor speed
        Motor_SetSpeed((int16_t)control);

        HAL_Delay(10);  // 100Hz control loop
    }
}
```

## 8.3 Project 3: Data Logger with SD Card

### Overview

Data logging system using:
- SD card (SPI interface)
- RTC for timestamps
- Multiple sensors (ADC)
- FAT filesystem

### Main Implementation

```c
#include "main.h"
#include "fatfs.h"

FATFS fs;
FIL file;
FRESULT fres;
RTC_HandleTypeDef hrtc;

void DataLogger_Init(void)
{
    // Mount SD card
    fres = f_mount(&fs, "", 1);
    if (fres != FR_OK) {
        printf("SD card mount failed\r\n");
        return;
    }

    // Create log file
    fres = f_open(&file, "datalog.csv", FA_WRITE | FA_OPEN_APPEND);
    if (fres == FR_OK) {
        // Write header if new file
        f_puts("Timestamp,Sensor1,Sensor2,Sensor3\n", &file);
        f_close(&file);
    }
}

void DataLogger_LogData(float sensor1, float sensor2, float sensor3)
{
    char buffer[128];
    RTC_TimeTypeDef sTime;
    RTC_DateTypeDef sDate;

    // Get timestamp
    HAL_RTC_GetTime(&hrtc, &sTime, RTC_FORMAT_BIN);
    HAL_RTC_GetDate(&hrtc, &sDate, RTC_FORMAT_BIN);

    // Format data
    sprintf(buffer, "%02d/%02d/%02d %02d:%02d:%02d,%.2f,%.2f,%.2f\n",
            sDate.Year, sDate.Month, sDate.Date,
            sTime.Hours, sTime.Minutes, sTime.Seconds,
            sensor1, sensor2, sensor3);

    // Write to file
    fres = f_open(&file, "datalog.csv", FA_WRITE | FA_OPEN_APPEND);
    if (fres == FR_OK) {
        f_puts(buffer, &file);
        f_close(&file);
    }
}

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    ADC_Init();
    RTC_Init();
    SPI_Init();
    DataLogger_Init();

    while (1)
    {
        // Read sensors
        float sensor1 = ADC_Read(ADC_CHANNEL_0);
        float sensor2 = ADC_Read(ADC_CHANNEL_1);
        float sensor3 = ADC_Read(ADC_CHANNEL_2);

        // Log data
        DataLogger_LogData(sensor1, sensor2, sensor3);

        // Wait 1 minute
        HAL_Delay(60000);
    }
}
```

## 8.4 Project 4: Wireless Sensor Node (ESP8266)

### Overview

IoT sensor node using:
- ESP8266 WiFi module (UART AT commands)
- DHT22 temperature/humidity sensor
- MQTT protocol
- Low power modes

### ESP8266 Communication

```c
#include "main.h"
#include "esp8266.h"

#define WIFI_SSID "YourSSID"
#define WIFI_PASS "YourPassword"
#define MQTT_BROKER "192.168.1.100"
#define MQTT_PORT 1883

bool ESP8266_SendCommand(char *cmd, char *expected_response, uint32_t timeout)
{
    HAL_UART_Transmit(&huart1, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);

    uint8_t rx_buffer[256];
    uint32_t start = HAL_GetTick();

    while (HAL_GetTick() - start < timeout) {
        if (HAL_UART_Receive(&huart1, rx_buffer, 1, 100) == HAL_OK) {
            // Check for expected response
            if (strstr((char*)rx_buffer, expected_response) != NULL) {
                return true;
            }
        }
    }

    return false;
}

bool ESP8266_ConnectWiFi(void)
{
    char cmd[128];

    // Set mode to station
    ESP8266_SendCommand("AT+CWMODE=1\r\n", "OK", 1000);

    // Connect to WiFi
    sprintf(cmd, "AT+CWJAP=\"%s\",\"%s\"\r\n", WIFI_SSID, WIFI_PASS);
    return ESP8266_SendCommand(cmd, "OK", 10000);
}

void ESP8266_PublishMQTT(char *topic, char *message)
{
    char cmd[256];

    // Connect to MQTT broker
    sprintf(cmd, "AT+CIPSTART=\"TCP\",\"%s\",%d\r\n", MQTT_BROKER, MQTT_PORT);
    ESP8266_SendCommand(cmd, "OK", 5000);

    // Publish message (simplified)
    sprintf(cmd, "AT+CIPSEND=%d\r\n", strlen(message));
    ESP8266_SendCommand(cmd, ">", 1000);
    HAL_UART_Transmit(&huart1, (uint8_t*)message, strlen(message), HAL_MAX_DELAY);
}

int main(void)
{
    float temperature, humidity;
    char payload[128];

    HAL_Init();
    SystemClock_Config();
    UART_Init();
    DHT22_Init();

    // Connect to WiFi
    if (ESP8266_ConnectWiFi()) {
        printf("WiFi connected\r\n");
    }

    while (1)
    {
        // Read sensor
        if (DHT22_Read(&temperature, &humidity)) {
            // Create JSON payload
            sprintf(payload, "{\"temp\":%.1f,\"hum\":%.1f}", temperature, humidity);

            // Publish to MQTT
            ESP8266_PublishMQTT("sensors/room1", payload);
        }

        // Sleep for 5 minutes
        HAL_Delay(300000);
    }
}
```

## 8.5 Project 5: CAN Bus Vehicle Dashboard

### Overview

Automotive dashboard using:
- CAN bus communication
- TFT display
- Multiple gauges (speed, RPM, fuel)
- Warning indicators

### CAN Message Handling

```c
#include "main.h"

CAN_HandleTypeDef hcan;

typedef struct {
    uint16_t speed;      // km/h
    uint16_t rpm;        // RPM
    uint8_t fuel;        // %
    uint8_t temperature; // °C
    uint8_t warnings;    // Bit flags
} VehicleData;

VehicleData vehicle = {0};

void CAN_FilterConfig(void)
{
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
}

void CAN_ProcessMessage(CAN_RxHeaderTypeDef *header, uint8_t *data)
{
    switch (header->StdId) {
        case 0x100:  // Speed message
            vehicle.speed = (data[0] << 8) | data[1];
            break;

        case 0x101:  // RPM message
            vehicle.rpm = (data[0] << 8) | data[1];
            break;

        case 0x102:  // Fuel level
            vehicle.fuel = data[0];
            break;

        case 0x103:  // Temperature
            vehicle.temperature = data[0];
            break;

        case 0x104:  // Warnings
            vehicle.warnings = data[0];
            break;
    }
}

void Dashboard_Update(void)
{
    char buffer[32];

    // Display speed
    sprintf(buffer, "Speed: %d km/h", vehicle.speed);
    TFT_DrawString(10, 10, buffer);

    // Display RPM
    sprintf(buffer, "RPM: %d", vehicle.rpm);
    TFT_DrawString(10, 40, buffer);

    // Display fuel
    TFT_DrawProgressBar(10, 70, 200, 20, vehicle.fuel);

    // Warning indicators
    if (vehicle.warnings & 0x01) {
        TFT_DrawIcon(220, 10, ICON_ENGINE_WARNING);
    }
    if (vehicle.warnings & 0x02) {
        TFT_DrawIcon(220, 40, ICON_OIL_WARNING);
    }
}

int main(void)
{
    CAN_RxHeaderTypeDef RxHeader;
    uint8_t RxData[8];

    HAL_Init();
    SystemClock_Config();
    CAN_Init();
    TFT_Init();

    CAN_FilterConfig();
    HAL_CAN_Start(&hcan);
    HAL_CAN_ActivateNotification(&hcan, CAN_IT_RX_FIFO0_MSG_PENDING);

    while (1)
    {
        // Check for CAN messages
        if (HAL_CAN_GetRxFifoFillLevel(&hcan, CAN_RX_FIFO0) > 0) {
            HAL_CAN_GetRxMessage(&hcan, CAN_RX_FIFO0, &RxHeader, RxData);
            CAN_ProcessMessage(&RxHeader, RxData);
        }

        // Update dashboard
        Dashboard_Update();

        HAL_Delay(50);  // 20Hz update rate
    }
}
```

## 8.6 Best Practices Summary

### Code Organization

1. **Modular Design**
   - Separate drivers from application code
   - Use header files for interfaces
   - Keep functions focused and small

2. **Error Handling**
   - Check return values
   - Implement timeout mechanisms
   - Provide fallback behavior

3. **Documentation**
   - Comment complex algorithms
   - Document pin assignments
   - Maintain changelog

### Testing and Debugging

1. **Incremental Development**
   - Test each peripheral separately
   - Verify hardware connections first
   - Use debug prints liberally

2. **Tools**
   - Logic analyzer for protocol debugging
   - Oscilloscope for signal verification
   - Serial monitor for logging

3. **Common Issues**
   - Clock configuration errors
   - Incorrect pin mapping
   - Missing peripheral clock enable
   - Buffer overflow
   - Stack overflow in RTOS

### Performance Optimization

1. **Use DMA** for data transfers
2. **Enable compiler optimization** (-O2 or -O3)
3. **Use appropriate data types** (uint8_t vs uint32_t)
4. **Minimize interrupt latency**
5. **Profile code** to find bottlenecks

## Conclusion

This tutorial series has covered the essential aspects of STM32 development, from basic concepts to advanced projects. Continue learning by:

- Building your own projects
- Reading ST application notes
- Joining STM32 communities
- Experimenting with different peripherals
- Contributing to open-source projects

Happy coding!
