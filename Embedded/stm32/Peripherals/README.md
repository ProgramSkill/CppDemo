# STM32 Peripherals Documentation
# STM32 外设详细文档

This directory contains comprehensive documentation for STM32 microcontroller peripherals, including theory, configuration, and practical examples.

本目录包含STM32微控制器外设的详细文档，包括理论知识、配置方法和实践示例。

---

## Directory Structure / 目录结构

### 01_GPIO - General Purpose Input/Output
**通用输入输出**
- GPIO configuration and modes
- Input/output operations
- Interrupt handling
- Practical examples (LED, button, etc.)

### 02_UART_USART - Universal Asynchronous/Synchronous Receiver-Transmitter
**通用异步/同步收发器**
- UART/USART basics
- Configuration and baud rate calculation
- DMA transfer
- Printf redirection
- Communication protocols

### 03_SPI - Serial Peripheral Interface
**串行外设接口**
- SPI protocol fundamentals
- Master/Slave configuration
- DMA transfer
- Common SPI devices (Flash, SD card, sensors)

### 04_I2C - Inter-Integrated Circuit
**I2C总线**
- I2C protocol and timing
- Master/Slave modes
- Multi-device communication
- Common I2C devices (EEPROM, sensors)

### 05_Timer - Timers and PWM
**定时器与PWM**
- Basic timer, general-purpose timer, advanced timer
- Time base configuration
- PWM generation
- Input capture and output compare
- Encoder interface

### 06_ADC_DAC - Analog to Digital / Digital to Analog Converter
**模数/数模转换器**
- ADC modes and configuration
- Multi-channel sampling
- DMA transfer
- DAC output
- Practical applications

### 07_DMA - Direct Memory Access
**直接内存访问**
- DMA fundamentals
- Channel configuration
- Memory-to-memory, peripheral-to-memory transfer
- DMA with UART, SPI, ADC

### 08_RTC - Real-Time Clock
**实时时钟**
- RTC configuration
- Calendar and alarm
- Backup domain
- Low-power timekeeping

### 09_Watchdog - Independent and Window Watchdog
**独立看门狗与窗口看门狗**
- IWDG (Independent Watchdog)
- WWDG (Window Watchdog)
- Configuration and usage
- System reliability

### 10_CAN - Controller Area Network
**CAN总线**
- CAN protocol basics
- Filter configuration
- Message transmission and reception
- Automotive applications

### 11_USB - Universal Serial Bus
**通用串行总线**
- USB Device/Host/OTG
- USB CDC (Virtual COM Port)
- USB HID (Human Interface Device)
- USB MSC (Mass Storage Class)

### 12_EXTI - External Interrupt
**外部中断**
- EXTI configuration
- GPIO interrupt
- Event mode
- Interrupt priority

### 13_Power_Management - Power and Low-Power Modes
**电源管理与低功耗模式**
- Power supply architecture
- Sleep, Stop, Standby modes
- Wake-up sources
- Low-power optimization

### 14_Flash_EEPROM - Internal Flash and EEPROM Emulation
**内部Flash与EEPROM模拟**
- Flash memory structure
- Flash programming and erasing
- EEPROM emulation
- Bootloader basics

---

## Documentation Format / 文档格式

Each peripheral documentation follows this structure:

每个外设文档遵循以下结构：

### 1. Overview / 概述
- Peripheral introduction
- Key features
- Application scenarios

### 2. Hardware Architecture / 硬件架构
- Block diagram
- Register description
- Pin configuration

### 3. Configuration / 配置方法
- HAL library configuration
- Register-level configuration
- Clock configuration

### 4. Programming Examples / 编程示例
- Basic examples
- Advanced applications
- Common pitfalls and solutions

### 5. Practical Projects / 实践项目
- Real-world applications
- Complete project code
- Debugging tips

---

## Learning Path / 学习路径

### Beginner Level / 初级
1. **GPIO** - Digital I/O basics
2. **UART** - Serial communication
3. **Timer** - Basic timing and PWM
4. **EXTI** - External interrupts

### Intermediate Level / 中级
5. **SPI** - High-speed serial communication
6. **I2C** - Multi-device bus communication
7. **ADC** - Analog signal acquisition
8. **DMA** - Efficient data transfer
9. **RTC** - Real-time clock

### Advanced Level / 高级
10. **CAN** - Automotive/industrial bus
11. **USB** - USB device development
12. **Power Management** - Low-power design
13. **Flash Programming** - Bootloader and firmware update
14. **Watchdog** - System reliability

---

## Prerequisites / 前置知识

- C language programming
- Basic digital circuit knowledge
- Understanding of STM32 architecture
- Familiarity with development tools (STM32CubeIDE, Keil, etc.)

- C语言编程
- 基本数字电路知识
- 了解STM32架构
- 熟悉开发工具（STM32CubeIDE、Keil等）

---

## Reference Resources / 参考资源

### Official Documentation / 官方文档
- STM32 Reference Manual (RM)
- STM32 Datasheet
- STM32 HAL Library Documentation
- STM32CubeMX User Manual

### Development Tools / 开发工具
- STM32CubeIDE
- STM32CubeMX
- Keil MDK
- IAR Embedded Workbench

### Hardware / 硬件
- STM32 Development Board (Nucleo, Discovery, etc.)
- ST-Link Debugger
- Logic Analyzer
- Oscilloscope

---

## Contributing / 贡献指南

This documentation is continuously updated. Contributions are welcome:
- Report errors or unclear content
- Suggest improvements
- Add practical examples
- Share project experiences

本文档持续更新中，欢迎贡献：
- 报告错误或不清晰的内容
- 提出改进建议
- 添加实践示例
- 分享项目经验

---

**Document Version**: 1.0
**Last Updated**: 2026-01-29
**Maintainer**: Embedded Development Team
