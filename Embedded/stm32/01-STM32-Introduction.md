# Chapter 1: STM32 Introduction

## 1.1 What is STM32?

STM32 is a family of 32-bit microcontroller integrated circuits by STMicroelectronics. Based on ARM Cortex-M processor cores, STM32 microcontrollers are widely used in embedded systems for industrial, consumer, and IoT applications.

### Key Features

- **ARM Cortex-M Core**: Various cores from M0 to M7, offering different performance levels
- **Rich Peripherals**: GPIO, timers, ADC, DAC, communication interfaces (UART, SPI, I2C, CAN, USB)
- **Low Power Consumption**: Multiple power-saving modes
- **Scalability**: Wide range of memory sizes and package options
- **Ecosystem**: Comprehensive development tools and libraries

## 1.2 STM32 Family Overview

### STM32F Series (Mainstream)

**STM32F0** - Entry-level (Cortex-M0)
- Clock: up to 48 MHz
- Flash: 16-256 KB
- Use cases: Simple control applications, cost-sensitive projects

**STM32F1** - Basic performance (Cortex-M3)
- Clock: up to 72 MHz
- Flash: 16-1024 KB
- Popular: STM32F103C8T6 ("Blue Pill")
- Use cases: General-purpose applications, motor control

**STM32F3** - Mixed-signal (Cortex-M4)
- Clock: up to 72 MHz
- Features: Enhanced analog peripherals, DSP instructions
- Use cases: Motor control, digital power conversion

**STM32F4** - High performance (Cortex-M4F)
- Clock: up to 180 MHz
- Features: FPU, DSP, advanced connectivity
- Popular: STM32F407VGT6
- Use cases: Audio processing, graphics, complex algorithms

**STM32F7** - Very high performance (Cortex-M7)
- Clock: up to 216 MHz
- Features: Double-precision FPU, L1 cache
- Use cases: High-end embedded applications, GUI

### STM32L Series (Ultra-low-power)

- Optimized for battery-powered applications
- Multiple low-power modes
- Use cases: Wearables, sensors, IoT devices

### STM32H Series (High performance)

- Cortex-M7 core up to 480 MHz
- Advanced features for demanding applications
- Use cases: Industrial automation, medical devices

### STM32G Series (Mainstream)

- Balanced performance and efficiency
- Modern peripherals
- Use cases: General-purpose applications

## 1.3 STM32 Architecture

### Core Components

```
┌─────────────────────────────────────┐
│         ARM Cortex-M Core           │
│  (CPU, NVIC, SysTick, Debug)        │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │   Bus Matrix │
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼───┐  ┌──▼────┐
│ Flash │  │ SRAM │  │ Periph│
│Memory │  │      │  │erals  │
└───────┘  └──────┘  └───────┘
```

### Memory Organization

**Flash Memory**
- Program storage (non-volatile)
- Typical address: 0x0800 0000
- Can be read/written in pages/sectors

**SRAM**
- Data storage (volatile)
- Typical address: 0x2000 0000
- Fast access for variables and stack

**Peripheral Registers**
- Memory-mapped I/O
- Typical address: 0x4000 0000
- Control hardware through register access

## 1.4 Selection Guide

### Choosing the Right STM32

Consider these factors:

1. **Performance Requirements**
   - Clock speed needed
   - Computational complexity
   - Real-time constraints

2. **Memory Requirements**
   - Program size (Flash)
   - Data size (SRAM)
   - Future expansion

3. **Peripherals Needed**
   - Communication interfaces
   - Analog features (ADC/DAC)
   - Timers and PWM channels

4. **Power Consumption**
   - Battery-powered vs. mains-powered
   - Sleep mode requirements

5. **Package and Pin Count**
   - PCB space constraints
   - Number of I/O pins needed

6. **Cost**
   - Budget constraints
   - Volume pricing

### Popular Choices for Learning

**STM32F103C8T6 (Blue Pill)**
- Pros: Inexpensive, widely available, good community support
- Specs: 72 MHz, 64 KB Flash, 20 KB SRAM
- Best for: Beginners, hobby projects

**STM32F407VGT6 (Discovery Board)**
- Pros: High performance, rich peripherals, official dev board
- Specs: 168 MHz, 1 MB Flash, 192 KB SRAM
- Best for: Advanced projects, learning complex features

**STM32L476RG (Nucleo Board)**
- Pros: Low power, modern architecture, official support
- Specs: 80 MHz, 1 MB Flash, 128 KB SRAM
- Best for: IoT, battery-powered applications

## 1.5 Development Tools

### Hardware Tools

**ST-Link Debugger**
- Official programmer/debugger
- Supports SWD interface
- Real-time debugging capabilities

**Development Boards**
- Discovery boards: Feature-rich, affordable
- Nucleo boards: Arduino-compatible, various MCU options
- Custom boards: For production designs

### Software Tools

**STM32CubeMX**
- Graphical configuration tool
- Generates initialization code
- Manages dependencies and conflicts

**IDEs**
- Keil MDK-ARM: Industry standard, excellent debugging
- STM32CubeIDE: Free, Eclipse-based, official
- IAR Embedded Workbench: Professional, optimized compiler

**Libraries**
- HAL (Hardware Abstraction Layer): High-level, portable
- LL (Low-Layer): Low-level, efficient
- Standard Peripheral Library: Legacy, still used for F1 series

## 1.6 Getting Started Checklist

Before diving into development:

- [ ] Choose your STM32 microcontroller
- [ ] Obtain development board or design minimal system
- [ ] Get ST-Link debugger
- [ ] Install STM32CubeMX
- [ ] Install your preferred IDE
- [ ] Download datasheets and reference manuals
- [ ] Set up serial terminal software
- [ ] Prepare basic electronic components

## Next Steps

Proceed to [Chapter 2: Hardware Basics](02-Hardware-Basics.md) to learn about circuit design and minimal system requirements.
