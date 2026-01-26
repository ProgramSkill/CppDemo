# Getting Started with 8051 Microcontroller

## Introduction

The 8051 microcontroller is an 8-bit microcontroller that has been the foundation of embedded systems education and industrial applications for over four decades.

## What You Need

### Hardware
- **8051 Development Board** (AT89C51, AT89S52, or compatible)
- **USB Programmer** (USB-to-Serial or ISP programmer)
- **Power Supply** (5V DC)
- **LEDs and Resistors** (for basic testing)
- **Breadboard and Jumper Wires**
- **Crystal Oscillator** (11.0592 MHz or 12 MHz)

### Software
- **Compiler**: Keil µVision, SDCC (Small Device C Compiler)
- **Programmer Software**: Flash Magic, Progisp
- **Simulator**: Proteus, Keil Simulator
- **Text Editor**: VS Code, Notepad++

## Basic Concepts

### Pin Configuration
- **Port 0 (P0.0-P0.7)**: 8-bit bidirectional I/O port
- **Port 1 (P1.0-P1.7)**: 8-bit bidirectional I/O port
- **Port 2 (P2.0-P2.7)**: 8-bit bidirectional I/O port
- **Port 3 (P3.0-P3.7)**: 8-bit bidirectional I/O port with alternate functions

### Power Pins
- **VCC**: +5V power supply
- **GND**: Ground
- **RST**: Reset pin (active high)

### Clock
- **XTAL1, XTAL2**: Crystal oscillator connections

## Your First Program

A simple LED blink program demonstrates the basics:

```c
#include <reg51.h>

void delay(unsigned int ms);

void main() {
    while(1) {
        P1 = 0x00;  // Turn ON LED (assuming active low)
        delay(500);
        P1 = 0xFF;  // Turn OFF LED
        delay(500);
    }
}

void delay(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}
```

## Next Steps

1. Study the hardware architecture
2. Learn programming in C or Assembly
3. Understand peripherals (timers, interrupts, serial)
4. Practice with examples
5. Build complete projects
