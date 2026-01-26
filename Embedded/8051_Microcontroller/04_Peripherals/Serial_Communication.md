# Serial Communication (UART)

## Overview

The 8051 has a full-duplex UART for serial communication:
- **RXD (P3.0)**: Receive Data
- **TXD (P3.1)**: Transmit Data

## Serial Modes

### Mode 0: Shift Register
- 8-bit synchronous mode
- Baud rate = fosc/12

### Mode 1: 8-bit UART (Most Common)
- 10-bit frame: 1 start + 8 data + 1 stop
- Variable baud rate (Timer 1)

### Mode 2: 9-bit UART
- 11-bit frame: 1 start + 8 data + 1 programmable + 1 stop
- Baud rate = fosc/32 or fosc/64

### Mode 3: 9-bit UART
- Same as Mode 2 but variable baud rate

## Baud Rate Generation

Using Timer 1 in Mode 2 (8-bit auto-reload):

**Formula**: Baud Rate = (2^SMOD / 32) × (fosc / (12 × (256 - TH1)))

**Common Values (11.0592 MHz crystal)**:
- 9600 baud: TH1 = 0xFD, SMOD = 1
- 4800 baud: TH1 = 0xFA, SMOD = 1
- 2400 baud: TH1 = 0xF4, SMOD = 1

## Example: Serial Transmission

```c
#include <reg51.h>

void serial_init() {
    TMOD = 0x20;  // Timer 1, Mode 2
    TH1 = 0xFD;   // 9600 baud at 11.0592MHz
    SCON = 0x50;  // Mode 1, REN enabled
    TR1 = 1;      // Start Timer 1
}

void send_char(char c) {
    SBUF = c;     // Load data
    while(TI == 0);  // Wait for transmission
    TI = 0;       // Clear flag
}

char receive_char() {
    while(RI == 0);  // Wait for reception
    RI = 0;       // Clear flag
    return SBUF;  // Return data
}
```
