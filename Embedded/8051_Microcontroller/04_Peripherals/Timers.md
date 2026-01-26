# Timers/Counters

## Overview

The 8051 has two 16-bit timers/counters:
- **Timer 0** (TH0, TL0)
- **Timer 1** (TH1, TL1)

Each can operate as a timer or counter.

## Timer vs Counter Mode

- **Timer Mode**: Counts machine cycles (clock/12)
- **Counter Mode**: Counts external pulses on T0 (P3.4) or T1 (P3.5)

## Timer Modes

### Mode 0: 13-bit Timer/Counter
- 8-bit TH + 5-bit TL
- Maximum count: 8192 (2^13)

### Mode 1: 16-bit Timer/Counter (Most Common)
- Full 16-bit timer
- Maximum count: 65536 (2^16)

### Mode 2: 8-bit Auto-Reload
- TL is 8-bit counter
- TH holds reload value
- Auto-reloads when TL overflows

### Mode 3: Split Timer Mode
- Timer 0 splits into two 8-bit timers
- Timer 1 stops

## Control Registers

### TMOD (Timer Mode Register)
```
GATE | C/T | M1 | M0 | GATE | C/T | M1 | M0
Timer 1              | Timer 0
```

### TCON (Timer Control Register)
- TF1, TF0: Timer overflow flags
- TR1, TR0: Timer run control bits

## Example: Delay using Timer

```c
#include <reg51.h>

void timer_delay() {
    TMOD = 0x01;  // Timer 0, Mode 1
    TH0 = 0xFC;   // Load high byte
    TL0 = 0x66;   // Load low byte (1ms delay at 12MHz)
    TR0 = 1;      // Start timer
    while(TF0 == 0);  // Wait for overflow
    TR0 = 0;      // Stop timer
    TF0 = 0;      // Clear flag
}
```
