# Interrupts

## Overview

The 8051 has 5 interrupt sources:
1. **INT0** - External Interrupt 0 (P3.2)
2. **INT1** - External Interrupt 1 (P3.3)
3. **TF0** - Timer 0 Overflow
4. **TF1** - Timer 1 Overflow
5. **RI/TI** - Serial Port Receive/Transmit

## Interrupt Priority

Two priority levels: High and Low
- High priority interrupts can interrupt low priority ISRs
- Set using IP (Interrupt Priority) register

## Control Registers

### IE (Interrupt Enable Register)
```
EA | - | - | ES | ET1 | EX1 | ET0 | EX0
```
- EA: Enable All (master enable)
- ES: Enable Serial
- ET1: Enable Timer 1
- EX1: Enable External 1
- ET0: Enable Timer 0
- EX0: Enable External 0

### IP (Interrupt Priority Register)
```
- | - | - | PS | PT1 | PX1 | PT0 | PX0
```

## Example: External Interrupt

```c
#include <reg51.h>

void ext0_isr() interrupt 0 {
    // ISR for External Interrupt 0
    P1 = ~P1;  // Toggle P1
}

void main() {
    EA = 1;   // Enable global interrupts
    EX0 = 1;  // Enable External Interrupt 0
    IT0 = 1;  // Edge triggered (falling edge)

    while(1) {
        // Main loop
    }
}
```

## Interrupt Vector Table

| Interrupt | Vector Address | Interrupt Number |
|-----------|---------------|------------------|
| Reset     | 0000H         | -                |
| INT0      | 0003H         | 0                |
| TF0       | 000BH         | 1                |
| INT1      | 0013H         | 2                |
| TF1       | 001BH         | 3                |
| RI/TI     | 0023H         | 4                |
