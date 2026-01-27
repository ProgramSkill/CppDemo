# Timer Examples

## Overview

This section demonstrates the use of 8051 timers for precise timing operations. Timers are essential for generating accurate delays, measuring time intervals, and creating periodic events.

## Examples

### Timer_Delay.c
**Difficulty:** ⭐⭐ Intermediate

**Description:**
Accurate delay generation using Timer 0 in Mode 1 (16-bit timer). Demonstrates:
- Timer initialization and configuration
- Timer value calculation
- Interrupt-driven timing
- Precise delay control

**Hardware Requirements:**
- 8051 microcontroller @ 11.0592MHz or 12MHz
- LED connected to P1.0 (optional, for visualization)

**Key Concepts:**
- Timer Mode 1 (16-bit)
- Timer overflow interrupts
- Reload value calculation
- Machine cycle timing

**Learning Outcomes:**
- Understand timer modes and their differences
- Calculate timer reload values for specific delays
- Implement interrupt-driven timing
- Compare polling vs interrupt approaches

**Timer Calculation:**
```
For 50ms delay @ 12MHz:
Machine Cycle = 12 / 12MHz = 1µs
Timer Value = 65536 - (50000µs / 1µs) = 15536 = 3CB0H
```

**Code Size:** ~80 lines
**Accuracy:** ±1 machine cycle

---

## Upcoming Examples

### PWM_Generation.c
PWM signal generation using Timer 2

### Frequency_Counter.c
Frequency measurement using Timer 0/1

### Stopwatch.c
Precise stopwatch with display

### Event_Counter.c
External event counting

---

## Timer Modes Quick Reference

| Mode | Description | Max Count | Use Case |
|------|-------------|-----------|----------|
| 0 | 13-bit timer | 8192 | Legacy compatibility |
| 1 | 16-bit timer | 65536 | ⭐ General purpose delays |
| 2 | 8-bit auto-reload | 256 | ⭐ Baud rate generation |
| 3 | Split timer | 2× 8-bit | Special cases |

---

## Timer Calculations

### Machine Cycle
```
Machine Cycle = 12 / Oscillator Frequency

@ 12MHz: 1 machine cycle = 1µs
@ 11.0592MHz: 1 machine cycle = 1.085µs
@ 24MHz: 1 machine cycle = 0.5µs
```

### Reload Value (Mode 1)
```
THx:TLx = 65536 - (Desired Delay / Machine Cycle)
```

### Reload Value (Mode 2)
```
THx = 256 - (Desired Delay / Machine Cycle)
```

---

## Common Issues

### Inaccurate Delays
- Verify oscillator frequency
- Check timer mode selection
- Ensure proper reload values

### Timer Not Overflowing
- Confirm timer is started (TRx = 1)
- Check interrupt enable bits
- Verify ISR vector address

### Mode 1 Not Reloading
- Mode 1 requires manual reload in ISR
- Consider Mode 2 for auto-reload

---

## Prerequisites

- Basic I/O operations
- Understanding of interrupts
- Number system conversions

**Recommended Reading:**
- [Timer/Counter Section](../02_Hardware_Architecture/README.md#timerscounters)
- [Interrupt System](../03_Interrupts/)

---

## Comparison: Polling vs Interrupt

### Polling Approach
```c
while(!TF0);   // Wait for overflow
TF0 = 0;       // Clear flag
```
- ✅ Simple to implement
- ❌ CPU blocked during wait
- ❌ Not precise for long delays

### Interrupt Approach
```c
// Timer ISR handles overflow
void timer0_isr() interrupt 1 {
    TH0 = reload_value;
    TF0 = 0;
}
```
- ✅ CPU free during counting
- ✅ More precise
- ✅ Can handle multiple events
- ❌ Slightly more complex

---

## Next Steps

After mastering timers, explore:
- [Interrupt Examples](../03_Interrupts/)
- [PWM and advanced timing](../05_Advanced/)
