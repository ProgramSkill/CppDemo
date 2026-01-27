# Interrupt Examples

## Overview

This section demonstrates interrupt handling in the 8051 microcontroller. Interrupts allow the CPU to respond to events asynchronously, without continuous polling.

## What are Interrupts?

Interrupts are signals that temporarily suspend the normal program execution to handle high-priority events. After handling the interrupt, the CPU returns to the original program.

**Key Benefits:**
- No CPU time wasted on polling
- Immediate response to events
- Ability to handle multiple events
- Precise timing control

## Interrupt Sources

| Source | Vector Address | Priority | Flag |
|--------|----------------|----------|------|
| INT0 (P3.2) | 0003H | Highest | IE0 |
| Timer 0 | 000BH | High | TF0 |
| INT1 (P3.3) | 0013H | Medium | IE1 |
| Timer 1 | 001BH | Low | TF1 |
| Serial | 0023H | Lowest | RI/TI |

## Examples

### Upcoming Examples

#### External_Interrupt.c
**Difficulty:** ⭐⭐ Intermediate

**Description:**
External interrupt handling using INT0 and INT1. Demonstrates:
- Edge-triggered interrupts
- Interrupt service routines (ISR)
- Button debouncing with interrupts

**Hardware:**
- Buttons on P3.2 (INT0) and P3.3 (INT1)
- LEDs on P1.0 and P1.1 for indication

**Key Concepts:**
- IT0/IT1 configuration (edge/level)
- Interrupt enable setup
- ISR structure and best practices

#### Timer_Interrupt.c
**Difficulty:** ⭐⭐ Intermediate

**Description:**
Periodic tasks using timer interrupts. Demonstrates:
- Timer overflow interrupts
- Multi-rate timing
- Real-time task scheduling

**Applications:**
- LED flashing without blocking
- Periodic sensor reading
- Real-time clock

#### Serial_Interrupt.c
**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Full-duplex serial communication using interrupts. Demonstrates:
- Transmit and receive interrupts
- Ring buffers
- Non-blocking I/O

#### Multi_Interrupt.c
**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Coordinating multiple interrupt sources. Demonstrates:
- Priority management
- Interrupt nesting
- Shared resource protection

---

## Interrupt Structure Template

```c
#include <reg51.h>

// 1. Global variables (use volatile for shared data)
volatile unsigned char flag = 0;

// 2. Interrupt Service Routine
void external0_isr(void) interrupt 0
{
    // Keep ISRs short!
    flag = 1;          // Set flag for main program
    // Perform time-critical actions only
}

// 3. Initialization
void init_interrupts(void)
{
    IT0 = 1;           // Edge-triggered
    EX0 = 1;           // Enable INT0
    EA = 1;            // Global interrupt enable
}

// 4. Main program
void main(void)
{
    init_interrupts();

    while(1)
    {
        if(flag)
        {
            flag = 0;  // Clear flag
            // Handle event
        }
        // Other tasks can run here
    }
}
```

---

## Best Practices

### ✅ DO's
1. **Keep ISRs Short** - Only set flags or do minimal processing
2. **Use volatile** - For variables shared with ISRs
3. **Save Context** - Push/pop used registers
4. **Clear Flags** - Clear interrupt flags appropriately
5. **Enable Globally Last** - Set EA bit last in initialization

### ❌ DON'Ts
1. **Long Processing in ISR** - Blocks other interrupts
2. **Heavy Calculations** - Move to main program
3. **Call Complex Functions** - Keep ISRs simple
4. **Forget to Clear Flags** - Causes repeated interrupts
5. **Enable All Interrupts** - Only enable what you need

---

## Interrupt Priority

The 8051 has two priority levels: **Low** (default) and **High**.

**Priority Rules:**
1. High-priority interrupts can interrupt low-priority ISRs
2. Same-priority interrupts cannot interrupt each other
3. Simultaneous interrupts serviced in fixed order (INT0 → T0 → INT1 → T1 → Serial)

**Setting Priority:**
```c
PT0 = 1;  // Timer 0 high priority
PX0 = 1;  // INT0 high priority
```

---

## Common Issues

### Interrupt Not Triggering
- Check individual interrupt enable (EX0, ET0, etc.)
- Verify global interrupt enable (EA = 1)
- Confirm interrupt flag is being set
- Check trigger type (edge vs level)

### System Crash/Hang
- ISR too complex or long
- Stack overflow (check SP initialization)
- Forgetting to clear interrupt flag
- Interrupting critical sections

### Unwanted Repeated Interrupts
- Level-triggered mode with low signal held
- Not clearing interrupt flag properly
- Hardware noise/glitches

### Interrupt Timing Issues
- High-priority interrupt blocking low-priority
- ISR taking too long
- Nested interrupts causing stack issues

---

## Interrupt vs Polling Comparison

| Aspect | Interrupt | Polling |
|--------|-----------|---------|
| CPU Usage | Efficient | Wasteful |
| Response Time | Immediate | Delayed |
| Implementation | Complex | Simple |
| Reliability | High | Low |
| Multi-tasking | Excellent | Poor |

---

## Prerequisites

- Basic I/O operations
- Understanding of stack and memory
- Timer basics

**Recommended Reading:**
- [Interrupt System](../02_Hardware_Architecture/README.md#interrupt-system)
- [Timer Examples](../02_Timers/)

---

## Debugging Tips

1. **LED Indicator** - Toggle LED in ISR to verify it's executing
2. **Counter** - Increment counter in ISR, check in main loop
3. **Scope** - Use oscilloscope to monitor interrupt pins
4. **Single Step** - Use debugger to step through ISR
5. **Simplify** - Start with one interrupt, add more gradually

---

## Real-World Applications

- **Embedded Systems**: Button presses, sensor triggers
- **Communications**: Serial data reception
- **Motor Control**: Encoder feedback, limit switches
- **Instrumentation**: Alarm conditions, threshold crossing
- **Automotive**: Crash sensors, timeout detection

---

## Next Steps

After mastering interrupts:
- [Serial Port Examples](../04_Serial_Port/)
- [Advanced Multi-Interrupt Systems](../05_Advanced/)
