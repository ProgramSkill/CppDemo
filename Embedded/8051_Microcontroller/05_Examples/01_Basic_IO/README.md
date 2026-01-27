# Basic I/O Examples

## Overview

This section contains fundamental I/O operations examples for the 8051 microcontroller. These examples demonstrate how to interact with ports, control LEDs, read buttons, and perform basic input/output operations.

## Examples

### LED_Blink.c
**Difficulty:** ⭐ Beginner

**Description:**
Simple LED blinking program using timer delays. Demonstrates:
- Port configuration as output
- Basic delay loops
- Bit manipulation operations

**Hardware Requirements:**
- 8051 microcontroller
- LED connected to P1.0 (with current-limiting resistor)

**Key Concepts:**
- Port initialization
- Output control
- Simple delay functions

**Learning Outcomes:**
- Understand how to configure I/O ports
- Learn basic timing control
- Master bitwise operations

**Code Size:** ~50 lines
**Execution Time:** Adjustable (default 500ms blink)

---

## Upcoming Examples

### Button_LED.c
Button-controlled LED with debouncing

### Traffic_Light.c
Simple traffic light controller

### Relay_Control.c
Relay on/off control

---

## Prerequisites

Before working through these examples, you should understand:
- 8051 architecture basics (CPU, memory, ports)
- Number systems (binary, hexadecimal)
- Basic C programming concepts

**Recommended Reading:**
- [Hardware Architecture](../02_Hardware_Architecture/README.md)
- [I/O Ports Section](../02_Hardware_Architecture/README.md#io-ports)

---

## Common Issues

### LED Not Blinking
- Check LED polarity (anode to port pin, cathode to ground through resistor)
- Verify port is configured correctly
- Check resistor value (typically 220Ω-1kΩ)

### Port Not Responding
- Ensure correct port address
- Check if alternate functions are enabled
- Verify hardware connections

---

## Next Steps

After mastering Basic I/O, proceed to:
- [Timer Examples](../02_Timers/)
- [Interrupt Examples](../03_Interrupts/)
