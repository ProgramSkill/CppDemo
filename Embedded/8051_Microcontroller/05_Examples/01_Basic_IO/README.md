# Basic I/O Examples

## Overview

This section contains fundamental I/O operations examples for the 8051 microcontroller. These examples demonstrate how to interact with ports, control LEDs, read buttons, and perform basic input/output operations.

---

## Example 1: LED Blink

### 📝 Complete Source Code

**File:** `LED_Blink.c`

```c
// LED Blink Example
// Blinks an LED connected to P1.0
// Target: 8051 @ 12MHz
// Hardware: LED on P1.0 with 220Ω resistor to ground

#include <reg51.h>

// Bit-addressable LED declaration
sbit LED = P1^0;  // LED connected to P1.0

/**
 * @brief  Simple delay function
 * @param  ms: Delay time in milliseconds
 * @retval None
 * Note:   Approximate timing, not precise
 */
void delay(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);  // Calibrated for ~1ms at 12MHz
}

void main() {
    // Main program loop
    while(1) {  // Infinite loop
        LED = 0;      // Turn ON LED (active low)
        delay(500);   // Wait 500ms

        LED = 1;      // Turn OFF LED
        delay(500);   // Wait 500ms
    }
}
```

---

### 🔌 Hardware Connection

```
         8051                    LED Circuit
       ┌──────┐              ┌──────────────┐
       │      │              │              │
       │  P1.0├──────────────┤▌││        220Ω│
       │      │              │ │    LED    │  │
       │  GND ├──────────────┤ │           ├──┴──┴→ GND
       │      │              │ │           │
       └──────┘              └──────────────┘

Alternative (Active High):
       ┌──────┐              ┌──────────────┐
       │      │              │              │
       │  P1.0├───220Ω────────┤▌││           │
       │      │              │ │    LED    │──┴──┴→ GND
       │  VCC │              │ │           │
       └──────┘              └──────────────┘
```

**Component List:**
- 1× LED (any color)
- 1× Resistor 220Ω (or 330Ω, 470Ω)
- Breadboard and jumper wires
- 8051 development board

---

### 📖 Code Explanation

#### 1. Include Header File
```c
#include <reg51.h>
```
- Includes 8051 special function register definitions
- Provides access to P0, P1, P2, P3, and other SFRs

#### 2. Bit-Addressable LED Declaration
```c
sbit LED = P1^0;  // LED connected to P1.0
```
- `sbit`: Single bit declaration
- `P1^0`: Bit 0 of Port 1
- Allows individual bit access: `LED = 0` or `LED = 1`

**Alternative methods:**
```c
// Method 1: Direct bit access
sbit LED = P1^0;

// Method 2: Use entire port
#define LED P1_0  // Or use bit-mask

// Method 3: Direct port manipulation (not recommended for single bit)
// LED = 0;  // Would affect entire P1 port!
```

#### 3. Delay Function
```c
void delay(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);  // Calibrated for ~1ms at 12MHz
}
```
- Simple software delay using nested loops
- **Not precise** - depends on compiler and optimization
- **Calibration**:
  - Inner loop count (123) calibrated for 12MHz crystal
  - May need adjustment for different frequencies
  - For precise timing, use hardware timers (see Timer examples)

**Timing Calculation:**
```
At 12MHz: 1 machine cycle = 1µs
Each loop iteration ≈ 8-10 machine cycles
123 iterations × 8 cycles = ~1ms
```

#### 4. Main Function
```c
void main() {
    while(1) {  // Infinite loop
        LED = 0;      // Turn ON LED (active low)
        delay(500);   // Wait 500ms
        LED = 1;      // Turn OFF LED
        delay(500);   // Wait 500ms
    }
}
```
- `while(1)`: Infinite loop (embedded programs never exit)
- `LED = 0`: Output LOW (turns on LED in active-low configuration)
- `LED = 1`: Output HIGH (turns off LED)

---

### 🎯 Active Low vs Active High

#### Active Low (Common in 8051)
```c
LED = 0;  // LED ON  (sinks current)
LED = 1;  // LED OFF
```
- LED anode to VCC through resistor
- LED cathode to port pin
- Port pin sinks current (can sink ~20mA)

#### Active High
```c
LED = 1;  // LED ON  (sources current)
LED = 0;  // LED OFF
```
- LED anode to port pin through resistor
- LED cathode to GND
- Port pin sources current (can source ~60µA only - WEAK!)

**Recommendation:** Use active-low configuration for better drive capability.

---

### 🔧 Testing and Troubleshooting

#### Expected Behavior
- LED blinks at 1 Hz (500ms ON, 500ms OFF)
- Pattern repeats continuously

#### If LED Doesn't Blink

**1. Check Polarity**
```
LED Long Lead (Anode)  ──→ Port Pin (for active high)
LED Short Lead (Cathode) ──→ GND
```

**2. Measure Voltage**
- P1.0 should toggle between 0V and 5V
- Use multimeter or oscilloscope

**3. Check Connections**
- Verify 8051 is powered (5V on VCC pin)
- Check GND connection
- Confirm crystal oscillator is running

**4. Simple Test**
```c
// Test: Constant ON
void main() {
    LED = 0;  // Should be always ON
    while(1);
}
```

**5. Reduce Delay**
```c
delay(100);  // Faster blink (100ms)
```

#### Common Issues and Solutions

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| LED always ON | Wrong polarity | Reverse LED connections |
| LED always OFF | No power | Check VCC and GND |
| Dim LED | Wrong resistor | Use 220Ω-470Ω |
| No blink | Not programmed | Verify programming succeeded |
| Erratic blink | Wrong crystal | Adjust delay loop count |

---

### 📊 Variations and Extensions

#### Variation 1: Multiple LEDs
```c
sbit LED1 = P1^0;
sbit LED2 = P1^1;
sbit LED3 = P1^2;
sbit LED4 = P1^3;

void main() {
    while(1) {
        LED1 = 0; LED2 = 1; LED3 = 0; LED4 = 1;
        delay(500);
        LED1 = 1; LED2 = 0; LED3 = 1; LED4 = 0;
        delay(500);
    }
}
```

#### Variation 2: Running LED (Chaser)
```c
void main() {
    unsigned char pattern;
    while(1) {
        for(pattern = 0; pattern < 8; pattern++) {
            P1 = ~(1 << pattern);  // Shift LED
            delay(200);
        }
    }
}
```

#### Variation 3: Morse Code SOS
```c
void dot() {
    LED = 0; delay(200); LED = 1; delay(200);
}

void dash() {
    LED = 0; delay(600); LED = 1; delay(200);
}

void main() {
    while(1) {
        // SOS: ... --- ...
        dot(); dot(); dot();    // S
        delay(400);
        dash(); dash(); dash();  // O
        delay(400);
        dot(); dot(); dot();    // S
        delay(1000);            // Long pause
    }
}
```

#### Variation 4: Blink Rate Control
```c
void main() {
    unsigned int speed = 500;
    while(1) {
        // Increase speed each cycle
        if(speed > 50) speed -= 50;

        LED = 0;
        delay(speed);
        LED = 1;
        delay(speed);
    }
}
```

---

### 🔬 Understanding Port Operations

#### Writing to Individual Bits
```c
sbit LED = P1^0;
LED = 0;  // Clear bit 0 only
```

#### Writing to Entire Port
```c
P1 = 0x55;  // Binary: 01010101
           // Turns on LEDs on pins 1, 3, 5, 7 (active low)
```

#### Bit Manipulation
```c
// Set bit (turn OFF if active low)
P1 |= (1 << 0);   // P1 = P1 | 0x01

// Clear bit (turn ON if active low)
P1 &= ~(1 << 0);  // P1 = P1 & 0xFE

// Toggle bit
P1 ^= (1 << 0);   // P1 = P1 ^ 0x01
```

---

### ⚡ Power Considerations

**Current Limits:**
- Single I/O pin: 20mA max (sink), 60µA (source)
- Entire port: 71mA max
- Entire chip: 150mA max

**LED Current Calculation:**
```
I = (VCC - VLED) / R
I = (5V - 2V) / 220Ω = 13.6mA  (Safe!)
```

**Safe Values:**
- Resistor: 220Ω to 1kΩ
- LED current: 5-15mA per LED
- Max LEDs per port: 8 (but check total current!)

---

### 📚 What You've Learned

✅ **Port Configuration**
- How to declare bit-addressable pins
- Writing to individual port pins
- Writing to entire ports

✅ **Digital Output**
- Active-low vs active-high configurations
- Current sinking vs sourcing
- LED drive calculations

✅ **Timing**
- Software delay loops
- Limitations of software delays
- Need for hardware timers (next lesson!)

✅ **Program Structure**
- Infinite loop in embedded systems
- Basic C syntax for 8051
- Including header files

---

### 🚀 Next Steps

After mastering this example:

1. **Modify the Code:**
   - Change blink rate
   - Add more LEDs
   - Create patterns

2. **Learn Timers:**
   - Replace delay with hardware timer
   - See: [Timer Examples](../02_Timers/)

3. **Add Input:**
   - Read button to control LED
   - Learn conditional statements

4. **Use Interrupts:**
   - Timer interrupt for precise blinking
   - See: [Interrupt Examples](../03_Interrupts/)

---

## Example 2: Button Control LED (Coming Soon)

**Description:** Read a button to control LED state

**Preview:**
```c
sbit BUTTON = P3^2;
sbit LED = P1^0;

void main() {
    LED = 1;  // Start with LED OFF
    while(1) {
        if(BUTTON == 0) {  // Button pressed (active low)
            LED = ~LED;    // Toggle LED
            delay(200);    // Debounce delay
            while(BUTTON == 0);  // Wait for release
        }
    }
}
```

---

## Example 3: Traffic Light (Coming Soon)

**Description:** Simulate traffic light with Red, Yellow, Green LEDs

**Preview:**
```c
sbit RED = P1^0;
sbit YELLOW = P1^1;
sbit GREEN = P1^2;

void main() {
    while(1) {
        GREEN = 0; YELLOW = 1; RED = 1;    // Green ON
        delay(5000);                         // 5 seconds

        YELLOW = 0; GREEN = 1;              // Yellow ON
        delay(2000);                         // 2 seconds

        RED = 0; YELLOW = 1;                // Red ON
        delay(5000);                         // 5 seconds
    }
}
```

---

## 🎓 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **sbit** | Single bit declaration for port pins |
| **Active Low** | Logic 0 = ON, Logic 1 = OFF |
| **Port Write** | Writing to entire port or individual bits |
| **Software Delay** | Inaccurate but simple timing method |
| **Infinite Loop** | Embedded programs never exit |
| **Current Sink** | Port pin sinks current to ground |
| **Current Source** | Port pin sources current from VCC (weak!) |

---

## 🐛 Debug Checklist

Before asking for help, check:
- [ ] 5V power supply connected
- [ ] GND connected
- [ ] Crystal oscillator working
- [ ] LED polarity correct
- [ ] Resistor value appropriate (220Ω-470Ω)
- [ ] Code compiled without errors
- [ ] Programming succeeded
- [ ] Correct port pin used
- [ ] Delay value reasonable (not 0 or too large)

---

## 📖 Additional Resources

### Datasheets
- [AT89C51 Datasheet](https://www.atmel.com/Images/Atmel-8051-Microcontroller-AT89C51-Datasheet.pdf)
- [8051 Architecture Reference](../../02_Hardware_Architecture/README.md)

### Learning Materials
- [I/O Ports Detailed Guide](../../02_Hardware_Architecture/README.md#io-ports)
- [Pin Configuration](../../02_Hardware_Architecture/README.md#pin-configuration-40-pin-dip)

### Tools
- **Compilers:** SDCC, Keil C51
- **Simulators:** Proteus, Keil Simulator
- **Programmers:** USBasp, ISP

---

## 🤝 Contributing

Have improvements? Found a bug?
- Add more variations
- Improve documentation
- Fix errors
- Add troubleshooting tips

---

**Difficulty:** ⭐ Beginner
**Time to Complete:** 30 minutes
**Hardware Required:** LED, resistor, 8051 board

**Happy Coding!** 💡
