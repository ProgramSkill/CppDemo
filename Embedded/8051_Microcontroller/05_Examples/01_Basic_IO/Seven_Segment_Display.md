# Example 3: Seven-Segment Display

## Overview

The seven-segment display is one of the most common output devices for embedded systems. It consists of 7 LEDs (segments) arranged in a figure-8 pattern, plus an optional decimal point. By controlling which segments are lit, you can display digits 0-9 and some letters (A, b, C, d, E, F).

---

## 📝 Complete Source Code

### Version 1: Single Digit Counter (0-9)

**File:** `Seven_Segment_Counter.c`

```c
// Seven-Segment Display Counter Example
// Displays numbers 0-9 on a 7-segment display
// Target: 8051 @ 12MHz
// Hardware: Common cathode 7-segment on P1

#include <reg51.h>

// Seven-segment patterns for digits 0-9 (Common Cathode)
// Segment mapping: a=P1.0, b=P1.1, c=P1.2, d=P1.3, e=P1.4, f=P1.5, g=P1.6, dp=P1.7
//                   --- a ---
//                  |         |
//                  f         b
//                  |         |
//                   --- g ---
//                  |         |
//                  e         c
//                  |         |
//                   --- d ---    dp

unsigned char code digit_pattern[10] = {
    0x3F,  // 0: a,b,c,d,e,f ON
    0x06,  // 1: b,c ON
    0x5B,  // 2: a,b,d,e,g ON
    0x4F,  // 3: a,b,c,d,g ON
    0x66,  // 4: b,c,f,g ON
    0x6D,  // 5: a,c,d,f,g ON
    0x7D,  // 6: a,c,d,e,f,g ON
    0x07,  // 7: a,b,c ON
    0x7F,  // 8: All segments ON
    0x6F   // 9: a,b,c,d,f,g ON
};

/**
 * @brief  Simple delay function
 * @param  ms: Delay time in milliseconds
 * @retval None
 */
void delay(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    unsigned char i;
    while(1) {  // Infinite loop
        for(i = 0; i < 10; i++) {
            P1 = digit_pattern[i];  // Display digit
            delay(1000);             // Display for 1 second
        }
    }
}
```

---

### Version 2: Multiple Digits (Scanning Method)

**File:** `Seven_Segment_4Digit.c`

```c
// 4-Digit Seven-Segment Display with Scanning
// Displays different numbers on 4 digits using multiplexing
// Target: 8051 @ 12MHz
// Hardware: 4x Common cathode 7-segment displays

#include <reg51.h>

// Segment data on P1 (Common Cathode)
unsigned char code digit_pattern[10] = {
    0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F
};

// Digit selection on P2 (Active LOW)
sbit DIG1 = P2^0;  // Thousands digit
sbit DIG2 = P2^1;  // Hundreds digit
sbit DIG3 = P2^2;  // Tens digit
sbit DIG4 = P2^3;  // Units digit

// Display buffer (4 digits)
unsigned char display_value[4] = {1, 2, 3, 4};  // Display "1234"

/**
 * @brief  Display one digit
 * @param  digit: Number to display (0-9)
 * @param  position: Digit position (1-4)
 * @retval None
 */
void display_digit(unsigned char digit, unsigned char position) {
    P1 = 0x00;  // Turn off all segments first

    // Select digit position
    switch(position) {
        case 1: DIG1 = 0; DIG2 = 1; DIG3 = 1; DIG4 = 1; break;
        case 2: DIG1 = 1; DIG2 = 0; DIG3 = 1; DIG4 = 1; break;
        case 3: DIG1 = 1; DIG2 = 1; DIG3 = 0; DIG4 = 1; break;
        case 4: DIG1 = 1; DIG2 = 1; DIG3 = 1; DIG4 = 0; break;
    }

    P1 = digit_pattern[digit];  // Set segment data
}

/**
 * @brief  Scan all 4 digits rapidly (multiplexing)
 * @param  value: Array of 4 digits to display
 * @retval None
 */
void display_scan(unsigned char *value) {
    unsigned char i;
    for(i = 0; i < 4; i++) {
        display_digit(value[i], i + 1);
        delay(5);  // 5ms per digit = 20ms total cycle
    }
}

void main() {
    unsigned int counter = 0;

    while(1) {
        // Update display value (counter from 0 to 9999)
        display_value[3] = counter % 10;          // Units
        display_value[2] = (counter / 10) % 10;   // Tens
        display_value[1] = (counter / 100) % 10;  // Hundreds
        display_value[0] = (counter / 1000) % 10; // Thousands

        // Scan display continuously
        display_scan(display_value);

        // Increment counter every 200 scans (approx 4 seconds)
        // In real application, use timer interrupt!
        if(++counter >= 10000) counter = 0;
    }
}
```

---

## 🔌 Hardware Connection

### Single Digit Display

```
         8051                   Common Cathode 7-Segment
       ┌──────┐                    ┌────────────┐
       │      │                    │            │
       │ P1.0 ├────────────────────┤ a          │
       │ P1.1 ├────────────────────┤ b          │
       │ P1.2 ├────────────────────┤ c          │
       │ P1.3 ├────────────────────┤ d          │
       │ P1.4 ├────────────────────┤ e          │
       │ P1.5 ├────────────────────┤ f          │
       │ P1.6 ├────────────────────┤ g          │
       │ P1.7 ├────────────────────┤ dp         │
       │      │                    │            │
       │ GND  ├────────────────────┤ Cathode    │
       │      │                    └────────────┘
       └──────┘

         Segment Layout:
              --- a ---
             |         |
             f         b
             |         |
              --- g ---
             |         |
             e         c
             |         |
              --- d ---    dp
```

### 4-Digit Display (Multiplexed)

```
         8051                    4x Common Cathode 7-Segment
       ┌──────┐                    ┌────────────────────────┐
       │      │                    │                        │
       │ P1   ├───[8 lines]────────┤ Segments (a-dp, common) │
       │      │                    │                        │
       │ P2.0 ├────────────────────┤ Digit 1 Cathode        │
       │ P2.1 ├────────────────────┤ Digit 2 Cathode        │
       │ P2.2 ├────────────────────┤ Digit 3 Cathode        │
       │ P2.3 ├────────────────────┤ Digit 4 Cathode        │
       │      │                    │                        │
       │ GND  ├────────────────────┤ Common Ground          │
       │      │                    └────────────────────────┘
       └──────┘

Note: Each digit needs:
       - 8× 220Ω resistors on segment lines (P1.0-P1.7)
       - 1× PNP transistor (or direct connection if low current)
```

### Component List

**For Single Digit:**
- 1× Common cathode 7-segment display
- 8× Resistors 220Ω (one per segment)
- Breadboard and jumper wires
- 8051 development board

**For 4 Digits:**
- 4× Common cathode 7-segment displays
- 8× Resistors 220Ω (shared across all digits)
- 4× PNP transistors (for digit switching) OR direct connection
- 4× 1kΩ resistors (for transistor bases, if used)
- Breadboard and jumper wires
- 8051 development board

---

## 📖 Code Explanation

### 1. Segment Pattern Array

```c
unsigned char code digit_pattern[10] = {
    0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F
};
```

**How to calculate segment patterns:**

For digit **0** (segments a, b, c, d, e, f ON):
```
Bit:  7   6   5   4   3   2   1   0
Seg: dp   g   f   e   d   c   b   a
Val: 0   0   1   1   1   1   1   1
Hex: 0x3F = 0011 1111
```

For digit **1** (segments b, c ON):
```
Bit:  7   6   5   4   3   2   1   0
Seg: dp   g   f   e   d   c   b   a
Val: 0   0   0   0   0   1   1   0
Hex: 0x06 = 0000 0110
```

**Reference Table (Common Cathode):**

| Digit | Segments ON | Binary | Hex |
|-------|-------------|--------|-----|
| 0 | a,b,c,d,e,f | 0011 1111 | 0x3F |
| 1 | b,c | 0000 0110 | 0x06 |
| 2 | a,b,d,e,g | 0101 1011 | 0x5B |
| 3 | a,b,c,d,g | 0100 1111 | 0x4F |
| 4 | b,c,f,g | 0110 0110 | 0x66 |
| 5 | a,c,d,f,g | 0110 1101 | 0x6D |
| 6 | a,c,d,e,f,g | 0111 1101 | 0x7D |
| 7 | a,b,c | 0000 0111 | 0x07 |
| 8 | all | 0111 1111 | 0x7F |
| 9 | a,b,c,d,f,g | 0110 1111 | 0x6F |

**For Common Anode displays:** Invert all values (e.g., 0x3F → 0xC0)

### 2. Display Multiplexing (Scanning)

**Why multiplexing?**

Without multiplexing:
- 4 digits × 8 segments = 32 I/O pins needed
- 8051 only has 32 pins total (not all I/O)

With multiplexing:
- 8 segment lines + 4 digit lines = 12 I/O pins
- Display 4 different numbers by rapidly switching

**How it works:**

```
Time →
Digit 1: Display '1' for 5ms
Digit 2: Display '2' for 5ms
Digit 3: Display '3' for 5ms
Digit 4: Display '4' for 5ms
Repeat...

Total cycle: 20ms = 50Hz refresh rate
Human eye sees all 4 digits simultaneously!
```

**Key points:**
- Each digit is ON for 5ms, OFF for 15ms
- Duty cycle: 25% (each digit)
- Refresh rate: 50Hz (no flicker visible)
- Average brightness = 25% of full brightness

---

## 🔬 Common Anode vs Common Cathode

### Common Cathode (Used in examples)

```
Internal connection:
       ┌───┐
       │ ┃ │
    a─┤ ┃ ├──GND ← Common cathode
    b─┤ ┃ ├──GND
    c─┤ ┃ ├──GND
       ...
       └───┘

Logic:
- Logic 1 (5V) → Segment ON
- Logic 0 (0V) → Segment OFF
```

### Common Anode

```
Internal connection:
       ┌───┐
       │ ┃ │
    a─┤ ┃ ├──VCC ← Common anode
    b─┤ ┃ ├──VCC
    c─┤ ┃ ├──VCC
       ...
       └───┘

Logic:
- Logic 0 (0V) → Segment ON
- Logic 1 (5V) → Segment OFF

Pattern values are INVERTED!
```

**Common Anode Patterns:**

```c
unsigned char code digit_pattern_ca[10] = {
    0xC0,  // 0: ~0x3F
    0xF9,  // 1: ~0x06
    0xA4,  // 2: ~0x5B
    0xB0,  // 3: ~0x4F
    0x99,  // 4: ~0x66
    0x92,  // 5: ~0x6D
    0x82,  // 6: ~0x7D
    0xF8,  // 7: ~0x07
    0x80,  // 8: ~0x7F
    0x90   // 9: ~0x6F
};
```

**How to identify:**

| Type | How to Test | Typical Part Number |
|------|-------------|---------------------|
| Common Cathode | Multimeter: Common to GND, segments light with positive probe | 5161AS, KC52101 |
| Common Anode | Multimeter: Common to VCC, segments light with negative probe | 5161BS, KC52102 |

---

## 🛠️ Testing and Troubleshooting

### Expected Behavior

**Single Digit:**
- Displays 0, 1, 2, ..., 9 sequentially
- Each digit displays for 1 second
- Pattern repeats continuously

**4-Digit:**
- All 4 digits appear to display simultaneously
- Numbers increment slowly

### Common Issues and Solutions

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| All segments OFF | Wrong connection type | Check common cathode vs anode |
| Wrong segments light | Bit mapping error | Verify P1.0→a, P1.1→b, etc. |
| Display is dim | Current limiting too high | Reduce resistor to 150Ω-220Ω |
| Flickering display | Scan rate too slow | Reduce delay to 3-5ms |
| Ghosting (segments partially on) | Turn-off delay needed | Add blank period between digits |
| Some segments always ON | Short circuit or wrong code | Check for solder bridges |
| Nothing displays | No power or wrong port | Check VCC, GND, and port connections |

### Debugging Steps

**1. Test All Segments:**
```c
void main() {
    P1 = 0xFF;  // All segments ON
    while(1);
    // All segments should light up
}
```

**2. Test Individual Segments:**
```c
void main() {
    unsigned char i;
    while(1) {
        for(i = 0; i < 8; i++) {
            P1 = (1 << i);  // Light one segment at a time
            delay(500);
        }
    }
}
```

**3. Check Pattern Array:**
```c
void main() {
    while(1) {
        P1 = 0x3F;  // Should display '0'
        delay(1000);
        P1 = 0x06;  // Should display '1'
        delay(1000);
        // Test each pattern individually
    }
}
```

**4. Verify Display Type:**
```c
// For common cathode
void main() {
    P1 = 0xFF;  // Should light all segments
    while(1);
}

// If all segments OFF, you have common anode!
// Invert: P1 = 0x00;
```

---

## 📊 Variations and Extensions

### Variation 1: Display Letters (A-F)

```c
unsigned char code letter_pattern[6] = {
    0x77,  // A: a,b,c,e,f,g
    0x7C,  // b: c,d,e,f,g
    0x39,  // C: a,d,e,f
    0x5E,  // d: b,c,d,e,g
    0x79,  // E: a,d,e,f,g
    0x71   // F: a,e,f,g
};

void main() {
    unsigned char i;
    while(1) {
        for(i = 0; i < 6; i++) {
            P1 = letter_pattern[i];
            delay(1000);
        }
    }
}
```

### Variation 2: Digital Clock Display

```c
// Display HH:MM format
unsigned char hours = 12, minutes = 34;
unsigned char display_buffer[4];

void update_time() {
    display_buffer[0] = hours / 10;
    display_buffer[1] = hours % 10;
    display_buffer[2] = minutes / 10;
    display_buffer[3] = minutes % 10;
}

void main() {
    while(1) {
        update_time();
        display_scan(display_buffer);

        // In real application, use timer interrupt to count time!
    }
}
```

### Variation 3: Decimal Point Control

```c
void display_with_dp(unsigned char digit, bit show_dp) {
    if(show_dp) {
        P1 = digit_pattern[digit] | 0x80;  // Set bit 7 (dp)
    } else {
        P1 = digit_pattern[digit];
    }
}

void main() {
    // Display "12.34"
    display_digit(1, 1); delay(5);
    display_digit(2, 2); delay(5);
    P1 = digit_pattern[3] | 0x80;  // Add DP
    // ... etc
}
```

### Variation 4: Brightness Control (PWM)

```c
void display_with_brightness(unsigned char digit, unsigned char brightness) {
    // brightness: 0-100 (percentage)
    unsigned char on_time = (brightness * 5) / 100;

    P1 = digit_pattern[digit];  // Turn ON
    delay(on_time);              // ON time
    P1 = 0x00;                  // Turn OFF
    delay(5 - on_time);          // Complete 5ms cycle
}
```

### Variation 5: Button Counter

```c
sbit BUTTON = P3^2;
unsigned int count = 0;

void main() {
    P1 = digit_pattern[0];  // Start with 0
    while(1) {
        if(BUTTON == 0) {
            count++;
            if(count > 9) count = 0;
            P1 = digit_pattern[count];
            delay(200);
            while(BUTTON == 0);  // Wait for release
        }
    }
}
```

### Variation 6: Rolling Number Effect

```c
void roll_number(unsigned char target) {
    unsigned char i;
    for(i = 0; i <= target; i++) {
        P1 = digit_pattern[i];
        delay(50);
    }
}

void main() {
    while(1) {
        roll_number(9);  // Roll from 0 to 9
        delay(1000);
    }
}
```

---

## ⚡ Hardware Considerations

### Current Limitations

**Per segment:**
- Forward current: 5-20mA typical
- Peak current: 30mA max
- Average current (with 4-digit multiplexing): 5mA × 25% duty = 1.25mA

**Per digit:**
- Max current: All 8 segments × 20mA = 160mA
- **Problem:** Exceeds 8051 port limit!

**Solutions:**

1. **Current Limiting Resistors:**
   ```
   R = (VCC - VLED) / I
   R = (5V - 2V) / 10mA = 300Ω
   Use 220Ω-330Ω per segment
   ```

2. **Use Transistor Drivers:**
   ```
   For high brightness displays:
   - Use ULN2003 Darlington array for sink (common cathode)
   - Use UDN2981 source driver for source (common anode)
   ```

### Segment Resistors vs Digit Resistors

**Per-segment resistors (recommended):**
```
8051 P1.0 ──220Ω──┤ a ├─┐
8051 P1.1 ──220Ω──┤ b ├─┤
8051 P1.2 ──220Ω──┤ c ├─┼─┤ Common ├─ GND
8051 P1.3 ──220Ω──┤ d ├─┤ Cathode │
8051 P1.4 ──220Ω──┤ e ├─┤         │
8051 P1.5 ──220Ω──┤ f ├─┤         │
8051 P1.6 ──220Ω──┤ g ├─┘         │
8051 P1.7 ──220Ω──┤dp ├───────────┘
```

**Advantages:**
- Consistent brightness across all segments
- Protects each LED individually
- Can compensate for different LED forward voltages

### Ghosting Prevention

**Problem:** When switching digits, previous digit's segments briefly light the wrong digit.

**Solution:** Add blank period
```c
void display_digit_clean(unsigned char digit, unsigned char position) {
    P1 = 0x00;  // Turn OFF all segments

    // Select digit position
    switch(position) {
        case 1: DIG1 = 0; DIG2 = 1; DIG3 = 1; DIG4 = 1; break;
        case 2: DIG1 = 1; DIG2 = 0; DIG3 = 1; DIG4 = 1; break;
        case 3: DIG1 = 1; DIG2 = 1; DIG3 = 0; DIG4 = 1; break;
        case 4: DIG1 = 1; DIG2 = 1; DIG3 = 1; DIG4 = 0; break;
    }

    delay(1);  // Short blank period to prevent ghosting

    P1 = digit_pattern[digit];  // Turn ON segments
}
```

---

## 📚 What You've Learned

✅ **Seven-Segment Display Basics**
- Common cathode vs common anode
- Segment mapping and layout
- Pattern calculation

✅ **Display Control**
- Writing segment data to ports
- Displaying numbers 0-9
- Displaying letters A-F

✅ **Multiplexing Technique**
- Scanning multiple digits
- Timing considerations
- Refresh rate calculation

✅ **Hardware Design**
- Current limiting resistors
- Brightness control
- Ghosting prevention

✅ **Code Organization**
- Lookup tables
- Display buffer
- Modular functions

---

## 🚀 Next Steps

After mastering this example:

1. **Build a Digital Clock:**
   - Add timer interrupts for accurate timing
   - Display HH:MM:SS format
   - Add buttons to set time

2. **Create a Counter:**
   - Button input to increment
   - Multi-digit display (0000-9999)
   - Overflow detection

3. **Display Sensor Data:**
   - Temperature from ADC
   - Voltage measurement
   - Speed/rotation counter

4. **Advanced Features:**
   - Scrolling text
   - Brightness adjustment
   - Special effects (blinking, rolling)

5. **Learn Timers:**
   - Replace delay with hardware timers
   - Precise timing for clocks
   - See: [Timer Examples](../02_Timers/)

6. **Use Interrupts:**
   - Timer interrupt for display scanning
   - Button interrupt for counter
   - See: [Interrupt Examples](../03_Interrupts/)

---

## 🎓 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Common Cathode** | All cathodes connected together to GND |
| **Common Anode** | All anodes connected together to VCC |
| **Segment Mapping** | a,b,c,d,e,f,g,dp → port bits |
| **Lookup Table** | Pre-calculated patterns for each digit |
| **Multiplexing** | Rapidly switching digits to display multiple values |
| **Duty Cycle** | Percentage of time each digit is ON |
| **Refresh Rate** | How often the display is updated per second |
| **Ghosting** | Unintended partial illumination during transitions |

---

## 🐛 Debug Checklist

Before asking for help, check:

- [ ] Correct display type (common cathode vs anode)
- [ ] Pattern array matches display type
- [ ] All 8 segment lines connected (P1.0-P1.7)
- [ ] Resistors on each segment (220Ω-470Ω)
- [ ] Digit control lines connected properly
- [ ] Scan delay 3-5ms (no flicker)
- [ ] VCC and GND connected
- [ ] No solder bridges on display pins
- [ ] Brightness not too dim or too bright

---

## 📖 Additional Resources

### Datasheets
- [Seven-Segment Display Guide](https://www.sparkfun.com/datasheets/Components/LED/7-Segment.pdf)
- [8051 I/O Ports](../../02_Hardware_Architecture/README.md#io-ports)

### Learning Materials
- [Multiplexing Tutorial](https://www.electronics-tutorials.ws/blog/7-segment-display-tutorial.html)
- [LED Current Limiting](https://www.allaboutcircuits.com/textbook/direct-current/chpt-5/led-arrays/)

### Tools
- **7-Segment Pattern Calculator:** Online tools to generate hex codes
- **Simulators:** Proteus (includes 7-segment models)
- **Debuggers:** Real hardware + logic analyzer

---

## 🤝 Contributing

Have improvements? Found a bug?
- Add more variations
- Improve documentation
- Fix errors
- Add troubleshooting tips

---

**Difficulty:** ⭐⭐ Beginner-Intermediate
**Time to Complete:** 1-2 hours
**Hardware Required:** 7-segment display, resistors, 8051 board

**Happy Coding!** 💡
