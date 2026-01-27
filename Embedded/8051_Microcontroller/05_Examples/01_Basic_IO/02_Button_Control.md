# Button Control LED

## Overview

This example demonstrates how to control an LED using a push button with the 8051 microcontroller. You'll learn about digital input, button debouncing, pull-up resistors, and edge detection techniques.

---

## 📝 Complete Source Code

**File:** `Button_Control_LED.c`

```c
// Button Control LED Example
// Controls LED with push button
// Target: 8051 @ 12MHz
// Hardware: Button on P3.2, LED on P1.0

#include <reg51.h>

// Pin declarations
sbit LED = P1^0;     // LED connected to P1.0
sbit BUTTON = P3^2;  // Button connected to P3.2 (INT0)

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
    LED = 1;  // Start with LED OFF (active low)
    while(1) {  // Infinite loop
        if(BUTTON == 0) {        // Button pressed (active low)
            LED = ~LED;          // Toggle LED state
            delay(50);           // Debounce delay (50ms for most switches)
            while(BUTTON == 0);  // Wait for button release
        }
    }
}
```

---

### 🔌 Hardware Connection

```
         8051                    Button Circuit
       ┌──────┐
       │      │         VCC (5V)
       │  P3.2├──────────┐
       │      │          │
       │      │          ├───┐ 10kΩ
       │      │          │   │ (上拉电阻)
       │      │          │   └──┐
      ─┴      │          │      │
       │      │          └──────┴────┐
       │      │                       │
       │      │                      ┌┴┐  按钮
       │      │                      │ │  (开关)
       │      │                      └┬┘
       │      │                       │
       │  GND ├───────────────────────┘
       │      │
       │      │              LED Circuit
       │  P1.0├──────────────┤▌││        220Ω│
       │      │              │ │    LED    │  │
       │  GND ├──────────────┤ │           ├──┴──┴→ GND
       │      │              │ │           │
       └──────┘              └──────────────┘
```

**Component List:**
- 1× LED (any color)
- 2× Resistors: 220Ω (LED), 10kΩ (button pull-up)
- 1× Push button (normally open)
- Breadboard and jumper wires
- 8051 development board

---

### 🔬 Button Circuit Principles

#### 1. **Why Use Active Low Configuration?**

**Button NOT Pressed (Released):**
```
Button Open → P3.2 connected to VCC(5V) through pull-up resistor
            → Reads as Logic 1 (HIGH)
            → if(BUTTON == 0) is FALSE, no action taken
```

**Button Pressed:**
```
Button Closed → P3.2 directly connected to GND
              → Reads as Logic 0 (LOW)
              → if(BUTTON == 0) is TRUE, LED toggles
```

**Reasons for Using Active Low:**
1. **8051 Quasi-Bidirectional Port Characteristics**: Internal weak pull-up with only 60µA drive capability
2. **Better Noise Immunity**: Low-level signals are less susceptible to noise than high-level signals
3. **Fail-Safe Design**: Default state is HIGH (not pressed) if wire disconnects, preventing false triggers

#### 2. **Purpose of Pull-Up Resistor**

**Problem Without Pull-Up Resistor:**
```
Button Open:   P3.2 is floating → Reads undefined (could be 0 or 1, susceptible to noise)
Button Closed: P3.2 connected to GND → Reads as 0
```

**Benefits With Pull-Up Resistor:**
```
Button Open:   Current flows VCC → 10kΩ → P3.2 → Stable at 5V (Logic 1)
Button Closed: Current flows VCC → 10kΩ → GND (~0.5mA, P3.2 pulled to 0V)
```

**Resistor Value Selection:**
- **Too Large (e.g., 100kΩ)**: Slow response, susceptible to EMI (electromagnetic interference)
- **Too Small (e.g., 1kΩ)**: High power consumption, 5mA current when pressed
- **10kΩ (Recommended)**: Balances power consumption and noise immunity, only 0.5mA when pressed

---

### 📖 Code Explanation

#### 1. Pin Declarations
```c
sbit LED = P1^0;     // LED connected to P1.0
sbit BUTTON = P3^2;  // Button connected to P3.2 (INT0)
```
- Use `sbit` to declare bit-addressable pins
- P3.2 is also INT0 external interrupt pin (can be upgraded to interrupt-driven design later)

#### 2. Main Loop
```c
while(1) {  // Infinite loop
    if(BUTTON == 0) {        // Check button pressed
        LED = ~LED;          // Toggle LED state
        delay(50);           // Debounce delay (50ms for most switches)
        while(BUTTON == 0);  // Wait for release
    }
}
```

**Key Steps Analysis:**

**Step 1: Button Detection - `if(BUTTON == 0)`**
- Continuously scan P3.2 pin state
- Execute operation when low level is detected

**Step 2: Toggle LED - `LED = ~LED`**
- Use bitwise NOT operator
- If LED=0 (ON), it becomes LED=1 (OFF)
- If LED=1 (OFF), it becomes LED=0 (ON)

**Step 3: Debounce Delay - `delay(50)`**
- **Why is debouncing necessary?**
  - Mechanical buttons produce 10-50ms bounce when pressed/released
  - Can cause multiple triggers, mistaken as multiple presses

  ```
  Actual Signal: ┐┌┐┌┐┌┐┌──────┐┌┐┌┐┌┐┌┐
  5V ────────────┘└┘└┘└┘└┘      └┘└┘└┘└┘───
                ↑10-50ms bounce ↑bounce period

  Program Read:  ┌────────────────────┐
  5V ────────────┘                    └───────
                ←50ms masks bounce→
  ```

**Step 4: Wait for Release - `while(BUTTON == 0)`**
- Prevents repeated triggers while button is held down
- Next trigger only possible after complete release
- Ensures single execution per press

#### 3. Complete Timing Diagram
```
Time Axis →

Button State: ┌──────────────────┐
              │  Pressed         │  Released
5V ────────────┘                   └─────────

P3.2 Level:   ┌──────────────────┐
              │  0V              │  5V
5V ────────────┘                   └─────────

LED State:      ─────────────┐
           OFF                ON                  OFF
                             ← Toggle →

Program Exec:               Detect→Toggle→Debounce→Wait Release
```

---

### 🛠️ Testing and Troubleshooting

#### Expected Behavior
- LED initial state: OFF (dark)
- Press button: LED toggles state (ON→OFF or OFF→ON)
- Single toggle per button press
- No repeated toggling while holding button

#### Common Issues and Solutions

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| LED has no response | Button not connected properly | Check button connection and pull-up resistor |
| LED toggles multiple times per press | Debounce delay too short | Increase delay to 50-100ms |
| Must press quickly to work | Missing wait-for-release | Add while(BUTTON==0) |
| Self-triggering (false trigger) | Pin floating or interference | Ensure pull-up resistor connected |
| Button not sensitive | Pull-up resistor too large | Use 4.7kΩ-10kΩ resistor |

#### Debugging Steps

**1. Test Button Hardware:**
```c
void main() {
    while(1) {
        LED = BUTTON;  // LED directly reflects button state
        // LED should be ON when pressed, OFF when released
    }
}
```

**2. Test LED:**
```c
void main() {
    LED = 0;  // LED should be constantly ON
    while(1);
}
```

**3. Test Debouncing:**
```c
void main() {
    unsigned char count = 0;
    while(1) {
        if(BUTTON == 0) {
            count++;
            LED = ~LED;
            delay(50);  // Reduce delay to observe multiple triggers
        }
    }
    // If count > 1, need to increase debounce delay
}
```

---

### 📊 Variations and Extensions

#### Variation 1: Multiple LEDs with One Button
```c
sbit BUTTON = P3^2;

void main() {
    unsigned char state = 0;
    P1 = 0xFF;  // All LEDs OFF
    while(1) {
        if(BUTTON == 0) {
            state++;
            if(state > 3) state = 0;

            // State machine
            switch(state) {
                case 0: P1 = 0xFF;      break;  // All OFF
                case 1: P1 = 0xAA;      break;  // Pattern 1
                case 2: P1 = 0x55;      break;  // Pattern 2
                case 3: P1 = 0x00;      break;  // All ON
            }
            delay(50);           // Debounce delay
            while(BUTTON == 0);
        }
    }
}
```

#### Variation 2: Long Press Detection
```c
void main() {
    unsigned int press_time = 0;
    LED = 1;
    while(1) {
        if(BUTTON == 0) {
            delay(50);  // Short debounce
            if(BUTTON == 0) {  // Still pressed
                press_time = 0;
                while(BUTTON == 0) {
                    delay(100);
                    press_time++;
                    if(press_time > 20) {  // 2 seconds
                        // Long press action
                        LED = 0;  // Turn ON
                    }
                }
                if(press_time <= 20) {
                    // Short press action
                    LED = ~LED;  // Toggle
                }
            }
        }
    }
}
```

#### Variation 3: Double Click Detection
```c
void main() {
    unsigned int click_count = 0;
    unsigned int timeout;
    LED = 1;
    while(1) {
        if(BUTTON == 0) {
            delay(50);           // Debounce delay
            while(BUTTON == 0);  // Wait for release
            click_count++;

            // Wait for potential second click (300ms window)
            timeout = 0;
            while(timeout < 300) {
                if(BUTTON == 0) {
                    // Second click detected
                    delay(50);           // Debounce
                    while(BUTTON == 0);  // Wait for release
                    click_count++;
                    break;
                }
                delay(1);
                timeout++;
            }

            // Process clicks
            if(click_count >= 2) {
                LED = ~LED;  // Double click action
            } else {
                LED = 0;     // Single click action
            }
            click_count = 0;
        }
    }
}
```

#### Variation 4: Button with LED Indicator
```c
sbit BUTTON = P3^2;
sbit STATUS_LED = P1^0;
sbit CONTROL_LED = P1^1;

void main() {
    STATUS_LED = 1;   // OFF
    CONTROL_LED = 1;  // OFF
    while(1) {
        if(BUTTON == 0) {
            STATUS_LED = 0;     // Button pressed indicator
            CONTROL_LED = ~CONTROL_LED;
            delay(50);           // Debounce delay
            while(BUTTON == 0);
            STATUS_LED = 1;     // Button released
        }
    }
}
```

---

### ⚡ Hardware Considerations

#### Button Types
1. **Normally Open (NO)**: Most common, open circuit when not pressed
2. **Normally Closed (NC)**: Closed circuit when not pressed (requires code logic modification)
3. **Momentary**: Conducts when pressed, auto-returns after release
4. **Latch**: Locks on first press, releases on second press

#### Pull-up vs Pull-down
```c
// Pull-up (Active Low) - Recommended
if(BUTTON == 0) { ... }  // Button to GND

// Pull-down (Active High)
if(BUTTON == 1) { ... }  // Button to VCC
```

#### Internal vs External Pull-up
```c
// 8051 has internal weak pull-up (~60µA), but external 10kΩ is recommended
// Using only internal pull-up:
P3 = 0xFF;  // Enable internal pull-up
// External strong pull-up is more reliable
```

---

### 🔍 Polling vs Interrupt (Preview)

**Current code uses polling method:**
```c
while(1) {
    if(BUTTON == 0) { ... }  // Continuous checking
}
```

**Advantages:** Simple and easy to understand
**Disadvantages:** Consumes CPU resources, slow response

**Interrupt method (to be learned in 03_Interrupts):**
```c
// P3.2 is INT0 external interrupt pin
void ISR_Ex0(void) interrupt 0 {
    LED = ~LED;  // Triggers immediately on press, no polling needed
}

void main() {
    IT0 = 1;     // Edge trigger
    EX0 = 1;     // Enable INT0 interrupt
    EA = 1;      // Global interrupt enable
    while(1);    // Main program can do other tasks
}
```

**Advantages:** Real-time response, doesn't occupy main program
**Disadvantages:** Slightly more complex configuration

---

### 📚 What You've Learned

✅ **Digital Input**
- Reading button state
- Active-low vs active-high
- Pull-up resistors

✅ **Button Debouncing**
- Understanding switch bounce
- Software debouncing with delay
- Hardware debouncing options

✅ **Edge Detection**
- Detecting button press
- Waiting for release
- Preventing multiple triggers

✅ **Conditional Logic**
- if statements
- State machines
- Toggle operations

---

### 🚀 Next Steps

1. **Modify the Code:**
   - Add multiple buttons
   - Implement long-press detection
   - Create button combinations

2. **Learn Timers:**
   - Replace delay with hardware timer
   - More accurate debouncing
   - See: [Timer Examples](../02_Timers/)

3. **Use Interrupts:**
   - Convert to interrupt-driven design
   - Better response time
   - See: [Interrupt Examples](../03_Interrupts/)

---

