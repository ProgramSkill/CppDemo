# Timer Examples

## Overview

This section demonstrates the use of 8051 timers for precise timing operations. Timers are essential for generating accurate delays, measuring time intervals, and creating periodic events.

---

## Example 1: Timer Delay (Interrupt-Driven)

### 📝 Complete Source Code

**File:** `Timer_Delay.c`

```c
// Timer Delay Example - Interrupt-Driven
// Generates precise 50ms delays using Timer 0 Mode 1
// Target: 8051 @ 12MHz
// Hardware: LED on P1.0 (optional, for visualization)

#include <reg51.h>

// LED pin (optional, for visualization)
sbit LED = P1^0;

// Timer reload values for 50ms @ 12MHz
// Calculation: 65536 - (50000µs / 1µs) = 15536 = 0x3CB0
#define TH0_RELOAD 0x3C
#define TL0_RELOAD 0xB0

// Global variables
volatile unsigned char timer0_overflow = 0;

/**
 * @brief  Timer 0 Interrupt Service Routine
 * @param  None
 * @retval None
 * @note   Called every time Timer 0 overflows
 */
void timer0_isr(void) interrupt 1
{
    // Reload timer values (Mode 1 requires manual reload)
    TH0 = TH0_RELOAD;
    TL0 = TL0_RELOAD;

    // Indicate timer event (toggle LED if connected)
    LED = ~LED;

    // Increment overflow counter
    timer0_overflow++;
}

/**
 * @brief  Initialize Timer 0 for 50ms periodic interrupts
 * @param  None
 * @retval None
 */
void timer0_init(void)
{
    // Configure Timer 0
    TMOD &= 0xF0;      // Clear Timer 0 mode bits
    TMOD |= 0x01;      // Timer 0, Mode 1 (16-bit timer)

    // Load initial timer values
    TH0 = TH0_RELOAD;
    TL0 = TL0_RELOAD;

    // Enable Timer 0 interrupt
    ET0 = 1;           // Enable Timer 0 overflow interrupt
    EA = 1;            // Enable global interrupts

    // Start Timer 0
    TR0 = 1;           // Start Timer 0
}

/**
 * @brief  Create delay using Timer 0 interrupts
 * @param  ms: Delay time in milliseconds (multiples of 50ms)
 * @retval None
 * @note   Blocks for specified time using polling method
 */
void timer_delay(unsigned int ms)
{
    unsigned int count = 0;
    unsigned int target = ms / 50;  // 50ms per interrupt

    timer0_overflow = 0;

    while(count < target) {
        if(timer0_overflow) {
            count++;
            timer0_overflow = 0;
        }
    }
}

/**
 * @brief  Simple software delay (for testing)
 * @param  ms: Delay time in milliseconds
 * @retval None
 */
void simple_delay_ms(unsigned int ms)
{
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main(void)
{
    // Initialize Timer 0
    timer0_init();

    // Main program loop
    while(1) {
        // Toggle LED to show program running
        LED = 0;
        timer_delay(500);  // Wait 500ms (10 × 50ms)

        LED = 1;
        timer_delay(500);  // Wait 500ms
    }
}
```

---

### 🔌 Hardware Connection (Optional LED)

```
         8051                    LED Circuit
       ┌──────┐              ┌──────────────┐
       │      │              │              │
       │  P1.0├──────────────┤▌││        220Ω│
       │      │              │ │    LED      │  │
       │  GND ├──────────────┤ │           ├──┴──┴→ GND
       │      │              └──────────────┘
       └──────┘

Active Low Configuration:
- LED ON when P1.0 = 0
- LED OFF when P1.0 = 1
```

**Component List:**
- 8051 microcontroller @ 12MHz (or 11.0592MHz)
- LED (any color)
- Resistor 220Ω (or 330Ω, 470Ω)
- Breadboard and jumper wires

---

### 📖 Code Explanation

#### 1. Timer Calculation

```c
// For 50ms delay @ 12MHz crystal
// Machine Cycle = 12 / 12MHz = 1µs
// Desired Delay = 50ms = 50,000µs
// Timer Counts Needed = 50,000µs / 1µs = 50,000
// Timer Value = 65,536 - 50,000 = 15,536
// In Hex: 15,536 = 0x3CB0
// TH0 = 0x3C (high byte)
// TL0 = 0xB0 (low byte)
```

**Calculation Formula:**
```
Timer Value = 65536 - (Desired Delay / Machine Cycle)

Where:
- 65536 = 2^16 (for 16-bit timer)
- Desired Delay = delay time in microseconds
- Machine Cycle = 12 / Crystal Frequency in MHz
```

#### 2. Timer Initialization

```c
void timer0_init(void)
{
    TMOD &= 0xF0;      // Clear Timer 0 mode bits
    TMOD |= 0x01;      // Timer 0, Mode 1 (16-bit timer)

    TH0 = TH0_RELOAD;
    TL0 = TL0_RELOAD;

    ET0 = 1;           // Enable Timer 0 interrupt
    EA = 1;            // Global interrupt enable
    TR0 = 1;           // Start Timer 0
}
```

**Step by Step:**
1. **TMOD Configuration**
   - Clear Timer 0 bits (keep Timer 1 unchanged)
   - Set Mode 1 (16-bit timer)

2. **Load Initial Values**
   - Set high and low bytes
   - Timer starts counting from these values

3. **Interrupt Enable**
   - ET0 = 1: Enable Timer 0 overflow interrupt
   - EA = 1: Master interrupt enable

4. **Start Timer**
   - TR0 = 1: Timer begins counting

#### 3. Interrupt Service Routine

```c
void timer0_isr(void) interrupt 1
{
    // Critical: Reload timer FIRST!
    TH0 = TH0_RELOAD;
    TL0 = TL0_RELOAD;

    // Timer-specific code here
    LED = ~LED;  // Toggle LED

    // Update counters
    timer0_overflow++;
}
```

**Important Notes:**
- **interrupt 1**: Specifies this is Timer 0 ISR (vector address 000BH)
- **Reload First**: Always reload timer before any other operation
- **Keep ISR Short**: Don't put long delays in ISR

#### 4. Delay Function

```c
void timer_delay(unsigned int ms)
{
    unsigned int count = 0;
    unsigned int target = ms / 50;  // 50ms per interrupt

    timer0_overflow = 0;

    while(count < target) {
        if(timer0_overflow) {
            count++;
            timer0_overflow = 0;
        }
    }
}
```

**How It Works:**
1. Set target number of interrupts (ms / 50)
2. Reset overflow counter
3. Wait until counter reaches target
4. Uses volatile variable to see ISR updates

---

### 🎯 Understanding Timer Modes

#### Mode 0: 13-bit Timer (Legacy)

```
┌─────────┬─────────┐
│   THx   │ TLx<4:0>│  13-bit counter
│ (8 bits)│ (5 bits)│
└─────────┴─────────┘
     │
     └──→ Counts from 0 to 8191
```

**Use Case:** Compatibility with 8048 microcontroller

#### Mode 1: 16-bit Timer (Most Common) ⭐

```
┌─────────┬─────────┐
│   THx   │   TLx   │  16-bit counter
│ (8 bits)│ (8 bits)│
└─────────┴─────────┘
     │
     └──→ Counts from 0 to 65535
```

**Use Case:** General-purpose timing, precise delays

**Overflow Calculation:**
```
@ 12MHz:
Time to Overflow = 65536 × 1µs = 65.536ms

@ 11.0592MHz:
Time to Overflow = 65536 × 1.085µs = 71.1ms
```

#### Mode 2: 8-bit Auto-Reload

```
┌─────────┐
│   THx   │  Reload value (static)
└────┬────┘
     │
     ├──> TLx acts as counter (8-bit)
     │
     └──> When TLx overflows:
          TLx = THx (auto reload)
          Interrupt occurs
```

**Use Case:** Baud rate generation for serial port

#### Mode 3: Split Timer (Timer 0 Only)

```
Timer 0 is split into two 8-bit timers:
┌─────────┐         ┌─────────┐
│   TL0   │         │   TH0   │
│ (8-bit) │         │ (8-bit) │
└─────────┘         └─────────┘
  Uses T0 control       Uses T1 control
```

**Use Case:** When you need two independent 8-bit timers

---

### 🔧 Testing and Verification

#### Expected Behavior

**Without LED:**
- Program runs continuously
- Timer generates interrupts every 50ms

**With LED:**
- LED toggles every 50ms
- Result: 10 Hz blink rate (100ms period)

#### Measurement Methods

**1. Frequency Counter/Oscilloscope**
- Measure P1.0 toggle frequency
- Should be 10 Hz (50ms ON + 50ms OFF)

**2. Logic Analyzer**
- Capture timer interrupt timing
- Verify 50ms period

**3. Simple Observation**
- LED should blink steadily
- Compare with known-good delay

---

### 📊 Timing Accuracy Analysis

#### Polling Method (Not Recommended)

```c
// Poll TF0 flag
void delay_polling(unsigned int ms)
{
    unsigned int i;
    for(i = 0; i < ms; i++) {
        TH0 = 0x3C;
        TL0 = 0xB0;
        TR0 = 1;
        while(!TF0);  // Wait here (CPU blocked!)
        TR0 = 0;
        TF0 = 0;
    }
}
```

**Disadvantages:**
- ❌ CPU blocked in while loop
- ❌ Can't do other tasks
- ❌ Accumulated error from overhead

#### Interrupt Method (Recommended) ✅

```c
// Interrupt-driven
volatile unsigned char flag = 0;

void timer0_isr(void) interrupt 1 {
    TH0 = 0x3C;
    TL0 = 0xB0;
    flag = 1;  // Signal main program
}

void delay_interrupt(unsigned int ms) {
    // CPU can do other work here!
    while(!flag);
    flag = 0;
}
```

**Advantages:**
- ✅ CPU free to do other tasks
- ✅ More accurate timing
- ✅ Can handle multiple events
- ✅ Better for real-time systems

---

### 🎓 Variations and Extensions

#### Variation 1: Variable Delay

```c
// Delay function with variable time
void timer_delay_variable(unsigned int ms)
{
    unsigned int count = 0;
    unsigned int target = (ms + 25) / 50;  // Round to nearest 50ms

    timer0_overflow = 0;

    while(count < target) {
        if(timer0_overflow) {
            count++;
            timer0_overflow = 0;
        }
    }
}
```

#### Variation 2: Multiple Timer Intervals

```c
// Generate 100ms, 500ms, 1s delays
void delay_100ms(void)  { timer_delay(100); }
void delay_500ms(void)  { timer_delay(500); }
void delay_1sec(void)   { timer_delay(1000); }
```

#### Variation 3: Long Delays (>65ms)

```c
// Delay for multiple timer overflows
void timer_delay_long(unsigned int ms)
{
    unsigned int overflows = ms / 50;

    timer0_overflow = 0;

    while(timer0_overflow < overflows) {
        // Wait for ISR to increment counter
    }
}
```

#### Variation 4: Precise Frequency Generation

```c
// Generate 1 Hz square wave on P1.0
// Toggle every 500ms (10 interrupts @ 50ms each)

volatile unsigned char half_sec = 0;

void timer0_isr(void) interrupt 1
{
    TH0 = 0x3C;
    TL0 = 0xB0;

    half_sec++;
    if(half_sec >= 10) {  // 10 × 50ms = 500ms
        half_sec = 0;
        LED = ~LED;  // Toggle LED
    }
}
```

---

### ⚡ Reload Value Calculator

#### Quick Reference Table (12MHz Crystal)

| Delay | THx | TLx | Hex Value | Calculation |
|-------|-----|-----|-----------|-------------|
| 1ms | 0xFC | 0x18 | FC18H | 65536 - 1000 |
| 10ms | 0xD8 | 0xF0 | D8F0H | 65536 - 10000 |
| 50ms | 0x3C | 0xB0 | 3CB0H | 65536 - 50000 |
| 100ms | 0x76 | 0x60 | 7660H | 65536 - 100000 |
| 500ms | 0x0C | 0x20 | 0C20H | 65536 - 500000 |

#### For Other Crystal Frequencies

**11.0592MHz:**
```c
// Machine Cycle = 12 / 11.0592MHz = 1.085µs
// For 50ms: 50000 / 1.085 = 46082 = 0xB422
TH0 = 0xB4;
TL0 = 0x22;
```

**24MHz:**
```c
// Machine Cycle = 12 / 24MHz = 0.5µs
// For 50ms: 50000 / 0.5 = 100000 = 0x7B20 (overflow!)
// Need multiple overflows or adjust delay
```

---

### 🔬 Advanced: Timer Capture Mode

**Not available in standard 8051**, but available in some variants:

```c
// Timer 2 in capture mode (8052)
// Captures timer value on external event
unsigned int capture_period(void)
{
    unsigned int rising_edge, falling_edge;

    // Wait for rising edge on T2EX pin
    while(!T2EX);  // Capture occurs
    rising_edge = T2;  // Read captured value

    // Wait for falling edge
    while(T2EX);
    falling_edge = T2;

    // Calculate period
    return falling_edge - rising_edge;
}
```

---

### 🐛 Common Issues and Solutions

#### Problem 1: Inaccurate Delays

**Symptoms:**
- LED blinks at wrong rate
- Delay too long or too short

**Solutions:**
1. **Check Crystal Frequency**
   ```c
   // If using 11.0592MHz instead of 12MHz:
   // Machine Cycle = 1.085µs (not 1µs)
   // Recalculate reload values
   ```

2. **Verify Timer Mode**
   ```c
   // Ensure Mode 1 (16-bit)
   if((TMOD & 0x0F) != 0x01) {
       TMOD = (TMOD & 0xF0) | 0x01;
   }
   ```

3. **Check Interrupt Priority**
   ```c
   // Higher priority interrupts may delay Timer 0 ISR
   PT0 = 1;  // Set Timer 0 to high priority if needed
   ```

#### Problem 2: Timer Not Overflowing

**Symptoms:**
- Program hangs in delay
- LED doesn't toggle

**Solutions:**
1. **Timer Not Started**
   ```c
   TR0 = 1;  // Make sure timer is running!
   ```

2. **Interrupt Not Enabled**
   ```c
   ET0 = 1;  // Enable Timer 0 interrupt
   EA = 1;   // Enable global interrupts
   ```

3. **Wrong ISR Vector**
   ```c
   void timer0_isr(void) interrupt 1  // Must be interrupt 1!
   ```

#### Problem 3: Wrong Reload Values

**Symptoms:**
- Timing is completely off
- Delay much shorter/longer than expected

**Solutions:**
1. **Recalculate for Your Crystal**
   ```c
   // At 11.0592MHz:
   // Delay = 50ms = 50,000µs
   // Counts = 50,000 / 1.085 = 46,082
   // Reload = 65,536 - 46,082 = 19,454 = 0x4BFE
   TH0 = 0x4B;
   TL0 = 0xFE;
   ```

2. **Use Online Calculator**
   ```
   https://www.ee-diary.com/8051-calculator/
   ```

#### Problem 4: ISR Not Executing

**Symptoms:**
- LED stays steady
- Program seems stuck

**Debug Steps:**
1. Add LED toggle in main loop:
   ```c
   while(1) {
       LED = 0;
       simple_delay_ms(100);
       LED = 1;
       simple_delay_ms(100);
   }
   ```

2. If LED blinks, main works → Check interrupt setup

3. Add test in ISR:
   ```c
   void timer0_isr(void) interrupt 1
   {
       static unsigned char test = 0;
       test = 1;  // Set breakpoint here in debugger
       TH0 = 0x3C;
       TL0 = 0xB0;
   }
   ```

---

### 📊 Comparison: Timer Modes

| Feature | Mode 0 | Mode 1 | Mode 2 | Mode 3 |
|---------|--------|--------|--------|--------|
| **Bit Width** | 13-bit | 16-bit | 8-bit | 2×8-bit |
| **Max Count** | 8192 | 65536 | 256 | 256 each |
| **Auto Reload** | No | No | Yes | Yes (TH only) |
| **Use Case** | Legacy | ⭐ General | ⭐ Baud Rate | Special |
| **Accuracy** | Low | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

### 💡 Tips for Success

1. **Always Use Interrupts**
   - More accurate
   - CPU can do other work
   - Better for real-time systems

2. **Keep ISR Short**
   - Just reload timer and set flag
   - Do processing in main loop

3. **Use volatile for Shared Variables**
   ```c
   volatile unsigned char flag;  // Must be volatile!
   ```

4. **Recalculate for Your Crystal**
   - Don't assume 12MHz
   - Verify your board's actual frequency

5. **Test with LED First**
   - Visual feedback helps debugging
   - Verify timing before trusting it

6. **Use Mode 1 for Most Applications**
   - 16-bit gives good range
   - Simple to understand
   - Widely compatible

---

### 🚀 Next Steps

After mastering this example:

1. **Modify the Code:**
   - Change delay values
   - Add second timer (Timer 1)
   - Create variable delays

2. **Learn Mode 2:**
   - Auto-reload timers
   - Serial baud rate generation
   - See: [Serial Port](../04_Serial_Port/)

3. **Advanced Timers:**
   - PWM generation
   - Input capture
   - Watchdog timer
   - See: [PWM Examples](../05_Advanced/01_PWM/)

4. **Real Applications:**
   - Real-time clock
   - Periodic task scheduling
   - Pulse generation
   - Frequency measurement

---

## Example 2: Two Timers Simultaneously

### Application: Dual Delays

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Use both Timer 0 and Timer 1 independently for different tasks.

### Source Code

```c
// Two Independent Timers
// Timer 0: 50ms periodic interrupt (LED toggle)
// Timer 1: 100ms periodic interrupt (beep)

#include <reg51.h>

sbit LED = P1^0;
sbit BEEP = P1^1;

volatile unsigned char timer0_flag = 0;
volatile unsigned char timer1_flag = 0;

// Timer 0 ISR - 50ms
void timer0_isr(void) interrupt 1
{
    // Reload for 50ms @ 12MHz
    TH0 = 0x3C;
    TL0 = 0xB0;

    LED = ~LED;  // Toggle LED
    timer0_flag = 1;
}

// Timer 1 ISR - 100ms
void timer1_isr(void) interrupt 3
{
    // Reload for 100ms @ 12MHz
    TH1 = 0x76;
    TL1 = 0x60;

    BEEP = ~BEEP;  // Toggle beep
    timer1_flag = 1;
}

void init_timers(void)
{
    // Timer 0: Mode 1, 50ms
    TMOD = 0x11;      // Both timers in Mode 1
    TH0 = 0x3C;
    TL0 = 0xB0;

    // Timer 1: Mode 1, 100ms
    TH1 = 0x76;
    TL1 = 0x60;

    // Enable interrupts
    ET0 = 1;
    ET1 = 1;
    EA = 1;

    // Start timers
    TR0 = 1;
    TR1 = 1;
}

void main(void)
{
    init_timers();

    while(1) {
        if(timer0_flag) {
            timer0_flag = 0;
            // Timer 0 event handler
        }

        if(timer1_flag) {
            timer1_flag = 0;
            // Timer 1 event handler
        }
    }
}
```

---

## 📚 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Machine Cycle** | 12 oscillator periods |
| **Timer Mode 1** | 16-bit timer, manual reload |
| **Timer Mode 2** | 8-bit timer, auto reload |
| **Overflow** | Timer rolls from FFFFH to 0000H |
| **ISR Vector** | Timer 0 = INT1, Timer 1 = INT3 |
| **TF Flag** | Set by hardware on overflow |
| **TR Bit** | Start/stop timer |
| **Volatile** | Required for variables modified in ISR |

---

## 🎓 Complete Learning Path

### Step 1: Understand the Basics
- [ ] Read hardware architecture docs
- [ ] Understand machine cycles
- [ ] Learn timer register structure

### Step 2: Simple Delays
- [ ] Implement polling delay
- [ ] Implement interrupt delay
- [ ] Measure accuracy

### Step 3: Advanced Usage
- [ ] Use both timers
- [ ] Implement counter mode
- [ ] Create real-time clock

### Step 4: Real Applications
- [ ] PWM generation
- [ ] Serial baud rate
- [ ] Event counting

---

## 🔧 Tools Needed

**Hardware:**
- 8051 development board
- LED + resistor (optional)
- Oscilloscope (optional, for verification)

**Software:**
- SDCC or Keil C51 compiler
- Programmer (USBasp, ISP)
- Serial terminal software

---

**Difficulty:** ⭐⭐ Intermediate
**Time to Complete:** 2-3 hours
**Hardware:** LED + resistor (optional)
**Prerequisites:** [Basic I/O](../01_Basic_IO/)

**Happy Timing!** ⏱️
