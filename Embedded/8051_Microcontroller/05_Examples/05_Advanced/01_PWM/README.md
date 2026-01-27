# PWM (Pulse Width Modulation)

## Overview

Pulse Width Modulation (PWM) is a technique for controlling analog circuits with digital outputs. It's widely used for:
- LED brightness control
- Motor speed control
- Servo position control
- Power supply regulation
- Audio signal generation

## PWM Fundamentals

### What is PWM?

PWM varies the duty cycle of a digital signal to simulate an analog voltage:

```
         Duty Cycle = 50%           Duty Cycle = 25%
    ┌───┐       ┌───┐           ┌─┐    ┌─┐
    │   │       │   │           │ │    │ │
───┘   └───────┘   └───────    ──┘ └────┘ └────
    ON  OFF   ON  OFF          ON OFF ON OFF
    ↑                        ↑
    │  Period = 10ms         │ Period = 10ms
    │  ON time = 5ms         │ ON time = 2.5ms
    │  Duty cycle = 50%      │ Duty cycle = 25%
    ↓                        ↓
Average voltage = 2.5V      Average voltage = 1.25V
(assuming 5V VCC)
```

### Key Parameters

- **Frequency**: How fast the PWM cycles (typically 1kHz - 20kHz)
- **Duty Cycle**: Percentage of time signal is ON (0% - 100%)
- **Resolution**: Number of discrete duty cycle levels (8-bit = 256 levels)

## 8051 PWM Implementation

The standard 8051 doesn't have dedicated PWM hardware, but we can:
1. **Use Timer 2** (8052 variant) - Built-in PWM mode
2. **Software PWM** - Use Timer 0/1 interrupts
3. **External PWM** - Use dedicated PWM controller

---

## Example 1: Software PWM using Timer 0

### Application: LED Brightness Control

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Generate 8-bit PWM signal using Timer 0 interrupts for smooth LED brightness control.

```c
// Software PWM for LED Brightness Control
// Target: 8051 @ 12MHz
// PWM Frequency: ~122 Hz (8-bit resolution)
// Output: P1.0

#include <reg51.h>

sbit PWM_OUT = P1^0;

unsigned char pwm_duty = 128;  // 50% duty cycle (0-255)
unsigned char pwm_counter = 0;

// Timer 0 Interrupt - Generates PWM waveform
void timer0_isr(void) interrupt 1
{
    // Reload timer for next interrupt
    TH0 = 0xFF;  // Reload immediately (fast PWM)
    TL0 = 0x00;

    // Increment counter
    pwm_counter++;

    // PWM logic
    if(pwm_counter <= pwm_duty) {
        PWM_OUT = 0;  // LED ON (active low)
    } else {
        PWM_OUT = 1;  // LED OFF
    }

    // Reset counter at 256
    if(pwm_counter == 0) {  // Overflow from 255 to 0
        pwm_counter = 0;
    }
}

// Initialize Timer 0 for fast PWM
void timer0_init(void)
{
    TMOD &= 0xF0;      // Clear Timer 0 mode bits
    TMOD |= 0x01;      // Timer 0, Mode 1 (16-bit)

    // Initialize timer values
    TH0 = 0xFF;
    TL0 = 0x00;

    ET0 = 1;           // Enable Timer 0 interrupt
    EA = 1;            // Global interrupt enable
    TR0 = 1;           // Start Timer 0
}

void main(void)
{
    timer0_init();

    // Demonstration: Fade in/out
    unsigned char i;
    while(1) {
        // Fade in (0 to 100%)
        for(i = 0; i < 255; i++) {
            pwm_duty = i;
            // Delay to see fading effect
            // Use delay function or timer
        }

        // Fade out (100% to 0%)
        for(i = 255; i > 0; i--) {
            pwm_duty = i;
        }
    }
}
```

**PWM Frequency Calculation:**
```
For 8-bit PWM (256 steps):
Interrupt every timer overflow
At 12MHz with reload value:
PWM Frequency = Fosc / (12 × 256) = 12MHz / 3072 ≈ 3.9 kHz
```

---

## Example 2: Hardware PWM using Timer 2 (8052)

### Application: Precision Motor Speed Control

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Use Timer 2 in PWM mode for hardware-generated PWM signal.

```c
// Hardware PWM using Timer 2 (8052 only)
// Output: P1.7 (T2EX)

#include <reg52.h>

// Timer 2 PWM configuration
void timer2_pwm_init(void)
{
    T2MOD = 0x00;      // Timer 2 as PWM
    T2CON = 0x00;      // Configure Timer 2

    // Set PWM frequency and duty cycle
    RCAP2H = 0xFF;     // Reload value high byte
    RCAP2L = 0x00;     // Reload value low byte

    TR2 = 1;           // Start Timer 2
}

void set_pwm_duty(unsigned char duty)
{
    // Duty cycle 0-255
    TH2 = duty;
}
```

---

## Example 3: Servo Motor Control

### Application: Hobby Servo (SG90/SG5010)

**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Generate 50Hz PWM for servo motor position control.

**Servo Timing:**
- **Frequency:** 50Hz (20ms period)
- **Pulse Width:** 1ms to 2ms
  - 1ms = 0° (left)
  - 1.5ms = 90° (center)
  - 2ms = 180° (right)

```c
// Servo Control using Timer 0
// Servo on P1.0
// 50Hz PWM, 1-2ms pulse width

#include <reg51.h>

sbit SERVO = P1^0;

unsigned int servo_pulse = 1500;  // 1500µs = center position

// Timer 0 interrupt - Generates servo PWM
void timer0_isr(void) interrupt 1
{
    static unsigned int pwm_counter = 0;

    // Reload for 1µs ticks (adjust for your crystal)
    TH0 = 0xFF;
    TL0 = 0x00;  // Adjust based on crystal frequency

    pwm_counter++;

    // 20ms period (20000µs)
    if(pwm_counter <= 20000) {
        // Generate pulse
        if(pwm_counter <= servo_pulse) {
            SERVO = 1;  // Pulse ON
        } else {
            SERVO = 0;  // Pulse OFF
        }
    } else {
        pwm_counter = 0;  // Reset for next period
    }
}

void set_servo_angle(unsigned char angle)
{
    // Convert angle (0-180) to pulse width (1000-2000µs)
    servo_pulse = 1000 + ((unsigned int)angle * 1000 / 180);
}

void main(void)
{
    // Initialize timer
    TMOD = 0x01;      // Timer 0, Mode 1
    TH0 = 0xFF;
    TL0 = 0x00;
    ET0 = 1;
    EA = 1;
    TR0 = 1;

    // Sweep servo 0-180-0
    while(1) {
        for(unsigned char i = 0; i <= 180; i += 10) {
            set_servo_angle(i);
            // Delay (approximate)
            for(unsigned int j = 0; j < 10000; j++);
        }
        for(unsigned char i = 180; i > 0; i -= 10) {
            set_servo_angle(i);
            for(unsigned int j = 0; j < 10000; j++);
        }
    }
}
```

---

## Example 4: Multi-Channel PWM

### Application: RGB LED Color Control

**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Generate 3 independent PWM channels for RGB LED color mixing.

```c
// RGB LED PWM Control
// Red LED: P1.0, Green LED: P1.1, Blue LED: P1.2

#include <reg51.h>

sbit RED = P1^0;
sbit GREEN = P1^1;
sbit BLUE = P1^2;

unsigned char pwm_r = 0, pwm_g = 0, pwm_b = 0;
unsigned char pwm_counter = 0;

void timer0_isr(void) interrupt 1
{
    TH0 = 0xFF;
    TL0 = 0x00;

    pwm_counter++;

    RED = (pwm_counter <= pwm_r) ? 0 : 1;
    GREEN = (pwm_counter <= pwm_g) ? 0 : 1;
    BLUE = (pwm_counter <= pwm_b) ? 0 : 1;
}

void set_color(unsigned char r, unsigned char g, unsigned char b)
{
    pwm_r = r;
    pwm_g = g;
    pwm_b = b;
}

void main(void)
{
    // Initialize timer
    TMOD = 0x01;
    TH0 = 0xFF;
    TL0 = 0x00;
    ET0 = 1;
    EA = 1;
    TR0 = 1;

    // Color cycle
    while(1) {
        // Red
        set_color(255, 0, 0);
        // Delay...

        // Green
        set_color(0, 255, 0);
        // Delay...

        // Blue
        set_color(0, 0, 255);
        // Delay...

        // White
        set_color(255, 255, 255);
        // Delay...

        // Purple
        set_color(255, 0, 255);
        // Delay...
    }
}
```

---

## PWM Calculations

### Frequency vs Resolution Trade-off

```
PWM Frequency = Fosc / (12 × 2^N × (Prescaler))

Where:
- Fosc = Crystal frequency (e.g., 12MHz)
- N = Resolution in bits
- Prescaler = Timer prescaler value

Example @ 12MHz, 8-bit resolution:
Frequency = 12MHz / (12 × 256) = 3.9 kHz
```

### Optimal Frequencies

| Application | Frequency Range | Reason |
|-------------|----------------|--------|
| LED Brightness | 100 Hz - 1 kHz | Above flicker fusion |
| Motor Speed | 10 kHz - 20 kHz | Above audible range |
| Servo Control | 50 Hz (fixed) | Standard requirement |
| Power Supply | 20 kHz - 100 kHz | Reduce filter size |
| Audio Output | 32 kHz - 44 kHz | Audio quality |

---

## Hardware Considerations

### 1. LED PWM
- **Minimum Frequency:** 100 Hz (avoid visible flicker)
- **Recommended:** 200 Hz - 1 kHz
- **Resolution:** 8-bit (256 levels) sufficient

### 2. Motor PWM
- **Minimum Frequency:** 1 kHz
- **Recommended:** 10 kHz - 20 kHz (above hearing range)
- **Note:** Higher frequency = less torque ripple but more switching loss

### 3. Servo PWM
- **Frequency:** 50 Hz exactly (20ms period)
- **Pulse Width:** 1000-2000µs
- **Resolution:** 1µs = 0.09° (1000 steps for 90° range)

---

## Common Issues

### 1. PWM Not Working
**Symptoms:** Output always HIGH or LOW

**Solutions:**
- Check timer initialization
- Verify interrupt is enabled (ET0 = 1, EA = 1)
- Confirm timer is running (TR0 = 1)
- Check duty cycle variable is being updated

### 2. Wrong Frequency
**Symptoms:** PWM frequency incorrect

**Solutions:**
- Recalculate reload values for your crystal
- Verify timer mode (8-bit vs 16-bit)
- Check if timer overflow is being handled

### 3. Flickering LED
**Symptoms:** Visible flicker at low brightness

**Solutions:**
- Increase PWM frequency (should be >100 Hz)
- Check for interrupt conflicts
- Reduce interrupt processing time

### 4. Motor Jitter
**Symptoms:** Motor vibrates but doesn't rotate smoothly

**Solutions:**
- Increase PWM frequency
- Improve power supply decoupling
- Use flyback diode for inductive loads
- Check for electrical noise

---

## Advanced Techniques

### 1. Dead Time Insertion
For H-bridge motor control to prevent shoot-through:

```c
// Add delay between switching outputs
if(direction_change) {
    PWM_OUT1 = 0;
    PWM_OUT2 = 0;
    delay_us(10);  // Dead time
    // Now switch safely
}
```

### 2. Synchronous PWM
Multiple channels synchronized to same timer:

```c
// All channels update simultaneously
void update_all_pwm(unsigned char ch1, unsigned char ch2, unsigned char ch3)
{
    // Disable interrupts briefly
    EA = 0;

    duty1 = ch1;
    duty2 = ch2;
    duty3 = ch3;

    EA = 1;  // Re-enable interrupts
}
```

### 3. Phase-Correct PWM
Symmetrical PWM for reduced harmonic distortion:

```c
// Count up then down (triangle wave)
if(up_direction) {
    counter++;
    if(counter >= 255) up_direction = 0;
} else {
    counter--;
}
```

---

## Real-World Applications

### 1. LED Dimmer
- Room lighting control
- RGB mood lighting
- Display backlight control
- Indicator brightness

### 2. Motor Control
- DC motor speed regulation
- Fan speed control
- Robot propulsion
- Conveyor belt speed

### 3. Servo Control
- Robotic arms
- Camera gimbals
- Model aircraft control surfaces
- Antenna positioning

### 4. Power Conversion
- DC-DC converters
- Battery chargers
- Solar charge controllers
- Class-D audio amplifiers

### 5. Signal Generation
- Tone generation
- Synthesizer
- Function generator
- Modulation

---

## Components Needed

### For LED PWM
- LED (any color)
- Resistor (220Ω-470Ω)
- 8051 microcontroller

### For Motor PWM
- DC motor
- H-bridge (L293D, L298N)
- Flyback diode (1N4007)
- Power supply for motor

### For Servo PWM
- Hobby servo (SG90, MG996R)
- External 5V supply (servos draw more current)
- Capacitor (100µF across servo power)

---

## Prerequisites

- ✅ [Basic I/O](../../01_Basic_IO/)
- ✅ [Timers](../../02_Timers/)
- ✅ [Interrupts](../../03_Interrupts/)

**Recommended Reading:**
- [Timer Mode 2](../../02_Timers/)
- [Timer Interrupts](../../03_Interrupts/)
- [Motor Control](../02_Motor_Control/)

---

## Next Steps

After mastering PWM:
- [Motor Control](../02_Motor_Control/) - Apply PWM to motors
- [Display](../05_Display/) - LED dimming, backlight control
- [Projects](../../../06_Projects/) - Complete PWM-based projects

---

**Difficulty:** ⭐⭐⭐ Advanced
**Time to Master:** 4-6 hours
**Hardware:** LED + 8051 (basic), Motor + H-Bridge (advanced)
