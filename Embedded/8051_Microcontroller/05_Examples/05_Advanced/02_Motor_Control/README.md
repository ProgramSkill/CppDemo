# Motor Control

## Overview

This section covers motor control techniques using the 8051 microcontroller. You'll learn to control various types of motors for robotics, automation, and embedded applications.

## Motor Types

| Motor Type | Control Method | Complexity | Applications |
|------------|----------------|------------|--------------|
| **DC Motor** | PWM (speed) + H-Bridge (direction) | ⭐⭐⭐ | Robotics, fans, conveyors |
| **Stepper Motor** | Step sequence | ⭐⭐⭐⭐ | CNC, 3D printers, precision positioning |
| **Servo Motor** | PWM position control | ⭐⭐⭐ | Robotics arms, camera gimbals |
| **Brushless DC (BLDC)** | Electronic commutation | ⭐⭐⭐⭐⭐ | Drones, high-performance tools |

---

## Example 1: DC Motor Speed Control

### Application: Variable Speed Fan

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Control DC motor speed using PWM and H-bridge for bidirectional control.

### Hardware Setup

```
         8051                   L293D H-Bridge
       ┌──────┐              ┌──────────────────┐
       │      │              │  Enable 1,2 ─────┼──→ PWM_OUT (P1.0)
       │  P1.0├──────────────┤                  │
       │  P1.1├──────────────┤  Input 1    ─────┼──→ DIR_A
       │  P1.2├──────────────┤  Input 2    ─────┼──→ DIR_B
       │      │              │                  │
       │      │         ┌────┤  Output 1   ─────┼─┐
       │      │         │    └──────────────────┘  │
       │      │         │                           │
       │      │         │    ┌──────────────────┐  │
       │      │         │    │  L293D           │  │
       │      │         │    │  Output 2   ─────┼──┘
       │      │         │    └──────────────────┘
       │      │         │
       │      │         │    ┌────────────────┐
       │      │         └────┤  DC MOTOR      │
       │      │              │  +  ─       │  │
       │  GND ├──────────────┤  -          │  │
       │      │              └────────────────┘
       └──────┘

Power: External 5V-12V for motor
```

### Component List
- DC motor (6V-12V)
- L293D or L298N H-bridge driver
- Flyback diodes (1N4007) - usually built into L293D
- Capacitor (100µF) across motor power
- External power supply for motor

### Source Code

```c
// DC Motor Control with PWM
// Uses Timer 0 for PWM generation
// Motor on L293D H-Bridge

#include <reg51.h>

// Control pins
sbit PWM_PIN = P1^0;   // PWM output (speed)
sbit DIR_A = P1^1;     // Direction control A
sbit DIR_B = P1^2;     // Direction control B

// PWM variables
unsigned char motor_speed = 0;  // 0-255
unsigned char pwm_counter = 0;

// Timer 0 interrupt for PWM generation
void timer0_isr(void) interrupt 1
{
    // Reload timer (adjust based on desired frequency)
    TH0 = 0xFF;
    TL0 = 0x00;

    pwm_counter++;

    // Generate PWM waveform
    if(pwm_counter <= motor_speed) {
        PWM_PIN = 1;  // PWM ON
    } else {
        PWM_PIN = 0;  // PWM OFF
    }
}

// Initialize Timer 0 for PWM
void pwm_init(void)
{
    TMOD = 0x01;      // Timer 0, Mode 1 (16-bit)
    TH0 = 0xFF;
    TL0 = 0x00;

    ET0 = 1;          // Enable Timer 0 interrupt
    EA = 1;           // Global interrupt enable
    TR0 = 1;          // Start Timer 0
}

// Set motor direction
// direction: 0 = forward, 1 = reverse
void set_direction(unsigned char direction)
{
    if(direction == 0) {
        DIR_A = 1;
        DIR_B = 0;    // Forward
    } else {
        DIR_A = 0;
        DIR_B = 1;    // Reverse
    }
}

// Set motor speed (0-255)
void set_speed(unsigned char speed)
{
    motor_speed = speed;
}

// Stop motor (brake)
void motor_stop(void)
{
    DIR_A = 0;
    DIR_B = 0;
    motor_speed = 0;
}

// Coast to stop (freewheel)
void motor_coast(void)
{
    motor_speed = 0;
}

void main(void)
{
    pwm_init();

    // Demo: Accelerate forward
    set_direction(0);  // Forward
    for(unsigned char i = 0; i < 255; i += 5) {
        set_speed(i);
        // Delay loop (approximate)
        for(unsigned int j = 0; j < 10000; j++);
    }

    // Run at full speed for 2 seconds
    for(unsigned int j = 0; j < 200; j++)
        for(unsigned int k = 0; k < 10000; k++);

    // Decelerate to stop
    for(unsigned char i = 255; i > 0; i -= 5) {
        set_speed(i);
        for(unsigned int j = 0; j < 10000; j++);
    }

    motor_stop();

    // Demo: Reverse
    set_direction(1);  // Reverse
    set_speed(128);    // Half speed

    while(1);  // Keep running
}
```

### Control Modes

**Mode 1: Speed Control Only**
```c
set_speed(200);  // 78% speed
```

**Mode 2: Direction + Speed**
```c
set_direction(0);  // Forward
set_speed(180);    // 70% speed
```

**Mode 3: Brake (Quick Stop)**
```c
motor_stop();  // Both outputs LOW = brake
```

**Mode 4: Coast (Freewheel)**
```c
motor_coast();  // PWM = 0% = freewheel
```

---

## Example 2: Stepper Motor Control

### Application: Precision Positioning

**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Control bipolar stepper motor using ULN2003 driver.

### Hardware Setup

```
         8051                ULN2003 Driver
       ┌──────┐            ┌──────────────────┐
       │      │            │  IN1 ───────────┼─→ Coil 1
       │  P1.0├────────────┤  IN2 ───────────┼─→ Coil 2
       │  P1.1├────────────┤  IN3 ───────────┼─→ Coil 3
       │  P1.2├────────────┤  IN4 ───────────┼─→ Coil 4
       │      │            │                  │
       │  GND ├────────────┤  GND             │
       │      │            └──────────────────┘
       │      │                      │
       │      │             ┌────────────────┐
       │      │             │  Stepper Motor │
       │      │             │  (28BYJ-48)    │
       │  +5V ├─────────────┤  +  -  Red      │
       └──────┘             └────────────────┘
```

### Stepper Motor Sequences

**Full Step (2 phases)**
```
Step  Coil1  Coil2  Coil3  Coil4
 1      1      0      1      0
 2      1      0      0      1
 3      0      1      0      1
 4      0      1      1      0
```

**Half Step (better resolution)**
```
Step  Coil1  Coil2  Coil3  Coil4
 1      1      0      0      0
 2      1      0      1      0
 3      0      0      1      0
 4      0      1      1      0
 5      0      1      0      0
 6      0      1      0      1
 7      0      0      0      1
 8      1      0      0      1
```

### Source Code

```c
// Stepper Motor Control
// 28BYJ-48 with ULN2003
// Full step sequence

#include <reg51.h>

// Motor control pins
sbit COIL1 = P1^0;
sbit COIL2 = P1^1;
sbit COIL3 = P1^2;
sbit COIL4 = P1^3;

// Step sequence for full step
unsigned char step_sequence[] = {0x09, 0x05, 0x06, 0x0A};
// Binary: 1001, 0101, 0110, 1010
//         Coil1+3, Coil1+2, Coil2+3, Coil1+4

// Simple delay
void delay_ms(unsigned int ms)
{
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

// Single step forward
void step_forward(void)
{
    static unsigned char step = 0;

    P1 = (P1 & 0xF0) | step_sequence[step];
    step = (step + 1) % 4;

    delay_ms(10);  // Speed control (lower = faster)
}

// Single step backward
void step_backward(void)
{
    static unsigned char step = 3;

    P1 = (P1 & 0xF0) | step_sequence[step];
    step = (step - 1) % 4;

    delay_ms(10);
}

// Rotate specified number of steps
void rotate_steps(unsigned int steps, unsigned char direction)
{
    unsigned int i;
    for(i = 0; i < steps; i++) {
        if(direction == 0) {
            step_forward();
        } else {
            step_backward();
        }
    }
}

// Rotate continuously (blocking)
void rotate_continuous(unsigned char direction)
{
    while(1) {
        if(direction == 0) {
            step_forward();
        } else {
            step_backward();
        }
    }
}

void main(void)
{
    // Initialize port
    P1 = 0xF0;  // Motor pins low

    // Demo: 360° rotation (2048 steps for 28BYJ-48)
    rotate_steps(512, 0);    // Forward 90°
    delay_ms(500);

    rotate_steps(512, 1);    // Backward 90°
    delay_ms(500);

    // Or continuous rotation
    // rotate_continuous(0);  // Forward forever
}
```

---

## Motor Control Concepts

### 1. Speed Control (PWM)

**Why PWM?**
- Efficient (no linear power loss)
- Precise control
- Low heat generation
- Wide speed range

**PWM Frequency for Motors:**
- Minimum: 1 kHz
- Recommended: 10-20 kHz
- Too high: Switching losses
- Too low: Audible whine, vibration

### 2. Direction Control (H-Bridge)

**H-Bridge Truth Table:**

| IN1 | IN2 | Motor State |
|-----|-----|-------------|
| 0   | 0   | Brake/Stop  |
| 0   | 1   | Reverse     |
| 1   | 0   | Forward     |
| 1   | 1   | Brake       |

### 3. Acceleration/Deceleration

**Ramp Function:**
```c
void smooth_start(unsigned char target_speed)
{
    for(unsigned char i = 0; i < target_speed; i++) {
        set_speed(i);
        delay_ms(50);  // Ramp rate
    }
}
```

### 4. Current Limiting

**Why Important:**
- Prevent motor burnout
- Protect driver IC
- Reduce power supply stress

**Methods:**
- Current sensing resistor
- Back EMF measurement
- Timeout protection

---

## Safety Considerations

### 1. Flyback Diodes
Always use flyback diodes with inductive loads:

```
Motor ────▶│──────┐
           │      │
         ──┴──    GND
```

Most H-bridge ICs (L293D, L298N) include built-in diodes.

### 2. Power Supply Decoupling
```c
Place 100µF capacitor across motor power terminals:
VCC ───||────→ Motor
     100µF     │
GND ───────────┴──→ Motor
```

### 3. Separate Power Supplies
- Logic supply: 5V for 8051
- Motor supply: Higher voltage for motor
- Common ground connection

### 4. Current Ratings
- L293D: 600mA per channel
- L298N: 2A per channel
- 8051 GPIO: 20mA max (use driver!)

---

## Troubleshooting

### Motor Not Turning
**Check:**
1. Power supply connected
2. PWM signal present (scope/logic analyzer)
3. Direction signals correct
4. Enable pin active (if used)
5. Motor not stalled/mechanically blocked

### Motor Runs Wrong Direction
**Solution:**
- Swap DIR_A and DIR_B in software
- Or swap motor leads

### Motor Jitters
**Causes:**
- PWM frequency too low
- Missing flyback diode
- Power supply noise
- Insufficient decoupling

### Erratic Behavior
**Check:**
1. Ground connections
2. Shared power supply noise
3. Interrupt conflicts
4. Timer configuration

---

## Advanced Topics

### 1. Closed-Loop Speed Control

```c
// PID-like speed control
unsigned char target_rpm = 100;
unsigned int current_rpm = 0;
unsigned char motor_output = 0;

void speed_control_loop(void)
{
    int error = target_rpm - current_rpm;

    // Simple P control
    motor_output += error / 10;

    if(motor_output > 255) motor_output = 255;
    if(motor_output < 0) motor_output = 0;

    set_speed(motor_output);
}
```

### 2. Position Control (Encoder Feedback)

```c
// Quadrature encoder reading
volatile int encoder_count = 0;

void read_encoder(void)
{
    static unsigned char last_state = 0;
    unsigned char current_state = (ENC_A << 1) | ENC_B;

    if(last_state != current_state) {
        // Decode direction and increment/decrement
        encoder_count += encoder_table[last_state][current_state];
        last_state = current_state;
    }
}
```

### 3. Multi-Motor Coordination

```c
// Differential drive robot
void set_robot_velocity(int linear, int angular)
{
    int left_speed = linear - angular;
    int right_speed = linear + angular;

    set_motor_speed(MOTOR_LEFT, left_speed);
    set_motor_speed(MOTOR_RIGHT, right_speed);
}
```

---

## Real-World Applications

### 1. Robotics
- Differential drive robots
- Robotic arm joints
- Gripper control
- Omni-directional robots

### 2. Automation
- Conveyor belts
- Linear actuators
- Valve control
- Camera pan/tilt

### 3. Precision Motion
- CNC machines
- 3D printers
- Plotters
- Camera sliders

### 4. Consumer Products
- Electric fans
- Power tools
- Toys
- Appliances

---

## Component Selection

### DC Motors

| Type | Voltage | Current | Use Case |
|------|---------|---------|----------|
| Small | 3V-6V | 100mA | Toys, small robots |
| Medium | 6V-12V | 500mA | Fans, conveyors |
| Large | 12V-24V | 2A+ | Heavy-duty applications |

### Motor Drivers

| Driver | Current | Voltage | Features |
|--------|---------|---------|----------|
| L293D | 600mA | 4.5V-36V | Dual H-bridge, built-in diodes |
| L298N | 2A | 5V-35V | Dual H-bridge, heat sink |
| TB6612 | 1.2A | 2.5V-13.5V | Compact, efficient |
| DRV8871 | 3.6A | 6.5V-45V | Single channel, high current |

### Stepper Drivers

| Driver | Type | Voltage | Current |
|--------|------|---------|---------|
| ULN2003 | Darlington | 5V-50V | 500mA |
| A4988 | Chopper | 8V-35V | 2A |
| DRV8825 | Chopper | 8.2V-45V | 1.5A |

---

## Prerequisites

- ✅ [Basic I/O](../../01_Basic_IO/)
- ✅ [Timers](../../02_Timers/)
- ✅ [Interrupts](../../03_Interrupts/)
- ✅ [PWM](../01_PWM/)

**Recommended Reading:**
- [PWM Generation](../01_PWM/)
- [Timer Interrupts](../../03_Interrupts/)

---

## Next Steps

After mastering motor control:
- [Sensors](../04_Sensors/) - Add feedback
- [Projects](../../../06_Projects/) - Build complete robot

---

**Difficulty:** ⭐⭐⭐⭐ Expert
**Time to Master:** 8-12 hours
**Hardware:** Motor + Driver + Power Supply
