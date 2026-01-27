# Example 4: Relay Control

## Overview

Relays are electrically operated switches that allow low-voltage microcontrollers (like 8051) to control high-voltage/high-power devices. They provide electrical isolation between the control circuit and the load circuit, making them essential for controlling AC mains, motors, heaters, lamps, and other high-power devices.

---

## ⚠️ Safety Warning

**WARNING: This example involves controlling high-voltage circuits (110V/220V AC).**

**Before proceeding:**
- ⚡ High voltage can cause serious injury or death
- 🔌 Always work with DC loads (batteries) when learning
- 💡 Never touch live AC circuits
- 🔧 Use proper insulation and enclosures
- 📖 Follow local electrical codes
- 🧪 Test with low-voltage DC first (5V-12V)

**Beginner recommendation:** Start with a 5V relay module controlling a 12V LED strip or small DC motor.

---

## 📝 Complete Source Code

### Version 1: Basic Relay ON/OFF Control

**File:** `Relay_Basic_Control.c`

```c
// Relay Basic Control Example
// Simple ON/OFF control of a relay
// Target: 8051 @ 12MHz
// Hardware: Relay module on P1.0 with transistor driver

#include <reg51.h>

// Relay control pin
sbit RELAY = P1^0;  // Relay connected to P1.0

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
    RELAY = 1;  // Start with relay OFF (active low relay module)
    while(1) {  // Infinite loop
        RELAY = 0;      // Turn ON relay (active low)
        delay(2000);    // Keep ON for 2 seconds

        RELAY = 1;      // Turn OFF relay
        delay(2000);    // Keep OFF for 2 seconds
    }
}
```

---

### Version 2: Relay with Button Control

**File:** `Relay_Button_Control.c`

```c
// Relay Control with Button
// Toggle relay with push button
// Target: 8051 @ 12MHz
// Hardware: Relay on P1.0, Button on P3.2

#include <reg51.h>

// Pin declarations
sbit RELAY = P1^0;     // Relay control pin
sbit BUTTON = P3^2;    // Button input pin

/**
 * @brief  Simple delay function
 */
void delay(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    RELAY = 1;  // Start with relay OFF
    while(1) {
        if(BUTTON == 0) {        // Button pressed (active low)
            RELAY = ~RELAY;      // Toggle relay state
            delay(200);          // Debounce delay
            while(BUTTON == 0);  // Wait for release
        }
    }
}
```

---

### Version 3: Multiple Relay Control

**File:** `Relay_Multiple_Control.c`

```c
// Multiple Relay Control Example
// Control 4 relays independently
// Target: 8051 @ 12MHz
// Hardware: 4 relays on P1.0-P1.3

#include <reg51.h>

// Relay control pins
sbit RELAY1 = P1^0;
sbit RELAY2 = P1^1;
sbit RELAY3 = P1^2;
sbit RELAY4 = P1^3;

/**
 * @brief  Simple delay function
 */
void delay(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    // Initialize all relays OFF
    RELAY1 = 1; RELAY2 = 1; RELAY3 = 1; RELAY4 = 1;

    while(1) {
        // Sequence 1: Turn on one at a time
        RELAY1 = 0; delay(1000); RELAY1 = 1;
        RELAY2 = 0; delay(1000); RELAY2 = 1;
        RELAY3 = 0; delay(1000); RELAY3 = 1;
        RELAY4 = 0; delay(1000); RELAY4 = 1;
        delay(500);

        // Sequence 2: Turn on all, then off
        RELAY1 = 0; RELAY2 = 0; RELAY3 = 0; RELAY4 = 0;
        delay(2000);

        RELAY1 = 1; RELAY2 = 1; RELAY3 = 1; RELAY4 = 1;
        delay(2000);

        // Sequence 3: Alternating pattern
        RELAY1 = 0; RELAY3 = 0;
        delay(1000);
        RELAY1 = 1; RELAY3 = 1;
        RELAY2 = 0; RELAY4 = 0;
        delay(1000);
        RELAY2 = 1; RELAY4 = 1;
    }
}
```

---

### Version 4: Relay with Timer Delay

**File:** `Relay_Timer_Control.c`

```c
// Relay with Automatic Timer Control
// Turn on relay for specified duration
// Target: 8051 @ 12MHz
// Hardware: Relay on P1.0, Button on P3.2

#include <reg51.h>

sbit RELAY = P1^0;
sbit BUTTON = P3^2;

unsigned int timer_count = 0;
bit relay_active = 0;

// Timer0 interrupt - generates 1ms ticks
void timer0_isr(void) interrupt 1 {
    TH0 = 0xFC;   // Reload for 1ms at 12MHz
    TL0 = 0x18;

    if(relay_active) {
        timer_count++;
        if(timer_count >= 5000) {  // 5 seconds
            RELAY = 1;             // Turn OFF relay
            relay_active = 0;
            timer_count = 0;
        }
    }
}

void timer0_init() {
    TMOD &= 0xF0;     // Clear Timer 0 mode bits
    TMOD |= 0x01;     // Timer 0, Mode 1 (16-bit)
    TH0 = 0xFC;       // Initial value for 1ms
    TL0 = 0x18;
    TR0 = 1;          // Start Timer 0
    ET0 = 1;          // Enable Timer 0 interrupt
    EA = 1;           // Global interrupt enable
}

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    RELAY = 1;  // Start with relay OFF
    timer0_init();

    while(1) {
        if(BUTTON == 0) {
            delay_ms(200);  // Debounce
            if(BUTTON == 0) {
                RELAY = 0;         // Turn ON relay
                relay_active = 1;  // Start timer
                timer_count = 0;
                while(BUTTON == 0);  // Wait for release
            }
        }
    }
}
```

---

## 🔌 Hardware Connection

### Relay Module with Optocoupler (Recommended)

```
         8051                    Relay Module
       ┌──────┐                 ┌─────────────┐
       │      │                 │             │
       │ 5V   ├─────────────────┤ VCC         │
       │      │                 │             │
       │ GND  ├─────────────────┤ GND         │
       │      │                 │             │
       │ P1.0 ├─────────────────┤ IN1         │
       │      │                 │             │
       └──────┘                 │   ┌───┐     │
                                 │   │ ● │     │
                    ┌────────────┤   │   │ COM │
                    │ Load       │   └─┬─┘     │
                    │ Power      │     │ NO    │
                    │ Supply     │     │ NC    │
                    │            └─────┴───────┘
                               AC/DC Load
                               (Lamp, Motor, etc.)

Module Pinout:
VCC  - 5V power supply
GND  - Ground
IN1  - Relay control signal
     - LOW = Relay ON
     - HIGH = Relay OFF
COM  - Common terminal
NO   - Normally Open (disconnected when OFF)
NC   - Normally Closed (connected when OFF)
```

### Discrete Relay Circuit with Transistor Driver

```
         8051                    Relay Driver Circuit
       ┌──────┐
       │      │                    5V Supply
       │      │                        │
       │ P1.0 ├──────1kΩ──────────────┤││
       │      │                      │   NPN
       │      │                      │  (2N2222
       │      │                      │   BC547
       │      │                      │   etc.)
       │      │                       └─┬─┘
       │      │                         │
       │      │                         ├───┐  Diode
       │      │                         │   │  (1N4007
       │      │                         │   │  Flyback)
       │      │                         └───┘
       │      │                          │ │
       │ GND  ├──────────────────────────┴─┴──────┐
       │      │                                     │
       └──────┘                    ┌──────┐         │
                                   │      │         │
                                   │ Relay├─────────┘
                                   │ Coil │
                                   └──┬───┘
                                      │
                    ┌─────────────────┴─────┐
                    │ AC/DC Power Supply     │
                    │ (for load only)        │
                    └────────────────────────┘

Components:
- 1× NPN Transistor (2N2222, BC547, S8050)
- 1× Diode (1N4007) - Flyback protection
- 1× Resistor 1kΩ (base current limiting)
- 1× Relay (5V coil, rated for load)
```

### ULN2003 Darlington Array Driver (For Multiple Relays)

```
         8051                    ULN2003 Driver
       ┌──────┐                 ┌─────────────────┐
       │      │                 │   ┌──────────┐  │
       │ P1.0 ├─────────────────┤ 1│IN1    OUT1│──┼──→ Relay 1
       │ P1.1 ├─────────────────┤ 2│IN2    OUT2│──┼──→ Relay 2
       │ P1.2 ├─────────────────┤ 3│IN3    OUT3│──┼──→ Relay 3
       │ P1.3 ├─────────────────┤ 4│IN4    OUT4│──┼──→ Relay 4
       │      │                 │   │           │  │
       │ 5V   ├─────────────────┤ 9│COM   ...  │  │
       │ GND  ├─────────────────┤ 8│GND       │  │
       │      │                 │   └──────────┘  │
       └──────┘                 └─────────────────┘

Advantages:
- Drives up to 7 relays with one chip
- Built-in flyback diodes
- High current capability (500mA per channel)
- Simple connection (no external components)
```

---

## 📖 Code Explanation

### 1. Active Low Logic

Most relay modules use **active low** logic:

```c
RELAY = 0;  // Relay ON
RELAY = 1;  // Relay OFF
```

**Why active low?**
- Optocoupler input LED needs to sink current
- Compatible with 8051's better sink capability
- Safer: Relay OFF if pin floats or program crashes

### 2. Flyback Diode (Protection)

**Problem:** When relay coil turns off, magnetic field collapses, generating high voltage spike:

```
Voltage:  ┐┐┐┐┐┐┐┐┐┐┐  (can reach hundreds of volts!)
          │
Relay OFF │
```

**Solution:** Diode across coil provides current path:

``
Diode placement:       Current flow during turn-off:
    ┌──┐                        ┌──┐
    │  │                        ↑  │
Coil └──┘                  Coil │  └──┘
     │  ←                      │  ←
    ─┴─ Diode                ─┴─ Diode

     Off: Off                      On:  On
```

**Diode orientation:** Cathode (stripe) to positive side

### 3. Base Resistor Calculation

For NPN transistor driving relay coil:

```
Given:
- Relay coil current: 70mA (typical for 5V relay)
- Transistor gain (hFE): 100 (minimum)
- 8051 output voltage: 5V
- VBE (base-emitter): 0.7V

Required base current:
IB = IC / hFE = 70mA / 100 = 0.7mA

Base resistor:
RB = (VOUTPUT - VBE) / IB
RB = (5V - 0.7V) / 0.7mA = 6.14kΩ

Use standard value: 1kΩ (provides good margin)
```

### 4. Optocoupler Isolation

**Purpose:** Electrically isolate 8051 from relay circuit

```
8051 Side          Load Side
─────────         ─────────
Low voltage       High voltage
Low current       High current
Clean ground      Noisy ground

Optocoupler creates:
- No electrical connection
- Optical coupling only
- Protection from voltage spikes
- Ground loop isolation
```

---

## 🔬 Relay Types and Selection

### By Coil Voltage

| Coil Voltage | Use Case | Power Source |
|--------------|----------|--------------|
| 5V | Direct 8051 control | 8051 5V supply |
| 9V | Battery systems | 9V battery |
| 12V | Automotive | Car battery |
| 24V | Industrial | Industrial PSU |

**Beginner choice:** 5V coil relay (works with 8051 supply)

### By Contact Configuration

```
SPST - Single Pole Single Throw:
      COM ──┐
            ├── NO (Normally Open)
      NC ───┘

SPDT - Single Pole Double Throw:
             NO
              ↑
      COM ────┼
              ↓
             NC

DPDT - Double Pole Double Throw:
      COM1 ───┼── NO1
      COM2 ───┼── NO2
      (Controls 2 independent circuits)
```

### By Contact Rating

**For resistive loads (lamps, heaters):**
```
Rated current × 0.7 = Safe current
Example: 10A relay → 7A safe continuous
```

**For inductive loads (motors, solenoids):**
```
Rated current × 0.4 = Safe current
Example: 10A relay → 4A safe continuous
```

**Why derate?**
- Inductive loads cause arcing when contacts open
- Reduces contact wear and welding risk

---

## 🛠️ Testing and Troubleshooting

### Test with LED First (Safe)

```c
void main() {
    while(1) {
        RELAY = 0;  // Should hear "click" and see LED ON
        delay(2000);
        RELAY = 1;  // Another "click", LED OFF
        delay(2000);
    }
}
```

**Expected:**
- Audible "click" when relay switches
- LED turns ON and OFF
- Measure 0V across coil when ON (active low)

### Common Issues and Solutions

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| Relay doesn't click | No power to coil | Check 5V and GND connections |
| Relay always ON | Wrong logic | Invert: RELAY = 1 for OFF |
| 8051 resets when relay switches | Voltage spike | Add flyback diode, check power supply |
| Relay chatters (rapid on/off) | Insufficient drive current | Use stronger transistor or ULN2003 |
| Contacts weld together | Load too high | Use higher current relay or solid-state relay |
| Relay gets hot | Coil voltage too high | Match coil voltage to supply |

### Multimeter Testing

**Test coil resistance:**
```
Multimeter: Resistance mode (Ω)
Measure across coil terminals
Expected: 50-500Ω (depending on relay)

If 0Ω: Coil shorted
If ∞: Coil open
```

**Test contact operation:**
```
1. Measure COM-NO: Should be ∞ (OFF state)
2. Energize coil
3. Measure COM-NO: Should be 0Ω (ON state)
4. De-energize coil
5. Measure COM-NO: Should return to ∞
```

---

## 📊 Variations and Extensions

### Variation 1: Temperature Fan Control

```c
// Turn on fan relay when temperature exceeds threshold
// Simulated with potentiometer on ADC channel

#include <reg51.h>

sbit RELAY_FAN = P1^0;
unsigned int temperature;

// Simulated ADC read (replace with real ADC)
unsigned char read_temperature() {
    // Returns temperature 0-100°C
    // In real application, read ADC and convert
    return 35;  // Example: 35°C
}

void main() {
    RELAY_FAN = 1;  // Fan OFF

    while(1) {
        temperature = read_temperature();

        if(temperature > 30) {
            RELAY_FAN = 0;  // Turn ON fan
        } else if(temperature < 25) {
            RELAY_FAN = 1;  // Turn OFF fan
        }

        delay(1000);  // Check every second
    }
}
```

### Variation 2: Light Timer (Automatic ON/OFF)

```c
// Turn on lamp at specific time, off after duration
// Simple simulation using counters

#include <reg51.h>

sbit RELAY_LAMP = P1^0;
unsigned int seconds = 0;
unsigned char hours = 0;

// Timer interrupt - increments seconds
void timer0_isr(void) interrupt 1 {
    // Timer setup code here...
    seconds++;
    if(seconds >= 3600) {
        seconds = 0;
        hours++;
        if(hours >= 24) hours = 0;
    }
}

void main() {
    RELAY_LAMP = 1;  // Start with lamp OFF

    while(1) {
        // Turn ON at 18:00 (6 PM)
        if(hours == 18 && seconds == 0) {
            RELAY_LAMP = 0;  // Lamp ON
        }

        // Turn OFF at 23:00 (11 PM)
        if(hours == 23 && seconds == 0) {
            RELAY_LAMP = 1;  // Lamp OFF
        }

        // Manual override with button could be added here
    }
}
```

### Variation 3: Motor Direction Control (2 Relays)

```c
// Control DC motor direction using 2 relays
// H-bridge configuration

#include <reg51.h>

sbit RELAY1 = P1^0;  // Forward control
sbit RELAY2 = P1^1;  // Reverse control
sbit BUTTON_FWD = P3^2;
sbit BUTTON_REV = P3^3;

void stop_motor() {
    RELAY1 = 1;
    RELAY2 = 1;
}

void motor_forward() {
    RELAY1 = 0;  // Active low
    RELAY2 = 1;
}

void motor_reverse() {
    RELAY1 = 1;
    RELAY2 = 0;  // Active low
}

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    stop_motor();

    while(1) {
        if(BUTTON_FWD == 0) {
            delay_ms(200);
            stop_motor();  // Stop before changing direction
            delay_ms(100);
            motor_forward();
            while(BUTTON_FWD == 0);
        }

        if(BUTTON_REV == 0) {
            delay_ms(200);
            stop_motor();  // Stop before changing direction
            delay_ms(100);
            motor_reverse();
            while(BUTTON_REV == 0);
        }
    }
}
```

### Variation 4: Sequential Relay Control

```c
// Control relays in sequence (conveyor belt example)

#include <reg51.h>

sbit RELAY1 = P1^0;  // Station 1
sbit RELAY2 = P1^1;  // Station 2
sbit RELAY3 = P1^2;  // Station 3
sbit RELAY4 = P1^3;  // Station 4
sbit START = P3^2;

void all_off() {
    RELAY1 = 1; RELAY2 = 1; RELAY3 = 1; RELAY4 = 1;
}

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    all_off();

    while(1) {
        if(START == 0) {
            delay_ms(200);
            if(START == 0) {
                // Start sequence
                RELAY1 = 0; delay_ms(1000);
                RELAY2 = 0; delay_ms(1000);
                RELAY3 = 0; delay_ms(1000);
                RELAY4 = 0; delay_ms(2000);

                // All on briefly
                delay_ms(1000);

                // Turn off in reverse
                RELAY4 = 1; delay_ms(500);
                RELAY3 = 1; delay_ms(500);
                RELAY2 = 1; delay_ms(500);
                RELAY1 = 1; delay_ms(500);

                while(START == 0);
            }
        }
    }
}
```

### Variation 5: Pulse Output (Door Lock Control)

```c
// Brief pulse to trigger door lock solenoid

#include <reg51.h>

sbit RELAY_LOCK = P1^0;
sbit BUTTON = P3^2;

void pulse_relay() {
    RELAY_LOCK = 0;  // Energize solenoid
    delay_ms(500);   // Keep energized for 0.5 seconds
    RELAY_LOCK = 1;  // De-energize
}

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    RELAY_LOCK = 1;  // Start OFF

    while(1) {
        if(BUTTON == 0) {
            delay_ms(200);
            if(BUTTON == 0) {
                pulse_relay();
                while(BUTTON == 0);
            }
        }
    }
}
```

---

## ⚡ Power Supply Design

### Separate Supplies for Control and Load

**Recommended for high-power loads:**

```
┌─────────────────┐     ┌──────────────────┐
│  Control Supply │     │   Load Supply    │
│     (5V)        │     │   (12V/24V/AC)   │
├─────────────────┤     ├──────────────────┤
│  8051           │     │                  │
│  Relay Coils    │     │   Load (Motor,   │
│  Optocouplers   │     │   Lamp, Heater)  │
└─────────────────┘     └──────────────────┘
        │                       │
        │        GND            │
        └───────────┬───────────┘
                    │ (Common ground at ONE point)
                   ─┴─
```

**Benefits:**
- Relay coil switching doesn't affect 8051
- Load noise isolated from control logic
- Better voltage regulation

### Decoupling Capacitors

```
8051 Power Supply:
    5V ──┬── 100µF ───┬── GND
         │            │
         ├── 0.1µF ───┤
         │            │
         └── 8051 ────┘

Relay Coil Supply:
    5V ──┬── 100µF ───┬── GND
         │            │
         ├── 0.1µF ───┤
         │            │
         └── Relay ───┘
```

**Purpose:**
- 100µF: Handles current surges when relay switches
- 0.1µF: Filters high-frequency noise

---

## 🔌 Load Types and Considerations

### Resistive Loads (Easy)

**Examples:** Incandescent lamps, heaters, soldering irons

```c
// Simple ON/OFF control
sbit RELAY = P1^0;
RELAY = 0;  // Lamp ON
```

**Considerations:**
- High inrush current (cold filament)
- Use relay rated for 2× load current

### Inductive Loads (Moderate)

**Examples:** Motors, solenoids, transformers

```c
// Motor control with flyback protection
// Note: Relay module should have snubber circuit
sbit RELAY_MOTOR = P1^0;
RELAY_MOTOR = 0;  // Motor ON
```

**Considerations:**
- Back EMF when turning OFF
- Use snubber circuit (RC network across contacts)
- Derate relay to 40% of rated current

### Capacitive Loads (Difficult)

**Examples:** LED drivers, electronic ballasts

**Considerations:**
- Huge inrush current when charging
- Best solution: Solid-state relay (SSR)
- Alternative: Pre-charge circuit

### AC Mains Loads (Dangerous!)

**⚠️ EXTREME CAUTION REQUIRED**

**If you must control AC:**
1. Use **opto-isolated relay module**
2. Use **relay rated for AC** (specific voltage)
3. Use **enclosed relay module** (no exposed contacts)
4. **Double-insulate** all connections
5. Use **AC-rated connector** (not terminal block)
6. **Never work on live circuit**
7. Test with **low-voltage AC first** (12V AC from transformer)

**Better alternative:** Use solid-state relay (SSR) with zero-crossing detection

---

## 📚 What You've Learned

✅ **Relay Basics**
- Normally open (NO) vs normally closed (NC)
- Coil voltage and current requirements
- Contact ratings and derating

✅ **Driver Circuits**
- Transistor driver design
- ULN2003 Darlington array
- Optocoupler isolation

✅ **Protection Circuits**
- Flyback diode for coil
- Snubber for inductive loads
- Decoupling capacitors

✅ **Control Logic**
- Active low operation
- Sequential control
- Timer-based control
- Pulse output

✅ **Safety**
- Electrical isolation
- Proper component ratings
- Ground separation
- High voltage precautions

---

## 🚀 Next Steps

After mastering this example:

1. **Build Practical Projects:**
   - Automatic light timer
   - Temperature-controlled fan
   - Motorized door lock
   - Irrigation system controller

2. **Add Sensors:**
   - Temperature sensor (DS18B20, LM35)
   - Light sensor (LDR)
   - Motion sensor (PIR)
   - Humidity sensor (DHT11)

3. **Advanced Features:**
   - LCD status display
   - Real-time clock (DS1307)
   - EEPROM logging
   - Remote control (RF/IR)

4. **Learn Solid State Relays:**
   - Silent operation
   - Longer lifespan
   - Faster switching
   - Zero-crossing detection

5. **Explore Motor Control:**
   - H-bridge with transistors
   - PWM speed control
   - Stepper motors
   - Servo motors

---

## 🎓 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **SPST** | Single Pole Single Throw (on/off) |
| **SPDT** | Single Pole Double Throw (changeover) |
| **NO** | Normally Open (disconnected when OFF) |
| **NC** | Normally Closed (connected when OFF) |
| **Coil** | Electromagnetic actuator |
| **Contact** | Switching terminals |
| **Flyback Diode** | Protects against voltage spikes |
| **Optocoupler** | Optical isolation between circuits |
| **ULN2003** | Darlington transistor array driver |
| **Active Low** | Logic 0 turns device ON |
| **Derating** | Operating below rated capacity |

---

## 🐛 Debug Checklist

Before asking for help, check:

- [ ] Relay coil voltage matches supply (5V for 8051)
- [ ] Flyback diode installed correctly (cathode to positive)
- [ ] Base resistor value appropriate (1kΩ typical)
- [ ] Optocoupler LED current sufficient (5-20mA)
- [ ] Load within relay contact rating
- [ ] AC loads: Using proper AC-rated relay
- [ ] Power supply can handle coil + load current
- [ ] Decoupling capacitors installed
- [ ] Ground connections proper (single point)
- [ ] Tested with LED before connecting real load

---

## 📖 Additional Resources

### Datasheets
- [SRD-05VDC-SL-C Relay Datasheet](https://www.datasheetcafe.com/srd-05vdc-sl-c-datasheet/)
- [ULN2003 Datasheet](https://www.ti.com/lit/ds/symlink/uln2003a.pdf)
- [2N2222 Transistor Datasheet](https://www.onsemi.com/pub/Collateral/P2N2222A-D.PDF)

### Learning Materials
- [Relay Working Principle](https://www.electronics-tutorials.ws/blog/relay.html)
- [Transistor as Switch](https://www.electronics-tutorials.ws/transistor/tran_4.html)
- [Flyback Diode Explanation](https://www.learningaboutelectronics.com/Articles/Flyback-diode)

### Safety References
- [Electrical Safety Guidelines](https://www.osha.gov/safe-working/safety-guidelines)
- [High Voltage Safety](https://www.physics.ohio-state.edu/~k6.1/ safety/safety.html)

---

## 🤝 Contributing

Have improvements? Found a bug?
- Add more variations
- Improve documentation
- Fix errors
- Add safety tips

---

**Difficulty:** ⭐⭐ Beginner-Intermediate
**Time to Complete:** 2-3 hours
**Hardware Required:** Relay module, transistor/ULN2003, load device
**Safety:** ⚠️ High voltage caution required

**Happy Coding - and stay safe!** 💡
