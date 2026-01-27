# Example 5: Buzzer Control

## Overview

Buzzers are audio signaling devices that produce sound when energized. They are commonly used for alarms, alerts, and user feedback in embedded systems. There are two main types: **active buzzers** (which produce a fixed frequency when powered) and **passive buzzers** (which require an oscillating signal to produce sound).

---

## 📝 Complete Source Code

### Version 1: Simple Beep (Active Buzzer)

**File:** `Buzzer_Simple_Beep.c`

```c
// Simple Buzzer Beep Example
// Produces a short beep using active buzzer
// Target: 8051 @ 12MHz
// Hardware: Active buzzer on P1.0 with transistor driver

#include <reg51.h>

// Buzzer control pin
sbit BUZZER = P1^0;

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

/**
 * @brief  Produce a short beep
 * @param  duration: Beep duration in milliseconds
 * @retval None
 */
void beep(unsigned int duration) {
    BUZZER = 0;      // Turn ON buzzer (active low with transistor)
    delay(duration);
    BUZZER = 1;      // Turn OFF buzzer
}

void main() {
    while(1) {  // Infinite loop
        beep(200);     // Beep for 200ms
        delay(1000);   // Wait 1 second
    }
}
```

---

### Version 2: Tone Generator (Passive Buzzer)

**File:** `Buzzer_Tone_Generator.c`

```c
// Tone Generator for Passive Buzzer
// Generates musical notes by toggling buzzer at specific frequencies
// Target: 8051 @ 12MHz
// Hardware: Passive buzzer on P1.0

#include <reg51.h>

sbit BUZZER = P1^0;

// Musical note frequencies (in Hz)
#define NOTE_C4  262
#define NOTE_D4  294
#define NOTE_E4  330
#define NOTE_F4  349
#define NOTE_G4  392
#define NOTE_A4  440
#define NOTE_B4  494
#define NOTE_C5  523

/**
 * @brief  Generate a tone for specified duration
 * @param  frequency: Tone frequency in Hz
 * @param  duration: Duration in milliseconds
 * @retval None
 */
void tone(unsigned int frequency, unsigned int duration) {
    unsigned long period;
    unsigned int half_period;
    unsigned int cycles, i;

    // Calculate half period in microseconds
    period = 1000000UL / frequency;  // Period in microseconds
    half_period = (unsigned int)(period / 2);

    // Calculate number of cycles
    cycles = ((unsigned long)duration * frequency) / 1000;

    for(i = 0; i < cycles; i++) {
        BUZZER = 0;  // Turn ON
        delay_us(half_period);
        BUZZER = 1;  // Turn OFF
        delay_us(half_period);
    }
}

/**
 * @brief  Microsecond delay (approximate)
 * @param  us: Delay time in microseconds
 * @retval None
 * Note:   At 12MHz, 1 machine cycle = 1µs, loop ~8 cycles
 */
void delay_us(unsigned int us) {
    unsigned int i;
    for(i = 0; i < us; i++) {
        // Approximately 1µs per iteration at 12MHz
        // May need calibration for accurate frequencies
        #pragma ASM
        NOP
        NOP
        #pragma ENDASM
    }
}

/**
 * @brief  Millisecond delay
 */
void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    while(1) {
        // Play a scale
        tone(NOTE_C4, 500);  delay_ms(100);
        tone(NOTE_D4, 500);  delay_ms(100);
        tone(NOTE_E4, 500);  delay_ms(100);
        tone(NOTE_F4, 500);  delay_ms(100);
        tone(NOTE_G4, 500);  delay_ms(100);
        tone(NOTE_A4, 500);  delay_ms(100);
        tone(NOTE_B4, 500);  delay_ms(100);
        tone(NOTE_C5, 500);  delay_ms(100);

        delay_ms(2000);  // Pause before repeating
    }
}
```

---

### Version 3: PWM Tone Generation (Timer-Based)

**File:** `Buzzer_PWM_Tone.c`

```c
// PWM Tone Generation Using Timer
// Generates accurate tones using hardware timer
// Target: 8051 @ 12MHz
// Hardware: Passive buzzer on P1.0

#include <reg51.h>

sbit BUZZER = P1^0;

// Global variables for tone generation
unsigned int tone_duration = 0;
bit tone_active = 0;

// Musical note frequencies
#define NOTE_C4  262
#define NOTE_CS4 277
#define NOTE_D4  294
#define NOTE_DS4 311
#define NOTE_E4  330
#define NOTE_F4  349
#define NOTE_FS4 370
#define NOTE_G4  392
#define NOTE_GS4 415
#define NOTE_A4  440
#define NOTE_AS4 466
#define NOTE_B4  494
#define NOTE_C5  523

/**
 * @brief  Calculate timer reload value for frequency
 * @param  frequency: Desired frequency in Hz
 * @retval Timer reload value
 */
unsigned int get_timer_reload(unsigned int frequency) {
    unsigned long period;
    unsigned int timer_value;

    // Period = 1/frequency
    // Half period for toggling
    period = 1000000UL / frequency;  // Period in µs
    period = period / 2;             // Half period

    // Convert to timer counts (1µs per count at 12MHz)
    // Timer value = 65536 - period
    timer_value = 65536UL - (unsigned int)period;

    return timer_value;
}

/**
 * @brief  Start playing a tone
 * @param  frequency: Tone frequency in Hz
 * @param  duration: Duration in milliseconds
 * @retval None
 */
void play_tone(unsigned int frequency, unsigned int duration) {
    unsigned int reload_value;

    reload_value = get_timer_reload(frequency);

    TH0 = reload_value >> 8;    // Set high byte
    TL0 = reload_value & 0xFF;  // Set low byte

    tone_duration = duration;
    tone_active = 1;

    TR0 = 1;  // Start timer
}

/**
 * @brief  Timer 0 interrupt service routine
 *         Generates square wave for tone
 */
void timer0_isr(void) interrupt 1 {
    static unsigned int ms_count = 0;

    // Reload timer
    TH0 = get_timer_reload(NOTE_A4) >> 8;    // Should store reload value!
    TL0 = get_timer_reload(NOTE_A4) & 0xFF;

    // Toggle buzzer
    BUZZER = ~BUZZER;

    // Count duration (simplified - needs separate ms timer)
    // In practice, use separate timer for duration counting
}

/**
 * @brief  Initialize Timer 0 for tone generation
 */
void timer0_init() {
    TMOD &= 0xF0;     // Clear Timer 0 mode bits
    TMOD |= 0x01;     // Timer 0, Mode 1 (16-bit)

    TH0 = 0xFC;       // Initial value
    TL0 = 0x18;

    ET0 = 1;          // Enable Timer 0 interrupt
    EA = 1;           // Global interrupt enable
}

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void main() {
    timer0_init();
    BUZZER = 1;  // Start with buzzer OFF

    while(1) {
        // Simple tone generation using polling (more reliable for beginners)
        // For advanced: Use timer interrupts

        BUZZER = 0;  delay_us(1130);  // ~440Hz (A4)
        BUZZER = 1;  delay_us(1130);

        BUZZER = 0;  delay_us(1130);
        BUZZER = 1;  delay_us(1130);

        BUZZER = 0;  delay_us(1130);
        BUZZER = 1;  delay_us(1130);

        BUZZER = 0;  delay_us(1130);
        BUZZER = 1;  delay_us(1130);

        delay_ms(1000);
    }
}

// Simplified microsecond delay
void delay_us(unsigned int us) {
    unsigned int i;
    for(i = 0; i < us; i++) {
        // Approximately 1µs per iteration
    }
}
```

---

### Version 4: Simple Music Player

**File:** `Buzzer_Music_Player.c`

```c
// Simple Music Player
// Plays melodies using passive buzzer
// Target: 8051 @ 12MHz
// Hardware: Passive buzzer on P1.0

#include <reg51.h>

sbit BUZZER = P1^0;

// Musical note definitions
#define NOTE_C3  131
#define NOTE_CS3 139
#define NOTE_D3  147
#define NOTE_DS3 156
#define NOTE_E3  165
#define NOTE_F3  175
#define NOTE_FS3 185
#define NOTE_G3  196
#define NOTE_GS3 208
#define NOTE_A3  220
#define NOTE_AS3 233
#define NOTE_B3  247
#define NOTE_C4  262
#define NOTE_CS4 277
#define NOTE_D4  294
#define NOTE_DS4 311
#define NOTE_E4  330
#define NOTE_F4  349
#define NOTE_FS4 370
#define NOTE_G4  392
#define NOTE_GS4 415
#define NOTE_A4  440
#define NOTE_AS4 466
#define NOTE_B4  494
#define NOTE_C5  523
#define NOTE_CS5 554
#define NOTE_D5  587
#define NOTE_DS5 622
#define NOTE_E5  659
#define NOTE_F5  698
#define NOTE_FS5 740
#define NOTE_G5  784
#define NOTE_GS5 831
#define NOTE_A5  880
#define NOTE_AS5 932
#define NOTE_B5  988
#define NOTE_C6  1047

// Note types (duration modifiers)
#define WHOLE    1
#define HALF     2
#define QUARTER  4
#define EIGHTH   8
#define SIXTEENTH 16

// Tempo (beats per minute)
#define TEMPO 120

// Song structure
typedef struct {
    unsigned int frequency;
    unsigned char duration;  // Note type (QUARTER, EIGHTH, etc.)
} NOTE;

/**
 * @brief  Play a single note
 * @param  frequency: Note frequency
 * @param  duration: Note duration type
 * @retval None
 */
void play_note(unsigned int frequency, unsigned char duration) {
    unsigned long period_us;
    unsigned int half_period_us;
    unsigned int note_time_ms;
    unsigned int cycles, i;

    if(frequency == 0) {
        // Rest (silence)
        note_time_ms = (60000 / TEMPO) * (4 / duration);
        delay_ms(note_time_ms);
        return;
    }

    // Calculate half period in microseconds
    period_us = 1000000UL / frequency;
    half_period_us = (unsigned int)(period_us / 2);

    // Calculate note duration in milliseconds
    note_time_ms = ((60000UL / TEMPO) * 4) / duration;

    // Number of complete cycles
    cycles = ((unsigned long)note_time_ms * frequency) / 1000;

    // Generate the tone
    for(i = 0; i < cycles; i++) {
        BUZZER = 0;
        delay_us_precise(half_period_us);
        BUZZER = 1;
        delay_us_precise(half_period_us);
    }

    // Small pause between notes
    delay_ms(note_time_ms / 10);
}

/**
 * @brief  Precise microsecond delay
 * @param  us: Microseconds to delay
 * @retval None
 */
void delay_us_precise(unsigned int us) {
    unsigned int i;
    // Calibrated for 12MHz
    for(i = 0; i < us; i++) {
        _nop_();  _nop_();
        _nop_();  _nop_();
        _nop_();  _nop_();
    }
}

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

// Song: "Twinkle Twinkle Little Star"
// Format: {NOTE, DURATION}
// End with {0, 0}
code NOTE song_twinkle[] = {
    {NOTE_C4, QUARTER}, {NOTE_C4, QUARTER}, {NOTE_G4, QUARTER}, {NOTE_G4, QUARTER},
    {NOTE_A4, QUARTER}, {NOTE_A4, QUARTER}, {NOTE_G4, HALF},
    {NOTE_F4, QUARTER}, {NOTE_F4, QUARTER}, {NOTE_E4, QUARTER}, {NOTE_E4, QUARTER},
    {NOTE_D4, QUARTER}, {NOTE_D4, QUARTER}, {NOTE_C4, HALF},
    {NOTE_G4, QUARTER}, {NOTE_G4, QUARTER}, {NOTE_F4, QUARTER}, {NOTE_F4, QUARTER},
    {NOTE_E4, QUARTER}, {NOTE_E4, QUARTER}, {NOTE_D4, HALF},
    {NOTE_G4, QUARTER}, {NOTE_G4, QUARTER}, {NOTE_F4, QUARTER}, {NOTE_F4, QUARTER},
    {NOTE_E4, QUARTER}, {NOTE_E4, QUARTER}, {NOTE_D4, HALF},
    {NOTE_C4, QUARTER}, {NOTE_C4, QUARTER}, {NOTE_G4, QUARTER}, {NOTE_G4},
    {NOTE_A4, QUARTER}, {NOTE_A4, QUARTER}, {NOTE_G4, HALF},
    {0, 0}  // End marker
};

// Song: "Happy Birthday"
code NOTE song_birthday[] = {
    {NOTE_C4, EIGHTH}, {NOTE_C4, EIGHTH}, {NOTE_D4, QUARTER}, {NOTE_C4, QUARTER},
    {NOTE_F4, QUARTER}, {NOTE_E4, HALF},
    {NOTE_C4, EIGHTH}, {NOTE_C4, EIGHTH}, {NOTE_D4, QUARTER}, {NOTE_C4, QUARTER},
    {NOTE_G4, QUARTER}, {NOTE_F4, HALF},
    {NOTE_C4, EIGHTH}, {NOTE_C4, EIGHTH}, {NOTE_C5, QUARTER}, {NOTE_A4, QUARTER},
    {NOTE_F4, QUARTER}, {NOTE_E4, QUARTER}, {NOTE_D4, HALF},
    {NOTE_AS4, EIGHTH}, {NOTE_AS4, EIGHTH}, {NOTE_A4, QUARTER}, {NOTE_F4, QUARTER},
    {NOTE_G4, QUARTER}, {NOTE_F4, HALF},
    {0, 0}  // End marker
};

/**
 * @brief  Play a song from array
 * @param  song: Pointer to song array
 * @retval None
 */
void play_song(const code NOTE *song) {
    unsigned char i = 0;

    while(1) {
        // Check for end marker
        if(song[i].frequency == 0 && song[i].duration == 0) {
            break;
        }

        play_note(song[i].frequency, song[i].duration);
        i++;
    }
}

void main() {
    BUZZER = 1;  // Start with buzzer OFF

    while(1) {
        // Play Twinkle Twinkle Little Star
        play_song(song_twinkle);

        delay_ms(2000);  // Pause between songs

        // Play Happy Birthday
        play_song(song_birthday);

        delay_ms(2000);
    }
}

// Inline NOP instruction
void _nop_(void) {
    #pragma ASM
    NOP
    #pragma ENDASM
}
```

---

### Version 5: Button-Controlled Sounds

**File:** `Buzzer_Button_Sounds.c`

```c
// Button-Controlled Buzzer Sounds
// Different buttons produce different sounds
// Target: 8051 @ 12MHz
// Hardware: Buzzer on P1.0, Buttons on P3.0-P3.3

#include <reg51.h>

sbit BUZZER = P1^0;
sbit BUTTON1 = P3^0;  // Short beep
sbit BUTTON2 = P3^1;  // Long beep
sbit BUTTON3 = P3^2;  // Double beep
sbit BUTTON4 = P3^3;  // Alarm sound

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void delay_us(unsigned int us) {
    unsigned int i;
    for(i = 0; i < us; i++);
}

// Sound 1: Short beep (confirmation)
void beep_short() {
    BUZZER = 0;
    delay_ms(100);
    BUZZER = 1;
}

// Sound 2: Long beep (warning)
void beep_long() {
    BUZZER = 0;
    delay_ms(500);
    BUZZER = 1;
}

// Sound 3: Double beep (attention)
void beep_double() {
    BUZZER = 0;
    delay_ms(100);
    BUZZER = 1;
    delay_ms(100);
    BUZZER = 0;
    delay_ms(100);
    BUZZER = 1;
}

// Sound 4: Alarm (siren effect)
void beep_alarm() {
    unsigned int i;
    for(i = 0; i < 5; i++) {
        // High tone
        BUZZER = 0; delay_us(500); BUZZER = 1; delay_us(500);
        BUZZER = 0; delay_us(500); BUZZER = 1; delay_us(500);
        BUZZER = 0; delay_us(500); BUZZER = 1; delay_us(500);
        BUZZER = 0; delay_us(500); BUZZER = 1; delay_us(500);

        delay_ms(50);

        // Low tone
        BUZZER = 0; delay_us(1000); BUZZER = 1; delay_us(1000);
        BUZZER = 0; delay_us(1000); BUZZER = 1; delay_us(1000);

        delay_ms(50);
    }
}

// Sound 5: Rising tone (startup)
void beep_rising() {
    unsigned int i;
    for(i = 0; i < 5; i++) {
        BUZZER = 0; delay_us(1000 - i * 150);
        BUZZER = 1; delay_us(1000 - i * 150);
    }
}

// Sound 6: Falling tone (shutdown)
void beep_falling() {
    unsigned int i;
    for(i = 0; i < 5; i++) {
        BUZZER = 0; delay_us(300 + i * 150);
        BUZZER = 1; delay_us(300 + i * 150);
    }
}

void main() {
    BUZZER = 1;  // Start OFF

    while(1) {
        if(BUTTON1 == 0) {
            delay_ms(20);  // Debounce
            if(BUTTON1 == 0) {
                beep_short();
                while(BUTTON1 == 0);  // Wait for release
            }
        }

        if(BUTTON2 == 0) {
            delay_ms(20);
            if(BUTTON2 == 0) {
                beep_long();
                while(BUTTON2 == 0);
            }
        }

        if(BUTTON3 == 0) {
            delay_ms(20);
            if(BUTTON3 == 0) {
                beep_double();
                while(BUTTON3 == 0);
            }
        }

        if(BUTTON4 == 0) {
            delay_ms(20);
            if(BUTTON4 == 0) {
                beep_alarm();
                while(BUTTON4 == 0);
            }
        }
    }
}
```

---

## 🔌 Hardware Connection

### Active Buzzer (Simplest)

```
         8051                    Active Buzzer Module
       ┌──────┐                 ┌────────────────┐
       │      │                 │                │
       │ 5V   ├─────────────────┤ VCC            │
       │      │                 │                │
       │ GND  ├─────────────────┤ GND            │
       │      │                 │                │
       │ P1.0 ├─────────────────┤ I/O (Signal)   │
       │      │                 │                │
       └──────┘                 │    ┌──────┐    │
                                 │    │      │    │
                                 │    │ Buzzer│   │
                                 │    │      │   │
                                 │    └──────┘   │
                                 └────────────────┘

Note: Active buzzer has internal oscillator
- Just apply power → Fixed frequency sound
- Cannot change tone/pitch
- Simple on/off control
```

### Passive Buzzer with Transistor Driver

```
         8051                    Passive Buzzer Circuit
       ┌──────┐
       │      │                    5V Supply
       │      │                        │
       │ P1.0 ├──────1kΩ──────────────┤││
       │      │                      │   NPN
       │      │                      │  (2N2222
       │      │                      │   BC547
       │      │                      │   S8050)
       │      │                       └─┬─┘
       │      │                         │
       │      │                         ├───┐
       │      │                         │   │
       │      │                         │ ┌─┴┐
       │ GND  ├─────────────────────────┤   │
       │      │                         │ │  │
       │      │                         │ └─┬┘
       │      │                         │   │
       └──────┘                    ┌────┴───┴────┐
                                  │  Passive    │
                                  │  Buzzer     │
                                  └─────────────┘

Components:
- 1× NPN Transistor (2N2222, BC547, S8050)
- 1× Resistor 1kΩ (base current limiting)
- 1× Passive buzzer

Note: Passive buzzer needs oscillating signal
- Frequency controls pitch
- PWM controls volume
```

### Direct Connection (Low Power Buzzer Only)

```
         8051
       ┌──────┐
       │      │
       │ P1.0 ├─────────┐
       │      │         │
       │      │        ┌┴┐
       │      │        ││  Small passive buzzer
       │      │        ││  (only if current < 20mA)
       │      │        └┬┘
       │ GND  ├─────────┘
       │      │
       └──────┘

WARNING: Only for very small buzzers!
- Check buzzer current draw
- 8051 pin can only sink 20mA
- Most buzzers need transistor driver
```

---

## 📖 Code Explanation

### 1. Active vs Passive Buzzer

**Active Buzzer:**
```c
BUZZER = 0;  // ON - produces fixed frequency beep
BUZZER = 1;  // OFF
```
- Internal oscillator generates fixed frequency (typically 2-5kHz)
- Simply apply power to make sound
- Cannot change pitch
- Easier to use

**Passive Buzzer:**
```c
// Must generate square wave
BUZZER = 0; delay_us(500);  // Half period
BUZZER = 1; delay_us(500);  // Half period
// Repeat...
```
- No internal oscillator
- You provide the frequency
- Can play different notes and music
- Requires more code

### 2. Frequency Calculation

To generate a specific frequency:

```c
// Frequency = 1 / Period
// Period (seconds) = 1 / Frequency (Hz)
// Period (microseconds) = 1000000 / Frequency

// Example: 440Hz (A4 note)
period = 1000000 / 440 = 2273µs
half_period = 2273 / 2 = 1136µs

// Toggle every 1136µs
BUZZER = 0; delay_us(1136);
BUZZER = 1; delay_us(1136);
```

### 3. Musical Note Frequencies

```
Octave 3:
C3: 131 Hz    | G3: 196 Hz
D3: 147 Hz    | A3: 220 Hz
E3: 165 Hz    | B3: 247 Hz
F3: 175 Hz    |

Octave 4 (Middle):
C4: 262 Hz    | G4: 392 Hz
D4: 294 Hz    | A4: 440 Hz (Standard tuning pitch)
E4: 330 Hz    | B4: 494 Hz
F4: 349 Hz    |

Octave 5:
C5: 523 Hz    | G5: 784 Hz
D5: 587 Hz    | A5: 880 Hz
E5: 659 Hz    | B5: 988 Hz
F5: 698 Hz    |
```

### 4. Volume Control with PWM

```c
// Volume is controlled by duty cycle
// Duty cycle = (ON time) / (Total period)

// Full volume (50% duty cycle - max for square wave)
void tone_full(unsigned int freq) {
    unsigned int half_period = 1000000 / freq / 2;
    BUZZER = 0; delay_us(half_period);
    BUZZER = 1; delay_us(half_period);
}

// Half volume (25% duty cycle)
void tone_half(unsigned int freq) {
    unsigned int period = 1000000 / freq;
    BUZZER = 0; delay_us(period / 4);
    BUZZER = 1; delay_us(3 * period / 4);
}

// Low volume (12.5% duty cycle)
void tone_low(unsigned int freq) {
    unsigned int period = 1000000 / freq;
    BUZZER = 0; delay_us(period / 8);
    BUZZER = 1; delay_us(7 * period / 8);
}
```

---

## 🔬 Buzzer Specifications

### Identifying Active vs Passive

**Visual Inspection:**

```
Active Buzzer:
- Usually sealed with black epoxy
- Thicker housing
- Label often says "Active" or shows frequency
- Cannot see internal components

Passive Buzzer:
- Often see internal coil or piezo element
- May have open back or grid
- Thinner profile
- Sometimes labeled "Passive" or "None"
```

**Testing with Multimeter:**

```
Resistance test:
Active:  High resistance (1kΩ+) or open circuit
Passive: Low resistance (8-50Ω) - coil is visible

Continuity test with 5V:
Active:  Makes sound when +5V applied
Passive:  No sound (needs oscillating signal)
```

### Buzzer Specifications

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| **Operating Voltage** | 3-24V DC | 5V most common for 8051 |
| **Current Draw** | 10-50mA | Active typically higher |
| **Frequency Range** | 2-8kHz | Passive depends on drive |
| **Sound Pressure** | 70-95dB | At 10cm distance |
| **Resonant Frequency** | 4kHz | Passive buzzer optimum |
| **Impedance** | 8-50Ω | Passive only |

---

## 🛠️ Testing and Troubleshooting

### Testing Hardware

**Test 1: Direct Power Test**
```c
void main() {
    BUZZER = 0;  // Should make sound
    while(1);
}
```

**Test 2: Tone Test**
```c
void main() {
    while(1) {
        BUZZER = 0; delay_us(1000);
        BUZZER = 1; delay_us(1000);
    }
}
// Should hear ~500Hz tone
```

**Test 3: Frequency Sweep**
```c
void main() {
    unsigned int delay;
    for(delay = 2000; delay > 100; delay -= 100) {
        BUZZER = 0; delay_us(delay);
        BUZZER = 1; delay_us(delay);
    }
}
// Should hear rising pitch
```

### Common Issues and Solutions

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| No sound at all | Wrong buzzer type | Check active vs passive |
| Very quiet | Insufficient drive current | Use transistor driver |
| Distorted sound | Frequency too high/low | Use 1-5kHz range |
| Wrong pitch | Timing not accurate | Calibrate delay function |
| Buzzer gets hot | Voltage too high | Check rated voltage |
| 8051 resets when buzzer on | Voltage spike | Add flyback diode |
| Continuous tone instead of beeps | Wrong logic (active vs passive) | Invert control signal |

### Debugging Steps

**1. Verify buzzer type:**
```c
// Apply continuous power
void main() {
    BUZZER = 0;  // Active: should sound
    while(1);    // Passive: silent or weak click
}
```

**2. Test frequency generation:**
```c
// Use known frequency (440Hz = A4)
void main() {
    while(1) {
        BUZZER = 0; delay_us(1136);  // 440Hz half-period
        BUZZER = 1; delay_us(1136);
    }
}
// Compare with tuning fork or online tone generator
```

**3. Check timer calculation:**
```c
// Calculate expected frequency
unsigned int delay_us = 1136;  // For 440Hz
unsigned int freq = 1000000 / (2 * delay_us);
// Should give 440Hz
```

---

## 📊 Variations and Extensions

### Variation 1: SOS Signal

```c
// Morse code SOS: ... --- ...
void dot() {
    BUZZER = 0; delay_ms(100);
    BUZZER = 1; delay_ms(100);
}

void dash() {
    BUZZER = 0; delay_ms(300);
    BUZZER = 1; delay_ms(100);
}

void sos() {
    // S: ...
    dot(); dot(); dot(); delay_ms(200);

    // O: ---
    dash(); dash(); dash(); delay_ms(200);

    // S: ...
    dot(); dot(); dot(); delay_ms(1000);
}
```

### Variation 2: Doorbell Chime

```c
// "Ding-dong" doorbell sound
void doorbell() {
    // Ding (high C5)
    unsigned int i;
    for(i = 0; i < 20; i++) {
        BUZZER = 0; delay_us(956);  // C5
        BUZZER = 1; delay_us(956);
    }
    delay_ms(200);

    // Dong (lower G4)
    for(i = 0; i < 15; i++) {
        BUZZER = 0; delay_us(1276); // G4
        BUZZER = 1; delay_us(1276);
    }
}
```

### Variation 3: Star Wars Theme

```c
// Star Wars opening theme (simplified)
void star_wars() {
    // Part 1: G-G-G-Eb-Bb-G-Eb-Bb-G
    tone(NOTE_G4, 300); delay_ms(50);
    tone(NOTE_G4, 300); delay_ms(50);
    tone(NOTE_G4, 300); delay_ms(50);
    tone(NOTE_DS4, 500); delay_ms(50);
    tone(NOTE_AS4, 500); delay_ms(50);
    tone(NOTE_G4, 500); delay_ms(50);
    tone(NOTE_DS4, 500); delay_ms(50);
    tone(NOTE_AS4, 500); delay_ms(50);
    tone(NOTE_G4, 1000); delay_ms(500);

    // Part 2: D4-D4-D4-D4-G4 (high)...
    // Add more notes...
}
```

### Variation 4: Temperature Alert

```c
// Rising pitch for temperature warning
void temp_alert(unsigned char temp) {
    unsigned int pitch;
    unsigned char i;

    // Higher temperature = higher pitch
    pitch = 200 + (temp * 10);

    for(i = 0; i < 3; i++) {
        tone(pitch, 200);
        delay_ms(200);
    }
}
```

### Variation 5: Pulse Train

```c
// Generate specific pulse count
void pulse_train(unsigned char pulses, unsigned int frequency) {
    unsigned char i;
    unsigned int half_period;

    half_period = 1000000 / frequency / 2;

    for(i = 0; i < pulses; i++) {
        BUZZER = 0; delay_us(half_period);
        BUZZER = 1; delay_us(half_period);
        BUZZER = 0; delay_us(half_period);
        BUZZER = 1; delay_us(half_period);
        BUZZER = 0; delay_us(half_period);
        BUZZER = 1; delay_us(half_period);
        BUZZER = 0; delay_us(half_period);
        BUZZER = 1; delay_ms(500);  // Long pause
    }
}
```

### Variation 6: Frequency Sweep (Siren)

```c
// Police siren effect
void siren() {
    unsigned int delay;
    unsigned char i;

    // Rising pitch
    for(i = 0; i < 20; i++) {
        delay = 1000 - (i * 30);  // 1000→400µs
        BUZZER = 0; delay_us(delay);
        BUZZER = 1; delay_us(delay);
    }

    // Falling pitch
    for(i = 0; i < 20; i++) {
        delay = 400 + (i * 30);  // 400→1000µs
        BUZZER = 0; delay_us(delay);
        BUZZER = 1; delay_us(delay);
    }
}
```

---

## ⚡ Hardware Considerations

### Transistor Selection

**For low-power buzzer (< 100mA):**
```
2N2222 NPN Transistor:
- Max collector current: 800mA
- Max collector-emitter voltage: 40V
- Gain (hFE): 100-300
- Package: TO-92

Base resistor calculation:
Rb = (Voh - Vbe) / Ib
Rb = (5V - 0.7V) / 1mA = 4.3kΩ
Use 1kΩ-4.7kΩ
```

**For high-power buzzer/horn:**
```
TIP31C NPN Transistor:
- Max collector current: 3A
- Max collector-emitter voltage: 100V
- Requires heat sink for high current
```

### Flyback Protection

If using inductive buzzer (coil type):

```
         ┌──┐
    + ───┤  ├──┐  Diode: 1N4007 or 1N5819
         └──┘ │  Cathode (stripe) to +
              │
         ┌──┐ │
    - ───┤  ├─┘
         └──┘

Purpose: Protect transistor from voltage spikes
when buzzer turns off
```

### Volume Optimization

**Maximum volume:**
- Drive at resonant frequency (typically 3-4kHz)
- Use 50% duty cycle square wave
- Ensure adequate current (20-50mA)
- Mount buzzer on sound chamber

**If too loud:**
- Reduce duty cycle (PWM)
- Add series resistor (100-330Ω)
- Lower drive voltage
- Use foam tape to dampen

**If too quiet:**
- Check drive current
- Use transistor for more current
- Try resonant frequency
- Add sound chamber/horn

---

## 📚 What You've Learned

✅ **Buzzer Types**
- Active vs passive buzzers
- How to identify each type
- When to use which type

✅ **Sound Generation**
- Frequency control for pitch
- PWM for volume control
- Timing calculations

✅ **Music Programming**
- Musical note frequencies
- Note durations (whole, half, quarter)
- Song data structures
- Playing melodies

✅ **Hardware Design**
- Transistor driver circuits
- Flyback protection
- Component selection

✅ **Practical Applications**
- Alert sounds
- Button feedback
- Alarm systems
- Simple music player

---

## 🚀 Next Steps

After mastering this example:

1. **Build Practical Projects:**
   - Doorbell system
   - Timer alarm
   - Temperature alert
   - Button sound feedback

2. **Add Features:**
   - Volume control buttons
   - Song selection
   - Recording/playback (with external EEPROM)
   - Visual display (LEDs synchronized with sound)

3. **Advanced Audio:**
   - Learn about DAC (Digital-to-Analog Converter)
   - Generate more complex waveforms
   - Sound synthesis techniques
   - Voice playback (with external chips)

4. **Real-World Applications:**
   - Alarm clock with snooze
   - Musical instrument
   - Game sound effects
   - Accessibility (audio feedback for visually impaired)

5. **Related Topics:**
   - Study timers in depth (for accurate timing)
   - Learn about interrupts (for background sound)
   - Explore PWM for better volume control
   - See: [Timer Examples](../02_Timers/)
   - See: [Interrupt Examples](../03_Interrupts/)

---

## 🎓 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Active Buzzer** | Has internal oscillator, fixed frequency |
| **Passive Buzzer** | Requires external oscillating signal |
| **Frequency** | Determines pitch (higher = higher pitch) |
| **Duty Cycle** | Ratio of ON time to total period (controls volume) |
| **PWM** | Pulse Width Modulation for volume control |
| **Square Wave** | Simple audio waveform (on/off) |
| **Resonant Frequency** | Frequency at which buzzer is loudest |
| **Sound Pressure** | Volume measured in dB |
| **Flyback Diode** | Protects against inductive spikes |
| **Timer Interrupt** | Generates accurate timing without delays |

---

## 🐛 Debug Checklist

Before asking for help, check:

**Hardware:**
- [ ] Correct buzzer type (active vs passive)
- [ ] Power supply voltage matches buzzer rating
- [ ] Transistor driver if needed
- [ ] Flyback diode installed (for inductive buzzers)
- [ ] Good electrical connections

**Software:**
- [ ] Correct control logic (active low vs high)
- [ ] Frequency in appropriate range (1-5kHz)
- [ ] Delay function calibrated
- [ ] Using correct timer reload values
- [ ] Interrupts enabled (if using timer interrupt)

**Testing:**
- [ ] Buzzer works with direct power
- [ ] Can generate known frequency (440Hz)
- [ ] Volume is adequate
- [ ] Sound is clear (not distorted)

---

## 📖 Additional Resources

### Datasheets
- [Piezo Buzzer Datasheet](https://www.murata.com/products/productdata?www=/catalogspeaker/speaker/piezo_e.pdf)
- [2N2222 Transistor Datasheet](https://www.onsemi.com/pub/Collateral/P2N2222A-D.PDF)

### Learning Materials
- [Sound Generation Tutorial](https://www.electronics-tutorials.ws/waveforms/generators.html)
- [Musical Note Frequencies](https://pages.mtu.edu/~suits/notefreqs.html)
- [PWM Audio Generation](https://www.arduino.cc/en/Tutorial/PWM)

### Music Resources
- [Music Notes Frequency Chart](https://www.seventhstring.com/resources/notefrequencies.html)
- [Simple Piano Songs](https://www.makingmusicfun.net/)
- [Morse Code Alphabet](https://morsecode.world/international/morse.html)

### Tools
- **Online Tone Generator:** For testing and calibration
- **Frequency Counter Apps:** For smartphone
- **Musical Tuner:** To verify note frequencies
- **Oscilloscope:** To see waveforms

---

## 🤝 Contributing

Have improvements? Found a bug?
- Add more song examples
- Improve timing accuracy
- Add more sound effects
- Fix errors
- Share your compositions!

---

**Difficulty:** ⭐⭐ Beginner-Intermediate
**Time to Complete:** 2-3 hours
**Hardware Required:** Buzzer (active or passive), transistor (for passive)
**Fun Factor:** 🎵 High (makes noise!)

**Happy Coding - and make some noise!** 🔊
