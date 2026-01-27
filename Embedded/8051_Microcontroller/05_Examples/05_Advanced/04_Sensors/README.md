# Sensors

## Overview

Sensors allow your 8051 to perceive the physical world. This section covers interfacing various sensors for temperature, humidity, distance, motion, light, and more.

---

## Sensor Categories

| Type | Sensors | Output | Applications |
|------|---------|--------|--------------|
| **Temperature** | LM35, DS18B20, DHT11 | Analog/Digital | Weather, HVAC, industrial |
| **Humidity** | DHT11, DHT22 | Digital | Weather, greenhouses |
| **Distance** | HC-SR04, Sharp IR | Pulse/Analog | Robotics, parking sensors |
| **Motion** | PIR, Accelerometer | Digital/Digital | Security, drones |
| **Light** | LDR, BH1750 | Analog/Digital | Lighting control |
| **Pressure** | BMP180, MPX5010 | Digital/Analog | Altitude, weather |
| **Gas** | MQ-2, MQ-5 | Analog | Air quality, safety |

---

## Example 1: Temperature Sensing (LM35)

### Application: Digital Thermometer

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Read temperature using LM35 analog sensor and ADC.

### LM35 Specifications
- **Output:** 10mV/°C (linear)
- **Range:** -55°C to +150°C
- **Accuracy:** ±0.5°C
- **Supply:** 4V to 30V

### Hardware Connection

```
         8051                 LM35          ADC0809
       ┌──────┐             ┌──────┐      ┌─────┐
       │      │             │      │      │     │
       │      │         VCC─┤├     │      │ IN0 │
       │      │             │      │      │     │
       │  ADC │─────────────┤ Vout ├──────┤     │
       │      │             │      │      │     │
       │  GND ├─────────────┤ GND  │      │     │
       │      │             └──────┘      └─────┘
       └──────┘
```

### Source Code

```c
// LM35 Temperature Sensor
// Output: 10mV/°C

#include <reg51.h>

sbit EOC = P2^2;
sbit OE = P2^1;
sbit START = P2^3;
#define ADC_DATA P0

// Read ADC from channel 0
unsigned char read_adc_temp(void)
{
    // Select channel 0
    P3 = P3 & 0xF8;

    // Start conversion
    START = 1;
    START = 0;

    while(EOC == 0);

    OE = 1;
    unsigned char val = ADC_DATA;
    OE = 0;

    return val;
}

// Convert ADC to temperature
float get_temperature_c(unsigned char adc)
{
    // VREF = 5V, LM35 = 10mV/°C
    // Temp(°C) = (ADC / 255) × 5000mV / 10mV/°C
    return ((float)adc * 5000.0) / 2550.0;
}

unsigned char get_temperature_int(unsigned char adc)
{
    // Return integer temperature
    return (unsigned char)(((unsigned int)adc * 500) / 255);
}

void main(void)
{
    unsigned char adc_val;
    unsigned char temp_int;

    while(1) {
        adc_val = read_adc_temp();
        temp_int = get_temperature_int(adc_val);

        // Use temp_int here
        // Display on LCD or send via serial

        for(unsigned int i = 0; i < 50000; i++);  // Delay
    }
}
```

---

## Example 2: Digital Temperature (DS18B20)

### Application: Precision Temperature Logging

**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Interface DS18B20 digital temperature sensor using 1-Wire protocol.

### DS18B20 Specifications
- **Range:** -55°C to +125°C
- **Accuracy:** ±0.5°C
- **Resolution:** 9-12 bit (configurable)
- **Interface:** 1-Wire

### Hardware Connection

```
         8051               DS18B20
       ┌──────┐            ┌──────┐
       │      │            │      │
       │  P1.0├──────┬─────┤ DQ   │
       │      │      │     │      │
       │  VCC ├──────┼─────┤ VDD  │
       │      │   4.7kΩ     │      │
       │  GND ├──────┴─────┤ GND  │
       └──────┘            └──────┘
```

**Important:** 4.7kΩ pull-up resistor on DQ line!

### 1-Wire Protocol

**Reset Pulse:**
```
Master    ───┐      ┌────────────
              │      │
              └──480µs└───────┘
              ↓
           Pull low 480µs
```

**Presence Pulse:**
```
Slave          ────┐
                    │
                    └──60-240µs
                    ↓
              Pulls low after reset
```

### Source Code

```c
// DS18B20 Temperature Sensor
// 1-Wire Protocol

#include <reg51.h>

sbit DQ = P1^0;

// Microsecond delay (approximate)
void delay_us(unsigned int us)
{
    while(us--) {
        // Calibrate for your crystal
        for(unsigned char i = 0; i < 2; i++);
    }
}

// Reset 1-Wire bus
unsigned char ow_reset(void)
{
    unsigned char presence = 0;

    DQ = 0;        // Pull low
    delay_us(480); // 480µs reset pulse
    DQ = 1;        // Release
    delay_us(70);  // Wait for presence

    presence = DQ; // Sample presence

    delay_us(410); // Complete reset

    return presence; // 0 = device present
}

// Write bit to 1-Wire
void ow_write_bit(unsigned char bit)
{
    DQ = 0;        // Pull low

    if(bit) {
        delay_us(6);   // 6µs for '1'
        DQ = 1;        // Release
        delay_us(64);  // Wait
    } else {
        delay_us(60);  // 60µs for '0'
        DQ = 1;        // Release
        delay_us(10);  // Recovery
    }
}

// Read bit from 1-Wire
unsigned char ow_read_bit(void)
{
    unsigned char bit = 0;

    DQ = 0;        // Pull low
    delay_us(6);   // 6µs
    DQ = 1;        // Release
    delay_us(9);   // Wait 15µs total

    bit = DQ;      // Sample

    delay_us(55);  // Complete slot

    return bit;
}

// Write byte to 1-Wire
void ow_write_byte(unsigned char data)
{
    for(unsigned char i = 0; i < 8; i++) {
        ow_write_bit(data & 0x01);
        data >>= 1;
    }
}

// Read byte from 1-Wire
unsigned char ow_read_byte(void)
{
    unsigned char data = 0;

    for(unsigned char i = 0; i < 8; i++) {
        data >>= 1;
        if(ow_read_bit()) {
            data |= 0x80;
        }
    }

    return data;
}

// Start temperature conversion
void ds18b20_start(void)
{
    ow_reset();
    ow_write_byte(0xCC);  // Skip ROM
    ow_write_byte(0x44);  // Convert T
}

// Read temperature from DS18B20
float ds18b20_read_temp(void)
{
    unsigned char temp_l, temp_h;
    int temp_raw;
    float temp_c;

    ow_reset();
    ow_write_byte(0xCC);  // Skip ROM
    ow_write_byte(0xBE);  // Read scratchpad

    temp_l = ow_read_byte();
    temp_h = ow_read_byte();

    temp_raw = (temp_h << 8) | temp_l;
    temp_c = temp_raw / 16.0;

    return temp_c;
}

void main(void)
{
    float temperature;

    while(1) {
        ds18b20_start();
        delay_us(750000);  // Wait 750ms for conversion

        temperature = ds18b20_read_temp();

        // Use temperature value
        // Display or send via serial

        delay_us(100000);  // Wait before next reading
    }
}
```

---

## Example 3: Ultrasonic Distance Sensor (HC-SR04)

### Application: Distance Measurement

**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Measure distance using ultrasonic sensor.

### HC-SR04 Specifications
- **Range:** 2cm to 400cm
- **Accuracy:** ±3mm
- **Frequency:** 40kHz
- **Interface:** 2 pins (Trigger, Echo)

### Hardware Connection

```
         8051                HC-SR04
       ┌──────┐             ┌────────┐
       │      │             │        │
       │  P1.0├─────────────┤ Trig   │
       │      │             │        │
       │  P1.1├─────────────┤ Echo   │
       │      │             │        │
       │  VCC ├─────────────┤ VCC    │
       │  GND ├─────────────┤ GND    │
       └──────┘             └────────┘
```

### Timing Diagram

```
Trigger:
    ┌───┐
    │   │
────┘   └────────────────────────
    10µs pulse

Echo:
                  ┌─────────────────────
                  │                     │
──────────────────┘                     └──
                  ←─ t (time to return) →

Distance = (t × 340) / 2
         = (t × 0.034) / 2
         = t × 0.017  (cm/µs)
```

### Source Code

```c
// HC-SR04 Ultrasonic Distance Sensor
// Measure distance in cm

#include <reg51.h>

sbit TRIG = P1^0;
sbit ECHO = P1^1;

// Timer 1 for measuring pulse width
void timer1_init(void)
{
    TMOD &= 0x0F;  // Clear Timer 1 mode
    TMOD |= 0x10;  // Timer 1, Mode 1 (16-bit)
    TR1 = 0;       // Start stopped
}

// Send trigger pulse
void send_trigger(void)
{
    TRIG = 0;
    TRIG = 1;
    delay_us(10);  // 10µs pulse
    TRIG = 0;
}

// Measure distance (returns cm)
unsigned int measure_distance(void)
{
    unsigned int distance;
    unsigned int time;

    send_trigger();

    // Wait for Echo to go high
    while(ECHO == 0);

    // Start Timer 1
    TH1 = 0;
    TL1 = 0;
    TR1 = 1;

    // Wait for Echo to go low
    while(ECHO == 1);

    // Stop Timer 1
    TR1 = 0;

    // Calculate time (in µs)
    time = (TH1 << 8) | TL1;

    // Calculate distance (cm)
    // Speed of sound = 340 m/s = 0.034 cm/µs
    // Distance = (time × 0.034) / 2
    distance = time / 59;  // Approximate for 12MHz

    return distance;
}

void main(void)
{
    unsigned int distance_cm;

    timer1_init();

    while(1) {
        distance_cm = measure_distance();

        // Use distance value
        // Display on LCD or send via serial

        // Delay between measurements
        for(unsigned int i = 0; i < 50000; i++);
    }
}
```

---

## Example 4: PIR Motion Sensor

### Application: Motion Detection

**Difficulty:** ⭐⭐ Intermediate

**Description:**
Detect motion using PIR (Passive Infrared) sensor.

### Hardware Connection

```
         8051                  PIR Sensor
       ┌──────┐              ┌──────────┐
       │      │              │          │
       │  P3.2├──────────────┤ OUT      │
       │      │              │          │
       │  VCC ├──────────────┤ VCC      │
       │  GND ├──────────────┤ GND      │
       └──────┘              └──────────┘

Optional: Mode jumper for repeat/single trigger
```

### Source Code

```c
// PIR Motion Sensor (HC-SR501)
// Detect motion and trigger alarm

#include <reg51.h>

sbit PIR = P3^2;    // Connect to INT0
sbit ALARM = P1^0;  // LED or buzzer

// External interrupt 0
void pir_isr(void) interrupt 0
{
    if(PIR == 1) {
        ALARM = 0;  // Motion detected (active low LED)
    } else {
        ALARM = 1;  // No motion
    }
}

void main(void)
{
    ALARM = 1;      // Start with alarm OFF

    // Configure INT0
    IT0 = 1;       // Edge-triggered
    EX0 = 1;       // Enable INT0
    EA = 1;        // Global interrupt enable

    while(1) {
        // Main loop does nothing
        // Interrupt handles motion detection
    }
}
```

---

## Example 5: Light Sensor (LDR)

### Application: Automatic Light Control

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Read ambient light using LDR (Light Dependent Resistor).

### Hardware Connection

```
         8051
       ┌──────┐             LDR Circuit
       │      │            ┌────────┐
       │  ADC │────────────┤  Vin   │ ADC Input
       │      │            │        │
       │  VCC ├────────────┤  VCC   │
       │  GND ├────────────┤  GND   │
       └──────┘            └────────┘
                           VCC
                            │
                           LDR (10kΩ dark)
                            │
                           ─┴─
                           10kΩ
                            │
                           GND

Voltage divider: Vout = VCC × R2 / (R1 + R2)
```

### Source Code

```c
// LDR Light Sensor
// Read ambient light level

#include <reg51.h>

#define ADC_DATA P0
sbit EOC = P2^2;
sbit OE = P2^1;
sbit START = P2^3;

unsigned char read_light(void)
{
    // Select channel 0
    P3 = P3 & 0xF8;

    START = 1;
    START = 0;

    while(EOC == 0);

    OE = 1;
    unsigned char light = ADC_DATA;
    OE = 0;

    return light;  // 0 = dark, 255 = bright
}

void main(void)
{
    unsigned char light_level;

    while(1) {
        light_level = read_light();

        if(light_level < 50) {
            // Dark - turn on light
            // P1 = 0x00;
        } else {
            // Bright - turn off light
            // P1 = 0xFF;
        }

        for(unsigned int i = 0; i < 50000; i++);
    }
}
```

---

## Sensor Calibration

### 1. Linear Sensors (LM35, LDR)

**Two-point calibration:**
```
1. Measure at known point 1 (e.g., 0°C)
2. Measure at known point 2 (e.g., 100°C)
3. Calculate slope and offset

Slope = (ADC2 - ADC1) / (Temp2 - Temp1)
Offset = ADC1 - Slope × Temp1
```

### 2. Non-Linear Sensors

**Use lookup table:**
```c
unsigned int temp_lookup[] = {
    0,      // ADC 0 = 0°C
    50,     // ADC 1 = 5°C
    100,    // ADC 2 = 10°C
    // ...
};
```

---

## Signal Processing

### 1. Filtering

**Moving Average:**
```c
#define FILTER_SIZE 8

unsigned char filter_buffer[FILTER_SIZE];
unsigned char filter_index = 0;

unsigned char moving_average(unsigned char new_value)
{
    unsigned int sum = 0;

    filter_buffer[filter_index] = new_value;
    filter_index = (filter_index + 1) % FILTER_SIZE;

    for(unsigned char i = 0; i < FILTER_SIZE; i++) {
        sum += filter_buffer[i];
    }

    return sum / FILTER_SIZE;
}
```

### 2. Hysteresis

```c
// Prevent oscillation near threshold
#define HIGH_THRESHOLD 100
#define LOW_THRESHOLD  90

if(value > HIGH_THRESHOLD) {
    state = ON;
} else if(value < LOW_THRESHOLD) {
    state = OFF;
}
```

---

## Troubleshooting

### Temperature Sensors
**Wrong readings:**
- Check wiring
- Verify reference voltage
- Ensure proper calibration
- Check for thermal gradients

**No response:**
- Verify supply voltage
- Check pull-up resistor (DS18B20)
- Test with multimeter

### Ultrasonic Sensors
**Always reads 0:**
- Check trigger pulse timing
- Verify echo pin connection
- Test distance range (2-400cm)

**Erratic readings:**
- Check for soft targets (absorb sound)
- Ensure proper angle
- Filter multiple readings

### PIR Sensors
**False triggers:**
- Adjust sensitivity potentiometer
- Check for heat sources
- Ensure proper warm-up time
**No detection:**
- Verify power supply
- Check mode jumper setting
- Test within range (3-7m)

---

## Applications

### Home Automation
- Temperature monitoring
- Motion detection for lighting
- Light level control
- Gas/smoke detection

### Robotics
- Distance sensors (obstacle avoidance)
- IMU (orientation)
- Line following (IR array)
- Encoders (position)

### Weather Stations
- Temperature
- Humidity
- Barometric pressure
- Light intensity

### Industrial
- Proximity sensors
- Current sensing
- Vibration detection
- Level sensing

---

## Component Selection

### Temperature Sensors

| Sensor | Type | Range | Accuracy | Interface | Price |
|--------|------|-------|----------|-----------|-------|
| LM35 | Analog | -55 to 150°C | ±0.5°C | Analog | Low |
| DS18B20 | Digital | -55 to 125°C | ±0.5°C | 1-Wire | Medium |
| DHT11 | Digital | 0-50°C | ±2°C | Digital | Low |
| DHT22 | Digital | -40 to 80°C | ±0.5°C | Digital | Medium |

### Distance Sensors

| Sensor | Type | Range | Accuracy | Price |
|--------|------|-------|----------|-------|
| HC-SR04 | Ultrasonic | 2-400cm | ±3mm | Low |
| Sharp IR | Infrared | 10-80cm | ±10% | Medium |
| VL53L0X | Laser ToF | 0-200cm | ±3mm | High |

---

## Prerequisites

- ✅ [Basic I/O](../../01_Basic_IO/)
- ✅ [ADC/DAC](../03_ADC_DAC/)
- ✅ [Interrupts](../../03_Interrupts/)

**Recommended Reading:**
- [ADC Interface](../03_ADC_DAC/)
- [External Interrupts](../../03_Interrupts/)

---

## Next Steps

After mastering sensors:
- [Display](../05_Display/) - Visualize sensor data
- [Communication](../06_Communication/) - Transmit sensor data
- [Projects](../../../06_Projects/) - Weather station, data logger

---

**Difficulty:** ⭐⭐⭐⭐ Expert
**Time to Master:** 12-20 hours
**Hardware:** Various sensors, resistors, capacitors
