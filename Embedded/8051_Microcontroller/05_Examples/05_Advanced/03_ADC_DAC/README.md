# ADC & DAC

## Overview

Analog-to-Digital (ADC) and Digital-to-Analog (DAC) converters bridge the gap between the analog physical world and digital microcontrollers. Learn to interface sensors, generate analog signals, and process real-world data.

---

## ADC (Analog-to-Digital Converter)

### What is ADC?

Converts analog voltage (continuous) to digital value (discrete):

```
Analog Input    Digital Output
    5V    ────→    11111111 (255)
    2.5V  ────→    10000000 (128)
    0V    ────→    00000000 (0)
```

### ADC Specifications

| Parameter | Description |
|-----------|-------------|
| **Resolution** | Number of bits (8-bit = 256 levels) |
| **Reference Voltage** | Maximum measurable voltage |
| **Conversion Time** | Time to complete one conversion |
| **Channels** | Number of input channels |
| **Interface** | Parallel, serial (SPI, I2C) |

---

## Example 1: ADC0809 Interfacing

### Application: Temperature Monitoring

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Interface ADC0809 (8-channel, 8-bit ADC) with 8051 for analog measurements.

### Hardware Setup

```
         8051                    ADC0809
       ┌──────┐              ┌─────────────────┐
       │      │              │                 │
       │  P0  ╞══╡╞══╞══╞══┤  D0-D7  (Data)  │
       │      │              │                 │
       │  P2.0├──────────────┤  ALE            │
       │  P2.1├──────────────┤  OE             │
       │  P2.2├──────────────┤  EOC            │
       │  P2.3├──────────────┤  START          │
       │  P2.4├──────────────┤  CLOCK          │
       │  P3.0├──────────────┤  ADD A          │
       │  P3.1├──────────────┤  ADD B          │
       │  P3.2├──────────────┤  ADD C          │
       │      │              │                 │
       │  GND ├──────────────┤  GND            │
       └──────┘              └─────────────────┘
                                      │
                    ┌─────────────────┤
                    │  Analog Inputs   │
                    │  IN0-IN7         │
                    │  (Sensors)       │
                    └─────────────────┘

Reference: VREF+ = 5V, VREF- = GND
```

### Component List
- ADC0809 IC
- Temperature sensor (LM35)
- Potentiometer (for testing)
- 5V reference voltage
- Clock circuit (or Timer 0 output)

### Source Code

```c
// ADC0809 Interfacing with 8051
// Read analog voltage from channel 0
// Display value on serial port

#include <reg51.h>

// ADC Control Pins
sbit ALE = P2^0;    // Address Latch Enable
sbit OE = P2^1;     // Output Enable
sbit EOC = P2^2;    // End of Conversion
sbit START = P2^3;  // Start Conversion
sbit CLOCK = P2^4;  // Clock Input

// Address Lines
sbit ADD_A = P3^0;
sbit ADD_B = P3^1;
sbit ADD_C = P3^2;

// ADC Data Port
#define ADC_DATA P0

unsigned char adc_value;

// Timer 0 for ADC clock (500kHz)
void timer0_init(void)
{
    TMOD = 0x02;     // Timer 0, Mode 2 (8-bit auto-reload)
    TH0 = 0xFD;      // ~500kHz @ 12MHz
    TL0 = 0xFD;
    TR0 = 1;         // Start timer
}

// Initialize ADC
void adc_init(void)
{
    timer0_init();   // Start clock

    ALE = 0;
    OE = 0;
    START = 0;
}

// Select ADC channel (0-7)
void select_channel(unsigned char channel)
{
    ADD_A = channel & 0x01;
    ADD_B = (channel >> 1) & 0x01;
    ADD_C = (channel >> 2) & 0x01;
}

// Start ADC conversion
void adc_start(unsigned char channel)
{
    select_channel(channel);

    ALE = 1;         // Latch address
    START = 1;       // Start conversion
    ALE = 0;
    START = 0;
}

// Wait for conversion to complete
unsigned char adc_read(void)
{
    while(EOC == 0); // Wait for EOC

    OE = 1;          // Enable output
    adc_value = ADC_DATA;
    OE = 0;          // Disable output

    return adc_value;
}

// Convert ADC value to voltage (mV)
unsigned int adc_to_voltage(unsigned char adc)
{
    // VREF = 5V, Resolution = 8-bit
    // Voltage(mV) = (ADC × 5000) / 255
    return ((unsigned int)adc * 5000) / 255;
}

void main(void)
{
    unsigned char channel = 0;
    unsigned int voltage_mv;

    adc_init();

    // Initialize serial for output
    // ... serial initialization code ...

    while(1) {
        adc_start(channel);
        adc_value = adc_read();
        voltage_mv = adc_to_voltage(adc_value);

        // Use voltage_mv here
        // Or display via serial port

        // Delay between readings
        for(unsigned int i = 0; i < 10000; i++);
    }
}
```

---

## Example 2: DAC with PWM

### Application: Analog Voltage Generation

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Generate analog voltage using PWM and RC low-pass filter.

### Hardware Setup

```
         8051                  RC Filter          Output
       ┌──────┐              ┌──────────┐       ┌─────┐
       │      │              │          │       │     │
       │  P1.0├───PWM───────┤  1kΩ     ├─┬─────┤  V  │
       │      │              │  Resistor│ │     │     │
       │      │              └──────────┘ │     └──┬──┘
       │      │                            │       │
       │  GND ├────────────────────────────┴───────┤
       └──────┘                            10µF    │
                                                  │
                                              Analog Out
                                             (0-5V smoothed)
```

### Component List
- Resistor: 1kΩ
- Capacitor: 10µF (electrolytic)
- Optional: Op-amp buffer

### Source Code

```c
// DAC using PWM + RC Filter
// Generate 0-5V analog output

#include <reg51.h>

sbit DAC_OUT = P1^0;

unsigned char dac_value = 0;
unsigned char pwm_counter = 0;

// Timer 0 interrupt for PWM
void timer0_isr(void) interrupt 1
{
    TH0 = 0xFF;
    TL0 = 0x00;

    pwm_counter++;

    // Generate PWM
    if(pwm_counter <= dac_value) {
        DAC_OUT = 1;
    } else {
        DAC_OUT = 0;
    }
}

// Set DAC output (0-255)
void dac_write(unsigned char value)
{
    dac_value = value;
}

// Generate sine wave (lookup table)
void generate_sine_wave(void)
{
    // 8-bit sine wave lookup table
    unsigned char sine_table[] = {
        128, 131, 134, 137, 140, 143, 146, 149,
        152, 155, 158, 162, 165, 167, 170, 173,
        176, 179, 182, 185, 188, 190, 193, 196,
        198, 201, 203, 206, 208, 211, 213, 215,
        218, 220, 222, 224, 226, 228, 230, 232,
        234, 235, 237, 238, 240, 241, 243, 244,
        245, 247, 248, 249, 250, 251, 252, 253,
        254, 254, 255, 255, 256, 256, 256, 256
        // ... (half table, mirror for other half)
    };

    unsigned char i;
    while(1) {
        for(i = 0; i < 64; i++) {
            dac_write(sine_table[i]);
            // Delay for frequency control
            for(unsigned int j = 0; j < 1000; j++);
        }
    }
}

// Generate ramp (sawtooth) wave
void generate_ramp(void)
{
    while(1) {
        for(unsigned char i = 0; i < 255; i++) {
            dac_write(i);
            for(unsigned int j = 0; j < 100; j++);
        }
    }
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

    // Generate waveform
    generate_sine_wave();
    // Or: generate_ramp();
}
```

---

## Example 3: Temperature Sensing

### Application: Digital Thermometer

**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Read temperature using LM35 sensor and ADC0809.

### LM35 Sensor
- **Output:** 10mV/°C
- **Range:** -55°C to +150°C
- **Accuracy:** ±0.5°C

### Calculation
```
At 25°C:
Voltage = 25 × 10mV = 250mV

ADC Value = (250mV / 5000mV) × 255 = 12.75 ≈ 13

Temperature(°C) = (ADC × 5000) / (255 × 10)
               = (ADC × 5000) / 2550
               ≈ ADC × 1.96
```

### Source Code

```c
// Digital Thermometer using LM35 + ADC0809
// Display temperature on serial port

#include <reg51.h>

// ADC control pins (same as previous example)
sbit ALE = P2^0;
sbit OE = P2^1;
sbit EOC = P2^2;
sbit START = P2^3;
sbit CLOCK = P2^4;

sbit ADD_A = P3^0;
sbit ADD_B = P3^1;
sbit ADD_C = P3^2;

#define ADC_DATA P0

unsigned char adc_read_temp(void)
{
    // Select channel 0
    ADD_A = 0;
    ADD_B = 0;
    ADD_C = 0;

    // Start conversion
    ALE = 1;
    START = 1;
    ALE = 0;
    START = 0;

    while(EOC == 0); // Wait

    OE = 1;
    unsigned char value = ADC_DATA;
    OE = 0;

    return value;
}

// Convert ADC to temperature (°C)
float adc_to_temperature(unsigned char adc)
{
    // LM35: 10mV/°C
    // VREF = 5V
    // Temp = (ADC / 255) × 5000mV / 10mV/°C
    return ((float)adc * 5000.0) / 255.0 / 10.0;
}

void main(void)
{
    unsigned char adc_val;
    float temperature;

    // Initialize ADC and serial...

    while(1) {
        adc_val = adc_read_temp();
        temperature = adc_to_temperature(adc_val);

        // Display temperature
        // printf("Temperature: %.1f C\r\n", temperature);

        // Delay
        for(unsigned int i = 0; i < 50000; i++);
    }
}
```

---

## DAC Methods

### Method 1: PWM + RC Filter
- **Pros:** Simple, no extra IC
- **Cons:** Limited resolution, needs filtering
- **Resolution:** Same as PWM (8-bit typical)

### Method 2: R-2R Ladder
```
         VCC
          │
         2kΩ    2kΩ    2kΩ
    D7 ───┤├───┬──┤├───┬──┤├───┬──→ Vout
          │   │   │   │   │
         1kΩ 1kΩ 1kΩ 1kΩ 1kΩ
          │   │   │   │   │
    D6 ───┴───┤├──┴───┤├──┴
              │       │
             GND     GND
```
- **Pros:** Parallel, fast
- **Cons:** Many resistors, impedance matching

### Method 3: DAC IC (e.g., DAC0832)
- **Pros:** Accurate, easy interface
- **Cons:** Extra component, cost

---

## Signal Conditioning

### 1. Voltage Divider
```
Vin ── R1 ──┬── Vout
             │
            R2
             │
            GND

Vout = Vin × R2 / (R1 + R2)
```

### 2. Non-Inverting Amplifier
```
          R2
Vin ──┬──┴─────┬── Vout
      │        │
     R1       |
      │        │
      └──┬─────┘
         │
        GND
    (Op-Amp)

Gain = 1 + R2/R1
```

### 3. Low-Pass Filter
```
           C
Vin ── R ──┬── Vout
           │
          GND

Cutoff = 1 / (2π × R × C)
```

---

## Troubleshooting

### ADC Issues

**Wrong Readings:**
- Check reference voltage
- Verify clock frequency
- Ensure proper grounding
- Check sensor connections

**No Conversion:**
- Verify START pulse
- Check EOC signal
- Confirm OE is enabled

**Noisy Readings:**
- Add decoupling capacitors
- Average multiple readings
- Check for electrical interference
- Use shielded cables

### DAC Issues

**No Output:**
- Check PWM generation
- Verify RC filter values
- Test with oscilloscope

**Poor Output Quality:**
- Adjust RC filter values
- Increase PWM frequency
- Add op-amp buffer

---

## Applications

### Input (ADC)
- Temperature sensing
- Light sensing (LDR)
- Pressure sensing
- Battery monitoring
- Current sensing
- Audio sampling

### Output (DAC)
- Audio generation
- Motor speed reference
- Display brightness
- Analog signal generation
- Control voltage output

---

## Component Selection

### ADC Chips

| Part | Resolution | Channels | Interface | Price |
|------|------------|----------|-----------|-------|
| ADC0809 | 8-bit | 8 | Parallel | Low |
| ADC0831 | 8-bit | 1 | Serial | Low |
| MCP3008 | 10-bit | 8 | SPI | Medium |
| ADS1115 | 16-bit | 4 | I2C | High |

### DAC Chips

| Part | Resolution | Channels | Interface | Price |
|------|------------|----------|-----------|-------|
| DAC0832 | 8-bit | 1 | Parallel | Low |
| MCP4725 | 12-bit | 1 | I2C | Medium |
| MCP4921 | 12-bit | 1 | SPI | Medium |

---

## Prerequisites

- ✅ [Basic I/O](../../01_Basic_IO/)
- ✅ [Timers](../../02_Timers/)
- ✅ [PWM](../01_PWM/)
- ✅ [Serial Port](../../04_Serial_Port/)

**Recommended Reading:**
- [PWM Generation](../01_PWM/)
- [Sensors](../04_Sensors/)

---

## Next Steps

After mastering ADC/DAC:
- [Sensors](../04_Sensors/) - Real-world sensor integration
- [Display](../05_Display/) - Visualize sensor data
- [Projects](../../../06_Projects/) - Data logger, weather station

---

**Difficulty:** ⭐⭐⭐⭐ Expert
**Time to Master:** 10-15 hours
**Hardware:** ADC0809, LM35 sensor, resistors, capacitors
