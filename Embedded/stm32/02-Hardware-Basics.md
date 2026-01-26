# Chapter 2: Hardware Basics

## 2.1 STM32 Minimal System

A minimal system is the basic circuit required for an STM32 to operate. Understanding this is crucial for both using development boards and designing custom hardware.

### Essential Components

```
        VDD (3.3V)
         │
    ┌────┴────┐
    │  100nF  │  Decoupling capacitors
    │  10uF   │  (near each VDD pin)
    └────┬────┘
         │
    ┌────▼────────────────┐
    │                     │
    │   STM32 MCU         │
    │                     │
    │  NRST ──────────┐   │
    │                 │   │
    │  BOOT0 ─────────┼───┤
    │                 │   │
    │  OSC_IN  ───────┼───┤
    │  OSC_OUT ───────┼───┤
    │                 │   │
    └─────────────────┼───┘
                      │
                     GND
```

### 1. Power Supply

**Voltage Requirements**
- VDD: 2.0V - 3.6V (typically 3.3V)
- VDDA: Analog power supply (same as VDD or separate for better ADC performance)
- VBAT: Battery backup for RTC (optional)

**Decoupling Capacitors**
- Place 100nF ceramic capacitor near each VDD pin
- Add 10µF tantalum/electrolytic capacitor for bulk decoupling
- Keep traces short to minimize inductance

**Power Supply Circuit Example**
```
5V ──┬──[LDO 3.3V]──┬──[10µF]──┬── VDD
     │              │          │
     │              └─[100nF]──┤
     │                         │
    GND ────────────────────────┴── GND
```

### 2. Reset Circuit

**NRST Pin**
- Active low reset
- Internal pull-up resistor (30-50 kΩ)
- External 10kΩ pull-up recommended for noise immunity

**Basic Reset Circuit**
```
VDD (3.3V)
  │
 [10kΩ]
  │
  ├──── NRST
  │
 [100nF]──[Button]
  │        │
 GND ──────┘
```

**Reset Sources**
- External reset (NRST pin)
- Power-on reset (POR)
- Brown-out reset (BOR)
- Software reset
- Watchdog reset

### 3. Boot Configuration

**BOOT0 Pin**
- Determines boot mode at startup
- Sampled at reset

**Boot Modes**

| BOOT0 | Boot Mode | Memory Location |
|-------|-----------|-----------------|
| 0 (GND) | Main Flash | 0x0800 0000 |
| 1 (VDD) | System Memory | 0x1FFF 0000 (bootloader) |

**Typical Configuration**
```
VDD
 │
[10kΩ]
 │
 ├──[Switch]── BOOT0
 │
GND
```

For normal operation: BOOT0 = 0 (connected to GND through 10kΩ)
For bootloader/programming: BOOT0 = 1 (connected to VDD)

### 4. Clock Source

**External Crystal Oscillator (HSE)**

Most STM32 applications use an external crystal for accurate timing.

**Common Crystal Values**
- 8 MHz (most common)
- 16 MHz
- 25 MHz (for Ethernet applications)

**Crystal Circuit**
```
        C1
OSC_IN ─┤├─┬─[Crystal]─┬─┤├─ OSC_OUT
         │  │           │  │
        GND │           │ GND
            └─[1MΩ]─────┘
               C2
```

**Load Capacitor Calculation**
```
CL = (C1 × C2) / (C1 + C2) + Cstray

Typical values:
- Crystal load capacitance (CL): 10-20 pF
- C1 = C2 = 2 × (CL - Cstray)
- Cstray ≈ 2-5 pF
- Result: C1 = C2 ≈ 15-22 pF
```

**Alternative: Ceramic Resonator**
- Integrated load capacitors
- Less accurate than crystal
- Cheaper and smaller
- Suitable for non-critical timing applications

## 2.2 Pin Configuration

### Pin Types

**GPIO (General Purpose I/O)**
- Configurable as input or output
- Multiple modes: push-pull, open-drain, pull-up, pull-down
- Maximum current: typically 25 mA per pin

**Alternate Functions**
- Pins can be configured for peripheral functions
- Examples: UART TX/RX, SPI MOSI/MISO, I2C SDA/SCL
- Refer to datasheet for pin mapping

**Analog Pins**
- ADC input channels
- DAC output channels
- Typically labeled as PA0-PA7, etc.

**Special Pins**
- NRST: Reset
- BOOT0: Boot mode selection
- VBAT: RTC backup battery
- OSC_IN/OSC_OUT: Crystal oscillator

### Pin Electrical Characteristics

**Input Modes**
- Floating: High impedance, susceptible to noise
- Pull-up: Internal ~40kΩ resistor to VDD
- Pull-down: Internal ~40kΩ resistor to GND
- Analog: For ADC/DAC, disable digital input

**Output Modes**
- Push-pull: Can drive both high and low
- Open-drain: Can only pull low, needs external pull-up
- Maximum speed: 2 MHz, 10 MHz, 50 MHz (affects slew rate)

**Voltage Levels**
- VIL (Input Low): < 0.3 × VDD
- VIH (Input High): > 0.7 × VDD
- VOL (Output Low): < 0.4V
- VOH (Output High): > VDD - 0.4V

### 5V Tolerance

Some STM32 pins are 5V-tolerant (marked as FT in datasheet):
- Can accept 5V input signals
- Useful for interfacing with 5V logic
- Check datasheet for specific pins
- Output is still 3.3V

## 2.3 Power Considerations

### Power Consumption

**Operating Modes**
- Run mode: Full operation, all peripherals available
- Sleep mode: CPU stopped, peripherals running
- Stop mode: Most clocks stopped, SRAM retained
- Standby mode: Lowest power, only RTC and backup registers

**Typical Current Consumption (STM32F103)**
- Run mode @ 72 MHz: ~30 mA
- Sleep mode: ~15 mA
- Stop mode: ~20 µA
- Standby mode: ~2 µA

### Power Supply Design

**Linear Regulator (LDO)**
```
5V Input ──[LDO]── 3.3V Output
           (e.g., AMS1117-3.3)
```
- Simple, low cost
- Inefficient (heat dissipation)
- Suitable for low-power applications

**Switching Regulator**
- Higher efficiency (>85%)
- More complex circuit
- Suitable for battery-powered devices
- May introduce noise (use filtering)

**Battery Power**
- Li-ion: 3.7V nominal (needs regulation)
- 3× AA/AAA: 4.5V (needs regulation)
- CR2032 coin cell: 3V (direct connection possible with low-power STM32L series)

## 2.4 Debug Interface

### SWD (Serial Wire Debug)

Modern STM32 devices use SWD for programming and debugging.

**SWD Pins**
- SWDIO: Data I/O
- SWCLK: Clock
- GND: Ground
- VDD: Power (for level detection)
- NRST: Reset (optional)

**Connection to ST-Link**
```
ST-Link          STM32
───────          ─────
SWDIO ────────── SWDIO (PA13)
SWCLK ────────── SWCLK (PA14)
GND ──────────── GND
3.3V ────────── VDD
NRST ────────── NRST
```

**Important Notes**
- PA13 and PA14 are SWD pins by default
- Can be reconfigured as GPIO (but lose debug capability)
- Always keep SWD enabled during development
- Use external pull-ups if reconfiguring

### JTAG Interface

Older debugging interface, requires more pins:
- TMS, TCK, TDI, TDO, TRST
- More comprehensive debugging features
- Most modern development uses SWD instead

## 2.5 Peripheral Connections

### LED Connection

**Active High (Common)**
```
GPIO ──[330Ω]──[LED]──GND
```

**Active Low**
```
VDD ──[330Ω]──[LED]──GPIO
```

**Current Limiting Resistor**
```
R = (VDD - VLED) / ILED
Example: (3.3V - 2.0V) / 10mA = 130Ω (use 150Ω or 330Ω)
```

### Button Connection

**Pull-up Configuration**
```
VDD
 │
[10kΩ]
 │
 ├──── GPIO (Input)
 │
[Button]
 │
GND
```
- Idle: GPIO reads HIGH
- Pressed: GPIO reads LOW
- Internal pull-up can be used (saves external resistor)

**Debouncing**
- Add 100nF capacitor parallel to button
- Or use software debouncing

### UART Connection

**Direct Connection (3.3V devices)**
```
STM32 TX ────── RX Device
STM32 RX ────── TX Device
GND ──────────── GND
```

**Level Shifting (5V devices)**
- Use level shifter IC (e.g., TXS0108E)
- Or voltage divider for RX (not recommended for production)

### I2C Pull-up Resistors

```
VDD (3.3V)
 │     │
[R]   [R]  (typically 4.7kΩ)
 │     │
SDA   SCL
```

**Resistor Selection**
- Standard mode (100 kHz): 4.7kΩ - 10kΩ
- Fast mode (400 kHz): 2.2kΩ - 4.7kΩ
- Consider bus capacitance and number of devices

## 2.6 PCB Design Guidelines

### Layout Best Practices

1. **Power Supply**
   - Place decoupling capacitors close to VDD pins
   - Use wide traces for power and ground
   - Consider power plane for complex designs

2. **Crystal Oscillator**
   - Keep traces short and symmetric
   - Place close to MCU
   - Avoid routing signals underneath
   - Ground plane underneath recommended

3. **Signal Integrity**
   - Keep high-speed signals short
   - Use ground plane for return path
   - Separate analog and digital grounds (connect at one point)

4. **Debug Interface**
   - Provide easy access to SWD pins
   - Consider adding debug header
   - Keep NRST accessible

### Common Mistakes to Avoid

- Insufficient decoupling capacitors
- Long crystal traces
- Floating input pins
- Inadequate current capacity for power traces
- No pull-up on NRST
- Forgetting boot mode configuration

## Next Steps

Proceed to [Chapter 3: Development Environment Setup](03-Development-Environment-Setup.md) to learn how to install and configure your development tools.
