# Optocoupler

## Overview

An **optocoupler** (also called optoisolator or photocoupler) is an electronic component that transfers electrical signals between two isolated circuits using light. Optocouplers provide electrical isolation while allowing signal transmission, protecting sensitive circuits from high voltages, ground loops, and electrical noise.

## Basic Operation

### Light-Based Signal Transfer

**Principle**: Electrical signal converted to light, then back to electrical signal

**Input side**: LED (Light Emitting Diode) converts electrical signal to light

**Isolation barrier**: Physical gap between input and output (no electrical connection)

**Output side**: Photodetector (phototransistor, photodiode, or photo-TRIAC) converts light back to electrical signal

**Key advantage**: Complete electrical isolation between input and output circuits

### Isolation Voltage

The maximum voltage difference that can exist between input and output circuits

**Typical values**: 2.5kV, 5kV, 7.5kV

**Safety**: Protects low-voltage circuits from high-voltage transients and faults

### Current Transfer Ratio (CTR)

Ratio of output current to input current, expressed as percentage

```
CTR = (IC / IF) × 100%
```

Where:
- **IC** = Output collector current
- **IF** = Input LED forward current

**Typical values**: 50% to 200%

**Example**: CTR = 100%, IF = 10mA → IC = 10mA

## Types of Optocouplers

### Phototransistor Optocoupler

**Construction**: LED coupled to phototransistor (NPN)

**Characteristics**:
- Most common type
- Good current gain
- Moderate switching speed (1-10 μs)
- CTR typically 50-200%

**Applications**: Digital signal isolation, microcontroller I/O isolation, sensor isolation

### Photodarlington Optocoupler

**Construction**: LED coupled to Darlington transistor pair

**Characteristics**:
- Very high current gain
- Higher CTR (100-500%)
- Slower switching speed (10-100 μs)
- Higher saturation voltage

**Applications**: Relay drivers, high-gain applications, low-speed switching

### Photodiode Optocoupler

**Construction**: LED coupled to photodiode

**Characteristics**:
- Fast switching speed (< 1 μs)
- Low CTR (requires external amplifier)
- Linear response
- Wide bandwidth

**Applications**: High-speed data transmission, analog signal isolation, linear applications

### Photo-TRIAC Optocoupler

**Construction**: LED coupled to photo-triggered TRIAC or SCR

**Characteristics**:
- Can switch AC loads directly
- Zero-crossing detection available
- Isolation for AC circuits
- Bidirectional switching

**Applications**: AC load control, solid-state relays, TRIAC gate drivers, AC motor control

### Logic Gate Optocoupler

**Construction**: LED coupled to integrated photodetector and logic circuit

**Characteristics**:
- Digital output (TTL/CMOS compatible)
- Fast switching (1-10 Mbps)
- Built-in Schmitt trigger
- No external components needed

**Applications**: Digital communication, I²C/SPI isolation, high-speed data links

## Key Specifications

### Current Transfer Ratio (CTR)

Efficiency of light coupling from LED to photodetector

**Typical values**: 50% to 500% depending on type

**Degradation**: CTR decreases with LED aging and temperature

**Design consideration**: Design for minimum CTR over lifetime

### Isolation Voltage

Maximum voltage withstand between input and output

**Typical values**: 2.5kV, 3.75kV, 5kV, 7.5kV

**Test standards**: IEC 60747-5, UL 1577

**Safety**: Critical for protecting low-voltage circuits from high-voltage faults

### Forward Voltage (VF)

LED forward voltage drop

**Typical values**: 1.2V to 1.5V

**Input current calculation**: IF = (Vin - VF) / Rin

### Collector-Emitter Saturation Voltage (VCE(sat))

Output transistor voltage drop when fully on

**Typical values**: 0.2V to 0.5V

**Impact**: Affects output logic levels and power dissipation

### Switching Speed

Time to turn on and off

**Rise time (tr)**: Time for output to go from 10% to 90%

**Fall time (tf)**: Time for output to go from 90% to 10%

**Typical values**:
- Phototransistor: 1-10 μs
- Photodarlington: 10-100 μs
- Photodiode: < 1 μs
- Logic gate: 1-10 Mbps data rate

## Common Applications

### High-Voltage Isolation

**Purpose**: Protect low-voltage circuits from high-voltage transients

**Example**: Isolate microcontroller from AC mains circuits

**Safety**: Prevents electric shock and equipment damage

### Ground Loop Elimination

**Problem**: Different ground potentials cause noise and interference

**Solution**: Optocoupler breaks electrical connection while maintaining signal transfer

**Applications**: Audio systems, industrial control, data acquisition

### Level Shifting

**Purpose**: Interface between circuits with different voltage levels

**Example**: 3.3V microcontroller controlling 5V or 12V circuit

**Advantage**: No voltage dividers or level shifters needed

### Noise Immunity

**Purpose**: Isolate sensitive circuits from noisy environments

**Applications**: Industrial sensors, motor control feedback, switching power supplies

**Benefit**: Common-mode noise rejection

### AC Load Control

**Photo-TRIAC optocouplers**: Drive TRIACs for AC switching

**Applications**: Light dimmers, heater control, AC motor speed control

**Zero-crossing**: Reduces EMI by switching at voltage zero-crossing

## Practical Considerations

### Input Circuit Design

**Current limiting resistor**: Required to limit LED current

**Calculation**:
```
Rin = (Vin - VF) / IF
```

**Example**: 5V input, VF = 1.2V, IF = 10mA
- Rin = (5 - 1.2) / 0.01 = 380Ω (use 390Ω standard value)

**Typical IF values**: 5mA to 20mA

### Output Circuit Design

**Pull-up resistor**: Required for phototransistor output

**Calculation**:
```
Rload = (Vcc - VOL) / IC
```

Where IC = IF × CTR

**Example**: Vcc = 5V, IF = 10mA, CTR = 100%
- IC = 10mA × 1.0 = 10mA
- Rload = (5 - 0.3) / 0.01 = 470Ω

**Speed consideration**: Lower Rload = faster switching (less RC delay)

### CTR Degradation

**LED aging**: CTR decreases over time (typically 20-30% over lifetime)

**Temperature**: CTR decreases at high temperatures

**Design margin**: Design for minimum CTR (typically 50% of initial value)

**Example**: If initial CTR = 100%, design for CTR = 50%

### Switching Speed Optimization

**Faster switching techniques**:
- Use lower pull-up resistance (increases power consumption)
- Add speed-up capacitor across pull-up resistor (10-100pF)
- Use photodiode optocoupler with external amplifier
- Use logic gate optocoupler for high-speed applications

**Trade-off**: Speed vs power consumption

### PCB Layout

**Creepage and clearance**: Maintain minimum spacing for isolation voltage

**Typical requirements**: 4mm for 2.5kV, 8mm for 5kV

**Slot in PCB**: Cut slot between input and output for better isolation

**Ground planes**: Separate input and output ground planes

### Common Mistakes to Avoid

- **No current limiting resistor**: Destroys LED from overcurrent
- **Insufficient CTR margin**: Circuit fails as optocoupler ages
- **Wrong isolation voltage rating**: Safety hazard in high-voltage applications
- **No pull-up resistor**: Phototransistor output doesn't work properly
- **Exceeding LED current rating**: Accelerates aging and reduces lifetime
- **Poor PCB layout**: Reduces effective isolation voltage
- **Ignoring switching speed**: Slow optocoupler causes signal distortion
- **Connecting grounds together**: Defeats purpose of isolation

## Summary

Optocouplers provide electrical isolation between circuits while allowing signal transmission through light. Understanding optocoupler types, specifications, and proper circuit design is essential for safe and reliable isolation in high-voltage, noisy, or ground-loop-prone applications.

**Key Takeaways**:
- Optocouplers transfer signals using light (LED to photodetector)
- Complete electrical isolation between input and output circuits
- Isolation voltage: 2.5kV to 7.5kV typical
- Current Transfer Ratio (CTR): IC/IF ratio, typically 50-200%
- Types: Phototransistor (most common), photodarlington (high gain), photodiode (fast), photo-TRIAC (AC switching), logic gate (digital)
- Input circuit: Requires current limiting resistor, Rin = (Vin - VF) / IF
- Output circuit: Requires pull-up resistor for phototransistor types
- CTR degrades over time (20-30%), design for minimum CTR
- Switching speed: 1-10 μs typical for phototransistor
- Applications: High-voltage isolation, ground loop elimination, level shifting, noise immunity
- PCB layout: Maintain creepage/clearance distances (4mm for 2.5kV, 8mm for 5kV)
- Design margin: Account for CTR degradation and temperature effects
- Photo-TRIAC types: Direct AC load control with zero-crossing detection
- Logic gate types: Fast digital isolation (1-10 Mbps)

Proper optocoupler selection based on isolation voltage, CTR, switching speed, and application requirements ensures safe and reliable electrical isolation in power control, signal processing, and communication applications.

## References

- Optocoupler construction and operating principles
- IEC 60747-5: Optocouplers for basic and reinforced insulation
- UL 1577: Optical isolators standard
- Optocoupler manufacturer datasheets and application notes
- PCB creepage and clearance requirements for isolation


