# Potentiometer

## Overview

A **potentiometer** (often called "pot") is a three-terminal variable resistor that allows manual adjustment of resistance. Potentiometers are used for voltage division, signal attenuation, position sensing, and user input controls in electronic circuits.

## Basic Operation

### Construction

**Resistive element**: Fixed resistance track (carbon, cermet, or wire-wound)

**Wiper**: Movable contact that slides along the resistive element

**Three terminals**:
- **Terminal 1**: One end of resistive element
- **Terminal 2 (Wiper)**: Movable contact
- **Terminal 3**: Other end of resistive element

### Variable Resistor vs Voltage Divider

**Variable resistor (Rheostat)**: Use two terminals (wiper + one end)
- Adjustable resistance from 0 to maximum value
- Controls current in circuit

**Voltage divider**: Use all three terminals
- Input voltage across terminals 1 and 3
- Output voltage from wiper (terminal 2)
- Adjustable voltage from 0 to input voltage

### Voltage Division Formula

```
Vout = Vin × (R2 / (R1 + R2))
```

Where R1 and R2 are the resistances on either side of the wiper.

## Types of Potentiometers

### Rotary Potentiometers

**Construction**: Circular resistive track with rotating wiper

**Characteristics**:
- Most common type
- Rotation typically 270° to 300°
- Single-turn or multi-turn (10-turn, 25-turn)
- Panel mount or PCB mount

**Applications**: Volume controls, tone controls, user adjustable settings

### Linear (Slide) Potentiometers

**Construction**: Straight resistive track with sliding wiper

**Characteristics**:
- Linear motion instead of rotary
- Visual indication of position
- Typical travel: 30mm to 100mm
- Single or dual gang available

**Applications**: Audio mixers, graphic equalizers, lighting controls

### Trimmer Potentiometers (Trimpots)

**Construction**: Miniature potentiometer for PCB mounting

**Characteristics**:
- Small size (3mm to 10mm)
- Single-turn or multi-turn
- Adjusted with screwdriver
- Not intended for frequent adjustment
- Lower power rating than panel-mount pots

**Applications**: Circuit calibration, bias adjustment, fine-tuning

### Digital Potentiometers

**Construction**: Electronic resistance network controlled by digital interface

**Characteristics**:
- No mechanical parts (solid-state)
- Controlled via I²C, SPI, or up/down pins
- Discrete steps (64, 128, 256 positions typical)
- Non-volatile memory option
- Lower power handling than mechanical pots

**Applications**: Microcontroller-controlled gain, programmable filters, digital audio

## Key Specifications

### Resistance Value

Total resistance between terminals 1 and 3

**Common values**: 1kΩ, 10kΩ, 50kΩ, 100kΩ, 1MΩ

**Selection**: Choose based on circuit impedance and current requirements

### Taper (Resistance Law)

Relationship between wiper position and resistance

**Linear taper (B taper)**:
- Resistance changes proportionally with position
- 50% rotation = 50% resistance
- Used for general applications, voltage division

**Logarithmic taper (A taper)**:
- Resistance changes logarithmically with position
- Used for audio volume controls (matches human hearing perception)
- 50% rotation ≈ 10-20% resistance

**Anti-logarithmic (C taper)**: Inverse of logarithmic, less common

### Power Rating

Maximum power dissipation

**Typical values**: 0.1W, 0.25W, 0.5W, 1W, 2W

**Calculation**: P = V² / R or P = I² × R

**Derating**: Reduce power rating at high temperatures

### Tolerance

Accuracy of total resistance value

**Typical values**: ±5%, ±10%, ±20%

**Impact**: Affects precision in voltage divider applications

### Rotational Life

Number of rotations before failure

**Typical values**: 10,000 to 100,000 cycles for standard pots

**Factors**: Wiper material, resistive element quality, operating conditions

### Temperature Coefficient

Change in resistance with temperature

**Typical values**: ±100 to ±500 ppm/°C

## Common Applications

### Volume and Tone Controls

**Audio equipment**: Adjust volume, bass, treble, balance

**Taper selection**: Logarithmic taper for volume (matches human hearing)

**Typical values**: 10kΩ to 100kΩ

### Voltage Division and Reference

**Adjustable voltage source**: Create variable reference voltage

**Circuit**: Potentiometer across power supply, wiper provides adjustable output

**Applications**: Threshold adjustment, bias voltage setting

### User Input Controls

**Analog input**: Provide variable input to microcontroller ADC

**Position sensing**: Detect knob or slider position

**Applications**: Brightness control, speed control, parameter adjustment

### Sensor Calibration

**Offset adjustment**: Trim sensor output to match reference

**Gain adjustment**: Scale sensor signal to desired range

**Applications**: Temperature sensor calibration, pressure sensor trimming

### Current Limiting

**Variable resistor mode**: Use two terminals as adjustable resistance

**Applications**: LED current adjustment, motor speed control (simple applications)

**Note**: Power dissipation must be considered

## Practical Considerations

### Loading Effects

**Problem**: Load resistance affects voltage divider output

**Output voltage with load**:
```
Vout = Vin × (R2 || RL) / (R1 + (R2 || RL))
```

Where R2 || RL = (R2 × RL) / (R2 + RL)

**Solution**: Choose potentiometer resistance much lower than load resistance (typically 10× lower)

**Example**: For 100kΩ load, use 10kΩ potentiometer

### Wiper Noise

**Problem**: Mechanical wear causes intermittent contact and noise

**Symptoms**: Crackling sound in audio, erratic readings

**Solutions**:
- Use conductive plastic track (lower noise than carbon)
- Clean with contact cleaner
- Replace worn potentiometers
- Use digital potentiometers for critical applications

### Power Dissipation

**Calculate power**: P = V² / R (at maximum voltage position)

**Example**: 12V across 1kΩ pot: P = 144 / 1000 = 0.144W

**Safety margin**: Use potentiometer rated at least 2× calculated power

### Mechanical Considerations

**Mounting**: Ensure secure mounting to prevent rotation or movement

**Shaft coupling**: Use flexible couplers for panel-mount pots to avoid binding

**Dust and moisture**: Use sealed potentiometers in harsh environments

**Vibration**: Can cause wiper bounce and intermittent contact

### Microcontroller Interface

**ADC connection**: Connect wiper to ADC input, terminals to Vcc and GND

**Input impedance**: Most MCU ADCs have high input impedance (>10MΩ), minimal loading

**Filtering**: Add capacitor (0.1μF) across potentiometer to reduce noise

**Software**: Implement averaging or filtering in code to smooth readings

### Common Mistakes to Avoid

- **Exceeding power rating**: Causes overheating and failure
- **Wrong taper selection**: Linear taper for audio volume sounds unnatural
- **High impedance with low load**: Causes loading effects and inaccurate voltage division
- **No consideration for wear**: Potentiometers have limited lifespan
- **Ignoring wiper resistance**: Can affect precision in some applications
- **Poor mechanical mounting**: Causes premature failure

## Summary

Potentiometers are three-terminal variable resistors that enable manual adjustment of resistance, voltage, or current in electronic circuits. Understanding potentiometer types, specifications, and proper application is essential for user interface design and circuit calibration.

**Key Takeaways**:
- Three terminals: two ends of resistive element, one wiper (movable contact)
- Variable resistor mode: Use two terminals (wiper + one end)
- Voltage divider mode: Use all three terminals for adjustable voltage output
- Voltage division: Vout = Vin × (R2 / (R1 + R2))
- Types: Rotary (most common), linear/slide, trimmer, digital
- Taper: Linear (B) for general use, logarithmic (A) for audio volume
- Common values: 1kΩ, 10kΩ, 50kΩ, 100kΩ, 1MΩ
- Power rating: 0.1W to 2W typical, derate at high temperatures
- Loading effects: Choose pot resistance 10× lower than load resistance
- Rotational life: 10,000 to 100,000 cycles for mechanical pots
- Digital pots: Solid-state, microcontroller-controlled, no mechanical wear
- Wiper noise: Use conductive plastic track for lower noise
- Audio applications: Use logarithmic taper for volume controls
- MCU interface: Connect wiper to ADC, add filtering capacitor

Proper potentiometer selection based on resistance value, taper, power rating, and application requirements ensures reliable operation in user controls, voltage division, and calibration applications.

## References

- Potentiometer construction and operating principles
- Resistor networks and voltage divider theory
- Audio taper curves and human hearing perception
- Digital potentiometer datasheets and specifications


