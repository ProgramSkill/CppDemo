# Resistor

## Overview

A **resistor** is a passive two-terminal electrical component that implements electrical resistance as a circuit element. Resistors are one of the most fundamental and commonly used components in electronic circuits, serving to reduce current flow, adjust signal levels, divide voltages, and dissipate power.

## What is Resistance?

**Resistance** is the opposition to the flow of electric current through a conductor. It is measured in **Ohms (Ω)**, named after German physicist Georg Simon Ohm.

### Ohm's Law

The relationship between voltage (V), current (I), and resistance (R) is defined by Ohm's Law:

```
V = I × R
```

Where:
- **V** = Voltage (Volts)
- **I** = Current (Amperes)
- **R** = Resistance (Ohms)

From this formula, we can derive:
- `I = V / R` (Current equals voltage divided by resistance)
- `R = V / I` (Resistance equals voltage divided by current)

## Physical Construction

Resistors are typically made from materials with specific resistive properties:

- **Carbon composition**: Mixed carbon particles with a binding resin
- **Carbon film**: Thin carbon film deposited on a ceramic substrate
- **Metal film**: Thin metal layer on a ceramic core (more precise)
- **Wire-wound**: Resistance wire wrapped around an insulating core (high power)
- **Metal oxide**: Metal oxide film on a ceramic substrate (high temperature)

## Resistor Color Code

Most through-hole resistors use a color band system to indicate their resistance value and tolerance.

### 4-Band Resistor Code

| Band | Position | Meaning |
|------|----------|---------|
| 1st  | Digit 1  | First significant digit |
| 2nd  | Digit 2  | Second significant digit |
| 3rd  | Multiplier | Number of zeros (power of 10) |
| 4th  | Tolerance | Accuracy percentage |

### Color Values

| Color  | Digit | Multiplier | Tolerance |
|--------|-------|------------|-----------|
| Black  | 0     | ×1         | -         |
| Brown  | 1     | ×10        | ±1%       |
| Red    | 2     | ×100       | ±2%       |
| Orange | 3     | ×1K        | -         |
| Yellow | 4     | ×10K       | -         |
| Green  | 5     | ×100K      | ±0.5%     |
| Blue   | 6     | ×1M        | ±0.25%    |
| Violet | 7     | ×10M       | ±0.1%     |
| Gray   | 8     | ×100M      | ±0.05%    |
| White  | 9     | ×1G        | -         |
| Gold   | -     | ×0.1       | ±5%       |
| Silver | -     | ×0.01      | ±10%      |

### Example

**Brown-Black-Red-Gold**:
- Brown (1) - Black (0) - Red (×100) - Gold (±5%)
- Resistance: 10 × 100 = **1,000Ω = 1kΩ ±5%**

### 5-Band and 6-Band Resistors

Precision resistors use 5 or 6 bands:
- **5-Band**: 3 digits + multiplier + tolerance
- **6-Band**: 3 digits + multiplier + tolerance + temperature coefficient

## Types of Resistors

### Fixed Resistors

**Carbon Composition**
- Inexpensive, general purpose
- High tolerance (±5% to ±20%)
- Good for high-energy pulses

**Carbon Film**
- Better tolerance (±2% to ±5%)
- Lower noise than carbon composition
- Common in consumer electronics

**Metal Film**
- High precision (±0.1% to ±1%)
- Low temperature coefficient
- Low noise, stable
- Used in precision circuits

**Metal Oxide**
- High temperature tolerance
- Good stability
- Used in high-temperature applications

**Wire-Wound**
- High power ratings (up to hundreds of watts)
- Very low resistance values possible
- Used in power supplies and industrial equipment

### Variable Resistors

**Potentiometer (Pot)**
- Three terminals
- Adjustable voltage divider
- Used for volume controls, tuning, calibration

**Rheostat**
- Two terminals (variable resistor)
- Controls current in a circuit
- Used for dimmer switches, motor speed control

**Trimmer (Trimpot)**
- Small potentiometer for infrequent adjustment
- Used for circuit calibration
- Adjusted with screwdriver

### Special Purpose Resistors

**Thermistor**
- Resistance changes with temperature
- NTC (Negative Temperature Coefficient): resistance decreases as temperature increases
- PTC (Positive Temperature Coefficient): resistance increases as temperature increases
- Used in temperature sensing and compensation

**Photoresistor (LDR)**
- Resistance changes with light intensity
- Used in light-sensing applications
- Common in automatic lighting systems

**Varistor (VDR)**
- Voltage-dependent resistor
- Protects circuits from voltage spikes
- Used in surge protection

## Power Rating

The **power rating** of a resistor indicates the maximum power it can safely dissipate as heat without damage. Power is calculated using:

```
P = V × I = I² × R = V² / R
```

Where:
- **P** = Power (Watts)
- **V** = Voltage across the resistor
- **I** = Current through the resistor
- **R** = Resistance

### Common Power Ratings

| Type | Power Rating | Typical Use |
|------|--------------|-------------|
| SMD (Surface Mount) | 1/16W, 1/10W, 1/8W, 1/4W | PCB circuits, low power |
| Through-hole | 1/8W, 1/4W, 1/2W, 1W, 2W | General electronics |
| Power resistors | 5W, 10W, 25W, 50W+ | Power supplies, motor control |

**Important**: Always use a resistor with a power rating at least 2× the expected power dissipation for safety and reliability.

### Example Calculation

If a 100Ω resistor has 5V across it:
- Current: I = V/R = 5V / 100Ω = 0.05A (50mA)
- Power: P = V²/R = 25 / 100 = 0.25W (250mW)
- **Recommended resistor**: At least 1/2W rating (2× safety margin)

## Resistor Configurations

### Series Connection

When resistors are connected in series (end-to-end), the total resistance is the sum of individual resistances:

```
R_total = R1 + R2 + R3 + ... + Rn
```

**Characteristics**:
- Same current flows through all resistors
- Voltage divides across resistors proportionally
- Total resistance increases

**Example**: 100Ω + 220Ω + 330Ω = 650Ω

### Parallel Connection

When resistors are connected in parallel (side-by-side), the total resistance is calculated using:

```
1/R_total = 1/R1 + 1/R2 + 1/R3 + ... + 1/Rn
```

For two resistors in parallel:
```
R_total = (R1 × R2) / (R1 + R2)
```

**Characteristics**:
- Same voltage across all resistors
- Current divides among resistors
- Total resistance decreases (always less than the smallest resistor)

**Example**: Two 100Ω resistors in parallel = (100 × 100) / (100 + 100) = 50Ω

## Standard Resistor Values (E-Series)

Resistors are manufactured in standardized values based on E-series (IEC 60063):

### E12 Series (±10% tolerance)
12 values per decade: 10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82

### E24 Series (±5% tolerance)
24 values per decade: 10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91

### E96 Series (±1% tolerance)
96 values per decade (used for precision resistors)

**Note**: These base values repeat for each decade (×1, ×10, ×100, ×1K, ×10K, etc.)

**Examples**:
- E12: 10Ω, 100Ω, 1kΩ, 10kΩ, 100kΩ, 1MΩ
- E24: 11Ω, 110Ω, 1.1kΩ, 11kΩ, 110kΩ, 1.1MΩ

## Surface Mount Device (SMD) Resistors

SMD resistors are compact components designed for automated PCB assembly. They use numerical codes instead of color bands.

### SMD Package Sizes

| Size Code | Dimensions (mm) | Dimensions (inches) | Typical Power |
|-----------|-----------------|---------------------|---------------|
| 0201 | 0.6 × 0.3 | 0.02 × 0.01 | 1/20W |
| 0402 | 1.0 × 0.5 | 0.04 × 0.02 | 1/16W |
| 0603 | 1.6 × 0.8 | 0.06 × 0.03 | 1/10W |
| 0805 | 2.0 × 1.25 | 0.08 × 0.05 | 1/8W |
| 1206 | 3.2 × 1.6 | 0.12 × 0.06 | 1/4W |
| 1210 | 3.2 × 2.5 | 0.12 × 0.10 | 1/2W |
| 2512 | 6.4 × 3.2 | 0.25 × 0.12 | 1W |

### SMD Marking Codes

**3-Digit Code** (Standard tolerance):
- First 2 digits: significant figures
- Last digit: multiplier (number of zeros)
- Example: **473** = 47 × 10³ = 47kΩ

**4-Digit Code** (Precision, ±1%):
- First 3 digits: significant figures
- Last digit: multiplier
- Example: **1002** = 100 × 10² = 10kΩ

**Special Cases**:
- **R** indicates decimal point: **4R7** = 4.7Ω
- **0** or **000** = 0Ω (jumper)

## Common Applications

### Current Limiting
Protect LEDs and other components from excessive current.
```
R = (V_supply - V_component) / I_desired
```
Example: 5V supply, LED with 2V forward voltage, 20mA desired current:
R = (5V - 2V) / 0.02A = 150Ω

### Voltage Division
Create a specific voltage from a higher voltage source.
```
V_out = V_in × (R2 / (R1 + R2))
```

### Pull-up/Pull-down Resistors
- **Pull-up**: Connects signal line to positive voltage (typically 1kΩ - 10kΩ)
- **Pull-down**: Connects signal line to ground
- Used in digital circuits to define default logic states

### Biasing
Set operating points for transistors and amplifiers.

### Timing Circuits
Combined with capacitors to create RC time constants for delays and oscillators.
```
τ = R × C (time constant in seconds)
```

### Signal Conditioning
- Impedance matching
- Gain control in amplifiers
- Filter networks (low-pass, high-pass, band-pass)

## Practical Considerations

### Temperature Coefficient
Resistance changes with temperature. The **temperature coefficient** (TC) is measured in ppm/°C (parts per million per degree Celsius).

- Carbon film: ±200 to ±500 ppm/°C
- Metal film: ±50 to ±100 ppm/°C
- Wire-wound: ±20 to ±50 ppm/°C

### Tolerance Selection
Choose tolerance based on application requirements:
- **±20%**: Non-critical applications (rarely used)
- **±10%**: General purpose (E12 series)
- **±5%**: Standard precision (E24 series)
- **±1%**: High precision (E96 series)
- **±0.1%**: Ultra-precision (measurement, calibration)

### Noise
Resistors generate thermal (Johnson) noise. Lower resistance values and lower temperatures produce less noise. Use metal film resistors for low-noise applications.

### Selection Guide

When choosing a resistor, consider:

1. **Resistance value**: Calculate using Ohm's Law
2. **Power rating**: At least 2× expected power dissipation
3. **Tolerance**: Based on precision requirements
4. **Temperature coefficient**: For temperature-sensitive applications
5. **Package type**: Through-hole vs SMD (based on assembly method)
6. **Physical size**: Ensure adequate heat dissipation
7. **Cost**: Balance performance with budget constraints

### Common Mistakes to Avoid

- **Insufficient power rating**: Leads to overheating and failure
- **Wrong resistance value**: Double-check color code or SMD marking
- **Ignoring tolerance**: Can cause circuit malfunction in precision applications
- **Poor heat dissipation**: Ensure adequate spacing and airflow
- **Using carbon composition for precision**: Use metal film instead
- **Exceeding voltage rating**: High-value resistors have maximum voltage limits

## Summary

Resistors are fundamental passive components that control current flow and voltage levels in electronic circuits. Understanding their characteristics—resistance value, power rating, tolerance, and temperature coefficient—is essential for proper circuit design.

**Key Takeaways**:
- Use Ohm's Law (V = I × R) for basic calculations
- Always select power ratings with adequate safety margin (2× minimum)
- Choose appropriate tolerance based on application precision requirements
- Understand color codes (through-hole) and numerical codes (SMD)
- Consider series/parallel combinations to achieve non-standard values
- Match resistor type to application (metal film for precision, wire-wound for power)

Proper resistor selection ensures reliable circuit operation, prevents component failure, and optimizes overall system performance.

## References

- IEC 60063: Preferred number series for resistors and capacitors
- Ohm's Law and basic circuit theory
- Manufacturer datasheets for specific component specifications

