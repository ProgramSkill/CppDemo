# Thermistor

## Overview

A **thermistor** (thermal resistor) is a temperature-sensitive resistor whose resistance changes significantly with temperature. Thermistors are widely used for temperature sensing, measurement, and control applications due to their high sensitivity, fast response, and low cost.

## Basic Operation

### Temperature-Dependent Resistance

**Principle**: Resistance varies predictably with temperature changes

**Physical mechanism**: Semiconductor material properties change with thermal energy

**High sensitivity**: Large resistance change per degree of temperature change

**Key advantage**: More sensitive than RTDs (Resistance Temperature Detectors) or thermocouples

### Types of Thermistors

**NTC (Negative Temperature Coefficient)**:
- Resistance decreases as temperature increases
- Most common type for temperature sensing
- Exponential resistance-temperature relationship

**PTC (Positive Temperature Coefficient)**:
- Resistance increases as temperature increases
- Used for overcurrent protection and self-regulating heaters
- Sharp resistance increase at Curie temperature

## NTC Thermistor Characteristics

### Resistance-Temperature Relationship

**Steinhart-Hart Equation**: Most accurate model for NTC thermistors

```
1/T = A + B×ln(R) + C×(ln(R))³
```

Where:
- **T** = Temperature in Kelvin
- **R** = Resistance in ohms
- **A, B, C** = Steinhart-Hart coefficients (from datasheet)

**Simplified Beta Equation**: Commonly used approximation

```
R(T) = R₀ × e^(β×(1/T - 1/T₀))
```

Where:
- **R₀** = Resistance at reference temperature T₀ (usually 25°C)
- **β** = Beta coefficient (material constant, typically 3000-5000K)
- **T** = Temperature in Kelvin

### Temperature Coefficient

**Negative coefficient**: Resistance decreases exponentially with temperature

**Typical sensitivity**: -3% to -6% per °C at 25°C

**Example**: 10kΩ NTC at 25°C with β=3950K
- At 0°C: ~32kΩ
- At 25°C: 10kΩ
- At 50°C: ~3.6kΩ

## PTC Thermistor Characteristics

### Switching Behavior

**Low temperature region**: Low resistance, normal operation

**Curie temperature (Tc)**: Sharp resistance increase (switching point)

**High temperature region**: Very high resistance, current limiting

**Typical Tc values**: 60°C, 80°C, 120°C, 150°C

### Self-Heating Effect

**Current flow**: Generates heat in thermistor (I²R)

**Temperature rise**: Increases resistance (for PTC)

**Self-regulating**: Limits current automatically

**Applications**: Overcurrent protection, motor starting, degaussing

### Resistance Ratio

**Rmin/Rmax ratio**: Can exceed 1000:1 at switching temperature

**Example**: 100Ω at 25°C → 100kΩ at 150°C

## Key Specifications

### Resistance at 25°C (R25)

Nominal resistance value at standard reference temperature

**Common NTC values**: 1kΩ, 2.2kΩ, 5kΩ, 10kΩ, 47kΩ, 100kΩ

**Common PTC values**: 10Ω, 50Ω, 100Ω, 1kΩ

### Beta Coefficient (β)

Material constant describing temperature sensitivity (NTC only)

**Typical range**: 3000K to 5000K

**Higher β**: Greater sensitivity to temperature changes

**Specified between two temperatures**: e.g., β25/85 (between 25°C and 85°C)

### Tolerance

Accuracy of resistance value at 25°C

**Typical values**: ±1%, ±2%, ±5%, ±10%, ±20%

**Tighter tolerance**: Higher cost, better accuracy

### Temperature Range

Operating temperature limits

**NTC typical**: -55°C to +150°C (some to +300°C)

**PTC typical**: -40°C to +125°C

### Power Rating

Maximum power dissipation without damage

**Typical values**: 50mW to 1W

**Derating**: Reduce power at high ambient temperatures

### Response Time

Time to reach 63.2% of final temperature change

**Thermal time constant (τ)**: Typically 1 to 30 seconds

**Smaller thermistors**: Faster response

## Common Applications

### Temperature Sensing (NTC)

**Digital thermometers**: Accurate temperature measurement

**HVAC systems**: Room and ambient temperature monitoring

**Automotive**: Engine coolant, oil, air intake temperature

**Battery management**: Monitor battery temperature for safety

**Medical devices**: Body temperature measurement

### Temperature Compensation (NTC)

**Circuit stabilization**: Compensate for temperature drift in other components

**Oscillator frequency**: Maintain stable frequency over temperature

**Voltage references**: Temperature-compensated voltage sources

### Inrush Current Limiting (NTC)

**Power supplies**: Limit startup surge current

**Motor soft-start**: Reduce mechanical stress during startup

**LED drivers**: Protect from initial current spike

**Operation**: Cold resistance limits current, self-heating reduces resistance

### Overcurrent Protection (PTC)

**Resettable fuses**: Automatic circuit protection (Polyfuse)

**Motor protection**: Prevent overheating and overcurrent

**Battery protection**: Limit excessive discharge current

**Operation**: Overcurrent causes heating, resistance increases, limits current

### Self-Regulating Heaters (PTC)

**Constant temperature heating**: Automotive mirror defrosters

**Fluid heating**: Maintain constant temperature without thermostat

**Hair dryers**: Temperature limiting for safety

**Operation**: As temperature rises, resistance increases, power decreases

## Practical Considerations

### Voltage Divider Circuit (NTC Temperature Sensing)

**Basic circuit**: Thermistor in series with fixed resistor

**Voltage output**:
```
Vout = Vcc × (Rfixed / (Rthermistor + Rfixed))
```

**Resistor selection**: Choose Rfixed ≈ R25 for maximum sensitivity at 25°C

**ADC connection**: Connect Vout to microcontroller ADC input

### Linearization Techniques

**Problem**: Exponential resistance-temperature relationship is nonlinear

**Software linearization**: Use Steinhart-Hart or Beta equation in firmware

**Hardware linearization**: Parallel resistor to reduce nonlinearity

**Lookup tables**: Store resistance-temperature pairs for fast conversion

### Self-Heating Error

**Problem**: Measurement current causes self-heating, affecting accuracy

**Power dissipation**: P = I² × R generates heat

**Minimize error**: Use low measurement current (< 100μA typical)

**High-impedance circuits**: Reduce current through thermistor

### Thermal Coupling

**Good thermal contact**: Ensure thermistor touches measured surface

**Thermal paste**: Improves heat transfer for surface mounting

**Immersion**: For liquid temperature, ensure full immersion

**Air gap**: Avoid air gaps that slow response and reduce accuracy

### Common Mistakes to Avoid

- **Excessive measurement current**: Causes self-heating and inaccurate readings
- **Wrong β coefficient**: Using incorrect value leads to temperature errors
- **Ignoring self-heating**: Especially critical in low-power applications
- **Poor thermal contact**: Slow response and measurement errors
- **Exceeding power rating**: Permanent damage or drift
- **Not accounting for tolerance**: ±20% tolerance = significant temperature error
- **Linear interpolation**: Exponential curve requires proper equation

## Summary

Thermistors are temperature-sensitive resistors offering high sensitivity and fast response for temperature sensing and control applications. Understanding NTC and PTC characteristics, proper circuit design, and linearization techniques is essential for accurate temperature measurement and protection applications.

**Key Takeaways**:
- Two types: NTC (negative coefficient) and PTC (positive coefficient)
- NTC: Resistance decreases with temperature, used for sensing
- PTC: Resistance increases with temperature, used for protection
- NTC sensitivity: -3% to -6% per °C at 25°C
- Beta equation: R(T) = R₀ × e^(β×(1/T - 1/T₀))
- Steinhart-Hart equation: Most accurate for NTC thermistors
- Common R25 values: 1kΩ, 10kΩ, 47kΩ, 100kΩ (NTC)
- Beta coefficient: 3000K to 5000K typical
- Temperature range: -55°C to +150°C (NTC), -40°C to +125°C (PTC)
- Applications: Temperature sensing, compensation, inrush limiting, overcurrent protection
- Voltage divider: Choose Rfixed ≈ R25 for maximum sensitivity
- Self-heating: Use low measurement current (< 100μA)
- Linearization: Use Steinhart-Hart equation or lookup tables
- PTC switching: Sharp resistance increase at Curie temperature
- Response time: 1 to 30 seconds typical

Proper thermistor selection based on resistance value, beta coefficient, tolerance, and temperature range ensures accurate temperature measurement and reliable protection in sensing and control applications.

## References

- Thermistor theory and temperature coefficient
- Steinhart-Hart equation and Beta equation
- NTC and PTC thermistor characteristics
- Thermistor manufacturer datasheets and specifications
- Temperature measurement circuit design


