# Photoresistor (Light Dependent Resistor - LDR)

## Overview

A **photoresistor** (also called Light Dependent Resistor or LDR) is a passive component whose resistance decreases with increasing light intensity. Photoresistors are widely used for light detection, automatic lighting control, and light-level sensing applications due to their simplicity, low cost, and ease of use.

## Basic Operation

### Light-Dependent Resistance

**Principle**: Resistance varies inversely with light intensity

**Dark resistance**: Very high resistance (MΩ range) in darkness

**Light resistance**: Low resistance (hundreds of Ω) in bright light

**Photoconductivity**: Light photons excite electrons, increasing conductivity

### Spectral Response

**Sensitive wavelength range**: Typically 400-700nm (visible light spectrum)

**Peak sensitivity**: Around 540nm (green light) for CdS photoresistors

**Material dependent**: Different materials have different spectral responses

**Applications**: Match photoresistor to light source spectrum

## Types of Photoresistors

### Cadmium Sulfide (CdS)

**Most common type**: Widely used for visible light detection

**Spectral range**: 400-700nm (visible light)

**Peak sensitivity**: ~540nm (green)

**Characteristics**: Good sensitivity, moderate response time

**Applications**: Light meters, automatic lighting, camera exposure control

### Cadmium Selenide (CdSe)

**Infrared sensitive**: Extended response into near-infrared

**Spectral range**: 500-750nm

**Peak sensitivity**: ~650nm (red)

**Applications**: Infrared detection, flame sensors

### Lead Sulfide (PbS)

**Infrared photoresistor**: Sensitive to infrared radiation

**Spectral range**: 1000-3000nm (near to mid-infrared)

**Applications**: Infrared spectroscopy, thermal imaging, gas analysis

**Requires cooling**: Often cooled for better sensitivity

## Key Specifications

### Dark Resistance (Rdark)

Resistance in complete darkness

**Typical values**: 1MΩ to 10MΩ for CdS photoresistors

**Higher values**: Indicates better light sensitivity

### Light Resistance (R10lux or R100lux)

Resistance at specified light level (typically 10 lux or 100 lux)

**Typical values**: 1kΩ to 20kΩ at 10 lux

**Lower values**: Better conductivity in light

### Resistance Ratio (Rdark/Rlight)

Ratio of dark resistance to light resistance

**Typical values**: 100:1 to 1000:1

**Higher ratio**: Greater sensitivity to light changes

### Response Time

Time to reach steady-state resistance after light change

**Rise time**: Darkness to light (typically 10-30ms)

**Fall time**: Light to darkness (typically 20-100ms)

**Slower than photodiodes**: Due to material properties

### Power Rating

Maximum power dissipation

**Typical values**: 100mW to 200mW

**Derating**: Reduce power at high temperatures

## Common Applications

### Automatic Lighting Control

**Street lights**: Turn on at dusk, off at dawn

**Garden lights**: Solar-powered lights with automatic activation

**Indoor lighting**: Adjust brightness based on ambient light

**Energy saving**: Reduce unnecessary lighting during daylight

### Light Detection and Measurement

**Light meters**: Photography exposure measurement

**Brightness sensors**: Display backlight adjustment

**Day/night detection**: Security systems, cameras

**Ambient light sensing**: Smartphones, tablets, laptops

### Alarm Systems

**Laser trip alarms**: Detect beam interruption

**Shadow detection**: Detect objects blocking light

**Perimeter security**: Detect changes in light patterns

### Camera Exposure Control

**Automatic exposure**: Adjust shutter speed and aperture

**Flash trigger**: Activate flash in low light

**Legacy cameras**: Mechanical exposure meters

## Practical Considerations

### Voltage Divider Circuit

**Basic circuit**: Photoresistor in series with fixed resistor

**Voltage output**:
```
Vout = Vcc × (Rfixed / (RLDR + Rfixed))
```

**Resistor selection**: Choose Rfixed in middle of LDR's resistance range

**Example**: For LDR with 1kΩ (light) to 1MΩ (dark), use 10kΩ to 100kΩ

**ADC connection**: Connect Vout to microcontroller ADC input

### Response Time Considerations

**Slow response**: 20-100ms fall time limits high-speed applications

**Not suitable for**: Fast light modulation, high-speed communication

**Suitable for**: Ambient light sensing, day/night detection

**Hysteresis**: Add to prevent oscillation at threshold

### Temperature Effects

**Resistance drift**: Temperature changes affect resistance

**Compensation**: Use temperature-stable reference resistor

**Calibration**: May require temperature compensation in precision applications

### Nonlinear Response

**Logarithmic relationship**: Resistance vs light intensity is nonlinear

**Linearization**: Use lookup tables or logarithmic conversion

**Software processing**: Apply calibration curve in firmware

### Common Mistakes to Avoid

- **Excessive current**: Causes self-heating and resistance drift
- **Wrong fixed resistor value**: Reduces sensitivity or limits range
- **No hysteresis**: Causes oscillation at switching threshold
- **Expecting fast response**: LDRs are slow compared to photodiodes
- **Ignoring spectral response**: CdS not suitable for infrared detection
- **Direct sunlight exposure**: Can cause permanent damage or drift
- **No protection from ambient light**: Stray light affects accuracy

## Summary

Photoresistors (LDRs) are light-sensitive resistors that provide simple, low-cost light detection for ambient sensing and automatic control applications. Understanding photoresistor characteristics, proper circuit design, and response limitations is essential for effective light sensing applications.

**Key Takeaways**:
- Light-dependent resistance: High resistance in dark, low resistance in light
- Most common: Cadmium Sulfide (CdS) for visible light (400-700nm)
- Dark resistance: 1MΩ to 10MΩ typical
- Light resistance: 1kΩ to 20kΩ at 10 lux
- Resistance ratio: 100:1 to 1000:1 (dark/light)
- Response time: 10-30ms rise, 20-100ms fall (slow)
- Spectral response: Peak at ~540nm for CdS (green light)
- Applications: Automatic lighting, light meters, alarm systems, exposure control
- Voltage divider: Choose Rfixed in middle of LDR range
- Nonlinear response: Logarithmic relationship with light intensity
- Temperature sensitive: May require compensation
- Not suitable for: High-speed applications, precise measurements
- Power rating: 100-200mW typical
- Slower than photodiodes: Due to material properties

Proper photoresistor selection based on spectral response, resistance range, and response time ensures reliable light detection in automatic lighting control and ambient sensing applications.

## References

- Photoresistor theory and photoconductivity
- CdS, CdSe, and PbS material properties
- Light measurement and lux units
- Photoresistor manufacturer datasheets and specifications


