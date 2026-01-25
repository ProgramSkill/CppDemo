# Inductor

## Overview

An **inductor** is a passive two-terminal electrical component that stores energy in a magnetic field when electric current flows through it. Inductors are fundamental components used for filtering, energy storage, impedance matching, and signal processing in electronic circuits.

## What is Inductance?

**Inductance** is the property of a conductor to oppose changes in current flow. It is measured in **Henries (H)**, named after American scientist Joseph Henry.

Common units:
- **Henry (H)**: Base unit
- **Millihenry (mH)**: 10⁻³ H (common)
- **Microhenry (μH or uH)**: 10⁻⁶ H (common)
- **Nanohenry (nH)**: 10⁻⁹ H (RF applications)

### Basic Formula

The relationship between voltage (V), inductance (L), and rate of current change:

```
V = L × (dI/dt)
```

Where:
- **V** = Induced voltage (Volts)
- **L** = Inductance (Henries)
- **dI/dt** = Rate of current change (Amperes per second)

This is known as **Faraday's Law of Induction**.

### Energy Storage

Energy stored in an inductor:

```
E = ½ × L × I²
```

Where:
- **E** = Energy (Joules)
- **L** = Inductance (Henries)
- **I** = Current (Amperes)

## Physical Construction

An inductor typically consists of a coil of wire wound around a core material.

### Inductance Formula

For a simple solenoid (coil):

```
L = (μ₀ × μᵣ × N² × A) / l
```

Where:
- **μ₀** = Permeability of free space (4π × 10⁻⁷ H/m)
- **μᵣ** = Relative permeability of core material
- **N** = Number of turns
- **A** = Cross-sectional area of coil (m²)
- **l** = Length of coil (m)

**Key factors affecting inductance**:
- More turns = higher inductance
- Larger core area = higher inductance
- Shorter coil = higher inductance
- Higher permeability core = higher inductance

## Types of Inductors

### Air Core Inductors

**Construction**: Coil wound without any core material (air core)

**Characteristics**:
- Low inductance values (typically nH to μH)
- No core saturation
- Low losses at high frequencies
- Temperature stable
- Used in RF applications

**Applications**: RF circuits, oscillators, antenna tuning

### Ferrite Core Inductors

**Construction**: Coil wound on ferrite (iron oxide ceramic) core

**Characteristics**:
- High permeability (μᵣ = 10 to 10,000)
- Higher inductance in smaller size
- Good for high-frequency applications (kHz to MHz)
- Can saturate at high currents
- Various shapes: rod, toroid, E-core

**Applications**: EMI filtering, switching power supplies, transformers

### Iron Core Inductors

**Construction**: Coil wound on laminated iron or steel core

**Characteristics**:
- Very high permeability
- High inductance values
- Good for low-frequency applications (50/60 Hz)
- Heavy and bulky
- Core losses at high frequencies

**Applications**: Power transformers, chokes, audio equipment

### Toroidal Inductors

**Construction**: Wire wound on donut-shaped (toroidal) core

**Characteristics**:
- Closed magnetic path (minimal EMI)
- High efficiency
- Compact size
- Self-shielding
- Available in ferrite, iron powder, or other materials

**Applications**: Power supplies, filters, audio circuits

### SMD (Surface Mount) Inductors

**Types**:
- **Chip inductors**: Ceramic or ferrite core, small size
- **Molded inductors**: Wire wound, encapsulated in resin
- **Multilayer inductors**: Ceramic construction, very small

**Characteristics**:
- Compact size for PCB assembly
- Inductance range: 1nH to 1mH
- Current ratings: mA to several Amps
- Various package sizes (0402, 0603, 0805, 1206, etc.)

**Applications**: Mobile devices, compact electronics, DC-DC converters

### Variable Inductors

**Types**:
- **Adjustable core**: Ferrite slug moves in/out of coil
- **Tapped inductors**: Multiple connection points for different inductance values

**Applications**: Tuning circuits, impedance matching, filter adjustment

### Chokes

**Common Mode Choke**: Two coils wound on same core, blocks common-mode noise

**Differential Mode Choke**: Single coil, blocks differential-mode noise

**Applications**: EMI/RFI filtering, power line filtering

## Inductor Markings and Specifications

### Color Code (Through-hole Inductors)

Similar to resistors, some inductors use color bands:
- First 2-3 bands: significant digits
- Last band: multiplier
- Unit: microhenries (μH)

**Example**: Brown-Black-Brown = 10 × 10 = 100μH

### SMD Inductor Codes

**Direct marking**: Value printed directly (e.g., "100" = 100μH, "4R7" = 4.7μH)

**3-digit code**: Similar to capacitors
- First 2 digits: significant figures
- Last digit: multiplier
- Unit: microhenries (μH)

**Example**: "101" = 10 × 10¹ = 100μH

### Key Specifications

**Inductance (L)**: Measured in H, mH, μH, or nH

**DC Resistance (DCR)**: Wire resistance, causes power loss
- Lower DCR = better efficiency
- Measured in Ohms (Ω)

**Saturation Current (Isat)**: Maximum current before core saturates
- Beyond this, inductance drops significantly

**Rated Current (Irms)**: Maximum continuous current based on temperature rise
- Typically 20-40°C temperature rise

**Self-Resonant Frequency (SRF)**: Frequency where parasitic capacitance resonates with inductance
- Inductor acts as capacitor above SRF
- Use inductor well below its SRF

**Q Factor (Quality Factor)**: Ratio of energy stored to energy lost
- Higher Q = lower losses
- Important for RF and resonant circuits

## Inductor Configurations

### Series Connection

When inductors are connected in series (assuming no mutual coupling), the total inductance is the sum:

```
L_total = L1 + L2 + L3 + ... + Ln
```

**Characteristics**:
- Total inductance increases
- Same current flows through all inductors
- Voltage divides across inductors

**Example**: 100μH + 220μH + 470μH = 790μH

### Parallel Connection

When inductors are connected in parallel (assuming no mutual coupling), the total inductance is:

```
1/L_total = 1/L1 + 1/L2 + 1/L3 + ... + 1/Ln
```

For two inductors in parallel:
```
L_total = (L1 × L2) / (L1 + L2)
```

**Characteristics**:
- Total inductance decreases (always less than smallest inductor)
- Same voltage across all inductors
- Current divides among inductors

**Example**: Two 100μH inductors in parallel = 50μH

**Note**: Mutual coupling between inductors can significantly affect these calculations. Keep inductors physically separated or use shielded types.

## Common Applications

### Power Supply Filtering

**Smoothing chokes**: Filter ripple current in power supplies

**LC filters**: Combined with capacitors for better filtering
```
f_cutoff = 1 / (2π√(L × C))
```

### DC-DC Converters

**Buck converter**: Steps down voltage (inductor stores/releases energy)
**Boost converter**: Steps up voltage
**Buck-boost converter**: Can step up or down

Inductors are critical energy storage elements in switching regulators.

### EMI/RFI Filtering

**Common mode chokes**: Block electromagnetic interference
**Ferrite beads**: Suppress high-frequency noise on signal/power lines

### Resonant Circuits

**LC tank circuits**: Create oscillations at resonant frequency
```
f_resonant = 1 / (2π√(L × C))
```

**Applications**: Radio tuning, oscillators, filters

### Transformers

Two or more inductors with mutual coupling transfer energy between circuits.

**Applications**: Power conversion, isolation, impedance matching

### Motor Control

**Smoothing inductors**: Reduce current ripple in motor drives
**Snubber circuits**: Protect against voltage spikes

## Practical Considerations

### Core Saturation

When current exceeds saturation current (Isat), the core saturates and inductance drops dramatically.

**Effects**:
- Reduced filtering effectiveness
- Increased ripple current
- Potential circuit malfunction

**Solution**: Choose inductor with Isat > maximum expected current

### DC Resistance (DCR)

Wire resistance causes power loss and voltage drop.

**Power loss**: P = I² × DCR

**Considerations**:
- Lower DCR = higher efficiency
- Thicker wire = lower DCR but larger size
- Important in high-current applications

### Self-Resonant Frequency (SRF)

Parasitic capacitance between windings creates resonance.

**Guidelines**:
- Use inductor at frequencies well below SRF (typically < SRF/3)
- Above SRF, inductor behaves as capacitor
- Check datasheet for SRF specification

### Magnetic Coupling and EMI

Inductors generate magnetic fields that can interfere with nearby components.

**Solutions**:
- Use shielded inductors
- Orient inductors perpendicular to each other
- Maintain adequate spacing
- Use toroidal cores (self-shielding)

### Temperature Effects

**Core material temperature coefficient**: Inductance changes with temperature

**Current rating**: Based on acceptable temperature rise (typically 20-40°C)

**Considerations**:
- Ensure adequate cooling/airflow
- Derate for high ambient temperatures
- Check maximum operating temperature

### Common Mistakes to Avoid

- **Exceeding saturation current**: Causes inductance collapse
- **Ignoring DCR**: Results in unexpected power loss and voltage drop
- **Operating above SRF**: Inductor behaves as capacitor
- **Poor layout**: Magnetic coupling causes EMI and interference
- **Inadequate current rating**: Leads to overheating
- **Wrong core material**: Use ferrite for high frequency, iron for low frequency

## Summary

Inductors are essential energy storage components that oppose changes in current flow. They are fundamental to power supplies, filters, and signal processing circuits. Understanding inductor characteristics and proper selection is critical for reliable circuit design.

**Key Takeaways**:
- Inductance (L = V/(dI/dt)) measured in Henries
- Energy stored in magnetic field (E = ½LI²)
- Core material affects inductance and frequency response
- Saturation current (Isat) must exceed maximum circuit current
- DC resistance (DCR) causes power loss
- Self-resonant frequency (SRF) limits usable frequency range
- Series connection increases inductance, parallel decreases it
- Critical in DC-DC converters, filters, and EMI suppression

Proper inductor selection requires considering inductance value, saturation current, DC resistance, SRF, core material, and physical size to ensure optimal circuit performance.

## References

- Faraday's Law of Induction
- Magnetic core materials and properties
- Inductor design and application notes
- Manufacturer datasheets for specific component specifications

