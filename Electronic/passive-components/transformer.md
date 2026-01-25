# Transformer

## Overview

A **transformer** is a passive electrical device that transfers electrical energy between two or more circuits through electromagnetic induction. Transformers are used to increase (step up) or decrease (step down) AC voltages, provide electrical isolation, and match impedances between circuits.

## Basic Operation

### Electromagnetic Induction

**Principle**: Changing magnetic field induces voltage in nearby conductor (Faraday's Law)

**Primary winding**: Input coil connected to AC source, creates changing magnetic field

**Magnetic core**: Concentrates and guides magnetic flux between windings

**Secondary winding**: Output coil, voltage induced by changing magnetic flux

**Key requirement**: AC voltage (changing current) required - transformers don't work with DC

### Turns Ratio

The relationship between primary and secondary voltages depends on the number of turns:

```
Vs / Vp = Ns / Np
```

Where:
- **Vp** = Primary voltage
- **Vs** = Secondary voltage
- **Np** = Number of primary turns
- **Ns** = Number of secondary turns

### Power Relationship

In an ideal transformer, power in equals power out:

```
Vp × Ip = Vs × Is
```

Therefore:
```
Is / Ip = Np / Ns = Vp / Vs
```

**Step-up transformer**: Ns > Np, increases voltage, decreases current
**Step-down transformer**: Ns < Np, decreases voltage, increases current

## Types of Transformers

### Power Transformers

**Purpose**: Transfer electrical power between circuits at different voltage levels

**Characteristics**:
- Large size and weight
- High power ratings (VA or kVA)
- Laminated iron core
- 50/60 Hz operation

**Applications**: Power supplies, AC adapters, distribution systems

### Isolation Transformers

**Purpose**: Provide electrical isolation between primary and secondary

**Characteristics**:
- 1:1 turns ratio (same voltage in/out)
- Breaks ground loops
- Protects against electric shock
- Reduces noise coupling

**Applications**: Medical equipment, audio systems, test equipment, safety isolation

### Audio Transformers

**Purpose**: Couple audio signals between stages, match impedances

**Characteristics**:
- Optimized for audio frequency range (20Hz-20kHz)
- Low distortion
- Impedance matching capability
- Small size

**Applications**: Microphone preamps, guitar amplifiers, audio interfaces, speaker matching

### RF Transformers

**Purpose**: Couple RF signals, impedance matching at high frequencies

**Characteristics**:
- Optimized for MHz frequency range
- Air core or ferrite core
- Minimal parasitic capacitance
- Precise impedance ratios

**Applications**: Radio transmitters/receivers, antenna matching, RF amplifiers

### Current Transformers (CT)

**Purpose**: Measure AC current by stepping down to measurable level

**Characteristics**:
- High turns ratio (primary has few turns, secondary has many)
- Primary in series with load
- Secondary provides proportional current
- Never open-circuit secondary (dangerous high voltage)

**Applications**: Power metering, current sensing, protection relays

### Autotransformer

**Construction**: Single winding with tap, not electrically isolated

**Characteristics**:
- More compact and efficient than two-winding
- No electrical isolation between input and output
- Lower cost
- Common tap point

**Applications**: Variable AC supplies (Variac), voltage adjustment, motor starting

## Key Specifications

### Power Rating (VA)

Maximum power the transformer can handle continuously

**Calculation**: VA = Voltage × Current

**Example**: 12V @ 2A = 24VA transformer

**Important**: Derate for continuous operation and high ambient temperature

### Turns Ratio

Ratio of primary to secondary turns (Np:Ns)

**Examples**:
- 230V:12V = approximately 19:1 (step-down)
- 12V:120V = 1:10 (step-up)

### Efficiency

Ratio of output power to input power

```
Efficiency = (Pout / Pin) × 100%
```

**Typical values**: 85-98% depending on size and design

**Losses**: Core losses (hysteresis, eddy currents), copper losses (I²R)

### Voltage Regulation

Change in output voltage from no-load to full-load

```
Regulation = ((Vno-load - Vfull-load) / Vfull-load) × 100%
```

**Better regulation**: Lower percentage (1-5% typical)

### Frequency Range

Operating frequency range for the transformer

**Power transformers**: 50/60 Hz (mains frequency)
**Audio transformers**: 20 Hz - 20 kHz
**RF transformers**: kHz to MHz range

### Insulation Class

Temperature rating of insulation materials

**Classes**: A (105°C), B (130°C), F (155°C), H (180°C)

## Common Applications

### Power Supplies

**AC-DC conversion**: Step down mains voltage, then rectify to DC

**Multiple voltages**: Center-tapped or multiple secondaries for different voltages

**Isolation**: Safety isolation from mains voltage

### Impedance Matching

**Audio systems**: Match microphone/speaker impedances to amplifiers

**RF systems**: Match antenna impedance to transmitter/receiver

**Maximum power transfer**: Occurs when source and load impedances are matched

### Voltage Level Conversion

**Distribution systems**: Step up voltage for transmission, step down for use

**Industrial equipment**: Convert between different voltage standards

**International adapters**: Convert between 110V and 220V systems

### Signal Isolation

**Ground loop elimination**: Break ground loops in audio/video systems

**Noise reduction**: Isolate noisy circuits from sensitive circuits

**Safety isolation**: Protect users from high voltages

## Practical Considerations

### Core Material

**Laminated iron**: Power transformers, low frequency
- Reduces eddy current losses
- Silicon steel laminations

**Ferrite**: High-frequency transformers
- Lower losses at high frequencies
- Used in switching power supplies

**Air core**: RF transformers
- No core losses
- Lower inductance

### Inrush Current

**Problem**: High initial current when transformer is first energized (5-10× normal)

**Cause**: Magnetizing the core from zero

**Solutions**:
- Use slow-blow fuses
- Soft-start circuits
- NTC thermistor in series

### Heat Dissipation

**Losses generate heat**: Core losses + copper losses

**Cooling methods**:
- Natural convection (most common)
- Forced air cooling (fans)
- Oil cooling (large power transformers)

**Temperature rise**: Typically 40-60°C above ambient at full load

### Common Mistakes to Avoid

- **Exceeding power rating**: Causes overheating and insulation failure
- **Wrong voltage rating**: Can damage equipment or create safety hazard
- **Ignoring inrush current**: Causes fuses to blow unnecessarily
- **Poor ventilation**: Leads to overheating and reduced lifespan
- **Connecting DC to transformer**: Saturates core, causes overheating
- **Open-circuit current transformer secondary**: Creates dangerous high voltage
- **Incorrect phasing**: Multiple secondaries must be connected with correct polarity

## Summary

Transformers transfer electrical energy between circuits through electromagnetic induction, enabling voltage transformation and electrical isolation. Understanding transformer types, specifications, and proper application is essential for power supply design and signal processing.

**Key Takeaways**:
- Transformers work on AC only (changing magnetic field required)
- Voltage ratio equals turns ratio: Vs/Vp = Ns/Np
- Power in equals power out (ideal): Vp×Ip = Vs×Is
- Step-up: Increases voltage, decreases current
- Step-down: Decreases voltage, increases current
- Types: Power, isolation, audio, RF, current, autotransformer
- Key specs: Power rating (VA), turns ratio, efficiency, voltage regulation
- Efficiency typically 85-98%
- Inrush current 5-10× normal when first energized
- Use slow-blow fuses for transformer circuits
- Isolation transformers provide safety and noise reduction
- Never open-circuit current transformer secondary

Proper transformer selection based on power rating, voltage ratio, frequency range, and isolation requirements ensures reliable operation in power conversion and signal coupling applications.

## References

- Transformer theory and electromagnetic induction
- IEC 61558: Safety of transformers
- Transformer manufacturer datasheets and specifications
- Core material selection and magnetic design

