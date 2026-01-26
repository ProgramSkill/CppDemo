# DIAC (Diode for Alternating Current)

## Overview

A **DIAC** (Diode for Alternating Current) is a two-terminal bidirectional semiconductor device used primarily as a trigger element for TRIACs and SCRs. Unlike regular diodes, DIACs conduct current in both directions once the breakover voltage is reached. They provide symmetric triggering and are commonly used in AC phase control circuits, light dimmers, and motor speed controllers.

## Basic Structure

### Two Terminals

**A1 (Anode 1)**: First terminal

**A2 (Anode 2)**: Second terminal

**No polarity**: Symmetric device, terminals are interchangeable

### Bidirectional Structure

**Construction**: Two diodes in anti-parallel or three-layer structure

**Symmetry**: Identical characteristics in both directions

**Breakover voltage**: Same magnitude for positive and negative voltages

## Basic Operation

### Blocking State

**Below breakover voltage**: DIAC blocks current in both directions

**Leakage current**: Very small (μA range)

**Behavior**: Acts as open circuit

### Breakover and Conduction

**Breakover voltage (VBO)**: Voltage at which DIAC switches to conducting state

**Typical values**: 28V to 36V (most common: 32V)

**Negative resistance region**: After breakover, voltage drops and current increases

**Conducting state**: Low impedance, voltage drop typically 5-15V

**Turn-off**: Current must fall below holding current

## Key Specifications

### Breakover Voltage (VBO)

Voltage at which DIAC switches from blocking to conducting state.

**Typical values**: 28V to 36V (DB3: 32V ±4V)

**Symmetry**: ±2V to ±5V difference between positive and negative breakover

**Importance**: Determines trigger point in phase control circuits

### Breakover Current (IBO)

Current at breakover point.

**Typical values**: 50μA to 200μA

**Low current**: Minimal power required to trigger

### On-State Voltage

Voltage drop across DIAC when conducting.

**Typical values**: 5V to 15V

**Negative resistance**: Voltage decreases as current increases after breakover

### Peak Current Rating

Maximum current DIAC can handle.

**Typical values**: 1A to 2A peak

**Pulse duration**: Usually specified for short pulses (μs range)

## DIAC in TRIAC Trigger Circuits

### Typical Circuit Configuration

**RC timing network**: Resistor and capacitor determine phase delay

**DIAC trigger**: Provides sharp trigger pulse when capacitor voltage reaches VBO

**TRIAC gate**: Receives symmetric trigger pulses from DIAC

**Potentiometer**: Adjusts RC time constant for variable phase control

### Operation Principle

**Capacitor charging**: Through resistor from AC line

**Breakover**: When capacitor voltage reaches DIAC VBO

**Trigger pulse**: DIAC conducts, discharging capacitor into TRIAC gate

**TRIAC turns ON**: Conducts for remainder of half-cycle

**Symmetric triggering**: DIAC provides identical triggering in both AC half-cycles

### Advantages of DIAC Triggering

**Symmetric operation**: Equal triggering in positive and negative half-cycles

**Sharp trigger pulse**: Fast switching reduces TRIAC stress

**Simple circuit**: Minimal components required

**Reliable triggering**: Consistent breakover voltage ensures predictable operation

## Common Applications

### Light Dimmer Circuits

**Purpose**: Control lamp brightness with smooth, symmetric dimming

**Circuit**: DIAC-TRIAC phase control with potentiometer

**Advantage**: Symmetric triggering prevents DC component in load

### Motor Speed Controllers

**Purpose**: Control AC motor speed in fans, drills, and appliances

**Operation**: Phase angle control reduces effective voltage to motor

**Benefit**: Simple, low-cost speed control solution

### Heater Control

**Purpose**: Temperature control for heating elements

**Operation**: Proportional power control using phase angle modulation

**Applications**: Soldering irons, hot plates, ovens

### Relaxation Oscillators

**Purpose**: Generate pulses or oscillations

**Operation**: RC charging with DIAC breakover creates periodic pulses

**Applications**: Timing circuits, pulse generators

## Practical Considerations

### Breakover Voltage Tolerance

**Variation**: ±10% to ±15% typical tolerance

**Impact**: Affects trigger timing in phase control circuits

**Design consideration**: Account for VBO variation in RC timing calculations

### RC Network Design

**Capacitor selection**: Typically 0.1μF to 1μF for 50/60Hz AC

**Resistor selection**: 10kΩ to 100kΩ range for phase control

**Potentiometer**: Variable resistor for adjustable phase angle

### Common Mistakes to Avoid

- **Exceeding peak current**: Can damage DIAC permanently
- **Wrong VBO selection**: DIAC VBO must be compatible with circuit voltage levels
- **No current limiting**: RC network must limit current to safe levels
- **Ignoring symmetry**: Assuming perfect symmetry without checking datasheet
- **Inadequate capacitor rating**: Capacitor voltage rating must exceed peak AC voltage

## Summary

DIACs (Diode for Alternating Current) are bidirectional trigger devices that provide symmetric switching for AC power control applications. With their negative resistance characteristic and consistent breakover voltage, DIACs are ideal for triggering TRIACs in phase control circuits such as light dimmers and motor speed controllers.

**Key Takeaways**:
- Bidirectional trigger device: Conducts in both directions after breakover
- Two terminals: No polarity, symmetric operation
- Breakover voltage: Typically 28V to 36V (DB3: 32V ±4V)
- Negative resistance: Voltage drops as current increases after breakover
- Primary application: TRIAC trigger element in phase control circuits
- Symmetric triggering: Provides equal triggering in both AC half-cycles
- Simple circuits: Minimal components (RC network + DIAC + TRIAC)
- Key specs: Breakover voltage (VBO), breakover current (IBO), peak current rating
- Applications: Light dimmers, motor speed controllers, heater control, relaxation oscillators
- Advantages: Symmetric operation, sharp trigger pulse, simple circuit, reliable triggering

Proper DIAC selection based on breakover voltage and peak current rating ensures reliable symmetric triggering in AC phase control applications.

## References

- DIAC operation principles and negative resistance characteristics
- Common DIAC datasheets (DB3, DB4, ST2)
- DIAC-TRIAC trigger circuit design
- RC timing network calculations for phase control
- Symmetric triggering and AC power control applications


