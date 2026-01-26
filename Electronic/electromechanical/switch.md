# Switch

## Overview

A **switch** is an electromechanical device that opens or closes an electrical circuit by making or breaking the connection between two or more conductors. Switches are fundamental components in electronics, used for manual control, signal routing, and circuit configuration.

## Basic Terminology

### Contact Configuration

**Pole**: Number of separate circuits controlled by the switch
**Throw**: Number of positions each pole can connect to

**Common configurations**:
- **SPST** (Single Pole Single Throw): Simple ON/OFF switch
- **SPDT** (Single Pole Double Throw): One input, two outputs
- **DPST** (Double Pole Single Throw): Two separate ON/OFF switches
- **DPDT** (Double Pole Double Throw): Two separate SPDT switches

### Contact Action

**Momentary**: Returns to default position when released (e.g., push button)
**Latching**: Stays in position until manually changed (e.g., toggle switch)

### Contact Type

**Normally Open (NO)**: Circuit open when not actuated
**Normally Closed (NC)**: Circuit closed when not actuated
**Changeover (CO)**: Has both NO and NC contacts

## Types of Switches

### Push Button Switch

**Operation**: Momentary contact when pressed

**Types**:
- **Tactile switch**: Provides tactile feedback (click feeling)
- **Membrane switch**: Flat, sealed design
- **Illuminated button**: Contains LED indicator

**Configurations**: SPST-NO (most common), SPST-NC, SPDT

**Applications**: User input, reset buttons, control panels, keyboards

### Toggle Switch

**Operation**: Latching switch with lever actuator

**Positions**: 2-position (ON-OFF) or 3-position (ON-OFF-ON)

**Configurations**: SPST, SPDT, DPDT

**Applications**: Power switches, mode selection, circuit switching

### Slide Switch

**Operation**: Latching switch with sliding actuator

**Positions**: 2 or 3 positions

**Configurations**: SPST, SPDT, DPDT

**Applications**: Power switches, mode selection, compact devices

### Rotary Switch

**Operation**: Multiple positions selected by rotating shaft

**Positions**: 2 to 12+ positions

**Configurations**: Multi-pole, multi-position

**Applications**: Channel selection, mode switching, multi-function control

### DIP Switch

**Operation**: Dual In-line Package switch array

**Configuration**: Multiple SPST switches in one package (2 to 12 positions)

**Operation**: Latching, set with small tool or fingernail

**Applications**: Configuration settings, address selection, option selection

### Microswitch

**Operation**: Snap-action switch with precise actuation point

**Actuator types**: Lever, roller, pin plunger

**Characteristics**: High reliability, long life, precise switching

**Applications**: Limit switches, safety interlocks, position sensing

## Key Specifications

### Voltage Rating

Maximum voltage the switch can safely handle.

**Typical values**: 12V, 24V, 125V AC, 250V AC

**Important**: AC and DC ratings differ

### Current Rating

Maximum current the switch contacts can carry.

**Typical values**: 100mA to 10A+

**Considerations**: Inductive loads (motors, relays) require derating

### Contact Resistance

Resistance of closed contacts.

**Typical values**: 10mΩ to 100mΩ

**Impact**: Affects voltage drop and power dissipation

### Mechanical Life

Number of operations before mechanical failure.

**Typical values**: 10,000 to 10,000,000+ cycles

### Electrical Life

Number of operations under rated load before contact failure.

**Typical values**: Usually less than mechanical life, depends on load

## Practical Considerations

### Contact Bounce

**Problem**: Mechanical contacts bounce when closing, creating multiple pulses

**Duration**: Typically 1-20ms

**Solutions**:
- Hardware debouncing: RC filter, Schmitt trigger
- Software debouncing: Delay or state machine in microcontroller

### Inductive Load Switching

**Problem**: Inductive loads (motors, relays, solenoids) generate voltage spikes

**Solution**: Use snubber circuit or flyback diode across load

**Derating**: Reduce current rating by 50% or more for inductive loads

### Common Mistakes to Avoid

- **Exceeding voltage/current ratings**: Causes contact welding or arcing
- **No debouncing**: Causes false triggering in digital circuits
- **Ignoring inductive load requirements**: Shortens switch life
- **Wrong contact configuration**: Using NO when NC needed or vice versa
- **Poor mechanical mounting**: Causes premature failure
- **No arc suppression**: Reduces life when switching inductive loads

## Summary

Switches are fundamental electromechanical components that control electrical circuits by making or breaking connections. Understanding switch types, configurations, and ratings is essential for proper circuit design and reliable operation.

**Key Takeaways**:
- Contact configurations: SPST, SPDT, DPST, DPDT (poles and throws)
- Contact action: Momentary (returns to default) vs Latching (stays in position)
- Contact type: Normally Open (NO), Normally Closed (NC), Changeover (CO)
- Common types: Push button, toggle, slide, rotary, DIP, microswitch
- Key specs: Voltage rating, current rating, contact resistance, mechanical/electrical life
- Contact bounce: 1-20ms, requires debouncing in digital circuits
- Inductive loads require derating and arc suppression
- Always verify voltage and current ratings match application requirements

Proper switch selection based on contact configuration, voltage/current ratings, mechanical life, and application requirements ensures reliable circuit control.

## References

- Switch contact configurations and terminology
- Manufacturer datasheets for specific switch types
- Contact bounce characteristics and debouncing techniques
- Inductive load switching and arc suppression methods
- Switch life ratings and derating factors
