# Relay

## Overview

A **relay** is an electrically operated switch that uses an electromagnet to mechanically open or close electrical contacts. Relays enable low-power control circuits to switch high-power loads, providing electrical isolation between control and load circuits.

## Basic Operation

### Electromagnetic Relay Construction

**Key Components**:
- **Coil**: Electromagnet that creates magnetic field when energized
- **Armature**: Movable iron piece attracted by electromagnet
- **Contacts**: Switch contacts that open/close
- **Spring**: Returns armature to rest position when coil is de-energized

### Operating Principle

1. **Coil energized**: Current through coil creates magnetic field
2. **Armature attracted**: Magnetic field pulls armature toward coil
3. **Contacts switch**: Armature movement opens/closes contacts
4. **Coil de-energized**: Spring returns armature to rest position

### Contact Configurations

**SPST (Single Pole Single Throw)**:
- One switch, two terminals
- Simple on/off switching

**SPDT (Single Pole Double Throw)**:
- One switch, three terminals (Common, NO, NC)
- Can switch between two circuits

**DPST (Double Pole Single Throw)**:
- Two switches, four terminals
- Switch two circuits simultaneously

**DPDT (Double Pole Double Throw)**:
- Two switches, six terminals
- Switch two circuits between two states

**Terminology**:
- **NO (Normally Open)**: Contact open when coil de-energized
- **NC (Normally Closed)**: Contact closed when coil de-energized
- **COM (Common)**: Common terminal

## Types of Relays

### Electromechanical Relays (EMR)

**Standard relays**: Most common type, mechanical contacts

**Characteristics**:
- Audible click when switching
- Contact bounce (1-10ms)
- Mechanical wear over time
- Complete electrical isolation
- Can switch AC or DC loads

**Typical lifespan**: 100,000 to 1,000,000 operations

### Reed Relays

**Construction**: Glass-enclosed magnetic reed switches

**Characteristics**:
- Fast switching (< 1ms)
- Low contact bounce
- Hermetically sealed
- Lower current capacity than standard relays
- Long lifespan (billions of operations)

**Applications**: Test equipment, telecommunications, high-speed switching

### Solid State Relays (SSR)

**Construction**: Semiconductor switching (MOSFET, TRIAC, or thyristor)

**Advantages**:
- No mechanical parts, no wear
- Silent operation
- Very long lifespan (unlimited switching cycles)
- Fast switching (μs)
- No contact bounce
- Immune to shock and vibration

**Disadvantages**:
- Higher on-state voltage drop
- Heat generation
- More expensive
- Leakage current when off

**Applications**: Industrial control, heating systems, motor control

## Key Specifications

### Coil Specifications

**Coil Voltage**: Nominal operating voltage (5V, 12V, 24V common)

**Coil Current**: Current drawn by coil when energized

**Coil Resistance**: DC resistance of coil winding

**Pull-in Voltage**: Minimum voltage to activate relay (typically 70-80% of nominal)

**Drop-out Voltage**: Maximum voltage where relay releases (typically 10-50% of nominal)

### Contact Ratings

**Contact Voltage**: Maximum voltage contacts can switch

**Contact Current**: Maximum current contacts can handle
- **Resistive load**: Full rated current
- **Inductive load**: Derate to 30-50% of resistive rating
- **Lamp load**: Derate due to inrush current

**Switching Capacity**: Maximum power (VA or Watts)

**Example**: 10A @ 250VAC, 10A @ 30VDC

### Electrical Life

Number of switching operations before failure

**Factors affecting life**:
- Load type (resistive, inductive, capacitive)
- Switching frequency
- Contact current
- Environmental conditions

**Typical values**: 100,000 to 10,000,000 operations

### Switching Time

**Operate time**: Time from coil energization to contact closure (typically 5-15ms)

**Release time**: Time from coil de-energization to contact opening (typically 3-10ms)

### Isolation Voltage

Voltage withstand between coil and contacts

**Typical values**: 1kV to 4kV

## Common Applications

### Power Control

**High-power switching**: Control motors, heaters, pumps, lighting

**AC load switching**: Household appliances, HVAC systems

**Isolation**: Separate control circuit from high-voltage load

### Automation and Control

**Industrial control**: PLCs, process control, machinery

**Home automation**: Smart switches, thermostats, security systems

**Automotive**: Starter motors, horn, lights, fuel pump

### Signal Routing

**Audio/video switching**: Route signals between sources and destinations

**Test equipment**: Switch measurement paths, calibration circuits

**Telecommunications**: Line switching, routing

### Safety and Protection

**Emergency stop circuits**: Disconnect power in emergency

**Overcurrent protection**: Disconnect load when current exceeds limit

**Ground fault protection**: Detect and isolate ground faults

## Practical Considerations

### Relay Driver Circuits

**Problem**: Microcontroller outputs cannot directly drive relay coils (insufficient current)

**Solution**: Use transistor or MOSFET driver

**Basic BJT Driver Circuit**:
```
MCU GPIO → Base Resistor → NPN Transistor (Base)
                           Collector → Relay Coil → V+
                           Emitter → Ground
```

**Component selection**:
- **Base resistor**: Rb = (Vgpio - 0.7V) / (Icoil / β)
- **Transistor**: Must handle coil current (e.g., 2N2222, BC547)
- **Flyback diode**: Essential for inductive kickback protection

### Flyback Diode Protection

**Critical**: Always use flyback diode across relay coil

**Purpose**: Suppress voltage spike when coil is de-energized

**Connection**: Cathode to V+, anode to ground (reverse biased during normal operation)

**Diode selection**: 1N4001-1N4007 suitable for most relays

**Without flyback diode**: Voltage spike can destroy driver transistor

### Contact Protection

**Arc suppression**: Contacts can arc when switching inductive loads

**RC snubber**: Resistor and capacitor in series across contacts
- Typical values: 47Ω to 100Ω, 0.1μF to 0.47μF
- Reduces arcing, extends contact life

**MOV (Metal Oxide Varistor)**: Clamps voltage spikes across contacts

### Load Considerations

**Inductive loads**: Motors, solenoids, transformers
- Generate back-EMF when switched off
- Derate contact current to 30-50% of resistive rating
- Use contact protection

**Capacitive loads**: Power supplies, large capacitors
- High inrush current when switched on
- Can weld contacts if not properly rated

**Lamp loads**: Incandescent bulbs
- High inrush current (10-15× steady state)
- Derate contact current

### Common Mistakes to Avoid

- **No flyback diode**: Destroys driver transistor from inductive kickback
- **Insufficient driver current**: Relay doesn't fully activate, contacts chatter
- **Wrong coil voltage**: Relay doesn't activate or coil overheats
- **Exceeding contact ratings**: Causes contact welding or burning
- **No contact protection**: Reduces relay life with inductive loads
- **Ignoring contact bounce**: Can cause issues in timing-critical applications
- **Poor wire sizing**: Voltage drop affects relay operation

## Summary

Relays are electromechanical switches that enable low-power control circuits to switch high-power loads with electrical isolation. Understanding relay types, specifications, and proper driver circuits is essential for reliable switching applications.

**Key Takeaways**:
- Relays use electromagnet to mechanically switch contacts
- Contact configurations: SPST, SPDT, DPST, DPDT
- Electromechanical relays: Mechanical contacts, audible click, wear over time
- Solid state relays: No mechanical parts, silent, unlimited life, but higher cost
- **Always use flyback diode** across relay coil to protect driver circuit
- Use transistor or MOSFET driver for microcontroller control
- Derate contact current for inductive loads (30-50% of resistive rating)
- Contact protection (RC snubber, MOV) extends relay life
- Consider contact bounce (1-10ms) in timing-critical applications
- Typical coil voltages: 5V, 12V, 24V
- Electrical isolation between coil and contacts (1-4kV)

Proper relay selection based on coil voltage, contact ratings, load type, and switching frequency ensures reliable operation in power control and automation applications.

## References

- Relay construction and operating principles
- Relay manufacturer datasheets and specifications
- Inductive load switching and contact protection
- Relay driver circuit design

