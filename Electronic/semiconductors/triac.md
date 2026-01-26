# TRIAC (Triode for Alternating Current)

## Overview

A **TRIAC** (Triode for Alternating Current) is a three-terminal semiconductor device that can conduct current in both directions when triggered. It is essentially two SCRs connected in anti-parallel, allowing bidirectional control of AC power. TRIACs are widely used in AC power control applications such as light dimmers, motor speed controllers, and heating element control.

## Basic Structure

### Three Terminals

**MT1 (Main Terminal 1)**: First main current terminal

**MT2 (Main Terminal 2)**: Second main current terminal

**Gate (G)**: Control terminal for triggering

### Bidirectional Structure

**Construction**: Two SCRs in anti-parallel configuration

**Current flow**: Can conduct in both directions between MT1 and MT2

**Triggering**: Can be triggered by positive or negative gate current

## Basic Operation

### Four Quadrant Operation

TRIACs can be triggered in four different quadrants based on MT2 voltage polarity and gate current polarity:

**Quadrant I (I+)**: MT2 positive, Gate positive

**Quadrant II (I-)**: MT2 positive, Gate negative

**Quadrant III (III+)**: MT2 negative, Gate positive

**Quadrant IV (III-)**: MT2 negative, Gate negative

### Triggering and Conduction

**OFF state**: TRIAC blocks current in both directions

**Triggering**: Apply gate current pulse (positive or negative)

**ON state**: TRIAC conducts current in direction determined by MT2-MT1 voltage

**Latching**: Remains ON after gate signal removed (like SCR)

**Turn-off**: Current must fall below holding current (natural at AC zero crossing)

## Key Specifications

### Voltage Rating (VDRM/VRRM)

Maximum repetitive peak off-state voltage.

**Typical values**: 200V to 800V

**Selection**: Choose rating > 1.5× peak AC voltage for safety margin

### Current Rating (IT(RMS))

Maximum RMS on-state current.

**Typical values**: 1A to 40A

**Derating**: Reduce current at elevated temperatures

### Gate Trigger Current (IGT)

Minimum gate current to turn ON TRIAC.

**Typical values**: 5mA to 50mA

**Varies by quadrant**: Different sensitivity in each quadrant

### Holding Current (IH)

Minimum main terminal current to keep TRIAC conducting.

**Typical values**: 10mA to 50mA

**Importance**: Must exceed IH to maintain conduction

### dV/dt Rating

Maximum rate of voltage rise without false triggering.

**Typical values**: 10V/μs to 500V/μs

**Protection**: Use snubber circuit to limit dV/dt

## Phase Control

### AC Power Control Principle

**Phase angle control**: Delay TRIAC triggering within each AC half-cycle

**Firing angle (α)**: Delay angle from zero crossing to trigger point

**Conduction angle**: Portion of cycle where TRIAC conducts

**Power control**: Average power = f(firing angle)

### Power Calculation

**RMS voltage**: VRMS = VAC × √[(π - α + sin(2α)/2) / π]

**Power delivered**: P ≈ Pmax × (1 - α/π) for resistive loads

**Full power**: α = 0° (trigger at zero crossing)

**Half power**: α ≈ 90°

**Minimum power**: α ≈ 180° (no conduction)

## Common Applications

### Light Dimmers

**Purpose**: Control lamp brightness by varying AC power

**Operation**: Phase angle control adjusts average power to lamp

**Typical circuit**: TRIAC with DIAC trigger circuit and potentiometer

### Motor Speed Control

**Purpose**: Control AC motor speed

**Operation**: Reduce voltage to motor using phase control

**Applications**: Fans, drills, kitchen appliances

### Heating Control

**Purpose**: Control electric heaters and heating elements

**Operation**: Proportional power control for temperature regulation

**Applications**: Ovens, water heaters, soldering irons

### Soft Start Circuits

**Purpose**: Gradually increase voltage to reduce inrush current

**Operation**: Start with large firing angle, gradually reduce to zero

**Applications**: Motor starting, transformer energizing, lamp protection

## Practical Considerations

### Snubber Circuit

**Purpose**: Protect TRIAC from dV/dt triggering and voltage spikes

**Typical values**: 100Ω resistor + 100nF capacitor in series across MT1-MT2

**Importance**: Essential for inductive loads and reliable operation

### Heat Sinking

**Power dissipation**: P = VT × IT(avg)

**On-state voltage drop**: Typically 1-2V

**Heat sink required**: For currents above 1-2A

### Gate Triggering

**DIAC trigger**: Common method for symmetric triggering in both directions

**Microcontroller control**: Zero-crossing detection + optoisolator for isolation

**Minimum pulse width**: Typically 10-20μs

### Common Mistakes to Avoid

- **No snubber circuit**: False triggering from dV/dt, especially with inductive loads
- **Insufficient gate current**: TRIAC fails to trigger reliably
- **Wrong quadrant operation**: Some TRIACs have poor sensitivity in certain quadrants
- **Exceeding current rating**: Overheating and thermal runaway
- **No heat sink**: Thermal shutdown or device failure at high currents
- **Inductive load without protection**: Voltage spikes can damage TRIAC

## Summary

TRIACs (Triode for Alternating Current) are bidirectional thyristors that enable AC power control by conducting current in both directions when triggered. Essentially two SCRs in anti-parallel, TRIACs are ideal for phase angle control applications such as light dimmers, motor speed controllers, and heating element control.

**Key Takeaways**:
- Bidirectional thyristor: Conducts current in both directions
- Three terminals: MT1, MT2, and Gate
- Four quadrant operation: Can be triggered with positive or negative gate current
- Latching behavior: Remains ON after gate signal removed until current drops below holding current
- Phase angle control: Varies AC power by delaying trigger point in each half-cycle
- Key specs: Voltage rating (VDRM), current rating (IT(RMS)), gate trigger current (IGT), holding current (IH), dV/dt rating
- Applications: Light dimmers, motor speed control, heating control, soft start circuits
- Snubber circuit essential: Protects from dV/dt triggering and voltage spikes
- Heat sinking required: For currents above 1-2A
- Common trigger methods: DIAC for simple circuits, microcontroller with optoisolator for precise control

Proper TRIAC selection based on voltage rating, current capacity, and gate sensitivity ensures reliable AC power control with appropriate snubber circuits and thermal management.

## References

- TRIAC operation principles and four quadrant triggering
- Common TRIAC datasheets (BT series, MAC series, TIC series)
- Phase angle control and power calculation
- Snubber circuit design for inductive loads
- DIAC trigger circuits and zero-crossing detection


