# MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor)

## Overview

A **MOSFET** is a voltage-controlled semiconductor device used for switching and amplifying electronic signals. MOSFETs are the most common type of transistor in digital and power electronics, offering high input impedance, fast switching speeds, and efficient power handling. They are fundamental building blocks in microprocessors, power supplies, and motor control circuits.

## Basic Operation

### Voltage-Controlled Device

**Gate control**: Voltage applied to gate controls current flow between drain and source

**High input impedance**: Gate is insulated by oxide layer, draws virtually no current

**Three terminals**: Gate (G), Drain (D), Source (S)

**Channel formation**: Gate voltage creates conductive channel between drain and source

### N-Channel vs P-Channel

**N-Channel MOSFET**:
- Electrons are majority carriers
- Conducts when VGS > VTH (positive gate voltage)
- More common due to higher electron mobility
- Lower on-resistance for same die size

**P-Channel MOSFET**:
- Holes are majority carriers
- Conducts when VGS < VTH (negative gate voltage)
- Used in high-side switching and complementary circuits
- Higher on-resistance than N-channel

## Key Specifications

### Threshold Voltage (VTH or VGS(th))

Gate-source voltage required to turn on MOSFET

**Typical values**: 1V to 4V for standard MOSFETs, 0.5V to 2V for logic-level

**Logic-level MOSFETs**: Designed for 3.3V or 5V gate drive

**Temperature coefficient**: VTH decreases with increasing temperature

### On-Resistance (RDS(on))

Resistance between drain and source when fully on

**Typical values**: mΩ to Ω range depending on voltage rating and size

**Power dissipation**: P = I² × RDS(on)

**Lower is better**: Reduces conduction losses and heat

**VGS dependent**: RDS(on) decreases with higher gate voltage

### Maximum Drain Current (ID)

Maximum continuous current through drain-source

**Typical values**: 1A to 200A+ for power MOSFETs

**Pulsed current**: Higher than continuous rating

**Temperature limited**: Derate at high temperatures

### Drain-Source Voltage (VDS)

Maximum voltage between drain and source when off

**Typical values**: 20V, 30V, 60V, 100V, 200V, 600V, 1000V

**Breakdown voltage**: MOSFET fails if exceeded

**Safety margin**: Choose VDS > 2× maximum circuit voltage

## Common Applications

### Power Switching

**DC-DC converters**: Buck, boost, buck-boost topologies

**Motor control**: PWM speed control, H-bridge drivers

**Load switching**: High-side and low-side switches

**Efficiency**: Low RDS(on) reduces power loss

### Digital Logic

**CMOS circuits**: Complementary N-channel and P-channel pairs

**Microprocessors**: Billions of MOSFETs in modern CPUs

**Memory**: DRAM, Flash memory cells

**Low power**: Minimal static power consumption

### Linear Amplification

**Source follower**: Unity-gain buffer with high input impedance

**Common source amplifier**: Voltage amplification

**Differential pairs**: Op-amp input stages

### Reverse Polarity Protection

**P-channel high-side**: Protects against reverse voltage

**Low voltage drop**: Better than diode protection

**Active control**: Can be switched on/off

## Practical Considerations

### Gate Drive Requirements

**Gate charge (Qg)**: Total charge needed to switch MOSFET

**Drive current**: Higher current = faster switching

**Gate resistor**: Limits switching speed and reduces ringing

**Logic-level vs standard**: Logic-level MOSFETs work with 3.3V/5V, standard need 10V+

### Body Diode

**Intrinsic diode**: Built-in diode from source to drain

**Freewheeling**: Conducts reverse current in inductive loads

**Slow recovery**: Body diode has poor reverse recovery compared to external diode

**Solution**: Add external fast diode in parallel for high-frequency switching

### Gate Protection

**ESD sensitive**: Gate oxide can be damaged by static discharge

**Gate-source voltage limit**: Typically ±20V maximum

**Protection**: Zener diode or TVS across gate-source

**Handling**: Use ESD precautions when handling MOSFETs

### Common Mistakes to Avoid

- **Insufficient gate voltage**: MOSFET not fully on, high RDS(on), overheating
- **No gate resistor**: Excessive ringing and EMI
- **Exceeding VDS rating**: Breakdown and failure
- **Floating gate**: Unpredictable behavior, possible damage
- **No heat sink**: Thermal runaway in high-power applications
- **Ignoring gate charge**: Slow switching, high switching losses

## Summary

MOSFETs are voltage-controlled transistors offering high input impedance, fast switching, and efficient power handling. Understanding gate drive requirements, on-resistance, and voltage ratings is essential for proper MOSFET selection and circuit design.

**Key Takeaways**:
- Voltage-controlled device: Gate voltage controls drain-source current
- High input impedance: Gate draws virtually no current
- N-channel vs P-channel: N-channel more common, lower RDS(on)
- Threshold voltage (VTH): 1-4V standard, 0.5-2V logic-level
- On-resistance (RDS(on)): mΩ to Ω, determines conduction losses
- Maximum ratings: VDS (20V-1000V), ID (1A-200A+)
- Applications: Power switching, digital logic, amplification, protection
- Gate charge (Qg): Determines switching speed
- Body diode: Intrinsic diode, slow recovery
- ESD sensitive: Requires gate protection
- Logic-level MOSFETs: Work with 3.3V/5V gate drive
- Choose VDS > 2× maximum circuit voltage
- Lower RDS(on) = higher efficiency, less heat

Proper MOSFET selection based on voltage rating, current capability, on-resistance, and gate drive requirements ensures efficient and reliable operation in power switching and control applications.

## References

- MOSFET theory and operation principles
- Gate drive circuit design
- MOSFET manufacturer datasheets and specifications
- Power MOSFET application notes


