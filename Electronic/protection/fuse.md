# Fuse

## Overview

A **fuse** is a sacrificial overcurrent protection device that protects electrical circuits and equipment from damage caused by excessive current. When current exceeds the fuse's rating, the fuse element melts and opens the circuit, preventing damage to downstream components.

## Basic Operation

### Principle

**Fuse element**: Thin wire or metal strip that melts when excessive current flows through it

**Heat generation**: Current flowing through resistance generates heat (P = I²R)

**Melting point**: When current exceeds rating, heat generation exceeds dissipation, causing element to melt

**Circuit interruption**: Melted element creates open circuit, stopping current flow

### Fuse Characteristics

**Normal operation**: Current below rating, fuse conducts with minimal voltage drop

**Overload condition**: Current exceeds rating, fuse heats up and eventually melts

**Short circuit**: Very high current causes rapid melting (milliseconds)

**One-time protection**: Fuse must be replaced after operation (except resettable fuses)

## Types of Fuses

### Glass Tube Fuses

**Construction**: Fuse element enclosed in glass tube

**Characteristics**:
- Visual inspection possible (can see if blown)
- Common sizes: 5×20mm, 6×30mm
- Current ratings: 100mA to 10A typical
- Voltage ratings: 250V, 500V

**Applications**: Consumer electronics, power supplies, test equipment

### Ceramic Fuses

**Construction**: Fuse element in ceramic body filled with sand

**Characteristics**:
- Higher breaking capacity than glass
- Better arc suppression
- Cannot visually inspect
- More robust construction

**Applications**: High-voltage circuits, industrial equipment

### Blade Fuses (Automotive)

**Construction**: Plastic body with two metal blade terminals

**Types**:
- **Mini**: 10.9×3.6×16.3mm
- **Standard (ATO/ATC)**: 19.1×5.1×18.5mm
- **Maxi**: 29.2×8.5×34.3mm

**Color coding**: Different colors indicate current ratings

**Applications**: Automotive, marine, RV electrical systems

### Surface Mount Fuses (SMD)

**Construction**: Chip-style fuse for PCB mounting

**Characteristics**:
- Very small size (0603, 1206, 2410 packages)
- Fast response time
- Low profile
- Automated assembly compatible

**Applications**: Mobile devices, compact electronics, PCB protection

### Resettable Fuses (PPTC - Polymeric Positive Temperature Coefficient)

**Construction**: Polymer with conductive particles

**Operation**:
- Normal: Low resistance, conducts current
- Fault: Heats up, expands, resistance increases dramatically
- Recovery: Cools down, resistance returns to normal (resettable)

**Characteristics**:
- Self-resetting (no replacement needed)
- Slower response than traditional fuses
- Higher resistance in normal operation
- Trip time increases with age

**Applications**: USB ports, battery protection, automotive circuits

**Common names**: PolySwitch, Polyfuse, Multifuse

## Key Specifications

### Current Rating (In)

Nominal current the fuse can carry continuously without blowing

**Important**: Derate for ambient temperature and mounting conditions

**Example**: 1A, 2A, 5A, 10A, 15A, 20A

### Voltage Rating

Maximum voltage the fuse can safely interrupt

**AC voltage**: 125V, 250V, 600V common
**DC voltage**: Often lower than AC rating (harder to interrupt DC arc)

**Important**: Never exceed voltage rating

### Breaking Capacity (Interrupting Rating)

Maximum fault current the fuse can safely interrupt

**Typical values**: 35A, 100A, 1500A, 10kA

**Important**: Must exceed maximum available fault current in circuit

### Time-Current Characteristics

Relationship between current and time to blow

**I²t Rating**: Energy required to melt fuse element (Ampere² × seconds)

**Melting time**: Time for fuse to blow at given overcurrent

### Fuse Speed (Blow Characteristics)

**Fast-acting (F)**: Blows quickly, minimal time delay
- Used for sensitive electronics
- Quick response to overcurrent

**Medium-acting (M)**: Moderate time delay
- General purpose applications

**Slow-blow / Time-delay (T)**: Tolerates temporary surge currents
- Used with inductive loads (motors, transformers)
- Allows startup inrush current
- Blows on sustained overcurrent

**Example**: T3.15A = Time-delay, 3.15A rating

## Common Applications

### Power Supply Protection

**Input protection**: Protect against mains overcurrent and short circuits

**Output protection**: Protect load from power supply failure

### Equipment Protection

**Appliances**: Protect internal circuits from faults

**Industrial equipment**: Protect motors, transformers, control circuits

### Battery Protection

**Overcurrent protection**: Prevent excessive discharge current

**Short circuit protection**: Protect battery from damage

### PCB Protection

**Trace protection**: Prevent PCB trace damage from overcurrent

**Component protection**: Protect sensitive ICs and components

## Practical Considerations

### Fuse Selection

**Current rating**: Choose 125-150% of normal operating current
- Too low: Nuisance blowing during normal operation
- Too high: Inadequate protection

**Voltage rating**: Must exceed maximum circuit voltage

**Breaking capacity**: Must exceed maximum available fault current

**Speed**: Match to load characteristics
- Fast-acting for electronics
- Slow-blow for motors, transformers

### Fuse Placement

**Location**: Install at power source, before load

**Accessibility**: Easy to inspect and replace

**Both poles**: For AC circuits, fuse both live and neutral (or use double-pole fuse)

### Voltage Drop

Fuses have small resistance, causing voltage drop

**Typical**: 0.1V to 0.5V at rated current

**Impact**: Consider in low-voltage circuits

### Aging and Degradation

**Thermal cycling**: Repeated heating/cooling weakens fuse element

**Vibration**: Can cause premature failure

**Corrosion**: Moisture and contaminants affect contacts

**Recommendation**: Replace fuses periodically in critical applications

### Common Mistakes to Avoid

- **Oversizing fuse**: Provides inadequate protection
- **Using wrong voltage rating**: Dangerous, can cause fire
- **Bypassing fuse**: Defeats protection, extremely dangerous
- **Replacing with wire**: Never acceptable, fire hazard
- **Ignoring breaking capacity**: Fuse may explode during fault
- **Wrong speed rating**: Fast fuse on motor causes nuisance blowing

## Summary

Fuses are sacrificial overcurrent protection devices that protect circuits and equipment from damage caused by excessive current. Understanding fuse types, specifications, and proper selection is essential for safe and reliable circuit protection.

**Key Takeaways**:
- Fuses protect by melting when current exceeds rating
- Types: Glass tube, ceramic, blade (automotive), SMD, resettable (PPTC)
- Key specs: Current rating, voltage rating, breaking capacity, speed
- Speed ratings: Fast-acting (F), medium (M), slow-blow/time-delay (T)
- Select fuse at 125-150% of normal operating current
- Voltage rating must exceed maximum circuit voltage
- Breaking capacity must exceed maximum fault current
- Use slow-blow fuses for inductive loads (motors, transformers)
- Resettable fuses (PPTC) self-reset but have slower response
- Never bypass or replace fuse with wire - fire hazard
- Install fuse at power source, before load

Proper fuse selection based on current rating, voltage rating, breaking capacity, and speed characteristics ensures effective overcurrent protection and circuit safety.

## References

- Fuse construction and operating principles
- IEC 60269: Low-voltage fuses
- UL 248: Fuses for use in electrical equipment
- Fuse manufacturer datasheets and application notes

