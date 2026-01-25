# Varistor (Metal Oxide Varistor - MOV)

## Overview

A **varistor** (voltage-dependent resistor) is a nonlinear protective device whose resistance decreases dramatically when voltage exceeds a threshold. Metal Oxide Varistors (MOVs) are the most common type, used to protect circuits from voltage transients, surges, and spikes by clamping excessive voltages.

## Basic Operation

### Voltage-Dependent Resistance

**Principle**: Resistance varies with applied voltage

**Normal operation**: High resistance (MΩ range), minimal current flow

**Overvoltage condition**: Resistance drops to low value (Ω range), conducts surge current

**Clamping action**: Limits voltage to safe level by shunting excess current to ground

### Nonlinear I-V Characteristic

**Low voltage region**: Very high resistance, acts as open circuit

**Threshold voltage**: Varistor begins to conduct

**High current region**: Low resistance, clamps voltage

**Bidirectional**: Works for both positive and negative voltage spikes

## MOV Construction and Materials

### Metal Oxide Composition

**Primary material**: Zinc oxide (ZnO) ceramic with additives

**Grain structure**: Microscopic ZnO grains separated by grain boundaries

**Conduction mechanism**: Electron tunneling across grain boundaries at high voltage

**Disc or block form**: Compressed and sintered ceramic material

### Electrode Configuration

**Metal electrodes**: Applied to both faces of ceramic disc

**Lead wires**: Attached to electrodes for circuit connection

**Epoxy coating**: Protective insulation and moisture barrier

**Color coding**: Often blue or orange for identification

## Key Specifications

### Varistor Voltage (VN)

Voltage at which varistor begins to conduct (measured at 1mA)

**Common values**: 14V, 25V, 130V, 275V, 385V, 510V, 680V

**Selection**: Choose VN 1.2-1.5× higher than maximum circuit operating voltage

**Example**: For 120VAC circuit (170V peak), use 275V MOV

### Maximum Clamping Voltage (VC)

Peak voltage across varistor during surge event

**Depends on**: Surge current magnitude and varistor size

**Critical parameter**: Must be below protected circuit's damage threshold

**Typical ratio**: VC ≈ 2-3× VN at rated surge current

### Energy Rating (Joules)

Maximum energy the varistor can absorb in a single pulse

**Typical values**: 1J to 1000J depending on size

**Larger disc**: Higher energy absorption capability

**Multiple surges**: Cumulative energy degrades varistor over time

### Peak Surge Current (Imax)

Maximum current the varistor can handle for a single 8/20μs pulse

**Typical values**: 400A to 70,000A

**8/20μs waveform**: Standard surge test waveform (8μs rise, 20μs to half value)

**Larger varistor**: Higher surge current capability

## Common Applications

### AC Power Line Protection

**Mains surge protection**: Protect equipment from lightning and switching transients

**Installation**: Connected line-to-neutral and line-to-ground

**Typical ratings**: 275V or 385V MOV for 120/240VAC systems

**Surge protector strips**: Multiple MOVs for comprehensive protection

### DC Power Supply Protection

**Input protection**: Shield power supplies from voltage spikes

**Automotive**: Protect electronics from load dump transients (12V/24V systems)

**Battery systems**: Prevent overvoltage damage

### Signal Line Protection

**Data lines**: Protect communication interfaces (RS-232, RS-485)

**Telephone lines**: Lightning and power cross protection

**Lower voltage MOVs**: 14V, 25V for signal-level protection

### Equipment Protection

**Motor circuits**: Suppress inductive kickback and switching transients

**Relay coils**: Protect against back-EMF

**Transformer primary**: Lightning and surge protection

## Practical Considerations

### Varistor Selection

**Voltage rating**: VN should be 1.2-1.5× maximum circuit voltage

**AC circuits**: Consider peak voltage (Vpeak = Vrms × 1.414)

**Energy rating**: Must exceed expected surge energy

**Clamping voltage**: Must be below protected circuit's damage threshold

**Example**: 120VAC circuit (170V peak) → use 275V MOV (VN = 275V)

### Series Fuse or Thermal Cutoff

**Problem**: Repeated surges degrade MOV, can cause thermal runaway

**Solution**: Series fuse or thermal cutoff disconnects failed MOV

**Thermal fuse**: Opens if MOV overheats from degradation

**Safety**: Prevents fire hazard from failed varistor

### Lead Length and Placement

**Short leads**: Minimize inductance for fast response

**Close to protected circuit**: Reduce lead inductance effects

**Typical**: Keep leads < 10cm for best performance

### Degradation and Aging

**Surge exposure**: Each surge slightly degrades MOV performance

**Leakage current increase**: Indicates degradation

**End of life**: MOV may short circuit or open (with thermal fuse)

**Replacement**: Consider periodic replacement in critical applications

### Common Mistakes to Avoid

- **Wrong voltage rating**: Too low causes premature conduction, too high provides inadequate protection
- **No series fuse**: Failed MOV can overheat and cause fire
- **Ignoring clamping voltage**: VC must be below circuit damage threshold
- **Long lead lengths**: Increases inductance, reduces effectiveness
- **Single MOV on 3-phase**: Need MOVs on all phases and phase-to-ground
- **Exceeding energy rating**: Causes catastrophic failure
- **No consideration for continuous AC voltage**: MOV must withstand continuous operating voltage

## Summary

Varistors (MOVs) are voltage-dependent resistors that provide fast, effective protection against voltage transients and surges by clamping excessive voltages. Understanding varistor specifications, proper selection, and installation practices is essential for reliable circuit protection.

**Key Takeaways**:
- Voltage-dependent resistance: High resistance at normal voltage, low resistance during surge
- Metal oxide construction: Zinc oxide (ZnO) ceramic with grain boundary conduction
- Bidirectional protection: Works for both positive and negative voltage spikes
- Varistor voltage (VN): Choose 1.2-1.5× maximum circuit operating voltage
- Common ratings: 14V, 25V, 130V, 275V, 385V, 510V, 680V
- Clamping voltage (VC): Must be below protected circuit's damage threshold
- Energy rating: 1J to 1000J depending on size
- Peak surge current: 400A to 70,000A for 8/20μs pulse
- Applications: AC power line protection, DC supply protection, signal line protection
- Series fuse required: Prevents fire hazard from failed MOV
- Short lead lengths: Keep < 10cm to minimize inductance
- Degradation: Each surge slightly degrades performance
- AC circuits: Consider peak voltage (Vpeak = Vrms × 1.414)
- Thermal protection: Use thermal fuse or cutoff for safety

Proper varistor selection based on voltage rating, energy capacity, clamping voltage, and surge current capability ensures effective transient protection in power and signal circuits.

## References

- Metal oxide varistor theory and operation
- Surge protection standards (IEEE, IEC)
- MOV manufacturer datasheets and application notes
- Transient voltage suppression design guidelines


