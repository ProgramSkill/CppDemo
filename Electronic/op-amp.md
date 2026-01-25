# Operational Amplifier (Op-Amp)

## Overview

An **operational amplifier** (op-amp) is a high-gain differential amplifier with very high input impedance and low output impedance. Op-amps are among the most versatile and widely used integrated circuits in analog electronics, serving as building blocks for amplifiers, filters, comparators, and signal processing circuits.

## Ideal Op-Amp Characteristics

An ideal op-amp has the following properties:

1. **Infinite open-loop gain (A)**: Amplifies the difference between inputs infinitely
2. **Infinite input impedance**: Draws no current from input sources
3. **Zero output impedance**: Can drive any load without voltage drop
4. **Infinite bandwidth**: Amplifies all frequencies equally
5. **Zero offset voltage**: Output is zero when inputs are equal
6. **Infinite slew rate**: Output can change instantaneously

## Basic Operation

### Terminals

- **Non-inverting input (+)**: Positive input terminal
- **Inverting input (-)**: Negative input terminal
- **Output (Vout)**: Amplified signal output
- **Positive supply (V+)**: Positive power supply
- **Negative supply (V-)**: Negative power supply (or ground)

### Fundamental Equation

```
Vout = A × (V+ - V-)
```

Where:
- **Vout** = Output voltage
- **A** = Open-loop gain (typically 100,000 to 1,000,000)
- **V+** = Voltage at non-inverting input
- **V-** = Voltage at inverting input

### Golden Rules

When op-amp operates with negative feedback:

1. **No current flows into inputs**: Input impedance is very high
2. **Input voltages are equal**: V+ = V- (virtual short circuit)

## Common Op-Amp Configurations

### Inverting Amplifier

**Circuit**: Input connected to inverting input through resistor, non-inverting input grounded

**Gain Formula**:
```
Vout = -(Rf / Rin) × Vin
Gain = -Rf / Rin
```

**Characteristics**:
- Negative gain (180° phase shift)
- Input impedance = Rin
- Virtual ground at inverting input

**Example**: Rf = 100kΩ, Rin = 10kΩ → Gain = -10

### Non-Inverting Amplifier

**Circuit**: Input connected to non-inverting input, feedback to inverting input

**Gain Formula**:
```
Vout = (1 + Rf / Rin) × Vin
Gain = 1 + Rf / Rin
```

**Characteristics**:
- Positive gain (no phase shift)
- Very high input impedance
- Minimum gain = 1 (voltage follower)

**Example**: Rf = 90kΩ, Rin = 10kΩ → Gain = 10

### Voltage Follower (Buffer)

**Circuit**: Output connected directly to inverting input, input to non-inverting input

**Gain Formula**:
```
Vout = Vin
Gain = 1
```

**Characteristics**:
- Unity gain
- Very high input impedance
- Very low output impedance
- Excellent for impedance matching

### Differential Amplifier

**Circuit**: Two inputs, amplifies the difference between them

**Gain Formula**:
```
Vout = (Rf / Rin) × (V2 - V1)
```

**Characteristics**:
- Amplifies difference, rejects common-mode signals
- Used for noise rejection
- Instrumentation amplifiers use this principle

### Summing Amplifier

**Circuit**: Multiple inputs to inverting input through separate resistors

**Gain Formula**:
```
Vout = -Rf × (V1/R1 + V2/R2 + V3/R3 + ...)
```

**Applications**: Audio mixers, digital-to-analog converters (DAC)

### Integrator

**Circuit**: Capacitor in feedback path

**Output Formula**:
```
Vout = -(1 / (Rin × C)) × ∫Vin dt
```

**Applications**: Signal processing, waveform generation, analog computation

### Differentiator

**Circuit**: Capacitor at input, resistor in feedback

**Output Formula**:
```
Vout = -Rin × C × (dVin / dt)
```

**Applications**: Edge detection, high-pass filtering

### Comparator

**Circuit**: No feedback, operates in open-loop mode

**Operation**:
- If V+ > V-, output goes to positive rail
- If V+ < V-, output goes to negative rail

**Applications**: Voltage comparison, zero-crossing detection, ADC

## Key Specifications

### Open-Loop Gain (Aol)

Gain without feedback, typically 100,000 to 1,000,000 (100-120 dB)

**Importance**: Higher gain provides better accuracy in closed-loop circuits

### Gain-Bandwidth Product (GBW)

Product of gain and bandwidth remains constant

```
GBW = Gain × Bandwidth
```

**Example**: Op-amp with GBW = 1MHz
- At gain of 10: Bandwidth = 100kHz
- At gain of 100: Bandwidth = 10kHz

### Slew Rate (SR)

Maximum rate of output voltage change

```
SR = dVout/dt (V/μs)
```

**Importance**: Limits high-frequency, large-amplitude signals

**Example**: SR = 1V/μs can handle 10V peak-to-peak at ~16kHz

### Input Offset Voltage (Vos)

Output voltage when both inputs are at same voltage (ideally zero)

**Typical values**: 0.1mV to 10mV

**Impact**: Causes DC error in output

### Input Bias Current (Ib)

Current flowing into input terminals

**Typical values**: pA (FET input) to nA or μA (BJT input)

**Impact**: Causes voltage drop across source resistance

### Common-Mode Rejection Ratio (CMRR)

Ability to reject signals common to both inputs

```
CMRR = 20 × log(Adiff / Acm) (dB)
```

**Typical values**: 70-120 dB

**Higher is better**: Rejects noise and interference

### Power Supply Rejection Ratio (PSRR)

Ability to reject power supply voltage variations

**Typical values**: 60-100 dB

**Higher is better**: Output less affected by supply noise

## Common Applications

### Signal Amplification

**Audio amplifiers**: Microphone preamps, headphone amplifiers, line drivers

**Sensor signal conditioning**: Amplify weak sensor signals (thermocouples, strain gauges)

**Instrumentation amplifiers**: High-precision differential amplification

### Active Filters

**Low-pass filter**: Passes low frequencies, attenuates high frequencies

**High-pass filter**: Passes high frequencies, attenuates low frequencies

**Band-pass filter**: Passes specific frequency range

**Band-stop (notch) filter**: Rejects specific frequency range

**Advantages over passive filters**: Gain, no loading effects, sharper cutoff

### Signal Processing

**Precision rectifiers**: Full-wave and half-wave rectification without diode drop

**Peak detectors**: Capture and hold peak voltage

**Sample and hold**: Capture instantaneous voltage value

**Logarithmic amplifiers**: Compress wide dynamic range signals

### Oscillators

**Wien bridge oscillator**: Sine wave generation

**Phase-shift oscillator**: Audio frequency generation

**Relaxation oscillator**: Square/triangle wave generation

### Voltage References and Regulators

**Precision voltage reference**: Stable reference voltage

**Active voltage regulator**: Improved regulation over passive circuits

## Practical Considerations

### Power Supply Decoupling

**Always use bypass capacitors** near op-amp power pins:
- 0.1μF ceramic capacitor close to IC
- 10μF electrolytic for bulk filtering
- Prevents oscillation and improves stability

### Input Protection

**ESD protection**: Op-amp inputs are sensitive to static discharge

**Overvoltage protection**: Use series resistors and clamp diodes for inputs exceeding supply rails

### Stability and Oscillation

**Causes of instability**:
- Insufficient phase margin
- Capacitive loading
- Poor PCB layout

**Solutions**:
- Add compensation capacitor if needed
- Use series output resistor with capacitive loads
- Keep feedback path short
- Use ground plane

### Single vs Dual Supply

**Dual supply (±V)**: Output can swing positive and negative, true ground reference

**Single supply (0 to V+)**: Output referenced to V+/2, requires biasing, AC coupling

### Output Limitations

**Output voltage swing**: Cannot reach supply rails (rail-to-rail op-amps get close)

**Output current**: Typically 20-50mA maximum

**Short circuit protection**: Most op-amps have built-in protection

### Common Mistakes to Avoid

- **No power supply decoupling**: Causes oscillation and instability
- **Exceeding input voltage range**: Damages op-amp or causes latch-up
- **Ignoring bandwidth limitations**: Gain-bandwidth product limits performance
- **Capacitive loading without compensation**: Causes oscillation
- **Poor PCB layout**: Long feedback traces cause instability
- **Using comparator as op-amp or vice versa**: Different optimization for each
- **Forgetting input bias current**: Causes DC offset errors

## Summary

Operational amplifiers are versatile analog building blocks that enable a wide range of signal processing functions. Understanding op-amp characteristics, configurations, and limitations is essential for analog circuit design.

**Key Takeaways**:
- Op-amps amplify the difference between two inputs with very high gain
- Golden rules: No input current, input voltages equal (with negative feedback)
- Common configurations: Inverting (gain = -Rf/Rin), non-inverting (gain = 1+Rf/Rin)
- Key specs: Open-loop gain, GBW, slew rate, offset voltage, CMRR, PSRR
- Applications: Amplification, filtering, signal processing, oscillators
- Always use power supply decoupling capacitors
- Consider bandwidth limitations (GBW) and slew rate for signal requirements
- Proper PCB layout critical for stability

Proper op-amp selection based on gain-bandwidth product, slew rate, input characteristics, and power requirements ensures optimal circuit performance.

## References

- Op-amp theory and ideal characteristics
- Common op-amp IC datasheets (LM358, TL071, LM741, OPAx series)
- Active filter design principles
- Stability and compensation techniques

