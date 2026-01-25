# Capacitor

## Overview

A **capacitor** is a passive two-terminal electrical component that stores electrical energy in an electric field. Capacitors are one of the most fundamental components in electronics, used for energy storage, filtering, coupling, decoupling, timing, and signal processing.

## What is Capacitance?

**Capacitance** is the ability to store electrical charge. It is measured in **Farads (F)**, named after English physicist Michael Faraday.

Common units:
- **Farad (F)**: Base unit (very large)
- **Millifarad (mF)**: 10⁻³ F
- **Microfarad (μF or uF)**: 10⁻⁶ F (common)
- **Nanofarad (nF)**: 10⁻⁹ F (common)
- **Picofarad (pF)**: 10⁻¹² F (common)

### Basic Formula

The relationship between charge (Q), voltage (V), and capacitance (C):

```
Q = C × V
```

Where:
- **Q** = Charge (Coulombs)
- **C** = Capacitance (Farads)
- **V** = Voltage (Volts)

### Energy Storage

Energy stored in a capacitor:

```
E = ½ × C × V²
```

Where:
- **E** = Energy (Joules)
- **C** = Capacitance (Farads)
- **V** = Voltage (Volts)

## Physical Construction

A capacitor consists of two conductive plates separated by an insulating material called a **dielectric**.

**Key parameters**:
- **Plate area**: Larger area = higher capacitance
- **Distance between plates**: Smaller distance = higher capacitance
- **Dielectric material**: Different materials have different permittivity

### Capacitance Formula

```
C = ε₀ × εᵣ × A / d
```

Where:
- **ε₀** = Permittivity of free space (8.854 × 10⁻¹² F/m)
- **εᵣ** = Relative permittivity of dielectric material
- **A** = Plate area (m²)
- **d** = Distance between plates (m)

## Types of Capacitors

### Ceramic Capacitors

**Construction**: Ceramic dielectric material between metal plates

**Characteristics**:
- Small size, low cost
- Non-polarized (can be connected either way)
- Wide capacitance range: 1pF to 100μF
- Good high-frequency performance
- Temperature-dependent capacitance

**Classes**:
- **Class 1 (C0G/NP0)**: Stable, low loss, ±30ppm/°C, precision applications
- **Class 2 (X7R, X5R)**: Higher capacitance, ±15% tolerance, general purpose
- **Class 3 (Y5V, Z5U)**: Highest capacitance, -20% to +80% tolerance, coupling/decoupling

**Applications**: Decoupling, filtering, timing, RF circuits

### Electrolytic Capacitors

**Construction**: Aluminum or tantalum with oxide dielectric layer

**Characteristics**:
- **Polarized** (must observe polarity - negative lead marked)
- High capacitance: 1μF to 100,000μF
- Relatively large size
- Limited lifespan (electrolyte dries out)
- Higher ESR (Equivalent Series Resistance)

**Types**:
- **Aluminum electrolytic**: Most common, bulk energy storage
- **Tantalum**: Smaller size, more stable, more expensive
- **Polymer**: Low ESR, longer life, better performance

**Applications**: Power supply filtering, energy storage, audio coupling

**Warning**: Reverse polarity or overvoltage can cause explosion!

### Film Capacitors

**Construction**: Thin plastic film dielectric (polyester, polypropylene, etc.)

**Characteristics**:
- Non-polarized
- Excellent stability and reliability
- Low ESR and ESL (Equivalent Series Inductance)
- Long lifespan
- Self-healing properties
- Capacitance range: 100pF to 100μF

**Types**:
- **Polyester (PET)**: General purpose, economical
- **Polypropylene (PP)**: Low loss, high frequency, audio applications
- **Polystyrene (PS)**: High precision, low temperature coefficient
- **PTFE (Teflon)**: High temperature, RF applications

**Applications**: Audio circuits, timing, power electronics, snubber circuits

### Supercapacitors (Ultracapacitors)

**Construction**: Electrochemical double-layer capacitor

**Characteristics**:
- Extremely high capacitance: 0.1F to 1000F+
- Low voltage rating: typically 2.5V to 5.5V per cell
- Bridge between capacitors and batteries
- Fast charge/discharge
- Long cycle life (millions of cycles)

**Applications**: Energy harvesting, backup power, peak power delivery, regenerative braking

### Variable Capacitors

**Types**:
- **Trimmer capacitors**: Small adjustable capacitors for tuning
- **Varactor diodes**: Voltage-controlled capacitance
- **Tuning capacitors**: Mechanical adjustment for radio tuning

**Applications**: Oscillator tuning, frequency adjustment, impedance matching

## Capacitor Markings and Codes

### Ceramic Capacitor Codes

**3-Digit Code** (most common):
- First 2 digits: significant figures
- Last digit: multiplier (number of zeros)
- Unit: picofarads (pF)

**Examples**:
- **104** = 10 × 10⁴ pF = 100,000pF = 100nF = 0.1μF
- **223** = 22 × 10³ pF = 22,000pF = 22nF = 0.022μF
- **475** = 47 × 10⁵ pF = 4,700,000pF = 4.7μF

**Letter codes**:
- **p** = picofarads (e.g., 4p7 = 4.7pF)
- **n** = nanofarads (e.g., 10n = 10nF)
- **μ or u** = microfarads (e.g., 2u2 = 2.2μF)

### Tolerance Codes

| Letter | Tolerance |
|--------|-----------|
| B | ±0.1pF |
| C | ±0.25pF |
| D | ±0.5pF |
| F | ±1% |
| G | ±2% |
| J | ±5% |
| K | ±10% |
| M | ±20% |
| Z | +80%, -20% |

### Voltage Rating

Always marked on electrolytic capacitors (e.g., 16V, 25V, 50V).
Ceramic capacitors may have voltage codes or ratings printed.

## Capacitor Configurations

### Series Connection

When capacitors are connected in series, the total capacitance decreases:

```
1/C_total = 1/C1 + 1/C2 + 1/C3 + ... + 1/Cn
```

For two capacitors in series:
```
C_total = (C1 × C2) / (C1 + C2)
```

**Characteristics**:
- Total capacitance is less than the smallest capacitor
- Voltage divides across capacitors
- Same charge on all capacitors
- Used to increase voltage rating

**Example**: Two 100μF capacitors in series = 50μF

### Parallel Connection

When capacitors are connected in parallel, the total capacitance is the sum:

```
C_total = C1 + C2 + C3 + ... + Cn
```

**Characteristics**:
- Total capacitance increases
- Same voltage across all capacitors
- Charge divides among capacitors
- Used to increase total capacitance

**Example**: 100μF + 220μF + 470μF = 790μF

## Common Applications

### Power Supply Filtering

**Smoothing/Bulk Capacitors**: Large electrolytic capacitors (100μF - 10,000μF) smooth rectified AC voltage.

**Decoupling Capacitors**: Placed near IC power pins (0.1μF ceramic) to filter high-frequency noise and provide local energy storage.

**Bypass Capacitors**: Remove AC noise from DC power lines.

### AC Coupling

Block DC voltage while allowing AC signals to pass. Common in audio circuits and amplifier stages.

```
f_cutoff = 1 / (2π × R × C)
```

### Timing Circuits

Combined with resistors to create time delays and oscillators.

**RC Time Constant**:
```
τ = R × C
```

After time τ, capacitor charges to 63.2% of supply voltage.

### Filtering

**Low-pass filter**: Passes low frequencies, blocks high frequencies
**High-pass filter**: Passes high frequencies, blocks low frequencies
**Band-pass filter**: Passes specific frequency range

### Energy Storage

- Camera flash circuits
- Power backup systems
- Pulse power applications
- Motor starting

### Signal Processing

- Tuning circuits (resonance with inductors)
- Phase shifting
- Impedance matching
- Snubber circuits (suppress voltage spikes)

## Practical Considerations

### ESR (Equivalent Series Resistance)

Real capacitors have internal resistance that causes power loss and heating.

- **Low ESR**: Better for high-frequency applications, power supplies
- **High ESR**: Can cause ripple voltage, reduced efficiency
- **Polymer and ceramic**: Typically low ESR
- **Standard electrolytic**: Higher ESR

### ESL (Equivalent Series Inductance)

Parasitic inductance limits high-frequency performance.

- Important in RF and high-speed digital circuits
- Shorter leads = lower ESL
- SMD capacitors have lower ESL than through-hole

### Voltage Derating

Always use capacitors with voltage rating higher than maximum circuit voltage.

**Recommended derating**:
- **Ceramic**: 50% (use 10V cap for 5V circuit)
- **Electrolytic**: 20-50% (use 25V cap for 16V circuit)
- **Film**: 30-50%

### Temperature Effects

Capacitance changes with temperature based on dielectric type.

- **C0G/NP0**: ±30ppm/°C (very stable)
- **X7R**: ±15% over -55°C to +125°C
- **Y5V**: -82% to +22% over -30°C to +85°C

### Aging and Lifespan

**Electrolytic capacitors**:
- Limited lifespan (1,000 - 10,000 hours at rated temperature)
- Electrolyte evaporates over time
- Higher temperature = shorter life
- Rule of thumb: Life doubles for every 10°C reduction

**Ceramic and film**: Very long lifespan (decades)

### Common Mistakes to Avoid

- **Wrong polarity on electrolytic**: Can cause explosion
- **Insufficient voltage rating**: Leads to dielectric breakdown
- **Using high-ESR caps in switching supplies**: Causes excessive ripple and heating
- **Ignoring temperature coefficient**: Capacitance drift in precision circuits
- **Inadequate decoupling**: Place 0.1μF ceramic caps close to IC power pins
- **Mixing capacitor types incorrectly**: Use appropriate type for each application

## Summary

Capacitors are essential energy storage components used throughout electronics for filtering, coupling, timing, and signal processing. Understanding capacitor types, specifications, and proper application is critical for reliable circuit design.

**Key Takeaways**:
- Capacitance (C = Q/V) measured in Farads
- Electrolytic capacitors are polarized and have high capacitance
- Ceramic capacitors are non-polarized, small, and good for high frequency
- Film capacitors offer excellent stability and reliability
- Always derate voltage ratings (50% for ceramic, 20-50% for electrolytic)
- Consider ESR and ESL for high-frequency applications
- Use proper decoupling techniques (0.1μF ceramic near ICs)
- Series connection decreases capacitance, parallel increases it

Proper capacitor selection based on capacitance value, voltage rating, ESR, temperature stability, and lifespan requirements ensures optimal circuit performance.

## References

- Capacitor dielectric types and characteristics
- IEC 60384: Fixed capacitors for use in electronic equipment
- Manufacturer datasheets for specific component specifications

