# Serial Port Examples

## Overview

This section demonstrates serial communication using the 8051's built-in UART (Universal Asynchronous Receiver/Transmitter). Serial communication is essential for:

- Debugging and logging
- PC interfacing
- Module communication (Bluetooth, WiFi, GPS)
- Multi-processor systems

## Serial Communication Basics

**The 8051 UART provides:**
- Full-duplex communication (simultaneous TX and RX)
- Programmable baud rates
- 4 operating modes
- Interrupt-driven or polled operation

**Hardware Connection:**
```
8051                  Level Shifter              PC
P3.1 (TXD) ──────→   MAX232/TTL       ──────→   RXD
P3.0 (RXD) ←──────   MAX232/TTL       ←──────   TXD
GND     ────────→   GND              ────────→   GND
```

## Serial Modes

| Mode | Frame Size | Baud Rate | Use Case |
|------|------------|-----------|----------|
| 0 | 8-bit sync | Fixed (f/12) | Shift registers, IC expansion |
| 1 | 10-bit async | Variable | ⭐ Most common, PC communication |
| 2 | 11-bit async | Fixed (f/32 or f/64) | Multi-processor |
| 3 | 11-bit async | Variable | Multi-processor with variable baud |

## Examples

### Serial_Echo.c
**Difficulty:** ⭐⭐ Intermediate

**Description:**
Simple echo program - receives character and sends it back. Demonstrates:
- Serial port initialization (Mode 1)
- Baud rate configuration using Timer 1
- Polled transmit and receive
- Basic data handling

**Hardware Requirements:**
- 8051 @ 11.0592MHz (standard for serial comms)
- MAX232 or USB-serial adapter
- Terminal software (PuTTY, TeraTerm, Arduino Serial Monitor)

**Configuration:**
- Mode: 1 (8-bit UART)
- Baud Rate: 9600
- Timer 1: Mode 2 (auto-reload)
- TH1: 0FDH (for 9600 baud @ 11.0592MHz)

**Key Concepts:**
- SCON register configuration
- Timer 1 as baud rate generator
- TI/RI flag checking
- SBUF read/write operations

**Code Size:** ~60 lines

**Testing:**
1. Open terminal at 9600 baud, 8N1
2. Type characters
3. Should see echo back

---

## Upcoming Examples

#### Serial_Transmit.c
String transmission with formatting

#### Serial_Receive.c
Buffered receive with processing

#### Serial_Interrupt.c
Interrupt-driven serial I/O

#### Serial_Commands.c
Command parser and processor

#### Bluetooth_HC05.c
Wireless communication via Bluetooth

#### GPS_Interface.c
NMEA data parsing from GPS module

---

## Baud Rate Calculator

### Common Baud Rates @ 11.0592MHz

| Baud | SMOD | Timer 1 Mode | TH1 Value | Error |
|------|------|--------------|-----------|-------|
| 9600 | 0 | 2 | FDH (253) | 0.00% |
| 4800 | 0 | 2 | FAH (250) | 0.00% |
| 2400 | 0 | 2 | F4H (244) | 0.00% |
| 19200 | 1 | 2 | FDH (253) | 0.00% |

**Formula:**
```
Baud Rate = (2^SMOD / 32) × (Oscillator / (12 × (256 - TH1)))

For 9600 baud @ 11.0592MHz, SMOD=0:
TH1 = 256 - (11059200 / (32 × 12 × 9600)) = 253 = FDH
```

### Why 11.0592MHz?

Standard crystal frequencies that work perfectly:
- ✅ **11.0592MHz** - Standard for serial (exact baud rates)
- ✅ 12MHz - Works but has slight error
- ❌ 16MHz - Poor for standard baud rates

---

## Initialization Template

```c
void serial_init(unsigned char baud_rate)
{
    // Timer 1 for baud rate
    TMOD &= 0x0F;        // Clear Timer 1 mode bits
    TMOD |= 0x20;        // Timer 1, Mode 2 (8-bit auto-reload)

    TH1 = baud_rate;     // Set baud rate
    TR1 = 1;             // Start Timer 1

    // Serial control
    SCON = 0x50;         // Mode 1, Receive enabled
    //   SM0=0, SM1=1 (Mode 1)
    //   REN=1 (Receive enable)

    TI = 0;              // Clear transmit flag
    RI = 0;              // Clear receive flag
}

void transmit_byte(unsigned char data)
{
    SBUF = data;         // Write to buffer
    while(!TI);          // Wait for transmission
    TI = 0;              // Clear flag
}

unsigned char receive_byte(void)
{
    while(!RI);          // Wait for reception
    RI = 0;              // Clear flag
    return SBUF;         // Read data
}
```

---

## Serial Data Frame (Mode 1)

```
┌───┬───────────┬───────────┬───────────┬───┬───┐
│ 0 │ D0        │ D1        │ D2 ... D7 │ 1 │   │
└───┴───────────┴───────────┴───────────┴───┴───┘
│   │           │           │           │   │
Start  Data (LSB first)      Data     Stop  Next
Bit                            (MSB)   Bit   Frame

Total: 10 bits per character
Time @ 9600 baud: 10 / 9600 = 1.04ms per character
```

---

## Common Issues

### Garbage Characters
**Symptoms:** Random/wrong characters received

**Solutions:**
- Verify baud rate matches (check crystal frequency!)
- Check Timer 1 configuration (must be Mode 2)
- Confirm SMOD bit setting
- Ensure correct TH1 reload value

### Nothing Transmitted
**Symptoms:** No data on TX pin

**Solutions:**
- Check TX pin (P3.1) with scope/logic analyzer
- Verify Timer 1 is running (TR1 = 1)
- Confirm SCON settings (Mode 1)
- Check TI flag clearing

### Nothing Received
**Symptoms:** RI flag never set

**Solutions:**
- Enable receive (REN = 1 in SCON)
- Verify RX pin (P3.0) connection
- Check baud rate matching
- Confirm cable wiring (TX→RX, RX→TX)

### Framing Errors
**Symptoms:** Occasional wrong characters

**Solutions:**
- Improve signal quality (shorter cables)
- Add decoupling capacitors
- Check for noise interference
- Verify ground connection

---

## Advanced Features

### Interrupt-Driven Serial

```c
void serial_isr(void) interrupt 4
{
    if(RI)              // Receive interrupt
    {
        RI = 0;
        // Process received data
        buffer[index++] = SBUF;
    }

    if(TI)              // Transmit interrupt
    {
        TI = 0;
        // Next character ready?
        if(tx_index < tx_length)
            SBUF = tx_buffer[tx_index++];
    }
}
```

### Ring Buffer for Receive

```c
#define BUF_SIZE 32

unsigned char rx_buffer[BUF_SIZE];
unsigned char rx_head = 0;
unsigned char rx_tail = 0;

void serial_isr(void) interrupt 4
{
    if(RI)
    {
        RI = 0;
        rx_buffer[rx_head] = SBUF;
        rx_head = (rx_head + 1) % BUF_SIZE;
    }
}

unsigned char get_byte(void)
{
    while(rx_head == rx_tail);    // Wait for data
    unsigned char data = rx_buffer[rx_tail];
    rx_tail = (rx_tail + 1) % BUF_SIZE;
    return data;
}
```

---

## Communication Protocols

### Simple Command Protocol

```
Format: [CMD][DATA][CHECKSUM]\r\n

Example:
"L1\r\n" - Turn on LED 1
"V?\r\n" - Request version
"R0\r\n" - Read sensor 0
```

### NMEA GPS Data Format

```
$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47
```

---

## Hardware Considerations

### Voltage Levels
- 8051: TTL (0V = 0, 5V = 1)
- PC: RS-232 (±12V)
- Solution: MAX232 or USB-Serial converter

### Connection Diagram
```
8051          MAX232          DB9         PC
TXD (P3.1) →→ T1IN   T1OUT →→  TX  ────→  RXD
RXD (P3.0) ←← R1OUT  R1IN ←←  RX  ←────  TXD
GND      ──── GND    GND   ───  GND  ────  GND

    C1-C4: 1µF capacitors
    C5: 1µF capacitor
```

---

## Debugging Tips

1. **Loopback Test** - Connect TX to RX, should echo
2. **Scope TX Pin** - Verify data is being transmitted
3. **Known Good** - Test with Arduino/USB-Serial
4. **Check Baud** - Verify with frequency counter
5. **Terminal Settings** - Confirm 8N1, no flow control

---

## Prerequisites

- Basic I/O operations
- Timer fundamentals
- Interrupt basics (for interrupt-driven I/O)

**Recommended Reading:**
- [Serial Port Section](../02_Hardware_Architecture/README.md#serial-port-uart)
- [Timer Examples](../02_Timers/)
- [Interrupt Examples](../03_Interrupts/)

---

## Real-World Applications

- **Debug Output**: Printf-style debugging
- **PC Interface**: Control panels, data loggers
- **Wireless**: Bluetooth, WiFi modules
- **GPS**: NMEA data parsing
- **Sensor Networks**: Multi-drop communication
- **Modbus**: Industrial protocol

---

## Next Steps

After mastering basic serial:
- [Interrupt-Driven Serial](../03_Interrupts/)
- [Advanced Protocols](../05_Advanced/)
