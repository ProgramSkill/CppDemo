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

**Complete Code:**

```c
/**
 * Serial_Echo.c
 * 8051 Serial Port Echo Program
 *
 * Description: Receives characters from serial port and echoes them back
 * Hardware: 8051 @ 11.0592MHz, MAX232 level shifter
 * Baud Rate: 9600 (8N1)
 */

#include <reg51.h>

/* Baud rate reload value for 9600 baud @ 11.0592MHz */
#define BAUD_9600 0xFD

/**
 * Initialize serial port for 9600 baud, 8N1
 * Mode 1: 8-bit UART with variable baud rate
 */
void serial_init(void)
{
    /* Configure Timer 1 as baud rate generator */
    TMOD &= 0x0F;        /* Clear Timer 1 mode bits (keep Timer 0) */
    TMOD |= 0x20;        /* Timer 1, Mode 2 (8-bit auto-reload) */

    TH1 = BAUD_9600;     /* Set baud rate reload value */
    TL1 = BAUD_9600;     /* Initial load (auto-reloads from TH1) */
    TR1 = 1;             /* Start Timer 1 */

    /* Configure Serial Control Register */
    SCON = 0x50;         /* Mode 1 (8-bit UART), Receive enabled */
    /*   SM0 = 0, SM1 = 1: Mode 1 selected */
    /*   SM2 = 0: Disable multiprocessor mode */
    /*   REN = 1: Enable receiver */
    /*   TB8, RB8: Not used in Mode 1 */
    /*   TI, RI: Cleared to 0 */

    /* Clear interrupt flags (for polled operation) */
    TI = 0;              /* Clear Transmit Interrupt flag */
    RI = 0;              /* Clear Receive Interrupt flag */
}

/**
 * Transmit a single byte via serial port
 * Polled mode: waits until transmission completes
 */
void serial_send(unsigned char data)
{
    SBUF = data;         /* Write data to serial buffer */
    while (!TI);         /* Wait for transmission to complete */
    TI = 0;              /* Clear transmit interrupt flag */
}

/**
 * Receive a single byte from serial port
 * Polled mode: waits until data is received
 */
unsigned char serial_receive(void)
{
    while (!RI);         /* Wait for reception to complete */
    RI = 0;              /* Clear receive interrupt flag */
    return SBUF;         /* Return received data */
}

/**
 * Transmit a null-terminated string
 */
void serial_send_string(unsigned char *str)
{
    while (*str != '\0')
    {
        serial_send(*str);
        str++;
    }
}

/**
 * Main program - echo received characters
 */
void main(void)
{
    unsigned char received_char;

    /* Initialize serial port */
    serial_init();

    /* Send welcome message */
    serial_send_string("8051 Serial Echo Program\r\n");
    serial_send_string("Type characters and see them echoed back\r\n");
    serial_send_string("Press Ctrl+C to stop\r\n\r\n");

    /* Main loop - echo characters */
    while (1)
    {
        /* Wait for character */
        received_char = serial_receive();

        /* Echo it back */
        serial_send(received_char);

        /* Optional: echo with prompt for visual feedback */
        if (received_char == '\r')
        {
            /* If Enter key pressed, add line feed */
            serial_send('\n');
        }
    }
}
```

**Code Breakdown:**

1. **Timer 1 Configuration** (Lines 23-26)
   - Timer 1 in Mode 2 (8-bit auto-reload)
   - TH1 holds reload value (0xFD for 9600 baud)
   - Timer generates baud rate clock

2. **SCON Register** (Lines 29-36)
   - Sets UART to Mode 1
   - Enables receiver (REN = 1)
   - Clears interrupt flags

3. **Transmit Function** (Lines 46-51)
   - Writes data to SBUF
   - Waits for TI flag (transmission complete)
   - Clears TI flag for next transmission

4. **Receive Function** (Lines 57-62)
   - Waits for RI flag (data received)
   - Clears RI flag
   - Returns data from SBUF

5. **String Function** (Lines 67-74)
   - Helper to send complete strings
   - Iterates until null terminator

6. **Main Loop** (Lines 79-104)
   - Sends startup message
   - Continuously echoes received characters
   - Handles carriage return for proper line endings

**Testing:**
1. Connect 8051 to PC via MAX232/USB-serial
2. Open terminal at 9600 baud, 8N1
3. Reset 8051 - should see welcome message
4. Type characters - should see echo back
5. Press Enter - should get new line

**Expected Output:**
```
8051 Serial Echo Program
Type characters and see them echoed back
Press Ctrl+C to stop

Hello World!
Hello World!
```

---

## More Examples

### Serial_Transmit.c
**Difficulty:** ⭐⭐ Intermediate

**Description:**
Demonstrates formatted string transmission including numbers in different formats (decimal, hexadecimal, binary). This example shows how to create a complete output library for serial communication.

**Features:**
- String transmission
- Decimal byte/word formatting (0-65535)
- Signed decimal (-128 to 127)
- Hexadecimal output (byte and word)
- Binary output (8-bit)
- Mixed formatting examples
- Data table generation

**Key Functions:**
- `serial_send_decimal_byte()` - Format 0-255
- `serial_send_decimal_word()` - Format 0-65535
- `serial_send_hex()` - 2-digit hex
- `serial_send_binary()` - 8-bit binary

**Use For:**
- Debugging output
- Data logging
- Status displays
- User interfaces

---

### Serial_Receive.c
**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Buffered serial data reception with line-by-line processing. Implements a circular receive buffer with interrupt-driven data collection and command parsing.

**Features:**
- Circular receive buffer (64 bytes)
- Interrupt-driven reception
- Line buffering and processing
- Command parsing system
- Built-in commands (HELP, STATUS, CLEAR)
- Echo with backspace support
- Overflow detection

**Key Concepts:**
- Ring buffer implementation
- Interrupt service routines
- Line termination handling
- Buffer overflow protection

**Built-in Commands:**
- `HELP` - Show command list
- `STATUS` - Display buffer statistics
- `CLEAR` - Clear receive buffers
- `HELLO` - Test greeting

**Use For:**
- Command-line interfaces
- Data logging systems
- Protocol implementations

---

### Serial_Interrupt.c
**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Full-duplex serial communication using interrupts for both transmission and reception. Provides non-blocking I/O with dual circular buffers.

**Features:**
- Full-duplex operation (TX and RX)
- Non-blocking I/O
- Dual circular buffers (64 bytes each)
- Interrupt-driven transmit and receive
- Buffer space monitoring
- Overflow detection
- Status reporting

**Key Functions:**
- `serial_write()` - Non-blocking write
- `serial_write_string()` - Write string to buffer
- `serial_read()` - Read from buffer
- `serial_available()` - Check for data
- `serial_flush()` - Wait for TX complete
- `serial_tx_count()` - Get TX buffer usage
- `serial_rx_count()` - Get RX buffer usage

**Interrupt Handler:**
```c
void serial_isr(void) interrupt 4
{
    if (RI) { /* Receive */ }
    if (TI) { /* Transmit */ }
}
```

**Built-in Commands:**
- `STATUS` - Show buffer status
- `ECHO` - Test transmission
- `CLEAR` - Clear buffers
- `FLUSH` - Wait for TX complete

**Use For:**
- High-speed communication
- Real-time systems
- Multi-tasking applications
- Background data transfer

---

### Serial_Commands.c
**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Complete command-line interface with argument parsing, LED control, port I/O, and multiple command types. Demonstrates building a user interface over serial.

**Features:**
- Command parser with arguments
- Case-insensitive command matching
- LED control (individual and patterns)
- Port I/O operations
- Number conversion utilities
- Help system
- Error handling

**Available Commands:**
- `HELP` - Display command list
- `LED <num> <state>` - Control individual LED
- `LEDS <value>` - Set all LEDs (0-255)
- `PATTERN <type>` - LED patterns (CHASE/BLINK/ALL)
- `STATUS` - Show system status
- `READ <port>` - Read port (0-3)
- `WRITE <port> <val>` - Write port (0-3, 0-255)
- `CLEAR` - Clear screen
- `ECHO <text>` - Echo text back
- `HEX <value>` - Convert to hex/bin

**Key Functions:**
- `parse_command()` - Split command line
- `strcmp_i()` - Case-insensitive compare
- `ascii_to_byte()` - String to integer

**Examples:**
```
LED 0 ON          # Turn on LED 0
LEDS 0xAA         # Set pattern 10101010
PATTERN CHASE     # Run chase pattern
STATUS            # Show port states
HEX 255           # Display 0xFF 11111111
```

**Use For:**
- User interfaces
- Configuration tools
- Debugging consoles
- Control systems

---

### Bluetooth_HC05.c
**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Wireless communication using HC-05 Bluetooth module. Includes AT command mode for configuration and data mode for communication.

**Hardware Connections:**
```
8051 TXD (P3.1) -> HC-05 RXD
8051 RXD (P3.0) <- HC-05 TXD
8051 P3.2      -> HC-05 KEY  (AT mode)
8051 P3.7      -> HC-05 EN   (Enable)
```

**Features:**
- AT command mode entry/exit
- Module configuration
- Name/address/version queries
- Data mode communication
- Connection status monitoring
- LED feedback indicator

**Available Commands:**
- `AT` - Enter AT command mode
- `DATA` - Exit to data mode
- `NAME` - Get module name
- `ADDR` - Get module address
- `VER` - Get firmware version
- `TEST` - Send test message
- `STATUS` - Show connection status

**Key Functions:**
- `hc05_init()` - Initialize module
- `hc05_enter_at_mode()` - Enter configuration mode
- `hc05_send_at_command()` - Send AT command
- `hc05_get_name()` - Query module name
- `hc05_get_address()` - Query MAC address

**Common AT Commands:**
```
AT+NAME<name>     - Set module name
AT+PSWD<password> - Set PIN (default 1234)
AT+UART<baud>,0,0 - Set baud rate
AT+ROLE=<role>    - 0=slave, 1=master
```

**Use For:**
- Wireless control
- Remote monitoring
- Smartphone interfaces
- Wireless sensor networks

---

### GPS_Interface.c
**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
GPS module interface with NMEA sentence parsing. Supports multiple NMEA message types (GPGGA, GPRMC, GPGSA, GPGSV) and extracts position, time, date, and satellite data.

**Hardware Connections:**
```
8051 RXD (P3.0) <- GPS TXD
GPS VCC         <- 3.3V or 5V
GPS GND         <- GND
```

**Supported NMEA Sentences:**
- **GPGGA** - Fix data (time, position, fix type, satellites, altitude)
- **GPRMC** - Recommended minimum (time, date, position, speed, course)
- **GPGSA** - Satellite data (fix type, PDOP, satellites in use)
- **GPGSV** - Satellites in view (count, elevation, azimuth, SNR)

**Features:**
- NMEA sentence parsing
- Multiple sentence type support
- Position validation
- Time and date extraction
- Speed and course tracking
- Satellite count monitoring
- Raw sentence display

**Data Extracted:**
- UTC time
- Date (DDMMYY)
- Latitude (DDMM.MMMM)
- Longitude (DDDMM.MMMM)
- North/South indicator
- East/West indicator
- Fix quality (0=none, 1=GPS, 2=DGPS)
- Number of satellites
- Altitude (meters)
- Speed over ground (knots)
- Course over ground (degrees)

**Available Commands:**
- `DATA` - Show parsed GPS data
- `RAW` - Show raw NMEA sentence
- `STATUS` - Show fix status and satellite count
- `TIME` - Show UTC time
- `POS` - Show position (lat/lon/alt)

**Key Functions:**
- `parse_gpgga()` - Parse fix data
- `parse_gprmc()` - Parse minimum data
- `parse_nmea_field()` - Extract comma-separated field
- `process_nmea_sentence()` - Route to correct parser

**Example NMEA Sentences:**
```
$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47
$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A
```

**Use For:**
- Navigation systems
- Vehicle tracking
- Time synchronization
- Location-based services
- Data logging

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
