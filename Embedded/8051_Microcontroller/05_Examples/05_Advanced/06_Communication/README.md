# Communication Protocols

## Overview

Learn to implement various communication protocols with the 8051 for interfacing with external modules, sensors, and other microcontrollers.

---

## Protocol Types

| Protocol | Type | Speed | Pins | Complexity | Applications |
|----------|------|-------|------|------------|--------------|
| **UART** | Serial | 300-115200 bps | 2 (TX, RX) | ⭐⭐ | PC, GPS, Bluetooth |
| **I2C** | Serial | 100kHz-400kHz | 2 (SDA, SCL) | ⭐⭐⭐ | EEPROM, RTC, sensors |
| **SPI** | Serial | Mbps range | 4 (MOSI, MISO, SCK, CS) | ⭐⭐⭐ | Flash, displays, SD card |
| **1-Wire** | Serial | ~15kbps | 1 | ⭐⭐⭐⭐ | DS18B20, EEPROM |
| **CAN** | Serial | 1Mbps | 2 | ⭐⭐⭐⭐⭐ | Automotive, industrial |

---

## Example 1: I2C Master

### Application: RTC (DS1307) Interface

**Difficulty:** ⭐⭐⭐⭐ Expert

**Description:**
Implement I2C master protocol for DS1307 Real-Time Clock.

### I2C Protocol

**Start Condition:**
```
SDA    ───┐       ┌────────────
           │       │
SCL    ────┴───────┴────────────
           ↓
        SDA goes LOW while SCL is HIGH
```

**Stop Condition:**
```
SDA         ┌──────────┐
            │          │
SCL    ─────┴──────────┴────
            ↓
        SDA goes HIGH while SCL is HIGH
```

**Data Transfer:**
```
     SCL  ┐   ┐   ┐   ┐   ┐
          └───┴───┴───┴───┴
          ↑   ↑   ↑   ↑   ↑
     SDA  D7  D6  D5  D4  D3...
          │   │   │   │
        ACK (pull SDA LOW)
```

### Hardware Connection

```
         8051               DS1307 RTC
       ┌──────┐             ┌──────┐
       │      │             │      │
       │  P1.0├───SDA───────┤ SDA  │
       │      │    ┌───┐    │      │
       │  P1.1├───SCL──────┤ SCL  │
       │      │    ┌───┘    │      │
       │  VCC ├─────────────┤ VCC  │
       │  GND ├─────────────┤ GND  │
       └──────┘             │ SQW  │ (Optional 1Hz output)
                           └──────┘
                           4.7kΩ pull-up on SDA and SCL
```

### Source Code

```c
// I2C Master Implementation
// DS1307 RTC Interface

#include <reg51.h>

sbit SDA = P1^0;
sbit SCL = P1^1;

// Microsecond delay
void delay_us(unsigned int us)
{
    while(us--) {
        for(unsigned char i = 0; i < 2; i++);
    }
}

// I2C Start condition
void i2c_start(void)
{
    SDA = 1;
    SCL = 1;
    delay_us(5);

    SDA = 0;  // SDA goes LOW while SCL is HIGH
    delay_us(5);

    SCL = 0;
}

// I2C Stop condition
void i2c_stop(void)
{
    SDA = 0;
    SCL = 1;
    delay_us(5);

    SDA = 1;  // SDA goes HIGH while SCL is HIGH
    delay_us(5);
}

// I2C Write bit
void i2c_write_bit(unsigned char bit)
{
    SDA = bit;
    delay_us(5);

    SCL = 1;
    delay_us(5);

    SCL = 0;
    delay_us(5);
}

// I2C Read bit
unsigned char i2c_read_bit(void)
{
    unsigned char bit = 0;

    SDA = 1;  // Release SDA (float high)
    delay_us(5);

    SCL = 1;
    delay_us(3);

    bit = SDA;  // Sample SDA

    delay_us(2);
    SCL = 0;
    delay_us(5);

    return bit;
}

// I2C Write byte
unsigned char i2c_write_byte(unsigned char data)
{
    unsigned char ack;
    unsigned char i;

    for(i = 0; i < 8; i++) {
        i2c_write_bit(data & 0x80);  // Send MSB first
        data <<= 1;
    }

    ack = i2c_read_bit();  // Read ACK

    return ack;  // 0 = ACK, 1 = NACK
}

// I2C Read byte
unsigned char i2c_read_byte(unsigned char ack)
{
    unsigned char data = 0;
    unsigned char i;

    for(i = 0; i < 8; i++) {
        data <<= 1;
        data |= i2c_read_bit();  // Read MSB first
    }

    i2c_write_bit(ack);  // Send ACK/NACK

    return data;
}

// DS1307 Write register
void ds1307_write(unsigned char reg, unsigned char data)
{
    i2c_start();
    i2c_write_byte(0xD0);  // DS1307 address + write
    i2c_write_byte(reg);    // Register address
    i2c_write_byte(data);   // Data
    i2c_stop();
}

// DS1307 Read register
unsigned char ds1307_read(unsigned char reg)
{
    unsigned char data;

    i2c_start();
    i2c_write_byte(0xD0);  // DS1307 address + write
    i2c_write_byte(reg);    // Register address
    i2c_stop();

    i2c_start();
    i2c_write_byte(0xD1);  // DS1307 address + read
    data = i2c_read_byte(1); // Read data, send NACK
    i2c_stop();

    return data;
}

// BCD to decimal conversion
unsigned char bcd_to_dec(unsigned char bcd)
{
    return ((bcd >> 4) * 10) + (bcd & 0x0F);
}

// Decimal to BCD conversion
unsigned char dec_to_bcd(unsigned char dec)
{
    return ((dec / 10) << 4) + (dec % 10);
}

// Get time from DS1307
void get_time(unsigned char *hours, unsigned char *minutes, unsigned char *seconds)
{
    *seconds = bcd_to_dec(ds1307_read(0x00));
    *minutes = bcd_to_dec(ds1307_read(0x01));
    *hours = bcd_to_dec(ds1307_read(0x02));
}

// Set time on DS1307
void set_time(unsigned char hours, unsigned char minutes, unsigned char seconds)
{
    ds1307_write(0x00, dec_to_bcd(seconds));
    ds1307_write(0x01, dec_to_bcd(minutes));
    ds1307_write(0x02, dec_to_bcd(hours));
}

void main(void)
{
    unsigned char h, m, s;

    // Set time: 12:30:45
    set_time(12, 30, 45);

    while(1) {
        get_time(&h, &m, &s);

        // Use h, m, s here
        // Display on LCD or send via serial

        delay_us(1000000);  // Wait 1 second
    }
}
```

---

## Example 2: SPI Master

### Application: SD Card Interface

**Difficulty:** ⭐⭐⭐⭐⭐ Expert

**Description:**
Implement SPI master for SD card data logging.

### SPI Protocol

**4-Wire SPI:**
- **MOSI**: Master Out Slave In (data from master to slave)
- **MISO**: Master In Slave Out (data from slave to master)
- **SCK**: Serial Clock (generated by master)
- **CS**: Chip Select (active low)

**Mode 0 (CPOL=0, CPHA=0):**
```
Clock:  ┌─┐ ┌─┐ ┌─┐ ┌─┐
        │ │ │ │ │ │ │ │
        └─┘ └─┘ └─┘ └─┘
MOSI:   ┌─┬─┬─┬─┬─┐
        D0 D1 D2 D3
MISO:   ┌─┬─┬─┬─┬─┐
        D0 D1 D2 D3 (slightly delayed)

Data: MSB first, sample on rising edge
```

### Hardware Connection

```
         8051                  SD Card
       ┌──────┐              ┌────────┐
       │      │              │        │
       │  P1.0├───MOSI───────┤ DI     │
       │  P1.1├───MISO───────┤ DO     │
       │  P1.2├───SCK────────┤ CLK    │
       │  P1.3├───CS─────────┤ CS     │
       │      │              │        │
       │  VCC ├──────────────┤ VCC    │
       │  GND ├──────────────┤ GND    │
       └──────┘              └────────┘
```

### Source Code (Basic SPI)

```c
// SPI Master Implementation
// SD Card SPI Mode (simplified)

#include <reg51.h>

sbit MOSI = P1^0;
sbit MISO = P1^1;
sbit SCK = P1^2;
sbit CS = P1^3;

// SPI initialization (Mode 0)
void spi_init(void)
{
    CS = 1;      // Deselect
    SCK = 0;     // Clock idle low
    MOSI = 1;    // Idle state
}

// SPI transmit/receive
unsigned char spi_transfer(unsigned char data)
{
    unsigned char received = 0;

    for(unsigned char i = 0; i < 8; i++) {
        // Send MSB first
        MOSI = (data & 0x80) ? 1 : 0;
        data <<= 1;

        // Generate clock pulse
        SCK = 1;

        // Read MISO on rising edge
        received <<= 1;
        if(MISO) {
            received |= 0x01;
        }

        SCK = 0;
    }

    return received;
}

// SD Card initialization (simplified)
unsigned char sd_init(void)
{
    unsigned char response;

    CS = 1;
    for(unsigned char i = 0; i < 10; i++) {
        spi_transfer(0xFF);  // Send dummy clocks
    }
    CS = 0;

    // Send CMD0 (reset)
    spi_transfer(0x40);
    spi_transfer(0x00);
    spi_transfer(0x00);
    spi_transfer(0x00);
    spi_transfer(0x00);
    spi_transfer(0x95);
    response = spi_transfer(0xFF);

    CS = 1;

    return (response == 0x01) ? 1 : 0;
}

// Write block to SD card
unsigned char sd_write_block(unsigned long address, unsigned char *buffer)
{
    // ... implementation ...
    return 1;
}

// Read block from SD card
unsigned char sd_read_block(unsigned long address, unsigned char *buffer)
{
    // ... implementation ...
    return 1;
}
```

---

## Communication Protocol Comparison

### Speed Comparison

| Protocol | Typical Speed | Data Rate |
|----------|---------------|-----------|
| UART | 9600-115200 bps | ~12 KB/s max |
| I2C | 100kHz (Standard) | ~12 KB/s |
| I2C | 400kHz (Fast) | ~48 KB/s |
| SPI | 1-10 MHz | ~1.2 MB/s |
| CAN | 125k-1 Mbps | ~125 KB/s |

### Complexity Comparison

| Protocol | Code Complexity | Hardware Complexity |
|----------|----------------|---------------------|
| UART | ⭐⭐ | ⭐ (2 pins, level shifter) |
| I2C | ⭐⭐⭐ | ⭐⭐ (2 pins + pull-ups) |
| SPI | ⭐⭐⭐ | ⭐⭐ (4 pins, CS per device) |
| 1-Wire | ⭐⭐⭐⭐ | ⭐ (1 pin + pull-up) |
| CAN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (transceiver needed) |

---

## Common Issues

### I2C

**Bus hangs (SDA stuck low):**
- Send additional clock pulses
- Reset all devices
- Check for short circuits

**No ACK received:**
- Verify device address
- Check pull-up resistors
- Ensure proper timing

**Wrong data:**
- Check for bus contention
- Verify bit ordering (MSB first)
- Ensure pull-up values correct

### SPI

**Garbage data:**
- Check mode compatibility (CPOL, CPHA)
- Verify clock frequency
- Ensure proper chip select timing

**No response:**
- Check CS pin toggling
- Verify MOSI/MISO connections
- Test with known-good device

**Wrong byte order:**
- Remember SPI is MSB first
- Some devices may use LSB first
- Check device datasheet

---

## Protocol Selection Guide

### Choose I2C when:
- Multiple slaves on same bus
- Low pin count is critical
- Speed requirements < 400kHz
- Devices are I2C native

### Choose SPI when:
- High speed required (>1MHz)
- Full-duplex communication needed
- Point-to-point communication
- Devices are SPI native

### Choose UART when:
- Point-to-point long distance
- PC communication needed
- Simple protocol sufficient
- Hardware already available

---

## Applications

### I2C Applications
- RTC (DS1307, PCF8563)
- EEPROM (AT24Cxx)
- Temperature sensors (DS1621, LM75)
- I/O expanders (PCF8574)
- DAC/ADC (PCF8591)

### SPI Applications
- SD/microSD cards
- Flash memory (AT45DB, W25Q)
- Displays (Nokia 5110, TFT)
- RF modules (nRF24L01)
- Ethernet controllers (ENC28J60)

### UART Applications
- GPS modules
- Bluetooth modules (HC-05)
- Wi-Fi modules (ESP8266)
- USB-UART bridges
- PC communication

---

## Component Selection

### I2C Devices

| Device | Type | Address | Voltage | Price |
|--------|------|---------|---------|-------|
| DS1307 | RTC | 0xD0 | 5V | Low |
| AT24C32 | EEPROM (32KB) | 0xAE | 5V | Low |
| PCF8574 | I/O expander | 0x40-0x47 | 5V | Low |
| BMP180 | Pressure sensor | 0xEE | 3.3V | Medium |

### SPI Devices

| Device | Type | Voltage | Price |
|--------|------|---------|-------|
| AT45DB161 | Flash (16Mbit) | 2.7-3.6V | Medium |
| nRF24L01 | 2.4GHz RF | 1.9-3.6V | Low |
| W25Q32 | Flash (32Mbit) | 2.7-3.6V | Low |

---

## Prerequisites

- ✅ [Basic I/O](../../01_Basic_IO/)
- ✅ [Timers](../../02_Timers/) - For baud rate generation
- ✅ [Serial Port](../../04_Serial_Port/) - UART basics
- ✅ Bit manipulation

**Recommended Reading:**
- [Serial Port](../../04_Serial_Port/)
- [Advanced Timer Usage](../../02_Timers/)
- [Sensors](../04_Sensors/) - Many use I2C/SPI

---

## Next Steps

After mastering communication protocols:
- [Sensors](../04_Sensors/) - I2C/SPI sensors
- [Display](../05_Display/) - SPI displays
- [Projects](../../../06_Projects/) - Data logger, weather station

---

**Difficulty:** ⭐⭐⭐⭐⭐ Expert
**Time to Master:** 20-30 hours
**Hardware:** Various modules, pull-up resistors, level shifters
