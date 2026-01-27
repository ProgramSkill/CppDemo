# Display Systems

## Overview

Learn to interface various display types with the 8051 for visual output, user interfaces, and data visualization.

---

## Display Types

| Display | Type | Resolution | Interface | Complexity |
|---------|------|------------|-----------|------------|
| **LED** | Simple | 1 pixel | GPIO | ⭐ Beginner |
| **7-Segment** | Numeric | 1 digit | GPIO/MUX | ⭐⭐ Intermediate |
| **LCD 16x2** | Character | 16×2 chars | Parallel | ⭐⭐⭐ Advanced |
| **OLED 128x64** | Graphical | 128×64 | I2C/SPI | ⭐⭐⭐⭐ Expert |
| **TFT LCD** | Color | Various | Parallel/SPI | ⭐⭐⭐⭐⭐ Expert |

---

## Example 1: 7-Segment Display

### Application: Digital Counter

**Difficulty:** ⭐⭐ Intermediate

**Description:**
Display numbers on 7-segment LED display.

### Hardware Connection (Common Cathode)

```
         8051               7-Segment Display
       ┌──────┐              ┌─────────────┐
       │      │              │      a      │
       │  P0.0├──────────────┤┌───┐  f │ b │
       │  P0.1├──────────────┤│   │      │
       │  P0.2├──────────────┤└───┘  g │   │
       │  P0.3├──────────────┤      e ───┤ c │
       │  P0.4├──────────────┤      d      │
       │  P0.5├──────────────┤      ┌────┐│
       │  P0.6├──────────────┤      │ DP ││
       │      │              └─────┴─────┴┘
       │  GND ├─────────────────┐  │
       │      │                 └──┴─ Common Cathode
       └──────┘
```

### Segment Pattern (0-9)

```c
// Hex values for common cathode display
// gfedcba
unsigned char digit_table[] = {
    0x3F,  // 0: 0011 1111
    0x06,  // 1: 0000 0110
    0x5B,  // 2: 0101 1011
    0x4F,  // 3: 0100 1111
    0x66,  // 4: 0110 0110
    0x6D,  // 5: 0110 1101
    0x7D,  // 6: 0111 1101
    0x07,  // 7: 0000 0111
    0x7F,  // 8: 0111 1111
    0x6F   // 9: 0110 1111
};
```

### Source Code

```c
// 7-Segment Display Counter
// Counts 0-9 continuously

#include <reg51.h>

#define SEGMENT_PORT P0

// Digit patterns (common cathode)
unsigned char digit_table[] = {
    0x3F,  // 0
    0x06,  // 1
    0x5B,  // 2
    0x4F,  // 3
    0x66,  // 4
    0x6D,  // 5
    0x7D,  // 6
    0x07,  // 7
    0x7F,  // 8
    0x6F   // 9
};

// Simple delay
void delay_ms(unsigned int ms)
{
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

// Display digit 0-9
void display_digit(unsigned char digit)
{
    if(digit <= 9) {
        SEGMENT_PORT = digit_table[digit];
    }
}

void main(void)
{
    unsigned char counter = 0;

    while(1) {
        display_digit(counter);

        delay_ms(1000);  // Display for 1 second

        counter = (counter + 1) % 10;  // 0-9
    }
}
```

---

## Example 2: Multi-Digit Multiplexing

### Application: 4-Digit Counter

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Display 4-digit number using multiplexing.

### Hardware Connection

```
         8051            4× 7-Segment Displays
       ┌──────┐              (Common Cathode)
       │      │              ┌──────────────┐
       │  P0  ╞═╡╡╡╡╡╡╡╡╡╡╡╡╡╡╡→  Segments (a-g, DP)
       │      │              └──────────────┘
       │  P2.0├──────────────┤ Digit 1 Common │
       │  P2.1├──────────────┤ Digit 2 Common │
       │  P2.2├──────────────┤ Digit 3 Common │
       │  P2.3├──────────────┤ Digit 4 Common │
       └──────┘
```

### Multiplexing Principle

**Rapidly switch between digits:**
```
Time:
0-5ms:   Display Digit 1 (others off)
5-10ms:  Display Digit 2 (others off)
10-15ms: Display Digit 3 (others off)
15-20ms: Display Digit 4 (others off)
Repeat...

Persistence of vision makes all appear ON simultaneously.
```

### Source Code

```c
// 4-Digit 7-Segment Display with Multiplexing
// Displays counter 0000-9999

#include <reg51.h>

#define SEGMENT_PORT P0

// Digit selection lines
sbit DIG1 = P2^0;
sbit DIG2 = P2^1;
sbit DIG3 = P2^2;
sbit DIG4 = P2^3;

// Digit patterns
unsigned char digit_table[] = {
    0x3F, 0x06, 0x5B, 0x4F, 0x66,
    0x6D, 0x7D, 0x07, 0x7F, 0x6F
};

// Number to display
unsigned int display_number = 0;

// Display one digit
void show_digit(unsigned char position, unsigned char value)
{
    // Turn off all digits
    DIG1 = DIG2 = DIG3 = DIG4 = 1;

    // Set segment pattern
    SEGMENT_PORT = digit_table[value];

    // Turn on selected digit (active low)
    switch(position) {
        case 0: DIG1 = 0; break;
        case 1: DIG2 = 0; break;
        case 2: DIG3 = 0; break;
        case 3: DIG4 = 0; break;
    }

    // Short delay for brightness control
    for(unsigned int i = 0; i < 100; i++);
}

// Multiplex all 4 digits
void multiplex_display(void)
{
    unsigned char digits[4];
    unsigned int temp = display_number;

    // Extract digits
    digits[3] = temp % 10;
    temp /= 10;
    digits[2] = temp % 10;
    temp /= 10;
    digits[1] = temp % 10;
    temp /= 10;
    digits[0] = temp % 10;

    // Display each digit rapidly
    show_digit(0, digits[0]);
    show_digit(1, digits[1]);
    show_digit(2, digits[2]);
    show_digit(3, digits[3]);
}

// Simple delay
void delay_sec(unsigned char sec)
{
    unsigned int i, j;
    for(i = 0; i < sec; i++)
        for(j = 0; j < 1000; j++)
            multiplex_display();  // Keep refreshing
}

void main(void)
{
    while(1) {
        for(display_number = 0; display_number < 9999; display_number++) {
            delay_sec(1);  // Display for 1 second
        }
    }
}
```

---

## Example 3: LCD 16x2 (HD44780)

### Application: Text Display

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Interface 16×2 character LCD using HD44780 controller.

### Hardware Connection (4-bit mode)

```
         8051                  LCD 16x2
       ┌──────┐              ┌────────────┐
       │      │              │            │
       │  P2.0├──────────────┤ RS         │
       │  P2.1├──────────────┤ RW         │
       │  P2.2├──────────────┤ E          │
       │  P2.4├──────────────┤ D4         │
       │  P2.5├──────────────┤ D5         │
       │  P2.6├──────────────┤ D6         │
       │  P2.7├──────────────┤ D7         │
       │      │              │            │
       │  VCC ├──────────────┤ VDD        │
       │  GND ├──────────────┤ VSS, V0    │
       │      │           ┌───┤ VO         │
       │      │           │   └────────────┘
       │      │           │    10kΩ Pot
       │      │           │     (Contrast)
       └──────┘           │
                          GND
```

### LCD Commands

| Command | Hex | Description |
|---------|-----|-------------|
| Clear display | 01H | Clear all characters |
| Return home | 02H | Cursor to position 0 |
| Entry mode set | 06H | Increment, no shift |
| Display on/off | 0EH | Display on, cursor off |
| Function set | 28H | 4-bit, 2 lines, 5×7 |
| Set DDRAM address | 80H | Position cursor |

### Source Code

```c
// LCD 16x2 (HD44780)
// 4-bit interface mode

#include <reg51.h>

// LCD control pins
sbit RS = P2^0;
sbit RW = P2^1;
sbit EN = P2^2;

// LCD data pins (P2.4 - P2.7)
#define LCD_DATA P2

// Microsecond delay
void delay_us(unsigned int us)
{
    while(us--) {
        for(unsigned char i = 0; i < 2; i++);
    }
}

// Millisecond delay
void delay_ms(unsigned int ms)
{
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

// Send 4 bits to LCD
void lcd_send_nibble(unsigned char nibble)
{
    LCD_DATA = (LCD_DATA & 0x0F) | (nibble & 0xF0);
    EN = 1;
    delay_us(1);
    EN = 0;
    delay_us(100);
}

// Send command to LCD
void lcd_cmd(unsigned char cmd)
{
    RS = 0;  // Command mode
    RW = 0;  // Write

    lcd_send_nibble(cmd & 0xF0);        // High nibble
    lcd_send_nibble((cmd << 4) & 0xF0); // Low nibble

    delay_ms(2);  // Wait for command to execute
}

// Send data to LCD
void lcd_data(unsigned char data)
{
    RS = 1;  // Data mode
    RW = 0;  // Write

    lcd_send_nibble(data & 0xF0);
    lcd_send_nibble((data << 4) & 0xF0);

    delay_us(100);
}

// Initialize LCD
void lcd_init(void)
{
    delay_ms(20);  // Wait for LCD power up

    lcd_send_nibble(0x30);  // Function set (8-bit)
    delay_ms(5);
    lcd_send_nibble(0x30);
    delay_us(100);
    lcd_send_nibble(0x30);
    delay_us(100);

    lcd_send_nibble(0x20);  // Set 4-bit mode
    delay_us(100);

    lcd_cmd(0x28);  // 4-bit, 2 lines, 5×7 dots
    lcd_cmd(0x0C);  // Display on, cursor off
    lcd_cmd(0x06);  // Auto-increment cursor
    lcd_cmd(0x01);  // Clear display
    delay_ms(2);
}

// Clear display
void lcd_clear(void)
{
    lcd_cmd(0x01);
    delay_ms(2);
}

// Set cursor position
void lcd_goto(unsigned char row, unsigned char col)
{
    unsigned char pos;

    if(row == 0) {
        pos = 0x80 + col;  // Line 1
    } else {
        pos = 0xC0 + col;  // Line 2
    }

    lcd_cmd(pos);
}

// Display string
void lcd_print(const char *str)
{
    while(*str) {
        lcd_data(*str++);
    }
}

// Display number
void lcd_print_number(unsigned int num)
{
    char buffer[16];
    unsigned char i = 0;

    // Convert number to string
    if(num == 0) {
        lcd_data('0');
        return;
    }

    while(num > 0) {
        buffer[i++] = (num % 10) + '0';
        num /= 10;
    }

    // Display in reverse order
    while(i > 0) {
        lcd_data(buffer[--i]);
    }
}

void main(void)
{
    lcd_init();

    lcd_clear();
    lcd_goto(0, 0);
    lcd_print("8051 LCD Demo");

    lcd_goto(1, 0);
    lcd_print("Count: ");

    unsigned int counter = 0;
    while(1) {
        lcd_goto(1, 7);
        lcd_print_number(counter);

        counter++;
        delay_ms(1000);
    }
}
```

---

## Example 4: Custom Characters

### Application: Create Special Symbols

**Description:**
Create custom characters for LCD (e.g., degree symbol, arrows).

### Source Code

```c
// Define custom character (5×8 pixels)
// Each byte is one row (5 lower bits used)

// Degree symbol (°)
unsigned char degree_symbol[8] = {
    0b00001100,  //   **
    0b00010010,  //  *  *
    0b00010010,  //  *  *
    0b00001100,  //   **
    0b00000000,  //
    0b00000000,  //
    0b00000000,  //
    0b00000000   //
};

// Save custom character to CGRAM
void lcd_create_char(unsigned char location, unsigned char *pattern)
{
    lcd_cmd(0x40 + (location * 8));  // Set CGRAM address

    for(unsigned char i = 0; i < 8; i++) {
        lcd_data(pattern[i]);
    }
}

// Display custom character
void main(void)
{
    lcd_init();

    // Create degree symbol at location 0
    lcd_create_char(0, degree_symbol);

    lcd_goto(0, 0);
    lcd_print("Temp: 25");

    // Display custom character
    lcd_data(0);  // Character 0 from CGRAM
    lcd_print("C");
}
```

---

## Display Selection Guide

### When to Use What

| Application | Recommended Display | Reason |
|-------------|---------------------|--------|
| Simple status | LED | Lowest cost, simplest |
| Numbers (0-9) | 7-segment | Easy to read, low cost |
| Text messages | LCD 16x2 | Standard, well-documented |
| Graphics | OLED/TFT | High resolution, colorful |
| Battery powered | OLED | Low power consumption |
| Bright sunlight | Reflective LCD | Good visibility |
| Dark environment | Backlit LED/OLED | Bright display |

---

## Troubleshooting

### 7-Segment Display
**Wrong segments lit:**
- Check pattern table
- Verify wiring (a-g connections)
- Test with known good display

**Dim display:**
- Increase current (lower resistor)
- Check supply voltage
- Verify common anode/cathode type

**Digits flicker:**
- Increase multiplex frequency
- Reduce delay between digits
- Check for timing conflicts

### LCD
**Blank screen:**
- Check contrast voltage (VO)
- Verify VCC and GND connections
- Adjust potentiometer
- Test with known good LCD

**Garbage characters:**
- Check timing delays
- Verify 4-bit vs 8-bit mode
- Check data line connections
- Ensure proper initialization

**No display after reset:**
- Add proper power-up delay
- Reinitialize LCD
- Check initialization sequence

---

## Advanced Topics

### 1. Scrolling Text

```c
void lcd_scroll_left(const char *str, unsigned char delay_time)
{
    unsigned char len = 0;
    const char *temp = str;

    while(*temp++) len++;

    for(unsigned char i = 0; i < len; i++) {
        lcd_goto(0, 0);
        lcd_print(&str[i]);
        delay_ms(delay_time);
        lcd_clear();
    }
}
```

### 2. Blinking Cursor

```c
void lcd_blink_cursor(unsigned char times)
{
    for(unsigned char i = 0; i < times; i++) {
        lcd_cmd(0x0F);  // Cursor on, blink on
        delay_ms(500);
        lcd_cmd(0x0C);  // Cursor off
        delay_ms(500);
    }
}
```

### 3. Animation

```c
// Loading bar animation
void lcd_loading_bar(unsigned char position)
{
    lcd_goto(1, 0);
    for(unsigned char i = 0; i < position; i++) {
        lcd_data(0xFF);  // Full block
    }
}
```

---

## Applications

### User Interface
- Menu systems
- Status displays
- Error messages
- Input prompts

### Data Display
- Sensor readings
- Counter values
- Time/date
- Setpoints

### Visual Feedback
- Progress bars
- Level indicators
- Status icons
- Animations

---

## Component Selection

### 7-Segment Displays

| Type | Color | Common | Price |
|------|-------|--------|-------|
| Standard | Red | Cathode | Low |
| Large | Red/Green | Anode | Medium |
| Mini | Red | Cathode | Low |

### Character LCDs

| Size | Interface | Backlight | Price |
|------|-----------|-----------|-------|
| 16×2 | Parallel | None/Yellow | Low |
| 20×4 | Parallel | Blue | Medium |
| 16×2 | I2C | White | Medium |

### Graphic Displays

| Type | Resolution | Interface | Color | Price |
|------|------------|-----------|-------|-------|
| OLED 128×64 | 128×64 | I2C/SPI | Mono | Medium |
| Nokia 5110 | 84×48 | SPI | Mono | Low |
| TFT 1.8″ | 128×160 | SPI | 65K | Medium |
| TFT 2.4″ | 240×320 | Parallel | 262K | High |

---

## Prerequisites

- ✅ [Basic I/O](../../01_Basic_IO/)
- ✅ [Timers](../../02_Timers/) - For multiplexing

**Recommended Reading:**
- [Timers for Multiplexing](../../02_Timers/)
- [Sensors](../04_Sensors/) - Display sensor data

---

## Next Steps

After mastering displays:
- [Sensors](../04_Sensors/) - Display sensor readings
- [Communication](../06_Communication/) - Remote display
- [Projects](../../../06_Projects/) - Complete UI systems

---

**Difficulty:** ⭐⭐⭐⭐ Expert
**Time to Master:** 15-25 hours
**Hardware:** Displays, resistors, potentiometer
