# GPIO (General Purpose I/O)

## Port Operations

### Writing to Ports
```c
P1 = 0xFF;  // All pins HIGH
P1 = 0x00;  // All pins LOW
P1 = 0xAA;  // Alternate pattern
```

### Reading from Ports
```c
unsigned char value = P1;  // Read entire port
if (P1_0 == 0) {          // Read single pin
    // Pin is LOW
}
```

### Bit-Level Operations
```c
sbit LED = P1^0;   // Define bit at P1.0
sbit BUTTON = P3^2; // Define bit at P3.2

LED = 0;           // Set LOW
LED = 1;           // Set HIGH
if (BUTTON == 0) { // Check if pressed
    LED = 0;
}
```

## Port Characteristics

### Port 0
- Open-drain output
- Requires external pull-up resistors (10kΩ)
- Used for address/data bus with external memory

### Port 1
- Internal pull-ups
- Best for general I/O
- No alternate functions

### Port 2
- Internal pull-ups
- Used for high-order address with external memory

### Port 3
- Internal pull-ups
- Has alternate functions (serial, interrupts, timers)

## Example: LED and Button
```c
#include <reg51.h>

sbit LED = P1^0;
sbit BUTTON = P3^2;

void main() {
    while(1) {
        if (BUTTON == 0) {  // Button pressed (active low)
            LED = 0;        // Turn ON LED
        } else {
            LED = 1;        // Turn OFF LED
        }
    }
}
```
