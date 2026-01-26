# C Programming Examples

## Port Operations

### Writing to Ports
```c
#include <reg51.h>

void main() {
    P1 = 0xFF;  // Set all pins HIGH
    P1 = 0x00;  // Set all pins LOW
    P1 = 0xAA;  // Set alternate pins (10101010)
}
```

### Reading from Ports
```c
unsigned char value;
value = P1;  // Read entire port
if (P1_0 == 0) {  // Read individual pin
    // Pin P1.0 is LOW
}
```

### Bit Operations
```c
sbit LED = P1^0;  // Define bit variable for P1.0

void main() {
    LED = 0;  // Turn ON LED (active low)
    LED = 1;  // Turn OFF LED
}
```

## Functions

### Function Declaration
```c
void delay(unsigned int ms);
unsigned char add(unsigned char a, unsigned char b);

void main() {
    unsigned char result;
    result = add(5, 10);
    delay(1000);
}

void delay(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

unsigned char add(unsigned char a, unsigned char b) {
    return a + b;
}
```

## Loops and Conditionals

### For Loop
```c
unsigned char i;
for(i = 0; i < 10; i++) {
    P1 = i;
}
```

### While Loop
```c
while(1) {  // Infinite loop
    // Code here
}
```

### If-Else
```c
if (P1_0 == 0) {
    P2 = 0x00;
} else {
    P2 = 0xFF;
}
```
