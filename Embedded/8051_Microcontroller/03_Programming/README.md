# Programming the 8051 Microcontroller

## Programming Languages

### 1. C Language (Recommended)
- **Advantages**: Easier to learn, portable, maintainable
- **Compilers**: Keil C51, SDCC (open-source)
- **Use Case**: Most applications

### 2. Assembly Language
- **Advantages**: Direct hardware control, smaller code size
- **Assemblers**: A51 (Keil), ASEM-51
- **Use Case**: Time-critical operations, bootloaders

## C Programming Basics

### Header Files
```c
#include <reg51.h>  // 8051 register definitions
```

### Data Types
- `bit`: Single bit (0 or 1)
- `unsigned char`: 8-bit (0-255)
- `signed char`: 8-bit (-128 to 127)
- `unsigned int`: 16-bit (0-65535)
- `signed int`: 16-bit (-32768 to 32767)

### Memory Keywords
- `code`: Program memory (ROM)
- `data`: Internal RAM (0x00-0x7F)
- `idata`: Indirect internal RAM
- `xdata`: External RAM
- `pdata`: Paged external RAM

### Example: Variable Declaration
```c
unsigned char data counter;      // Internal RAM
unsigned int xdata buffer[100];  // External RAM
code unsigned char table[] = {1,2,3,4,5};  // ROM
bit flag;                        // Bit variable
```
