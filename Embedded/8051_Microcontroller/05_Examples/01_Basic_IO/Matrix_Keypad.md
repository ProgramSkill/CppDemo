# Example 6: Matrix Keypad (4×4 and 4×3)

## Overview

Matrix keypads are an efficient way to interface multiple buttons using fewer I/O pins. Instead of using 16 pins for 16 buttons, a 4×4 matrix keypad uses only 8 pins (4 rows + 4 columns). This is achieved through a technique called **scanning**, where rows and columns are sequentially activated and read.

---

## 📝 Complete Source Code

### Version 1: 4×4 Matrix Keypad (Basic Scanning)

**File:** `Matrix_Keypad_4x4.c`

```c
// 4×4 Matrix Keypad Example
// Reads and displays pressed keys
// Target: 8051 @ 12MHz
// Hardware: 4×4 keypad on P1 and P2

#include <reg51.h>

// Keypad pin definitions
// Rows (outputs)
sbit ROW1 = P1^0;
sbit ROW2 = P1^1;
sbit ROW3 = P1^2;
sbit ROW4 = P1^3;

// Columns (inputs)
sbit COL1 = P1^4;
sbit COL2 = P1^5;
sbit COL3 = P1^6;
sbit COL4 = P1^7;

// Key values for 4×4 keypad
unsigned char code key_map[4][4] = {
    {'1', '2', '3', 'A'},
    {'4', '5', '6', 'B'},
    {'7', '8', '9', 'C'},
    {'*', '0', '#', 'D'}
};

/**
 * @brief  Initialize keypad pins
 * @retval None
 */
void keypad_init() {
    // Set all rows as outputs (initially HIGH)
    ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 1;

    // Set all columns as inputs with pull-up
    // 8051 has internal weak pull-ups, but external recommended
    COL1 = 1; COL2 = 1; COL3 = 1; COL4 = 1;
}

/**
 * @brief  Scan keypad for key press
 * @retval Pressed key value, or 0 if no key pressed
 */
unsigned char keypad_scan() {
    unsigned char row, col;

    // Scan each row
    for(row = 0; row < 4; row++) {
        // Set current row LOW, others HIGH
        switch(row) {
            case 0: ROW1 = 0; ROW2 = 1; ROW3 = 1; ROW4 = 1; break;
            case 1: ROW1 = 1; ROW2 = 0; ROW3 = 1; ROW4 = 1; break;
            case 2: ROW1 = 1; ROW2 = 1; ROW3 = 0; ROW4 = 1; break;
            case 3: ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 0; break;
        }

        // Small delay for signal stabilization
        // In real application, may need 10-50µs

        // Check each column
        if(COL1 == 0) {
            // Wait for key release
            while(COL1 == 0);
            return key_map[row][0];
        }
        if(COL2 == 0) {
            while(COL2 == 0);
            return key_map[row][1];
        }
        if(COL3 == 0) {
            while(COL3 == 0);
            return key_map[row][2];
        }
        if(COL4 == 0) {
            while(COL4 == 0);
            return key_map[row][3];
        }
    }

    // No key pressed
    return 0;
}

/**
 * @brief  Simple delay function
 */
void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

/**
 * @brief  Display key on LEDs (for testing)
 * @param  key: Key character to display
 */
void display_key(unsigned char key) {
    // Simple: Show binary on P2
    // In real application, send to LCD or serial
    P2 = key;
}

void main() {
    unsigned char key;

    keypad_init();

    while(1) {
        key = keypad_scan();

        if(key != 0) {
            // Key pressed
            display_key(key);

            // Simple beep or LED indicator could go here
            delay_ms(200);  // Small delay
        }
    }
}
```

---

### Version 2: 4×4 Matrix Keypad with Debounce

**File:** `Matrix_Keypad_Debounce.c`

```c
// 4×4 Matrix Keypad with Enhanced Debouncing
// Includes proper debounce and repeat prevention
// Target: 8051 @ 12MHz

#include <reg51.h>

// Row pins (outputs)
sbit ROW1 = P1^0;
sbit ROW2 = P1^1;
sbit ROW3 = P1^2;
sbit ROW4 = P1^3;

// Column pins (inputs)
sbit COL1 = P1^4;
sbit COL2 = P1^5;
sbit COL3 = P1^6;
sbit COL4 = P1^7;

// LED indicators
sbit LED_PRESS = P2^0;  // Lights when key pressed

unsigned char code key_map[4][4] = {
    {'1', '2', '3', 'A'},
    {'4', '5', '6', 'B'},
    {'7', '8', '9', 'C'},
    {'*', '0', '#', 'D'}
};

void keypad_init() {
    ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 1;
    COL1 = 1; COL2 = 1; COL3 = 1; COL4 = 1;
    LED_PRESS = 1;  // LED OFF (active low)
}

/**
 * @brief  Debounce delay
 * @param  ms: Milliseconds to delay
 */
void debounce_delay(unsigned char ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

/**
 * @brief  Scan keypad with debouncing
 * @retval Pressed key or 0 if none
 */
unsigned char keypad_scan() {
    unsigned char row, col;

    for(row = 0; row < 4; row++) {
        // Activate current row
        switch(row) {
            case 0: ROW1 = 0; ROW2 = 1; ROW3 = 1; ROW4 = 1; break;
            case 1: ROW1 = 1; ROW2 = 0; ROW3 = 1; ROW4 = 1; break;
            case 2: ROW1 = 1; ROW2 = 1; ROW3 = 0; ROW4 = 1; break;
            case 3: ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 0; break;
        }

        // Debounce delay for row change
        debounce_delay(1);

        // Check columns
        for(col = 0; col < 4; col++) {
            unsigned char col_pressed = 0;

            switch(col) {
                case 0: col_pressed = (COL1 == 0); break;
                case 1: col_pressed = (COL2 == 0); break;
                case 2: col_pressed = (COL3 == 0); break;
                case 3: col_pressed = (COL4 == 0); break;
            }

            if(col_pressed) {
                // Key pressed - debounce
                debounce_delay(20);  // 20ms debounce

                // Verify still pressed
                unsigned char still_pressed = 0;
                switch(col) {
                    case 0: still_pressed = (COL1 == 0); break;
                    case 1: still_pressed = (COL2 == 0); break;
                    case 2: still_pressed = (COL3 == 0); break;
                    case 3: still_pressed = (COL4 == 0); break;
                }

                if(still_pressed) {
                    unsigned char key = key_map[row][col];

                    // Wait for key release
                    while(1) {
                        unsigned char released = 1;
                        switch(col) {
                            case 0: if(COL1 == 0) released = 0; break;
                            case 1: if(COL2 == 0) released = 0; break;
                            case 2: if(COL3 == 0) released = 0; break;
                            case 3: if(COL4 == 0) released = 0; break;
                        }
                        if(released) break;
                    }

                    debounce_delay(20);  // Release debounce
                    return key;
                }
            }
        }
    }

    return 0;
}

void main() {
    unsigned char key;

    keypad_init();

    while(1) {
        key = keypad_scan();

        if(key != 0) {
            LED_PRESS = 0;  // LED ON
            debounce_delay(100);
            LED_PRESS = 1;  // LED OFF

            // Use key value here
            // Could send to LCD, serial, etc.
        }
    }
}
```

---

### Version 3: 4×3 Matrix Keypad

**File:** `Matrix_Keypad_4x3.c`

```c
// 4×3 Matrix Keypad (Telephone Style)
// Common layout: 1-9, *, 0, #
// Target: 8051 @ 12MHz

#include <reg51.h>

// Rows (outputs)
sbit ROW1 = P1^0;
sbit ROW2 = P1^1;
sbit ROW3 = P1^2;
sbit ROW4 = P1^3;

// Columns (inputs)
sbit COL1 = P1^4;
sbit COL2 = P1^5;
sbit COL3 = P1^6;

// Key layout for 4×3 keypad
unsigned char code key_map_4x3[4][3] = {
    {'1', '2', '3'},
    {'4', '5', '6'},
    {'7', '8', '9'},
    {'*', '0', '#'}
};

void keypad_init() {
    ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 1;
    COL1 = 1; COL2 = 1; COL3 = 1;
}

unsigned char keypad_scan_4x3() {
    unsigned char row, col;

    for(row = 0; row < 4; row++) {
        // Activate row
        switch(row) {
            case 0: ROW1 = 0; ROW2 = 1; ROW3 = 1; ROW4 = 1; break;
            case 1: ROW1 = 1; ROW2 = 0; ROW3 = 1; ROW4 = 1; break;
            case 2: ROW1 = 1; ROW2 = 1; ROW3 = 0; ROW4 = 1; break;
            case 3: ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 0; break;
        }

        // Check columns
        for(col = 0; col < 3; col++) {
            unsigned char pressed = 0;

            switch(col) {
                case 0: pressed = (COL1 == 0); break;
                case 1: pressed = (COL2 == 0); break;
                case 2: pressed = (COL3 == 0); break;
            }

            if(pressed) {
                // Debounce
                unsigned int i;
                for(i = 0; i < 2000; i++);  // ~20ms

                // Verify
                unsigned char still_pressed = 0;
                switch(col) {
                    case 0: still_pressed = (COL1 == 0); break;
                    case 1: still_pressed = (COL2 == 0); break;
                    case 2: still_pressed = (COL3 == 0); break;
                }

                if(still_pressed) {
                    unsigned char key = key_map_4x3[row][col];

                    // Wait for release
                    switch(col) {
                        case 0: while(COL1 == 0); break;
                        case 1: while(COL2 == 0); break;
                        case 2: while(COL3 == 0); break;
                    }

                    return key;
                }
            }
        }
    }

    return 0;
}

void main() {
    unsigned char key;
    keypad_init();

    while(1) {
        key = keypad_scan_4x3();

        if(key != 0) {
            // Process key
            P2 = key;  // Display on port 2
        }
    }
}
```

---

### Version 4: Password Entry System

**File:** `Keypad_Password.c`

```c
// Password Entry System with 4×4 Keypad
// Enter 4-digit password to unlock
// Target: 8051 @ 12MHz

#include <reg51.h>

// Keypad pins
sbit ROW1 = P1^0; sbit ROW2 = P1^1; sbit ROW3 = P1^2; sbit ROW4 = P1^3;
sbit COL1 = P1^4; sbit COL2 = P1^5; sbit COL3 = P1^6; sbit COL4 = P1^7;

// LED indicators
sbit LED_UNLOCK = P2^0;  // Green LED - unlocked
sbit LED_LOCKED = P2^1;  // Red LED - locked
sbit LED_ACCESS = P2^2;  // Yellow LED - processing

unsigned char code key_map[4][4] = {
    {'1', '2', '3', 'A'},
    {'4', '5', '6', 'B'},
    {'7', '8', '9', 'C'},
    {'*', '0', '#', 'D'}
};

// Correct password
unsigned char code password[5] = "1234";
unsigned char input_buffer[5];
unsigned char input_index = 0;

#define PASSWORD_LENGTH 4

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void keypad_init() {
    ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 1;
    COL1 = 1; COL2 = 1; COL3 = 1; COL4 = 1;
    LED_UNLOCK = 1;   // OFF
    LED_LOCKED = 0;   // ON (active low)
    LED_ACCESS = 1;   // OFF
}

unsigned char keypad_scan() {
    unsigned char row, col;

    for(row = 0; row < 4; row++) {
        switch(row) {
            case 0: ROW1 = 0; ROW2 = 1; ROW3 = 1; ROW4 = 1; break;
            case 1: ROW1 = 1; ROW2 = 0; ROW3 = 1; ROW4 = 1; break;
            case 2: ROW1 = 1; ROW2 = 1; ROW3 = 0; ROW4 = 1; break;
            case 3: ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 0; break;
        }

        for(col = 0; col < 4; col++) {
            unsigned char pressed = 0;
            switch(col) {
                case 0: pressed = (COL1 == 0); break;
                case 1: pressed = (COL2 == 0); break;
                case 2: pressed = (COL3 == 0); break;
                case 3: pressed = (COL4 == 0); break;
            }

            if(pressed) {
                delay_ms(20);
                unsigned char still = 0;
                switch(col) {
                    case 0: still = (COL1 == 0); break;
                    case 1: still = (COL2 == 0); break;
                    case 2: still = (COL3 == 0); break;
                    case 3: still = (COL4 == 0); break;
                }
                if(still) {
                    unsigned char key = key_map[row][col];
                    switch(col) {
                        case 0: while(COL1 == 0); break;
                        case 1: while(COL2 == 0); break;
                        case 2: while(COL3 == 0); break;
                        case 3: while(COL4 == 0); break;
                    }
                    return key;
                }
            }
        }
    }
    return 0;
}

bit compare_password() {
    unsigned char i;
    for(i = 0; i < PASSWORD_LENGTH; i++) {
        if(input_buffer[i] != password[i]) {
            return 0;  // Wrong
        }
    }
    return 1;  // Correct
}

void unlock_success() {
    LED_LOCKED = 1;   // OFF
    LED_UNLOCK = 0;   // ON
    LED_ACCESS = 1;   // OFF

    // Stay unlocked for 5 seconds
    delay_ms(5000);

    // Relock
    LED_LOCKED = 0;
    LED_UNLOCK = 1;
}

void unlock_failed() {
    LED_ACCESS = 0;   // Flash yellow
    delay_ms(100);
    LED_ACCESS = 1;
    delay_ms(100);
    LED_ACCESS = 0;
    delay_ms(100);
    LED_ACCESS = 1;
    delay_ms(100);
    LED_ACCESS = 0;
    delay_ms(100);
    LED_ACCESS = 1;
}

void reset_input() {
    input_index = 0;
    unsigned char i;
    for(i = 0; i < 5; i++) {
        input_buffer[i] = 0;
    }
}

void main() {
    unsigned char key;

    keypad_init();
    reset_input();

    while(1) {
        key = keypad_scan();

        if(key != 0) {
            // Check if key is a digit
            if(key >= '0' && key <= '9') {
                if(input_index < PASSWORD_LENGTH) {
                    input_buffer[input_index++] = key;
                    LED_ACCESS = 0;  // Brief flash
                    delay_ms(50);
                    LED_ACCESS = 1;
                }
            }
            // Check for clear (*)
            else if(key == '*') {
                reset_input();
            }
            // Check for enter (#)
            else if(key == '#') {
                if(input_index == PASSWORD_LENGTH) {
                    if(compare_password()) {
                        unlock_success();
                    } else {
                        unlock_failed();
                    }
                }
                reset_input();
            }
        }
    }
}
```

---

### Version 5: Keypad to 7-Segment Display

**File:** `Keypad_To_Segment.c`

```c
// Keypad Input to 7-Segment Display
// Shows pressed number on display
// Target: 8051 @ 12MHz

#include <reg51.h>

// Keypad on P1
sbit ROW1 = P1^0; sbit ROW2 = P1^1; sbit ROW3 = P1^2; sbit ROW4 = P1^3;
sbit COL1 = P1^4; sbit COL2 = P1^5; sbit COL3 = P1^6; sbit COL4 = P1^7;

// 7-segment display on P2
unsigned char code segment_pattern[11] = {
    0x3F,  // 0
    0x06,  // 1
    0x5B,  // 2
    0x4F,  // 3
    0x66,  // 4
    0x6D,  // 5
    0x7D,  // 6
    0x07,  // 7
    0x7F,  // 8
    0x6F,  // 9
    0x00   // Blank (for non-digit keys)
};

unsigned char code key_map[4][4] = {
    {'1', '2', '3', 'A'},
    {'4', '5', '6', 'B'},
    {'7', '8', '9', 'C'},
    {'*', '0', '#', 'D'}
};

void delay_ms(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);
}

void keypad_init() {
    ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 1;
    COL1 = 1; COL2 = 1; COL3 = 1; COL4 = 1;
    P2 = 0x00;  // Start with display off
}

unsigned char keypad_scan() {
    unsigned char row, col;

    for(row = 0; row < 4; row++) {
        switch(row) {
            case 0: ROW1 = 0; ROW2 = 1; ROW3 = 1; ROW4 = 1; break;
            case 1: ROW1 = 1; ROW2 = 0; ROW3 = 1; ROW4 = 1; break;
            case 2: ROW1 = 1; ROW2 = 1; ROW3 = 0; ROW4 = 1; break;
            case 3: ROW1 = 1; ROW2 = 1; ROW3 = 1; ROW4 = 0; break;
        }

        for(col = 0; col < 4; col++) {
            unsigned char pressed = 0;
            switch(col) {
                case 0: pressed = (COL1 == 0); break;
                case 1: pressed = (COL2 == 0); break;
                case 2: pressed = (COL3 == 0); break;
                case 3: pressed = (COL4 == 0); break;
            }

            if(pressed) {
                delay_ms(20);
                unsigned char still = 0;
                switch(col) {
                    case 0: still = (COL1 == 0); break;
                    case 1: still = (COL2 == 0); break;
                    case 2: still = (COL3 == 0); break;
                    case 3: still = (COL4 == 0); break;
                }
                if(still) {
                    unsigned char key = key_map[row][col];
                    switch(col) {
                        case 0: while(COL1 == 0); break;
                        case 1: while(COL2 == 0); break;
                        case 2: while(COL3 == 0); break;
                        case 3: while(COL4 == 0); break;
                    }
                    return key;
                }
            }
        }
    }
    return 0;
}

unsigned char key_to_segment(unsigned char key) {
    if(key >= '0' && key <= '9') {
        return segment_pattern[key - '0'];
    }
    return segment_pattern[10];  // Blank for non-digits
}

void main() {
    unsigned char key, segment;

    keypad_init();

    while(1) {
        key = keypad_scan();

        if(key != 0) {
            segment = key_to_segment(key);
            P2 = segment;  // Display on 7-segment
        }
    }
}
```

---

## 🔌 Hardware Connection

### 4×4 Matrix Keypad Connection

```
         8051                    4×4 Matrix Keypad
       ┌──────┐                 ┌─────────────────┐
       │      │                 │                 │
       │ P1.0 ├─────────────────┤ Row 1           │
       │ P1.1 ├─────────────────┤ Row 2           │
       │ P1.2 ├─────────────────┤ Row 3           │
       │ P1.3 ├─────────────────┤ Row 4           │
       │      │                 │                 │
       │ P1.4 ├─────────────────┤ Col 1           │
       │ P1.5 ├─────────────────┤ Col 2           │
       │ P1.6 ├─────────────────┤ Col 3           │
       │ P1.7 ├─────────────────┤ Col 4           │
       │      │                 │                 │
       │ GND  ├─────────────────┤ GND (common)    │
       │      │                 │                 │
       └──────┘                 │   [1] [2] [3] [A]│
                                 │   [4] [5] [6] [B]│
                                 │   [7] [8] [9] [C]│
                                 │   [*] [0] [#] [D]│
                                 └─────────────────┘

Keypad Layout:
    Col1  Col2  Col3  Col4
Row1:   1     2     3     A
Row2:   4     5     6     B
Row3:   7     8     9     C
Row4:   *     0     #     D
```

### With External Pull-up Resistors (Recommended)

```
         8051
       ┌──────┐
       │      │        VCC (5V)
       │      │           │
       │ P1.4 ├───────────┐││
       │ P1.5 ├───────────┤││  10kΩ Pull-up
       │ P1.6 ├───────────┤││  (on columns)
       │ P1.7 ├───────────┤││
       │      │           │ └┐
       │      │           ┌─┴┐│
       │      │           │ │ │
       └──────┘           └─┬─┘
                          │

Better noise immunity with external pull-ups
```

### 4×3 Matrix Keypad Connection

```
         8051                    4×3 Matrix Keypad
       ┌──────┐                 ┌─────────────────┐
       │      │                 │                 │
       │ P1.0 ├─────────────────┤ Row 1           │
       │ P1.1 ├─────────────────┤ Row 2           │
       │ P1.2 ├─────────────────┤ Row 3           │
       │ P1.3 ├─────────────────┤ Row 4           │
       │      │                 │                 │
       │ P1.4 ├─────────────────┤ Col 1           │
       │ P1.5 ├─────────────────┤ Col 2           │
       │ P1.6 ├─────────────────┤ Col 3           │
       │      │                 │                 │
       │ GND  ├─────────────────┤ GND             │
       │      │                 │                 │
       └──────┘                 │   [1] [2] [3]   │
                                 │   [4] [5] [6]   │
                                 │   [7] [8] [9]   │
                                 │   [*] [0] [#]   │
                                 └─────────────────┘

Telephone-style layout (4×3)
```

---

## 📖 Code Explanation

### 1. Matrix Keypad Principle

**How it works:**

A matrix keypad arranges switches in a grid of rows and columns. Each key sits at the intersection of a row and column.

```
     Col1  Col2  Col3  Col4
Row1:  [1]   [2]   [3]   [A]
Row2:  [4]   [5]   [6]   [B]
Row3:  [7]   [8]   [9]   [C]
Row4:  [*]   [0]   [#]   [D]

Example: Key '5' is at Row2, Col2
         Key 'A' is at Row1, Col4
```

**Scanning process:**

```
Step 1: Set Row1 LOW, others HIGH
        → Check columns
        → If Col1=LOW, key '1' pressed
        → If Col2=LOW, key '2' pressed
        → etc.

Step 2: Set Row2 LOW, others HIGH
        → Check columns
        → If Col2=LOW, key '5' pressed
        → etc.

Step 3: Repeat for Row3, Row4
```

### 2. Why Matrix Scanning Works

**Without pull-ups on columns:**
```
Row1=LOW, Row2=HIGH, Row3=HIGH, Row4=HIGH

If key '5' (Row2,Col2) pressed:
  Row2 connects to Col2 through the switch
  But Row2 is HIGH, so Col2 reads HIGH
  → Cannot detect the press!

Solution: Use pull-ups on columns
  When no key pressed: Col2 pulled HIGH
  When key pressed:
    If Row1=LOW and key '2' pressed:
      Col2 connected to LOW → reads LOW
    If Row2=HIGH and key '5' pressed:
      Col2 connected to HIGH → still HIGH
    → Only detect when scanning row is LOW
```

### 3. Debouncing Explained

**Mechanical switch bounce:**
```
Actual Signal: ┐┌┐┌┐┌┐┌┐┌──────┐
5V ────────────┘└┘└┘└┘└┘└┘      └─────
              ← 20-50ms bounce →

Program should read:
5V ────────────┐──────────────┐
                └──────────────┘
              Wait 20ms → Read again → Verify
```

**Debounce strategy:**
1. Detect initial press
2. Wait 20ms (debounce delay)
3. Read again to verify
4. If still pressed → valid key
5. Wait for release
6. Another 20ms debounce on release

### 4. Alternative: Port-Based Scanning (Faster)

```c
// Using entire port for faster scanning
unsigned char keypad_scan_fast() {
    unsigned char row, col, temp;

    for(row = 0; row < 4; row++) {
        // Set row pattern
        P1 = ~(1 << row);  // Only one row LOW

        // Small delay
        // ...

        // Read columns
        temp = P1 >> 4;  // Get column bits

        // Check each column
        for(col = 0; col < 4; col++) {
            if((temp & (1 << col)) == 0) {
                // Key pressed at (row, col)
                return key_map[row][col];
            }
        }
    }

    return 0;
}
```

---

## 🔬 Keypad Types

### 4×4 Matrix Keypad

**Layout:**
```
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ A │
├───┼───┼───┼───┤
│ 4 │ 5 │ 6 │ B │
├───┼───┼───┼───┤
│ 7 │ 8 │ 9 │ C │
├───┼───┼───┼───┤
│ * │ 0 │ # │ D │
└───┴───┴───┴───┘
```

**Key codes (ASCII):**
- Digits: '0'-'9' (0x30-0x39)
- Letters: 'A', 'B', 'C', 'D' (0x41-0x44)
- Symbols: '*' (0x2A), '#' (0x23)

**Applications:**
- Industrial control panels
- Security systems
- Calculator input
- Menu navigation

### 4×3 Matrix Keypad (Telephone Style)

**Layout:**
```
┌───┬───┬───┐
│ 1 │ 2 │ 3 │
├───┼───┼───┤
│ 4 │ 5 │ 6 │
├───┼───┼───┤
│ 7 │ 8 │ 9 │
├───┼───┼───┤
│ * │ 0 │ # │
└───┴───┴───┘
```

**Key codes:**
- Digits: '0'-'9'
- Symbols: '*', '#'

**Applications:**
- Telephone systems
- Security PIN entry
- Simple numeric input
- Door access control

### Custom Key Maps

```c
// Example: Control panel layout
unsigned char code custom_map[4][4] = {
    {'U', 'D', 'L', 'R'},  // Up, Down, Left, Right
    {'+', '-', 'M', 'P'},  // Plus, Minus, Mode, Power
    {'S', 'E', 'C', 'R'},  // Start, Enter, Cancel, Reset
    {'1', '2', '3', '4'}   // Preset values
};
```

---

## 🛠️ Testing and Troubleshooting

### Test 1: Verify Hardware Connections

```c
void test_connections() {
    while(1) {
        // Test Row 1
        ROW1 = 0; ROW2 = 1; ROW3 = 1; ROW4 = 1;

        // If any column reads LOW, there's a short or pressed key
        if(COL1 == 0) P2 = 0x01;  // Indicate on LEDs
        if(COL2 == 0) P2 = 0x02;
        if(COL3 == 0) P2 = 0x04;
        if(COL4 == 0) P2 = 0x08;
    }
}
```

### Test 2: Single Key Test

```c
// Test if a specific key works
void test_single_key() {
    while(1) {
        ROW1 = 0; ROW2 = 1; ROW3 = 1; ROW4 = 1;

        if(COL1 == 0) {
            // Key '1' should be pressed
            P2 = 0xFF;  // All LEDs ON
        } else {
            P2 = 0x00;  // All LEDs OFF
        }
    }
}
```

### Common Issues and Solutions

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| No keys detected | Rows/columns swapped | Check pin assignment |
| Multiple keys detected | Stuck key or short circuit | Check keypad hardware |
| Wrong key detected | Key map mismatch | Verify key_map array |
| Erratic readings | No pull-ups on columns | Add external 10kΩ pull-ups |
| Keys repeat | Missing release wait | Add while(COLx == 0) loop |
| Slow response | Scan delay too long | Reduce debounce delay |
| Ghost keys | 3+ keys pressed simultaneously | Normal for matrix keypads |

### Debugging Steps

**1. Check Pin Assignments:**
```c
// Verify with multimeter:
// - Set each row LOW one at a time
// - Measure column voltages
// - Should be 0V when key pressed
```

**2. Test Pull-ups:**
```c
// With no keys pressed, all columns should be HIGH
void test_pullups() {
    ROW1 = 0;  // Any row LOW

    // All columns should read HIGH (1)
    if(COL1 == 0) {
        // Missing pull-up or short to ground
    }
}
```

**3. Verify Scanning Sequence:**
```c
// Add LED indicators to see scanning
void debug_scan() {
    unsigned char row;

    for(row = 0; row < 4; row++) {
        switch(row) {
            case 0: ROW1 = 0; ROW2 = 1; ROW3 = 1; ROW4 = 1;
                    P2 = 0x01; break;  // LED 1 = scanning row 1
            case 1: ROW1 = 1; ROW2 = 0; ROW3 = 1; ROW4 = 1;
                    P2 = 0x02; break;  // LED 2 = scanning row 2
            // etc.
        }
        delay_ms(100);
    }
}
```

---

## 📊 Variations and Extensions

### Variation 1: Multi-Key Press Detection

```c
// Detect up to 2 simultaneous keys
unsigned char read_multiple_keys() {
    static unsigned char key1 = 0, key2 = 0;
    unsigned char key, count = 0;

    key = keypad_scan();
    if(key != 0) {
        if(key1 == 0) {
            key1 = key;
            count++;
        } else if(key2 == 0 && key != key1) {
            key2 = key;
            count++;
        }
    }

    // Clear keys after processing
    // ...

    return count;
}
```

### Variation 2: Long Press Detection

```c
// Detect normal press vs long press
unsigned char keypad_scan_with_duration() {
    unsigned int press_time = 0;

    // ... detect key press ...

    if(key != 0) {
        while(COLx == 0) {  // While still pressed
            delay_ms(100);
            press_time++;
            if(press_time > 20) {  // 2 seconds
                // Long press detected
                return key | 0x80;  // Set bit 7 to indicate long press
            }
        }
    }

    return key;  // Normal press
}
```

### Variation 3: Key Repeat (Typematic)

```c
// Like computer keyboard: repeat after holding
unsigned char keypad_scan_repeat() {
    static unsigned int repeat_count = 0;
    static unsigned char last_key = 0;
    unsigned char key;

    key = keypad_scan();

    if(key != 0) {
        if(key == last_key) {
            repeat_count++;
            if(repeat_count > 10) {  // Initial delay
                repeat_count = 8;  // Faster repeat
                return key;  // Return repeated key
            }
        } else {
            last_key = key;
            repeat_count = 0;
            return key;  // New key
        }
    } else {
        last_key = 0;
        repeat_count = 0;
    }

    return 0;
}
```

### Variation 4: Keypad with LCD Display

```c
// Display key presses on LCD
void key_to_lcd(unsigned char key) {
    if(key >= '0' && key <= '9') {
        lcd_write_char(key);
    } else if(key == '*') {
        lcd_command(CLEAR_DISPLAY);
    } else if(key == '#') {
        lcd_write_string("ENTER");
    } else {
        lcd_write_char(key);
    }
}
```

### Variation 5: Calculator Mode

```c
// Simple calculator using keypad
long calculator_mode() {
    unsigned char buffer[16];
    unsigned char index = 0;
    long operand1 = 0, operand2 = 0;
    char operation = 0;

    while(1) {
        unsigned char key = keypad_scan();

        if(key != 0) {
            if(key >= '0' && key <= '9') {
                buffer[index++] = key;
                display_to_7segment(key);
            }
            else if(key == '+' || key == '-' || key == '*') {
                operand1 = atoi(buffer);
                operation = key;
                index = 0;
            }
            else if(key == '#') {  // Enter
                operand2 = atoi(buffer);
                long result = 0;
                switch(operation) {
                    case '+': result = operand1 + operand2; break;
                    case '-': result = operand1 - operand2; break;
                    case '*': result = operand1 * operand2; break;
                }
                display_number(result);
                index = 0;
            }
            else if(key == '*') {  // Clear
                index = 0;
            }
        }
    }
}
```

### Variation 6: Menu Navigation

```c
// Use keypad for menu navigation
unsigned char menu_navigation() {
    unsigned char menu_item = 0;
    unsigned char num_items = 5;

    while(1) {
        display_menu_item(menu_item);

        unsigned char key = keypad_scan();

        if(key == '2') {  // Up
            if(menu_item > 0) menu_item--;
        }
        else if(key == '8') {  // Down
            if(menu_item < num_items - 1) menu_item++;
        }
        else if(key == '5') {  // Select
            return menu_item;
        }
        else if(key == '*') {  // Exit
            return 0xFF;
        }
    }
}
```

---

## ⚡ Hardware Considerations

### Pin Efficiency Comparison

**Direct connection (non-matrix):**
```
16 keys = 16 I/O pins
```

**Matrix connection:**
```
4×4 keypad = 4 + 4 = 8 I/O pins (50% reduction!)
4×3 keypad = 4 + 3 = 7 I/O pins
```

**Maximum keys with 8 pins:**
```
Matrix: 4×4 = 16 keys
Direct: 8 keys
```

### Pull-up Resistor Selection

**Internal pull-ups (8051):**
- Weak (~60µA source)
- May not be reliable for all keypads
- Susceptible to noise

**External pull-ups (recommended):**
```c
Value: 4.7kΩ - 10kΩ
Power: 5V / 10kΩ = 0.5mA (acceptable)
Response: Fast
Noise immunity: Good
```

**Why not larger?**
- 100kΩ: Too slow, susceptible to EMI
- 1kΩ: Too much current (5mA), unnecessary

### Ghosting and Masking

**Ghosting:**
```
When 3 keys are pressed:
  Row1+Col1 (Key 1), Row1+Col2 (Key 2), Row2+Col1 (Key 4)
A 4th "ghost" key appears at Row2+Col2

This is normal for matrix keypads
Most applications ignore ghost keys
```

**Solutions:**
1. **Diode matrix** (complex, expensive)
2. **Software filtering** (detect and ignore)
3. **Limit to 2 simultaneous keys** (most common)

### PCB Layout Tips

```
Good layout:
- Keep traces short
- Add decoupling capacitors near keypad
- Use ground plane if possible
- Add ESD protection for exposed buttons

Bad layout:
- Long parallel traces (crosstalk)
- No pull-ups nearby
- Sharp corners on traces
```

---

## 📚 What You've Learned

✅ **Matrix Keypad Principles**
- Row and column scanning
- How matrix reduces pin count
- Key mapping and detection

✅ **Scanning Algorithms**
- Sequential row scanning
- Column checking
- Key code generation

✅ **Debouncing Techniques**
- Software delay debouncing
- Verification reading
- Release detection

✅ **Practical Applications**
- Password entry systems
- Menu navigation
- Numeric input
- Calculator input

✅ **Hardware Design**
- Pull-up resistor selection
- Pin assignment
- Connection schemes
- Troubleshooting

---

## 🚀 Next Steps

After mastering this example:

1. **Build Practical Projects:**
   - Digital door lock with password
   - Calculator with LCD display
   - Menu-driven control system
   - Security system keypad

2. **Add Display:**
   - LCD character display
   - 7-segment display
   - OLED display
   - TFT touchscreen

3. **Advanced Features:**
   - Multi-key detection
   - Long press recognition
   - Key repeat (typematic)
   - Macro keys

4. **Security Enhancements:**
   - Password hashing
   - Attempt limiting
   - Lockout after failures
   - Random PIN challenges

5. **Integration:**
   - Relay control (unlock door)
   - Buzzer feedback
   - EEPROM password storage
   - Real-time clock for time-based access

6. **Related Topics:**
   - Learn about I2C keypad controllers
   - Explore touchscreen interfaces
   - Study rotary encoders
   - See: [Timer Examples](../02_Timers/)
   - See: [Interrupt Examples](../03_Interrupts/)

---

## 🎓 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Matrix Scanning** | Sequential activation of rows to detect column states |
| **Row/Column** | Grid arrangement for efficient key detection |
| **Pull-up Resistor** | Keeps column HIGH when no key pressed |
| **Active Low** | Key detected when line goes LOW |
| **Debouncing** | Wait for contact bounce to settle |
| **Ghosting** | False key detection with 3+ simultaneous presses |
| **Key Map** | Array mapping (row,col) to key values |
| **4×4 Keypad** | 16 keys using 8 I/O pins |
| **4×3 Keypad** | 12 keys using 7 I/O pins (telephone style) |
| **Scanning Frequency** | How fast keypad is checked (typically 50-100Hz) |

---

## 🐛 Debug Checklist

Before asking for help, check:

**Hardware:**
- [ ] Correct pin connections (rows and columns)
- [ ] Pull-up resistors on columns (10kΩ recommended)
- [ ] Good ground connection
- [ ] No solder bridges on keypad
- [ ] Proper voltage (5V)

**Software:**
- [ ] Correct key mapping array
- [ ] Rows set as outputs
- [ ] Columns set as inputs
- [ ] Debounce delay appropriate (20ms)
- [ ] Wait for key release
- [ ] Scanning all rows and columns

**Testing:**
- [ ] Each key tested individually
- [ ] No stuck keys
- [ ] Response time acceptable
- [ ] No ghost keys with normal use
- [ ] Works reliably over time

---

## 📖 Additional Resources

### Datasheets
- [4×4 Matrix Keypad Datasheet](https://www.sparkfun.com/datasheets/Components/Buttons/4x4-Button-Pad-Datasheet.pdf)
- [Keypad Interface Guide](https://www.maximintegrated.com/en/design/technical-documents/app-notes/4/4802.html)

### Learning Materials
- [Matrix Keypad Tutorial](https://www.engineersgarage.com/matrix-keypad-tutorial/)
- [Scanning Techniques](https://www.electronics-tutorials.ws/io/switch-debouncing.html)
- [Keyboard Matrix Theory](https://www.avrfreaks.net/sites/default/files/keyboard%20scan.pdf)

### Tools
- **Online Keypad Designer:** Design custom layouts
- **Keypad Testers:** Diagnostic tools
- **PCB Design Software:** Eagle, KiCad

### Projects
- Digital Door Lock
- Calculator Project
- Menu System Tutorial
- Security Access Control

---

## 🤝 Contributing

Have improvements? Found a bug?
- Add more variations
- Improve documentation
- Fix errors
- Add troubleshooting tips
- Share your keypad layouts!

---

**Difficulty:** ⭐⭐ Intermediate
**Time to Complete:** 3-4 hours
**Hardware Required:** 4×4 or 4×3 matrix keypad, pull-up resistors
**Fun Factor:** 🔢 High (interactive input!)

**Happy Coding!** 💡
