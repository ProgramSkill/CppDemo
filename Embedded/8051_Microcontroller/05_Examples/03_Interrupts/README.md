# Interrupt Examples

## Overview

This section demonstrates interrupt handling in the 8051 microcontroller. Interrupts allow the CPU to respond to events asynchronously, without continuous polling.

## What are Interrupts?

Interrupts are signals that temporarily suspend the normal program execution to handle high-priority events. After handling the interrupt, the CPU returns to the original program.

**Key Benefits:**
- No CPU time wasted on polling
- Immediate response to events
- Ability to handle multiple events
- Precise timing control

## Interrupt Sources

| Source | Vector Address | Priority | Flag |
|--------|----------------|----------|------|
| INT0 (P3.2) | 0003H | Highest | IE0 |
| Timer 0 | 000BH | High | TF0 |
| INT1 (P3.3) | 0013H | Medium | IE1 |
| Timer 1 | 001BH | Low | TF1 |
| Serial | 0023H | Lowest | RI/TI |

## Examples

### External_Interrupt.c

**Difficulty:** ⭐⭐ Intermediate

**Description:**
External interrupt handling using INT0 and INT1. Demonstrates:
- Edge-triggered interrupts
- Interrupt service routines (ISR)
- Button debouncing with interrupts

**Hardware:**
- Buttons on P3.2 (INT0) and P3.3 (INT1)
- LEDs on P1.0 and P1.1 for indication

**Key Concepts:**
- IT0/IT1 configuration (edge/level)
- Interrupt enable setup
- ISR structure and best practices

```c
/**
 * External Interrupt Example for 8051 Microcontroller
 *
 * Description:
 *   Demonstrates external interrupt handling using INT0 (P3.2) and INT1 (P3.3)
 *   Each interrupt toggles an LED on P1.0 (INT0) and P1.1 (INT1)
 *
 * Hardware:
 *   - Button 1 connected to P3.2 (INT0) with pull-up resistor
 *   - Button 2 connected to P3.3 (INT1) with pull-up resistor
 *   - LED 1 connected to P1.0
 *   - LED 2 connected to P1.1
 *
 * Author: Example Code
 * Date: 2024
 */

#include <reg51.h>

// Define LED pins
#define LED1 P1_0    // Controlled by INT0
#define LED2 P1_1    // Controlled by INT1

// Debounce delay in milliseconds
#define DEBOUNCE_DELAY 20

/**
 * Delay function for debouncing
 * @param ms: milliseconds to delay (approximately)
 */
void delay_ms(unsigned int ms)
{
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 127; j++);
}

/**
 * Interrupt Service Routine for INT0 (External Interrupt 0)
 * Triggered by falling edge on P3.2
 * Vector Address: 0x0003
 */
void external0_isr(void) interrupt 0
{
    // Simple debounce delay
    delay_ms(DEBOUNCE_DELAY);

    // Toggle LED1
    LED1 = ~LED1;

    // Wait for button release (optional, prevents multiple triggers)
    while(INT0 == 0);  // Wait while button is pressed
    delay_ms(DEBOUNCE_DELAY);
}

/**
 * Interrupt Service Routine for INT1 (External Interrupt 1)
 * Triggered by falling edge on P3.3
 * Vector Address: 0x0013
 */
void external1_isr(void) interrupt 2
{
    // Simple debounce delay
    delay_ms(DEBOUNCE_DELAY);

    // Toggle LED2
    LED2 = ~LED2;

    // Wait for button release
    while(INT1 == 0);  // Wait while button is pressed
    delay_ms(DEBOUNCE_DELAY);
}

/**
 * Initialize external interrupts
 */
void init_external_interrupts(void)
{
    // Configure interrupt trigger type (1 = falling edge, 0 = low level)
    IT0 = 1;    // INT0 edge-triggered
    IT1 = 1;    // INT1 edge-triggered

    // Clear any pending interrupt flags
    IE0 = 0;
    IE1 = 0;

    // Enable individual interrupts
    EX0 = 1;    // Enable INT0
    EX1 = 1;    // Enable INT1

    // Enable global interrupts
    EA = 1;
}

/**
 * Initialize ports
 */
void init_ports(void)
{
    // Set LED pins as output
    LED1 = 0;   // Turn off LED1
    LED2 = 0;   // Turn off LED2
}

/**
 * Main program
 */
void main(void)
{
    // Initialize hardware
    init_ports();
    init_external_interrupts();

    // Main loop - does nothing, waiting for interrupts
    while(1)
    {
        // Normal program execution continues here
        // Interrupts will be handled asynchronously

        // You can add other tasks here
        // For example: display updates, background processing
    }
}
```

### Timer_Interrupt.c

**Difficulty:** ⭐⭐ Intermediate

**Description:**
Periodic tasks using timer interrupts. Demonstrates:
- Timer overflow interrupts
- Multi-rate timing
- Real-time task scheduling

**Applications:**
- LED flashing without blocking
- Periodic sensor reading
- Real-time clock

```c
/**
 * Timer Interrupt Example for 8051 Microcontroller
 *
 * Description:
 *   Demonstrates timer-based interrupts for periodic task execution
 *   Creates two different timing intervals using software counters
 *
 * Hardware:
 *   - LED connected to P1.0 (flashes every 500ms)
 *   - LED connected to P1.1 (flashes every 1000ms)
 *
 * Author: Example Code
 * Date: 2024
 */

#include <reg51.h>

#define LED1 P1_0
#define LED2 P1_1

// Timer 0 interrupt generates a tick every 1ms (assuming 12MHz crystal)
// Using 16-bit timer mode (Mode 1)
#define TIMER0_RELOAD_VALUE 0xFC18  // Reload for 1ms at 12MHz

// Software counters for different timing intervals
volatile unsigned int counter_500ms = 0;
volatile unsigned int counter_1000ms = 0;

volatile bit flag_500ms = 0;
volatile bit flag_1000ms = 0;

/**
 * Timer 0 Interrupt Service Routine
 * Called every 1ms when timer overflows
 * Vector Address: 0x000B
 */
void timer0_isr(void) interrupt 1
{
    // Reload timer for next interrupt
    TH0 = (TIMER0_RELOAD_VALUE >> 8) & 0xFF;  // High byte
    TL0 = TIMER0_RELOAD_VALUE & 0xFF;          // Low byte

    // Increment counters
    counter_500ms++;
    counter_1000ms++;

    // Check 500ms interval
    if(counter_500ms >= 500)
    {
        counter_500ms = 0;
        flag_500ms = 1;
    }

    // Check 1000ms interval
    if(counter_1000ms >= 1000)
    {
        counter_1000ms = 0;
        flag_1000ms = 1;
    }
}

/**
 * Initialize Timer 0 for periodic interrupts
 */
void init_timer0(void)
{
    // Stop timer
    TR0 = 0;

    // Set timer mode (Mode 1: 16-bit timer)
    TMOD &= 0xF0;  // Clear Timer 0 mode bits
    TMOD |= 0x01;  // Set Timer 0 to Mode 1

    // Load initial values
    TH0 = (TIMER0_RELOAD_VALUE >> 8) & 0xFF;
    TL0 = TIMER0_RELOAD_VALUE & 0xFF;

    // Clear timer flag
    TF0 = 0;

    // Enable Timer 0 interrupt
    ET0 = 1;

    // Enable global interrupts
    EA = 1;

    // Start timer
    TR0 = 1;
}

/**
 * Initialize ports
 */
void init_ports(void)
{
    LED1 = 0;
    LED2 = 0;
}

/**
 * Main program
 */
void main(void)
{
    // Initialize hardware
    init_ports();
    init_timer0();

    // Main loop
    while(1)
    {
        // Check 500ms flag
        if(flag_500ms)
        {
            flag_500ms = 0;  // Clear flag
            LED1 = ~LED1;    // Toggle LED1
        }

        // Check 1000ms flag
        if(flag_1000ms)
        {
            flag_1000ms = 0;  // Clear flag
            LED2 = ~LED2;     // Toggle LED2
        }

        // Other tasks can run here
        // The timer interrupts continue independently
    }
}
```

### Serial_Interrupt.c

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Full-duplex serial communication using interrupts. Demonstrates:
- Transmit and receive interrupts
- Ring buffers
- Non-blocking I/O

```c
/**
 * Serial Interrupt Example for 8051 Microcontroller
 *
 * Description:
 *   Demonstrates interrupt-driven serial communication
 *   Uses ring buffers for efficient data handling
 *
 * Hardware:
 *   - Serial connection (RXD/TXD)
 *   - Optional: MAX232 level converter
 *
 * Baud Rate: 9600 (assuming 11.0592MHz crystal)
 *
 * Author: Example Code
 * Date: 2024
 */

#include <reg51.h>

// Buffer size
#define BUFFER_SIZE 64

// Ring buffers for serial communication
unsigned char xdata tx_buffer[BUFFER_SIZE];
unsigned char xdata rx_buffer[BUFFER_SIZE];

volatile unsigned char tx_head = 0;
volatile unsigned char tx_tail = 0;
volatile unsigned char rx_head = 0;
volatile unsigned char rx_tail = 0;

volatile bit tx_busy = 0;

/**
 * Check if TX buffer has space
 */
bit tx_buffer_empty(void)
{
    return (tx_head == tx_tail);
}

/**
 * Check if RX buffer has data
 */
bit rx_buffer_data_ready(void)
{
    return (rx_head != rx_tail);
}

/**
 * Put byte in TX buffer (non-blocking)
 * Returns 1 on success, 0 if buffer full
 */
bit tx_buffer_put(unsigned char data)
{
    unsigned char next_head = (tx_head + 1) % BUFFER_SIZE;

    if(next_head == tx_tail)
        return 0;  // Buffer full

    tx_buffer[tx_head] = data;
    tx_head = next_head;
    return 1;
}

/**
 * Get byte from RX buffer
 * Returns 1 on success, 0 if buffer empty
 */
bit rx_buffer_get(unsigned char *data)
{
    if(rx_head == rx_tail)
        return 0;  // Buffer empty

    *data = rx_buffer[rx_tail];
    rx_tail = (rx_tail + 1) % BUFFER_SIZE;
    return 1;
}

/**
 * Serial Transmit Interrupt Service Routine
 * Triggered when transmit buffer is empty
 * Vector Address: 0x0023
 */
void serial_tx_isr(void) interrupt 4
{
    if(TI)  // Transmit interrupt
    {
        TI = 0;  // Clear interrupt flag

        if(tx_head != tx_tail)
        {
            // Send next byte
            SBUF = tx_buffer[tx_tail];
            tx_tail = (tx_tail + 1) % BUFFER_SIZE;
        }
        else
        {
            tx_busy = 0;  // No more data to send
        }
    }
}

/**
 * Serial Receive Interrupt Service Routine
 * Triggered when data is received
 */
void serial_rx_isr(void) interrupt 4
{
    if(RI)  // Receive interrupt
    {
        RI = 0;  // Clear interrupt flag

        unsigned char next_head = (rx_head + 1) % BUFFER_SIZE;

        if(next_head != rx_tail)
        {
            // Store received byte
            rx_buffer[rx_head] = SBUF;
            rx_head = next_head;
        }
        // If buffer full, data is lost (could add error handling)
    }
}

/**
 * Send string via serial (non-blocking)
 */
void serial_send_string(char *str)
{
    while(*str)
    {
        // Wait if buffer is full (with timeout)
        unsigned char timeout = 255;
        while(!tx_buffer_put(*str) && timeout--)
            ;

        if(!tx_busy)
        {
            tx_busy = 1;
            SBUF = tx_buffer[tx_tail];
            tx_tail = (tx_tail + 1) % BUFFER_SIZE;
        }

        str++;
    }
}

/**
 * Initialize serial port with interrupts
 */
void init_serial(void)
{
    // Initialize buffer indices
    tx_head = tx_tail = 0;
    rx_head = rx_tail = 0;
    tx_busy = 0;

    // Set up timer for baud rate generation
    // Using Timer 1 in Mode 2 (8-bit auto-reload)
    TMOD &= 0x0F;      // Clear Timer 1 mode bits
    TMOD |= 0x20;      // Timer 1, Mode 2

    // Set baud rate to 9600 for 11.0592MHz
    TH1 = 0xFD;        // Reload value
    TL1 = 0xFD;
    TR1 = 1;           // Start Timer 1

    // Configure serial port
    SCON = 0x50;       // Mode 1 (8-bit UART), enable receiver

    // Enable serial interrupts
    ES = 1;            // Enable serial interrupt
    EA = 1;            // Enable global interrupts
}

/**
 * Main program
 */
void main(void)
{
    unsigned char received_char;

    // Initialize serial
    init_serial();

    // Send welcome message
    serial_send_string("Serial Interrupt Example\r\n");
    serial_send_string("Type something and press Enter...\r\n");

    // Main loop
    while(1)
    {
        // Check for received data
        if(rx_buffer_data_ready())
        {
            // Get received character
            if(rx_buffer_get(&received_char))
            {
                // Echo back the character
                tx_buffer_put(received_char);

                // Start transmission if not busy
                if(!tx_busy && !tx_buffer_empty())
                {
                    tx_busy = 1;
                    SBUF = tx_buffer[tx_tail];
                    tx_tail = (tx_tail + 1) % BUFFER_SIZE;
                }
            }
        }

        // Other tasks can run here
    }
}
```

### Multi_Interrupt.c

**Difficulty:** ⭐⭐⭐ Advanced

**Description:**
Coordinating multiple interrupt sources. Demonstrates:
- Priority management
- Interrupt nesting
- Shared resource protection

```c
/**
 * Multiple Interrupt Example for 8051 Microcontroller
 *
 * Description:
 *   Demonstrates coordination of multiple interrupt sources
 *   Uses external interrupts and timer interrupts together
 *
 * Hardware:
 *   - Button 1 on P3.2 (INT0) - high priority
 *   - Button 2 on P3.3 (INT1) - low priority
 *   - Timer 0 - generates periodic ticks
 *   - LED1 on P1.0 - indicates INT0 triggered
 *   - LED2 on P1.1 - indicates INT1 triggered
 *   - LED3 on P1.2 - indicates timer tick
 *
 * Author: Example Code
 * Date: 2024
 */

#include <reg51.h>

#define LED1 P1_0
#define LED2 P1_1
#define LED3 P1_2

// Shared variables (volatile for ISR access)
volatile unsigned int timer_counter = 0;
volatile unsigned char int0_count = 0;
volatile unsigned char int1_count = 0;

volatile bit flag_100ms = 0;
volatile bit flag_int0 = 0;
volatile bit flag_int1 = 0;

// Debounce delay
#define DEBOUNCE_DELAY 20

/**
 * Delay function
 */
void delay_ms(unsigned int ms)
{
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 127; j++);
}

/**
 * Timer 0 Interrupt Service Routine (Low Priority)
 * Called every 1ms
 * Vector Address: 0x000B
 */
void timer0_isr(void) interrupt 1
{
    // Reload timer for 1ms tick at 12MHz
    TH0 = 0xFC;
    TL0 = 0x18;

    // Increment counter
    timer_counter++;

    // Set flag every 100ms
    if(timer_counter >= 100)
    {
        timer_counter = 0;
        flag_100ms = 1;
    }

    // Indicate timer interrupt
    LED3 = 1;
}

/**
 * External Interrupt 0 Service Routine (High Priority)
 * Triggered by falling edge on P3.2
 * Vector Address: 0x0003
 */
void external0_isr(void) interrupt 0
{
    // Debounce
    delay_ms(DEBOUNCE_DELAY);

    // Update counter and set flag
    int0_count++;
    flag_int0 = 1;

    // Visual indication
    LED1 = ~LED1;

    // Wait for button release
    while(INT0 == 0);
    delay_ms(DEBOUNCE_DELAY);
}

/**
 * External Interrupt 1 Service Routine (Low Priority)
 * Triggered by falling edge on P3.3
 * Vector Address: 0x0013
 */
void external1_isr(void) interrupt 2
{
    // Debounce
    delay_ms(DEBOUNCE_DELAY);

    // Update counter and set flag
    int1_count++;
    flag_int1 = 1;

    // Visual indication
    LED2 = ~LED2;

    // Wait for button release
    while(INT1 == 0);
    delay_ms(DEBOUNCE_DELAY);
}

/**
 * Initialize Timer 0
 */
void init_timer0(void)
{
    TR0 = 0;              // Stop timer

    TMOD &= 0xF0;         // Clear Timer 0 mode bits
    TMOD |= 0x01;         // Mode 1: 16-bit timer

    TH0 = 0xFC;           // Initial values for 1ms
    TL0 = 0x18;

    TF0 = 0;              // Clear flag
    ET0 = 1;              // Enable Timer 0 interrupt
    PT0 = 0;              // Low priority

    TR0 = 1;              // Start timer
}

/**
 * Initialize external interrupts
 */
void init_external_interrupts(void)
{
    // Configure trigger type
    IT0 = 1;              // INT0 falling edge
    IT1 = 1;              // INT1 falling edge

    // Clear flags
    IE0 = 0;
    IE1 = 0;

    // Enable interrupts
    EX0 = 1;              // Enable INT0
    EX1 = 1;              // Enable INT1

    // Set priorities
    PX0 = 1;              // INT0 high priority
    PX1 = 0;              // INT1 low priority
}

/**
 * Initialize system
 */
void init_system(void)
{
    // Initialize ports
    LED1 = 0;
    LED2 = 0;
    LED3 = 0;

    // Initialize interrupts
    init_timer0();
    init_external_interrupts();

    // Enable global interrupts
    EA = 1;
}

/**
 * Main program
 */
void main(void)
{
    // Initialize system
    init_system();

    // Main loop
    while(1)
    {
        // Handle 100ms timer flag
        if(flag_100ms)
        {
            flag_100ms = 0;

            // Clear timer LED indication
            LED3 = 0;

            // Perform periodic tasks
            // For example: update display, check sensors, etc.
        }

        // Handle INT0 flag
        if(flag_int0)
        {
            flag_int0 = 0;

            // Process INT0 event
            // Could send message, update state, etc.
        }

        // Handle INT1 flag
        if(flag_int1)
        {
            flag_int1 = 0;

            // Process INT1 event
        }

        // Other background tasks
        // The system continues to respond to interrupts
    }
}
```

---

## Interrupt Structure Template

```c
#include <reg51.h>

// 1. Global variables (use volatile for shared data)
volatile unsigned char flag = 0;

// 2. Interrupt Service Routine
void external0_isr(void) interrupt 0
{
    // Keep ISRs short!
    flag = 1;          // Set flag for main program
    // Perform time-critical actions only
}

// 3. Initialization
void init_interrupts(void)
{
    IT0 = 1;           // Edge-triggered
    EX0 = 1;           // Enable INT0
    EA = 1;            // Global interrupt enable
}

// 4. Main program
void main(void)
{
    init_interrupts();

    while(1)
    {
        if(flag)
        {
            flag = 0;  // Clear flag
            // Handle event
        }
        // Other tasks can run here
    }
}
```

---

## Best Practices

### ✅ DO's
1. **Keep ISRs Short** - Only set flags or do minimal processing
2. **Use volatile** - For variables shared with ISRs
3. **Save Context** - Push/pop used registers
4. **Clear Flags** - Clear interrupt flags appropriately
5. **Enable Globally Last** - Set EA bit last in initialization

### ❌ DON'Ts
1. **Long Processing in ISR** - Blocks other interrupts
2. **Heavy Calculations** - Move to main program
3. **Call Complex Functions** - Keep ISRs simple
4. **Forget to Clear Flags** - Causes repeated interrupts
5. **Enable All Interrupts** - Only enable what you need

---

## Interrupt Priority

The 8051 has two priority levels: **Low** (default) and **High**.

**Priority Rules:**
1. High-priority interrupts can interrupt low-priority ISRs
2. Same-priority interrupts cannot interrupt each other
3. Simultaneous interrupts serviced in fixed order (INT0 → T0 → INT1 → T1 → Serial)

**Setting Priority:**
```c
PT0 = 1;  // Timer 0 high priority
PX0 = 1;  // INT0 high priority
```

---

## Common Issues

### Interrupt Not Triggering
- Check individual interrupt enable (EX0, ET0, etc.)
- Verify global interrupt enable (EA = 1)
- Confirm interrupt flag is being set
- Check trigger type (edge vs level)

### System Crash/Hang
- ISR too complex or long
- Stack overflow (check SP initialization)
- Forgetting to clear interrupt flag
- Interrupting critical sections

### Unwanted Repeated Interrupts
- Level-triggered mode with low signal held
- Not clearing interrupt flag properly
- Hardware noise/glitches

### Interrupt Timing Issues
- High-priority interrupt blocking low-priority
- ISR taking too long
- Nested interrupts causing stack issues

---

## Interrupt vs Polling Comparison

| Aspect | Interrupt | Polling |
|--------|-----------|---------|
| CPU Usage | Efficient | Wasteful |
| Response Time | Immediate | Delayed |
| Implementation | Complex | Simple |
| Reliability | High | Low |
| Multi-tasking | Excellent | Poor |

---

## Prerequisites

- Basic I/O operations
- Understanding of stack and memory
- Timer basics

**Recommended Reading:**
- [Interrupt System](../02_Hardware_Architecture/README.md#interrupt-system)
- [Timer Examples](../02_Timers/)

---

## Debugging Tips

1. **LED Indicator** - Toggle LED in ISR to verify it's executing
2. **Counter** - Increment counter in ISR, check in main loop
3. **Scope** - Use oscilloscope to monitor interrupt pins
4. **Single Step** - Use debugger to step through ISR
5. **Simplify** - Start with one interrupt, add more gradually

---

## Real-World Applications

- **Embedded Systems**: Button presses, sensor triggers
- **Communications**: Serial data reception
- **Motor Control**: Encoder feedback, limit switches
- **Instrumentation**: Alarm conditions, threshold crossing
- **Automotive**: Crash sensors, timeout detection

---

## Next Steps

After mastering interrupts:
- [Serial Port Examples](../04_Serial_Port/)
- [Advanced Multi-Interrupt Systems](../05_Advanced/)
