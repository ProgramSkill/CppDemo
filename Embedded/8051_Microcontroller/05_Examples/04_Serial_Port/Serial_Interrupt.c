/**
 * Serial_Interrupt.c
 * 8051 Interrupt-Driven Serial I/O
 *
 * Description: Full-duplex serial communication using interrupts
 *              Features include bidirectional buffers, non-blocking I/O
 * Hardware: 8051 @ 11.0592MHz, MAX232 level shifter
 * Baud Rate: 9600 (8N1)
 */

#include <reg51.h>

/* Baud rate reload value for 9600 baud @ 11.0592MHz */
#define BAUD_9600 0xFD

/* Buffer sizes */
#define TX_BUFFER_SIZE 64
#define RX_BUFFER_SIZE 64

/* Transmit buffer (circular) */
unsigned char tx_buffer[TX_BUFFER_SIZE];
volatile unsigned char tx_head = 0;
volatile unsigned char tx_tail = 0;

/* Receive buffer (circular) */
unsigned char rx_buffer[RX_BUFFER_SIZE];
volatile unsigned char rx_head = 0;
volatile unsigned char rx_tail = 0;

/* Status flags */
volatile bit tx_active = 0;
volatile bit rx_overflow = 0;
volatile bit tx_overflow = 0;

/**
 * Initialize serial port with interrupt support
 */
void serial_init(void)
{
    /* Configure Timer 1 as baud rate generator */
    TMOD &= 0x0F;
    TMOD |= 0x20;        /* Timer 1, Mode 2 (8-bit auto-reload) */

    TH1 = BAUD_9600;
    TL1 = BAUD_9600;
    TR1 = 1;             /* Start Timer 1 */

    /* Configure Serial Control Register */
    SCON = 0x50;         /* Mode 1, Receive enabled */

    /* Enable interrupts */
    ES = 1;              /* Enable Serial Interrupt */
    EA = 1;              /* Enable Global Interrupt */

    TI = 0;
    RI = 0;
}

/**
 * Write byte to transmit buffer (non-blocking)
 * Returns 1 on success, 0 if buffer full
 */
bit serial_write(unsigned char data)
{
    unsigned char next_head;

    /* Calculate next head position */
    next_head = (tx_head + 1) % TX_BUFFER_SIZE;

    /* Check if buffer is full */
    if (next_head == tx_tail)
    {
        tx_overflow = 1;
        return 0;  /* Buffer full */
    }

    /* Add data to buffer */
    tx_buffer[tx_head] = data;
    tx_head = next_head;

    /* Start transmission if not active */
    if (!tx_active)
    {
        tx_active = 1;
        TI = 1;  /* Trigger transmit interrupt */
    }

    return 1;  /* Success */
}

/**
 * Write string to transmit buffer (non-blocking)
 * Returns number of characters written
 */
unsigned char serial_write_string(unsigned char *str)
{
    unsigned char count = 0;

    while (*str != '\0')
    {
        if (!serial_write(*str))
            break;  /* Buffer full */
        str++;
        count++;
    }

    return count;
}

/**
 * Check if data is available in receive buffer
 */
bit serial_available(void)
{
    return (rx_head != rx_tail);
}

/**
 * Read byte from receive buffer (non-blocking)
 * Returns byte if available, 0xFF if empty
 */
unsigned char serial_read_nb(void)
{
    unsigned char data;

    if (!serial_available())
        return 0xFF;  /* Buffer empty */

    data = rx_buffer[rx_tail];
    rx_tail = (rx_tail + 1) % RX_BUFFER_SIZE;

    return data;
}

/**
 * Read byte from receive buffer (blocking)
 */
unsigned char serial_read(void)
{
    while (!serial_available());

    unsigned char data = rx_buffer[rx_tail];
    rx_tail = (rx_tail + 1) % RX_BUFFER_SIZE;

    return data;
}

/**
 * Get number of bytes in receive buffer
 */
unsigned char serial_rx_count(void)
{
    if (rx_head >= rx_tail)
        return rx_head - rx_tail;
    else
        return RX_BUFFER_SIZE - rx_tail + rx_head;
}

/**
 * Get number of bytes in transmit buffer
 */
unsigned char serial_tx_count(void)
{
    if (tx_head >= tx_tail)
        return tx_head - tx_tail;
    else
        return TX_BUFFER_SIZE - tx_tail + tx_tail;
}

/**
 * Get available space in transmit buffer
 */
unsigned char serial_tx_space(void)
{
    return TX_BUFFER_SIZE - serial_tx_count() - 1;
}

/**
 * Clear receive buffer
 */
void serial_clear_rx(void)
{
    rx_head = 0;
    rx_tail = 0;
}

/**
 * Clear transmit buffer
 */
void serial_clear_tx(void)
{
    tx_head = 0;
    tx_tail = 0;
}

/**
 * Wait for transmit buffer to empty
 */
void serial_flush(void)
{
    while (tx_active || (tx_head != tx_tail));
}

/**
 * Serial interrupt service routine
 * Handles both transmit and receive interrupts
 */
void serial_isr(void) interrupt 4
{
    /* Receive interrupt */
    if (RI)
    {
        RI = 0;  /* Clear receive interrupt */

        /* Calculate next head position */
        unsigned char next_head = (rx_head + 1) % RX_BUFFER_SIZE;

        /* Check for buffer overflow */
        if (next_head == rx_tail)
        {
            rx_overflow = 1;  /* Buffer full, data lost */
        }
        else
        {
            /* Store received data */
            rx_buffer[rx_head] = SBUF;
            rx_head = next_head;
        }
    }

    /* Transmit interrupt */
    if (TI)
    {
        TI = 0;  /* Clear transmit interrupt */

        /* Check if data to transmit */
        if (tx_head != tx_tail)
        {
            /* Send next byte */
            SBUF = tx_buffer[tx_tail];
            tx_tail = (tx_tail + 1) % TX_BUFFER_SIZE;
            tx_active = 1;
        }
        else
        {
            /* No more data */
            tx_active = 0;
        }
    }
}

/**
 * Display system status
 */
void display_status(void)
{
    serial_write_string("\r\n--- Serial Status ---\r\n");

    /* Receive buffer status */
    serial_write_string("RX Buffer: ");
    serial_write_decimal(serial_rx_count());
    serial_write_string("/");
    serial_write_decimal(RX_BUFFER_SIZE);
    serial_write_string(" bytes\r\n");

    /* Transmit buffer status */
    serial_write_string("TX Buffer: ");
    serial_write_decimal(serial_tx_count());
    serial_write_string("/");
    serial_write_decimal(TX_BUFFER_SIZE);
    serial_write_string(" bytes\r\n");

    /* Overflow flags */
    serial_write_string("RX Overflow: ");
    serial_write_string(rx_overflow ? "Yes" : "No");
    serial_write_string("\r\n");

    serial_write_string("TX Overflow: ");
    serial_write_string(tx_overflow ? "Yes" : "No");
    serial_write_string("\r\n");

    serial_write_string("TX Active: ");
    serial_write_string(tx_active ? "Yes" : "No");
    serial_write_string("\r\n");

    serial_write_string("--------------------\r\n\r\n");
}

/**
 * Display welcome message
 */
void display_welcome(void)
{
    serial_write_string("\r\n");
    serial_write_string("========================================\r\n");
    serial_write_string("  8051 Interrupt-Driven Serial Demo\r\n");
    serial_write_string("========================================\r\n");
    serial_write_string("Features:\r\n");
    serial_write_string("  - Full-duplex communication\r\n");
    serial_write_string("  - Non-blocking I/O\r\n");
    serial_write_string("  - Circular buffers\r\n");
    serial_write_string("\r\n");
    serial_write_string("Commands:\r\n");
    serial_write_string("  STATUS    - Show buffer status\r\n");
    serial_write_string("  ECHO      - Test transmission\r\n");
    serial_write_string("  CLEAR     - Clear buffers\r\n");
    serial_write_string("  FLUSH     - Wait for TX complete\r\n");
    serial_write_string("========================================\r\n\r\n");
}

/**
 * Process received command
 */
void process_command(unsigned char *cmd)
{
    /* Check for STATUS command */
    if (cmd[0] == 'S' && cmd[1] == 'T' &&
        cmd[2] == 'A' && cmd[3] == 'T' && cmd[4] == 'U' &&
        cmd[5] == 'S')
    {
        display_status();
        return;
    }

    /* Check for ECHO command */
    if (cmd[0] == 'E' && cmd[1] == 'C' &&
        cmd[2] == 'H' && cmd[3] == 'O')
    {
        serial_write_string("Echo test: ABCDEFGHIJKLMNOPQRSTUVWXYZ\r\n");
        return;
    }

    /* Check for CLEAR command */
    if (cmd[0] == 'C' && cmd[1] == 'L' &&
        cmd[2] == 'E' && cmd[3] == 'A' && cmd[4] == 'R')
    {
        serial_clear_rx();
        serial_clear_tx();
        rx_overflow = 0;
        tx_overflow = 0;
        serial_write_string("Buffers cleared\r\n");
        return;
    }

    /* Check for FLUSH command */
    if (cmd[0] == 'F' && cmd[1] == 'L' &&
        cmd[2] == 'U' && cmd[3] == 'S' && cmd[4] == 'H')
    {
        serial_write_string("Flushing TX buffer...\r\n");
        serial_flush();
        serial_write_string("Flush complete\r\n");
        return;
    }

    /* Unknown command */
    serial_write_string("Unknown command: ");
    serial_write_string(cmd);
    serial_write_string("\r\n");
}

/**
 * Main program
 */
void main(void)
{
    unsigned char data;
    unsigned char cmd_buffer[32];
    unsigned char cmd_index = 0;

    /* Initialize serial port */
    serial_init();

    /* Display welcome message */
    display_welcome();

    /* Send startup prompt */
    serial_write_string("Ready> ");

    /* Main loop */
    while (1)
    {
        /* Check for received data */
        if (serial_available())
        {
            data = serial_read();

            /* Echo character */
            serial_write(data);

            /* Check for line terminator */
            if (data == '\r' || data == '\n')
            {
                serial_write_string("\r\n");

                /* Process command if not empty */
                if (cmd_index > 0)
                {
                    cmd_buffer[cmd_index] = '\0';
                    process_command(cmd_buffer);
                    cmd_index = 0;
                }

                serial_write_string("Ready> ");
            }
            /* Handle backspace */
            else if (data == 8 || data == 127)
            {
                if (cmd_index > 0)
                {
                    cmd_index--;
                    serial_write_string("\b \b");
                }
            }
            /* Store regular character */
            else if (data >= 32 && data < 127)
            {
                if (cmd_index < 31)
                {
                    cmd_buffer[cmd_index++] = data;
                }
            }
        }

        /* Check for overflow errors */
        if (rx_overflow)
        {
            serial_write_string("\r\n! RX Buffer Overflow !\r\n");
            rx_overflow = 0;
        }

        if (tx_overflow)
        {
            serial_write_string("\r\n! TX Buffer Overflow !\r\n");
            tx_overflow = 0;
        }

        /* Other application code can go here */
        /* Serial I/O is non-blocking! */
    }
}
