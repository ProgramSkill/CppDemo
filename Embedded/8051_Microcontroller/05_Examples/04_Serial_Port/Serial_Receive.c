/**
 * Serial_Receive.c
 * 8051 Serial Port Buffered Receive with Processing
 *
 * Description: Demonstrates buffered serial data reception and processing
 *              Features include line buffering, command parsing, and echo
 * Hardware: 8051 @ 11.0592MHz, MAX232 level shifter
 * Baud Rate: 9600 (8N1)
 */

#include <reg51.h>

/* Baud rate reload value for 9600 baud @ 11.0592MHz */
#define BAUD_9600 0xFD

/* Buffer configuration */
#define RX_BUFFER_SIZE 64
#define LINE_BUFFER_SIZE 32

/* Global receive buffer (circular buffer) */
unsigned char rx_buffer[RX_BUFFER_SIZE];
unsigned char rx_head = 0;
unsigned char rx_tail = 0;

/* Line buffer for command processing */
unsigned char line_buffer[LINE_BUFFER_SIZE];
unsigned char line_index = 0;

/* Status flags */
bit line_ready = 0;
bit buffer_overflow = 0;

/**
 * Initialize serial port for 9600 baud, 8N1
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

    /* Enable serial interrupt */
    ES = 1;              /* Enable Serial Interrupt */
    EA = 1;              /* Enable Global Interrupt */

    TI = 0;
    RI = 0;
}

/**
 * Transmit a single byte
 */
void serial_send(unsigned char data)
{
    SBUF = data;
    while (!TI);
    TI = 0;
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
 * Transmit a byte as decimal (0-255)
 */
void serial_send_decimal(unsigned char value)
{
    unsigned char hundreds = 0;
    unsigned char tens = 0;
    unsigned char ones = 0;

    while (value >= 100)
    {
        hundreds++;
        value -= 100;
    }
    while (value >= 10)
    {
        tens++;
        value -= 10;
    }
    ones = value;

    if (hundreds > 0)
        serial_send(hundreds + 48);
    if (hundreds > 0 || tens > 0)
        serial_send(tens + 48);
    serial_send(ones + 48);
}

/**
 * Transmit newline
 */
void serial_send_newline(void)
{
    serial_send('\r');
    serial_send('\n');
}

/**
 * Check if data is available in receive buffer
 */
bit serial_available(void)
{
    return (rx_head != rx_tail);
}

/**
 * Read one byte from receive buffer
 * Returns byte if available, waits if empty
 */
unsigned char serial_read(void)
{
    while (!serial_available());

    unsigned char data = rx_buffer[rx_tail];
    rx_tail = (rx_tail + 1) % RX_BUFFER_SIZE;
    return data;
}

/**
 * Read one byte from receive buffer (non-blocking)
 * Returns byte if available, 0xFF if empty
 */
unsigned char serial_read_nb(void)
{
    if (!serial_available())
        return 0xFF;

    unsigned char data = rx_buffer[rx_tail];
    rx_tail = (rx_tail + 1) % RX_BUFFER_SIZE;
    return data;
}

/**
 * Get number of bytes in receive buffer
 */
unsigned char serial_available_count(void)
{
    if (rx_head >= rx_tail)
        return rx_head - rx_tail;
    else
        return RX_BUFFER_SIZE - rx_tail + rx_head;
}

/**
 * Clear receive buffer
 */
void serial_clear_buffer(void)
{
    rx_head = 0;
    rx_tail = 0;
}

/**
 * Process received line
 */
void process_line(unsigned char *line, unsigned char length)
{
    /* Check for empty line */
    if (length == 0)
    {
        serial_send_string("Empty line received\r\n");
        return;
    }

    /* Echo the line */
    serial_send_string("Received: ");
    serial_send_string(line);
    serial_send_newline();

    /* Display line info */
    serial_send_string("Length: ");
    serial_send_decimal(length);
    serial_send_string(" bytes\r\n");

    /* Check for specific commands */
    if (length >= 4)
    {
        /* Check for "HELLO" */
        if (line[0] == 'H' && line[1] == 'E' &&
            line[2] == 'L' && line[3] == 'L' && line[4] == 'O')
        {
            serial_send_string("Hello to you too!\r\n");
            return;
        }

        /* Check for "HELP" */
        if (line[0] == 'H' && line[1] == 'E' &&
            line[2] == 'L' && line[3] == 'P')
        {
            serial_send_string("Commands:\r\n");
            serial_send_string("  HELP    - Show this help\r\n");
            serial_send_string("  HELLO   - Greeting\r\n");
            serial_send_string("  STATUS  - Show buffer status\r\n");
            serial_send_string("  CLEAR   - Clear buffers\r\n");
            return;
        }

        /* Check for "STATUS" */
        if (line[0] == 'S' && line[1] == 'T' &&
            line[2] == 'A' && line[3] == 'T' && line[4] == 'U' &&
            line[5] == 'S')
        {
            serial_send_string("Buffer Status:\r\n");
            serial_send_string("  Available: ");
            serial_send_decimal(serial_available_count());
            serial_send_string(" bytes\r\n");
            serial_send_string("  Line Ready: ");
            serial_send_string(line_ready ? "Yes" : "No");
            serial_send_newline();
            return;
        }

        /* Check for "CLEAR" */
        if (line[0] == 'C' && line[1] == 'L' &&
            line[2] == 'E' && line[3] == 'A' && line[4] == 'R')
        {
            serial_clear_buffer();
            line_index = 0;
            line_ready = 0;
            serial_send_string("Buffers cleared\r\n");
            return;
        }
    }

    /* Unknown command */
    serial_send_string("Unknown command. Type HELP for usage\r\n");
}

/**
 * Process receive buffer and build lines
 */
void process_receive_buffer(void)
{
    unsigned char data;

    /* Process all available data */
    while (serial_available())
    {
        data = serial_read();

        /* Check for line terminator */
        if (data == '\r' || data == '\n')
        {
            /* End of line */
            if (line_index > 0)
            {
                line_buffer[line_index] = '\0';
                line_ready = 1;
                process_line(line_buffer, line_index);
                line_index = 0;
                line_ready = 0;
            }
        }
        /* Check for backspace */
        else if (data == 8 || data == 127)
        {
            if (line_index > 0)
            {
                line_index--;
                serial_send_string("\b \b");  /* Echo backspace */
            }
        }
        /* Regular character */
        else if (data >= 32 && data < 127)
        {
            /* Check buffer space */
            if (line_index < LINE_BUFFER_SIZE - 1)
            {
                line_buffer[line_index] = data;
                line_index++;
                serial_send(data);  /* Echo character */
            }
            else
            {
                /* Line buffer overflow */
                serial_send_string("\r\nLine too long!\r\n");
                line_index = 0;
            }
        }
    }
}

/**
 * Serial interrupt service routine
 * Receives data into circular buffer
 */
void serial_isr(void) interrupt 4
{
    if (RI)
    {
        RI = 0;  /* Clear receive interrupt */

        /* Calculate next head position */
        unsigned char next_head = (rx_head + 1) % RX_BUFFER_SIZE;

        /* Check for buffer overflow */
        if (next_head == rx_tail)
        {
            buffer_overflow = 1;
        }
        else
        {
            rx_buffer[rx_head] = SBUF;
            rx_head = next_head;
        }
    }

    /* Handle transmit interrupt if needed */
    if (TI)
    {
        TI = 0;  /* Clear transmit interrupt */
    }
}

/**
 * Display welcome banner
 */
void display_welcome(void)
{
    serial_send_string("\r\n");
    serial_send_string("========================================\r\n");
    serial_send_string("  8051 Serial Receive Demo\r\n");
    serial_send_string("========================================\r\n");
    serial_send_string("Type commands and press Enter\r\n");
    serial_send_string("Type HELP for available commands\r\n");
    serial_send_string("========================================\r\n\r\n");
    serial_send_string("> ");
}

/**
 * Main program
 */
void main(void)
{
    /* Initialize serial port */
    serial_init();

    /* Display welcome message */
    display_welcome();

    /* Main loop */
    while (1)
    {
        /* Process received data */
        process_receive_buffer();

        /* Check for buffer overflow */
        if (buffer_overflow)
        {
            serial_send_string("\r\nBuffer overflow! Data lost.\r\n> ");
            buffer_overflow = 0;
        }
    }
}
