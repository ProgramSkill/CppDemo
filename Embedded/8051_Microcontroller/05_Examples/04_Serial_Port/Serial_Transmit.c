/**
 * Serial_Transmit.c
 * 8051 Serial Port Transmission with Formatting
 *
 * Description: Demonstrates formatted string transmission including
 *              numbers in different formats (decimal, hex, binary)
 * Hardware: 8051 @ 11.0592MHz, MAX232 level shifter
 * Baud Rate: 9600 (8N1)
 */

#include <reg51.h>
#include <stdio.h>

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

    /* Clear interrupt flags */
    TI = 0;
    RI = 0;
}

/**
 * Transmit a single byte via serial port
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
 * Transmit a byte as hexadecimal (2 digits)
 */
void serial_send_hex(unsigned char value)
{
    unsigned char high_nibble = (value >> 4) & 0x0F;
    unsigned char low_nibble = value & 0x0F;

    /* Convert to ASCII */
    serial_send(high_nibble > 9 ? high_nibble + 55 : high_nibble + 48);
    serial_send(low_nibble > 9 ? low_nibble + 55 : low_nibble + 48);
}

/**
 * Transmit a 16-bit word as hexadecimal (4 digits)
 */
void serial_send_hex_word(unsigned int value)
{
    serial_send_hex((unsigned char)(value >> 8));
    serial_send_hex((unsigned char)value);
}

/**
 * Transmit a byte as binary (8 digits)
 */
void serial_send_binary(unsigned char value)
{
    unsigned char i;
    for (i = 0x80; i != 0; i >>= 1)
    {
        serial_send((value & i) ? '1' : '0');
    }
}

/**
 * Transmit an unsigned byte as decimal (0-255)
 */
void serial_send_decimal_byte(unsigned char value)
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
 * Transmit a signed byte as decimal (-128 to 127)
 */
void serial_send_decimal_signed(char value)
{
    if (value < 0)
    {
        serial_send('-');
        value = -value;
    }
    serial_send_decimal_byte((unsigned char)value);
}

/**
 * Transmit an unsigned 16-bit integer as decimal (0-65535)
 */
void serial_send_decimal_word(unsigned int value)
{
    unsigned char digit;
    unsigned char started = 0;
    unsigned int divisor;

    for (divisor = 10000; divisor >= 1; divisor /= 10)
    {
        digit = value / divisor;
        if (digit != 0 || started || divisor == 1)
        {
            serial_send(digit + 48);
            started = 1;
            value -= digit * divisor;
        }
    }
}

/**
 * Transmit newline sequence
 */
void serial_send_newline(void)
{
    serial_send('\r');
    serial_send('\n');
}

/**
 * Main program - demonstrate various formatting options
 */
void main(void)
{
    unsigned char test_byte = 0xAB;
    unsigned int test_word = 0x1234;
    unsigned char i;

    /* Initialize serial port */
    serial_init();

    /* Send program title */
    serial_send_string("========================================\r\n");
    serial_send_string("  8051 Serial Transmit Demo\r\n");
    serial_send_string("========================================\r\n\r\n");

    /* 1. String transmission */
    serial_send_string("1. String Transmission:\r\n");
    serial_send_string("   Hello, World!\r\n\r\n");

    /* 2. Decimal byte transmission */
    serial_send_string("2. Decimal Byte (0-255):\r\n");
    for (i = 0; i <= 10; i++)
    {
        serial_send_string("   Number ");
        serial_send_decimal_byte(i);
        serial_send_string("\r\n");
    }
    serial_send_newline();

    /* 3. Signed decimal transmission */
    serial_send_string("3. Signed Decimal (-128 to 127):\r\n");
    serial_send_string("   Positive: ");
    serial_send_decimal_signed(100);
    serial_send_newline();
    serial_send_string("   Negative: ");
    serial_send_decimal_signed(-50);
    serial_send_newline();
    serial_send_string("   Zero: ");
    serial_send_decimal_signed(0);
    serial_send_newline();
    serial_send_newline();

    /* 4. Decimal word transmission */
    serial_send_string("4. Decimal Word (0-65535):\r\n");
    serial_send_string("   Small: ");
    serial_send_decimal_word(123);
    serial_send_newline();
    serial_send_string("   Large: ");
    serial_send_decimal_word(54321);
    serial_send_newline();
    serial_send_string("   Max: ");
    serial_send_decimal_word(65535);
    serial_send_newline();
    serial_send_newline();

    /* 5. Hexadecimal transmission */
    serial_send_string("5. Hexadecimal:\r\n");
    serial_send_string("   Byte: 0x");
    serial_send_hex(test_byte);
    serial_send_newline();
    serial_send_string("   Word: 0x");
    serial_send_hex_word(test_word);
    serial_send_newline();
    serial_send_newline();

    /* 6. Binary transmission */
    serial_send_string("6. Binary:\r\n");
    serial_send_string("   Pattern: ");
    serial_send_binary(0b10101010);
    serial_send_newline();
    serial_send_string("   Counter: ");
    serial_send_binary(0b00001111);
    serial_send_newline();
    serial_send_newline();

    /* 7. Mixed formatting example */
    serial_send_string("7. Mixed Formatting:\r\n");
    serial_send_string("   Value ");
    serial_send_decimal_byte(255);
    serial_send_string(" = 0x");
    serial_send_hex(255);
    serial_send_string(" = ");
    serial_send_binary(255);
    serial_send_string("B\r\n\r\n");

    /* 8. Data table */
    serial_send_string("8. Data Table:\r\n");
    serial_send_string("   Dec  Hex    Bin\r\n");
    serial_send_string("   ---  ----   --------\r\n");
    for (i = 0; i < 8; i++)
    {
        serial_send_string("   ");
        serial_send_decimal_byte(i);
        serial_send_string("   0x");
        serial_send_hex(i);
        serial_send_string("   ");
        serial_send_binary(i);
        serial_send_newline();
    }
    serial_send_newline();

    serial_send_string("Demo Complete!\r\n");
    serial_send_string("========================================\r\n");

    /* End - just loop */
    while (1);
}
