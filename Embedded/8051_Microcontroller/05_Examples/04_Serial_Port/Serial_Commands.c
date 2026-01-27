/**
 * Serial_Commands.c
 * 8051 Serial Command Parser and Processor
 *
 * Description: Command-line interface with argument parsing
 *              Features include help system, LED control, I/O monitoring
 * Hardware: 8051 @ 11.0592MHz, MAX232 level shifter
 * Baud Rate: 9600 (8N1)
 */

#include <reg51.h>

/* Baud rate reload value for 9600 baud @ 11.0592MHz */
#define BAUD_9600 0xFD

/* Buffer sizes */
#define CMD_BUFFER_SIZE 64
#define ARG_BUFFER_SIZE 32

/* Command buffer */
unsigned char cmd_buffer[CMD_BUFFER_SIZE];
unsigned char cmd_index = 0;
bit cmd_ready = 0;

/* Argument storage */
unsigned char arg1[ARG_BUFFER_SIZE];
unsigned char arg2[ARG_BUFFER_SIZE];
unsigned char arg_count = 0;

/* Command return codes */
#define CMD_OK           0
#define CMD_ERROR        1
#define CMD_NOT_FOUND    2
#define CMD_BAD_ARGS     3

/* P1 connected to LEDs (assumed active low) */
#define LED_PORT P1

/**
 * Initialize serial port
 */
void serial_init(void)
{
    TMOD &= 0x0F;
    TMOD |= 0x20;
    TH1 = BAUD_9600;
    TL1 = BAUD_9600;
    TR1 = 1;
    SCON = 0x50;

    TI = 0;
    RI = 0;
}

/**
 * Send a byte
 */
void serial_send(unsigned char data)
{
    SBUF = data;
    while (!TI);
    TI = 0;
}

/**
 * Send a string
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
 * Send a decimal byte
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
 * Send a hex byte
 */
void serial_send_hex(unsigned char value)
{
    unsigned char high_nibble = (value >> 4) & 0x0F;
    unsigned char low_nibble = value & 0x0F;

    serial_send(high_nibble > 9 ? high_nibble + 55 : high_nibble + 48);
    serial_send(low_nibble > 9 ? low_nibble + 55 : low_nibble + 48);
}

/**
 * Send a binary byte
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
 * Send newline
 */
void serial_send_newline(void)
{
    serial_send('\r');
    serial_send('\n');
}

/**
 * Convert ASCII to uppercase
 */
unsigned char to_upper(unsigned char c)
{
    if (c >= 'a' && c <= 'z')
        return c - 32;
    return c;
}

/**
 * Compare strings (case-insensitive)
 */
unsigned char strcmp_i(unsigned char *s1, unsigned char *s2)
{
    while (*s1 != '\0' && *s2 != '\0')
    {
        if (to_upper(*s1) != to_upper(*s2))
            return 1;
        s1++;
        s2++;
    }
    return (*s1 == '\0' && *s2 == '\0') ? 0 : 1;
}

/**
 * Parse command line into command and arguments
 */
void parse_command(unsigned char *cmd_line)
{
    unsigned char i = 0;
    unsigned char arg_index = 0;
    bit in_arg = 0;
    unsigned char current_arg = 0;

    arg_count = 0;

    /* Skip leading spaces */
    while (cmd_line[i] == ' ')
        i++;

    /* Parse command line */
    while (cmd_line[i] != '\0' && cmd_line[i] != '\r' && cmd_line[i] != '\n')
    {
        if (cmd_line[i] == ' ' || cmd_line[i] == ',')
        {
            /* End of argument */
            if (in_arg)
            {
                if (current_arg == 0)
                {
                    /* This is the command itself */
                    /* Already stored in cmd_buffer */
                }
                else if (current_arg == 1)
                {
                    arg1[arg_index] = '\0';
                }
                else if (current_arg == 2)
                {
                    arg2[arg_index] = '\0';
                }

                arg_index = 0;
                in_arg = 0;
            }
        }
        else
        {
            /* Character part of argument */
            if (!in_arg)
            {
                in_arg = 1;
                current_arg++;

                if (current_arg > 2)
                {
                    /* Too many arguments */
                    break;
                }
            }

            if (current_arg == 1 && arg_index < ARG_BUFFER_SIZE - 1)
            {
                arg1[arg_index++] = cmd_line[i];
            }
            else if (current_arg == 2 && arg_index < ARG_BUFFER_SIZE - 1)
            {
                arg2[arg_index++] = cmd_line[i];
            }
        }
        i++;
    }

    /* Null-terminate last argument */
    if (in_arg)
    {
        if (current_arg == 1)
        {
            arg1[arg_index] = '\0';
        }
        else if (current_arg == 2)
        {
            arg2[arg_index] = '\0';
        }
    }

    arg_count = current_arg;
}

/**
 * Convert ASCII string to integer
 */
unsigned char ascii_to_byte(unsigned char *str)
{
    unsigned char value = 0;

    /* Handle decimal */
    while (*str >= '0' && *str <= '9')
    {
        value = value * 10 + (*str - '0');
        str++;
    }

    return value;
}

/**
 * Command: HELP
 * Display available commands
 */
unsigned char cmd_help(void)
{
    serial_send_string("\r\n");
    serial_send_string("Available Commands:\r\n");
    serial_send_string("====================\r\n");
    serial_send_string("  HELP              - Show this help\r\n");
    serial_send_string("  LED <num> <state> - Control LED (0-7, ON/OFF)\r\n");
    serial_send_string("  LEDS <value>      - Set all LEDs (0-255)\r\n");
    serial_send_string("  PATTERN <type>    - LED pattern (CHASE/BLINK/ALL)\r\n");
    serial_send_string("  STATUS            - Show system status\r\n");
    serial_send_string("  READ <port>       - Read port (0-3)\r\n");
    serial_send_string("  WRITE <port> <val> - Write port (0-3, 0-255)\r\n");
    serial_send_string("  CLEAR             - Clear screen\r\n");
    serial_send_string("  ECHO <text>       - Echo text back\r\n");
    serial_send_string("  HEX <value>       - Convert to hex/bin\r\n");
    serial_send_string("\r\n");
    serial_send_string("Examples:\r\n");
    serial_send_string("  LED 0 ON\r\n");
    serial_send_string("  LEDS 0xAA\r\n");
    serial_send_string("  PATTERN CHASE\r\n");
    serial_send_string("  STATUS\r\n");
    serial_send_string("\r\n");

    return CMD_OK;
}

/**
 * Command: LED
 * Control individual LED
 */
unsigned char cmd_led(void)
{
    unsigned char led_num;
    bit led_state;

    /* Check arguments */
    if (arg_count < 3)
    {
        serial_send_string("Usage: LED <num> <state>\r\n");
        return CMD_BAD_ARGS;
    }

    /* Parse LED number */
    led_num = ascii_to_byte(arg1);
    if (led_num > 7)
    {
        serial_send_string("Error: LED number must be 0-7\r\n");
        return CMD_BAD_ARGS;
    }

    /* Parse state */
    if (strcmp_i(arg2, (unsigned char*)"ON") == 0)
    {
        led_state = 0;  /* Active low */
        serial_send_string("LED ");
        serial_send_decimal(led_num);
        serial_send_string(" ON\r\n");
    }
    else if (strcmp_i(arg2, (unsigned char*)"OFF") == 0)
    {
        led_state = 1;  /* Active low */
        serial_send_string("LED ");
        serial_send_decimal(led_num);
        serial_send_string(" OFF\r\n");
    }
    else
    {
        serial_send_string("Error: State must be ON or OFF\r\n");
        return CMD_BAD_ARGS;
    }

    /* Set LED */
    if (led_state)
        LED_PORT |= (1 << led_num);
    else
        LED_PORT &= ~(1 << led_num);

    return CMD_OK;
}

/**
 * Command: LEDS
 * Set all LEDs to value
 */
unsigned char cmd_leds(void)
{
    unsigned char value;

    if (arg_count < 2)
    {
        serial_send_string("Usage: LEDS <value>\r\n");
        return CMD_BAD_ARGS;
    }

    value = ascii_to_byte(arg1);

    LED_PORT = ~value;  /* Active low */

    serial_send_string("LEDs set to 0x");
    serial_send_hex(value);
    serial_send_string(" (");
    serial_send_decimal(value);
    serial_send_string(")\r\n");

    return CMD_OK;
}

/**
 * Command: PATTERN
 * Display LED pattern
 */
unsigned char cmd_pattern(void)
{
    if (arg_count < 2)
    {
        serial_send_string("Usage: PATTERN <CHASE|BLINK|ALL>\r\n");
        return CMD_BAD_ARGS;
    }

    if (strcmp_i(arg1, (unsigned char*)"ALL") == 0)
    {
        LED_PORT = 0x00;  /* All on */
        serial_send_string("All LEDs ON\r\n");
    }
    else if (strcmp_i(arg1, (unsigned char*)"CHASE") == 0)
    {
        unsigned char i;
        for (i = 0; i < 8; i++)
        {
            LED_PORT = ~(1 << i);
        }
        LED_PORT = 0xFF;  /* All off */
        serial_send_string("Chase pattern complete\r\n");
    }
    else if (strcmp_i(arg1, (unsigned char*)"BLINK") == 0)
    {
        unsigned char i;
        for (i = 0; i < 5; i++)
        {
            LED_PORT = 0x00;  /* On */
            LED_PORT = 0xFF;  /* Off */
        }
        serial_send_string("Blink pattern complete\r\n");
    }
    else
    {
        serial_send_string("Error: Unknown pattern\r\n");
        return CMD_BAD_ARGS;
    }

    return CMD_OK;
}

/**
 * Command: STATUS
 * Display system status
 */
unsigned char cmd_status(void)
{
    serial_send_string("\r\n=== System Status ===\r\n");

    serial_send_string("Port 1 (LEDs): 0x");
    serial_send_hex(LED_PORT);
    serial_send_string("\r\n");

    serial_send_string("Port 0: 0x");
    serial_send_hex(P0);
    serial_send_string("\r\n");

    serial_send_string("Port 2: 0x");
    serial_send_hex(P2);
    serial_send_string("\r\n");

    serial_send_string("Port 3: 0x");
    serial_send_hex(P3);
    serial_send_string("\r\n");

    serial_send_string("====================\r\n\r\n");

    return CMD_OK;
}

/**
 * Command: READ
 * Read a port
 */
unsigned char cmd_read(void)
{
    unsigned char port_num;
    unsigned char value;

    if (arg_count < 2)
    {
        serial_send_string("Usage: READ <port>\r\n");
        return CMD_BAD_ARGS;
    }

    port_num = ascii_to_byte(arg1);

    if (port_num > 3)
    {
        serial_send_string("Error: Port must be 0-3\r\n");
        return CMD_BAD_ARGS;
    }

    switch (port_num)
    {
        case 0: value = P0; break;
        case 1: value = P1; break;
        case 2: value = P2; break;
        case 3: value = P3; break;
        default: value = 0; break;
    }

    serial_send_string("Port ");
    serial_send_decimal(port_num);
    serial_send_string(" = 0x");
    serial_send_hex(value);
    serial_send_string(" (");
    serial_send_decimal(value);
    serial_send_string(")\r\n");

    return CMD_OK;
}

/**
 * Command: WRITE
 * Write to a port
 */
unsigned char cmd_write(void)
{
    unsigned char port_num;
    unsigned char value;

    if (arg_count < 3)
    {
        serial_send_string("Usage: WRITE <port> <value>\r\n");
        return CMD_BAD_ARGS;
    }

    port_num = ascii_to_byte(arg1);
    value = ascii_to_byte(arg2);

    if (port_num > 3)
    {
        serial_send_string("Error: Port must be 0-3\r\n");
        return CMD_BAD_ARGS;
    }

    switch (port_num)
    {
        case 0: P0 = value; break;
        case 1:
            /* Skip P1 (LEDs) - read-only */
            serial_send_string("Error: Port 1 is read-only (LEDs)\r\n");
            return CMD_ERROR;
        case 2: P2 = value; break;
        case 3: P3 = value; break;
    }

    serial_send_string("Port ");
    serial_send_decimal(port_num);
    serial_send_string(" = 0x");
    serial_send_hex(value);
    serial_send_string("\r\n");

    return CMD_OK;
}

/**
 * Command: CLEAR
 * Clear screen
 */
unsigned char cmd_clear(void)
{
    serial_send_string("\033[2J\033[H");  /* ANSI escape codes */
    serial_send_string("Screen cleared\r\n");
    return CMD_OK;
}

/**
 * Command: ECHO
 * Echo text back
 */
unsigned char cmd_echo(void)
{
    serial_send_string(arg1);
    if (arg_count >= 2)
    {
        serial_send_string(" ");
        serial_send_string(arg2);
    }
    serial_send_newline();
    return CMD_OK;
}

/**
 * Command: HEX
 * Display value in hex and binary
 */
unsigned char cmd_hex(void)
{
    unsigned char value;

    if (arg_count < 2)
    {
        serial_send_string("Usage: HEX <value>\r\n");
        return CMD_BAD_ARGS;
    }

    value = ascii_to_byte(arg1);

    serial_send_string("Decimal: ");
    serial_send_decimal(value);
    serial_send_newline();

    serial_send_string("Hex: 0x");
    serial_send_hex(value);
    serial_send_newline();

    serial_send_string("Binary: ");
    serial_send_binary(value);
    serial_send_newline();

    return CMD_OK;
}

/**
 * Process command
 */
void process_command(unsigned char *cmd)
{
    unsigned char result;

    /* Parse command line */
    parse_command(cmd);

    /* Execute command */
    if (arg_count == 0)
    {
        serial_send_string("Empty command\r\n");
        return;
    }

    /* Match command */
    if (strcmp_i(arg1, (unsigned char*)"HELP") == 0 ||
        strcmp_i(arg1, (unsigned char*)"?") == 0)
    {
        result = cmd_help();
    }
    else if (strcmp_i(arg1, (unsigned char*)"LED") == 0)
    {
        result = cmd_led();
    }
    else if (strcmp_i(arg1, (unsigned char*)"LEDS") == 0)
    {
        result = cmd_leds();
    }
    else if (strcmp_i(arg1, (unsigned char*)"PATTERN") == 0)
    {
        result = cmd_pattern();
    }
    else if (strcmp_i(arg1, (unsigned char*)"STATUS") == 0)
    {
        result = cmd_status();
    }
    else if (strcmp_i(arg1, (unsigned char*)"READ") == 0)
    {
        result = cmd_read();
    }
    else if (strcmp_i(arg1, (unsigned char*)"WRITE") == 0)
    {
        result = cmd_write();
    }
    else if (strcmp_i(arg1, (unsigned char*)"CLEAR") == 0)
    {
        result = cmd_clear();
    }
    else if (strcmp_i(arg1, (unsigned char*)"ECHO") == 0)
    {
        result = cmd_echo();
    }
    else if (strcmp_i(arg1, (unsigned char*)"HEX") == 0)
    {
        result = cmd_hex();
    }
    else
    {
        serial_send_string("Unknown command: ");
        serial_send_string(arg1);
        serial_send_newline();
        result = CMD_NOT_FOUND;
    }

    /* Display result */
    if (result != CMD_OK && result != CMD_NOT_FOUND)
    {
        serial_send_string("Error code: ");
        serial_send_decimal(result);
        serial_send_newline();
    }
}

/**
 * Display welcome banner
 */
void display_welcome(void)
{
    serial_send_string("\r\n");
    serial_send_string("========================================\r\n");
    serial_send_string("  8051 Command Line Interface\r\n");
    serial_send_string("========================================\r\n");
    serial_send_string("Type HELP for available commands\r\n");
    serial_send_string("========================================\r\n\r\n");
}

/**
 * Main program
 */
void main(void)
{
    unsigned char data;

    /* Initialize */
    serial_init();
    LED_PORT = 0xFF;  /* All LEDs off */

    /* Display welcome */
    display_welcome();

    /* Send prompt */
    serial_send_string("CMD> ");

    /* Main loop */
    while (1)
    {
        /* Check for received data */
        if (RI)
        {
            RI = 0;
            data = SBUF;

            /* Echo character */
            serial_send(data);

            /* Check for line terminator */
            if (data == '\r' || data == '\n')
            {
                serial_send_newline();

                /* Process command */
                if (cmd_index > 0)
                {
                    cmd_buffer[cmd_index] = '\0';
                    process_command(cmd_buffer);
                    cmd_index = 0;
                }

                /* Send prompt */
                serial_send_string("CMD> ");
            }
            /* Handle backspace */
            else if (data == 8 || data == 127)
            {
                if (cmd_index > 0)
                {
                    cmd_index--;
                    serial_send_string("\b \b");
                }
            }
            /* Store regular character */
            else if (data >= 32 && data < 127)
            {
                if (cmd_index < CMD_BUFFER_SIZE - 1)
                {
                    cmd_buffer[cmd_index++] = data;
                }
            }
        }
    }
}
