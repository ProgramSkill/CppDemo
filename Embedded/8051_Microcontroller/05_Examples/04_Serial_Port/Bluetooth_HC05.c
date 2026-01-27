/**
 * Bluetooth_HC05.c
 * 8051 Bluetooth Communication via HC-05 Module
 *
 * Description: Wireless communication using HC-05 Bluetooth module
 *              Features include AT command mode, data transfer, status monitoring
 * Hardware: 8051 @ 11.0592MHz, HC-05 Bluetooth module
 * Connections:
 *   8051 TXD (P3.1) -> HC-05 RXD
 *   8051 RXD (P3.0) -> HC-05 TXD
 *   8051 P3.2      -> HC-05 KEY (for AT mode)
 *   8051 P3.7      -> HC-05 EN (enable)
 * Baud Rate: 9600 (8N1) - Default for HC-05
 */

#include <reg51.h>

/* Baud rate reload value for 9600 baud @ 11.0592MHz */
#define BAUD_9600 0xFD

/* HC-05 Control pins */
sbit HC05_KEY = P3^2;   /* KEY pin for AT command mode */
sbit HC05_EN  = P3^7;   /* EN pin (enable) */
sbit HC05_LED = P3^4;   /* LED status indicator (optional) */

/* Buffer sizes */
#define RX_BUFFER_SIZE 64
#define TX_BUFFER_SIZE 64

/* Buffers */
unsigned char rx_buffer[RX_BUFFER_SIZE];
volatile unsigned char rx_head = 0;
volatile unsigned char rx_tail = 0;

unsigned char tx_buffer[TX_BUFFER_SIZE];
volatile unsigned char tx_head = 0;
volatile unsigned char tx_tail = 0;

/* Status flags */
volatile bit tx_active = 0;
volatile bit connection_active = 0;
volatile bit data_received = 0;
unsigned char connection_count = 0;

/* Bluetooth states */
#define BT_STATE_DISCONNECTED  0
#define BT_STATE_CONNECTED     1
#define BT_STATE_AT_MODE       2

unsigned char bt_state = BT_STATE_DISCONNECTED;

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

    /* Enable interrupts */
    ES = 1;
    EA = 1;

    TI = 0;
    RI = 0;
}

/**
 * Initialize HC-05 module
 */
void hc05_init(void)
{
    /* Default mode: KEY low (data mode) */
    HC05_KEY = 0;
    HC05_EN = 1;  /* Enable module */

    /* Wait for module to initialize */
    /* Simple delay */
    unsigned int i;
    for (i = 0; i < 50000; i++);

    bt_state = BT_STATE_DISCONNECTED;
}

/**
 * Enter AT command mode
 */
void hc05_enter_at_mode(void)
{
    /* Set KEY high and power cycle */
    HC05_KEY = 1;
    HC05_EN = 0;

    /* Wait for power down */
    unsigned int i;
    for (i = 0; i < 10000; i++);

    /* Power on in AT mode */
    HC05_EN = 1;

    /* Wait for module initialization in AT mode */
    for (i = 0; i < 50000; i++);

    bt_state = BT_STATE_AT_MODE;
}

/**
 * Exit AT command mode
 */
void hc05_exit_at_mode(void)
{
    HC05_KEY = 0;

    /* Wait for mode switch */
    unsigned int i;
    for (i = 0; i < 50000; i++);

    bt_state = BT_STATE_DISCONNECTED;
}

/**
 * Send byte
 */
void serial_send(unsigned char data)
{
    SBUF = data;
    while (!TI);
    TI = 0;
}

/**
 * Send string via serial
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
 * Send decimal byte
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
 * Check if data available
 */
bit serial_available(void)
{
    return (rx_head != rx_tail);
}

/**
 * Read byte from buffer
 */
unsigned char serial_read(void)
{
    while (!serial_available());

    unsigned char data = rx_buffer[rx_tail];
    rx_tail = (rx_tail + 1) % RX_BUFFER_SIZE;
    return data;
}

/**
 * Write byte to buffer
 */
void serial_write(unsigned char data)
{
    unsigned char next_head = (tx_head + 1) % TX_BUFFER_SIZE;

    if (next_head != tx_tail)
    {
        tx_buffer[tx_head] = data;
        tx_head = next_head;

        if (!tx_active)
        {
            tx_active = 1;
            TI = 1;
        }
    }
}

/**
 * Write string to buffer
 */
void serial_write_string(unsigned char *str)
{
    while (*str != '\0')
    {
        serial_write(*str);
        str++;
    }
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
 * Send AT command
 * Returns 1 if OK received, 0 otherwise
 */
bit hc05_send_at_command(unsigned char *cmd)
{
    unsigned char response[32];
    unsigned char i = 0;
    unsigned char timeout;

    serial_clear_rx();

    /* Send command */
    serial_send_string(cmd);
    serial_send_string("\r\n");

    /* Wait for response */
    for (timeout = 0; timeout < 100; timeout++)
    {
        if (serial_available())
        {
            response[i++] = serial_read();

            /* Check for OK response */
            if (i >= 2 && response[i-2] == 'O' && response[i-1] == 'K')
            {
                return 1;
            }

            /* Check for ERROR */
            if (i >= 5 && response[i-5] == 'E' &&
                         response[i-4] == 'R' &&
                         response[i-3] == 'R' &&
                         response[i-2] == 'O' &&
                         response[i-1] == 'R')
            {
                return 0;
            }

            if (i >= 30)
                break;
        }

        /* Small delay */
        unsigned int j;
        for (j = 0; j < 1000; j++);
    }

    return 0;
}

/**
 * Get module name
 */
void hc05_get_name(void)
{
    unsigned char c;

    serial_clear_rx();
    serial_send_string("AT+NAME?\r\n");

    /* Read response */
    serial_send_string("Module Name: ");
    for (c = 0; c < 50; c++)
    {
        if (serial_available())
        {
            unsigned char data = serial_read();
            if (data == '\r' || data == '\n')
            {
                if (c > 0) break;
            }
            else
            {
                serial_send(data);
            }
        }
        unsigned int i;
        for (i = 0; i < 1000; i++);
    }
    serial_send_string("\r\n");
}

/**
 * Get module address
 */
void hc05_get_address(void)
{
    unsigned char c;

    serial_clear_rx();
    serial_send_string("AT+ADDR?\r\n");

    serial_send_string("Module Address: ");
    for (c = 0; c < 20; c++)
    {
        if (serial_available())
        {
            unsigned char data = serial_read();
            if (data == '\r' || data == '\n')
            {
                if (c > 0) break;
            }
            else
            {
                serial_send(data);
            }
        }
        unsigned int i;
        for (i = 0; i < 1000; i++);
    }
    serial_send_string("\r\n");
}

/**
 * Get module version
 */
void hc05_get_version(void)
{
    unsigned char c;

    serial_clear_rx();
    serial_send_string("AT+VERSION?\r\n");

    serial_send_string("Firmware Version: ");
    for (c = 0; c < 30; c++)
    {
        if (serial_available())
        {
            unsigned char data = serial_read();
            if (data == '\r' || data == '\n')
            {
                if (c > 0) break;
            }
            else
            {
                serial_send(data);
            }
        }
        unsigned int i;
        for (i = 0; i < 1000; i++);
    }
    serial_send_string("\r\n");
}

/**
 * Serial interrupt handler
 */
void serial_isr(void) interrupt 4
{
    if (RI)
    {
        RI = 0;

        unsigned char next_head = (rx_head + 1) % RX_BUFFER_SIZE;

        if (next_head != rx_tail)
        {
            rx_buffer[rx_head] = SBUF;
            rx_head = next_head;
            data_received = 1;
        }
    }

    if (TI)
    {
        TI = 0;

        if (tx_head != tx_tail)
        {
            SBUF = tx_buffer[tx_tail];
            tx_tail = (tx_tail + 1) % TX_BUFFER_SIZE;
            tx_active = 1;
        }
        else
        {
            tx_active = 0;
        }
    }
}

/**
 * Display welcome message
 */
void display_welcome(void)
{
    serial_send_string("\r\n");
    serial_send_string("========================================\r\n");
    serial_send_string("  8051 Bluetooth HC-05 Demo\r\n");
    serial_send_string("========================================\r\n");
    serial_send_string("Commands:\r\n");
    serial_send_string("  AT       - Enter AT mode\r\n");
    serial_send_string("  DATA     - Exit to data mode\r\n");
    serial_send_string("  NAME     - Get module name\r\n");
    serial_send_string("  ADDR     - Get module address\r\n");
    serial_send_string("  VER      - Get firmware version\r\n");
    serial_send_string("  TEST     - Send test message\r\n");
    serial_send_string("  STATUS   - Show connection status\r\n");
    serial_send_string("========================================\r\n\r\n");
}

/**
 * Display connection status
 */
void display_status(void)
{
    serial_send_string("\r\n--- Bluetooth Status ---\r\n");
    serial_send_string("State: ");

    switch (bt_state)
    {
        case BT_STATE_DISCONNECTED:
            serial_send_string("Disconnected\r\n");
            break;
        case BT_STATE_CONNECTED:
            serial_send_string("Connected\r\n");
            break;
        case BT_STATE_AT_MODE:
            serial_send_string("AT Command Mode\r\n");
            break;
    }

    serial_send_string("Connection Count: ");
    serial_send_decimal(connection_count);
    serial_send_string("\r\n");
    serial_send_string("------------------------\r\n\r\n");
}

/**
 * Process received data
 */
void process_received_data(void)
{
    unsigned char data;

    while (serial_available())
    {
        data = serial_read();

        /* Echo back to sender */
        serial_write(data);

        /* Indicate data received */
        data_received = 0;

        /* Toggle LED indicator */
        HC05_LED = !HC05_LED;
    }
}

/**
 * Main program
 */
void main(void)
{
    unsigned char data;
    unsigned char cmd_buffer[16];
    unsigned char cmd_index = 0;

    /* Initialize */
    serial_init();
    hc05_init();

    /* Display welcome */
    display_welcome();

    /* Send prompt */
    serial_write_string("BT> ");

    /* Main loop */
    while (1)
    {
        /* Process received data */
        if (data_received)
        {
            process_received_data();
        }

        /* Check for incoming data */
        if (serial_available())
        {
            data = serial_read();

            /* Echo locally */
            serial_send(data);

            /* Parse command */
            if (data == '\r' || data == '\n')
            {
                serial_send_string("\r\n");

                /* Process command */
                if (cmd_index > 0)
                {
                    cmd_buffer[cmd_index] = '\0';

                    /* Command matching */
                    if (cmd_buffer[0] == 'A' && cmd_buffer[1] == 'T')
                    {
                        hc05_enter_at_mode();
                        serial_send_string("Entered AT command mode\r\n");
                    }
                    else if (cmd_buffer[0] == 'D' && cmd_buffer[1] == 'A')
                    {
                        hc05_exit_at_mode();
                        serial_send_string("Exited to data mode\r\n");
                    }
                    else if (cmd_buffer[0] == 'N' && cmd_buffer[1] == 'A')
                    {
                        if (bt_state == BT_STATE_AT_MODE)
                            hc05_get_name();
                        else
                            serial_send_string("Enter AT mode first\r\n");
                    }
                    else if (cmd_buffer[0] == 'A' && cmd_buffer[1] == 'D')
                    {
                        if (bt_state == BT_STATE_AT_MODE)
                            hc05_get_address();
                        else
                            serial_send_string("Enter AT mode first\r\n");
                    }
                    else if (cmd_buffer[0] == 'V' && cmd_buffer[1] == 'E')
                    {
                        if (bt_state == BT_STATE_AT_MODE)
                            hc05_get_version();
                        else
                            serial_send_string("Enter AT mode first\r\n");
                    }
                    else if (cmd_buffer[0] == 'T' && cmd_buffer[1] == 'E')
                    {
                        serial_write_string("Hello from 8051 via Bluetooth!\r\n");
                    }
                    else if (cmd_buffer[0] == 'S' && cmd_buffer[1] == 'T')
                    {
                        display_status();
                    }
                    else
                    {
                        serial_send_string("Unknown command\r\n");
                    }

                    cmd_index = 0;
                }

                serial_write_string("BT> ");
            }
            else if (data == 8 || data == 127)
            {
                if (cmd_index > 0)
                {
                    cmd_index--;
                    serial_send_string("\b \b");
                }
            }
            else if (data >= 32 && data < 127)
            {
                if (cmd_index < 15)
                {
                    cmd_buffer[cmd_index++] = data;
                }
            }
        }
    }
}
