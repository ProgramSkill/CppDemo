/**
 * GPS_Interface.c
 * 8051 GPS Module Interface - NMEA Data Parsing
 *
 * Description: Interface with GPS module to parse NMEA sentences
 *              Supports GPGGA, GPRMC, GPGSA, GPGSV sentences
 * Hardware: 8051 @ 11.0592MHz, GPS module (TX only)
 * Connections:
 *   8051 RXD (P3.0) <- GPS TXD
 *   GPS VCC         <- 3.3V or 5V (check module)
 *   GPS GND         <- GND
 * Baud Rate: 9600 (8N1) - Standard for GPS modules
 */

#include <reg51.h>

/* Baud rate reload value for 9600 baud @ 11.0592MHz */
#define BAUD_9600 0xFD

/* Buffer sizes */
#define NMEA_BUFFER_SIZE 128
#define GPS_BUFFER_SIZE 256

/* NMEA sentence buffers */
unsigned char nmea_buffer[NMEA_BUFFER_SIZE];
unsigned char nmea_index = 0;
bit nmea_ready = 0;

/* GPS data storage */
unsigned char gps_time[10];      /* HHMMSS */
unsigned char gps_date[8];       /* DDMMYY */
unsigned char gps_lat[12];       /* Latitude */
unsigned char gps_ns;            /* North/South */
unsigned char gps_lon[13];       /* Longitude */
unsigned char gps_ew;            /* East/West */
unsigned char gps_fix;           /* Fix quality (0=no, 1=GPS, 2=DGPS) */
unsigned char gps_sats;          /* Number of satellites */
unsigned char gps_alt[8];        /* Altitude */
unsigned char gps_speed[8];      /* Speed over ground */
unsigned char gps_course[8];     /* Course over ground */

/* Status flags */
bit gps_valid = 0;
bit time_valid = 0;
bit date_valid = 0;
bit position_valid = 0;

/**
 * Initialize serial port for GPS reception
 */
void serial_init(void)
{
    TMOD &= 0x0F;
    TMOD |= 0x20;
    TH1 = BAUD_9600;
    TL1 = BAUD_9600;
    TR1 = 1;
    SCON = 0x50;     /* Mode 1, Receive enabled */

    /* Enable serial interrupt for GPS */
    ES = 1;
    EA = 1;

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
 * Send newline
 */
void serial_send_newline(void)
{
    serial_send('\r');
    serial_send('\n');
}

/**
 * Compare strings (case-sensitive for NMEA)
 */
unsigned char nmea_compare(unsigned char *s1, unsigned char *s2, unsigned char len)
{
    unsigned char i;
    for (i = 0; i < len; i++)
    {
        if (s1[i] != s2[i])
            return 0;
    }
    return 1;
}

/**
 * Convert ASCII character to number
 */
unsigned char ascii_to_num(unsigned char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    return 0;
}

/**
 * Parse comma-separated field from NMEA buffer
 * Copies field to destination buffer
 */
unsigned char parse_nmea_field(unsigned char *dest, unsigned char start_pos, unsigned char max_len)
{
    unsigned char i = 0;
    unsigned char pos = start_pos;

    while (pos < nmea_index && nmea_buffer[pos] != ',' && nmea_buffer[pos] != '*')
    {
        if (i < max_len - 1)
        {
            dest[i++] = nmea_buffer[pos];
        }
        pos++;
    }

    dest[i] = '\0';
    return pos + 1;  /* Return position after comma */
}

/**
 * Parse GPGGA sentence - Fix data
 * Format: $GPGGA,HHMMSS,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx*hh
 */
void parse_gpgga(void)
{
    unsigned char pos = 7;  /* Skip "$GPGGA," */
    unsigned char temp[16];

    /* Time (UTC) */
    pos = parse_nmea_field(gps_time, pos, 10);
    time_valid = (gps_time[0] != '\0');

    /* Latitude */
    pos = parse_nmea_field(gps_lat, pos, 12);

    /* North/South */
    pos = parse_nmea_field(temp, pos, 2);
    if (temp[0] != '\0')
        gps_ns = temp[0];

    /* Longitude */
    pos = parse_nmea_field(gps_lon, pos, 13);

    /* East/West */
    pos = parse_nmea_field(temp, pos, 2);
    if (temp[0] != '\0')
        gps_ew = temp[0];

    /* Fix quality */
    pos = parse_nmea_field(temp, pos, 2);
    if (temp[0] != '\0')
        gps_fix = ascii_to_num(temp[0]);

    /* Number of satellites */
    pos = parse_nmea_field(temp, pos, 4);
    gps_sats = 0;
    if (temp[0] != '\0')
    {
        gps_sats = ascii_to_num(temp[0]) * 10;
        if (temp[1] != '\0')
            gps_sats += ascii_to_num(temp[1]);
    }

    /* Altitude */
    pos = parse_nmea_field(temp, pos, 8);  /* Skip HDOP */
    pos = parse_nmea_field(gps_alt, pos, 8);

    /* Position valid if fix quality > 0 */
    position_valid = (gps_fix > 0);
    gps_valid = position_valid;
}

/**
 * Parse GPRMC sentence - Recommended minimum data
 * Format: $GPRMC,HHMMSS,A,llll.ll,a,yyyyy.yy,a,x.x,x.x,DDMMYY,x.x,a*hh
 */
void parse_gprmc(void)
{
    unsigned char pos = 7;  /* Skip "$GPRMC," */
    unsigned char temp[16];

    /* Time */
    pos = parse_nmea_field(gps_time, pos, 10);
    time_valid = (gps_time[0] != '\0');

    /* Status */
    pos = parse_nmea_field(temp, pos, 2);
    if (temp[0] == 'A' || temp[0] == 'V')
    {
        gps_valid = (temp[0] == 'A');
    }

    /* Latitude */
    pos = parse_nmea_field(gps_lat, pos, 12);

    /* North/South */
    pos = parse_nmea_field(temp, pos, 2);
    if (temp[0] != '\0')
        gps_ns = temp[0];

    /* Longitude */
    pos = parse_nmea_field(gps_lon, pos, 13);

    /* East/West */
    pos = parse_nmea_field(temp, pos, 2);
    if (temp[0] != '\0')
        gps_ew = temp[0];

    /* Speed over ground */
    pos = parse_nmea_field(gps_speed, pos, 8);

    /* Course over ground */
    pos = parse_nmea_field(gps_course, pos, 8);

    /* Date */
    pos = parse_nmea_field(gps_date, pos, 8);
    date_valid = (gps_date[0] != '\0');
}

/**
 * Parse GPGSA sentence - Satellite data
 * Format: $GPGSA,a,x,xx,xx,xx,xx,xx,xx,xx,xx,xx,xx,xx,x.x,x.x,x.x*hh
 */
void parse_gpgsa(void)
{
    unsigned char pos = 7;  /* Skip "$GPGSA," */
    unsigned char temp[16];

    /* Mode (M=Manual, A=Auto) */
    pos = parse_nmea_field(temp, pos, 2);

    /* Fix type (1=No fix, 2=2D fix, 3=3D fix) */
    pos = parse_nmea_field(temp, pos, 2);
    if (temp[0] != '\0')
    {
        unsigned char fix_type = ascii_to_num(temp[0]);
        position_valid = (fix_type >= 2);
    }
}

/**
 * Parse GPGSV sentence - Satellites in view
 * Format: $GPGSV,x,x,xx,xx,xx,xxx,xx,xx,xx,xxx,xx,xx,xx,xxx*hh
 */
void parse_gpgsv(void)
{
    /* Just count satellites - detailed parsing skipped for brevity */
    unsigned char pos = 7;  /* Skip "$GPGSV," */
    unsigned char temp[16];
    unsigned char i;

    /* Skip first 3 fields */
    for (i = 0; i < 3; i++)
    {
        while (pos < nmea_index && nmea_buffer[pos] != ',')
            pos++;
        pos++;
    }

    /* Count satellites (4 per sentence) */
    unsigned char sat_count = 0;
    while (pos < nmea_index && nmea_buffer[pos] != '*')
    {
        pos = parse_nmea_field(temp, pos, 4);
        if (temp[0] != '\0')
            sat_count++;
    }
}

/**
 * Process complete NMEA sentence
 */
void process_nmea_sentence(void)
{
    /* Check minimum length */
    if (nmea_index < 6)
        return;

    /* Check for start marker */
    if (nmea_buffer[0] != '$')
        return;

    /* Identify sentence type */
    if (nmea_compare(&nmea_buffer[1], (unsigned char*)"GPGGA", 5))
    {
        parse_gpgga();
    }
    else if (nmea_compare(&nmea_buffer[1], (unsigned char*)"GPRMC", 5))
    {
        parse_gprmc();
    }
    else if (nmea_compare(&nmea_buffer[1], (unsigned char*)"GPGSA", 5))
    {
        parse_gpgsa();
    }
    else if (nmea_compare(&nmea_buffer[1], (unsigned char*)"GPGSV", 5))
    {
        parse_gpgsv();
    }
}

/**
 * Display GPS data
 */
void display_gps_data(void)
{
    serial_send_string("\r\n========== GPS DATA ==========\r\n");

    /* Fix status */
    serial_send_string("Status: ");
    if (gps_valid)
    {
        serial_send_string("FIX (");
        serial_send_decimal(gps_fix);
        serial_send_string("D)\r\n");
    }
    else
    {
        serial_send_string("NO FIX\r\n");
    }

    /* Time */
    if (time_valid)
    {
        serial_send_string("Time (UTC): ");
        serial_send_string(gps_time);
        serial_send_newline();
    }

    /* Date */
    if (date_valid)
    {
        serial_send_string("Date: ");
        serial_send_string(gps_date);
        serial_send_newline();
    }

    /* Position */
    if (position_valid)
    {
        serial_send_string("Latitude: ");
        serial_send_string(gps_lat);
        serial_send(gps_ns);
        serial_send_newline();

        serial_send_string("Longitude: ");
        serial_send_string(gps_lon);
        serial_send(gps_ew);
        serial_send_newline();

        serial_send_string("Altitude: ");
        serial_send_string(gps_alt);
        serial_send_string("m\r\n");

        serial_send_string("Satellites: ");
        serial_send_decimal(gps_sats);
        serial_send_newline();

        if (gps_speed[0] != '\0')
        {
            serial_send_string("Speed: ");
            serial_send_string(gps_speed);
            serial_send_string(" knots\r\n");
        }

        if (gps_course[0] != '\0')
        {
            serial_send_string("Course: ");
            serial_send_string(gps_course);
            serial_send_string(" deg\r\n");
        }
    }

    serial_send_string("==============================\r\n\r\n");
}

/**
 * Display raw NMEA sentence
 */
void display_raw_nmea(void)
{
    unsigned char i;
    serial_send_string("RAW: ");
    for (i = 0; i < nmea_index; i++)
    {
        serial_send(nmea_buffer[i]);
    }
    serial_send_newline();
}

/**
 * Display welcome message
 */
void display_welcome(void)
{
    serial_send_string("\r\n");
    serial_send_string("========================================\r\n");
    serial_send_string("  8051 GPS NMEA Parser\r\n");
    serial_send_string("========================================\r\n");
    serial_send_string("Commands:\r\n");
    serial_send_string("  DATA     - Show parsed GPS data\r\n");
    serial_send_string("  RAW      - Show raw NMEA sentence\r\n");
    serial_send_string("  STATUS   - Show fix status\r\n");
    serial_send_string("  TIME     - Show UTC time\r\n");
    serial_send_string("  POS      - Show position\r\n");
    serial_send_string("========================================\r\n\r\n");
}

/**
 * Serial interrupt handler
 * Receives NMEA data character by character
 */
void serial_isr(void) interrupt 4
{
    if (RI)
    {
        RI = 0;
        unsigned char data = SBUF;

        /* Start of sentence */
        if (data == '$')
        {
            nmea_index = 0;
            nmea_buffer[nmea_index++] = data;
        }
        /* End of sentence */
        else if (data == '\r' || data == '\n')
        {
            if (nmea_index > 0)
            {
                nmea_buffer[nmea_index] = '\0';
                nmea_ready = 1;
                nmea_index = 0;
            }
        }
        /* Sentence data */
        else if (nmea_index > 0 && nmea_index < NMEA_BUFFER_SIZE - 1)
        {
            nmea_buffer[nmea_index++] = data;
        }
        else if (nmea_index >= NMEA_BUFFER_SIZE - 1)
        {
            /* Buffer overflow */
            nmea_index = 0;
        }
    }

    if (TI)
    {
        TI = 0;
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

    /* Initialize serial port */
    serial_init();

    /* Display welcome */
    display_welcome();

    /* Send prompt */
    serial_send_string("GPS> ");

    /* Main loop */
    while (1)
    {
        /* Process NMEA sentence if ready */
        if (nmea_ready)
        {
            process_nmea_sentence();
            nmea_ready = 0;

            /* Auto-display when fix acquired */
            if (gps_valid && !position_valid)
            {
                position_valid = 1;
                serial_send_string("\r\n! GPS FIX ACQUIRED !\r\n");
                display_gps_data();
            }
        }

        /* Check for serial input */
        if (RI)
        {
            RI = 0;
            data = SBUF;

            /* Echo */
            serial_send(data);

            /* Parse command */
            if (data == '\r' || data == '\n')
            {
                serial_send_string("\r\n");

                if (cmd_index > 0)
                {
                    cmd_buffer[cmd_index] = '\0';

                    /* Process commands */
                    if (cmd_buffer[0] == 'D' && cmd_buffer[1] == 'A')
                    {
                        display_gps_data();
                    }
                    else if (cmd_buffer[0] == 'R')
                    {
                        display_raw_nmea();
                    }
                    else if (cmd_buffer[0] == 'S')
                    {
                        serial_send_string("Fix Status: ");
                        serial_send_string(gps_valid ? "Valid" : "Invalid");
                        serial_send_string(", Satellites: ");
                        serial_send_decimal(gps_sats);
                        serial_send_newline();
                    }
                    else if (cmd_buffer[0] == 'T' && cmd_buffer[1] == 'I')
                    {
                        if (time_valid)
                        {
                            serial_send_string("UTC Time: ");
                            serial_send_string(gps_time);
                            serial_send_newline();
                        }
                        else
                        {
                            serial_send_string("Time not available\r\n");
                        }
                    }
                    else if (cmd_buffer[0] == 'P' && cmd_buffer[1] == 'O')
                    {
                        if (position_valid)
                        {
                            serial_send_string("Lat: ");
                            serial_send_string(gps_lat);
                            serial_send(gps_ns);
                            serial_send_newline();
                            serial_send_string("Lon: ");
                            serial_send_string(gps_lon);
                            serial_send(gps_ew);
                            serial_send_newline();
                            serial_send_string("Alt: ");
                            serial_send_string(gps_alt);
                            serial_send_string("m\r\n");
                        }
                        else
                        {
                            serial_send_string("No position fix\r\n");
                        }
                    }
                    else
                    {
                        serial_send_string("Unknown command\r\n");
                    }

                    cmd_index = 0;
                }

                serial_send_string("GPS> ");
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
