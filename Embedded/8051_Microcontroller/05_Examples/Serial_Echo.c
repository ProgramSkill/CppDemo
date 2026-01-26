// Serial Communication Echo Example
// Receives a character and echoes it back

#include <reg51.h>

void serial_init() {
    TMOD = 0x20;  // Timer 1, Mode 2 (8-bit auto-reload)
    TH1 = 0xFD;   // 9600 baud at 11.0592MHz
    SCON = 0x50;  // Mode 1, 8-bit UART, REN enabled
    TR1 = 1;      // Start Timer 1
}

void send_char(char c) {
    SBUF = c;         // Load character to buffer
    while(TI == 0);   // Wait for transmission complete
    TI = 0;           // Clear transmit flag
}

char receive_char() {
    while(RI == 0);   // Wait for reception complete
    RI = 0;           // Clear receive flag
    return SBUF;      // Return received character
}

void send_string(char *str) {
    while(*str) {
        send_char(*str++);
    }
}

void main() {
    char received;

    serial_init();
    send_string("8051 Serial Echo Ready\r\n");

    while(1) {
        received = receive_char();  // Receive character
        send_char(received);        // Echo it back
    }
}
