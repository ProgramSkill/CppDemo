// Timer-based Delay Example
// Uses Timer 0 for accurate 1ms delays

#include <reg51.h>

sbit LED = P1^0;

void timer0_delay_1ms() {
    TMOD = 0x01;  // Timer 0, Mode 1 (16-bit)
    TH0 = 0xFC;   // Load high byte for 1ms at 12MHz
    TL0 = 0x66;   // Load low byte
    TR0 = 1;      // Start timer
    while(TF0 == 0);  // Wait for overflow
    TR0 = 0;      // Stop timer
    TF0 = 0;      // Clear overflow flag
}

void delay_ms(unsigned int ms) {
    unsigned int i;
    for(i = 0; i < ms; i++)
        timer0_delay_1ms();
}

void main() {
    while(1) {
        LED = 0;
        delay_ms(1000);  // 1 second ON
        LED = 1;
        delay_ms(1000);  // 1 second OFF
    }
}
