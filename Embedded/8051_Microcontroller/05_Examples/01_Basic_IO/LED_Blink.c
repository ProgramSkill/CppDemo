// LED Blink Example
// Blinks an LED connected to P1.0

#include <reg51.h>

sbit LED = P1^0;  // LED connected to P1.0

void delay(unsigned int ms) {
    unsigned int i, j;
    for(i = 0; i < ms; i++)
        for(j = 0; j < 123; j++);  // Approximate 1ms at 12MHz
}

void main() {
    while(1) {
        LED = 0;      // Turn ON LED (active low)
        delay(500);   // Wait 500ms
        LED = 1;      // Turn OFF LED
        delay(500);   // Wait 500ms
    }
}
