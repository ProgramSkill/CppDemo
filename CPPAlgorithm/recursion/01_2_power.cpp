// recursion Power  -- Single-Branch Linear Recursion
#include <iostream>
using namespace std;

// Calculate base^exponent (exponent must be non-negative)
int power(int base, int exponent) {
    // Base case: any number to power 0 is 1
    if (exponent == 0) {
        return 1;
    }

    // Recursive case: ONE recursive call
    return base * power(base, exponent - 1);
}

int main() {
    cout << "2^5 = " << power(2, 5) << endl;  // Output: 32
    return 0;
}