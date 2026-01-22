//Greatest Common Divisor -- Single-Branch Linear Recursion
#include <iostream>
using namespace std;

// Find GCD using Euclidean algorithm
int gcd(int a, int b) {
    // Base case: when b is 0, a is the GCD
    if (b == 0) {
        return a;
    }

    // Recursive case: ONE recursive call
    return gcd(b, a % b);
}

int main() {
    cout << "GCD(48, 18) = " << gcd(48, 18) << endl;  // Output: 6
    return 0;
}