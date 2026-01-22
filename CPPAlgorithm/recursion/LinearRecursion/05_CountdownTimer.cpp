// Countdown Timer -- Single-Branch Linear Recursion
#include <iostream>
using namespace std;

// Print countdown from n to 1
void countdown(int n) {
    // Base case: stop when reaching 0
    if (n == 0) {
        cout << "Blastoff!" << endl;
        return;
    }

    // Print current number
    cout << n << endl;

    // Recursive case: ONE recursive call
    countdown(n - 1);
}

int main() {
    countdown(5);
    // Output:
    // 5
    // 4
    // 3
    // 2
    // 1
    // Blastoff!
    return 0;
}