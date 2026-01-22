// recursion ReverseString  -- Single-Branch Linear Recursion
#include <iostream>
#include <string>
using namespace std;

// Print string in reverse
void reverseString(string str, int index = 0) {
    // Base case: reached end of string
    if (index == str.length()) {
        return;
    }

    // Recursive case: ONE recursive call
    reverseString(str, index + 1);
    cout << str[index];
}

int main() {
    reverseString("Hello");  // Output: olleH
    cout << endl;
    return 0;
}