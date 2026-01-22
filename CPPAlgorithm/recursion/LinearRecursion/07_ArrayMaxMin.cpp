// Array Maximum/Minimum -- Single-Branch Linear Recursion
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

/**
 * Recursively find the maximum value in array
 * Base case: when index reaches end, return current maximum
 * Recursive case: compare current element with known maximum
 */
int arrayMax(vector<int>& arr, int index, int currentMax) {
    // Base case: reached the end
    if (index == arr.size()) {
        return currentMax;
    }

    // Recursive case: update maximum and continue
    return arrayMax(arr, index + 1, max(currentMax, arr[index]));
}

/**
 * Recursively find the minimum value in array
 * Similar approach to arrayMax but using min function
 */
int arrayMin(vector<int>& arr, int index, int currentMin) {
    // Base case: reached the end
    if (index == arr.size()) {
        return currentMin;
    }

    // Recursive case: update minimum and continue
    return arrayMin(arr, index + 1, min(currentMin, arr[index]));
}

int main() {
    vector<int> arr = { 3, 7, 2, 9, 1, 5 };

    cout << "Maximum value: " << arrayMax(arr, 0, arr[0]) << endl;   // Output: 9
    cout << "Minimum value: " << arrayMin(arr, 0, arr[0]) << endl;   // Output: 1

    return 0;
}