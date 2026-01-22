// Array Maximum/Minimum -- Single-Branch Linear Recursion
#include <iostream>
#include <algorithm>
using namespace std;

// Array Sum with raw array
int arraySum(int arr[], int n) {
    // Base case: when n <= 0
    if (n <= 0) {
        return 0;
    }

    // Recursive case: last element + sum of first n-1 elements
    return arr[n - 1] + arraySum(arr, n - 1);
}

// Array Maximum with raw array
int arrayMax(int arr[], int n) {
    // Base case: only one element
    if (n == 1) {
        return arr[0];
    }

    // Recursive case: max of last element and max of rest
    return max(arr[n - 1], arrayMax(arr, n - 1));
}

// Array Minimum with raw array
int arrayMin(int arr[], int n) {
    // Base case: only one element
    if (n == 1) {
        return arr[0];
    }

    // Recursive case: min of last element and min of rest
    return min(arr[n - 1], arrayMin(arr, n - 1));
}

int main() {
    int arr[] = { 3, 7, 2, 9, 1, 5 };
    int size = 6;

    cout << "Sum of array: " << arraySum(arr, size) << endl;          // Output: 27
    cout << "Maximum value: " << arrayMax(arr, size) << endl;         // Output: 9
    cout << "Minimum value: " << arrayMin(arr, size) << endl;         // Output: 1

    return 0;
}