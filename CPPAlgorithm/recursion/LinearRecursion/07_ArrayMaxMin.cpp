// Array Maximum/Minimum -- Single-Branch Linear Recursion
#include <iostream>
#include <algorithm> //contains max and min template functions

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
    return std::max(arr[n - 1], arrayMax(arr, n - 1));
}

// Array Minimum with raw array
int arrayMin(int arr[], int n) {
    // Base case: only one element
    if (n == 1) {
        return arr[0];
    }

    // Recursive case: min of last element and min of rest
    return std::min(arr[n - 1], arrayMin(arr, n - 1));
}

int main() {
    int arr[] = { 3, 7, 2, 9, 1, 5 };
    int size = 6;

    std::cout << "Sum of array: " << arraySum(arr, size) << std::endl; //27
    std::cout << "Maximum value: " << arrayMax(arr, size) << std::endl;//9
    std::cout << "Minimum value: " << arrayMin(arr, size) << std::endl;//1

    return 0;
}