// recursion LinearSearch  -- Single-Branch Linear Recursion
#include <iostream>
using namespace std;

// Search for target in array, starting from index start
int linearSearch(int arr[], int size, int target, int start = 0) {
    // Base case 1: reached end of array
    if (start == size) {
        return -1;  // Not found
    }

    // Base case 2: found the target
    if (arr[start] == target) {
        return start;
    }

    // Recursive case: ONE recursive call (search next element)
    return linearSearch(arr, size, target, start + 1);
}

int main() {
    int arr[] = { 1, 3, 5, 7, 9 };
    int index = linearSearch(arr, 5, 7);
    cout << "Found at index: " << index << endl;  // Output: 3
    return 0;
}