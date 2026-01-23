// recursion binary_search
#include <iostream>
using namespace std;

// Binary Search: Find target in a sorted array
// Time Complexity: O(log n) - Much faster than linear search O(n)!
// Precondition: Array must be sorted!
int binarySearch(int arr[], int left, int right, int target) {
    // [Base Case] Element not found
    if (left > right) {
        cout << "Range exhausted [" << left << ", " << right << "], target not found!" << endl;
        return -1;
    }

    // Find middle index
    int mid = left + (right - left) / 2;
    cout << "Searching in range [" << left << ", " << right << "], mid = " << mid << ", arr[mid] = " << arr[mid] << endl;

    // [Base Case] Element found
    if (arr[mid] == target) {
        cout << "Found! Target " << target << " at index " << mid << endl;
        return mid;
    }

    // [Recursive Step] Eliminate half of search space
    if (arr[mid] < target) {
        cout << arr[mid] << " < " << target << ", search right half" << endl;
        return binarySearch(arr, mid + 1, right, target);  // Search right
    }
    else {
        cout << arr[mid] << " > " << target << ", search left half" << endl;
        return binarySearch(arr, left, mid - 1, target);   // Search left
    }
}

// Linear Search for comparison: O(n) time
int linearSearch(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        cout << "Checking index " << i << ", arr[" << i << "] = " << arr[i] << endl;
        if (arr[i] == target) {
            cout << "Found at index " << i << endl;
            return i;
        }
    }
    cout << "Target not found!" << endl;
    return -1;
}

int main() {
    int arr[] = { 2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78 };
    int size = 11;
    int target = 23;

    cout << "Array: ";
    for (int i = 0; i < size; i++) {
        cout << arr[i] << " ";
    }
    cout << "\n" << endl;

    cout << "=== BINARY SEARCH ===" << endl;
    cout << "Searching for: " << target << endl;
    int resultBinary = binarySearch(arr, 0, size - 1, target);
    cout << "Result: " << resultBinary << "\n" << endl;

    cout << "=== LINEAR SEARCH (for comparison) ===" << endl;
    cout << "Searching for: " << target << endl;
    int resultLinear = linearSearch(arr, size, target);
    cout << "Result: " << resultLinear << "\n" << endl;

    return 0;
}

/*
Binary Search vs Linear Search Performance:

Array size: 1,000
  - Binary Search: ~10 comparisons (log2(1000) = 10)
  - Linear Search: ~500 comparisons (worst case)
  - Binary Search is 50x faster!

Array size: 1,000,000
  - Binary Search: ~20 comparisons (log2(1000000) = 20)
  - Linear Search: ~500,000 comparisons (worst case)
  - Binary Search is 25,000x faster!

Array size: 1,000,000,000
  - Binary Search: ~30 comparisons (log2(1000000000) = 30)
  - Linear Search: ~500,000,000 comparisons (worst case)
  - Binary Search is 16,666,667x faster!
*/