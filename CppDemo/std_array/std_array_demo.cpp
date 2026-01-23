// std::array - Fixed-size array container
// Header: <array>
// Time Complexity:
//   - Random access: O(1)
//   - Insert/Delete: Not supported (fixed size)
// Space Complexity: O(N) where N is fixed at compile time

#include <iostream>
#include <array>
#include <algorithm>
#include <numeric>

using namespace std;

// Basic demonstration of std::array
void basicArrayDemo() {
    cout << "=== Basic std::array Demo ===" << endl;

    // Declaration and initialization
    array<int, 5> arr1 = {1, 2, 3, 4, 5};           // Uniform initialization
    array<int, 5> arr2{};                            // Zero-initialized
    array<int, 5> arr3{10};                          // {10, 0, 0, 0, 0}
    array<string, 3> arr4 = {"hello", "world", "!"}; // String array

    cout << "arr1: ";
    for (const auto& elem : arr1) {
        cout << elem << " ";
    }
    cout << endl;

    cout << "arr2 (zero-initialized): ";
    for (const auto& elem : arr2) {
        cout << elem << " ";
    }
    cout << endl;

    cout << "arr3 (partial init): ";
    for (const auto& elem : arr3) {
        cout << elem << " ";
    }
    cout << endl;

    cout << "arr4 (strings): ";
    for (const auto& elem : arr4) {
        cout << elem << " ";
    }
    cout << endl << endl;
}

// Element access methods
void elementAccessDemo() {
    cout << "=== Element Access Demo ===" << endl;

    array<int, 5> arr = {10, 20, 30, 40, 50};

    // Operator[] - No bounds checking
    cout << "arr[2] (no bounds check): " << arr[2] << endl;

    // at() - Bounds checking, throws std::out_of_range
    cout << "arr.at(3) (with bounds check): " << arr.at(3) << endl;

    // front() and back()
    cout << "arr.front(): " << arr.front() << endl;
    cout << "arr.back(): " << arr.back() << endl;

    // data() - Pointer to underlying array
    int* ptr = arr.data();
    cout << "arr.data()[1]: " << ptr[1] << endl;

    // Demonstrating bounds checking
    try {
        cout << "Attempting arr.at(10)... ";
        cout << arr.at(10) << endl;
    } catch (const out_of_range& e) {
        cout << "Exception caught: " << e.what() << endl;
    }
    cout << endl;
}

// Capacity operations
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    array<int, 5> arr = {1, 2, 3, 4, 5};

    cout << "arr.size(): " << arr.size() << endl;        // 5 (fixed)
    cout << "arr.max_size(): " << arr.max_size() << endl; // 5 (same as size)
    cout << "arr.empty(): " << boolalpha << arr.empty() << endl;

    array<int, 0> emptyArr{};
    cout << "emptyArr.empty(): " << boolalpha << emptyArr.empty() << endl;
    cout << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    array<int, 5> arr = {1, 2, 3, 4, 5};

    // begin() and end()
    cout << "Forward iteration: ";
    for (auto it = arr.begin(); it != arr.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // cbegin() and cend() - const iterators
    cout << "Const forward iteration: ";
    for (auto it = arr.cbegin(); it != arr.cend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // rbegin() and rend() - reverse iterators
    cout << "Reverse iteration: ";
    for (auto it = arr.rbegin(); it != arr.rend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Range-based for loop
    cout << "Range-based for: ";
    for (const auto& elem : arr) {
        cout << elem << " ";
    }
    cout << endl << endl;
}

// Modifying operations
void modifyingDemo() {
    cout << "=== Modifying Operations Demo ===" << endl;

    array<int, 5> arr = {1, 2, 3, 4, 5};

    cout << "Original: ";
    for (const auto& elem : arr) cout << elem << " ";
    cout << endl;

    // fill() - Set all elements to a value
    arr.fill(100);
    cout << "After fill(100): ";
    for (const auto& elem : arr) cout << elem << " ";
    cout << endl;

    // swap() - Exchange contents with another array
    array<int, 5> other = {10, 20, 30, 40, 50};
    arr.swap(other);

    cout << "After swap with {10,20,30,40,50}: ";
    for (const auto& elem : arr) cout << elem << " ";
    cout << endl;

    // Direct element modification
    arr[0] = 999;
    arr.at(1) = 888;
    arr.front() = 777;
    arr.back() = 111;

    cout << "After direct modifications: ";
    for (const auto& elem : arr) cout << elem << " ";
    cout << endl << endl;
}

// STL Algorithms with std::array
void algorithmDemo() {
    cout << "=== STL Algorithms Demo ===" << endl;

    array<int, 10> arr = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};

    cout << "Original: ";
    for (const auto& elem : arr) cout << elem << " ";
    cout << endl;

    // sort()
    sort(arr.begin(), arr.end());
    cout << "After sort: ";
    for (const auto& elem : arr) cout << elem << " ";
    cout << endl;

    // Binary search algorithms (requires sorted array)
    bool found = binary_search(arr.begin(), arr.end(), 7);
    cout << "binary_search for 7: " << boolalpha << found << endl;

    auto lower = lower_bound(arr.begin(), arr.end(), 5);
    cout << "lower_bound(5): " << *lower << " at index " << (lower - arr.begin()) << endl;

    // accumulate() - Sum all elements
    int sum = accumulate(arr.begin(), arr.end(), 0);
    cout << "accumulate (sum): " << sum << endl;

    // for_each() - Apply function to each element
    cout << "for_each (multiply by 2): ";
    for_each(arr.begin(), arr.end(), [](int& n) { n *= 2; });
    for (const auto& elem : arr) cout << elem << " ";
    cout << endl;

    // count() and count_if()
    array<int, 10> arr2 = {1, 2, 3, 2, 4, 2, 5, 2, 6, 2};
    int count2 = count(arr2.begin(), arr2.end(), 2);
    cout << "count of 2 in arr2: " << count2 << endl;

    // find()
    auto it = find(arr2.begin(), arr2.end(), 4);
    if (it != arr2.end()) {
        cout << "find(4): found at index " << (it - arr2.begin()) << endl;
    }
    cout << endl;
}

// Comparison operations
void comparisonDemo() {
    cout << "=== Comparison Demo ===" << endl;

    array<int, 5> arr1 = {1, 2, 3, 4, 5};
    array<int, 5> arr2 = {1, 2, 3, 4, 5};
    array<int, 5> arr3 = {1, 2, 3, 4, 6};

    cout << "arr1: {1,2,3,4,5}" << endl;
    cout << "arr2: {1,2,3,4,5}" << endl;
    cout << "arr3: {1,2,3,4,6}" << endl;

    cout << boolalpha;
    cout << "arr1 == arr2: " << (arr1 == arr2) << endl;
    cout << "arr1 == arr3: " << (arr1 == arr3) << endl;
    cout << "arr1 < arr3: " << (arr1 < arr3) << endl;
    cout << "arr3 > arr1: " << (arr3 > arr1) << endl;
    cout << endl;
}

// Multi-dimensional arrays
void multiDimensionalDemo() {
    cout << "=== Multi-dimensional Array Demo ===" << endl;

    // 2D array using std::array
    array<array<int, 3>, 2> matrix = {{
        {1, 2, 3},
        {4, 5, 6}
    }};

    cout << "2D array (2x3 matrix):" << endl;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }

    // Accessing elements
    cout << "matrix[0][2]: " << matrix[0][2] << endl;
    cout << "matrix[1][1]: " << matrix[1][1] << endl;
    cout << endl;
}

// Tuple interface for std::array
void tupleInterfaceDemo() {
    cout << "=== Tuple Interface Demo ===" << endl;

    array<int, 3> arr = {10, 20, 30};

    // std::array can be used with std::tuple_element and std::get
    using ElementType = tuple_element<1, decltype(arr)>::type;
    cout << "Type of element at index 1: " << typeid(ElementType).name() << endl;

    // std::get access
    cout << "get<0>(arr): " << get<0>(arr) << endl;
    cout << "get<1>(arr): " << get<1>(arr) << endl;
    cout << "get<2>(arr): " << get<2>(arr) << endl;

    // std::tuple_size
    cout << "tuple_size<decltype(arr)>::value: " << tuple_size<decltype(arr)>::value << endl;
    cout << endl;
}

// Performance characteristics
void performanceDemo() {
    cout << "=== Performance Characteristics ===" << endl;

    array<int, 5> arr = {1, 2, 3, 4, 5};

    // Random access is O(1) - same as raw arrays
    cout << "Random access arr[4]: " << arr[4] << endl;

    // Stack allocation (no heap allocation)
    cout << "sizeof(arr): " << sizeof(arr) << " bytes" << endl;
    cout << "Elements on stack: Yes" << endl;
    cout << "Cache friendly: Yes (contiguous memory)" << endl;

    // Compile-time size
    constexpr size_t size = arr.size();
    cout << "Size known at compile time: " << size << endl;
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "    std::array Complete Demonstration" << endl;
    cout << "========================================" << endl << endl;

    basicArrayDemo();
    elementAccessDemo();
    capacityDemo();
    iteratorDemo();
    modifyingDemo();
    algorithmDemo();
    comparisonDemo();
    multiDimensionalDemo();
    tupleInterfaceDemo();
    performanceDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::array advantages:" << endl;
    cout << "  - Fixed size known at compile time" << endl;
    cout << "  - No dynamic allocation (stack memory)" << endl;
    cout << "  - Same performance as C-style arrays" << endl;
    cout << "  - Provides STL container interface" << endl;
    cout << "  - Bounds checking with at()" << endl;
    cout << "  - Works with STL algorithms" << endl;
    cout << "  - Never nullptr (unlike raw pointers)" << endl;

    return 0;
}

/*
Output Example:
========================================
    std::array Complete Demonstration
========================================

=== Basic std::array Demo ===
arr1: 1 2 3 4 5
arr2 (zero-initialized): 0 0 0 0 0
arr3 (partial init): 10 0 0 0 0
arr4 (strings): hello world !

=== Element Access Demo ===
arr[2] (no bounds check): 30
arr.at(3) (with bounds check): 40
arr.front(): 10
arr.back(): 50
arr.data()[1]: 20
Attempting arr.at(10)... Exception caught: array::at: __n (which is 10) >= _Nm (which is 5)

=== Capacity Demo ===
arr.size(): 5
arr.max_size(): 5
arr.empty(): false
emptyArr.empty(): true

=== Iterator Demo ===
Forward iteration: 1 2 3 4 5
Const forward iteration: 1 2 3 4 5
Reverse iteration: 5 4 3 2 1
Range-based for: 1 2 3 4 5

=== Modifying Operations Demo ===
Original: 1 2 3 4 5
After fill(100): 100 100 100 100 100
After swap with {10,20,30,40,50}: 10 20 30 40 50
After direct modifications: 999 888 30 40 111

=== STL Algorithms Demo ===
Original: 5 2 8 1 9 3 7 4 6 0
After sort: 0 1 2 3 4 5 6 7 8 9
binary_search for 7: true
lower_bound(5): 5 at index 5
accumulate (sum): 45
for_each (multiply by 2): 0 2 4 6 8 10 12 14 16 18
count of 2 in arr2: 5
find(4): found at index 3

=== Comparison Demo ===
arr1: {1,2,3,4,5}
arr2: {1,2,3,4,5}
arr3: {1,2,3,4,6}
arr1 == arr2: true
arr1 == arr3: false
arr1 < arr3: true
arr3 > arr1: true

=== Multi-dimensional Array Demo ===
2D array (2x3 matrix):
1 2 3
4 5 6
matrix[0][2]: 3
matrix[1][1]: 5

=== Tuple Interface Demo ===
Type of element at index 1: int
get<0>(arr): 10
get<1>(arr): 20
get<2>(arr): 30
tuple_size<decltype(arr)>::value: 3

=== Performance Characteristics ===
Random access arr[4]: 5
sizeof(arr): 20 bytes
Elements on stack: Yes
Cache friendly: Yes (contiguous memory)
Size known at compile time: 5

========================================
              Summary
========================================
std::array advantages:
  - Fixed size known at compile time
  - No dynamic allocation (stack memory)
  - Same performance as C-style arrays
  - Provides STL container interface
  - Bounds checking with at()
  - Works with STL algorithms
  - Never nullptr (unlike raw pointers)
*/
