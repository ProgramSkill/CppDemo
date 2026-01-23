// std::span - Non-owning view over contiguous sequence (C++20)
// Header: <span>
// Provides safe array/view access without ownership

#if __cplusplus >= 202002L

#include <iostream>
#include <span>
#include <vector>
#include <array>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::span Demo ===" << endl;
    int arr[] = {1, 2, 3, 4, 5};
    span<int> s1(arr);
    span<int, 5> s2(arr);  // Fixed extent

    cout << "Span size: " << s1.size() << endl;
    cout << "Elements: ";
    for (int x : s1) cout << x << " ";
    cout << endl;
    cout << endl;
}

void fromVectorDemo() {
    cout << "=== Span from Vector ===" << endl;
    vector<int> vec = {10, 20, 30, 40, 50};
    span<int> s(vec);

    cout << "Span from vector: ";
    for (int x : s) cout << x << " ";
    cout << endl;

    // Modifying through span
    s[0] = 99;
    cout << "After s[0] = 99, vec[0] = " << vec[0] << endl;
    cout << endl;
}

void subspanDemo() {
    cout << "=== Subspan ===" << endl;
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8};
    span<int> s(arr);

    auto s1 = s.subspan(2, 3);  // [2, 5) -> 3, 4, 5
    cout << "subspan(2, 3): ";
    for (int x : s1) cout << x << " ";
    cout << endl;

    auto s2 = s.first(3);
    cout << "first(3): ";
    for (int x : s2) cout << x << " ";
    cout << endl;

    auto s3 = s.last(3);
    cout << "last(3): ";
    for (int x : s3) cout << x << " ";
    cout << endl;
    cout << endl;
}

void sizeBytesDemo() {
    cout << "=== size_bytes ===" << endl;
    int arr[] = {1, 2, 3, 4, 5};
    span<int> s(arr);

    cout << "size(): " << s.size() << endl;
    cout << "size_bytes(): " << s.size_bytes() << endl;
    cout << endl;
}

void functionParameterDemo() {
    cout << "=== Function Parameter ===" << endl;
    auto printSpan = [](span<const int> s) {
        cout << "Span contents: ";
        for (int x : s) cout << x << " ";
        cout << endl;
    };

    int arr[] = {1, 2, 3};
    vector<int> vec = {4, 5, 6};
    array<int, 3> arr2 = {7, 8, 9};

    printSpan(arr);
    printSpan(vec);
    printSpan(arr2);
    cout << endl;
}

void noCopyDemo() {
    cout << "=== Zero-Copy View ===" << endl;
    vector<int> vec = {1, 2, 3, 4, 5};

    // Pass by reference (no copy)
    auto process = [](span<const int> s) {
        return s.size() * sizeof(int);
    };

    cout << "Bytes: " << process(vec) << endl;
    cout << "No vector copy was made!" << endl;
    cout << endl;
}

#else

#include <iostream>
using namespace std;

int main() {
    cout << "std::span requires C++20 or later!" << endl;
    cout << "Compile with: /std:c++20 (MSVC) or -std=c++20 (GCC/Clang)" << endl;
    return 0;
}

#endif

int main() {
#if __cplusplus >= 202002L
    cout << "========================================\n";
    cout << "        std::span Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    fromVectorDemo();
    subspanDemo();
    sizeBytesDemo();
    functionParameterDemo();
    noCopyDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::span: Non-owning view over contiguous memory\n";
    cout << "  - Zero-overhead abstraction\n";
    cout << "  - Works with arrays, vector, string\n";
    cout << "  - No ownership, no allocation\n";
    cout << "  - Prevents decay to pointer\n";
    cout << "  - Perfect for function parameters\n";

#else
    cout << "std::span requires C++20 or later!" << endl;
#endif

    return 0;
}

/*
C++20 Output Summary:
=== Basic ===
Span size: 5
Elements: 1 2 3 4 5

=== Subspan ===
subspan(2, 3): 3 4 5
first(3): 1 2 3
last(3): 6 7 8
*/
