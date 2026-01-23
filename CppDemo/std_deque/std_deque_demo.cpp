// std::deque - Double-ended queue
// Header: <deque>
// Time Complexity:
//   - Random access: O(1)
//   - Insert/Delete at front/back: O(1) amortized
//   - Insert/Delete in middle: O(n)
// Space Complexity: O(N)
// Implementation:分段连续内存（multiple fixed-size arrays）

#include <iostream>
#include <deque>
#include <algorithm>
#include <numeric>

using namespace std;

// Basic demonstration
void basicDequeDemo() {
    cout << "=== Basic std::deque Demo ===" << endl;

    // Declaration
    deque<int> dq1;                                // Empty deque
    deque<int> dq2(5);                             // 5 elements, value-initialized
    deque<int> dq3(5, 100);                       // 5 elements, all 100
    deque<int> dq4 = {1, 2, 3, 4, 5};             // Initializer list
    deque<int> dq5(dq4.begin(), dq4.end());       // Iterator range

    cout << "dq2 (5 default-initialized): ";
    for (const auto& elem : dq2) cout << elem << " ";
    cout << endl;

    cout << "dq3 (5 elements of 100): ";
    for (const auto& elem : dq3) cout << elem << " ";
    cout << endl;

    cout << "dq4 (initializer list): ";
    for (const auto& elem : dq4) cout << elem << " ";
    cout << endl;

    cout << "dq5 (from iterator range): ";
    for (const auto& elem : dq5) cout << elem << " ";
    cout << endl << endl;
}

// Element access
void elementAccessDemo() {
    cout << "=== Element Access Demo ===" << endl;

    deque<int> dq = {10, 20, 30, 40, 50};

    // operator[] - No bounds checking
    cout << "dq[2]: " << dq[2] << endl;

    // at() - With bounds checking
    cout << "dq.at(3): " << dq.at(3) << endl;

    // front() and back()
    cout << "dq.front(): " << dq.front() << endl;
    cout << "dq.back(): " << dq.back() << endl;

    // Demonstrating bounds checking
    try {
        cout << "Attempting dq.at(10)... ";
        cout << dq.at(10) << endl;
    } catch (const out_of_range& e) {
        cout << "Exception caught: " << e.what() << endl;
    }
    cout << endl;
}

// Capacity operations
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    deque<int> dq = {1, 2, 3, 4, 5};

    cout << "dq.size(): " << dq.size() << endl;
    cout << "dq.max_size(): " << dq.max_size() << endl;
    cout << "dq.empty(): " << boolalpha << dq.empty() << endl;

    // resize()
    dq.resize(10);
    cout << "After resize(10), size: " << dq.size() << endl;
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    dq.resize(3);
    cout << "After resize(3), size: " << dq.size() << endl;
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // shrink_to_fit() - Request to reduce capacity
    dq.resize(100);
    dq.resize(5);
    dq.shrink_to_fit();
    cout << "After shrink_to_fit(): " << dq.size() << endl;
    cout << endl;
}

// Double-ended operations (deque's specialty)
void doubleEndedOpsDemo() {
    cout << "=== Double-Ended Operations Demo ===" << endl;

    deque<int> dq;

    // push_front() and push_back()
    cout << "Building deque using push_front and push_back:" << endl;
    dq.push_back(1);
    dq.push_back(2);
    dq.push_front(0);
    dq.push_front(-1);

    cout << "After pushes: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // pop_front() and pop_back()
    dq.pop_front();
    dq.pop_back();

    cout << "After pops: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // emplace_front() and emplace_back()
    dq.emplace_front(100);
    dq.emplace_back(200);

    cout << "After emplaces: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // emplace() - Insert at position
    auto it = dq.begin() + 1;
    dq.emplace(it, 50);

    cout << "After emplace at position 1: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    deque<int> dq = {1, 2, 3, 4, 5};

    // Forward iteration
    cout << "Forward: ";
    for (auto it = dq.begin(); it != dq.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Reverse iteration
    cout << "Reverse: ";
    for (auto it = dq.rbegin(); it != dq.rend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Iterator arithmetic (random access)
    auto it = dq.begin() + 2;
    cout << "begin() + 2: " << *it << endl;
    cout << "end() - begin(): " << (dq.end() - dq.begin()) << endl;

    // Const iterators
    cout << "Const iteration: ";
    for (auto it = dq.cbegin(); it != dq.cend(); ++it) {
        cout << *it << " ";
    }
    cout << endl << endl;
}

// Modifying operations
void modifyingDemo() {
    cout << "=== Modifying Operations Demo ===" << endl;

    deque<int> dq = {1, 2, 3, 4, 5};

    cout << "Original: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // insert()
    auto it = dq.begin() + 2;
    dq.insert(it, 100);
    cout << "After insert(100) at position 2: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // insert() with count
    dq.insert(dq.begin(), 3, 0);
    cout << "After insert(3, 0) at begin: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // erase()
    dq.erase(dq.begin() + 1);
    cout << "After erase position 1: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // erase() range
    dq.erase(dq.begin() + 2, dq.begin() + 5);
    cout << "After erase range [2, 5): ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // clear()
    deque<int> dq2 = {1, 2, 3};
    dq2.clear();
    cout << "After clear(), size: " << dq2.size() << ", empty: " << boolalpha << dq2.empty() << endl;

    // assign()
    deque<int> dq3;
    dq3.assign(5, 42);
    cout << "After assign(5, 42): ";
    for (const auto& elem : dq3) cout << elem << " ";
    cout << endl;

    // swap()
    dq.swap(dq3);
    cout << "After swap: " << endl;
    cout << "  dq: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;
    cout << "  dq3: ";
    for (const auto& elem : dq3) cout << elem << " ";
    cout << endl << endl;
}

// STL Algorithms
void algorithmDemo() {
    cout << "=== STL Algorithms Demo ===" << endl;

    deque<int> dq = {5, 2, 8, 1, 9, 3, 7, 4, 6, 0};

    cout << "Original: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // sort()
    sort(dq.begin(), dq.end());
    cout << "After sort: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // reverse()
    reverse(dq.begin(), dq.end());
    cout << "After reverse: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // accumulate()
    int sum = accumulate(dq.begin(), dq.end(), 0);
    cout << "Sum: " << sum << endl;

    // count_if()
    auto count = count_if(dq.begin(), dq.end(), [](int n) { return n % 2 == 0; });
    cout << "Even numbers count: " << count << endl;

    // for_each()
    deque<int> dq2 = {1, 2, 3, 4, 5};
    for_each(dq2.begin(), dq2.end(), [](int& n) { n *= 2; });
    cout << "After for_each (multiply by 2): ";
    for (const auto& elem : dq2) cout << elem << " ";
    cout << endl << endl;
}

// Comparison operations
void comparisonDemo() {
    cout << "=== Comparison Demo ===" << endl;

    deque<int> dq1 = {1, 2, 3};
    deque<int> dq2 = {1, 2, 3};
    deque<int> dq3 = {1, 2, 4};

    cout << "dq1: {1, 2, 3}" << endl;
    cout << "dq2: {1, 2, 3}" << endl;
    cout << "dq3: {1, 2, 4}" << endl;

    cout << boolalpha;
    cout << "dq1 == dq2: " << (dq1 == dq2) << endl;
    cout << "dq1 < dq3: " << (dq1 < dq3) << endl;
    cout << "dq3 > dq1: " << (dq3 > dq1) << endl;
    cout << endl;
}

// Deque as a queue/stack
void dequeAsQueueStackDemo() {
    cout << "=== Deque as Queue/Stack Demo ===" << endl;

    deque<int> dq;

    // Use as a queue (FIFO)
    cout << "Using deque as a queue:" << endl;
    dq.push_back(1);
    dq.push_back(2);
    dq.push_back(3);

    while (!dq.empty()) {
        cout << "Front: " << dq.front() << ", ";
        dq.pop_front();
        cout << "popped, size now: " << dq.size() << endl;
    }

    // Use as a stack (LIFO)
    cout << "\nUsing deque as a stack:" << endl;
    dq.push_back(10);
    dq.push_back(20);
    dq.push_back(30);

    while (!dq.empty()) {
        cout << "Back: " << dq.back() << ", ";
        dq.pop_back();
        cout << "popped, size now: " << dq.size() << endl;
    }
    cout << endl;
}

// Performance characteristics
void performanceDemo() {
    cout << "=== Performance Characteristics ===" << endl;

    deque<int> dq;

    // Operations at both ends are O(1)
    for (int i = 0; i < 5; ++i) {
        dq.push_back(i);
    }
    for (int i = -1; i > -5; --i) {
        dq.push_front(i);
    }

    cout << "After pushing to both ends: ";
    for (const auto& elem : dq) cout << elem << " ";
    cout << endl;

    // Random access is O(1) but slower than vector
    cout << "Random access dq[3]: " << dq[3] << endl;

    // Memory structure
    cout << "\nMemory characteristics:" << endl;
    cout << "  - Segmented memory (multiple chunks)" << endl;
    cout << "  - O(1) access to any element" << endl;
    cout << "  - O(1) insertion at both ends" << endl;
    cout << "  - No reallocation when growing" << endl;
    cout << "  - Slightly slower than vector for random access" << endl;
    cout << endl;
}

// Use case: Sliding window
void slidingWindowDemo() {
    cout << "=== Sliding Window Use Case ===" << endl;

    deque<int> window;
    int data[] = {1, 3, -1, -3, 5, 3, 6, 7};
    int k = 3;  // Window size
    int n = sizeof(data) / sizeof(data[0]);

    cout << "Array: [1, 3, -1, -3, 5, 3, 6, 7]" << endl;
    cout << "Window size: " << k << endl;
    cout << "Maximum in each window:" << endl;

    for (int i = 0; i < n; ++i) {
        // Remove elements outside the window
        while (!window.empty() && window.front() <= i - k) {
            window.pop_front();
        }

        // Remove smaller elements from the back
        while (!window.empty() && data[window.back()] < data[i]) {
            window.pop_back();
        }

        window.push_back(i);

        // Print maximum for complete windows
        if (i >= k - 1) {
            cout << "  Window [" << i - k + 1 << " - " << i << "]: "
                 << data[window.front()] << endl;
        }
    }
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "    std::deque Complete Demonstration" << endl;
    cout << "========================================" << endl << endl;

    basicDequeDemo();
    elementAccessDemo();
    capacityDemo();
    doubleEndedOpsDemo();
    iteratorDemo();
    modifyingDemo();
    algorithmDemo();
    comparisonDemo();
    dequeAsQueueStackDemo();
    performanceDemo();
    slidingWindowDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::deque advantages:" << endl;
    cout << "  - O(1) insertion at both front and back" << endl;
    cout << "  - O(1) random access (like vector)" << endl;
    cout << "  - No reallocation when growing" << endl;
    cout << "  - Better than vector for frequent front operations" << endl;
    cout << "  - Excellent for sliding window algorithms" << endl;
    cout << "\nWhen to use deque:" << endl;
    cout << "  - Need frequent insertions at both ends" << endl;
    cout << "  - Implementing queues or double-ended queues" << endl;
    cout << "  - Sliding window algorithms" << endl;
    cout << "  - When memory reallocation is problematic" << endl;

    return 0;
}

/*
Output Summary:
========================================
    std::deque Complete Demonstration
========================================

=== Basic std::deque Demo ===
dq2 (5 default-initialized): 0 0 0 0 0
dq3 (5 elements of 100): 100 100 100 100 100
dq4 (initializer list): 1 2 3 4 5
dq5 (from iterator range): 1 2 3 4 5

=== Double-Ended Operations Demo ===
Building deque using push_front and push_back:
After pushes: -1 0 1 2
After pops: 0 1
After emplaces: 100 0 1 200
After emplace at position 1: 100 50 0 1 200

=== Sliding Window Use Case ===
Array: [1, 3, -1, -3, 5, 3, 6, 7]
Window size: 3
Maximum in each window:
  Window [0 - 2]: 3
  Window [1 - 3]: 3
  Window [2 - 4]: 5
  Window [3 - 5]: 5
  Window [4 - 6]: 6
  Window [5 - 7]: 7

========================================
              Summary
========================================
std::deque advantages:
  - O(1) insertion at both front and back
  - O(1) random access (like vector)
  - No reallocation when growing
  - Better than vector for frequent front operations
  - Excellent for sliding window algorithms
*/
