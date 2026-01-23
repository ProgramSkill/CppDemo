// std::multiset - Sorted container allowing duplicate elements
// Header: <set>
// Time Complexity:
//   - Insert: O(log n)
//   - Delete: O(log n)
//   - Find: O(log n)
//   - Count: O(log k) where k is count of elements
// Space Complexity: O(N)
// Implementation: Red-black tree (allows duplicates)

#include <iostream>
#include <set>
#include <string>
#include <algorithm>

using namespace std;

// Basic demonstration
void basicMultisetDemo() {
    cout << "=== Basic std::multiset Demo ===" << endl;

    // Declaration
    multiset<int> ms1;                                    // Empty multiset
    multiset<int> ms2 = {5, 2, 8, 2, 1, 9, 2};          // Initializer list with duplicates
    multiset<int> ms3(ms2.begin(), ms2.end());          // Iterator range

    // Note: Duplicates are preserved!
    cout << "ms2 (with duplicates {5,2,8,2,1,9,2}): ";
    for (const auto& elem : ms2) cout << elem << " ";
    cout << endl;

    cout << "ms3 (from iterator range): ";
    for (const auto& elem : ms3) cout << elem << " ";
    cout << endl;

    // Counting duplicates
    cout << "\nCount of element 2: " << ms2.count(2) << endl;
    cout << "Count of element 5: " << ms2.count(5) << endl;
    cout << "Count of element 10: " << ms2.count(10) << endl;
    cout << endl;
}

// Insertion operations
void insertionDemo() {
    cout << "=== Insertion Demo ===" << endl;

    multiset<int> ms;

    // insert() - Always succeeds (allows duplicates)
    ms.insert(5);
    ms.insert(5);  // Duplicate!
    ms.insert(5);  // Another duplicate!

    ms.insert(3);
    ms.insert(7);
    ms.insert(3);  // Duplicate of 3

    cout << "After inserts: ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << endl;

    // insert() returns iterator (not pair like std::set)
    auto it = ms.insert(4);
    cout << "Insert 4, iterator points to: " << *it << endl;

    // insert() with hint
    auto hint = ms.begin();
    advance(hint, 3);
    ms.insert(hint, 6);

    cout << "After insert with hint: ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << endl;

    // emplace()
    ms.emplace(4);
    cout << "After emplace(4): ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << endl;

    // insert() with range
    multiset<int> ms2 = {1, 1, 2, 2};
    ms.insert(ms2.begin(), ms2.end());
    cout << "After insert range: ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Deletion operations
void deletionDemo() {
    cout << "=== Deletion Demo ===" << endl;

    multiset<int> ms = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};

    cout << "Original: ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << endl;

    // erase() by value - removes ALL occurrences!
    size_t count = ms.erase(3);
    cout << "\nErase 3: removed " << count << " element(s)" << endl;
    cout << "After erase(3): ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << endl;

    // erase() by iterator - removes single element
    multiset<int> ms2 = {1, 2, 2, 3};
    cout << "\nms2: ";
    for (const auto& elem : ms2) cout << elem << " ";
    cout << endl;

    auto it = ms2.find(2);
    if (it != ms2.end()) {
        ms2.erase(it);  // Erases only one occurrence
        cout << "After erase iterator at 2 (one occurrence): ";
        for (const auto& elem : ms2) cout << elem << " ";
        cout << endl;
    }

    // erase() by range
    ms = {1, 2, 3, 4, 5, 6, 7, 8};
    auto start = ms.begin();
    auto end = ms.begin();
    advance(start, 2);
    advance(end, 5);
    ms.erase(start, end);
    cout << "\nAfter erase range [2, 5): ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << endl;

    // clear()
    multiset<int> ms3 = {1, 2, 3};
    ms3.clear();
    cout << "After clear(), size: " << ms3.size() << ", empty: " << boolalpha << ms3.empty() << endl;
    cout << endl;
}

// Lookup operations
void lookupDemo() {
    cout << "=== Lookup Demo ===" << endl;

    multiset<int> ms = {10, 20, 20, 30, 30, 30, 40};

    // find() - Returns iterator to first occurrence
    auto it = ms.find(30);
    if (it != ms.end()) {
        cout << "find(30): found " << *it << " (first occurrence)" << endl;
    }

    it = ms.find(25);
    if (it == ms.end()) {
        cout << "find(25): not found" << endl;
    }

    // count() - Returns number of occurrences!
    cout << "\ncount(20): " << ms.count(20) << " occurrences" << endl;
    cout << "count(30): " << ms.count(30) << " occurrences" << endl;
    cout << "count(50): " << ms.count(50) << " occurrences" << endl;

    // lower_bound() - First element >= value
    auto lb = ms.lower_bound(25);
    cout << "\nlower_bound(25): " << *lb << endl;

    // upper_bound() - First element > value
    auto ub = ms.upper_bound(30);
    cout << "upper_bound(30): " << *ub << endl;

    // equal_range() - Range of elements equal to value
    auto range = ms.equal_range(30);
    cout << "equal_range(30): ";
    for (auto it = range.first; it != range.second; ++it) {
        cout << *it << " ";
    }
    cout << endl;
    cout << "Count in range: " << distance(range.first, range.second) << endl;
    cout << endl;
}

// Capacity operations
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    multiset<int> ms = {1, 2, 2, 3, 3, 3};

    cout << "ms.size(): " << ms.size() << endl;
    cout << "ms.max_size(): " << ms.max_size() << endl;
    cout << "ms.empty(): " << boolalpha << ms.empty() << endl;

    multiset<int> emptyMS;
    cout << "emptyMS.empty(): " << boolalpha << emptyMS.empty() << endl;
    cout << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    multiset<int> ms = {1, 2, 2, 3, 3, 3};

    // Forward iteration (sorted order!)
    cout << "Forward (sorted): ";
    for (auto it = ms.begin(); it != ms.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Reverse iteration
    cout << "Reverse: ";
    for (auto it = ms.rbegin(); it != ms.rend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Const iterators
    cout << "Const iteration: ";
    for (auto it = ms.cbegin(); it != ms.cend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    cout << endl;
}

// Comparison operations
void comparisonDemo() {
    cout << "=== Comparison Demo ===" << endl;

    multiset<int> ms1 = {1, 2, 2, 3};
    multiset<int> ms2 = {1, 2, 2, 3};
    multiset<int> ms3 = {1, 2, 3, 3};

    cout << boolalpha;
    cout << "ms1 == ms2: " << (ms1 == ms2) << endl;
    cout << "ms1 == ms3: " << (ms1 == ms3) << endl;
    cout << "ms1 < ms3: " << (ms1 < ms3) << endl;
    cout << endl;
}

// Custom comparator
void customComparatorDemo() {
    cout << "=== Custom Comparator Demo ===" << endl;

    // Descending order
    multiset<int, greater<int>> ms1 = {1, 5, 3, 9, 2, 5, 1};
    cout << "Descending order: ";
    for (const auto& elem : ms1) cout << elem << " ";
    cout << endl;

    // Custom comparator with lambda
    auto cmp = [](int a, int b) { return abs(a) < abs(b); };
    multiset<int, decltype(cmp)> ms2(cmp);
    ms2.insert(-3);
    ms2.insert(1);
    ms2.insert(-5);
    ms2.insert(2);
    ms2.insert(3);  // Same abs as -3

    cout << "\nSorted by absolute value: ";
    for (const auto& elem : ms2) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// multiset vs set comparison
void multisetVsSetDemo() {
    cout << "=== multiset vs set ===" << endl;

    cout << "std::set:" << endl;
    cout << "  - Each element must be unique" << endl;
    cout << "  - insert() returns pair<iterator, bool>" << endl;
    cout << "  - erase(value) removes the single element" << endl;
    cout << "  - count() always returns 0 or 1" << endl;

    cout << "\nstd::multiset:" << endl;
    cout << "  - Allows duplicate elements" << endl;
    cout << "  - insert() always succeeds, returns iterator" << endl;
    cout << "  - erase(value) removes ALL occurrences" << endl;
    cout << "  - count() returns number of occurrences" << endl;

    // Demonstration
    cout << "\nDemonstration:" << endl;
    set<int> s = {1, 2, 2, 3, 3, 3};
    multiset<int> ms = {1, 2, 2, 3, 3, 3};

    cout << "set from {1,2,2,3,3,3}: ";
    for (const auto& elem : s) cout << elem << " ";
    cout << "(size: " << s.size() << ")" << endl;

    cout << "multiset from {1,2,2,3,3,3}: ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << "(size: " << ms.size() << ")" << endl;
    cout << endl;
}

// Use case: Event logging with timestamps
void eventLoggingDemo() {
    cout << "=== Use Case: Event Logging ===" << endl;

    struct Event {
        string description;
        int priority;

        bool operator<(const Event& other) const {
            return priority < other.priority;
        }
    };

    multiset<Event> log;

    log.insert({"System startup", 1});
    log.insert({"User login", 2});
    log.insert({"User login", 2});  // Another login
    log.insert({"Error occurred", 3});
    log.insert({"Warning", 2});
    log.insert({"Debug info", 1});

    cout << "Event log (sorted by priority):" << endl;
    for (const auto& event : log) {
        cout << "  [Priority " << event.priority << "] " << event.description << endl;
    }

    cout << "\nCount of priority 2 events: " << log.count({"", 2}) << endl;
    cout << endl;
}

// Use case: Finding all elements in range
void rangeQueryDemo() {
    cout << "=== Use Case: Range Query ===" << endl;

    multiset<int> ms = {1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 8};

    cout << "Multiset: ";
    for (const auto& elem : ms) cout << elem << " ";
    cout << endl;

    // Find all elements in range [3, 6]
    auto lower = ms.lower_bound(3);
    auto upper = ms.upper_bound(6);

    cout << "\nElements in range [3, 6]: ";
    for (auto it = lower; it != upper; ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Count elements in range
    auto range = ms.equal_range(5);
    cout << "\nElements equal to 5: ";
    for (auto it = range.first; it != range.second; ++it) {
        cout << *it << " ";
    }
    cout << "(" << distance(range.first, range.second) << " occurrences)" << endl;
    cout << endl;
}

// Use case: Bag/Multiset operations
void bagOperationsDemo() {
    cout << "=== Use Case: Bag Operations ===" << endl;

    // Bag union (merge)
    multiset<int> bag1 = {1, 2, 2, 3};
    multiset<int> bag2 = {2, 3, 3, 4};

    multiset<int> bagUnion;
    bagUnion.insert(bag1.begin(), bag1.end());
    bagUnion.insert(bag2.begin(), bag2.end());

    cout << "Bag 1: ";
    for (const auto& elem : bag1) cout << elem << " ";
    cout << endl;

    cout << "Bag 2: ";
    for (const auto& elem : bag2) cout << elem << " ";
    cout << endl;

    cout << "Union: ";
    for (const auto& elem : bagUnion) cout << elem << " ";
    cout << endl;

    // Bag intersection (keep minimum count)
    multiset<int> bagIntersect;
    for (const auto& elem : bag1) {
        auto count1 = bag1.count(elem);
        auto count2 = bag2.count(elem);
        auto minCount = min(count1, count2);
        for (size_t i = 0; i < minCount; ++i) {
            bagIntersect.insert(elem);
        }
    }

    cout << "\nIntersection: ";
    for (const auto& elem : bagIntersect) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Observer operations
void observerDemo() {
    cout << "=== Observer Operations Demo ===" << endl;

    multiset<int> ms = {1, 2, 3, 4, 5};

    // key_comp()
    auto comp = ms.key_comp();
    cout << "key_comp(): Is 1 < 2? " << boolalpha << comp(1, 2) << endl;

    // value_comp() - Same as key_comp()
    auto valComp = ms.value_comp();
    cout << "value_comp(): Is 4 < 5? " << boolalpha << valComp(4, 5) << endl;
    cout << endl;
}

// Performance characteristics
void performanceDemo() {
    cout << "=== Performance Characteristics ===" << endl;

    cout << "Time Complexity:" << endl;
    cout << "  Insert: O(log n)" << endl;
    cout << "  Delete: O(log n)" << endl;
    cout << "  Find: O(log n)" << endl;
    cout << "  Count: O(log n + k) where k is count" << endl;
    cout << "  Traversal: O(n)" << endl;

    cout << "\nImplementation:" << endl;
    cout << "  Red-black tree (same as std::set)" << endl;
    cout << "  Elements always sorted" << endl;
    cout << "  Duplicates allowed" << endl;

    cout << "\nMemory overhead:" << endl;
    cout << "  Same as std::set (red-black tree nodes)" << endl;
    cout << "  Each duplicate element is a separate node" << endl;
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "    std::multiset Complete Demo" << endl;
    cout << "========================================" << endl << endl;

    basicMultisetDemo();
    insertionDemo();
    deletionDemo();
    lookupDemo();
    capacityDemo();
    iteratorDemo();
    comparisonDemo();
    customComparatorDemo();
    multisetVsSetDemo();
    eventLoggingDemo();
    rangeQueryDemo();
    bagOperationsDemo();
    observerDemo();
    performanceDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::multiset characteristics:" << endl;
    cout << "  - Allows duplicate elements" << endl;
    cout << "  - Elements always sorted" << endl;
    cout << "  - O(log n) insert, delete, find" << endl;
    cout << "  - Implemented as red-black tree" << endl;
    cout << "\nKey differences from std::set:" << endl;
    cout << "  - Duplicates allowed" << endl;
    cout << "  - count() can return > 1" << endl;
    cout << "  - erase(value) removes all occurrences" << endl;
    cout << "\nWhen to use multiset:" << endl;
    cout << "  - Need to allow duplicates" << endl;
    cout << "  - Need to count occurrences" << endl;
    cout << "  - Need sorted order" << endl;
    cout << "  - Event logging, bag operations" << endl;

    return 0;
}

/*
Output Summary:
=== Basic std::multiset Demo ===
ms2 (with duplicates {5,2,8,2,1,9,2}): 1 2 2 2 5 8 9
Count of element 2: 3

=== Lookup Demo ===
find(30): found 30 (first occurrence)
count(20): 2 occurrences
count(30): 3 occurrences
equal_range(30): 30 30 30

=== multiset vs set ===
set from {1,2,2,3,3,3}: 1 2 3 (size: 3)
multiset from {1,2,2,3,3,3}: 1 2 2 3 3 3 (size: 6)

========================================
              Summary
========================================
std::multiset characteristics:
  - Allows duplicate elements
  - O(log n) operations
*/
*/
