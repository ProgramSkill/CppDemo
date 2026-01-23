// std::list - Doubly-linked list
// Header: <list>
// Time Complexity:
//   - Insert/Delete anywhere: O(1)
//   - Access by index: O(n)
//   - Front/Back access: O(1)
// Space Complexity: O(N) - each node has prev/next pointers
// Implementation: Doubly-linked list with node-based allocation

#include <iostream>
#include <list>
#include <algorithm>
#include <string>

using namespace std;

// Basic demonstration
void basicListDemo() {
    cout << "=== Basic std::list Demo ===" << endl;

    // Declaration
    list<int> lst1;                                // Empty list
    list<int> lst2(5);                             // 5 elements, value-initialized
    list<int> lst3(5, 100);                       // 5 elements, all 100
    list<int> lst4 = {1, 2, 3, 4, 5};             // Initializer list
    list<int> lst5(lst4.begin(), lst4.end());     // Iterator range

    cout << "lst2 (5 default-initialized): ";
    for (const auto& elem : lst2) cout << elem << " ";
    cout << endl;

    cout << "lst3 (5 elements of 100): ";
    for (const auto& elem : lst3) cout << elem << " ";
    cout << endl;

    cout << "lst4 (initializer list): ";
    for (const auto& elem : lst4) cout << elem << " ";
    cout << endl;

    cout << "lst5 (from iterator range): ";
    for (const auto& elem : lst5) cout << elem << " ";
    cout << endl << endl;
}

// Element access
void elementAccessDemo() {
    cout << "=== Element Access Demo ===" << endl;

    list<int> lst = {10, 20, 30, 40, 50};

    // front() and back() - O(1)
    cout << "lst.front(): " << lst.front() << endl;
    cout << "lst.back(): " << lst.back() << endl;

    // No operator[] or at() - list doesn't support random access!
    cout << "\nNote: list doesn't support operator[] or at()" << endl;
    cout << "To access elements, you must iterate from front/back" << endl;
    cout << endl;
}

// Capacity operations
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    list<int> lst = {1, 2, 3, 4, 5};

    cout << "lst.size(): " << lst.size() << endl;
    cout << "lst.max_size(): " << lst.max_size() << endl;
    cout << "lst.empty(): " << boolalpha << lst.empty() << endl;

    // resize()
    lst.resize(10);
    cout << "After resize(10), size: " << lst.size() << endl;
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    lst.resize(3);
    cout << "After resize(3), size: " << lst.size() << endl;
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Insertion operations (O(1) anywhere!)
void insertionDemo() {
    cout << "=== Insertion Operations Demo ===" << endl;

    list<int> lst = {2, 4, 6};

    cout << "Original: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // push_front() and push_back() - O(1)
    lst.push_front(1);
    lst.push_back(7);
    cout << "After push_front(1) and push_back(7): ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // emplace_front() and emplace_back() - O(1)
    lst.emplace_front(0);
    lst.emplace_back(8);
    cout << "After emplace_front(0) and emplace_back(8): ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // insert() - O(1) at any position!
    auto it = lst.begin();
    advance(it, 3);  // Move to position 3
    lst.insert(it, 99);
    cout << "After insert(99) at position 3: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // insert() with count
    lst.insert(lst.begin(), 3, -1);
    cout << "After insert(3, -1) at begin: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // insert() with range
    list<int> src = {100, 200};
    lst.insert(lst.end(), src.begin(), src.end());
    cout << "After insert range at end: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Deletion operations
void deletionDemo() {
    cout << "=== Deletion Operations Demo ===" << endl;

    list<int> lst = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    cout << "Original: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // pop_front() and pop_back()
    lst.pop_front();
    lst.pop_back();
    cout << "After pop_front() and pop_back(): ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // erase() single element - O(1)
    auto it = lst.begin();
    advance(it, 2);
    lst.erase(it);
    cout << "After erase position 2: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // erase() range - O(k) where k is range length
    auto start = lst.begin();
    auto end = lst.begin();
    advance(start, 1);
    advance(end, 4);
    lst.erase(start, end);
    cout << "After erase range [1, 4): ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    // clear()
    list<int> lst2 = {1, 2, 3};
    lst2.clear();
    cout << "After clear(), size: " << lst2.size() << ", empty: " << boolalpha << lst2.empty() << endl;
    cout << endl;
}

// Special list operations
void specialOperationsDemo() {
    cout << "=== Special List Operations Demo ===" << endl;

    list<int> lst1 = {1, 2, 3, 4, 5};
    list<int> lst2 = {10, 20, 30};

    // splice() - Move elements from one list to another O(1)
    cout << "splice() - Transfer elements:" << endl;
    cout << "lst1: ";
    for (const auto& elem : lst1) cout << elem << " ";
    cout << endl;
    cout << "lst2: ";
    for (const auto& elem : lst2) cout << elem << " ";
    cout << endl;

    auto it = lst1.begin();
    advance(it, 2);
    lst1.splice(it, lst2);  // Move all of lst2 into lst1 at position it
    cout << "After splice lst2 into lst1 at position 2:" << endl;
    cout << "lst1: ";
    for (const auto& elem : lst1) cout << elem << " ";
    cout << endl;
    cout << "lst2 (now empty): size=" << lst2.size() << endl;
    cout << endl;

    // remove() - Remove all elements with specific value
    list<int> lst3 = {1, 2, 3, 2, 4, 2, 5};
    cout << "remove() - Remove all 2's:" << endl;
    cout << "Before: ";
    for (const auto& elem : lst3) cout << elem << " ";
    cout << endl;
    lst3.remove(2);
    cout << "After: ";
    for (const auto& elem : lst3) cout << elem << " ";
    cout << endl;

    // remove_if() - Remove elements matching condition
    lst3.remove_if([](int n) { return n % 2 == 0; });
    cout << "After remove_if(even numbers): ";
    for (const auto& elem : lst3) cout << elem << " ";
    cout << endl;

    // unique() - Remove consecutive duplicates
    list<int> lst4 = {1, 1, 2, 2, 2, 3, 3, 4};
    cout << "\nunique() - Remove consecutive duplicates:" << endl;
    cout << "Before: ";
    for (const auto& elem : lst4) cout << elem << " ";
    cout << endl;
    lst4.unique();
    cout << "After: ";
    for (const auto& elem : lst4) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// List-specific merge and sort operations
void mergeSortOperationsDemo() {
    cout << "=== Merge and Sort Operations Demo ===" << endl;

    // merge() - Merge two sorted lists
    list<int> lst1 = {1, 3, 5, 7};
    list<int> lst2 = {2, 4, 6, 8};

    cout << "merge() - Merge two sorted lists:" << endl;
    cout << "lst1: ";
    for (const auto& elem : lst1) cout << elem << " ";
    cout << endl;
    cout << "lst2: ";
    for (const auto& elem : lst2) cout << elem << " ";
    cout << endl;

    lst1.merge(lst2);
    cout << "After lst1.merge(lst2):" << endl;
    cout << "lst1: ";
    for (const auto& elem : lst1) cout << elem << " ";
    cout << endl;
    cout << "lst2: size=" << lst2.size() << " (empty)" << endl;
    cout << endl;

    // sort() - List-specific sort (better than std::sort for lists)
    list<int> lst3 = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    cout << "sort() - List-specific sort:" << endl;
    cout << "Before: ";
    for (const auto& elem : lst3) cout << elem << " ";
    cout << endl;
    lst3.sort();
    cout << "After sort(): ";
    for (const auto& elem : lst3) cout << elem << " ";
    cout << endl;

    // sort() with custom comparator
    lst3.sort(greater<int>());
    cout << "After sort(greater): ";
    for (const auto& elem : lst3) cout << elem << " ";
    cout << endl;

    // reverse()
    lst3.reverse();
    cout << "After reverse(): ";
    for (const auto& elem : lst3) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    list<int> lst = {1, 2, 3, 4, 5};

    // Forward iteration
    cout << "Forward: ";
    for (auto it = lst.begin(); it != lst.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Reverse iteration
    cout << "Reverse: ";
    for (auto it = lst.rbegin(); it != lst.rend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Note: list iterators are NOT random access!
    // You can't do it + 5, you must use advance()
    auto it = lst.begin();
    advance(it, 3);  // Move iterator 3 positions forward
    cout << "After advance(it, 3): *it = " << *it << endl;

    // distance() to get distance between iterators
    auto start = lst.begin();
    auto end = lst.end();
    cout << "Distance between begin() and end(): " << distance(start, end) << endl;
    cout << endl;
}

// Comparison operations
void comparisonDemo() {
    cout << "=== Comparison Demo ===" << endl;

    list<int> lst1 = {1, 2, 3};
    list<int> lst2 = {1, 2, 3};
    list<int> lst3 = {1, 2, 4};

    cout << "lst1: {1, 2, 3}" << endl;
    cout << "lst2: {1, 2, 3}" << endl;
    cout << "lst3: {1, 2, 4}" << endl;

    cout << boolalpha;
    cout << "lst1 == lst2: " << (lst1 == lst2) << endl;
    cout << "lst1 < lst3: " << (lst1 < lst3) << endl;
    cout << endl;
}

// Use case: Implementing LRU Cache
void lruCacheDemo() {
    cout << "=== Use Case: LRU Cache ===" << endl;

    // Simulate LRU cache using list
    list<int> cache;
    const size_t CACHE_SIZE = 3;

    auto access = [&](int value) {
        cout << "Access " << value << ": ";

        // Check if value exists
        auto it = find(cache.begin(), cache.end(), value);
        if (it != cache.end()) {
            // Move to front (most recently used)
            cache.splice(cache.begin(), cache, it);
            cout << "found, moved to front. Cache: ";
        } else {
            // Add new value
            cache.push_front(value);
            cout << "added. Cache: ";

            // Remove least recently used if full
            if (cache.size() > CACHE_SIZE) {
                cache.pop_back();
            }
        }

        for (const auto& elem : cache) cout << elem << " ";
        cout << endl;
    };

    access(1);
    access(2);
    access(3);
    access(2);  // Already in cache
    access(4);  // Evicts 1 (least recently used)
    access(5);  // Evicts 3
    cout << endl;
}

// Performance characteristics
void performanceDemo() {
    cout << "=== Performance Characteristics ===" << endl;

    list<int> lst;

    // Insertion anywhere is O(1)
    for (int i = 0; i < 5; ++i) {
        lst.push_back(i);
    }

    cout << "List contents: ";
    for (const auto& elem : lst) cout << elem << " ";
    cout << endl;

    cout << "\nPerformance characteristics:" << endl;
    cout << "  Insert/Delete anywhere: O(1)" << endl;
    cout << "  Access front/back: O(1)" << endl;
    cout << "  Access middle: O(n) - must traverse" << endl;
    cout << "  No random access (no operator[])" << endl;
    cout << "  Excellent cache locality: No (scattered memory)" << endl;
    cout << "  Per-node overhead: 2 pointers (prev, next)" << endl;

    cout << "\nMemory overhead per element:" << endl;
    cout << "  For int (4 bytes): ~12 bytes per node" << endl;
    cout << "  - Data: 4 bytes" << endl;
    cout << "  - Next pointer: 4 bytes" << endl;
    cout << "  - Prev pointer: 4 bytes" << endl;
    cout << endl;
}

// Emplace vs Insert
void emplaceDemo() {
    cout << "=== Emplace vs Insert Demo ===" << endl;

    struct Person {
        string name;
        int age;
        Person(string n, int a) : name(n), age(a) {
            cout << "  Person constructed: " << name << endl;
        }
    };

    list<Person> people;

    cout << "emplace_front():" << endl;
    people.emplace_front("Alice", 25);

    cout << "\npush_front() with temporary:" << endl;
    people.push_front(Person("Bob", 30));

    cout << "\nemplace_back():" << endl;
    people.emplace_back("Charlie", 35);

    cout << "\nFinal list:" << endl;
    for (const auto& p : people) {
        cout << "  " << p.name << ", " << p.age << endl;
    }
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "    std::list Complete Demonstration" << endl;
    cout << "========================================" << endl << endl;

    basicListDemo();
    elementAccessDemo();
    capacityDemo();
    insertionDemo();
    deletionDemo();
    specialOperationsDemo();
    mergeSortOperationsDemo();
    iteratorDemo();
    comparisonDemo();
    lruCacheDemo();
    performanceDemo();
    emplaceDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::list advantages:" << endl;
    cout << "  - O(1) insertion/deletion at any position" << endl;
    cout << "  - O(1) insertion at both ends" << endl;
    cout << "  - Stable iterators (never invalidated)" << endl;
    cout << "  - Special operations: splice, merge, unique" << endl;
    cout << "\nWhen to use list:" << endl;
    cout << "  - Frequent insertions/deletions in middle" << endl;
    cout << "  - Need stable iterators" << endl;
    cout << "  - Implementing queues or LRU caches" << endl;
    cout << "  - When random access is not needed" << endl;
    cout << "\nWhen NOT to use list:" << endl;
    cout << "  - Need random access (use vector/deque)" << endl;
    cout << "  - Care about cache locality (use vector)" << endl;
    cout << "  - Small data types (pointer overhead is high)" << endl;

    return 0;
}

/*
Output Summary:
========================================
    std::list Complete Demonstration
========================================

=== Special List Operations Demo ===
splice() - Transfer elements:
lst1: 1 2 3 4 5
lst2: 10 20 30
After splice lst2 into lst1 at position 2:
lst1: 1 2 10 20 30 3 4 5
lst2 (now empty): size=0

remove() - Remove all 2's:
Before: 1 2 3 2 4 2 5
After: 1 3 4 5

=== Merge and Sort Operations Demo ===
merge() - Merge two sorted lists:
lst1: 1 3 5 7
lst2: 2 4 6 8
After lst1.merge(lst2):
lst1: 1 2 3 4 5 6 7 8

=== Use Case: LRU Cache ===
Access 1: added. Cache: 1
Access 2: added. Cache: 2 1
Access 3: added. Cache: 3 2 1
Access 2: found, moved to front. Cache: 2 3 1
Access 4: added. Cache: 4 2 3
Access 5: added. Cache: 5 4 2

========================================
              Summary
========================================
std::list advantages:
  - O(1) insertion/deletion at any position
  - Stable iterators (never invalidated)
  - Special operations: splice, merge, unique
*/
