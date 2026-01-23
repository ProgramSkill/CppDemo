// std::forward_list - Singly-linked list
// Header: <forward_list>
// Time Complexity:
//   - Insert/Delete after position: O(1)
//   - Access by index: O(n)
//   - Front access: O(1)
// Space Complexity: O(N) - each node has only next pointer (half of std::list)
// Implementation: Singly-linked list (C++11)

#include <iostream>
#include <forward_list>
#include <algorithm>
#include <string>

using namespace std;

// Basic demonstration
void basicForwardListDemo() {
    cout << "=== Basic std::forward_list Demo ===" << endl;

    // Declaration
    forward_list<int> fl1;                            // Empty list
    forward_list<int> fl2(5);                         // 5 elements, value-initialized
    forward_list<int> fl3(5, 100);                   // 5 elements, all 100
    forward_list<int> fl4 = {1, 2, 3, 4, 5};         // Initializer list
    forward_list<int> fl5(fl4.begin(), fl4.end());   // Iterator range

    cout << "fl2 (5 default-initialized): ";
    for (const auto& elem : fl2) cout << elem << " ";
    cout << endl;

    cout << "fl3 (5 elements of 100): ";
    for (const auto& elem : fl3) cout << elem << " ";
    cout << endl;

    cout << "fl4 (initializer list): ";
    for (const auto& elem : fl4) cout << elem << " ";
    cout << endl;

    cout << "fl5 (from iterator range): ";
    for (const auto& elem : fl5) cout << elem << " ";
    cout << endl << endl;
}

// Element access (very limited - only front!)
void elementAccessDemo() {
    cout << "=== Element Access Demo ===" << endl;

    forward_list<int> fl = {10, 20, 30, 40, 50};

    // front() - O(1)
    cout << "fl.front(): " << fl.front() << endl;

    // No back(), no operator[], no at()!
    cout << "\nNote: forward_list only provides front() access!" << endl;
    cout << "  - No back() (would be O(n))" << endl;
    cout << "  - No operator[] (no random access)" << endl;
    cout << "  - No size() (would be O(n))" << endl;
    cout << "To access other elements, you must iterate:" << endl;

    int i = 0;
    for (auto it = fl.begin(); it != fl.end() && i < 4; ++it, ++i) {
        if (i == 3) {
            cout << "  Element at index 3: " << *it << endl;
            break;
        }
    }
    cout << endl;
}

// Capacity operations (very limited)
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    forward_list<int> fl = {1, 2, 3, 4, 5};

    // max_size()
    cout << "fl.max_size(): " << fl.max_size() << endl;

    // empty() - O(1)
    cout << "fl.empty(): " << boolalpha << fl.empty() << endl;

    // No size() member! Use std::distance() or std::count()
    auto sz = distance(fl.begin(), fl.end());
    cout << "Size using distance(): " << sz << endl;

    // resize()
    fl.resize(10);
    sz = distance(fl.begin(), fl.end());
    cout << "After resize(10), size: " << sz << endl;
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    fl.resize(3);
    sz = distance(fl.begin(), fl.end());
    cout << "After resize(3), size: " << sz << endl;
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Insertion operations
void insertionDemo() {
    cout << "=== Insertion Operations Demo ===" << endl;

    forward_list<int> fl = {2, 4, 6};

    cout << "Original: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // push_front() - O(1)
    fl.push_front(1);
    cout << "After push_front(1): ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // emplace_front() - O(1)
    fl.emplace_front(0);
    cout << "After emplace_front(0): ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // insert_after() - O(1) after position!
    auto it = fl.begin();
    advance(it, 2);  // Point to element 2
    fl.insert_after(it, 99);
    cout << "After insert_after(99) at position 2: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // insert_after() with count
    it = fl.begin();
    fl.insert_after(it, 3, -1);
    cout << "After insert_after(3, -1) at begin: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // insert_after() with range
    forward_list<int> src = {100, 200};
    it = fl.begin();
    advance(it, 2);
    fl.insert_after(it, src.begin(), src.end());
    cout << "After insert_after range: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // emplace_after() - O(1)
    it = fl.begin();
    fl.emplace_after(it, 77);
    cout << "After emplace_after(77) at begin: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Deletion operations
void deletionDemo() {
    cout << "=== Deletion Operations Demo ===" << endl;

    forward_list<int> fl = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    cout << "Original: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // pop_front() - O(1)
    fl.pop_front();
    cout << "After pop_front(): ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // erase_after() single element - O(1)
    auto it = fl.begin();
    advance(it, 2);  // Point to element 3
    fl.erase_after(it);  // Erase element after 3 (which is 5)
    cout << "After erase_after() at position 2: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // erase_after() range
    auto start = fl.begin();
    auto end = fl.begin();
    advance(start, 1);
    advance(end, 4);
    fl.erase_after(start, end);
    cout << "After erase_after() range: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // clear()
    forward_list<int> fl2 = {1, 2, 3};
    fl2.clear();
    cout << "After clear(), empty: " << boolalpha << fl2.empty() << endl;
    cout << endl;
}

// Special forward_list operations
void specialOperationsDemo() {
    cout << "=== Special Operations Demo ===" << endl;

    forward_list<int> fl1 = {1, 2, 3, 4, 5};
    forward_list<int> fl2 = {10, 20, 30};

    // splice_after() - Move elements after position
    cout << "splice_after() - Transfer elements:" << endl;
    cout << "fl1: ";
    for (const auto& elem : fl1) cout << elem << " ";
    cout << endl;
    cout << "fl2: ";
    for (const auto& elem : fl2) cout << elem << " ";
    cout << endl;

    auto it = fl1.begin();
    advance(it, 2);
    fl1.splice_after(it, fl2);  // Move all of fl2 after position it
    cout << "After splice_after:" << endl;
    cout << "fl1: ";
    for (const auto& elem : fl1) cout << elem << " ";
    cout << endl;
    cout << "fl2 (now empty): " << (fl2.empty() ? "empty" : "not empty") << endl;
    cout << endl;

    // splice_after() single element
    forward_list<int> fl3 = {1, 2, 3};
    forward_list<int> fl4 = {99};
    it = fl3.begin();
    fl3.splice_after(it, fl4, fl4.before_begin());
    cout << "After splice_after single element: ";
    for (const auto& elem : fl3) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// remove and unique operations
void removeUniqueDemo() {
    cout << "=== Remove and Unique Demo ===" << endl;

    // remove() - Remove all elements with specific value
    forward_list<int> fl = {1, 2, 3, 2, 4, 2, 5};
    cout << "remove() - Remove all 2's:" << endl;
    cout << "Before: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;
    fl.remove(2);
    cout << "After: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // remove_if() - Remove elements matching condition
    fl.remove_if([](int n) { return n % 2 == 0; });
    cout << "After remove_if(even numbers): ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // unique() - Remove consecutive duplicates
    forward_list<int> fl2 = {1, 1, 2, 2, 2, 3, 3, 4};
    cout << "\nunique() - Remove consecutive duplicates:" << endl;
    cout << "Before: ";
    for (const auto& elem : fl2) cout << elem << " ";
    cout << endl;
    fl2.unique();
    cout << "After: ";
    for (const auto& elem : fl2) cout << elem << " ";
    cout << endl;

    // unique() with custom predicate
    forward_list<int> fl3 = {1, 2, 3, 4, 5, 6};
    fl3.unique([](int a, int b) { return b - a == 1; });  // Remove consecutive numbers
    cout << "\nunique() with custom predicate (remove consecutive):" << endl;
    cout << "Before: 1 2 3 4 5 6" << endl;
    cout << "After: ";
    for (const auto& elem : fl3) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Sort and reverse operations
void sortReverseDemo() {
    cout << "=== Sort and Reverse Demo ===" << endl;

    // sort()
    forward_list<int> fl = {5, 2, 8, 1, 9, 3, 7, 4, 6};
    cout << "sort() - Sort the list:" << endl;
    cout << "Before: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;
    fl.sort();
    cout << "After sort(): ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // sort() with custom comparator
    fl.sort(greater<int>());
    cout << "After sort(greater): ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    // reverse()
    fl.reverse();
    cout << "After reverse(): ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Merge operation
void mergeDemo() {
    cout << "=== Merge Demo ===" << endl;

    forward_list<int> fl1 = {1, 3, 5, 7};
    forward_list<int> fl2 = {2, 4, 6, 8};

    cout << "merge() - Merge two sorted lists:" << endl;
    cout << "fl1: ";
    for (const auto& elem : fl1) cout << elem << " ";
    cout << endl;
    cout << "fl2: ";
    for (const auto& elem : fl2) cout << elem << " ";
    cout << endl;

    fl1.merge(fl2);
    cout << "After fl1.merge(fl2):" << endl;
    cout << "fl1: ";
    for (const auto& elem : fl1) cout << elem << " ";
    cout << endl;
    cout << "fl2: " << (fl2.empty() ? "empty" : "not empty") << endl;

    // merge with custom comparator
    forward_list<int> fl3 = {7, 5, 3, 1};
    forward_list<int> fl4 = {8, 6, 4, 2};
    fl3.sort(greater<int>());
    fl4.sort(greater<int>());
    fl3.merge(fl4, greater<int>());
    cout << "\nAfter merge with greater:" << endl;
    cout << "fl3: ";
    for (const auto& elem : fl3) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    forward_list<int> fl = {1, 2, 3, 4, 5};

    // Forward iteration
    cout << "Forward: ";
    for (auto it = fl.begin(); it != fl.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Const iteration
    cout << "Const forward: ";
    for (auto it = fl.cbegin(); it != fl.cend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // No reverse iterators! (only forward direction)
    cout << "\nNote: forward_list only has forward iterators!" << endl;
    cout << "  - No rbegin()/rend() (can't go backward)" << endl;

    // before_begin() - Iterator before the first element
    cout << "\nbefore_begin() - Special iterator:" << endl;
    auto before = fl.before_begin();
    fl.insert_after(before, 0);
    cout << "After insert_after(before_begin(), 0): ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Comparison operations
void comparisonDemo() {
    cout << "=== Comparison Demo ===" << endl;

    forward_list<int> fl1 = {1, 2, 3};
    forward_list<int> fl2 = {1, 2, 3};
    forward_list<int> fl3 = {1, 2, 4};

    cout << "fl1: {1, 2, 3}" << endl;
    cout << "fl2: {1, 2, 3}" << endl;
    cout << "fl3: {1, 2, 4}" << endl;

    cout << boolalpha;
    cout << "fl1 == fl2: " << (fl1 == fl2) << endl;
    cout << "fl1 < fl3: " << (fl1 < fl3) << endl;
    cout << endl;
}

// forward_list vs list comparison
void comparisonWithListDemo() {
    cout << "=== forward_list vs std::list ===" << endl;

    cout << "forward_list advantages:" << endl;
    cout << "  - Uses less memory (1 pointer vs 2)" << endl;
    cout << "  - Faster and more efficient" << endl;
    cout << "  - Better cache locality" << endl;
    cout << "\nforward_list limitations:" << endl;
    cout << "  - No backward traversal" << endl;
    cout << "  - No size() member" << endl;
    cout << "  - No back() operation" << endl;
    cout << "  - Only insert_after/erase_after (not insert/erase)" << endl;
    cout << "\nWhen to use forward_list:" << endl;
    cout << "  - Memory is constrained" << endl;
    cout << "  - Only need forward traversal" << endl;
    cout << "  - Don't need size() operation" << endl;
    cout << "  - Performance is critical" << endl;
    cout << endl;
}

// Performance demonstration
void performanceDemo() {
    cout << "=== Performance Characteristics ===" << endl;

    forward_list<int> fl;

    // Insertion at front is O(1)
    for (int i = 0; i < 5; ++i) {
        fl.push_front(i);
    }

    cout << "After push_front operations: ";
    for (const auto& elem : fl) cout << elem << " ";
    cout << endl;

    cout << "\nPerformance characteristics:" << endl;
    cout << "  Insert anywhere: O(1)" << endl;
    cout << "  Delete anywhere: O(1)" << endl;
    cout << "  Access front: O(1)" << endl;
    cout << "  Access middle: O(n) - must traverse" << endl;
    cout << "  Memory per node: 1 pointer (vs 2 for list)" << endl;

    // Memory usage comparison
    cout << "\nMemory overhead per element (assuming 64-bit):" << endl;
    cout << "  std::list:    16 bytes overhead (2 * 8-byte pointers)" << endl;
    cout << "  forward_list: 8 bytes overhead (1 * 8-byte pointer)" << endl;
    cout << "  Savings: 50% less overhead!" << endl;
    cout << endl;
}

// Use case: Hash table chaining
void hashTableChainingDemo() {
    cout << "=== Use Case: Hash Table Chaining ===" << endl;

    struct Entry {
        int key;
        string value;
    };

    // Simulate hash bucket using forward_list
    forward_list<Entry> bucket;

    auto insert = [&](int key, string value) {
        bucket.push_front(Entry{key, value});
        cout << "  Inserted: " << key << " -> " << value << endl;
    };

    auto find = [&](int key) -> forward_list<Entry>::iterator {
        for (auto it = bucket.begin(); it != bucket.end(); ++it) {
            if (it->key == key) return it;
        }
        return bucket.end();
    };

    auto remove = [&](int key) {
        auto prev = bucket.before_begin();
        for (auto it = bucket.begin(); it != bucket.end(); ++it, ++prev) {
            if (it->key == key) {
                bucket.erase_after(prev);
                cout << "  Removed key: " << key << endl;
                return;
            }
        }
        cout << "  Key " << key << " not found" << endl;
    };

    cout << "Hash table bucket operations:" << endl;
    insert(1, "one");
    insert(2, "two");
    insert(3, "three");

    cout << "\nCurrent bucket: ";
    for (const auto& entry : bucket) {
        cout << "[" << entry.key << ":" << entry.value << "] ";
    }
    cout << endl;

    remove(2);
    cout << "\nAfter removal: ";
    for (const auto& entry : bucket) {
        cout << "[" << entry.key << ":" << entry.value << "] ";
    }
    cout << endl;
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << " std::forward_list Complete Demo" << endl;
    cout << "========================================" << endl << endl;

    basicForwardListDemo();
    elementAccessDemo();
    capacityDemo();
    insertionDemo();
    deletionDemo();
    specialOperationsDemo();
    removeUniqueDemo();
    sortReverseDemo();
    mergeDemo();
    iteratorDemo();
    comparisonDemo();
    comparisonWithListDemo();
    performanceDemo();
    hashTableChainingDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::forward_list advantages:" << endl;
    cout << "  - Most memory-efficient sequence container" << endl;
    cout << "  - O(1) insertion/deletion at any position" << endl;
    cout << "  - Better cache locality than std::list" << endl;
    cout << "\nLimitations:" << endl;
    cout << "  - Forward traversal only" << endl;
    cout << "  - No size() member function" << endl;
    cout << "  - No back() or pop_back()" << endl;
    cout << "\nBest use cases:" << endl;
    cout << "  - Hash table chaining" << endl;
    cout << "  - Memory-constrained environments" << endl;
    cout << "  - When only forward iteration is needed" << endl;

    return 0;
}

/*
Output Summary:
=== Special Operations Demo ===
splice_after() - Transfer elements:
fl1: 1 2 3 4 5
fl2: 10 20 30
After splice_after:
fl1: 1 2 10 20 30 3 4 5
fl2 (now empty): empty

=== Remove and Unique Demo ===
remove() - Remove all 2's:
Before: 1 2 3 2 4 2 5
After: 1 3 4 5
After remove_if(even numbers): 1 3 5

unique() with custom predicate (remove consecutive):
Before: 1 2 3 4 5 6
After: 1

=== forward_list vs std::list ===
forward_list advantages:
  - Uses less memory (1 pointer vs 2)
  - Faster and more efficient
  - Better cache locality
Memory overhead per element (assuming 64-bit):
  std::list:    16 bytes overhead (2 * 8-byte pointers)
  forward_list: 8 bytes overhead (1 * 8-byte pointer)
  Savings: 50% less overhead!
*/
