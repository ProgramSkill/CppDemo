// std::set - Sorted unique element container
// Header: <set>
// Time Complexity:
//   - Insert: O(log n)
//   - Delete: O(log n)
//   - Find: O(log n)
//   - Traversal: O(n)
// Space Complexity: O(N)
// Implementation: Red-black tree (balanced binary search tree)

#include <iostream>
#include <set>
#include <string>
#include <algorithm>

using namespace std;

// Basic demonstration
void basicSetDemo() {
    cout << "=== Basic std::set Demo ===" << endl;

    // Declaration
    set<int> s1;                                      // Empty set
    set<int> s2 = {5, 2, 8, 1, 9, 3};               // Initializer list
    set<int> s3(s2.begin(), s2.end());              // Iterator range
    set<int> s4(s2);                                // Copy constructor

    // Note: Elements are automatically sorted!
    cout << "s2 (initializer list {5,2,8,1,9,3}): ";
    for (const auto& elem : s2) cout << elem << " ";
    cout << endl;

    cout << "s3 (from iterator range): ";
    for (const auto& elem : s3) cout << elem << " ";
    cout << endl;

    // Duplicates are automatically removed!
    set<int> s5 = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    cout << "s5 (with duplicates): ";
    for (const auto& elem : s5) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Insertion operations
void insertionDemo() {
    cout << "=== Insertion Demo ===" << endl;

    set<int> s;

    // insert() - Returns pair<iterator, bool>
    auto result1 = s.insert(5);
    cout << "Insert 5: success=" << result1.second << ", value=" << *result1.first << endl;

    auto result2 = s.insert(5);  // Duplicate!
    cout << "Insert 5 again: success=" << result2.second << endl;

    s.insert(3);
    s.insert(7);
    s.insert(1);
    s.insert(9);

    cout << "After inserts {5,5,3,7,1,9}: ";
    for (const auto& elem : s) cout << elem << " ";
    cout << endl;

    // insert() with hint
    auto it = s.begin();
    advance(it, 2);
    s.insert(it, 4);  // Hint at position 2

    cout << "After insert with hint: ";
    for (const auto& elem : s) cout << elem << " ";
    cout << endl;

    // emplace()
    s.emplace(6);
    cout << "After emplace(6): ";
    for (const auto& elem : s) cout << elem << " ";
    cout << endl;

    // insert() with range
    set<int> s2 = {100, 200, 300};
    s.insert(s2.begin(), s2.end());
    cout << "After insert range: ";
    for (const auto& elem : s) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Deletion operations
void deletionDemo() {
    cout << "=== Deletion Demo ===" << endl;

    set<int> s = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    cout << "Original: ";
    for (const auto& elem : s) cout << elem << " ";
    cout << endl;

    // erase() by value
    size_t count = s.erase(5);
    cout << "Erase 5: removed " << count << " element(s)" << endl;
    cout << "After erase(5): ";
    for (const auto& elem : s) cout << elem << " ";
    cout << endl;

    // erase() by iterator
    auto it = s.find(3);
    if (it != s.end()) {
        s.erase(it);
        cout << "Erase iterator at 3: ";
        for (const auto& elem : s) cout << elem << " ";
        cout << endl;
    }

    // erase() by range
    auto start = s.begin();
    auto end = s.begin();
    advance(start, 2);
    advance(end, 5);
    s.erase(start, end);
    cout << "After erase range [2, 5): ";
    for (const auto& elem : s) cout << elem << " ";
    cout << endl;

    // clear()
    set<int> s2 = {1, 2, 3};
    s2.clear();
    cout << "After clear(), size: " << s2.size() << ", empty: " << boolalpha << s2.empty() << endl;
    cout << endl;
}

// Lookup operations
void lookupDemo() {
    cout << "=== Lookup Demo ===" << endl;

    set<int> s = {10, 20, 30, 40, 50};

    // find() - O(log n)
    auto it = s.find(30);
    if (it != s.end()) {
        cout << "find(30): found " << *it << endl;
    }

    it = s.find(35);
    if (it == s.end()) {
        cout << "find(35): not found" << endl;
    }

    // count() - Returns 0 or 1 (no duplicates!)
    cout << "count(20): " << s.count(20) << endl;
    cout << "count(25): " << s.count(25) << endl;

    // contains() - C++20
    cout << "contains(40): " << boolalpha << s.contains(40) << endl;
    cout << "contains(45): " << boolalpha << s.contains(45) << endl;

    // lower_bound() - First element >= value
    auto lb = s.lower_bound(25);
    cout << "lower_bound(25): " << *lb << endl;

    // upper_bound() - First element > value
    auto ub = s.upper_bound(30);
    cout << "upper_bound(30): " << *ub << endl;

    // equal_range() - Pair of lower_bound and upper_bound
    auto range = s.equal_range(30);
    cout << "equal_range(30): [" << *range.first << ", " << *range.second << ")" << endl;
    cout << endl;
}

// Capacity operations
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    set<int> s = {1, 2, 3, 4, 5};

    cout << "s.size(): " << s.size() << endl;
    cout << "s.max_size(): " << s.max_size() << endl;
    cout << "s.empty(): " << boolalpha << s.empty() << endl;

    set<int> emptySet;
    cout << "emptySet.empty(): " << boolalpha << emptySet.empty() << endl;
    cout << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    set<int> s = {1, 2, 3, 4, 5};

    // Forward iteration (sorted order!)
    cout << "Forward (sorted): ";
    for (auto it = s.begin(); it != s.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Reverse iteration
    cout << "Reverse: ";
    for (auto it = s.rbegin(); it != s.rend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Const iterators
    cout << "Const iteration: ";
    for (auto it = s.cbegin(); it != s.cend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    cout << endl;
}

// Comparison operations
void comparisonDemo() {
    cout << "=== Comparison Demo ===" << endl;

    set<int> s1 = {1, 2, 3};
    set<int> s2 = {1, 2, 3};
    set<int> s3 = {1, 2, 4};

    cout << boolalpha;
    cout << "s1 == s2: " << (s1 == s2) << endl;
    cout << "s1 < s3: " << (s1 < s3) << endl;

    // Lexicographic comparison
    set<int> s4 = {1, 2};
    cout << "{1,2,3} > {1,2}: " << (s1 > s4) << endl;
    cout << endl;
}

// Custom comparator
void customComparatorDemo() {
    cout << "=== Custom Comparator Demo ===" << endl;

    // Descending order
    set<int, greater<int>> s1 = {1, 5, 3, 9, 2};
    cout << "Descending order: ";
    for (const auto& elem : s1) cout << elem << " ";
    cout << endl;

    // Custom comparator with lambda
    auto cmp = [](int a, int b) { return abs(a) < abs(b); };
    set<int, decltype(cmp)> s2(cmp);
    s2.insert(-3);
    s2.insert(1);
    s2.insert(-5);
    s2.insert(2);

    cout << "Sorted by absolute value: ";
    for (const auto& elem : s2) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Set with custom objects
void customObjectDemo() {
    cout << "=== Custom Object Demo ===" << endl;

    struct Person {
        string name;
        int age;

        bool operator<(const Person& other) const {
            return age < other.age;  // Sort by age
        }
    };

    set<Person> people;
    people.insert({"Alice", 30});
    people.insert({"Bob", 25});
    people.insert({"Charlie", 35});

    cout << "People sorted by age:" << endl;
    for (const auto& person : people) {
        cout << "  " << person.name << " (" << person.age << ")" << endl;
    }
    cout << endl;
}

// Set operations
void setOperationsDemo() {
    cout << "=== Set Operations Demo ===" << endl;

    set<int> set1 = {1, 2, 3, 4, 5};
    set<int> set2 = {4, 5, 6, 7, 8};

    // Union
    set<int> unionSet;
    set_union(set1.begin(), set1.end(),
              set2.begin(), set2.end(),
              inserter(unionSet, unionSet.begin()));
    cout << "Union: ";
    for (const auto& elem : unionSet) cout << elem << " ";
    cout << endl;

    // Intersection
    set<int> intersectSet;
    set_intersection(set1.begin(), set1.end(),
                     set2.begin(), set2.end(),
                     inserter(intersectSet, intersectSet.begin()));
    cout << "Intersection: ";
    for (const auto& elem : intersectSet) cout << elem << " ";
    cout << endl;

    // Difference
    set<int> diffSet;
    set_difference(set1.begin(), set1.end(),
                   set2.begin(), set2.end(),
                   inserter(diffSet, diffSet.begin()));
    cout << "Difference (set1 - set2): ";
    for (const auto& elem : diffSet) cout << elem << " ";
    cout << endl;

    // Symmetric difference
    set<int> symDiffSet;
    set_symmetric_difference(set1.begin(), set1.end(),
                             set2.begin(), set2.end(),
                             inserter(symDiffSet, symDiffSet.begin()));
    cout << "Symmetric Difference: ";
    for (const auto& elem : symDiffSet) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Use case: Removing duplicates while maintaining order
void removeDuplicatesDemo() {
    cout << "=== Use Case: Remove Duplicates ===" << endl;

    vector<int> vec = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5};

    cout << "Original vector: ";
    for (const auto& elem : vec) cout << elem << " ";
    cout << endl;

    // Remove duplicates using set
    set<int> s(vec.begin(), vec.end());

    cout << "After removing duplicates: ";
    for (const auto& elem : s) cout << elem << " ";
    cout << endl;

    // Convert back to vector
    vector<int> uniqueVec(s.begin(), s.end());
    cout << "Back to vector: ";
    for (const auto& elem : uniqueVec) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Observer operations
void observerDemo() {
    cout << "=== Observer Operations Demo ===" << endl;

    set<int> s = {1, 2, 3, 4, 5};

    // key_comp() - Get the comparator
    auto comp = s.key_comp();
    cout << "key_comp(): Is 1 < 2? " << boolalpha << comp(1, 2) << endl;
    cout << "key_comp(): Is 3 < 2? " << boolalpha << comp(3, 2) << endl;

    // value_comp() - Same as key_comp() for set
    auto valComp = s.value_comp();
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
    cout << "  Traversal: O(n)" << endl;
    cout << "  Lower/Upper bound: O(log n)" << endl;

    cout << "\nImplementation:" << endl;
    cout << "  Red-black tree (self-balancing BST)" << endl;
    cout << "  Guaranteed O(log n) operations" << endl;
    cout << "  Elements always sorted" << endl;

    cout << "\nMemory overhead:" << endl;
    cout << "  Each node stores:" << endl;
    cout << "    - Value" << endl;
    cout << "    - Left child pointer" << endl;
    cout << "    - Right child pointer" << endl;
    cout << "    - Parent pointer" << endl;
    cout << "    - Color (red/black)" << endl;
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "      std::set Complete Demonstration" << endl;
    cout << "========================================" << endl << endl;

    basicSetDemo();
    insertionDemo();
    deletionDemo();
    lookupDemo();
    capacityDemo();
    iteratorDemo();
    comparisonDemo();
    customComparatorDemo();
    customObjectDemo();
    setOperationsDemo();
    removeDuplicatesDemo();
    observerDemo();
    performanceDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::set characteristics:" << endl;
    cout << "  - Stores unique elements only" << endl;
    cout << "  - Elements always sorted" << endl;
    cout << "  - O(log n) insert, delete, find" << endl;
    cout << "  - Implemented as red-black tree" << endl;
    cout << "\nWhen to use set:" << endl;
    cout << "  - Need unique elements" << endl;
    cout << "  - Need sorted order" << endl;
    cout << "  - Frequent lookups" << endl;
    cout << "  - Set operations (union, intersection)" << endl;
    cout << "\nAlternatives:" << endl;
    cout << "  - unordered_set: O(1) average, no sorting" << endl;
    cout << "  - multiset: Allows duplicates" << endl;
    cout << "  - vector + sort: When you need random access" << endl;

    return 0;
}

/*
Output Summary:
=== Basic std::set Demo ===
s2 (initializer list {5,2,8,1,9,3}): 1 2 3 5 8 9
s5 (with duplicates): 1 2 3 4

=== Insertion Demo ===
Insert 5: success=1, value=5
Insert 5 again: success=0

=== Set Operations Demo ===
Union: 1 2 3 4 5 6 7 8
Intersection: 4 5
Difference (set1 - set2): 1 2 3
Symmetric Difference: 1 2 3 6 7 8

========================================
              Summary
========================================
std::set characteristics:
  - Stores unique elements only
  - Elements always sorted
  - O(log n) insert, delete, find
*/
*/
