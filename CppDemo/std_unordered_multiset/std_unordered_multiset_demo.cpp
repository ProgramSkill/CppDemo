// std::unordered_multiset - Hash set allowing duplicates
// Header: <unordered_set>
// Time Complexity (Average): O(1) insert, delete, find
// Time Complexity (Worst): O(n)
// Implementation: Hash table with chaining

#include <iostream>
#include <unordered_set>
#include <string>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::unordered_multiset Demo ===" << endl;
    unordered_multiset<int> ms = {1, 2, 2, 3, 3, 3};
    cout << "Contents (with duplicates): ";
    for (const auto& e : ms) cout << e << " ";
    cout << endl;
    cout << "count(2): " << ms.count(2) << endl;
    cout << "count(3): " << ms.count(3) << endl;
    cout << endl;
}

void insertDemo() {
    cout << "=== Insertion ===" << endl;
    unordered_multiset<int> ms;
    ms.insert(5);
    ms.insert(5);
    ms.insert(5);
    cout << "After inserting 5 three times, count(5): " << ms.count(5) << endl;
    cout << endl;
}

void eraseDemo() {
    cout << "=== Deletion ===" << endl;
    unordered_multiset<int> ms = {1, 2, 2, 3, 3, 3};
    cout << "Before: count(3) = " << ms.count(3) << endl;
    ms.erase(3);  // Erases ALL 3s
    cout << "After erase(3): count(3) = " << ms.count(3) << endl;

    unordered_multiset<int> ms2 = {1, 2, 2, 3};
    auto it = ms2.find(2);
    if (it != ms2.end()) ms2.erase(it);  // Erases one 2
    cout << "After erasing one 2: count(2) = " << ms2.count(2) << endl;
    cout << endl;
}

void lookupDemo() {
    cout << "=== Lookup ===" << endl;
    unordered_multiset<int> ms = {1, 2, 2, 3, 3, 3};
    auto it = ms.find(2);
    if (it != ms.end()) cout << "find(2): " << *it << " (first occurrence)" << endl;
    cout << "count(3): " << ms.count(3) << endl;
    auto range = ms.equal_range(3);
    cout << "equal_range(3): ";
    for (auto i = range.first; i != range.second; ++i) cout << *i << " ";
    cout << endl << endl;
}

void comparisonDemo() {
    cout << "=== unordered_multiset vs unordered_set ===" << endl;
    cout << "unordered_set: unique elements only" << endl;
    cout << "unordered_multiset: allows duplicates" << endl;
    cout << "  - count() can return > 1" << endl;
    cout << "  - erase(value) removes all occurrences" << endl;
    cout << "  - equal_range() useful for finding all copies" << endl;
    cout << endl;
}

void bagDemo() {
    cout << "=== Use Case: Bag/Multiset ===" << endl;
    unordered_multiset<string> bag;
    bag.insert("apple");
    bag.insert("apple");
    bag.insert("banana");
    bag.insert("apple");
    cout << "Bag contents:" << endl;
    for (const auto& item : bag) cout << "  " << item << endl;
    cout << "Quantity of apples: " << bag.count("apple") << endl;
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "std::unordered_multiset Demo\n";
    cout << "========================================\n\n";

    basicDemo();
    insertDemo();
    eraseDemo();
    lookupDemo();
    comparisonDemo();
    bagDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::unordered_multiset:\n";
    cout << "  - Hash-based, allows duplicates\n";
    cout << "  - O(1) average operations\n";
    cout << "  - No ordering\n";
    cout << "  - Use for bags, counting occurrences\n";

    return 0;
}

/*
Output:
count(2): 2
count(3): 3

=== Lookup ===
equal_range(3): 3 3 3
*/
