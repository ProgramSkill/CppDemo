// std::unordered_multimap - Hash map allowing duplicate keys
// Header: <unordered_map>
// Time Complexity (Average): O(1) insert, delete, find
// Time Complexity (Worst): O(n)
// Implementation: Hash table with chaining

#include <iostream>
#include <unordered_map>
#include <string>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::unordered_multimap Demo ===" << endl;
    unordered_multimap<string, int> mm = {{"apple", 5}, {"apple", 2}, {"banana", 3}};
    cout << "Contents (duplicate keys allowed):" << endl;
    for (const auto& [k, v] : mm) cout << "  " << k << " -> " << v << endl;
    cout << "count(\"apple\"): " << mm.count("apple") << endl;
    cout << endl;
}

void insertDemo() {
    cout << "=== Insertion ===" << endl;
    unordered_multimap<string, int> mm;
    mm.insert({"apple", 5});
    mm.insert({"apple", 2});
    mm.insert({"apple", 10});
    cout << "After 3 inserts of \"apple\": count = " << mm.count("apple") << endl;
    cout << endl;
}

void eraseDemo() {
    cout << "=== Deletion ===" << endl;
    unordered_multimap<string, int> mm = {{"a", 1}, {"a", 2}, {"b", 3}};
    cout << "Before: count(\"a\") = " << mm.count("a") << endl;
    mm.erase("a");  // Erases ALL "a" entries
    cout << "After erase(\"a\"): count(\"a\") = " << mm.count("a") << endl;

    unordered_multimap<int, string> mm2 = {{1, "a"}, {1, "b"}, {2, "c"}};
    auto it = mm2.find(1);
    if (it != mm2.end()) mm2.erase(it);  // Erases one entry
    cout << "After erasing one entry with key 1: count(1) = " << mm2.count(1) << endl;
    cout << endl;
}

void lookupDemo() {
    cout << "=== Lookup ===" << endl;
    unordered_multimap<int, string> mm = {{1, "one-a"}, {1, "one-b"}, {2, "two"}};
    auto it = mm.find(1);
    if (it != mm.end()) cout << "find(1): " << it->first << " -> " << it->second << endl;
    cout << "count(1): " << mm.count(1) << endl;

    auto range = mm.equal_range(1);
    cout << "equal_range(1):" << endl;
    for (auto i = range.first; i != range.second; ++i) {
        cout << "  " << i->first << " -> " << i->second << endl;
    }
    cout << endl;
}

void comparisonDemo() {
    cout << "=== unordered_multimap vs unordered_map ===" << endl;
    cout << "unordered_map: unique keys only" << endl;
    cout << "unordered_multimap: allows duplicate keys" << endl;
    cout << "  - No operator[]" << endl;
    cout << "  - count() can return > 1" << endl;
    cout << "  - equal_range() for finding all values" << endl;
    cout << endl;
}

void studentGradesDemo() {
    cout << "=== Use Case: Student Grades ===" << endl;
    unordered_multimap<string, int> grades;
    grades.insert({"Math", 95});
    grades.insert({"Math", 88});
    grades.insert({"English", 92});
    grades.insert({"Math", 91});

    cout << "All grades:" << endl;
    for (const auto& [subj, grade] : grades) cout << "  " << subj << ": " << grade << endl;

    cout << "\nMath grades:" << endl;
    auto range = grades.equal_range("Math");
    for (auto i = range.first; i != range.second; ++i) {
        cout << "  " << i->second << endl;
    }
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "std::unordered_multimap Demo\n";
    cout << "========================================\n\n";

    basicDemo();
    insertDemo();
    eraseDemo();
    lookupDemo();
    comparisonDemo();
    studentGradesDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::unordered_multimap:\n";
    cout << "  - Hash-based, duplicate keys allowed\n";
    cout << "  - O(1) average operations\n";
    cout << "  - No operator[]\n";
    cout << "  - One-to-many key-value relationships\n";

    return 0;
}

/*
Output:
=== Basic ===
apple -> 5
apple -> 2
banana -> 3
count("apple"): 2

=== Lookup ===
equal_range(1): 1 -> one-a, 1 -> one-b
*/
