// std::multimap - Sorted key-value pair container allowing duplicate keys
// Header: <map>
// Time Complexity:
//   - Insert: O(log n)
//   - Delete: O(log n)
//   - Find: O(log n)
//   - Count: O(log n + k) where k is count of elements with key
// Space Complexity: O(N)
// Implementation: Red-black tree (allows duplicate keys)

#include <iostream>
#include <map>
#include <string>
#include <algorithm>

using namespace std;

// Basic demonstration
void basicMultimapDemo() {
    cout << "=== Basic std::multimap Demo ===" << endl;

    // Declaration
    multimap<string, int> mm1;                                         // Empty multimap
    multimap<string, int> mm2 = {{"apple", 5}, {"banana", 3}, {"apple", 2}};  // Duplicate keys!
    multimap<string, int> mm3(mm2.begin(), mm2.end());                 // Iterator range

    cout << "mm2 (with duplicate key \"apple\"):" << endl;
    for (const auto& [key, value] : mm2) {
        cout << "  " << key << " -> " << value << endl;
    }

    // Keys are sorted, duplicates are allowed
    multimap<int, string> mm4 = {{1, "one"}, {3, "three"}, {2, "two"}, {1, "ONE"}};
    cout << "\nmm4 (sorted by key, with duplicates):" << endl;
    for (const auto& [key, value] : mm4) {
        cout << "  " << key << " -> " << value << endl;
    }

    // Count entries with the same key
    cout << "\nCount of key \"apple\": " << mm2.count("apple") << endl;
    cout << "Count of key \"banana\": " << mm2.count("banana") << endl;
    cout << endl;
}

// Insertion operations
void insertionDemo() {
    cout << "=== Insertion Demo ===" << endl;

    multimap<string, int> mm;

    // insert() with pair - always succeeds!
    mm.insert(pair<string, int>("apple", 5));
    mm.insert(make_pair("banana", 3));
    mm.insert({"apple", 2});  // Duplicate key!

    cout << "After inserts (with duplicate \"apple\"):" << endl;
    for (const auto& [key, value] : mm) {
        cout << "  " << key << " -> " << value << endl;
    }

    // insert() returns iterator (not pair like std::map)
    auto it = mm.insert({"cherry", 8});
    cout << "\nInsert \"cherry\": success (iterator points to: " << it->first << " -> " << it->second << ")" << endl;

    // emplace()
    mm.emplace("apple", 10);  // Another duplicate!
    cout << "After emplace another \"apple\": " << mm.count("apple") << " entries" << endl;

    // insert() with range
    multimap<string, int> mm2 = {{"date", 4}, {"elderberry", 6}};
    mm.insert(mm2.begin(), mm2.end());

    cout << "\nFinal multimap:" << endl;
    for (const auto& [key, value] : mm) {
        cout << "  " << key << " -> " << value << endl;
    }
    cout << endl;
}

// Deletion operations
void deletionDemo() {
    cout << "=== Deletion Demo ===" << endl;

    multimap<string, int> mm = {
        {"apple", 5},
        {"apple", 2},
        {"banana", 3},
        {"cherry", 8},
        {"cherry", 4},
        {"date", 1}
    };

    cout << "Original:" << endl;
    for (const auto& [key, value] : mm) {
        cout << "  " << key << " -> " << value << endl;
    }

    // erase() by key - removes ALL entries with that key!
    size_t count = mm.erase("apple");
    cout << "\nErase \"apple\": removed " << count << " entry/entries" << endl;

    cout << "After erase(\"apple\"):" << endl;
    for (const auto& [key, value] : mm) {
        cout << "  " << key << " -> " << value << endl;
    }

    // erase() by iterator - removes single entry
    auto it = mm.find("cherry");
    if (it != mm.end()) {
        mm.erase(it);  // Erases only one occurrence
        cout << "\nAfter erase one \"cherry\" entry:" << endl;
        for (const auto& [key, value] : mm) {
            cout << "  " << key << " -> " << value << endl;
        }
    }

    // erase() by range
    multimap<int, string> mm2 = {{1, "a"}, {2, "b"}, {3, "c"}, {4, "d"}, {5, "e"}};
    auto start = mm2.begin();
    auto end = mm2.begin();
    advance(start, 1);
    advance(end, 4);
    mm2.erase(start, end);
    cout << "\nAfter erase range [1, 4): ";
    for (const auto& [key, value] : mm2) {
        cout << key << " ";
    }
    cout << endl;

    // clear()
    multimap<string, int> mm3 = {{"one", 1}, {"two", 2}};
    mm3.clear();
    cout << "After clear(), size: " << mm3.size() << ", empty: " << boolalpha << mm3.empty() << endl;
    cout << endl;
}

// Lookup operations
void lookupDemo() {
    cout << "=== Lookup Demo ===" << endl;

    multimap<int, string> mm = {
        {10, "ten-a"},
        {20, "twenty-a"},
        {20, "twenty-b"},
        {30, "thirty-a"},
        {30, "thirty-b"},
        {30, "thirty-c"}
    };

    // find() - Returns iterator to first entry with key
    auto it = mm.find(20);
    if (it != mm.end()) {
        cout << "find(20): " << it->first << " -> " << it->second << " (first occurrence)" << endl;
    }

    // count() - Returns number of entries with key!
    cout << "\ncount(10): " << mm.count(10) << " entry/entries" << endl;
    cout << "count(20): " << mm.count(20) << " entries" << endl;
    cout << "count(30): " << mm.count(30) << " entries" << endl;
    cout << "count(40): " << mm.count(40) << " entries" << endl;

    // lower_bound() - First entry >= key
    auto lb = mm.lower_bound(25);
    cout << "\nlower_bound(25): " << lb->first << " -> " << lb->second << endl;

    // upper_bound() - First entry > key
    auto ub = mm.upper_bound(20);
    cout << "upper_bound(20): " << ub->first << " -> " << ub->second << endl;

    // equal_range() - Range of entries with key
    auto range = mm.equal_range(30);
    cout << "\nequal_range(30):" << endl;
    for (auto it = range.first; it != range.second; ++it) {
        cout << "  " << it->first << " -> " << it->second << endl;
    }
    cout << "Count in range: " << distance(range.first, range.second) << endl;
    cout << endl;
}

// Capacity and size
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    multimap<string, int> mm = {{"a", 1}, {"b", 2}, {"b", 3}};

    cout << "mm.size(): " << mm.size() << endl;
    cout << "mm.max_size(): " << mm.max_size() << endl;
    cout << "mm.empty(): " << boolalpha << mm.empty() << endl;

    multimap<string, int> emptyMM;
    cout << "emptyMM.empty(): " << boolalpha << emptyMM.empty() << endl;
    cout << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    multimap<int, string> mm = {{1, "one"}, {2, "two-a"}, {2, "two-b"}, {3, "three"}};

    // Forward iteration (sorted by key!)
    cout << "Forward (sorted by key):" << endl;
    for (auto it = mm.begin(); it != mm.end(); ++it) {
        cout << "  " << it->first << " -> " << it->second << endl;
    }

    // Reverse iteration
    cout << "\nReverse:" << endl;
    for (auto it = mm.rbegin(); it != mm.rend(); ++it) {
        cout << "  " << it->first << " -> " << it->second << endl;
    }

    // Structured binding (C++17)
    cout << "\nStructured binding:" << endl;
    for (const auto& [key, value] : mm) {
        cout << "  " << key << " -> " << value << endl;
    }
    cout << endl;
}

// Comparison operations
void comparisonDemo() {
    cout << "=== Comparison Demo ===" << endl;

    multimap<string, int> mm1 = {{"a", 1}, {"b", 2}};
    multimap<string, int> mm2 = {{"a", 1}, {"b", 2}};
    multimap<string, int> mm3 = {{"a", 1}, {"b", 3}};

    cout << boolalpha;
    cout << "mm1 == mm2: " << (mm1 == mm2) << endl;
    cout << "mm1 < mm3: " << (mm1 < mm3) << endl;
    cout << endl;
}

// Custom comparator
void customComparatorDemo() {
    cout << "=== Custom Comparator Demo ===" << endl;

    // Descending order by key
    multimap<int, string, greater<int>> mm1 = {{1, "one"}, {3, "three"}, {2, "two"}, {1, "ONE"}};
    cout << "Descending order by key:" << endl;
    for (const auto& [key, value] : mm1) {
        cout << "  " << key << " -> " << value << endl;
    }

    // Custom comparator - sort by value length
    auto cmp = [](const string& a, const string& b) {
        return a.length() < b.length();
    };
    multimap<string, int, decltype(cmp)> mm2(cmp);
    mm2.insert({"a", 1});
    mm2.insert({"abc", 3});
    mm2.insert({"ab", 2});
    mm2.insert({"ab", 20});  // Duplicate key with same length

    cout << "\nSorted by key length:" << endl;
    for (const auto& [key, value] : mm2) {
        cout << "  \"" << key << "\" (" << key.length() << ") -> " << value << endl;
    }
    cout << endl;
}

// multimap vs map comparison
void multimapVsMapDemo() {
    cout << "=== multimap vs map ===" << endl;

    cout << "std::map:" << endl;
    cout << "  - Each key must be unique" << endl;
    cout << "  - insert() returns pair<iterator, bool>" << endl;
    cout << "  - operator[] creates/updates single entry" << endl;
    cout << "  - erase(key) removes the single entry" << endl;
    cout << "  - count() always returns 0 or 1" << endl;

    cout << "\nstd::multimap:" << endl;
    cout << "  - Allows duplicate keys" << endl;
    cout << "  - insert() always succeeds, returns iterator" << endl;
    cout << "  - No operator[]" << endl;
    cout << "  - erase(key) removes ALL entries with that key" << endl;
    cout << "  - count() returns number of entries with key" << endl;

    // Demonstration
    cout << "\nDemonstration:" << endl;
    map<string, int> m = {{"apple", 5}, {"apple", 2}, {"banana", 3}};
    multimap<string, int> mm = {{"apple", 5}, {"apple", 2}, {"banana", 3}};

    cout << "map from {\"apple\":5, \"apple\":2, \"banana\":3}:" << endl;
    for (const auto& [key, value] : m) {
        cout << "  " << key << " -> " << value << endl;
    }

    cout << "\nmultimap from same data:" << endl;
    for (const auto& [key, value] : mm) {
        cout << "  " << key << " -> " << value << endl;
    }
    cout << endl;
}

// Use case: Student grades by subject
void studentGradesDemo() {
    cout << "=== Use Case: Student Grades ===" << endl;

    struct Grade {
        string student;
        int score;

        Grade(string s, int sc) : student(s), score(sc) {}
    };

    // Subject -> Grades (multiple students per subject)
    multimap<string, Grade> grades;

    grades.insert({"Math", Grade("Alice", 95)});
    grades.insert({"Math", Grade("Bob", 87)});
    grades.insert({"Math", Grade("Charlie", 92)});
    grades.insert({"Physics", Grade("Alice", 88)});
    grades.insert({"Physics", Grade("Bob", 90)});
    grades.insert({"English", Grade("Alice", 91)});

    cout << "Grades by subject:" << endl;
    for (const auto& [subject, grade] : grades) {
        cout << "  " << subject << ": " << grade.student << " - " << grade.score << endl;
    }

    cout << "\nStudents in Math:" << endl;
    auto range = grades.equal_range("Math");
    for (auto it = range.first; it != range.second; ++it) {
        cout << "  " << it->second.student << ": " << it->second.score << endl;
    }
    cout << endl;
}

// Use case: Dictionary with multiple definitions
void dictionaryDemo() {
    cout << "=== Use Case: Dictionary ===" << endl;

    multimap<string, string> dictionary;

    dictionary.insert({"bank", "financial institution"});
    dictionary.insert({"bank", "land alongside river"});
    dictionary.insert({"bank", "to put in a bank"});
    dictionary.insert({"run", "to move fast"});
    dictionary.insert({"run", "to operate"});
    dictionary.insert({"run", "a series of events"});

    cout << "Dictionary entries:" << endl;
    for (const auto& [word, definition] : dictionary) {
        cout << "  " << word << ": " << definition << endl;
    }

    cout << "\nDefinitions of \"run\":" << endl;
    auto range = dictionary.equal_range("run");
    for (auto it = range.first; it != range.second; ++it) {
        cout << "  - " << it->second << endl;
    }
    cout << endl;
}

// Use case: Time-series data
void timeSeriesDemo() {
    cout << "=== Use Case: Time-Series Data ===" << endl;

    multimap<string, double> stockPrices;

    // Multiple price updates for the same stock
    stockPrices.insert({"AAPL", 150.25});
    stockPrices.insert({"GOOGL", 2800.50});
    stockPrices.insert({"AAPL", 151.30});  // Price update
    stockPrices.insert({"MSFT", 300.75});
    stockPrices.insert({"AAPL", 149.80});  // Another update
    stockPrices.insert({"GOOGL", 2810.20});

    cout << "Stock price history (sorted by symbol):" << endl;
    for (const auto& [symbol, price] : stockPrices) {
        cout << "  " << symbol << ": $" << price << endl;
    }

    cout << "\nAll prices for AAPL:" << endl;
    auto range = stockPrices.equal_range("AAPL");
    for (auto it = range.first; it != range.second; ++it) {
        cout << "  $" << it->second << endl;
    }
    cout << endl;
}

// Finding all values for a key
void findAllValuesDemo() {
    cout << "=== Finding All Values for a Key ===" << endl;

    multimap<string, int> mm = {
        {"fruit", 1},
        {"fruit", 2},
        {"vegetable", 3},
        {"fruit", 4},
        {"meat", 5}
    };

    string key = "fruit";

    cout << "Multimap contents:" << endl;
    for (const auto& [k, v] : mm) {
        cout << "  " << k << " -> " << v << endl;
    }

    // Method 1: Using equal_range
    cout << "\nValues for \"" << key << "\" (using equal_range):" << endl;
    auto range = mm.equal_range(key);
    for (auto it = range.first; it != range.second; ++it) {
        cout << "  " << it->second << endl;
    }

    // Method 2: Using lower_bound and upper_bound
    cout << "\nValues for \"" << key << "\" (using lower/upper_bound):" << endl;
    for (auto it = mm.lower_bound(key); it != mm.upper_bound(key); ++it) {
        cout << "  " << it->second << endl;
    }

    // Method 3: Using count and find
    cout << "\nValues for \"" << key << "\" (using count and find):" << endl;
    auto count = mm.count(key);
    auto it = mm.find(key);
    for (size_t i = 0; i < count && it != mm.end(); ++i, ++it) {
        cout << "  " << it->second << endl;
    }
    cout << endl;
}

// Performance characteristics
void performanceDemo() {
    cout << "=== Performance Characteristics ===" << endl;

    cout << "Time Complexity:" << endl;
    cout << "  Insert: O(log n)" << endl;
    cout << "  Delete: O(log n)" << endl;
    cout << "  Find: O(log n)" << endl;
    cout << "  Count: O(log n + k) where k is entries with key" << endl;
    cout << "  Access by key: No operator[]!" << endl;

    cout << "\nImplementation:" << endl;
    cout << "  Red-black tree (same as std::map)" << endl;
    cout << "  Keys always sorted" << endl;
    cout << "  Duplicate keys allowed" << endl;

    cout << "\nMemory overhead:" << endl;
    cout << "  Same as std::map (red-black tree nodes)" << endl;
    cout << "  Each duplicate key-value pair is a separate node" << endl;
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "    std::multimap Complete Demo" << endl;
    cout << "========================================" << endl << endl;

    basicMultimapDemo();
    insertionDemo();
    deletionDemo();
    lookupDemo();
    capacityDemo();
    iteratorDemo();
    comparisonDemo();
    customComparatorDemo();
    multimapVsMapDemo();
    studentGradesDemo();
    dictionaryDemo();
    timeSeriesDemo();
    findAllValuesDemo();
    performanceDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::multimap characteristics:" << endl;
    cout << "  - Stores key-value pairs" << endl;
    cout << "  - Allows duplicate keys" << endl;
    cout << "  - Keys always sorted" << endl;
    cout << "  - O(log n) insert, delete, find" << endl;
    cout << "  - No operator[]" << endl;
    cout << "\nWhen to use multimap:" << endl;
    cout << "  - One-to-many key-value relationships" << endl;
    cout << "  - Need to store multiple values per key" << endl;
    cout << "  - Need sorted keys" << endl;
    cout << "  - Dictionary, time-series, grouped data" << endl;
    cout << "\nAlternatives:" << endl;
    cout << "  - map: For unique keys only" << endl;
    cout << "  - unordered_multimap: O(1) average, no sorting" << endl;
    cout << "  - map<vector<Value>>: For grouped access patterns" << endl;

    return 0;
}

/*
Output Summary:
=== Basic std::multimap Demo ===
mm2 (with duplicate key "apple"):
  apple -> 5
  apple -> 2
  banana -> 3

Count of key "apple": 2

=== Lookup Demo ===
find(20): 20 -> twenty-a (first occurrence)
count(20): 2 entries
count(30): 3 entries
equal_range(30):
  30 -> thirty-a
  30 -> thirty-b
  30 -> thirty-c

=== multimap vs map ===
map from {"apple":5, "apple":2, "banana":3}:
  apple -> 2  (last one wins!)
multimap from same data:
  apple -> 5
  apple -> 2
  banana -> 3

========================================
              Summary
========================================
std::multimap characteristics:
  - One-to-many key-value relationships
  - No operator[]
  - O(log n) operations
*/
