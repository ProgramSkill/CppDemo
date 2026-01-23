// std::map - Sorted key-value pair container
// Header: <map>
// Time Complexity:
//   - Insert: O(log n)
//   - Delete: O(log n)
//   - Find: O(log n)
//   - Access by key: O(log n)
// Space Complexity: O(N)
// Implementation: Red-black tree (balanced binary search tree)

#include <iostream>
#include <map>
#include <string>
#include <algorithm>

using namespace std;

// Basic demonstration
void basicMapDemo() {
    cout << "=== Basic std::map Demo ===" << endl;

    // Declaration
    map<string, int> m1;                               // Empty map
    map<string, int> m2 = {{"apple", 5}, {"banana", 3}};  // Initializer list
    map<string, int> m3(m2.begin(), m2.end());        // Iterator range
    map<string, int> m4(m2);                          // Copy constructor

    cout << "m2 (initializer list):" << endl;
    for (const auto& [key, value] : m2) {
        cout << "  " << key << " -> " << value << endl;
    }

    // Keys are automatically sorted!
    map<int, string> m5 = {{5, "five"}, {2, "two"}, {8, "eight"}, {1, "one"}};
    cout << "\nm5 (sorted by key):" << endl;
    for (const auto& [key, value] : m5) {
        cout << "  " << key << " -> " << value << endl;
    }

    // Duplicate keys are not allowed!
    map<int, string> m6 = {{1, "one"}, {1, "ONE"}, {1, "1"}};
    cout << "\nm6 (duplicate keys - last one wins):" << endl;
    for (const auto& [key, value] : m6) {
        cout << "  " << key << " -> " << value << endl;
    }
    cout << endl;
}

// Element access
void elementAccessDemo() {
    cout << "=== Element Access Demo ===" << endl;

    map<string, int> m = {{"apple", 5}, {"banana", 3}, {"cherry", 8}};

    // operator[] - Creates default value if key doesn't exist!
    cout << "m[\"apple\"]: " << m["apple"] << endl;
    cout << "m[\"banana\"]: " << m["banana"] << endl;

    // Warning: operator[] creates entry if not found!
    int value = m["grape"];  // Creates entry with value 0
    cout << "m[\"grape\"] (auto-created): " << value << endl;
    cout << "Size after auto-create: " << m.size() << endl;

    // at() - Throws exception if key doesn't exist
    try {
        cout << "m.at(\"cherry\"): " << m.at("cherry") << endl;
        cout << "m.at(\"orange\"): " << m.at("orange") << endl;
    } catch (const out_of_range& e) {
        cout << "Exception: " << e.what() << endl;
    }

    // find() - Safe way to check existence
    auto it = m.find("banana");
    if (it != m.end()) {
        cout << "\nFound: " << it->first << " -> " << it->second << endl;
    }

    it = m.find("orange");
    if (it == m.end()) {
        cout << "\"orange\" not found" << endl;
    }
    cout << endl;
}

// Insertion operations
void insertionDemo() {
    cout << "=== Insertion Demo ===" << endl;

    map<string, int> m;

    // insert() with pair
    m.insert(pair<string, int>("apple", 5));
    m.insert(make_pair("banana", 3));

    // insert() with initializer list
    m.insert({"cherry", 8});

    // insert() - Returns pair<iterator, bool>
    auto result = m.insert({"date", 4});
    cout << "Insert \"date\": success=" << result->second << endl;

    // Duplicate key insert fails
    result = m.insert({"apple", 10});
    cout << "Insert \"apple\" again: success=" << result->second << endl;

    // insert_or_assign() - C++17, replaces if exists
    auto [it, inserted] = m.insert_or_assign("apple", 10);
    cout << "insert_or_assign(\"apple\", 10): inserted=" << inserted << ", value=" << it->second << endl;

    // emplace()
    m.emplace("elderberry", 6);

    // try_emplace() - C++17, doesn't construct if key exists
    auto [it2, inserted2] = m.try_emplace("fig", 7);
    cout << "try_emplace(\"fig\", 7): inserted=" << inserted2 << endl;

    // insert() with range
    map<string, int> m2 = {{"grape", 2}, {"honeydew", 1}};
    m.insert(m2.begin(), m2.end());

    cout << "\nFinal map:" << endl;
    for (const auto& [key, value] : m) {
        cout << "  " << key << " -> " << value << endl;
    }
    cout << endl;
}

// Deletion operations
void deletionDemo() {
    cout << "=== Deletion Demo ===" << endl;

    map<string, int> m = {{"apple", 5}, {"banana", 3}, {"cherry", 8}, {"date", 4}};

    cout << "Original:" << endl;
    for (const auto& [key, value] : m) {
        cout << "  " << key << " -> " << value << endl;
    }

    // erase() by key
    size_t count = m.erase("banana");
    cout << "\nErase \"banana\": removed " << count << " element(s)" << endl;

    // erase() by iterator
    auto it = m.find("cherry");
    if (it != m.end()) {
        m.erase(it);
        cout << "Erase iterator at \"cherry\"" << endl;
    }

    // erase() by range
    auto start = m.begin();
    auto end = m.begin();
    advance(start, 1);
    advance(end, 2);
    m.erase(start, end);
    cout << "Erase range [1, 2)" << endl;

    cout << "\nAfter deletions:" << endl;
    for (const auto& [key, value] : m) {
        cout << "  " << key << " -> " << value << endl;
    }

    // clear()
    map<string, int> m2 = {{"one", 1}, {"two", 2}};
    m2.clear();
    cout << "\nAfter clear(), size: " << m2.size() << ", empty: " << boolalpha << m2.empty() << endl;
    cout << endl;
}

// Lookup operations
void lookupDemo() {
    cout << "=== Lookup Demo ===" << endl;

    map<int, string> m = {{10, "ten"}, {20, "twenty"}, {30, "thirty"}};

    // find() - O(log n)
    auto it = m.find(20);
    if (it != m.end()) {
        cout << "find(20): " << it->first << " -> " << it->second << endl;
    }

    // count() - Returns 0 or 1 (keys are unique!)
    cout << "count(30): " << m.count(30) << endl;
    cout << "count(40): " << m.count(40) << endl;

    // contains() - C++20
    cout << "contains(10): " << boolalpha << m.contains(10) << endl;
    cout << "contains(50): " << boolalpha << m.contains(50) << endl;

    // lower_bound() - First element >= key
    auto lb = m.lower_bound(25);
    cout << "lower_bound(25): " << lb->first << " -> " << lb->second << endl;

    // upper_bound() - First element > key
    auto ub = m.upper_bound(20);
    cout << "upper_bound(20): " << ub->first << " -> " << ub->second << endl;

    // equal_range() - Pair of lower_bound and upper_bound
    auto range = m.equal_range(20);
    cout << "equal_range(20): [" << range.first->first << ", " << range.second->first << ")" << endl;
    cout << endl;
}

// Capacity and size
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    map<string, int> m = {{"a", 1}, {"b", 2}, {"c", 3}};

    cout << "m.size(): " << m.size() << endl;
    cout << "m.max_size(): " << m.max_size() << endl;
    cout << "m.empty(): " << boolalpha << m.empty() << endl;

    map<string, int> emptyMap;
    cout << "emptyMap.empty(): " << boolalpha << emptyMap.empty() << endl;
    cout << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    map<int, string> m = {{1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};

    // Forward iteration (sorted by key!)
    cout << "Forward (sorted by key):" << endl;
    for (auto it = m.begin(); it != m.end(); ++it) {
        cout << "  " << it->first << " -> " << it->second << endl;
    }

    // Reverse iteration
    cout << "\nReverse:" << endl;
    for (auto it = m.rbegin(); it != m.rend(); ++it) {
        cout << "  " << it->first << " -> " << it->second << endl;
    }

    // Structured binding (C++17)
    cout << "\nStructured binding:" << endl;
    for (const auto& [key, value] : m) {
        cout << "  " << key << " -> " << value << endl;
    }
    cout << endl;
}

// Comparison operations
void comparisonDemo() {
    cout << "=== Comparison Demo ===" << endl;

    map<string, int> m1 = {{"a", 1}, {"b", 2}};
    map<string, int> m2 = {{"a", 1}, {"b", 2}};
    map<string, int> m3 = {{"a", 1}, {"b", 3}};

    cout << boolalpha;
    cout << "m1 == m2: " << (m1 == m2) << endl;
    cout << "m1 < m3: " << (m1 < m3) << endl;
    cout << endl;
}

// Custom comparator
void customComparatorDemo() {
    cout << "=== Custom Comparator Demo ===" << endl;

    // Descending order
    map<int, string, greater<int>> m1 = {{1, "one"}, {3, "three"}, {2, "two"}};
    cout << "Descending order by key:" << endl;
    for (const auto& [key, value] : m1) {
        cout << "  " << key << " -> " << value << endl;
    }

    // Custom comparator - sort by string length
    auto cmp = [](const string& a, const string& b) {
        return a.length() < b.length();
    };
    map<string, int, decltype(cmp)> m2(cmp);
    m2["a"] = 1;
    m2["abc"] = 3;
    m2["ab"] = 2;

    cout << "\nSorted by key length:" << endl;
    for (const auto& [key, value] : m2) {
        cout << "  \"" << key << "\" (" << key.length() << ") -> " << value << endl;
    }
    cout << endl;
}

// Map with custom objects as values
void customValueDemo() {
    cout << "=== Custom Value Type Demo ===" << endl;

    struct Person {
        string name;
        int age;

        void print() const {
            cout << name << " (" << age << ")";
        }
    };

    map<int, Person> people;
    people[1] = {"Alice", 30};
    people[2] = {"Bob", 25};
    people[3] = {"Charlie", 35};

    cout << "People indexed by ID:" << endl;
    for (const auto& [id, person] : people) {
        cout << "  ID " << id << ": ";
        person.print();
        cout << endl;
    }
    cout << endl;
}

// Map with custom objects as keys
void customKeyDemo() {
    cout << "=== Custom Key Type Demo ===" << endl;

    struct Person {
        string name;
        int age;

        bool operator<(const Person& other) const {
            return age < other.age;  // Sort by age
        }
    };

    map<Person, string> m;
    m[{"Alice", 30}] = "Engineer";
    m[{"Bob", 25}] = "Designer";
    m[{"Charlie", 35}] = "Manager";

    cout << "People sorted by age (as key):" << endl;
    for (const auto& [person, role] : m) {
        cout << "  " << person.name << " (age " << person.age << "): " << role << endl;
    }
    cout << endl;
}

// Multimap comparison
void mapVsMultimapDemo() {
    cout << "=== map vs multimap ===" << endl;

    cout << "std::map:" << endl;
    cout << "  - Each key must be unique" << endl;
    cout << "  - Insert with duplicate key fails" << endl;
    cout << "  - operator[] creates default value" << endl;

    cout << "\nstd::multimap:" << endl;
    cout << "  - Allows duplicate keys" << endl;
    cout << "  - No operator[]" << endl;
    cout << "  - Always succeeds on insert" << endl;
    cout << endl;
}

// Use case: Word frequency counter
void wordFrequencyDemo() {
    cout << "=== Use Case: Word Frequency Counter ===" << endl;

    string text = "the quick brown fox jumps over the lazy dog the dog barked";
    map<string, int> frequency;

    // Count word frequencies
    string word;
    for (size_t i = 0; i < text.length(); ++i) {
        if (text[i] == ' ') {
            if (!word.empty()) {
                frequency[word]++;
                word.clear();
            }
        } else {
            word += text[i];
        }
    }
    if (!word.empty()) {
        frequency[word]++;
    }

    cout << "Word frequencies (sorted alphabetically):" << endl;
    for (const auto& [word, count] : frequency) {
        cout << "  \"" << word << "\": " << count << endl;
    }
    cout << endl;
}

// Use case: Configuration settings
void configDemo() {
    cout << "=== Use Case: Configuration Settings ===" << endl;

    map<string, string> config = {
        {"host", "localhost"},
        {"port", "8080"},
        {"debug", "true"},
        {"timeout", "30"}
    };

    cout << "Configuration settings:" << endl;
    for (const auto& [key, value] : config) {
        cout << "  " << key << " = " << value << endl;
    }

    cout << "\nAccessing settings:" << endl;
    cout << "  Server: " << config["host"] << ":" << config["port"] << endl;
    cout << "  Debug mode: " << (config["debug"] == "true" ? "enabled" : "disabled") << endl;
    cout << endl;
}

// Observer operations
void observerDemo() {
    cout << "=== Observer Operations Demo ===" << endl;

    map<int, string> m = {{1, "one"}, {2, "two"}, {3, "three"}};

    // key_comp()
    auto keyComp = m.key_comp();
    cout << "key_comp(): Is 1 < 2? " << boolalpha << keyComp(1, 2) << endl;

    // value_comp() - Compares keys, not key-value pairs!
    auto valueComp = m.value_comp();
    cout << "value_comp(): Is 2 < 3? " << boolalpha << valueComp(
        pair<const int, string>(2, "two"),
        pair<const int, string>(3, "three")
    ) << endl;
    cout << endl;
}

// Performance characteristics
void performanceDemo() {
    cout << "=== Performance Characteristics ===" << endl;

    cout << "Time Complexity:" << endl;
    cout << "  Insert: O(log n)" << endl;
    cout << "  Delete: O(log n)" << endl;
    cout << "  Find: O(log n)" << endl;
    cout << "  Access by key []: O(log n)" << endl;
    cout << "  Traversal: O(n)" << endl;

    cout << "\nImplementation:" << endl;
    cout << "  Red-black tree (self-balancing BST)" << endl;
    cout << "  Keys always sorted" << endl;
    cout << "  Guaranteed O(log n) operations" << endl;

    cout << "\nMemory overhead:" << endl;
    cout << "  Each node stores:" << endl;
    cout << "    - Key" << endl;
    cout << "    - Value" << endl;
    cout << "    - Left child pointer" << endl;
    cout << "    - Right child pointer" << endl;
    cout << "    - Parent pointer" << endl;
    cout << "    - Color (red/black)" << endl;
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "      std::map Complete Demonstration" << endl;
    cout << "========================================" << endl << endl;

    basicMapDemo();
    elementAccessDemo();
    insertionDemo();
    deletionDemo();
    lookupDemo();
    capacityDemo();
    iteratorDemo();
    comparisonDemo();
    customComparatorDemo();
    customValueDemo();
    customKeyDemo();
    mapVsMultimapDemo();
    wordFrequencyDemo();
    configDemo();
    observerDemo();
    performanceDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::map characteristics:" << endl;
    cout << "  - Stores key-value pairs" << endl;
    cout << "  - Keys are unique and sorted" << endl;
    cout << "  - O(log n) insert, delete, find" << endl;
    cout << "  - Implemented as red-black tree" << endl;
    cout << "\nWhen to use map:" << endl;
    cout << "  - Need associative array (dictionary)" << endl;
    cout << "  - Need sorted keys" << endl;
    cout << "  - Fast lookups by key" << endl;
    cout << "  - One-to-one key-value mapping" << endl;
    cout << "\nAlternatives:" << endl;
    cout << "  - unordered_map: O(1) average, no sorting" << endl;
    cout << "  - multimap: Allows duplicate keys" << endl;
    cout << "  - vector<pair>: When you need random access" << endl;

    return 0;
}

/*
Output Summary:
=== Basic std::map Demo ===
m2 (initializer list):
  apple -> 5
  banana -> 3

m5 (sorted by key):
  1 -> one
  2 -> two
  5 -> five
  8 -> eight

=== Element Access Demo ===
m["apple"]: 5
m["grape"] (auto-created): 0
m.at("orange"): throws out_of_range

=== Use Case: Word Frequency Counter ===
Word frequencies (sorted alphabetically):
  "barked": 1
  "brown": 1
  "dog": 2
  "fox": 1
  "jumps": 1
  "lazy": 1
  "over": 1
  "quick": 1
  "the": 2

========================================
              Summary
========================================
std::map characteristics:
  - Stores key-value pairs
  - Keys are unique and sorted
  - O(log n) operations
*/
