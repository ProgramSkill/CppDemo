// std::unordered_set - Hash set with unique elements (no sorting)
// Header: <unordered_set>
// Time Complexity (Average):
//   - Insert: O(1)
//   - Delete: O(1)
//   - Find: O(1)
// Time Complexity (Worst): O(n) - hash collisions
// Space Complexity: O(N)
// Implementation: Hash table

#include <iostream>
#include <unordered_set>
#include <string>
#include <algorithm>

using namespace std;

// Basic demonstration
void basicUnorderedSetDemo() {
    cout << "=== Basic std::unordered_set Demo ===" << endl;

    // Declaration
    unordered_set<int> us1;                                    // Empty set
    unordered_set<int> us2 = {5, 2, 8, 1, 9, 3};             // Initializer list
    unordered_set<int> us3(us2.begin(), us2.end());           // Iterator range

    cout << "us2 (initializer list {5,2,8,1,9,3}): ";
    for (const auto& elem : us2) cout << elem << " ";
    cout << endl;
    cout << "Note: Elements are NOT sorted (order depends on hash)" << endl;

    // Duplicates are automatically removed!
    unordered_set<int> us4 = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    cout << "\nus4 (with duplicates): ";
    for (const auto& elem : us4) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Insertion operations
void insertionDemo() {
    cout << "=== Insertion Demo ===" << endl;

    unordered_set<int> us;

    // insert() - Returns pair<iterator, bool>
    auto result1 = us.insert(5);
    cout << "Insert 5: success=" << result1.second << ", value=" << *result1.first << endl;

    auto result2 = us.insert(5);  // Duplicate!
    cout << "Insert 5 again: success=" << result2.second << endl;

    us.insert(3);
    us.insert(7);
    us.insert(1);
    us.insert(9);

    cout << "After inserts {5,5,3,7,1,9}: ";
    for (const auto& elem : us) cout << elem << " ";
    cout << endl;

    // emplace()
    us.emplace(6);
    cout << "After emplace(6): ";
    for (const auto& elem : us) cout << elem << " ";
    cout << endl;

    // insert() with range
    unordered_set<int> us2 = {100, 200, 300};
    us.insert(us2.begin(), us2.end());
    cout << "After insert range: ";
    for (const auto& elem : us) cout << elem << " ";
    cout << endl;

    // insert() with hint (not very useful for unordered containers)
    auto it = us.begin();
    us.insert(it, 50);
    cout << "After insert with hint: ";
    for (const auto& elem : us) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Deletion operations
void deletionDemo() {
    cout << "=== Deletion Demo ===" << endl;

    unordered_set<int> us = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    cout << "Original: ";
    for (const auto& elem : us) cout << elem << " ";
    cout << endl;

    // erase() by value
    size_t count = us.erase(5);
    cout << "Erase 5: removed " << count << " element(s)" << endl;
    cout << "After erase(5): ";
    for (const auto& elem : us) cout << elem << " ";
    cout << endl;

    // erase() by iterator
    auto it = us.find(3);
    if (it != us.end()) {
        us.erase(it);
        cout << "After erase iterator at 3: ";
        for (const auto& elem : us) cout << elem << " ";
        cout << endl;
    }

    // erase() by range
    auto start = us.begin();
    auto end = us.begin();
    advance(start, 2);
    advance(end, 5);
    us.erase(start, end);
    cout << "After erase range [2, 5): ";
    for (const auto& elem : us) cout << elem << " ";
    cout << endl;

    // clear()
    unordered_set<int> us2 = {1, 2, 3};
    us2.clear();
    cout << "After clear(), size: " << us2.size() << ", empty: " << boolalpha << us2.empty() << endl;
    cout << endl;
}

// Lookup operations
void lookupDemo() {
    cout << "=== Lookup Demo ===" << endl;

    unordered_set<int> us = {10, 20, 30, 40, 50};

    // find() - O(1) average
    auto it = us.find(30);
    if (it != us.end()) {
        cout << "find(30): found " << *it << endl;
    }

    it = us.find(35);
    if (it == us.end()) {
        cout << "find(35): not found" << endl;
    }

    // count() - Returns 0 or 1
    cout << "count(20): " << us.count(20) << endl;
    cout << "count(25): " << us.count(25) << endl;

    // contains() - C++20
    cout << "contains(40): " << boolalpha << us.contains(40) << endl;
    cout << "contains(45): " << boolalpha << us.contains(45) << endl;
    cout << endl;
}

// Hash policy
void hashPolicyDemo() {
    cout << "=== Hash Policy Demo ===" << endl;

    unordered_set<int> us = {1, 2, 3, 4, 5};

    cout << "Size: " << us.size() << endl;
    cout << "Bucket count: " << us.bucket_count() << endl;
    cout << "Load factor: " << us.load_factor() << endl;
    cout << "Max load factor: " << us.max_load_factor() << endl;

    // Bucket information
    cout << "\nBucket details:" << endl;
    for (size_t i = 0; i < us.bucket_count(); ++i) {
        cout << "  Bucket " << i << ": " << us.bucket_size(i) << " element(s)";
        if (us.bucket_size(i) > 0) {
            cout << " -> ";
            for (auto it = us.begin(i); it != us.end(i); ++it) {
                cout << *it << " ";
            }
        }
        cout << endl;
    }

    // Rehash
    cout << "\nBefore rehash:" << endl;
    cout << "  Bucket count: " << us.bucket_count() << endl;
    us.rehash(20);
    cout << "After rehash(20):" << endl;
    cout << "  Bucket count: " << us.bucket_count() << endl;

    // Reserve
    us.reserve(100);
    cout << "After reserve(100):" << endl;
    cout << "  Bucket count: " << us.bucket_count() << endl;
    cout << endl;
}

// Capacity operations
void capacityDemo() {
    cout << "=== Capacity Demo ===" << endl;

    unordered_set<int> us = {1, 2, 3, 4, 5};

    cout << "us.size(): " << us.size() << endl;
    cout << "us.max_size(): " << us.max_size() << endl;
    cout << "us.empty(): " << boolalpha << us.empty() << endl;

    unordered_set<int> emptyUS;
    cout << "emptyUS.empty(): " << boolalpha << emptyUS.empty() << endl;
    cout << endl;
}

// Iterators
void iteratorDemo() {
    cout << "=== Iterator Demo ===" << endl;

    unordered_set<int> us = {1, 2, 3, 4, 5};

    // Forward iteration (NO guaranteed order!)
    cout << "Forward (unordered): ";
    for (auto it = us.begin(); it != us.end(); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Local iterators for buckets
    cout << "\nLocal iteration for bucket 0: ";
    for (auto it = us.begin(0); it != us.end(0); ++it) {
        cout << *it << " ";
    }
    cout << endl;

    // Const iterators
    cout << "Const iteration: ";
    for (auto it = us.cbegin(); it != us.cend(); ++it) {
        cout << *it << " ";
    }
    cout << endl;
    cout << endl;
}

// Custom hash and equal
void customHashDemo() {
    cout << "=== Custom Hash Demo ===" << endl;

    // Custom struct
    struct Person {
        string name;
        int age;

        bool operator==(const Person& other) const {
            return name == other.name && age == other.age;
        }
    };

    // Custom hash function
    struct PersonHash {
        size_t operator()(const Person& p) const {
            return hash<string>()(p.name) ^ hash<int>()(p.age);
        }
    };

    unordered_set<Person, PersonHash> people;
    people.insert({"Alice", 30});
    people.insert({"Bob", 25});
    people.insert({"Charlie", 35});

    cout << "People with custom hash:" << endl;
    for (const auto& person : people) {
        cout << "  " << person.name << " (" << person.age << ")" << endl;
    }
    cout << endl;
}

// unordered_set vs set comparison
void unorderedSetVsSetDemo() {
    cout << "=== unordered_set vs set ===" << endl;

    cout << "std::set:" << endl;
    cout << "  - Elements sorted (red-black tree)" << endl;
    cout << "  - O(log n) insert, delete, find" << endl;
    cout << "  - Ordered iteration" << endl;
    cout << "  - Lower overhead per element" << endl;

    cout << "\nstd::unordered_set:" << endl;
    cout << "  - Elements NOT sorted (hash table)" << endl;
    cout << "  - O(1) average, O(n) worst case" << endl;
    cout << "  - Unordered iteration" << endl;
    cout << "  - Higher memory overhead (bucket array)" << endl;

    // Performance comparison
    cout << "\nPerformance comparison:" << endl;
    const int N = 100000;
    vector<int> data(N);
    for (int i = 0; i < N; ++i) data[i] = i;

    set<int> s(data.begin(), data.end());
    unordered_set<int> us(data.begin(), data.end());

    cout << "For " << N << " elements:" << endl;
    cout << "  set size: " << sizeof(s) << " (implementation dependent)" << endl;
    cout << "  unordered_set bucket_count: " << us.bucket_count() << endl;
    cout << "  unordered_set load_factor: " << us.load_factor() << endl;
    cout << endl;
}

// Use case: Removing duplicates efficiently
void removeDuplicatesDemo() {
    cout << "=== Use Case: Remove Duplicates ===" << endl;

    vector<int> vec = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5};

    cout << "Original vector: ";
    for (const auto& elem : vec) cout << elem << " ";
    cout << endl;

    // Remove duplicates using unordered_set
    unordered_set<int> us(vec.begin(), vec.end());

    cout << "After removing duplicates: ";
    for (const auto& elem : us) cout << elem << " ";
    cout << endl;

    // Note: order is NOT preserved
    cout << "Note: Original order is NOT preserved" << endl;

    // To preserve order, use different approach
    unordered_set<int> seen;
    vector<int> uniqueVec;
    for (const auto& elem : vec) {
        if (seen.insert(elem).second) {
            uniqueVec.push_back(elem);
        }
    }

    cout << "With order preserved: ";
    for (const auto& elem : uniqueVec) cout << elem << " ";
    cout << endl;
    cout << endl;
}

// Use case: Membership testing
void membershipDemo() {
    cout << "=== Use Case: Membership Testing ===" << endl;

    unordered_set<string> dictionary = {"apple", "banana", "cherry", "date", "elderberry"};

    vector<string> words = {"apple", "grape", "banana", "fig", "cherry"};

    cout << "Dictionary contains:" << endl;
    for (const auto& word : dictionary) {
        cout << "  " << word << endl;
    }

    cout << "\nSpell check:" << endl;
    for (const auto& word : words) {
        if (dictionary.find(word) != dictionary.end()) {
            cout << "  \"" << word << "\": VALID" << endl;
        } else {
            cout << "  \"" << word << "\": INVALID" << endl;
        }
    }
    cout << endl;
}

// Observer operations
void observerDemo() {
    cout << "=== Observer Operations Demo ===" << endl;

    unordered_set<int> us = {1, 2, 3, 4, 5};

    // hash_function()
    auto hasher = us.hash_function();
    cout << "hash_function(): hash of 42 = " << hasher(42) << endl;

    // key_eq()
    auto key_eq = us.key_eq();
    cout << "key_eq(): 5 == 5? " << boolalpha << key_eq(5, 5) << endl;
    cout << "key_eq(): 5 == 6? " << boolalpha << key_eq(5, 6) << endl;
    cout << endl;
}

// Performance characteristics
void performanceDemo() {
    cout << "=== Performance Characteristics ===" << endl;

    cout << "Time Complexity (Average):" << endl;
    cout << "  Insert: O(1)" << endl;
    cout << "  Delete: O(1)" << endl;
    cout << "  Find: O(1)" << endl;

    cout << "\nTime Complexity (Worst Case - many collisions):" << endl;
    cout << "  Insert: O(n)" << endl;
    cout << "  Delete: O(n)" << endl;
    cout << "  Find: O(n)" << endl;

    cout << "\nImplementation:" << endl;
    cout << "  Hash table with chaining" << endl;
    cout << "  Elements NOT sorted" << endl;
    cout << "  Fast average case operations" << endl;

    cout << "\nMemory overhead:" << endl;
    cout << "  Array of buckets" << endl;
    cout << "  Each element stored in a bucket" << endl;
    cout << "  Load factor affects performance" << endl;

    cout << "\nWhen to choose unordered_set vs set:" << endl;
    cout << "  - Need O(1) lookups -> unordered_set" << endl;
    cout << "  - Need sorted order -> set" << endl;
    cout << "  - Need range queries -> set" << endl;
    cout << "  - Memory constrained -> set (less overhead)" << endl;
    cout << "  - Custom hash function available -> unordered_set" << endl;
    cout << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << " std::unordered_set Complete Demo" << endl;
    cout << "========================================" << endl << endl;

    basicUnorderedSetDemo();
    insertionDemo();
    deletionDemo();
    lookupDemo();
    hashPolicyDemo();
    capacityDemo();
    iteratorDemo();
    customHashDemo();
    unorderedSetVsSetDemo();
    removeDuplicatesDemo();
    membershipDemo();
    observerDemo();
    performanceDemo();

    cout << "========================================" << endl;
    cout << "              Summary" << endl;
    cout << "========================================" << endl;
    cout << "std::unordered_set characteristics:" << endl;
    cout << "  - Hash-based implementation" << endl;
    cout << "  - Unique elements only" << endl;
    cout << "  - O(1) average insert, delete, find" << endl;
    cout << "  - NO guaranteed order" << endl;
    cout << "\nWhen to use unordered_set:" << endl;
    cout << "  - Fast lookups are critical" << endl;
    cout << "  - Don't need sorted order" << endl;
    cout << "  - Good hash function available" << endl;
    cout << "  - Membership testing" << endl;
    cout << "  - Duplicate removal" << endl;
    cout << "\nAlternatives:" << endl;
    cout << "  - set: When you need sorted order" << endl;
    cout << "  - vector: For small datasets" << endl;
    cout << "  - bitset: For boolean sets" << endl;

    return 0;
}

/*
Output Summary:
=== Basic std::unordered_set Demo ===
us2 (initializer list {5,2,8,1,9,3}): (order varies - hash dependent)
Note: Elements are NOT sorted

=== Hash Policy Demo ===
Size: 5
Bucket count: 7 (implementation dependent)
Load factor: 0.714
Max load factor: 1.0

=== unordered_set vs set ===
std::set: O(log n) operations, ordered
std::unordered_set: O(1) average, unordered

========================================
              Summary
========================================
std::unordered_set characteristics:
  - Hash-based, O(1) average
  - NO guaranteed order
*/
