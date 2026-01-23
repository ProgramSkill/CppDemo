// std::unordered_map - Hash map with unique keys (no sorting)
// Header: <unordered_map>
// Time Complexity (Average): O(1) insert, delete, find
// Time Complexity (Worst): O(n) - hash collisions
// Space Complexity: O(N)
// Implementation: Hash table

#include <iostream>
#include <unordered_map>
#include <string>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::unordered_map Demo ===" << endl;
    unordered_map<string, int> m = {{"apple", 5}, {"banana", 3}, {"cherry", 8}};
    cout << "Contents (unordered):" << endl;
    for (const auto& [k, v] : m) cout << "  " << k << " -> " << v << endl;
    cout << endl;
}

void accessDemo() {
    cout << "=== Element Access ===" << endl;
    unordered_map<string, int> m = {{"apple", 5}, {"banana", 3}};
    cout << "m[\"apple\"]: " << m["apple"] << endl;
    cout << "m.at(\"banana\"): " << m.at("banana") << endl;
    m["grape"] = 2;
    cout << "After m[\"grape\"] = 2" << endl;
    try { m.at("orange"); }
    catch (const out_of_range& e) { cout << "Exception: " << e.what() << endl; }
    cout << endl;
}

void insertDemo() {
    cout << "=== Insertion ===" << endl;
    unordered_map<string, int> m;
    m.insert({"apple", 5});
    m.emplace("banana", 3);
    auto [it, success] = m.insert_or_assign("apple", 10);
    cout << "insert_or_assign apple: inserted=" << success << ", value=" << it->second << endl;
    for (const auto& [k, v] : m) cout << "  " << k << " -> " << v << endl;
    cout << endl;
}

void eraseDemo() {
    cout << "=== Deletion ===" << endl;
    unordered_map<string, int> m = {{"a", 1}, {"b", 2}, {"c", 3}};
    m.erase("b");
    cout << "After erase(\"b\"): ";
    for (const auto& [k, v] : m) cout << k << " ";
    cout << endl << endl;
}

void lookupDemo() {
    cout << "=== Lookup ===" << endl;
    unordered_map<int, string> m = {{1, "one"}, {2, "two"}, {3, "three"}};
    auto it = m.find(2);
    if (it != m.end()) cout << "find(2): " << it->first << " -> " << it->second << endl;
    cout << "count(3): " << m.count(3) << endl;
    cout << "contains(4): " << boolalpha << m.contains(4) << endl;
    cout << endl;
}

void hashPolicyDemo() {
    cout << "=== Hash Policy ===" << endl;
    unordered_map<int, string> m = {{1, "a"}, {2, "b"}, {3, "c"}};
    cout << "bucket_count: " << m.bucket_count() << endl;
    cout << "load_factor: " << m.load_factor() << endl;
    cout << "max_load_factor: " << m.max_load_factor() << endl;
    m.rehash(20);
    cout << "After rehash(20), bucket_count: " << m.bucket_count() << endl;
    cout << endl;
}

void customHashDemo() {
    cout << "=== Custom Hash ===" << endl;
    struct Person { string name; int age; bool operator==(const Person& o) const { return name == o.name; } };
    auto person_hash = [](const Person& p) { return hash<string>()(p.name); };
    unordered_map<Person, string, decltype(person_hash)> m(0, person_hash);
    m[{"Alice", 30}] = "Engineer";
    m[{"Bob", 25}] = "Designer";
    for (const auto& [p, role] : m) cout << "  " << p.name << ": " << role << endl;
    cout << endl;
}

void unorderedMapVsMapDemo() {
    cout << "=== unordered_map vs map ===" << endl;
    cout << "unordered_map: O(1) average, unordered" << endl;
    cout << "map: O(log n), sorted by key" << endl;
    cout << "Choose unordered_map for:" << endl;
    cout << "  - Fast lookups" << endl;
    cout << "  - Don't need sorted keys" << endl;
    cout << "  - Good hash function available" << endl;
    cout << endl;
}

void wordCountDemo() {
    cout << "=== Use Case: Word Frequency ===" << endl;
    string text = "the quick brown fox jumps over the lazy dog";
    unordered_map<string, int> freq;
    string word;
    for (char c : text) {
        if (c == ' ') { if (!word.empty()) { freq[word]++; word.clear(); } }
        else word += c;
    }
    if (!word.empty()) freq[word]++;
    for (const auto& [w, c] : freq) cout << "  \"" << w << "\": " << c << endl;
    cout << endl;
}

void cacheDemo() {
    cout << "=== Use Case: Cache ===" << endl;
    unordered_map<string, int> cache;
    auto get = [&](string key) -> int {
        if (cache.find(key) != cache.end()) {
            cout << "  Cache HIT for " << key << endl;
            return cache[key];
        }
        cout << "  Cache MISS for " << key << ", computing..." << endl;
        cache[key] = key.length();
        return cache[key];
    };
    get("apple");
    get("banana");
    get("apple");  // Cache hit
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "  std::unordered_map Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    accessDemo();
    insertDemo();
    eraseDemo();
    lookupDemo();
    hashPolicyDemo();
    customHashDemo();
    unorderedMapVsMapDemo();
    wordCountDemo();
    cacheDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::unordered_map: Hash-based map\n";
    cout << "  - O(1) average operations\n";
    cout << "  - Unique keys, unordered\n";
    cout << "  - operator[] and at() for access\n";
    cout << "  - Perfect for caches, lookups\n";

    return 0;
}

/*
Output Summary:
=== Basic std::unordered_map Demo ===
Contents (unordered):
  apple -> 5
  banana -> 3
  cherry -> 8

=== Hash Policy ===
bucket_count: varies (implementation dependent)
load_factor: ~0.5-1.0

========================================
              Summary
========================================
std::unordered_map: O(1) average, unordered
*/
