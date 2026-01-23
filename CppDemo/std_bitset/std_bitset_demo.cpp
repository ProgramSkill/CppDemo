// std::bitset - Fixed-size bit set
// Header: <bitset>
// Efficient bit manipulation

#include <iostream>
#include <bitset>
#include <string>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::bitset Demo ===" << endl;
    bitset<8> b1;           // Default initialized (all zeros)
    bitset<8> b2(0b101010); // From binary literal
    bitset<8> b3(42);       // From integer
    bitset<8> b4(string("11001010"));

    cout << "b1 (default): " << b1 << endl;
    cout << "b2 (0b101010): " << b2 << endl;
    cout << "b3 (42): " << b3 << endl;
    cout << "b4 (string): " << b4 << endl;
    cout << endl;
}

void accessDemo() {
    cout << "=== Bit Access ===" << endl;
    bitset<8> b(0b10101010);

    cout << "Bitset: " << b << endl;
    cout << "b[0]: " << b[0] << endl;
    cout << "b[3]: " << b[3] << endl;
    cout << "b.test(5): " << b.test(5) << endl;

    b.set(0);
    cout << "After b.set(0): " << b << endl;

    b.reset(1);
    cout << "After b.reset(1): " << b << endl;

    b.flip(2);
    cout << "After b.flip(2): " << b << endl;
    cout << endl;
}

void operationsDemo() {
    cout << "=== Bit Operations ===" << endl;
    bitset<8> b1(0b1100);
    bitset<8> b2(0b1010);

    cout << "b1: " << b1 << " (" << b1.to_ulong() << ")" << endl;
    cout << "b2: " << b2 << " (" << b2.to_ulong() << ")" << endl;

    cout << "b1 & b2: " << (b1 & b2) << endl;
    cout << "b1 | b2: " << (b1 | b2) << endl;
    cout << "b1 ^ b2: " << (b1 ^ b2) << endl;
    cout << "~b1: " << (~b1) << endl;
    cout << "b1 << 2: " << (b1 << 2) << endl;
    cout << "b1 >> 1: " << (b1 >> 1) << endl;
    cout << endl;
}

void setResetFlipDemo() {
    cout << "=== Set/Reset/Flip All ===" << endl;
    bitset<8> b(0b10101010);

    cout << "Original: " << b << endl;
    b.set();
    cout << "After set(): " << b << endl;

    b.reset();
    cout << "After reset(): " << b << endl;

    b = 0b10101010;
    b.flip();
    cout << "After flip(): " << b << endl;
    cout << endl;
}

void countDemo() {
    cout << "=== Count Operations ===" << endl;
    bitset<16> b(0b1011011010110101);

    cout << "Bitset: " << b << endl;
    cout << "count() (ones): " << b.count() << endl;
    cout << "size() (total bits): " << b.size() << endl;
    cout << "Any set: " << boolalpha << b.any() << endl;
    cout << "All set: " << boolalpha << b.all() << endl;
    cout << "None set: " << boolalpha << b.none() << endl;
    cout << endl;
}

void conversionDemo() {
    cout << "=== Conversion ===" << endl;
    bitset<8> b(0b11001010);

    cout << "Bitset: " << b << endl;
    cout << "to_ulong(): " << b.to_ulong() << endl;
    cout << "to_ullong(): " << b.to_ullong() << endl;
    cout << "to_string(): " << b.to_string() << endl;
    cout << "to_string('o', 'x'): " << b.to_string('o', 'x') << endl;
    cout << endl;
}

void flagsDemo() {
    cout << "=== Use Case: Flags ===" << endl;
    enum Flags { READ = 0, WRITE = 1, EXECUTE = 2 };
    bitset<3> flags;

    flags[READ] = true;
    flags[WRITE] = true;
    flags[EXECUTE] = false;

    cout << "Flags: " << flags << endl;
    cout << "Can read: " << boolalpha << flags[READ] << endl;
    cout << "Can write: " << boolalpha << flags[WRITE] << endl;
    cout << "Can execute: " << boolalpha << flags[EXECUTE] << endl;
    cout << endl;
}

void subsetDemo() {
    cout << "=== Use Case: Subset ===" << endl;
    bitset<5> set1(0b10101);  // {0, 2, 4}
    bitset<5> set2(0b00101);  // {0, 2}

    cout << "Set1: " << set1 << endl;
    cout << "Set2: " << set2 << endl;

    bitset<5> intersection = set1 & set2;
    cout << "Intersection: " << intersection << endl;

    bitset<5> union_set = set1 | set2;
    cout << "Union: " << union_set << endl;

    bitset<5> difference = set1 & ~set2;
    cout << "Difference (set1 - set2): " << difference << endl;
    cout << endl;
}

void permutationDemo() {
    cout << "=== Use Case: Permutation ===" << endl;
    bitset<5> perm;  // Represents subset of {0,1,2,3,4}

    // Generate all 2^5 subsets
    int count = 0;
    for (size_t i = 0; i < (1 << 5); ++i) {
        perm = i;
        cout << perm << " ";
        if (++count % 8 == 0) cout << endl;
    }
    cout << "\nTotal subsets: " << count << endl;
    cout << endl;
}

void sieveDemo() {
    cout << "=== Use Case: Sieve of Eratosthenes ===" << endl;
    const int N = 20;
    bitset<N + 1> isPrime;
    isPrime.flip();  // Set all to true

    isPrime[0] = isPrime[1] = false;

    for (int i = 2; i * i <= N; ++i) {
        if (isPrime[i]) {
            for (int j = i * i; j <= N; j += i) {
                isPrime[j] = false;
            }
        }
    }

    cout << "Primes up to " << N << ": ";
    for (int i = 2; i <= N; ++i) {
        if (isPrime[i]) cout << i << " ";
    }
    cout << endl;
    cout << endl;
}

void grayCodeDemo() {
    cout << "=== Use Case: Gray Code ===" << endl;
    const int N = 4;

    for (size_t i = 0; i < (1 << N); ++i) {
        size_t gray = i ^ (i >> 1);
        bitset<N> b(gray);
        cout << "Decimal: " << i << ", Binary: " << bitset<N>(i)
             << ", Gray: " << b << endl;
    }
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "      std::bitset Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    accessDemo();
    operationsDemo();
    setResetFlipDemo();
    countDemo();
    conversionDemo();
    flagsDemo();
    subsetDemo();
    permutationDemo();
    sieveDemo();
    grayCodeDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::bitset: Fixed-size bit array\n";
    cout << "  - Size fixed at compile time\n";
    cout << "  - Efficient bit operations\n";
    cout << "  - Perfect for flags, sets, masks\n";
    cout << "  - Alternative: vector<bool> (dynamic)\n";
    cout << "  - Use for bit manipulation algorithms\n";

    return 0;
}

/*
Output Summary:
=== Basic ===
b1 (default): 00000000
b2 (0b101010): 00101010
b3 (42): 00101010

=== Operations ===
b1 & b2: 1000
b1 | b2: 1110

=== Count ===
count(): 8
any: true, all: false

=== Sieve ===
Primes up to 20: 2 3 5 7 11 13 17 19
*/
