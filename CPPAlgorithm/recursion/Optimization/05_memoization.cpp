// recursion memoization
// Memoization: Cache results to avoid redundant calculations
#include <iostream>
#include <unordered_map>
using namespace std;

// Global cache to store computed results
unordered_map<int, long long> memo;

// [Problem] Normal recursive fibonacci has O(2^n) time complexity
// fib(5) calls fib(3) twice, fib(2) three times!
long long fib_slow(int n) {
    if (n <= 1) return n;
    return fib_slow(n - 1) + fib_slow(n - 2);
}

// [Solution] Memoized fibonacci has O(n) time complexity
// Each fib(k) is computed only once and cached
long long fib_memo(int n) {
    // Base case
    if (n <= 1) return n;

    // Check if already computed
    if (memo.find(n) != memo.end()) {
        cout << "Cache hit: fib(" << n << ") = " << memo[n] << endl;
        return memo[n];
    }

    // Compute and store in cache
    cout << "Computing fib(" << n << ")..." << endl;
    memo[n] = fib_memo(n - 1) + fib_memo(n - 2);
    return memo[n];
}

int main() {
    cout << "=== MEMOIZATION DEMO ===" << endl << endl;

    // Demo with small number to show cache hits
    cout << "Computing fib(6) with memoization:" << endl;
    cout << "Result: " << fib_memo(6) << endl << endl;

    // Clear cache for next demo
    memo.clear();

    // Compare performance
    cout << "=== PERFORMANCE COMPARISON ===" << endl;
    int n = 40;

    cout << "\nComputing fib(" << n << ") with memoization..." << endl;
    memo.clear();
    long long result = fib_memo(n);
    cout << "fib(" << n << ") = " << result << endl;

    // Warning: fib_slow(40) takes several seconds!
    // Uncomment below to see the difference
    // cout << "\nComputing fib(" << n << ") without memoization..." << endl;
    // cout << "fib(" << n << ") = " << fib_slow(n) << endl;

    return 0;
}

/*
WHY MEMOIZATION WORKS:

Without memoization - fib(5) call tree:
                    fib(5)
                   /      \
              fib(4)        fib(3)      <- fib(3) computed twice!
             /      \       /      \
        fib(3)    fib(2)  fib(2)   fib(1)
        /    \
    fib(2) fib(1)

Total calls: 15 (exponential growth)

With memoization:
fib(5) -> compute fib(4), fib(3)
fib(4) -> compute fib(3), fib(2)
fib(3) -> compute fib(2), fib(1)
fib(2) -> compute fib(1), fib(0)
fib(1) -> return 1 (base case)
fib(0) -> return 0 (base case)
fib(2) -> CACHE HIT!
fib(3) -> CACHE HIT!

Total unique computations: 6 (linear growth)

Time Complexity:
- Without memo: O(2^n) - exponential
- With memo:    O(n)   - linear
*/
