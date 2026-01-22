// recursion tail_recursion
// Tail Recursion: Recursive call is the LAST operation
// Can be optimized by compiler to avoid stack overflow
#include <iostream>
using namespace std;

// [Normal Recursion] - NOT tail recursive
// After recursive call, still need to multiply by n
long long factorial_normal(int n) {
    if (n <= 1) return 1;
    return n * factorial_normal(n - 1);  // Must wait for result, then multiply
}

// [Tail Recursion] - IS tail recursive
// Recursive call is the last thing, nothing to do after it returns
long long factorial_tail(int n, long long acc = 1) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);  // Just return the recursive call
}

// [Normal Recursion] Fibonacci - NOT tail recursive
long long fib_normal(int n) {
    if (n <= 1) return n;
    return fib_normal(n - 1) + fib_normal(n - 2);
}

// [Tail Recursion] Fibonacci - IS tail recursive
// Use two accumulators to track fib(n-1) and fib(n-2)
long long fib_tail(int n, long long a = 0, long long b = 1) {
    if (n == 0) return a;
    if (n == 1) return b;
    return fib_tail(n - 1, b, a + b);
}

int main() {
    cout << "=== TAIL RECURSION DEMO ===" << endl << endl;

    // Factorial comparison
    cout << "Factorial:" << endl;
    cout << "Normal: 5! = " << factorial_normal(5) << endl;
    cout << "Tail:   5! = " << factorial_tail(5) << endl << endl;

    // Fibonacci comparison
    cout << "Fibonacci:" << endl;
    cout << "Normal: fib(10) = " << fib_normal(10) << endl;
    cout << "Tail:   fib(10) = " << fib_tail(10) << endl << endl;

    // Large number test (tail recursion won't stack overflow)
    cout << "Large factorial with tail recursion:" << endl;
    cout << "20! = " << factorial_tail(20) << endl;

    return 0;
}

/*
TAIL RECURSION EXPLAINED:

Normal recursion - factorial(5):
factorial(5)
  = 5 * factorial(4)
  = 5 * (4 * factorial(3))
  = 5 * (4 * (3 * factorial(2)))
  = 5 * (4 * (3 * (2 * factorial(1))))
  = 5 * (4 * (3 * (2 * 1)))        <- Stack holds all frames
  = 5 * (4 * (3 * 2))
  = 5 * (4 * 6)
  = 5 * 24
  = 120

Tail recursion - factorial_tail(5, 1):
factorial_tail(5, 1)
  = factorial_tail(4, 5)           <- Can reuse same stack frame!
  = factorial_tail(3, 20)
  = factorial_tail(2, 60)
  = factorial_tail(1, 120)
  = 120

WHY TAIL RECURSION IS BETTER:
1. Compiler can optimize to a loop (Tail Call Optimization)
2. No stack buildup - O(1) space instead of O(n)
3. Won't cause stack overflow for deep recursion

NOTE: C++ compilers may or may not optimize tail recursion.
Use -O2 flag with g++ to enable optimization.
*/
