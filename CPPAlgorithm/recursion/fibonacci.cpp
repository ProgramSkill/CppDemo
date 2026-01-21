//recursion fibonacci 
#include <iostream>
using namespace std;

// Calculate the nth Fibonacci number
// Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21...
// Each number is the sum of the two preceding ones
// f(n) = f(n-1) + f(n-2)
int fibonacci(int n) {
    // [Base Case] Stop condition
    if (n <= 1) {
        cout << "Base case reached: fib(" << n << ") = " << n << endl;
        return n;
    }

    // [Recursive Step] Break down the problem
    cout << "Computing fib(" << n << ") = fib(" << (n - 1) << ") + fib(" << (n - 2) << ")" << endl;
    int result = fibonacci(n - 1) + fibonacci(n - 2);  // Recursive calls
    cout << "fib(" << n << ") = " << result << endl;
    return result;
}

int main() {
    cout << "=== Fibonacci Sequence ===" << endl;
    for (int i = 0; i <= 6; i++) {
        cout << "\n--- Computing fib(" << i << ") ---" << endl;
        int answer = fibonacci(i);
        cout << "Final result: fib(" << i << ") = " << answer << "\n" << endl;
    }
    return 0;
}

/*
Example: Computing fib(5)

                    fib(5)
                   /      \
              fib(4)        fib(3)
             /      \       /      \
        fib(3)    fib(2)  fib(2)   fib(1)
        /    \     /   \   /   \
    fib(2) fib(1) fib(1) fib(0) fib(1) fib(0)
    /   \
fib(1) fib(0)

Notice: fib(3) and fib(2) are computed multiple times!
This is inefficient and why we need optimization (memoization)
*/