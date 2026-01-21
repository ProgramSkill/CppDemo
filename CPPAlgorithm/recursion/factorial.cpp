// The code is intended to demonstrate recursion by calculating the factorial of a number.
#include <iostream>
using namespace std;

// Calculate n! = n * (n-1) * (n-2) * ... * 1
// Example: 5! = 5 * 4 * 3 * 2 * 1 = 120
int factorial(int n) {
    // [Base Case] When does the recursion stop?
    if (n == 0 || n == 1) {
        cout << "Reached base case, n = " << n << ", returning 1" << endl;
        return 1;
    }

    // [Recursive Step] Break down the problem
    cout << "Computing " << n << "! = " << n << " * " << (n - 1) << "!" << endl;
    int result = n * factorial(n - 1);  // Recursive call
    cout << n << "! = " << result << endl;

    return result;
}

int main() {
    int num = 5;
    int answer = factorial(num);
    return 0;
}

/*
0! = 1
1! = 1
2! = 2 * 1 = 2
3! = 3 * 2 * 1 = 6
4! = 4 * 3 * 2 * 1 = 24
5! = 5 * 4 * 3 * 2 * 1 = 120
6! = 6 * 5 * 4 * 3 * 2 * 1 = 720
[Descending Phase - Recursive Calls]
factorial(5) called
  -> 5 * factorial(4) called
    -> 4 * factorial(3) called
      -> 3 * factorial(2) called
        -> 2 * factorial(1) called
          -> [Reached base case] returns 1
[Ascending Phase - Returning Values]
2 * 1 = 2 returned to factorial(3)
3 * 2 = 6 returned to factorial(4)
4 * 6 = 24 returned to factorial(5)
5 * 24 = 120 returned to main()
*/