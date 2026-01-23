// std::tuple - Fixed-size heterogeneous collection (C++11)
// Header: <tuple>
// Stores multiple values of different types

#include <iostream>
#include <tuple>
#include <string>
#include <utility>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::tuple Demo ===" << endl;
    tuple<int, string, double> t1 = make_tuple(42, "hello", 3.14);
    tuple<int, string, double> t2(100, "world", 2.71);

    cout << "t1: " << get<0>(t1) << ", " << get<1>(t1) << ", " << get<2>(t1) << endl;
    cout << "t2: " << get<int>(t2) << ", " << get<string>(t2) << ", " << get<double>(t2) << endl;
    cout << "Size of t1: " << tuple_size<decltype(t1)>::value << endl;
    cout << endl;
}

void structuredBindingDemo() {
    cout << "=== Structured Binding (C++17) ===" << endl;
    tuple<string, int, double> student = make_tuple("Alice", 25, 95.5);

    auto [name, age, score] = student;
    cout << "Name: " << name << endl;
    cout << "Age: " << age << endl;
    cout << "Score: " << score << endl;
    cout << endl;
}

void tieDemo() {
    cout << "=== std::tie Demo ===" << endl;
    tuple<int, int, int> t = make_tuple(1, 2, 3);
    int a, b, c;
    tie(a, b, c) = t;
    cout << "After tie: a=" << a << ", b=" << b << ", c=" << c << endl;

    tie(b, ignore, c) = t;  // Ignore middle element
    cout << "After tie with ignore: a=" << a << ", b=" << b << ", c=" << c << endl;
    cout << endl;
}

void concatenationDemo() {
    cout << "=== Tuple Concatenation (C++17) ===" << endl;
    tuple<int, char> t1(1, 'a');
    tuple<double, string> t2(3.14, "hello");

    auto t3 = tuple_cat(t1, t2);
    cout << "Concatenated tuple size: " << tuple_size<decltype(t3)>::value << endl;
    cout << "Elements: " << get<0>(t3) << ", " << get<1>(t3) << ", "
         << get<2>(t3) << ", " << get<3>(t3) << endl;
    cout << endl;
}

void comparisonDemo() {
    cout << "=== Tuple Comparison ===" << endl;
    tuple<int, int, int> t1(1, 2, 3);
    tuple<int, int, int> t2(1, 2, 4);
    tuple<int, int, int> t3(1, 2, 3);

    cout << "t1 == t3: " << boolalpha << (t1 == t3) << endl;
    cout << "t1 < t2: " << boolalpha << (t1 < t2) << endl;
    cout << endl;
}

void pairToTupleDemo() {
    cout << "=== Pair vs Tuple ===" << endl;
    pair<int, string> p = make_pair(42, "hello");
    tuple<int, string> t = make_tuple(42, "hello");

    cout << "pair: " << p.first << ", " << p.second << endl;
    cout << "tuple: " << get<0>(t) << ", " << get<1>(t) << endl;
    cout << endl;
}

void swapDemo() {
    cout << "=== Tuple Swap ===" << endl;
    tuple<int, string> t1(1, "one");
    tuple<int, string> t2(2, "two");

    cout << "Before swap: t1=" << get<0>(t1) << ", t2=" << get<0>(t2) << endl;
    t1.swap(t2);
    cout << "After swap: t1=" << get<0>(t1) << ", t2=" << get<0>(t2) << endl;
    cout << endl;
}

void multiReturnDemo() {
    cout << "=== Use Case: Multiple Return Values ===" << endl;

    auto divide = [](int a, int b) -> tuple<int, int, int> {
        return make_tuple(a / b, a % b, a * b);
    };

    auto [quotient, remainder, product] = divide(17, 5);
    cout << "17 / 5:" << endl;
    cout << "  Quotient: " << quotient << endl;
    cout << "  Remainder: " << remainder << endl;
    cout << "  Product: " << product << endl;
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "      std::tuple Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    structuredBindingDemo();
    tieDemo();
    concatenationDemo();
    comparisonDemo();
    pairToTupleDemo();
    swapDemo();
    multiReturnDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::tuple: Fixed-size heterogeneous collection\n";
    cout << "  - Store different types together\n";
    cout << "  - get<index>() or get<type>() to access\n";
    cout << "  - Structured binding (C++17)\n";
    cout << "  - tuple_cat for concatenation\n";
    cout << "  - Perfect for multiple return values\n";

    return 0;
}

/*
Output Summary:
=== Basic ===
get<0>: 42, get<1>: hello, get<2>: 3.14

=== Structured Binding ===
Name: Alice, Age: 25, Score: 95.5

=== Multiple Return Values ===
Quotient: 3, Remainder: 2, Product: 85
*/
