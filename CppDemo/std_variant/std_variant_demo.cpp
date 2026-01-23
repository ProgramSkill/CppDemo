// std::variant - Type-safe union (C++17)
// Header: <variant>
// Stores one of several types at a time

#include <iostream>
#include <variant>
#include <string>
#include <vector>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::variant Demo ===" << endl;
    variant<int, double, string> v1;
    variant<int, double, string> v2 = 42;
    variant<int, double, string> v3 = 3.14;
    variant<int, double, string> v4 = "hello";

    cout << "v2 index: " << v2.index() << ", value: " << get<int>(v2) << endl;
    cout << "v3 index: " << v3.index() << ", value: " << get<double>(v3) << endl;
    cout << "v4 index: " << v4.index() << ", value: " << get<string>(v4) << endl;
    cout << endl;
}

void访问Demo() {
    cout << "=== Access Methods ===" << endl;
    variant<int, string> v = 42;

    cout << "get<int>: " << get<int>(v) << endl;

    if (holds_alternative<int>(v)) {
        cout << "Holds int: " << get<int>(v) << endl;
    }

    v = "hello";
    cout << "After assignment, index: " << v.index() << endl;

    try {
        cout << get<int>(v) << endl;
    } catch (const bad_variant_access& e) {
        cout << "Exception: " << e.what() << endl;
    }

    auto visitor = [](auto&& arg) {
        cout << "Value: " << arg << ", type: " << typeid(arg).name() << endl;
    };
    visit(visitor, v);
    cout << endl;
}

void get_ifDemo() {
    cout << "=== get_if ===" << endl;
    variant<int, double, string> v = 3.14;

    if (auto ptr = get_if<double>(&v)) {
        cout << "Double value: " << *ptr << endl;
    }

    if (auto ptr = get_if<int>(&v)) {
        cout << "Int value: " << *ptr << endl;
    } else {
        cout << "Not holding int" << endl;
    }
    cout << endl;
}

void visitorDemo() {
    cout << "=== Visitor Pattern ===" << endl;
    variant<int, double, string> v = 42;

    auto visitor = [](auto&& arg) {
        using T = decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, int>) {
            cout << "Int: " << arg << endl;
        } else if constexpr (is_same_v<T, double>) {
            cout << "Double: " << arg << endl;
        } else if constexpr (is_same_v<T, string>) {
            cout << "String: " << arg << endl;
        }
    };

    visit(visitor, v);
    v = 3.14;
    visit(visitor, v);
    v = "hello";
    visit(visitor, v);
    cout << endl;
}

void comparisonDemo() {
    cout << "=== Comparison ===" << endl;
    variant<int, string> v1 = 42;
    variant<int, string> v2 = 42;
    variant<int, string> v3 = 100;

    cout << "v1 == v2: " << boolalpha << (v1 == v2) << endl;
    cout << "v1 < v3: " << boolalpha << (v1 < v3) << endl;

    variant<int, string> v4 = "hello";
    // cout << "v1 < v4: " << (v1 < v4) << endl;  // Won't compile!
    cout << endl;
}

void assignmentDemo() {
    cout << "=== Assignment ===" << endl;
    variant<int, string> v = 42;
    cout << "Initial: " << get<int>(v) << endl;

    v = "hello";
    cout << "After string assignment: " << get<string>(v) << endl;

    v.emplace<int>(100);
    cout << "After emplace: " << get<int>(v) << endl;
    cout << endl;
}

void anyTypeDemo() {
    cout << "=== Use Case: Any Type ===" << endl;
    using AnyType = variant<int, double, string, vector<int>>;

    vector<AnyType> values;
    values.push_back(42);
    values.push_back(3.14);
    values.push_back("hello");
    values.push_back(vector<int>{1, 2, 3});

    for (const auto& v : values) {
        visit([](auto&& arg) {
            using T = decay_t<decltype(arg)>;
            if constexpr (is_same_v<T, vector<int>>) {
                cout << "Vector: ";
                for (int x : arg) cout << x << " ";
                cout << endl;
            } else {
                cout << arg << endl;
            }
        }, v);
    }
    cout << endl;
}

void resultDemo() {
    cout << "=== Use Case: Result Type ===" << endl;
    struct Error {
        string message;
        int code;
    };

    using Result = variant<int, Error>;

    auto divide = [](int a, int b) -> Result {
        if (b == 0) {
            return Error{"Division by zero", 400};
        }
        return a / b;
    };

    auto r1 = divide(10, 2);
    auto r2 = divide(10, 0);

    if (holds_alternative<int>(r1)) {
        cout << "Result: " << get<int>(r1) << endl;
    }

    if (holds_alternative<Error>(r2)) {
        auto e = get<Error>(r2);
        cout << "Error: " << e.message << " (code: " << e.code << ")" << endl;
    }
    cout << endl;
}

void expressionDemo() {
    cout << "=== Use Case: Expression Evaluator ===" << endl;
    struct Add;
    struct Mul;
    struct Num;

    using Expr = variant<shared_ptr<Add>, shared_ptr<Mul>, int>;

    struct Add {
        Expr left, right;
        Add(Expr l, Expr r) : left(l), right(r) {}
    };

    struct Mul {
        Expr left, right;
        Mul(Expr l, Expr r) : left(l), right(r) {}
    };

    auto eval = [](auto&& self, Expr expr) -> int {
        return visit([&self](auto&& arg) -> int {
            using T = decay_t<decltype(arg)>;
            if constexpr (is_same_v<T, int>) {
                return arg;
            } else if constexpr (is_same_v<T, shared_ptr<Add>>) {
                return self(self, arg->left) + self(self, arg->right);
            } else if constexpr (is_same_v<T, shared_ptr<Mul>>) {
                return self(self, arg->left) * self(self, arg->right);
            }
        }, expr);
    };

    Expr expr = make_shared<Add>(
        make_shared<Mul>(2, 3),
        make_shared<Mul>(4, 5)
    );

    cout << "Expression: (2*3) + (4*5)" << endl;
    cout << "Result: " << eval(eval, expr) << endl;
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "      std::variant Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    访问Demo();
    get_ifDemo();
    visitorDemo();
    comparisonDemo();
    assignmentDemo();
    anyTypeDemo();
    resultDemo();
    expressionDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::variant: Type-safe union\n";
    cout << "  - Holds one of several types\n";
    cout << "  - Type-safe access with get<>, get_if<>\n";
    cout << "  - visit() for pattern matching\n";
    cout << "  - No invalid state (unlike union)\n";
    cout << "  - Perfect for result types, expressions\n";

    return 0;
}

/*
Output Summary:
=== Basic ===
v2 index: 0, value: 42
v3 index: 1, value: 3.14

=== Visitor ===
Int: 42
Double: 3.14
String: hello

=== Result Type ===
Result: 5
Error: Division by zero (code: 400)
*/
