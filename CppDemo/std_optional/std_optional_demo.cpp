// std::optional - Wrapper for optional values (C++17)
// Header: <optional>
// Represents a value that may or may not be present

#include <iostream>
#include <optional>
#include <string>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::optional Demo ===" << endl;
    optional<int> opt1;  // Empty
    optional<int> opt2 = 42;  // Has value
    optional<string> opt3 = "hello";
    optional<double> opt4 = nullopt;  // Empty

    cout << "opt1 has value: " << boolalpha << opt1.has_value() << endl;
    cout << "opt2 has value: " << boolalpha << opt2.has_value() << endl;
    cout << "opt2 value: " << *opt2 << endl;
    cout << "opt3 value: " << opt3.value() << endl;
    cout << "opt4 has value: " << boolalpha << opt4.has_value() << endl;
    cout << endl;
}

void accessDemo() {
    cout << "=== Value Access ===" << endl;
    optional<int> opt = 42;

    cout << "*opt: " << *opt << endl;
    cout << "opt.value(): " << opt.value() << endl;
    cout << "opt.value_or(100): " << opt.value_or(100) << endl;

    optional<int> empty;
    cout << "empty.value_or(100): " << empty.value_or(100) << endl;

    try {
        cout << "empty.value(): " << empty.value() << endl;
    } catch (const bad_optional_access& e) {
        cout << "Exception: " << e.what() << endl;
    }
    cout << endl;
}

void operationsDemo() {
    cout << "=== Operations ===" << endl;
    optional<string> opt = "hello";

    opt.emplace("world");
    cout << "After emplace: " << *opt << endl;

    opt.reset();
    cout << "After reset, has_value: " << boolalpha << opt.has_value() << endl;

    opt = "optional";
    if (opt) cout << "After assignment: " << *opt << endl;
    cout << endl;
}

void arrowDemo() {
    cout << "=== Arrow Operator ===" << endl;
    struct Person {
        string name;
        int age;
        string introduce() const { return "I'm " + name + ", " + to_string(age) + " years old"; }
    };

    optional<Person> opt = Person{"Alice", 30};
    cout << "opt->name: " << opt->name << endl;
    cout << "opt->introduce(): " << opt->introduce() << endl;
    cout << endl;
}

void comparisonDemo() {
    cout << "=== Comparison ===" << endl;
    optional<int> opt1 = 42;
    optional<int> opt2 = 42;
    optional<int> opt3 = 100;
    optional<int> opt4;

    cout << "opt1 == opt2: " << boolalpha << (opt1 == opt2) << endl;
    cout << "opt1 < opt3: " << boolalpha << (opt1 < opt3) << endl;
    cout << "opt1 == opt4: " << boolalpha << (opt1 == opt4) << endl;
    cout << "opt4 == nullopt: " << boolalpha << (opt4 == nullopt) << endl;
    cout << endl;
}

void makeOptionalDemo() {
    cout << "=== make_optional ===" << endl;
    auto opt1 = make_optional(42);
    auto opt2 = make_optional<string>("hello");

    cout << "*opt1: " << *opt1 << endl;
    cout << "*opt2: " << *opt2 << endl;
    cout << endl;
}

void findDemo() {
    cout << "=== Use Case: Find Operation ===" << endl;
    vector<int> vec = {1, 2, 3, 4, 5};

    auto find = [&](int value) -> optional<size_t> {
        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] == value) return i;
        }
        return nullopt;
    };

    auto result1 = find(3);
    auto result2 = find(10);

    if (result1) cout << "Found 3 at index: " << *result1 << endl;
    if (!result2) cout << "10 not found" << endl;
    cout << endl;
}

void configDemo() {
    cout << "=== Use Case: Configuration ===" << endl;
    struct Config {
        optional<string> host;
        optional<int> port;
        optional<bool> debug;

        void print() const {
            cout << "Host: " << host.value_or("localhost") << endl;
            cout << "Port: " << port.value_or(8080) << endl;
            cout << "Debug: " << boolalpha << debug.value_or(false) << endl;
        }
    };

    Config config = {"example.com", nullopt, true};
    config.print();
    cout << endl;
}

void factoryDemo() {
    cout << "=== Use Case: Factory Pattern ===" << endl;
    struct Shape {
        virtual ~Shape() = default;
        virtual string name() const = 0;
    };

    struct Circle : Shape {
        string name() const override { return "Circle"; }
    };

    struct Square : Shape {
        string name() const override { return "Square"; }
    };

    auto createShape = [](string type) -> optional<unique_ptr<Shape>> {
        if (type == "circle") return make_unique<Circle>();
        if (type == "square") return make_unique<Square>();
        return nullopt;
    };

    auto shape1 = createShape("circle");
    auto shape2 = createShape("triangle");

    if (shape1) cout << "Created: " << (*shape1)->name() << endl;
    if (!shape2) cout << "Unknown shape type" << endl;
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "      std::optional Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    accessDemo();
    operationsDemo();
    arrowDemo();
    comparisonDemo();
    makeOptionalDemo();
    findDemo();
    configDemo();
    factoryDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::optional: Optional value wrapper\n";
    cout << "  - Represents value that may or may not exist\n";
    cout << "  - Alternative to pointers/exceptions\n";
    cout << "  - *, ->, value(), value_or() for access\n";
    cout << "  - has_value() to check presence\n";
    cout << "  - Perfect for return values, config, factories\n";

    return 0;
}

/*
Output Summary:
=== Basic ===
has_value: false, value: 42

=== Value Access ===
*opt: 42
value_or(100): 42
empty.value_or(100): 100

=== Use Case: Find ===
Found 3 at index: 2
10 not found
*/
