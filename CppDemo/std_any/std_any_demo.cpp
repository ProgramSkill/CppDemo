// std::any - Type-safe container for single values of any type (C++17)
// Header: <any>
// Stores any copyable type

#include <iostream>
#include <any>
#include <string>
#include <vector>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::any Demo ===" << endl;
    any a1;
    any a2 = 42;
    any a3 = string("hello");
    any a4 = 3.14;

    cout << "a1.has_value: " << boolalpha << a1.has_value() << endl;
    cout << "a2.type: " << a2.type().name() << endl;
    cout << "any_cast<int>(a2): " << any_cast<int>(a2) << endl;
    cout << "any_cast<string>(a3): " << any_cast<string>(a3) << endl;
    cout << endl;
}

void castDemo() {
    cout << "=== any_cast ===" << endl;
    any a = 42;

    cout << "any_cast<int>: " << any_cast<int>(a) << endl;

    if (auto ptr = any_cast<int>(&a)) {
        cout << "Pointer cast: " << *ptr << endl;
    }

    try {
        cout << any_cast<string>(a) << endl;
    } catch (const bad_any_cast& e) {
        cout << "Exception: " << e.what() << endl;
    }
    cout << endl;
}

void assignmentDemo() {
    cout << "=== Assignment ===" << endl;
    any a;
    a = 42;
    cout << "After int assignment: " << any_cast<int>(a) << endl;

    a = string("hello");
    cout << "After string assignment: " << any_cast<string>(a) << endl;

    a.emplace<vector<int>>({1, 2, 3});
    auto vec = any_cast<vector<int>>(a);
    cout << "After emplace vector: ";
    for (int x : vec) cout << x << " ";
    cout << endl;

    a.reset();
    cout << "After reset, has_value: " << boolalpha << a.has_value() << endl;
    cout << endl;
}

void make_anyDemo() {
    cout << "=== make_any ===" << endl;
    auto a1 = make_any<int>(42);
    auto a2 = make_any<string>("hello");

    cout << "*any_cast<int>(&a1): " << *any_cast<int>(&a1) << endl;
    cout << "*any_cast<string>(&a2): " << *any_cast<string>(&a2) << endl;
    cout << endl;
}

void typeCheckDemo() {
    cout << "=== Type Checking ===" << endl;
    vector<any> values;
    values.push_back(42);
    values.push_back(3.14);
    values.push_back(string("hello"));
    values.push_back('a');

    for (const auto& a : values) {
        if (a.type() == typeid(int)) {
            cout << "int: " << any_cast<int>(a) << endl;
        } else if (a.type() == typeid(double)) {
            cout << "double: " << any_cast<double>(a) << endl;
        } else if (a.type() == typeid(string)) {
            cout << "string: " << any_cast<string>(a) << endl;
        } else if (a.type() == typeid(char)) {
            cout << "char: " << any_cast<char>(a) << endl;
        }
    }
    cout << endl;
}

void heterogeneousContainerDemo() {
    cout << "=== Heterogeneous Container ===" << endl;
    vector<any> items;

    items.push_back(42);
    items.push_back(string("hello"));
    items.push_back(3.14);
    items.push_back(vector<int>{1, 2, 3});

    for (size_t i = 0; i < items.size(); ++i) {
        const auto& item = items[i];
        cout << "Item " << i << " type: " << item.type().name() << endl;
    }
    cout << endl;
}

void messagePassingDemo() {
    cout << "=== Use Case: Message Passing ===" << endl;
    queue<any> messages;

    messages.push(42);
    messages.push(string("hello"));
    messages.push(3.14);

    while (!messages.empty()) {
        any msg = messages.front(); messages.pop();

        if (msg.type() == typeid(int)) {
            cout << "Int message: " << any_cast<int>(msg) << endl;
        } else if (msg.type() == typeid(string)) {
            cout << "String message: " << any_cast<string>(msg) << endl;
        } else if (msg.type() == typeid(double)) {
            cout << "Double message: " << any_cast<double>(msg) << endl;
        }
    }
    cout << endl;
}

void propertyMapDemo() {
    cout << "=== Use Case: Property Map ===" << endl;
    map<string, any> props;

    props["name"] = string("Alice");
    props["age"] = 30;
    props["height"] = 1.75;
    props["active"] = true;

    cout << "Properties:" << endl;
    for (const auto& [key, value] : props) {
        cout << "  " << key << ": ";

        if (value.type() == typeid(string)) {
            cout << any_cast<string>(value);
        } else if (value.type() == typeid(int)) {
            cout << any_cast<int>(value);
        } else if (value.type() == typeid(double)) {
            cout << any_cast<double>(value);
        } else if (value.type() == typeid(bool)) {
            cout << boolalpha << any_cast<bool>(value);
        }
        cout << endl;
    }
    cout << endl;
}

void comparisonDemo() {
    cout << "=== any vs variant vs void* ===" << endl;
    cout << "std::any:" << endl;
    cout << "  + Can hold any copyable type" << endl;
    cout << "  + Type-safe" << endl;
    cout << "  - Requires any_cast<>" << endl;
    cout << "  - Heap allocation (usually)" << endl;

    cout << "\nstd::variant:" << endl;
    cout << "  + No heap allocation (small types)" << endl;
    cout << "  + Type-safe" << endl;
    cout << "  - Must specify types upfront" << endl;

    cout << "\nvoid*:" << endl;
    cout << "  + Can hold anything" << endl;
    cout << "  - Not type-safe" << endl;
    cout << "  - Manual memory management" << endl;
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "        std::any Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    castDemo();
    assignmentDemo();
    make_anyDemo();
    typeCheckDemo();
    heterogeneousContainerDemo();
    messagePassingDemo();
    propertyMapDemo();
    comparisonDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::any: Type-safe container for any type\n";
    cout << "  - Holds any copyable type\n";
    cout << "  - Type-safe with any_cast<>\n";
    cout << "  - Check type with .type() == typeid(T)\n";
    cout << "  - Use when types are unknown at compile time\n";
    cout << "  - Prefer variant when types are known\n";

    return 0;
}

/*
Output Summary:
=== Basic ===
has_value: false
type: int
any_cast<int>(a2): 42

=== Type Checking ===
int: 42
double: 3.14
string: hello
char: a
*/
