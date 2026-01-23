// std::string_view - Non-owning reference to string (C++17)
// Header: <string_view>
// Avoids unnecessary string copies

#include <iostream>
#include <string_view>
#include <string>

using namespace std;

void basicDemo() {
    cout << "=== Basic std::string_view Demo ===" << endl;
    string str = "hello";
    string_view sv1 = str;
    string_view sv2 = "world";  // From string literal

    cout << "sv1: " << sv1 << endl;
    cout << "sv2: " << sv2 << endl;
    cout << "sv1 size: " << sv1.size() << endl;
    cout << sv2 << " length: " << sv2.length() << endl;
    cout << endl;
}

void noCopyDemo() {
    cout << "=== No Copy ===" << endl;
    string str = "hello world";
    string_view sv = str;

    cout << "String: " << str << endl;
    cout << "View: " << sv << endl;
    cout << "Same address: " << boolalpha << (sv.data() == str.data()) << endl;
    cout << endl;
}

void substringDemo() {
    cout << "=== Substring View ===" << endl;
    string str = "hello world";
    string_view sv(str);

    auto sub = sv.substr(0, 5);
    cout << "Original: " << sv << endl;
    cout << "Substring (0, 5): " << sub << endl;
    cout << "No allocation occurred!" << endl;
    cout << endl;
}

void comparisonDemo() {
    cout << "=== Comparison ===" << endl;
    string_view sv1 = "hello";
    string_view sv2 = "hello";
    string_view sv3 = "world";

    cout << "sv1 == sv2: " << boolalpha << (sv1 == sv2) << endl;
    cout << "sv1 < sv3: " << boolalpha << (sv1 < sv3) << endl;
    cout << endl;
}

void functionParameterDemo() {
    cout << "=== Function Parameter ===" << endl;
    auto print = [](string_view sv) {
        cout << "View: " << sv << ", size: " << sv.size() << endl;
    };

    string s = "string";
    const char* cstr = "C string";
    print(s);
    print(cstr);
    print("string literal");
    cout << endl;
}

void performanceDemo() {
    cout << "=== Performance Comparison ===" << endl;

    // With string (copy)
    auto withString = [](string s) {
        cout << "Got string: " << s << endl;
    };

    // With string_view (no copy)
    auto withView = [](string_view sv) {
        cout << "Got view: " << sv << endl;
    };

    string str = "hello world";
    withString(str);   // Copy made
    withView(str);     // No copy

    cout << "string_view avoids allocation!" << endl;
    cout << endl;
}

void removePrefixDemo() {
    cout << "=== remove_prefix/remove_suffix ===" << endl;
    string_view sv = "hello world";

    cout << "Original: " << sv << endl;
    sv.remove_prefix(6);
    cout << "After remove_prefix(6): " << sv << endl;

    string_view sv2 = "hello world";
    sv2.remove_suffix(6);
    cout << "After remove_suffix(6): " << sv2 << endl;
    cout << endl;
}

void startsEndsWithDemo() {
    cout << "=== starts_with/ends_with (C++20) ===" << endl;
    string_view sv = "hello world";

    cout << "sv: " << sv << endl;
    cout << "starts_with(\"hello\"): " << boolalpha << sv.starts_with("hello") << endl;
    cout << "ends_with(\"world\"): " << boolalpha << sv.ends_with("world") << endl;
    cout << endl;
}

void findDemo() {
    cout << "=== Find Operations ===" << endl;
    string_view sv = "hello world";

    auto pos = sv.find("world");
    cout << "find(\"world\"): " << pos << endl;

    if (pos != string_view::npos) {
        cout << "Substring found: " << sv.substr(pos) << endl;
    }
    cout << endl;
}

void parsingDemo() {
    cout << "=== Use Case: Parsing ===" << endl;
    string_view sv = "name=John&age=30";

    auto parseKey = [](string_view& sv) -> string_view {
        size_t pos = sv.find('=');
        string_view key = sv.substr(0, pos);
        sv.remove_prefix(pos + 1);
        return key;
    };

    auto parseValue = [](string_view& sv) -> string_view {
        size_t pos = sv.find('&');
        if (pos == string_view::npos) {
            string_view value = sv;
            sv = string_view{};
            return value;
        }
        string_view value = sv.substr(0, pos);
        sv.remove_prefix(pos + 1);
        return value;
    };

    while (!sv.empty()) {
        auto key = parseKey(sv);
        auto value = parseValue(sv);
        cout << key << " = " << value << endl;
    }
    cout << endl;
}

void lifetimeDemo() {
    cout << "=== Lifetime Warning ===" << endl;
    string_view sv;

    {
        string temp = "temporary";
        sv = temp;
        cout << "Inside scope: " << sv << endl;
    }
    // sv is now dangling!

    cout << "Outside scope: ";
    cout << "(undefined behavior - may crash!)" << endl;
    cout << "WARNING: string_view doesn't extend lifetime!" << endl;
    cout << endl;
}

int main() {
    cout << "========================================\n";
    cout << "    std::string_view Demonstration\n";
    cout << "========================================\n\n";

    basicDemo();
    noCopyDemo();
    substringDemo();
    comparisonDemo();
    functionParameterDemo();
    performanceDemo();
    removePrefixDemo();
    startsEndsWithDemo();
    findDemo();
    parsingDemo();
    lifetimeDemo();

    cout << "========================================\n";
    cout << "              Summary\n";
    cout << "========================================\n";
    cout << "std::string_view: Non-owning string reference\n";
    cout << "  - Zero-copy string passing\n";
    cout << "  - Works with string, char*, literals\n";
    cout << "  - Substring without allocation\n";
    cout << "  - WARNING: Doesn't extend lifetime!\n";
    cout << "  - Perfect for function parameters\n";
    cout << "  - Use when read-only access needed\n";

    return 0;
}

/*
Output Summary:
=== Basic ===
sv1: hello, sv2: world

=== No Copy ===
Same address: true

=== Substring ===
Original: hello world
Substring (0, 5): hello

=== Performance ===
string_view avoids allocation!
*/
