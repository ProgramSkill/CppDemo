// parameter_pack (Non-Type Template Parameters in Classes)
#include <iostream>

// Basic parameter pack
template<int... Values>
class DataSystem {
public:
    void sum_all() {
        int total = (Values + ... + 0);  // Binary right fold: (10 + (20 + (30 + (40 + (50 + 0)))))
        std::cout << "Sum: " << total << "\n";
    }

    void print_all() {
        std::cout << "Values: ";
		((std::cout << Values << " "), ...);  // Unary right fold: ((cout << 10 << " "), ((cout << 20 << " "), (cout << 30 << " ")))
        std::cout << "\n";
    }

    void count() {
        std::cout << "Count: " << sizeof...(Values) << "\n";
    }

    void product_all() {
        int product = (Values * ... * 1);  // Binary right fold: (2 * (3 * (4 * 1)))
        std::cout << "Product: " << product << "\n";
    }
};

// Parameter pack calculations
template<int... Numbers>
class Calculator {
public:
    void show_statistics() {
        std::cout << "Numbers: ";
        ((std::cout << Numbers << " "), ...);
        std::cout << "\n";

        int sum = (Numbers + ... + 0);
        int count = sizeof...(Numbers);
        int average = sum / count;

        std::cout << "Sum: " << sum << "\n";
        std::cout << "Count: " << count << "\n";
        std::cout << "Average: " << average << "\n";
    }

    void show_max() {
        int max_val = std::max({ Numbers... });
        int min_val = std::min({ Numbers... });

        std::cout << "Max: " << max_val << "\n";
        std::cout << "Min: " << min_val << "\n";
    }
};

// Character parameter pack
template<char... Chars>
class CharProcessor {
public:
    void print_chars() {
        std::cout << "Characters: ";
        ((std::cout << Chars), ...);
        std::cout << "\n";
    }

    void count_chars() {
        std::cout << "Character count: " << sizeof...(Chars) << "\n";
    }

    void print_with_commas() {
        std::cout << "Characters: ";
        bool first = true;
        ((std::cout << (first ? "" : ", ") << Chars, first = false), ...);
        std::cout << "\n";
    }
};

// Type parameter pack (template parameter pack)
template<typename... Types>
class TypeInfo {
public:
    void show_count() {
        std::cout << "Number of types: " << sizeof...(Types) << "\n";
    }
};

// Mixed parameter pack
template<double... Coefficients>
class PolynomialEvaluator {
public:
    void show_coefficients() {
        std::cout << "Coefficients: ";
        ((std::cout << Coefficients << " "), ...);
        std::cout << "\n";
    }

    double evaluate(double x) {
        double result = 0.0;
        int power = 0;
        ((result += Coefficients * std::pow(x, power++)), ...);
        return result;
    }

    void evaluate_and_print(double x) {
        double result = evaluate(x);
        std::cout << "f(" << x << ") = " << result << "\n";
    }
};

// Helper functions
namespace Helper {
    template<int... Nums>
    int total(DataSystem<Nums...>& ds) {
        return (Nums + ... + 0);
    }
}

int main() {
    std::cout << "=== 6 Parameter Pack Expansion ===\n\n";

    // Integer parameter pack
    std::cout << "--- Integer Parameter Pack ---\n";
    DataSystem<10, 20, 30, 40, 50> sys1;
    sys1.print_all();
    sys1.sum_all();
    sys1.product_all();
    sys1.count();

    std::cout << "\n";

    DataSystem<5, 15, 25> sys2;
    sys2.print_all();
    sys2.sum_all();
    sys2.count();

    // Calculator
    std::cout << "\n--- Calculator with Statistics ---\n";
    Calculator<15, 25, 35, 45, 55> calc;
    calc.show_statistics();
    std::cout << "\n";
    calc.show_max();

    std::cout << "\n";

    Calculator<100, 50, 75, 25> calc2;
    calc2.show_statistics();
    std::cout << "\n";
    calc2.show_max();

    // Character parameter pack
    std::cout << "\n--- Character Parameter Pack ---\n";
    CharProcessor<'H', 'e', 'l', 'l', 'o'> char_proc1;
    char_proc1.print_chars();
    char_proc1.count_chars();

    std::cout << "\n";

    CharProcessor<'C', '+', '+', '2', '0'> char_proc2;
    char_proc2.print_with_commas();
    char_proc2.count_chars();

    // Multi-element parameter pack
    std::cout << "\n--- Multiple Parameter Packs ---\n";
    DataSystem<1, 1, 2, 3, 5, 8, 13> fibonacci;
    std::cout << "Fibonacci sequence:\n";
    fibonacci.print_all();
    fibonacci.sum_all();

    std::cout << "\n";

    DataSystem<2, 4, 8, 16, 32, 64> powers_of_two;
    std::cout << "Powers of two:\n";
    powers_of_two.print_all();
    powers_of_two.product_all();

    // Floating point parameter pack (polynomial coefficients)
    std::cout << "\n--- Polynomial Coefficients ---\n";
    PolynomialEvaluator<1.0, 2.0, 3.0> poly;  // 1 + 2x + 3x^2
    poly.show_coefficients();
    poly.evaluate_and_print(0.0);
    poly.evaluate_and_print(1.0);
    poly.evaluate_and_print(2.0);

    std::cout << "\nParameter packs are expanded at compile time, supporting fold expression optimization!\n";

    return 0;
}