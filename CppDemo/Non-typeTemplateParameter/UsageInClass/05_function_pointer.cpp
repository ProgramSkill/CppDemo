// function_pointer (Non-Type Template Parameters in Classes)
#include <iostream>
#include <cmath>

// Basic arithmetic functions
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
int subtract(int a, int b) { return a - b; }
int divide(int a, int b) { return b != 0 ? a / b : 0; }

// Advanced operations
double power(int a, int b) {
    return std::pow(a, b);
}

double square_root(int a, int b) {
    return std::sqrt(a);
}

// Comparison functions
bool is_greater(int a, int b) { return a > b; }
bool is_equal(int a, int b) { return a == b; }

template<int (*Operation)(int, int)>
class DataSystem {
public:
    void calculate() {
        int result = Operation(10, 20);
        std::cout << "Result: " << result << "\n";
    }

    void calculate_custom(int a, int b) {
        int result = Operation(a, b);
        std::cout << "Operation(" << a << ", " << b << ") = " << result << "\n";
    }

    void show_values(int a, int b) {
        std::cout << "a = " << a << ", b = " << b << "\n";
        std::cout << "Operation result: " << Operation(a, b) << "\n";
    }
};

template<double (*MathFunc)(int, int)>
class AdvancedCalculator {
public:
    void calculate(int a, int b) {
        double result = MathFunc(a, b);
        std::cout << "Advanced Result: " << result << "\n";
    }
};

template<bool (*Predicate)(int, int)>
class Comparator {
public:
    void compare(int a, int b) {
        bool result = Predicate(a, b);
        std::string op_name = Predicate == is_greater ? ">" : "==";
        std::cout << "Is " << a << " " << op_name << " " << b << "? "
            << (result ? "Yes" : "No") << "\n";
    }
};

// Custom operation collection
namespace CustomOps {
    int max_value(int a, int b) { return a > b ? a : b; }
    int min_value(int a, int b) { return a < b ? a : b; }
    int gcd(int a, int b) {
        while (b) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }
    int lcm(int a, int b) {
        return (a / gcd(a, b)) * b;
    }
}

int main() {
    std::cout << "=== Function Pointer Invocation ===\n\n";

    // Basic arithmetic
    std::cout << "--- Basic Arithmetic ---\n";
    DataSystem<add> adder;
    adder.calculate();
    adder.show_values(25, 15);

    std::cout << "\n";

    DataSystem<multiply> multiplier;
    multiplier.calculate();
    multiplier.show_values(6, 7);

    std::cout << "\n";

    DataSystem<subtract> subtractor;
    subtractor.calculate();
    subtractor.show_values(100, 35);

    std::cout << "\n";

    DataSystem<divide> divisor;
    divisor.calculate();
    divisor.show_values(42, 6);

    // Custom operations
    std::cout << "\n--- Custom Operations ---\n";
    DataSystem<CustomOps::max_value> max_op;
    std::cout << "Max value operation:\n";
    max_op.show_values(15, 23);

    std::cout << "\n";

    DataSystem<CustomOps::min_value> min_op;
    std::cout << "Min value operation:\n";
    min_op.show_values(15, 23);

    std::cout << "\n";

    DataSystem<CustomOps::gcd> gcd_op;
    std::cout << "GCD operation:\n";
    gcd_op.show_values(48, 18);

    std::cout << "\n";

    DataSystem<CustomOps::lcm> lcm_op;
    std::cout << "LCM operation:\n";
    lcm_op.show_values(12, 18);

    // Comparison functions
    std::cout << "\n--- Comparators ---\n";
    Comparator<is_greater> gt_comp;
    gt_comp.compare(10, 5);
    gt_comp.compare(5, 10);

    std::cout << "\n";

    Comparator<is_equal> eq_comp;
    eq_comp.compare(10, 10);
    eq_comp.compare(10, 5);

    // Advanced calculations (floating point results)
    std::cout << "\n--- Advanced Calculations ---\n";
    AdvancedCalculator<power> power_calc;
    power_calc.calculate(2, 10);

    std::cout << "\nFunction pointers are bound at compile time, supporting full inline optimization!\n";

    return 0;
}