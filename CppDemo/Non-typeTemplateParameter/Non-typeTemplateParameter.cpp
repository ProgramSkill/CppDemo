//Non-typeTemplateParameter
#include <iostream>
#include <tuple>
#include <array>
#include <string>

#ifdef _WIN32
#include <windows.h>  // For SetConsoleOutputCP
#include <locale>    // For std::locale
#elif __linux__
#include <locale>    // For std::locale
#endif

// ============ Define Base Types ============

// Enum: Text alignment
enum class Alignment { Left, Center, Right };

// Global variable (for pointer parameter)
int global_config = 100;

// Processing functions (for function pointer parameter)
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
int subtract(int a, int b) { return a - b; }

// ============ Main Template: Showcase All Non-Type Parameters ============

template<
    int Priority,                      // ① int: priority level
    std::size_t BufferSize,            // ② std::size_t: buffer size
    bool EnableDebug,                  // ③ bool: enable debug mode
    Alignment TextAlign,               // ④ enum: text alignment
    double ScaleFactor,                // ⑤ double: scaling factor (⚠️ requires C++20)
    int* GlobalPtr,                    // ⑥ pointer: global config pointer
    int (*Operation)(int, int),        // ⑦ function pointer: operation function
    int... Values                      // ⑧ parameter pack: variable integers
>
class DataSystem {
private:
    std::array<char, BufferSize> buffer;

public:
    // Template function: demonstrate std::index_sequence usage
    template<typename... Args>
    void process_tuple(const std::tuple<Args...>& data) {
        process_impl(
            data,
            std::index_sequence_for<Args...>{}  // ⑨ std::index_sequence
        );
    }

    void display_all_params() {
        std::cout << "\n╔════ All Non-Type Parameters Display ════╗\n";

        // ① Display int parameter
        std::cout << "① int Priority: " << Priority << "\n";

        // ② Display std::size_t parameter
        std::cout << "② std::size_t BufferSize: " << BufferSize << " bytes\n";

        // ③ Display bool parameter (compile-time condition)
        std::cout << "③ bool EnableDebug: " << (EnableDebug ? "TRUE" : "FALSE") << "\n";

        // ④ Display enum parameter
        std::cout << "④ Alignment TextAlign: ";
        if constexpr (TextAlign == Alignment::Left) {
            std::cout << "Left\n";
        }
        else if constexpr (TextAlign == Alignment::Center) {
            std::cout << "Center\n";
        }
        else {
            std::cout << "Right\n";
        }

        // ⑤ Display double parameter
        std::cout << "⑤ double ScaleFactor: " << ScaleFactor << "\n";

        // ⑥ Display pointer parameter
        std::cout << "⑥ int* GlobalPtr: " << *GlobalPtr << "\n";

        // ⑦ Display function pointer parameter
        std::cout << "⑦ Function Operation(3, 5): " << Operation(3, 5) << "\n";

        // ⑧ Display parameter pack
        std::cout << "⑧ int... Values: ";
        print_values(std::index_sequence_for<decltype(Values)...>{});
        std::cout << "\n";

        std::cout << "╚════════════════════════════════════════╝\n\n";
    }

    void demonstrate_conditions() {
        std::cout << "╔════ Compile-Time Condition Demo ════╗\n";

        // Conditional compilation based on bool parameter
        if constexpr (EnableDebug) {
            std::cout << "[DEBUG] Debug mode is ON\n";
            std::cout << "[DEBUG] Priority: " << Priority << "\n";
            std::cout << "[DEBUG] Buffer: " << BufferSize << " bytes\n";
        }
        else {
            std::cout << "[PROD] Production mode\n";
        }

        std::cout << "╚═════════════════════════════════════╝\n\n";
    }

    void calculate_with_strategy() {
        std::cout << "╔════ Strategy Pattern Demo (Function Pointer) ════╗\n";
        std::cout << "Operation(10, 20) = " << Operation(10, 20) << "\n";
        std::cout << "Operation(7, 3) = " << Operation(7, 3) << "\n";
        std::cout << "╚══════════════════════════════════════════════╝\n\n";
    }

    void sum_values() {
        std::cout << "╔════ Parameter Pack Sum Demo ════╗\n";
        std::cout << "Values sum = " << (Values + ...) << "\n";
        std::cout << "Count = " << sizeof...(Values) << "\n";
        std::cout << "╚═════════════════════════════════╝\n\n";
    }

private:
    // Use std::index_sequence to unpack tuple
    template<typename Tuple, std::size_t... Is>
    void process_impl(const Tuple& t, std::index_sequence<Is...>) {
        std::cout << "╔════ Tuple Unpacking Demo (std::index_sequence) ════╗\n";
        std::cout << "Tuple elements: ";
        ((std::cout << std::get<Is>(t) << " "), ...);
        std::cout << "\n";
        std::cout << "╚════════════════════════════════════════════════╝\n\n";
    }

    // Print values from parameter pack
    template<std::size_t... Is>
    void print_values(std::index_sequence<Is...>) {
        ((std::cout << Values << " "), ...);
    }
};

// ============ Usage Examples ============

int main() {
    // Set console to UTF-8 encoding to display Unicode characters
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);//设置输出编码
    SetConsoleCP(CP_UTF8);//设置输入编码
    std::locale::global(std::locale("en_US.UTF-8"));//设置全局locale
#elif __linux__
    std::locale::global(std::locale("en_US.UTF-8"));//设置全局locale
#endif
    
    std::cout << "════════════════════════════════════════════════════════════\n";
    std::cout << "      Comprehensive Demonstration of All Non-Type Parameters\n";
    std::cout << "════════════════════════════════════════════════════════════\n";

    // Configuration 1: Debug mode with addition strategy
    using DebugConfig = DataSystem<
        1,                          // ① Priority = 1
        256,                        // ② BufferSize = 256
        true,                       // ③ EnableDebug = true
        Alignment::Center,          // ④ TextAlign = Center
        1.5,                        // ⑤ ScaleFactor = 1.5
        &global_config,             // ⑥ GlobalPtr = &global_config
        add,                        // ⑦ Operation = add
        10, 20, 30, 40, 50          // ⑧ Values = 10, 20, 30, 40, 50
    >;

    DebugConfig debug_system;
    debug_system.display_all_params();
    debug_system.demonstrate_conditions();
    debug_system.calculate_with_strategy();
    debug_system.sum_values();

    std::tuple<int, double, std::string> data1(42, 3.14, "hello");
    debug_system.process_tuple(data1);

    std::cout << "\n════════════════════════════════════════════════════════════\n\n";

    // Configuration 2: Production mode with multiplication strategy
    using ProdConfig = DataSystem<
        3,                          // ① Priority = 3
        4096,                       // ② BufferSize = 4096
        false,                      // ③ EnableDebug = false
        Alignment::Left,            // ④ TextAlign = Left
        2.5,                        // ⑤ ScaleFactor = 2.5
        &global_config,             // ⑥ GlobalPtr = &global_config
        multiply,                   // ⑦ Operation = multiply
        5, 15, 25, 35               // ⑧ Values = 5, 15, 25, 35
    >;

    ProdConfig prod_system;
    prod_system.display_all_params();
    prod_system.demonstrate_conditions();
    prod_system.calculate_with_strategy();
    prod_system.sum_values();

    std::tuple<std::string, int, bool> data2("config", 100, true);
    prod_system.process_tuple(data2);

    std::cout << "\n════════════════════════════════════════════════════════════\n\n";

    // Configuration 3: Subtraction strategy with right alignment
    using ThirdConfig = DataSystem<
        2,                          // ① Priority = 2
        512,                        // ② BufferSize = 512
        true,                       // ③ EnableDebug = true
        Alignment::Right,           // ④ TextAlign = Right
        0.75,                       // ⑤ ScaleFactor = 0.75
        &global_config,             // ⑥ GlobalPtr = &global_config
        subtract,                   // ⑦ Operation = subtract
        1, 2, 3                     // ⑧ Values = 1, 2, 3
    >;

    ThirdConfig third_system;
    third_system.display_all_params();
    third_system.demonstrate_conditions();
    third_system.calculate_with_strategy();
    third_system.sum_values();

    std::tuple<double, int> data3(99.9, 777);
    third_system.process_tuple(data3);

    return 0;
}