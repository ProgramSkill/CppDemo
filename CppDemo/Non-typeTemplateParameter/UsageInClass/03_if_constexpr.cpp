// if constexpr (Non-Type Template Parameters in Classes)
#include <iostream>

template<bool EnableDebug>  // Boolean value as template parameter
class DataSystem {
public:
    void log() {
        if constexpr (EnableDebug) {
            std::cout << "[DEBUG] Processing data...\n";
            std::cout << "[DEBUG] Resource allocation successful\n";
            std::cout << "[DEBUG] Data validation complete\n";
        }
        else {
            std::cout << "[PROD] Processing data...\n";
        }
    }

    void process(int data) {
        if constexpr (EnableDebug) {
            std::cout << "[DEBUG] Processing value: " << data << "\n";
            std::cout << "[DEBUG] Value is " << (data > 100 ? "large" : "small") << "\n";
        }

        // Actual processing logic
        int result = data * 2;

        if constexpr (EnableDebug) {
            std::cout << "[DEBUG] Result: " << result << "\n";
        }
    }
};

template<bool EnableTiming>
class PerformanceMonitor {
public:
    void start() {
        if constexpr (EnableTiming) {
            std::cout << "[TIMER] Starting timer...\n";
        }
    }

    void operation() {
        std::cout << "Executing operation...\n";

        if constexpr (EnableTiming) {
            std::cout << "[TIMER] Operation completed\n";
        }
    }
};

template<bool IsProduction>
class Config {
public:
    void show_settings() {
        if constexpr (IsProduction) {
            std::cout << "=== PRODUCTION CONFIG ===\n";
            std::cout << "Log Level: ERROR\n";
            std::cout << "Optimization: Maximum\n";
            std::cout << "Debug Info: OFF\n";
            std::cout << "Performance: Critical\n";
        }
        else {
            std::cout << "=== DEVELOPMENT CONFIG ===\n";
            std::cout << "Log Level: DEBUG\n";
            std::cout << "Optimization: Standard\n";
            std::cout << "Debug Info: ON\n";
            std::cout << "Performance: Flexible\n";
        }
    }

    int get_log_level() {
        if constexpr (IsProduction) {
            return 0;  // ERROR only
        }
        else {
            return 3;  // DEBUG
        }
    }

    void detailed_config() {
        std::cout << "\nDetailed Configuration:\n";
        std::cout << "Mode: ";

        if constexpr (IsProduction) {
            std::cout << "PRODUCTION\n";
            std::cout << "  - Minimal logging\n";
            std::cout << "  - Error handling only\n";
            std::cout << "  - Maximum performance\n";
        }
        else {
            std::cout << "DEVELOPMENT\n";
            std::cout << "  - Verbose logging\n";
            std::cout << "  - Full error tracking\n";
            std::cout << "  - Debug symbols included\n";
        }
    }
};

int main() {
    std::cout << "=== Compile-Time Conditional Logic (if constexpr) ===\n\n";

    // DEBUG version
    std::cout << "--- DEBUG Mode ---\n";
    DataSystem<true> debug_system;
    debug_system.log();
    debug_system.process(150);

    std::cout << "\n--- PRODUCTION Mode ---\n";
    DataSystem<false> prod_system;
    prod_system.log();
    prod_system.process(150);

    // Performance monitoring
    std::cout << "\n--- With Timing ---\n";
    PerformanceMonitor<true> monitor_with_timing;
    monitor_with_timing.start();
    monitor_with_timing.operation();

    std::cout << "\n--- Without Timing ---\n";
    PerformanceMonitor<false> monitor_no_timing;
    monitor_no_timing.start();
    monitor_no_timing.operation();

    // Configuration management
    std::cout << "\n--- Development Configuration ---\n";
    Config<false> dev_config;
    dev_config.show_settings();
    dev_config.detailed_config();

    std::cout << "\n--- Production Configuration ---\n";
    Config<true> prod_config;
    prod_config.show_settings();
    prod_config.detailed_config();

    std::cout << "\nCompile-time decision: DEBUG code completely removed in Release builds!\n";

    return 0;
}

/*
Aspect           if constexpr          Regular if
Decision Time    Compile-time          Runtime
Inactive Branch  Completely removed    Generated but not executed
Performance      Zero overhead         Conditional judgment overhead
Type Check       Only active branch    Both branches checked
*/