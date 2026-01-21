// enum_judgment (Non-Type Template Parameters in Classes)
#include <iostream>

enum class Alignment { Left, Center, Right };

template<Alignment TextAlign>
class DataSystem {
public:
    std::string get_alignment() {
        if constexpr (TextAlign == Alignment::Left) {
            return "Left";
        }
        else if constexpr (TextAlign == Alignment::Center) {
            return "Center";
        }
        else {
            return "Right";
        }
    }

    void print_text(const std::string& text, int width) {
        int text_len = text.length();
        int padding = width - text_len;

        if constexpr (TextAlign == Alignment::Left) {
            std::cout << text;
            for (int i = 0; i < padding; i++) std::cout << " ";
        }
        else if constexpr (TextAlign == Alignment::Center) {
            int left_pad = padding / 2;
            int right_pad = padding - left_pad;
            for (int i = 0; i < left_pad; i++) std::cout << " ";
            std::cout << text;
            for (int i = 0; i < right_pad; i++) std::cout << " ";
        }
        else {
            for (int i = 0; i < padding; i++) std::cout << " ";
            std::cout << text;
        }
        std::cout << "|\n";
    }

    void show_info() {
        std::cout << "Alignment: " << get_alignment() << "\n";
    }
};

enum class LogLevel { Error, Warning, Info, Debug };

template<LogLevel Level>
class Logger {
public:
    std::string get_level_name() {
        if constexpr (Level == LogLevel::Error) {
            return "ERROR";
        }
        else if constexpr (Level == LogLevel::Warning) {
            return "WARNING";
        }
        else if constexpr (Level == LogLevel::Info) {
            return "INFO";
        }
        else {
            return "DEBUG";
        }
    }

    void log(const std::string& message) {
        std::string prefix;

        if constexpr (Level == LogLevel::Error) {
            prefix = "[ERROR]";
        }
        else if constexpr (Level == LogLevel::Warning) {
            prefix = "[WARNING]";
        }
        else if constexpr (Level == LogLevel::Info) {
            prefix = "[INFO]";
        }
        else {
            prefix = "[DEBUG]";
        }

        std::cout << prefix << " " << message << "\n";
    }

    bool should_log() {
        if constexpr (Level == LogLevel::Error) {
            return true;
        }
        else if constexpr (Level == LogLevel::Warning) {
            return true;
        }
        else if constexpr (Level == LogLevel::Info) {
            return true;
        }
        else {
            return true;
        }
    }
};

enum class ColorMode { RGB, CMYK, Grayscale };

template<ColorMode Mode>
class ColorProcessor {
public:
    std::string get_mode_name() {
        if constexpr (Mode == ColorMode::RGB) {
            return "RGB (Red, Green, Blue)";
        }
        else if constexpr (Mode == ColorMode::CMYK) {
            return "CMYK (Cyan, Magenta, Yellow, Black)";
        }
        else {
            return "Grayscale";
        }
    }

    int get_channels() {
        if constexpr (Mode == ColorMode::RGB) {
            return 3;
        }
        else if constexpr (Mode == ColorMode::CMYK) {
            return 4;
        }
        else {
            return 1;
        }
    }

    void show_info() {
        std::cout << "Mode: " << get_mode_name() << "\n";
        std::cout << "Channels: " << get_channels() << "\n";
    }
};

int main() {
    std::cout << "=== Enum Value Judgment ===\n\n";

    // Text alignment
    std::cout << "--- Text Alignment ---\n";
    DataSystem<Alignment::Left> left_align;
    DataSystem<Alignment::Center> center_align;
    DataSystem<Alignment::Right> right_align;

    std::cout << "Left:  |";
    left_align.print_text("Hello", 20);

    std::cout << "Center:|";
    center_align.print_text("Hello", 20);

    std::cout << "Right: |";
    right_align.print_text("Hello", 20);

    // Log levels
    std::cout << "\n--- Log Levels ---\n";
    Logger<LogLevel::Debug> debug_logger;
    Logger<LogLevel::Info> info_logger;
    Logger<LogLevel::Warning> warning_logger;
    Logger<LogLevel::Error> error_logger;

    debug_logger.log("This is a debug message");
    info_logger.log("Application started");
    warning_logger.log("Memory usage high");
    error_logger.log("Critical failure detected!");

    // Color modes
    std::cout << "\n--- Color Modes ---\n";
    ColorProcessor<ColorMode::RGB> rgb;
    ColorProcessor<ColorMode::CMYK> cmyk;
    ColorProcessor<ColorMode::Grayscale> gray;

    std::cout << "RGB Mode:\n";
    rgb.show_info();

    std::cout << "\nCMYK Mode:\n";
    cmyk.show_info();

    std::cout << "\nGrayscale Mode:\n";
    gray.show_info();

    std::cout << "\nEnum parameters provide type-safe compile-time branch selection!\n";

    return 0;
}