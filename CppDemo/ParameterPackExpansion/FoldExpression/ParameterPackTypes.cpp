#include <iostream>

// 1. 类型参数包（运行时的值）
template<typename... Args>
void print_runtime(Args... args) {
    std::cout << "类型参数包（运行时值）: ";
    ((std::cout << args << " "), ...);
    std::cout << std::endl;
}

// 2. 非类型参数包（编译期的值）
template<int... nums>
void print_compile_time() {
    std::cout << "非类型参数包（编译期值）: ";
    ((std::cout << nums << " "), ...);
    std::cout << std::endl;
}

int main() {
    // 类型参数包：可以传变量
    int x = 10;
    int y = 20;
    print_runtime(x, y, 30);  // OK：运行时的值

    // 非类型参数包：必须是常量
    print_compile_time<10, 20, 30>();  // OK：编译期常量
    // print_compile_time<x, y, 30>();  // 错误！x 和 y 是变量

    return 0;
}