#include <iostream>

// 基础情况：单个参数
template<typename T>
void print_recursive(T arg) {
    std::cout << arg << "\n";
}

// 递归情况：展开第一个参数，递归处理剩余参数
template<typename T, typename... Rest>
void print_recursive(T first, Rest... rest) {
    std::cout << first << " ";
    print_recursive(rest...);  // rest... 展开为剩余参数
}

int main() {
    print_recursive(42, 3.14, "hello", true, 100);
    // Output:
    // 42 3.14 hello 1 
    // 100
}