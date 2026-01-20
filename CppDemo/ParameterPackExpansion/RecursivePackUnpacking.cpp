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
    // 执行流程：
    // 1. print_recursive(42, 3.14, "hello", true, 100)
    //    - 输出: 42 
    //    - 调用: print_recursive(3.14, "hello", true, 100)
    // 2. print_recursive(3.14, "hello", true, 100)
    //    - 输出: 3.14 
    //    - 调用: print_recursive("hello", true, 100)
    // 3. print_recursive("hello", true, 100)
    //    - 输出: hello 
    //    - 调用: print_recursive(true, 100)
    // 4. print_recursive(true, 100)
    //    - 输出: 1 (bool true 输出为 1)
    //    - 调用: print_recursive(100)
    // 5. print_recursive(100) [基础情况]
    //    - 输出: 100\n
    // 最终输出: 42 3.14 hello 1 100
    print_recursive(42, 3.14, "hello", true, 100);
}