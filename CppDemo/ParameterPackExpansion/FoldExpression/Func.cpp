#include <iostream>
#include <utility>

/*
函数调用折叠表达式示例

本示例展示如何使用折叠表达式对参数包中的每个参数调用函数。
这是折叠表达式在实际编程中的典型应用场景。

核心概念：
1. 完美转发 (std::forward) - 保持参数的值类别
2. 逗号运算符折叠 - 按顺序执行多个表达式
3. 泛型编程 - 支持任意类型的函数和参数
*/

/**
 * @brief 对所有参数调用指定函数
 * 
 * @tparam Func 函数类型（可以是函数指针、函数对象、lambda等）
 * @tparam Args 参数包类型
 * @param f 要调用的函数
 * @param args 要传递给函数的参数包
 * 
 * 使用一元右折叠表达式：(f(arg1), f(arg2), ..., f(argN))
 * 展开为：f(arg1), (f(arg2), (f(arg3), ... f(argN)))
 * 
 * 注意：虽然语法上是右折叠，但逗号运算符保证从左到右执行
 * 所以实际执行顺序是：f(arg1) → f(arg2) → f(arg3) → ... → f(argN)
 * 
 * std::forward<Args>(args) 确保参数以正确的值类别（左值/右值）传递
 */
template<typename Func, typename... Args>
void call_on_all(Func f, Args&&... args) {
    // 一元右折叠：对每个参数调用函数f
    // 使用完美转发保持参数的原始值类别
    (f(std::forward<Args>(args)), ...);
}

/**
 * @brief 打印整数的平方值
 * 
 * @param x 输入的整数
 * 
 * 这是一个普通函数，将被用作call_on_all的参数
 * 演示了如何将传统函数与折叠表达式结合使用
 */
void print_square(int x) {
    std::cout << x << "^2 = " << x * x << std::endl;
}

int main() {
    std::cout << "=== 函数调用折叠表达式示例 ===" << std::endl;
    std::cout << std::endl;

    // 示例1: 使用lambda表达式
    std::cout << "【示例1: 使用lambda表达式打印数字】" << std::endl;
    std::cout << "调用: call_on_all([](int x) { std::cout << x << \" \"; }, 1, 2, 3, 4, 5)" << std::endl;
    std::cout << "输出: ";
    // lambda表达式 [](int x) { std::cout << x << " "; } 对每个参数调用
    // 折叠展开：(lambda(1), (lambda(2), (lambda(3), (lambda(4), lambda(5)))))
    // 实际执行顺序：1 → 2 → 3 → 4 → 5（逗号运算符保证从左到右）
    call_on_all([](int x) { std::cout << x << " "; }, 1, 2, 3, 4, 5);
    std::cout << std::endl << std::endl;

    // 示例2: 使用普通函数
    std::cout << "【示例2: 使用普通函数计算平方】" << std::endl;
    std::cout << "调用: call_on_all(print_square, 2, 3, 4, 5)" << std::endl;
    std::cout << "输出:" << std::endl;
    // 普通函数print_square对每个参数调用
    // 折叠展开：(print_square(2), (print_square(3), (print_square(4), print_square(5))))
    // 实际执行顺序：2 → 3 → 4 → 5
    call_on_all(print_square, 2, 3, 4, 5);
    std::cout << std::endl;

    /*
    技术要点总结：
    
    1. 折叠表达式类型：
       - 使用一元右折叠：(pack op ...)
       - 这里是：(f(arg1), f(arg2), ...)
    
    2. 完美转发：
       - std::forward<Args>(args) 保持参数的值类别
       - 左值参数保持为左值，右值参数保持为右值
       - 避免不必要的拷贝，提高性能
    
    3. 逗号运算符特性：
       - 从左到右执行表达式
       - 返回最后一个表达式的值（这里被忽略）
       - 适合用于需要按顺序执行多个操作的场景
    
    4. 泛型编程优势：
       - 支持任意类型的函数对象
       - 支持任意数量和类型的参数
       - 编译时展开，零运行时开销
    */

    return 0;
}