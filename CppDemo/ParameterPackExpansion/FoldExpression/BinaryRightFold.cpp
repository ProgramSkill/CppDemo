#include <iostream>
#include <string>
#include <vector>

// 示例1: 带初始值的加法
template<typename... Args>
auto sum_with_init(Args... args) {
    return (args + ... + 0);  // 二元右折叠，初始值为0
}

// 示例2: 带初始值的减法
template<typename... Args>
auto subtract_with_init(Args... args) {
    return (args - ... - 100);  // 二元右折叠，初始值为100
}

// 示例3: 带初始值的乘法
template<typename... Args>
auto multiply_with_init(Args... args) {
    return (args * ... * 1);  // 二元右折叠，初始值为1
}

// 示例4: 带初始值的逻辑与
template<typename... Args>
bool all_true_with_init(Args... args) {
    return (args && ... && true);  // 二元右折叠，初始值为true
}

// 示例5: 带初始值的逻辑或
template<typename... Args>
bool any_true_with_init(Args... args) {
    return (args || ... || false);  // 二元右折叠，初始值为false
}

// 示例6: push_back 到 vector
template<typename T, typename... Args>
void push_all(std::vector<T>& vec, Args... args) {
    (vec.push_back(args), ...);  // 一元右折叠，使用逗号运算符
}

// 示例7: 打印所有参数
template<typename... Args>
void print_all(Args... args) {
    ((std::cout << args << " "), ...);  // 一元右折叠
    std::cout << std::endl;
}

// 示例8: 字符串连接带后缀
template<typename... Args>
std::string concat_with_prefix(Args... args) {
    return (args + ... + std::string(" [end]"));  // 二元右折叠，添加后缀
}

// 示例9: 空参数包的安全处理
template<typename... Args>
auto safe_sum(Args... args) {
    return (args + ... + 0);  // 即使参数包为空，也会返回0
}

int main() {
    // 示例1: 带初始值的加法
    std::cout << "【示例1: 带初始值的加法】" << std::endl;
    int result1 = sum_with_init(1, 2, 3, 4);
    std::cout << "sum_with_init(1, 2, 3, 4) = " << result1 << std::endl;
    // 展开为: 1 + (2 + (3 + (4 + 0))) = 10

    int result1_empty = sum_with_init();
    std::cout << "sum_with_init() [空参数包] = " << result1_empty << std::endl;
    // 展开为: 0

    // 示例2: 带初始值的减法
    std::cout << "【示例2: 带初始值的减法】" << std::endl;
    int result2 = subtract_with_init(10, 5, 3);
    std::cout << "subtract_with_init(10, 5, 3) = " << result2 << std::endl;
    // 展开为: 10 - (5 - (3 - 100)) = 10 - (5 - (-97)) = 10 - 102 = -92

    // 示例3: 带初始值的乘法
    std::cout << "【示例3: 带初始值的乘法】" << std::endl;
    int result3 = multiply_with_init(2, 3, 4);
    std::cout << "multiply_with_init(2, 3, 4) = " << result3 << std::endl;
    // 展开为: 2 * (3 * (4 * 1)) = 24

    int result3_empty = multiply_with_init();
    std::cout << "multiply_with_init() [空参数包] = " << result3_empty << std::endl;
    // 展开为: 1

    // 示例4: 带初始值的逻辑与
    std::cout << "【示例4: 带初始值的逻辑与】" << std::endl;
    bool result4 = all_true_with_init(true, true, true);
    std::cout << "all_true_with_init(true, true, true) = " << std::boolalpha << result4 << std::endl;
    // 展开为: true && (true && (true && true)) = true

    bool result4_empty = all_true_with_init();
    std::cout << "all_true_with_init() [空参数包] = " << result4_empty << std::endl;
    // 展开为: true

    // 示例5: 带初始值的逻辑或
    std::cout << "【示例5: 带初始值的逻辑或】" << std::endl;
    bool result5 = any_true_with_init(false, false, true);
    std::cout << "any_true_with_init(false, false, true) = " << std::boolalpha << result5 << std::endl;
    // 展开为: false || (false || (true || false)) = true

    bool result5_empty = any_true_with_init();
    std::cout << "any_true_with_init() [空参数包] = " << result5_empty << std::endl;
    // 展开为: false

    // 示例6: push_back 到 vector
    std::cout << "【示例6: push_back 到 vector】" << std::endl;
    std::vector<int> vec;
    push_all(vec, 10, 20, 30, 40, 50);
    std::cout << "push_all(vec, 10, 20, 30, 40, 50): ";
    for (int v : vec) {
        std::cout << v << " ";
    }
    // 展开为: (vec.push_back(10), (vec.push_back(20), (vec.push_back(30), (vec.push_back(40), vec.push_back(50)))))
    // 注意: 虽然是右折叠(括号从右嵌套)，但逗号运算符从左到右执行，所以实际顺序是 10→20→30→40→50

    // 示例7: 打印所有参数
    std::cout << "【示例7: 打印所有参数】" << std::endl;
    std::cout << "print_all(1, 2, 3, \"Hello\", 4.5): ";
    print_all(1, 2, 3, "Hello", 4.5);
    // 展开为: ((std::cout << 1 << " "), (... 递归展开))
    std::cout << std::endl;

    // 示例8: 字符串连接带后缀
    std::cout << "【示例8: 字符串连接带后缀】" << std::endl;
    std::string result8 = concat_with_prefix(std::string("Hello"), std::string(" "), std::string("World"));
    std::cout << "concat_with_prefix(\"Hello\", \" \", \"World\") = " << result8 << std::endl;
    // 展开为: "Hello" + (" " + ("World" + " [end]"))

    // 示例9: 空参数包的安全处理
    std::cout << "【示例9: 空参数包的安全处理】" << std::endl;
    int result9 = safe_sum();
    std::cout << "safe_sum() [空参数包] = " << result9 << std::endl;
    // 二元折叠表达式可以安全处理空参数包

    return 0;
}

/*
2. 二元右折叠（Binary Right Fold）
语法: (pack op ... op init)
展开为：E1 op (E2 op (E3 op ... op (En op init)))

优点：
1. 可以提供初始值
2. 可以安全处理空参数包
3. 更灵活的控制计算过程
*/
