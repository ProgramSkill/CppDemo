#include <iostream>
#include <string>

// 示例1: 加法运算
template<typename... Args>
auto sum_left(Args... args) {
    return (... + args);  // 一元左折叠
}

// 示例2: 减法运算（展示左折叠的结合性）
template<typename... Args>
auto subtract_left(Args... args) {
    return (... - args);  // 一元左折叠
}

// 示例3: 乘法运算
template<typename... Args>
auto multiply_left(Args... args) {
    return (... * args);
}

// 示例4: 逻辑与运算
template<typename... Args>
bool all_true(Args... args) {
    return (... && args);  // 一元左折叠
}

// 示例5: 逻辑或运算
template<typename... Args>
bool any_true(Args... args) {
    return (... || args);
}

// 示例6: 字符串连接
template<typename... Args>
std::string concat_left(Args... args) {
    return (... + args);
}

// 示例7: 逗号运算符（返回最后一个值）
template<typename... Args>
auto get_last(Args... args) {
    return (... , args);  // 一元左折叠
}

int main() {
    std::cout << "=== 一元左折叠示例 ===" << std::endl << std::endl;

    // 示例1: 加法
    int result1 = sum_left(1, 2, 3, 4);
    std::cout << "sum_left(1, 2, 3, 4) = " << result1 << std::endl;
    // 展开为: ((1 + 2) + 3) + 4 = 10

    // 示例2: 减法（展示左折叠的结合性）
    int result3 = subtract_left(10, 5, 3, 1);
    std::cout << "subtract_left(10, 5, 3, 1) = " << result3 << std::endl;
    // 展开为: ((10 - 5) - 3) - 1 = (5 - 3) - 1 = 2 - 1 = 1
    std::cout << "注意: 如果是右结合 10 - (5 - (3 - 1)) = 7" << std::endl;
    std::cout << std::endl;

    // 示例3: 乘法
    int result4 = multiply_left(2, 3, 4);
    std::cout << "multiply_left(2, 3, 4) = " << result4 << std::endl;
    // 展开为: (2 * 3) * 4 = 6 * 4 = 24

    // 示例4: 逻辑与
    bool result6 = all_true(true, false, true);
    std::cout << "all_true(true, false, true) = " << std::boolalpha << result6 << std::endl;
    // 展开为: (true && false) && true = false
    std::cout << std::endl;

    // 示例5: 逻辑或
    bool result7 = any_true(false, false, true);
    std::cout << "any_true(false, false, true) = " << std::boolalpha << result7 << std::endl;
    // 展开为: (false || false) || true = true

    // 示例6: 字符串连接
    std::string result9 = concat_left(std::string("Hello"), std::string(" "), std::string("World"), std::string("!"));
    std::cout << "concat_left(\"Hello\", \" \", \"World\", \"!\") = " << result9 << std::endl;
    // 展开为: (("Hello" + " ") + "World") + "!"

    // 示例7: 逗号运算符
    // 逗号运算符总是返回最右边的值
    // 例如: int y = (5, 10, 15); // y = 15
    int result10 = get_last(10, 20, 30, 40);
    std::cout << "get_last(10, 20, 30, 40) = " << result10 << std::endl;
    // 展开为: (((10, 20), 30), 40) = 40

    return 0;
}

/*
1. 一元左折叠（Unary Left Fold）
语法: (... op pack)
展开为：((E1 op E2) op E3) op ... op En
*/