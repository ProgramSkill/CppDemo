#include <iostream>
#include <string>
#include <array>

/*
参数包类型详解：类型参数包 vs 非类型参数包

===============================================================================
1. 类型参数包 (Type Parameter Pack)
===============================================================================
语法：template<typename... Args>
特点：
- 参数是类型，可以是任意类型
- 可以传递运行时值（变量、字面量、表达式等）
- 灵活性最高，适用于泛型编程
- 编译时推导类型，运行时传递值

===============================================================================
2. 非类型参数包 (Non-Type Parameter Pack)  
===============================================================================
语法：template<int... nums> 或 template<typename T, T... values>
特点：
- 参数是值，必须是编译期常量
- 只能传递编译期确定的值（字面量、constexpr变量）
- 适用于模板元编程、编译期计算
- 类型固定，值在编译期确定

===============================================================================
关键区别总结：
===============================================================================
| 特性                | 类型参数包              | 非类型参数包            |
|---------------------|------------------------|------------------------|
| 参数性质            | 类型                   | 值                     |
| 可传递的值          | 运行时值               | 编译期常量             |
| 灵活性              | 高（任意类型）         | 低（固定类型）         |
| 应用场景            | 泛型编程               | 模板元编程             |
| 编译时要求          | 推导类型               | 确定值                 |
*/

// ==================== 类型参数包示例 ====================

/**
 * @brief 类型参数包函数 - 可以传递任意类型的运行时值
 * @tparam Args 可变的类型参数包
 * @param args 可变的值参数包
 * 
 * 使用场景：需要处理不同类型数据的泛型函数
 */
template<typename... Args>
void print_type_pack(Args... args) {
    std::cout << "类型参数包示例: ";
    // 一元右折叠表达式：((std::cout << arg1 << " "), (std::cout << arg2 << " "), ...)
    // 逗号运算符保证从左到右执行
    ((std::cout << args << " "), ...);
    std::cout << std::endl;
}

/**
 * @brief 展示类型参数包的类型信息
 */
template<typename... Args>
void show_type_info(Args... args) {
    std::cout << "类型信息: ";
    // 使用折叠表达式和逗号运算符显示每个参数的类型
    ((std::cout << typeid(args).name() << ":" << args << " "), ...);
    std::cout << std::endl;
}

// ==================== 非类型参数包示例 ====================

/**
 * @brief 非类型参数包函数 - 只能传递编译期常量
 * @tparam nums 编译期整数常量包
 * 
 * 使用场景：编译期计算、数组大小、循环展开等
 */
template<int... nums>
void print_non_type_pack() {
    std::cout << "非类型参数包示例: ";
    ((std::cout << nums << " "), ...);
    std::cout << std::endl;
}

/**
 * @brief 编译期计算非类型参数包的和
 * @tparam nums 编译期整数常量包
 * @return 所有数的和（编译期计算）
 */
template<int... nums>
constexpr int sum_compile_time() {
    // 折叠表达式计算和：(nums + ... + 0)
    return (nums + ... + 0);
}

/**
 * @brief 通用非类型参数包 - 支持不同类型的编译期常量
 * @tparam T 值的类型
 * @tparam values 编译期常量包
 */
template<typename T, T... values>
void print_generic_non_type_pack() {
    std::cout << "通用非类型参数包 (" << typeid(T).name() << "): ";
    ((std::cout << values << " "), ...);
    std::cout << std::endl;
}

// ==================== 高级示例 ====================

/**
 * @brief 混合使用类型和非类型参数包
 * @tparam N 非类型参数（数组大小）
 * @tparam Args 类型参数包
 */
template<int N, typename... Args>
void mixed_pack_example(Args... args) {
    std::cout << "混合参数包 (N=" << N << "): ";
    ((std::cout << args << " "), ...);
    std::cout << "[共" << sizeof...(Args) << "个参数，数组大小" << N << "]" << std::endl;
}

/**
 * @brief 编译期数组生成
 * @tparam values 编译期常量包
 * @return 包含所有常量的数组
 */
template<int... values>
constexpr auto make_array() {
    // 使用初始化列表和折叠表达式创建数组
    return std::array<int, sizeof...(values)>{values...};
}

int main() {
    std::cout << "=== 参数包类型详解 ===" << std::endl << std::endl;
    
    // ==================== 类型参数包演示 ====================
    std::cout << "【类型参数包演示】" << std::endl;
    std::cout << "特点：可以传递任意类型的运行时值" << std::endl << std::endl;
    
    // 1. 传递不同类型的变量
    int int_var = 42;
    double double_var = 3.14;
    std::string str_var = "Hello";
    char char_var = 'A';
    
    std::cout << "1. 传递不同类型的变量:" << std::endl;
    print_type_pack(int_var, double_var, str_var, char_var);
    show_type_info(int_var, double_var, str_var, char_var);
    std::cout << std::endl;
    
    // 2. 传递字面量和表达式
    std::cout << "2. 传递字面量和表达式:" << std::endl;
    print_type_pack(100, 2.718, std::string("World"), 'Z', int_var * 2);
    show_type_info(100, 2.718, std::string("World"), 'Z', int_var * 2);
    std::cout << std::endl;
    
    // 3. 传递左值和右值引用
    std::cout << "3. 传递左值和右值:" << std::endl;
    int temp = 999;
    print_type_pack(temp, 888, std::move(temp));  // 左值、右值、移动语义
    std::cout << std::endl;
    
    // ==================== 非类型参数包演示 ====================
    std::cout << "【非类型参数包演示】" << std::endl;
    std::cout << "特点：只能是编译期常量" << std::endl << std::endl;
    
    // 1. 传递字面量常量
    std::cout << "1. 传递字面量常量:" << std::endl;
    print_non_type_pack<1, 2, 3, 4, 5>();
    std::cout << "编译期计算和: " << sum_compile_time<1, 2, 3, 4, 5>() << std::endl;
    std::cout << std::endl;
    
    // 2. 使用constexpr常量
    constexpr int const1 = 10;
    constexpr int const2 = 20;
    constexpr int const3 = 30;
    
    std::cout << "2. 使用constexpr常量:" << std::endl;
    print_non_type_pack<const1, const2, const3>();
    std::cout << "编译期计算和: " << sum_compile_time<const1, const2, const3>() << std::endl;
    std::cout << std::endl;
    
    // 3. 通用非类型参数包（不同类型）
    std::cout << "3. 通用非类型参数包:" << std::endl;
    print_generic_non_type_pack<int, 1, 2, 3>();
    print_generic_non_type_pack<double, 1.1, 2.2, 3.3>();
    print_generic_non_type_pack<char, 'A', 'B', 'C'>();
    std::cout << std::endl;
    
    // ==================== 混合使用演示 ====================
    std::cout << "【混合使用演示】" << std::endl;
    std::cout << "特点：同时使用类型和非类型参数包" << std::endl << std::endl;
    
    mixed_pack_example<5>(1, 2, 3);  // N=5, Args={int, int, int}
    mixed_pack_example<10>("Hello", "World");  // N=10, Args={const char*, const char*}
    std::cout << std::endl;
    
    // ==================== 编译期数组生成 ====================
    std::cout << "【编译期数组生成】" << std::endl;
    constexpr auto arr1 = make_array<1, 2, 3, 4, 5>();
    constexpr auto arr2 = make_array<10, 20, 30>();
    
    std::cout << "数组1: ";
    for (int v : arr1) std::cout << v << " ";
    std::cout << std::endl;
    
    std::cout << "数组2: ";
    for (int v : arr2) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << std::endl;
    
    // ==================== 错误示例展示 ====================
    std::cout << "【错误示例（无法编译）】" << std::endl;
    std::cout << "// 以下代码会导致编译错误：" << std::endl;
    std::cout << "// int runtime_var = 100;" << std::endl;
    std::cout << "// print_non_type_pack<runtime_var>();  // 错误！runtime_var不是编译期常量" << std::endl;
    std::cout << "// print_non_type_pack<int_var>();      // 错误！int_var是运行时变量" << std::endl;
    std::cout << std::endl;
    
    // ==================== 总结 ====================
    std::cout << "【总结】" << std::endl;
    std::cout << "1. 类型参数包：灵活性强，适用于泛型编程，可处理运行时数据" << std::endl;
    std::cout << "2. 非类型参数包：限制多，适用于模板元编程，编译期计算" << std::endl;
    std::cout << "3. 两者可以混合使用，实现更复杂的模板编程" << std::endl;
    std::cout << "4. 折叠表达式是处理参数包的强大工具" << std::endl;
    std::cout << "5. 选择哪种参数包取决于具体需求和应用场景" << std::endl;
    
    return 0;
}
