# std::index_sequence 详解

## 概述

`std::index_sequence` 是 C++14 引入的一个编译期整数序列工具，定义在 `<utility>` 头文件中。它主要用于模板元编程，特别是在需要生成编译期整数序列的场景中。

## 基本定义

```cpp
template<std::size_t... Ints>
struct index_sequence {
    using value_type = std::size_t;
    static constexpr std::size_t size() noexcept { return sizeof...(Ints); }
};
```

`std::index_sequence` 是一个类模板，它将一系列 `std::size_t` 类型的整数作为模板参数。

## 相关工具

### std::make_index_sequence

```cpp
template<std::size_t N>
using make_index_sequence = std::index_sequence<0, 1, 2, ..., N-1>;
```

`std::make_index_sequence<N>` 会生成一个从 0 到 N-1 的整数序列。

**示例：**
- `std::make_index_sequence<3>` 生成 `std::index_sequence<0, 1, 2>`
- `std::make_index_sequence<5>` 生成 `std::index_sequence<0, 1, 2, 3, 4>`

### std::index_sequence_for

```cpp
template<typename... T>
using index_sequence_for = std::make_index_sequence<sizeof...(T)>;
```

根据类型参数包的大小生成对应的索引序列。

## 核心原理

`std::index_sequence` 的核心思想是**将运行时的索引操作转换为编译期的模板参数展开**。

### 参数包展开

当我们有一个 `std::index_sequence<0, 1, 2>` 时，可以通过模板参数推导将其展开：

```cpp
template<std::size_t... Is>
void func(std::index_sequence<Is...>) {
    // Is... 会展开为 0, 1, 2
}
```

## 常见用例

### 1. Tuple 解包（最常见）

这是 `std::index_sequence` 最典型的应用场景。当我们想要将 tuple 的元素作为函数参数传递时，需要在编译期展开 tuple 的所有元素。

**问题：** 如何将 `std::tuple<int, double, string>` 的元素传递给函数 `void func(int, double, string)`？

**解决方案：**

```cpp
#include <tuple>
#include <utility>
#include <iostream>

void print(int a, double b, const char* c) {
    std::cout << a << ", " << b << ", " << c << "\n";
}

// 实现函数：使用 index_sequence 展开 tuple
template<typename Func, typename Tuple, std::size_t... Is>
void call_impl(Func f, Tuple t, std::index_sequence<Is...>) {
    f(std::get<Is>(t)...);  // 展开为: f(std::get<0>(t), std::get<1>(t), std::get<2>(t))
}

// 接口函数：生成 index_sequence
template<typename Func, typename Tuple>
void call(Func f, Tuple t) {
    call_impl(f, t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

int main() {
    auto args = std::make_tuple(42, 3.14, "hello");
    call(print, args);  // 输出: 42, 3.14, hello
}
```

**工作原理详解：**

1. `std::make_index_sequence<3>()` 生成 `std::index_sequence<0, 1, 2>`
2. 模板参数推导：`Is...` 被推导为 `0, 1, 2`
3. 参数包展开：`std::get<Is>(t)...` 展开为 `std::get<0>(t), std::get<1>(t), std::get<2>(t)`
4. 函数调用：`f(std::get<0>(t), std::get<1>(t), std::get<2>(t))`

### 2. 数组初始化

使用 `std::index_sequence` 可以在编译期生成数组。

```cpp
#include <array>
#include <utility>

// 生成平方数数组
template<std::size_t... Is>
constexpr auto make_square_array_impl(std::index_sequence<Is...>) {
    return std::array<std::size_t, sizeof...(Is)>{Is * Is...};
}

template<std::size_t N>
constexpr auto make_square_array() {
    return make_square_array_impl(std::make_index_sequence<N>());
}

int main() {
    constexpr auto squares = make_square_array<5>();
    // squares = {0, 1, 4, 9, 16}
}
```

### 3. 编译期遍历

可以使用 `std::index_sequence` 在编译期对参数包进行遍历操作。

```cpp
#include <iostream>
#include <utility>

// 打印所有参数
template<typename... Args, std::size_t... Is>
void print_all_impl(std::index_sequence<Is...>, Args... args) {
    // 使用折叠表达式（C++17）
    ((std::cout << Is << ": " << args << "\n"), ...);
}

template<typename... Args>
void print_all(Args... args) {
    print_all_impl(std::index_sequence_for<Args...>(), args...);
}

int main() {
    print_all(42, 3.14, "hello", 'x');
    // 输出:
    // 0: 42
    // 1: 3.14
    // 2: hello
    // 3: x
}
```

### 4. 反转索引序列

创建一个反转的索引序列。

```cpp
#include <utility>
#include <tuple>
#include <iostream>

// 反转 tuple
template<typename Tuple, std::size_t... Is>
auto reverse_tuple_impl(const Tuple& t, std::index_sequence<Is...>) {
    constexpr std::size_t N = std::tuple_size<Tuple>::value;
    return std::make_tuple(std::get<N - 1 - Is>(t)...);
}

template<typename Tuple>
auto reverse_tuple(const Tuple& t) {
    return reverse_tuple_impl(t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

int main() {
    auto t = std::make_tuple(1, 2.0, "three");
    auto reversed = reverse_tuple(t);
    // reversed = ("three", 2.0, 1)
}
```

## 实现原理

`std::make_index_sequence` 的实现通常使用递归模板实例化。以下是一个简化的实现示例：

```cpp
// 基础情况
template<std::size_t N, std::size_t... Is>
struct make_index_sequence_impl : make_index_sequence_impl<N - 1, N - 1, Is...> {};

// 递归终止
template<std::size_t... Is>
struct make_index_sequence_impl<0, Is...> {
    using type = std::index_sequence<Is...>;
};

// 辅助别名
template<std::size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;
```

**工作过程：**
- `make_index_sequence<3>`
- → `make_index_sequence_impl<3>`
- → `make_index_sequence_impl<2, 2>`
- → `make_index_sequence_impl<1, 1, 2>`
- → `make_index_sequence_impl<0, 0, 1, 2>`
- → `std::index_sequence<0, 1, 2>`

## 最佳实践

### 1. 使用辅助函数模式

推荐使用两层函数的设计模式：
- **接口函数**：负责生成 `index_sequence`
- **实现函数**：负责使用 `index_sequence` 进行参数包展开

```cpp
// 接口函数
template<typename Func, typename Tuple>
void call(Func f, Tuple t) {
    call_impl(f, t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

// 实现函数
template<typename Func, typename Tuple, std::size_t... Is>
void call_impl(Func f, Tuple t, std::index_sequence<Is...>) {
    f(std::get<Is>(t)...);
}
```

### 2. 使用 constexpr

尽可能使用 `constexpr` 以确保编译期计算。

```cpp
template<std::size_t N>
constexpr auto make_array() {
    return make_array_impl(std::make_index_sequence<N>());
}
```

### 3. 参数传递方式

对于 tuple 参数，根据使用场景选择合适的传递方式：

```cpp
// 按值传递（小对象）
template<typename Func, typename Tuple, std::size_t... Is>
void call_impl(Func f, Tuple t, std::index_sequence<Is...>) {
    f(std::get<Is>(t)...);
}

// 按常量引用传递（大对象）
template<typename Func, typename Tuple, std::size_t... Is>
void call_impl(Func f, const Tuple& t, std::index_sequence<Is...>) {
    f(std::get<Is>(t)...);
}

// 完美转发（通用）
template<typename Func, typename Tuple, std::size_t... Is>
void call_impl(Func f, Tuple&& t, std::index_sequence<Is...>) {
    f(std::get<Is>(std::forward<Tuple>(t))...);
}
```

## 常见陷阱

### 1. 参数包展开 vs 折叠表达式

**参数包展开**（Parameter Pack Expansion）：
```cpp
f(std::get<Is>(t)...)  // 展开为: f(std::get<0>(t), std::get<1>(t), std::get<2>(t))
```

**折叠表达式**（Fold Expression，C++17）：
```cpp
(... + args)  // 展开为: ((a1 + a2) + a3)
```

两者不同：
- 参数包展开生成多个参数
- 折叠表达式需要操作符，生成单个表达式

### 2. 编译期 vs 运行期

`std::index_sequence` 是**编译期**工具，索引值在编译期确定。不能用于运行期动态索引：

```cpp
// ❌ 错误：运行期索引
template<typename Tuple, std::size_t... Is>
void wrong(Tuple t, std::index_sequence<Is...>, int runtime_index) {
    // 无法使用 runtime_index 访问 tuple
}

// ✅ 正确：编译期索引
template<typename Tuple, std::size_t... Is>
void correct(Tuple t, std::index_sequence<Is...>) {
    f(std::get<Is>(t)...);  // Is 是编译期常量
}
```

### 3. 模板实例化深度

对于非常大的 N 值，递归模板实例化可能导致编译器达到实例化深度限制。大多数编译器对 `std::make_index_sequence` 进行了优化，使用编译器内建（builtin）实现。

## 性能考虑

### 编译期开销

- `std::index_sequence` 的所有操作都在**编译期**完成
- 不会产生运行期开销
- 生成的代码与手写展开的代码完全相同

### 零成本抽象

```cpp
// 使用 index_sequence
call(print, std::make_tuple(1, 2, 3));

// 编译后等价于
print(1, 2, 3);
```

两者生成的汇编代码完全相同，这就是 C++ 的"零成本抽象"原则。

## C++11 替代方案

在 C++11 中，可以手动实现类似功能：

```cpp
// C++11 实现
template<std::size_t... Is>
struct index_sequence {};

template<std::size_t N, std::size_t... Is>
struct make_index_sequence_impl : make_index_sequence_impl<N - 1, N - 1, Is...> {};

template<std::size_t... Is>
struct make_index_sequence_impl<0, Is...> {
    using type = index_sequence<Is...>;
};

template<std::size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;
```

但 C++14 标准库提供的版本通常有编译器优化，性能更好。

## 总结

### 核心要点

1. **编译期工具**：`std::index_sequence` 是编译期整数序列，用于模板元编程
2. **零运行期开销**：所有操作在编译期完成，生成的代码与手写展开完全相同
3. **主要用途**：Tuple 解包、数组初始化、编译期遍历等
4. **设计模式**：使用接口函数 + 实现函数的两层设计

### 何时使用

- 需要将 tuple/array 元素展开为函数参数
- 需要在编译期生成整数序列
- 需要对参数包进行索引访问
- 需要实现编译期算法

### 相关工具

- `std::index_sequence<Is...>` - 整数序列类型
- `std::make_index_sequence<N>` - 生成 0 到 N-1 的序列
- `std::index_sequence_for<Types...>` - 根据类型包大小生成序列

## 示例代码

本目录下的 `UnpackingTuples.cpp` 提供了一个完整的 tuple 解包示例，展示了 `std::index_sequence` 的典型用法。

## 快速参考

### 基本模板

```cpp
// 1. 接口函数 - 生成 index_sequence
template<typename... Args>
auto my_function(Args... args) {
    return my_function_impl(std::make_index_sequence<sizeof...(Args)>(), args...);
}

// 2. 实现函数 - 使用 index_sequence
template<std::size_t... Is, typename... Args>
auto my_function_impl(std::index_sequence<Is...>, Args... args) {
    // 使用 Is... 进行参数包展开
}
```

### 常用展开模式

```cpp
// 函数调用展开
f(std::get<Is>(tuple)...)

// 数组初始化展开
std::array{expr(Is)...}

// 折叠表达式配合使用（C++17）
((process(std::get<Is>(tuple))), ...)
```

---

**C++ 标准**：C++14
**头文件**：`<utility>`
**命名空间**：`std`
