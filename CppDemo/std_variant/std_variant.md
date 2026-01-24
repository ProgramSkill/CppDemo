# std::variant 详细解析

## 目录

1. [概述](#概述)
2. [核心特性](#核心特性)
3. [成员函数详解](#成员函数详解)
4. [访问方式](#访问方式)
5. [使用场景](#使用场景)
6. [注意事项](#注意事项)
7. [常见问题](#常见问题)

---

## 概述

`std::variant`是C++17引入的**类型安全联合体**，可以存储多个类型中的任意一个。

### 定义位置

```cpp
#include <variant>
```

### 模板声明

```cpp
template<class... Types>
class variant;
```

- **Types**: 可以存储的类型列表

### 为什么选择 std::variant？

```
┌──────────────────────────────────────────────┐
│         📦 std::variant 的优势                │
├──────────────────────────────────────────────┤
│ ✅ 类型安全：编译时检查，避免类型错误       │
│ ✅ 知道当前类型：运行时可查询当前类型       │
│ ✅ 允许非POD：支持复杂类型和析构函数       │
│ ✅ 零开销：编译为union+tag，无额外开销     │
│ ✅ 替代union：比C union更安全               │
└──────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | std::variant | union |
|------|-------------|-------|
| 类型安全 | ✅ | ❌ |
| 知道当前类型 | ✅ | ❌ |
| 允许非POD | ✅ | ❌ |
| 默认构造 | ✅ 首个类型 | ❌ |
| 析构函数 | ✅ 自动 | ❌ 手动 |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造（首个类型）
std::variant<int, double, std::string> v1;  // v1 = 0

// 2. 值构造
std::variant<int, double, std::string> v2 = 42;
std::variant<int, double, std::string> v3 = 3.14;
std::variant<int, double, std::string> v4 = "hello";

// 3. 原位构造
std::variant<int, double, std::string> v5(std::in_place_type<std::string>, "world");

// 4. 拷贝构造
std::variant<int, double, std::string> v6(v2);

// 5. 移动构造
std::variant<int, double, std::string> v7(std::move(v2));
```

### 赋值操作

| 函数 | 说明 |
|------|------|
| `operator=` | 赋值新值 |
| `emplace<T>()` | 原位构造新值 |

```cpp
std::variant<int, double, std::string> v = 42;

// 赋值
v = 3.14;           // 现在是double
v = "hello";        // 现在是string

// 原位构造
v.emplace<std::string>("world");
v.emplace<int>(100);
```

### 查询操作

| 函数 | 说明 |
|------|------|
| `index()` | 返回当前类型的索引 |
| `valueless_by_exception()` | 检查是否处于无效状态 |

```cpp
std::variant<int, double, std::string> v = 42;

std::cout << v.index();  // 0 (int是第一个类型)

v = 3.14;
std::cout << v.index();  // 1 (double是第二个类型)

v = "hello";
std::cout << v.index();  // 2 (string是第三个类型)
```

---

## 访问方式

### 1. get<T>() - 按类型访问

```cpp
std::variant<int, double, std::string> v = 42;

// 正确访问
int x = std::get<int>(v);           // 42

// 错误访问（抛出bad_variant_access）
// double d = std::get<double>(v);  // 异常

// 按索引访问
int y = std::get<0>(v);             // 42
```

### 2. get_if<T>() - 安全访问

```cpp
std::variant<int, double, std::string> v = 42;

// 返回指针，如果类型不匹配返回nullptr
if (auto ptr = std::get_if<int>(&v)) {
    std::cout << *ptr << std::endl;  // 42
}

if (auto ptr = std::get_if<double>(&v)) {
    std::cout << *ptr << std::endl;  // 不执行
}
```

### 3. holds_alternative<T>() - 类型检查

```cpp
std::variant<int, double, std::string> v = 42;

if (std::holds_alternative<int>(v)) {
    std::cout << "v holds int" << std::endl;
}

if (std::holds_alternative<double>(v)) {
    std::cout << "v holds double" << std::endl;  // 不执行
}
```

### 4. visit() - 访问者模式

```cpp
std::variant<int, double, std::string> v = 42;

// Lambda访问者
std::visit([](auto&& arg) {
    std::cout << arg << std::endl;
}, v);

// 多个variant
std::variant<int, double> v1 = 42;
std::variant<int, double> v2 = 3.14;

std::visit([](auto&& a, auto&& b) {
    std::cout << a << " " << b << std::endl;
}, v1, v2);
```

---

## 使用场景

### 1. 处理多种类型

```cpp
std::variant<int, std::string, std::vector<int>> data;

// 存储不同类型
data = 42;
data = "hello";
data = std::vector<int>{1, 2, 3};

// 访问
std::visit([](auto&& arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, int>) {
        std::cout << "int: " << arg << std::endl;
    } else if constexpr (std::is_same_v<T, std::string>) {
        std::cout << "string: " << arg << std::endl;
    } else if constexpr (std::is_same_v<T, std::vector<int>>) {
        std::cout << "vector size: " << arg.size() << std::endl;
    }
}, data);
```

### 2. 表达式求值

```cpp
struct Num { int value; };
struct Add { std::shared_ptr<struct Expr> left, right; };
struct Mul { std::shared_ptr<struct Expr> left, right; };

using Expr = std::variant<Num, std::shared_ptr<Add>, std::shared_ptr<Mul>>;

int evaluate(const Expr& expr) {
    return std::visit([](auto&& arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, Num>) {
            return arg.value;
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Add>>) {
            return evaluate(arg->left) + evaluate(arg->right);
        } else if constexpr (std::is_same_v<T, std::shared_ptr<Mul>>) {
            return evaluate(arg->left) * evaluate(arg->right);
        }
    }, expr);
}
```

### 3. 错误处理（代替异常）

```cpp
struct Error {
    int code;
    std::string message;
};

template<typename T>
using Result = std::variant<T, Error>;

Result<int> divide(int a, int b) {
    if (b == 0) {
        return Error{1, "Division by zero"};
    }
    return a / b;
}

// 使用
auto result = divide(10, 2);
if (std::holds_alternative<int>(result)) {
    std::cout << "Result: " << std::get<int>(result) << std::endl;
} else {
    auto err = std::get<Error>(result);
    std::cout << "Error: " << err.message << std::endl;
}
```

### 4. 状态机

```cpp
struct Idle {};
struct Running { int progress; };
struct Stopped { int reason; };

using State = std::variant<Idle, Running, Stopped>;

void handle_state(const State& state) {
    std::visit([](auto&& s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, Idle>) {
            std::cout << "System is idle" << std::endl;
        } else if constexpr (std::is_same_v<T, Running>) {
            std::cout << "Progress: " << s.progress << "%" << std::endl;
        } else if constexpr (std::is_same_v<T, Stopped>) {
            std::cout << "Stopped with reason: " << s.reason << std::endl;
        }
    }, state);
}
```

---

## 注意事项

### 1. 异常安全

```cpp
std::variant<int, std::string> v = 42;

try {
    // 如果构造失败，variant可能处于无效状态
    v = std::string(1000000000, 'a');  // 可能抛出异常
} catch (...) {
    // 检查是否有效
    if (v.valueless_by_exception()) {
        std::cout << "Variant is in invalid state" << std::endl;
    }
}
```

### 2. 类型歧义

```cpp
// ❌ 歧义：int可以隐式转换为double
// std::variant<int, double> v = 42;  // 编译错误

// ✅ 明确指定类型
std::variant<int, double> v(std::in_place_type<int>, 42);
```

### 3. 大小开销

```cpp
std::variant<int, double, std::string> v;

// 大小 = max(sizeof(int), sizeof(double), sizeof(string)) + tag
std::cout << sizeof(v) << std::endl;  // 通常 > 32字节
```

### 4. 访问者的返回类型

```cpp
std::variant<int, double> v = 42;

// ❌ 返回类型不一致
// auto result = std::visit([](auto&& arg) {
//     if (std::is_same_v<decltype(arg), int>) return 1;
//     else return 1.0;  // 类型不同
// }, v);

// ✅ 返回类型一致
auto result = std::visit([](auto&& arg) -> double {
    if (std::is_same_v<decltype(arg), int>) return arg;
    else return arg;
}, v);
```

---

## 常见问题

### Q1: variant 和 union 的区别？

| 特性 | std::variant | union |
|------|-------------|-------|
| 类型安全 | ✅ | ❌ |
| 知道当前类型 | ✅ | ❌ |
| 允许非POD | ✅ | ❌ |
| 自动析构 | ✅ | ❌ |
| 易用性 | ✅ | ❌ |

### Q2: 何时使用 variant？

✅ **适合**：
- 需要存储多种类型之一
- 类型安全很重要
- 需要知道当前类型
- 错误处理（代替异常）
- 状态机实现

❌ **不适合**：
- 只需要一种类型 → 使用该类型
- 需要动态类型 → 使用void*或多态
- 性能极其关键 → 考虑union

### Q3: 如何遍历 variant 中的所有可能类型？

```cpp
std::variant<int, double, std::string> v = 42;

// 方法1：使用visit
std::visit([](auto&& arg) {
    std::cout << typeid(arg).name() << std::endl;
}, v);

// 方法2：使用index
switch (v.index()) {
    case 0: std::cout << std::get<0>(v) << std::endl; break;
    case 1: std::cout << std::get<1>(v) << std::endl; break;
    case 2: std::cout << std::get<2>(v) << std::endl; break;
}
```

### Q4: variant 可以为空吗？

```cpp
// ❌ variant 不能为空
// std::variant<> v;  // 编译错误

// ✅ 如果需要"空"状态，添加std::monostate
std::variant<std::monostate, int, std::string> v;  // 默认为monostate
```

---

## 总结

### 何时使用 std::variant

✅ **适合**：
- 需要类型安全的多类型存储
- 需要知道当前类型
- 实现状态机
- 错误处理（Result类型）
- 表达式求值

❌ **不适合**：
- 只需要一种类型 → 使用该类型
- 需要动态类型 → 使用多态
- 性能极其关键 → 考虑其他方案

### 最佳实践

1. **使用visit()** 而非多个get_if()调用
2. **使用holds_alternative()** 进行类型检查
3. **使用in_place_type** 避免类型歧义
4. **处理valueless_by_exception()** 状态
5. **使用Result<T>模式** 进行错误处理

---

## 参考文档
- [cppreference - std::variant](https://en.cppreference.org/w/cpp/utility/variant)
