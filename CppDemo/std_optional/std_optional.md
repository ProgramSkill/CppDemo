# std::optional 详细解析

## 目录

1. [概述](#概述)
2. [核心特性](#核心特性)
3. [成员函数详解](#成员函数详解)
4. [使用场景](#使用场景)
5. [注意事项](#注意事项)
6. [常见问题](#常见问题)

---

## 概述

`std::optional` 是 C++17 引入的**可选值包装器**，用于表示一个值可能存在也可能不存在的情况。

### 定义位置

```cpp
#include <optional>
```

### 模板声明

```cpp
template<class T>
class optional;
```

### 为什么需要 std::optional？

```
┌──────────────────────────────────────────────┐
│         📦 std::optional 的优势               │
├──────────────────────────────────────────────┤
│ ✅ 类型安全：替代 nullptr 或特殊值            │
│ ✅ 语义清晰：明确表达"可能无值"的意图         │
│ ✅ 避免异常：不需要抛出异常表示无值           │
│ ✅ 零开销：通常只增加一个 bool 的开销         │
│ ✅ 值语义：存储值本身，非指针                 │
└──────────────────────────────────────────────┘
```

## 核心特性

| 特性 | std::optional<T> | T* | std::variant |
|------|-----------------|-----|-------------|
| 可选值 | ✅ | ✅ | ❌ |
| 值语义 | ✅ | ❌ | ✅ |
| 内存开销 | sizeof(T) + 1 | 指针大小 | sizeof(T) + 1 |
| 类型安全 | ✅ | ❌ | ✅ |
| 多种状态 | ❌ 有/无 | ❌ | ✅ |

---

## 成员函数详解

### 构造函数

| 函数 | 说明 |
|------|------|
| `optional()` | 默认构造，空值 |
| `optional(T)` | 从值构造 |
| `optional(std::nullopt_t)` | 显式空值 |
| `optional(std::in_place, args...)` | 原位构造 |
| `optional(const optional&)` | 拷贝构造 |

```cpp
// 1. 默认构造 - 空
std::optional<int> opt1;

// 2. 从值构造
std::optional<int> opt2 = 42;
std::optional<int> opt3(42);

// 3. 显式空值
std::optional<int> opt4 = std::nullopt;

// 4. 原位构造
std::optional<std::string> opt5(std::in_place, "hello");

// 5. 拷贝构造
std::optional<int> opt6(opt2);
```

### 赋值操作

| 函数 | 说明 |
|------|------|
| `operator=(T)` | 赋值值 |
| `operator=(std::nullopt_t)` | 赋值空值 |
| `emplace(args...)` | 原位构造赋值 |
| `reset()` | 清空值 |

```cpp
std::optional<int> opt;

// 赋值
opt = 10;           // 有值
opt = std::nullopt; // 清空

// 原位构造
opt.emplace(20);

// 清空
opt.reset();
```

### 访问值

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `operator*()` | O(1) | 解引用（无检查） |
| `operator->()` | O(1) | 指针访问（无检查） |
| `value()` | O(1) | 有检查，无值抛异常 |
| `value_or(default)` | O(1) | 有值返回值，无值返回默认值 |
| `has_value()` | O(1) | 检查是否有值 |
| `operator bool()` | O(1) | 隐式转换为bool |

```cpp
std::optional<int> opt = 42;

// 检查是否有值
if (opt.has_value()) { /* ... */ }
if (opt) { /* ... */ }  // 隐式转换为 bool

// 访问值
int x = *opt;              // 解引用（无检查）
int y = opt.value();       // 有检查，无值时抛异常
int z = opt.value_or(0);   // 有值返回值，无值返回默认值

// 指针访问
std::optional<std::string> str_opt = "hello";
std::cout << str_opt->length();  // 5
```

---

## 使用场景

---

## 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| 构造 | **O(1)** |
| 赋值 | **O(1)** |
| 访问 | **O(1)** |
| has_value() | **O(1)** |
| value_or() | **O(1)** |

---

## 使用场景

### 1. 可选的返回值

```cpp
std::optional<int> find_index(const std::vector<int>& v, int target) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == target) return i;
    }
    return std::nullopt;  // 未找到
}

auto result = find_index(vec, 42);
if (result) {
    std::cout << "Found at: " << *result << std::endl;
} else {
    std::cout << "Not found" << std::endl;
}
```

### 2. 配置项和可选参数

```cpp
struct Config {
    std::optional<std::string> host;
    std::optional<int> port;
    std::optional<bool> debug;
    std::optional<std::string> log_file;

    std::string getHost() const { return host.value_or("localhost"); }
    int getPort() const { return port.value_or(8080); }
    bool isDebug() const { return debug.value_or(false); }
};

Config cfg;
cfg.host = "example.com";
cfg.port = 9000;
// debug 和 log_file 保持未设置状态
```

### 3. 避免异常处理

```cpp
std::optional<int> to_int(const std::string& s) {
    try {
        return std::stoi(s);
    } catch (...) {
        return std::nullopt;
    }
}

// 使用
auto num = to_int("42");
if (num) {
    std::cout << "Value: " << *num << std::endl;
}
```

### 4. 替代特殊值

```cpp
// ❌ 不好：使用特殊值 -1 表示"未找到"
int find_index_old(const std::vector<int>& v, int target) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == target) return i;
    }
    return -1;  // 特殊值，容易混淆
}

// ✅ 好：使用 optional
std::optional<size_t> find_index_new(const std::vector<int>& v, int target) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == target) return i;
    }
    return std::nullopt;  // 清晰表达"无值"
}
```

### 5. 链式操作

```cpp
class User {
public:
    std::optional<std::string> get_email() const { return email_; }
    std::optional<std::string> get_phone() const { return phone_; }

private:
    std::optional<std::string> email_;
    std::optional<std::string> phone_;
};

User user;
// 安全地链式访问
auto contact = user.get_email().value_or(user.get_phone().value_or("No contact"));
```

---

## 注意事项

### 1. 访问空 optional 是未定义行为

```cpp
std::optional<int> opt;

// ❌ 未定义行为
// int x = *opt;

// ❌ 抛出 std::bad_optional_access
// int y = opt.value();

// ✅ 先检查
if (opt) {
    int z = *opt;
}

// ✅ 使用 value_or
int w = opt.value_or(0);
```

### 2. 使用 value_or 提供默认值

```cpp
std::optional<int> opt;

// 安全的默认值访问
int x = opt.value_or(0);
std::string s = opt.value_or("default");
```

### 3. 原位构造避免临时对象

```cpp
std::optional<std::string> opt;

// ❌ 创建临时对象
opt = std::string("hello");

// ✅ 原位构造，避免临时对象
opt.emplace("hello");
```

### 4. 比较操作

```cpp
std::optional<int> opt1 = 42;
std::optional<int> opt2 = 42;
std::optional<int> opt3;

opt1 == opt2;  // true
opt1 == opt3;  // false
opt3 == std::nullopt;  // true
opt1 > 40;  // true（与值比较）
```

---

## 常见问题

### Q1: optional 和指针的区别？

| 特性 | std::optional<T> | T* |
|------|-----------------|-----|
| 语义 | 值语义 | 指针语义 |
| 内存 | 栈上 | 可能堆上 |
| 空值表示 | nullopt | nullptr |
| 所有权 | 拥有值 | 不拥有 |
| 大小 | sizeof(T) + 1 | 指针大小 |

```cpp
// optional - 值语义
std::optional<int> opt = 42;
auto opt2 = opt;  // 拷贝值

// 指针 - 指针语义
int* ptr = new int(42);
int* ptr2 = ptr;  // 拷贝指针，指向同一对象
delete ptr;
```

### Q2: optional 和 variant 的区别？

| 特性 | std::optional<T> | std::variant<T, U> |
|------|-----------------|-------------------|
| 可选值 | ✅ 有/无 | ❌ 必须有值 |
| 多种类型 | ❌ 单一类型 | ✅ 多种类型 |
| 使用场景 | 可能无值 | 多种状态 |

```cpp
// optional - 单一类型，可能无值
std::optional<int> opt;

// variant - 多种类型，必须有值
std::variant<int, std::string> var = 42;
var = "hello";
```

### Q3: 何时使用 optional？

✅ **适合**：
- 函数可能无返回值
- 配置项可能未设置
- 避免使用特殊值（如 -1）
- 替代指针表示可选
- 避免异常处理

❌ **不适合**：
- 必须有值的情况
- 需要表示多种错误状态 → 使用 variant 或 expected
- 需要动态分配 → 使用指针

### Q4: 如何在容器中使用 optional？

```cpp
std::vector<std::optional<int>> vec = {1, std::nullopt, 3, std::nullopt, 5};

// 遍历并处理
for (const auto& opt : vec) {
    if (opt) {
        std::cout << *opt << " ";
    } else {
        std::cout << "empty ";
    }
}

// 过滤出有值的元素
std::vector<int> values;
for (const auto& opt : vec) {
    if (opt) {
        values.push_back(*opt);
    }
}
```

### Q5: optional 的内存开销是多少？

```cpp
std::optional<int> opt;

// 通常大小为 sizeof(int) + 1 字节（用于标记是否有值）
// 可能因对齐而增加到 sizeof(int) + 4 或 sizeof(int) + 8

std::cout << sizeof(opt);  // 通常 8 字节（int 4 + padding 4）

// 对于大对象，开销相对较小
std::optional<std::string> str_opt;
// 大约 sizeof(std::string) + 1 字节
```

---

## 总结

### 何时使用 std::optional

✅ **适合**：
- 函数可能无返回值
- 配置项可能未设置
- 避免使用特殊值
- 替代指针表示可选
- 避免异常处理

❌ **不适合**：
- 必须有值的情况
- 需要表示多种错误状态 → 使用 variant
- 需要动态分配 → 使用指针

### 最佳实践

1. **优先使用 value_or()** 提供默认值
2. **使用原位构造** 避免临时对象
3. **检查后再访问** 使用 has_value() 或隐式转换
4. **避免嵌套 optional** 使用 flatten 或 variant
5. **考虑 expected** 需要错误信息时

---

## 参考文档
- [cppreference - std::optional](https://en.cppreference.com/w/cpp/utility/optional)

