# std::string_view 详细解析

## 目录

1. [概述](#概述)
2. [核心特性](#核心特性)
3. [成员函数详解](#成员函数详解)
4. [时间复杂度](#时间复杂度)
5. [使用场景](#使用场景)
6. [注意事项](#注意事项)
7. [常见问题](#常见问题)

---

## 概述

`std::string_view`是C++17引入的**非拥有字符串引用**，提供对字符序列的零开销视图。

### 定义位置

```cpp
#include <string_view>
```

### 模板声明

```cpp
template<class CharT, class Traits = std::char_traits<CharT>>
class basic_string_view;

using string_view = basic_string_view<char>;
using wstring_view = basic_string_view<wchar_t>;
```

### 为什么选择 std::string_view？

```
┌──────────────────────────────────────────────┐
│      📦 std::string_view 的优势               │
├──────────────────────────────────────────────┤
│ ✅ 零拷贝：O(1)构造，无内存分配               │
│ ✅ 通用性：接受字符串字面量、std::string     │
│ ✅ 非拥有：避免所有权问题                     │
│ ✅ 只读：类型安全的只读视图                   │
│ ✅ 高效：适合函数参数和临时使用               │
└──────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | std::string | std::string_view |
|------|-------------|------------------|
| 拥有内存 | ✅ | ❌ |
| 拷贝开销 | O(n) | **O(1)** |
| 修改 | ✅ | ❌ 只读 |
| 内存管理 | 自动 | 无 |
| 生命周期 | 自管理 | 依赖源 |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造（空视图）
std::string_view sv1;

// 2. 从C字符串构造
std::string_view sv2("hello");

// 3. 从std::string构造
std::string str = "world";
std::string_view sv3(str);

// 4. 指定范围构造
std::string_view sv4("hello world", 5);  // "hello"

// 5. 从另一个string_view构造
std::string_view sv5(sv2);
```

### 元素访问

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `operator[]` | O(1) | 访问指定位置 |
| `at(pos)` | O(1) | 安全访问（范围检查） |
| `front()` | O(1) | 访问第一个字符 |
| `back()` | O(1) | 访问最后一个字符 |
| `data()` | O(1) | 获取指针 |

```cpp
std::string_view sv = "hello";

// 访问元素
char c = sv[0];           // 'h'
char first = sv.front();  // 'h'
char last = sv.back();    // 'o'
const char* ptr = sv.data();
```

### 子串操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `substr(pos, count)` | O(1) | 返回子串视图 |
| `remove_prefix(count)` | O(1) | 移除前缀 |
| `remove_suffix(count)` | O(1) | 移除后缀 |

```cpp
std::string_view sv = "hello world";

// 子串
auto sub = sv.substr(0, 5);      // "hello"
auto sub2 = sv.substr(6);        // "world"

// 移除前后缀
std::string_view sv2 = sv;
sv2.remove_prefix(6);            // "world"
sv2.remove_suffix(1);            // "worl"
```

### 查询操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `size()` / `length()` | O(1) | 字符数 |
| `empty()` | O(1) | 是否为空 |
| `find(sv)` | O(n*m) | 查找子串 |
| `starts_with(sv)` | O(n) | 检查前缀 (C++20) |
| `ends_with(sv)` | O(n) | 检查后缀 (C++20) |

```cpp
std::string_view sv = "hello world";

std::cout << sv.size();           // 11
std::cout << sv.empty();          // false
std::cout << sv.find("world");    // 6
std::cout << sv.starts_with("he"); // true (C++20)
```

---

## 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| 构造 | **O(1)** |
| 访问 | **O(1)** |
| 子串 | **O(1)** |
| 查找 | **O(n*m)** |
| 大小查询 | **O(1)** |

---

## 使用场景

### 1. 函数参数（避免拷贝）

```cpp
// 接受任何字符串源
void print(std::string_view sv) {
    std::cout << sv << std::endl;
}

// 可以传递不同类型
print("literal");                    // C字符串
print(std::string("temporary"));     // 临时string
std::string s = "persistent";
print(s);                            // 持久string
```

### 2. 字符串分割

```cpp
std::vector<std::string_view> split(std::string_view str, char delim) {
    std::vector<std::string_view> result;
    size_t start = 0;

    for (size_t i = 0; i <= str.size(); ++i) {
        if (i == str.size() || str[i] == delim) {
            result.push_back(str.substr(start, i - start));
            start = i + 1;
        }
    }
    return result;
}

// 使用
auto parts = split("hello,world,cpp", ',');
// parts = ["hello", "world", "cpp"]
```

### 3. 字符串处理

```cpp
// 移除前后空格
std::string_view trim(std::string_view sv) {
    sv.remove_prefix(std::min(sv.find_first_not_of(" "), sv.size()));
    sv.remove_suffix(std::min(sv.size() - sv.find_last_not_of(" ") - 1, sv.size()));
    return sv;
}

// 检查前缀/后缀
bool is_config_file(std::string_view filename) {
    return filename.ends_with(".cfg");  // C++20
}
```

### 4. 日志和调试

```cpp
void log_message(std::string_view level, std::string_view message) {
    std::cout << "[" << level << "] " << message << std::endl;
}

// 无需创建临时string
log_message("INFO", "Application started");
log_message("ERROR", "Failed to open file");
```

---

## 注意事项

### 1. 生命周期管理

```cpp
// ❌ 危险：string_view指向临时对象
std::string_view get_view() {
    std::string temp = "hello";
    return std::string_view(temp);  // temp生命周期结束
}

// ✅ 正确：string_view指向有效对象
std::string str = "hello";
std::string_view sv(str);  // str仍然有效
```

### 2. 不能修改

```cpp
std::string_view sv = "hello";

// ❌ 编译错误：string_view是只读的
// sv[0] = 'H';

// ✅ 如果需要修改，使用string
std::string s = "hello";
s[0] = 'H';
```

### 3. 不保证null终止

```cpp
std::string_view sv("hello", 3);  // "hel"

// ❌ 危险：可能没有null终止符
// const char* ptr = sv.data();  // 不安全

// ✅ 安全方式
std::string s(sv);  // 转换为string
```

### 4. 与C API交互

```cpp
// 如果C函数需要null终止符
void c_function(const char* str);

std::string_view sv = "hello";

// ❌ 不安全
// c_function(sv.data());

// ✅ 安全方式
std::string s(sv);
c_function(s.c_str());
```

---

## 常见问题

### Q1: string_view 和 string 的区别？

| 特性 | std::string | std::string_view |
|------|-------------|------------------|
| 拥有内存 | ✅ | ❌ |
| 拷贝开销 | O(n) | O(1) |
| 修改 | ✅ | ❌ |
| 生命周期 | 自管理 | 依赖源 |
| 用途 | 存储字符串 | 引用字符串 |

### Q2: 何时使用 string_view？

✅ **适合**：
- 函数参数（避免拷贝）
- 临时字符串处理
- 字符串分割和查询
- 与C API交互

❌ **不适合**：
- 需要拥有字符串 → 使用 string
- 需要修改字符串 → 使用 string
- 需要长期存储 → 使用 string

### Q3: 如何安全地使用 string_view？

```cpp
// 规则1：确保源对象有效
std::string str = "hello";
std::string_view sv(str);  // OK

// 规则2：避免临时对象
// std::string_view sv = std::string("temp");  // 危险

// 规则3：如果需要修改，转换为string
std::string s(sv);
s[0] = 'H';
```

### Q4: string_view 可以转换为 string 吗？

```cpp
std::string_view sv = "hello";

// 方法1：构造函数
std::string s1(sv);

// 方法2：显式转换
std::string s2 = std::string(sv);

// 方法3：赋值
std::string s3;
s3 = std::string(sv);
```

---

## 总结

### 何时使用 std::string_view

✅ **适合**：
- 函数参数接受任意字符串源
- 临时字符串处理
- 字符串分割和查询
- 避免不必要的拷贝

❌ **不适合**：
- 需要拥有字符串 → 使用 string
- 需要修改字符串 → 使用 string
- 需要长期存储 → 使用 string

### 最佳实践

1. **使用string_view作为函数参数** 而非const string&
2. **注意生命周期** 确保源对象有效
3. **避免存储string_view** 作为成员变量
4. **需要修改时转换为string** 而非尝试修改视图

---

## 参考文档
- [cppreference - std::string_view](https://en.cppreference.com/w/cpp/string/basic_string_view)
