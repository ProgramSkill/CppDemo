# std::variant 详细解析

## 概述

`std::variant`是C++17引入的类型安全联合体。

```cpp
#include <variant>
```

## 核心特性

| 特性 | std::variant | union |
|------|-------------|-------|
| 类型安全 | ✅ | ❌ |
| 知道当前类型 | ✅ | ❌ |
| 允许非POD | ✅ | ❌ |
| 默认构造 | ✅ 首个类型 | ❌ |

## 基本用法

```cpp
// 可以存储int, double, string之一
variant<int, double, string> v;

v = 42;
cout << get<int>(v);  // 42
cout << holds_alternative<int>(v);  // true

v = 3.14;
cout << get<double>(v);  // 3.14
// get<int>(v);  // 抛出bad_variant_access

v = "hello";
cout << get<string>(v);  // "hello"
```

## 访问方式

```cpp
variant<int, double, string> v = 42;

// 1. get<T>
if (holds_alternative<int>(v)) {
    int x = get<int>(v);
}

// 2. get_if
if (auto ptr = get_if<int>(&v)) {
    cout << *ptr;
}

// 3. visit
visit([](auto&& arg) {
    cout << arg;
}, v);
```

## 使用场景

```cpp
// 1. 处理多种类型
variant<int, string, vector<int>> data;

// 2. 表达式求值
struct Add; struct Mul; struct Num;
using Expr = variant<shared_ptr<Add>, shared_ptr<Mul>, int>;

// 3. 错误处理（代替异常）
variant<Value, Error> result;
if (holds_alternative<Error>(result)) {
    auto err = get<Error>(result);
    cout << err.message;
}
```

## 参考文档
- [cppreference - std::variant](https://en.cppreference.org/w/cpp/utility/variant)
