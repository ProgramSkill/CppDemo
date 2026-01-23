# std::optional 详细解析

## 概述

`std::optional`是C++17引入的可选值包装器，表示值可能存在或不存在。

```cpp
#include <optional>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 可选值 | 可能包含值也可能为空 |
| 类型安全 | 替代nullptr/异常 |
| 内存效率 | 通常与T+bool相同 |

## 基本用法

```cpp
optional<int> opt1;        // 空
optional<int> opt2 = 42;  // 有值
optional<int> opt3 = nullopt;  // 空

// 检查
if (opt2.has_value()) {
    cout << *opt2;
}

// 访问
int x = opt2.value();          // 有值时返回
int y = opt2.value_or(0);      // 有值返回，无值返回0
int z = *opt2;                 // 直接解引用

// 修改
opt1 = 10;
opt1.reset();
opt1.emplace(20);
```

## 使用场景

```cpp
// 1. 可选的返回值
optional<int> find(int key) {
    auto it = m.find(key);
    if (it != m.end()) {
        return it->second;
    }
    return nullopt;  // 或 {}
}

auto result = find(5);
if (result) {
    cout << "found: " << *result;
}

// 2. 配置项
struct Config {
    optional<string> host;
    optional<int> port;
    optional<bool> debug;
    string getHost() const { return host.value_or("localhost"); }
};

// 3. 避免异常
optional<int> to_int(string s) {
    try {
        return stoi(s);
    } catch (...) {
        return nullopt;
    }
}
```

## 参考文档
- [cppreference - std::optional](https://en.cppreference.com/w/cpp/utility/optional)
