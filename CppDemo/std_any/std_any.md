# std::any 详细解析

## 概述

`std::any`是C++17引入的类型擦除容器，可存储任何可拷贝类型。

```cpp
#include <any>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 类型擦除 | 可存储任何类型 |
| 类型安全 | 需要any_cast访问 |
| 运行时检查 | 访问时检查类型 |

## 基本用法

```cpp
any a = 42;
int x = any_cast<int>(a);  // 42

a = string("hello");
string s = any_cast<string>(a);  // "hello"

// any_cast<int>(a);  // 抛出bad_any_cast

// 指针版本（不抛异常）
if (auto ptr = any_cast<string>(&a)) {
    cout << *ptr;
}
```

## 使用场景

```cpp
// 1. 异构容器
vector<any> items;
items.push_back(42);
items.push_back(string("hello"));
items.push_back(3.14);

// 2. 消息传递
queue<any> messages;
messages.push(int(42));
messages.push(string("test"));

// 3. 属性映射
map<string, any> props;
props["name"] = string("Alice");
props["age"] = 30;
props["height"] = 1.75;
```

## 参考文档
- [cppreference - std::any](https://en.cppreference.com/w/cpp/utility/any)
