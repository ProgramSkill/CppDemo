# std::span 详细解析

## 概述

`std::span`是C++20引入的非拥有序列视图，提供对连续内存的零开销抽象。

```cpp
#include <span>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 非拥有 | 不管理内存生命周期 |
| 零开销 | 编译为单指针+大小 |
| 通用 | 适用于array, vector, C数组 |

## 基本用法

```cpp
int arr[] = {1, 2, 3, 4, 5};
span<int> s1(arr);
span<int, 5> s2(arr);  // 固定大小

cout << s1.size();
cout << s1[0];
```

## 子视图

```cpp
span<int> s(arr);
auto first = s.first(3);    // 前3个
auto last = s.last(2);      // 后2个
auto sub = s.subspan(1, 3); // [1, 4)
```

## 使用场景

```cpp
// 1. 函数参数（避免拷贝）
void process(span<const int> data);
process(vector{1,2,3});
process(array<int,3>{1,2,3});
int arr[] = {1,2,3}; process(arr);

// 2. 分块处理
span<int> data = /* ... */;
for (size_t i = 0; i < data.size(); i += 1024) {
    auto chunk = data.subspan(i, 1024);
    process(chunk);
}
```

## 参考文档
- [cppreference - std::span](https://en.cppreference.com/w/cpp/container/span)
