# std::unordered_multimap 详细解析

## 概述

允许重复键的哈希表。

```cpp
#include <unordered_map>
```

## 核心特性

- 允许重复键
- 无序存储
- 平均O(1)操作
- 无operator[]

## 使用场景

```cpp
// 允许重复键的快速查找字典
unordered_multimap<string, int> umm;
umm.insert({"key", 1});
umm.insert({"key", 2});
cout << umm.count("key");  // 2

// equal_range访问所有相同键的值
auto range = umm.equal_range("key");
```

## 参考文档
- [cppreference - std::unordered_multimap](https://en.cppreference.com/w/cpp/container/unordered_multimap)
