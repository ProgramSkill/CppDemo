# std::unordered_multiset 详细解析

## 概述

允许重复元素的哈希集合。

```cpp
#include <unordered_set>
```

## 核心特性

- 允许重复元素
- 无序存储
- 平均O(1)插入删除查找

## 使用场景

```cpp
// 允许重复的快速查找集合
unordered_multiset<int> ums;
ums.insert(1);
ums.insert(1);
ums.insert(1);
cout << ums.count(1);  // 3
```

## 参考文档
- [cppreference - std::unordered_multiset](https://en.cppreference.com/w/cpp/container/unordered_multiset)
