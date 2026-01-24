# std::unordered_multiset 详细解析

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

`std::unordered_multiset`是基于**哈希表**的关联容器，允许**重复元素**且**无序**存储。

### 定义位置

```cpp
#include <unordered_set>
```

### 模板声明

```cpp
template<class T, class Hash = std::hash<T>,
         class KeyEqual = std::equal_to<T>,
         class Allocator = std::allocator<T>>
class unordered_multiset;
```

---

## 核心特性

| 特性 | std::multiset | std::unordered_multiset |
|------|---------------|------------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找效率 | O(log n) | **O(1)平均** |
| 允许重复 | ✅ | ✅ |
| 迭代器稳定 | ✅ | ❌ rehash时失效 |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造
std::unordered_multiset<int> ums1;

// 2. 从范围构造
std::vector<int> v = {1, 2, 2, 3, 3, 3};
std::unordered_multiset<int> ums2(v.begin(), v.end());

// 3. 初始化列表构造
std::unordered_multiset<int> ums3 = {1, 2, 2, 3};

// 4. 拷贝构造
std::unordered_multiset<int> ums4(ums3);
```

### 插入操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `insert(const T&)` | O(1)平均 | 插入元素 |
| `insert(T&&)` | O(1)平均 | 移动插入 |
| `emplace(Args&&...)` | O(1)平均 | 原位构造插入 |

```cpp
std::unordered_multiset<int> ums;

// insert - 总是成功
ums.insert(5);
ums.insert(5);
ums.insert(5);

// emplace - 原位构造
ums.emplace(10);
```

### 删除操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `erase(const T&)` | O(n)平均 | 删除所有该值 |
| `erase(iterator)` | O(1)平均 | 删除单个元素 |
| `clear()` | O(n) | 清空所有元素 |

```cpp
std::unordered_multiset<int> ums = {1, 2, 2, 3, 3, 3};

// erase(value) - 删除所有该值
size_t count = ums.erase(3);  // 删除3个3，返回3

// erase(iterator) - 删除单个
auto it = ums.find(2);
if (it != ums.end()) {
    ums.erase(it);  // 只删除一个2
}
```

### 查询操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `find(const T&)` | O(1)平均 | 查找元素 |
| `count(const T&)` | O(1)平均 | 计数 |
| `contains(const T&)` | O(1)平均 | 检查是否存在 (C++20) |
| `equal_range(const T&)` | O(n)平均 | 返回所有相同元素 |

```cpp
std::unordered_multiset<int> ums = {1, 2, 2, 3, 3, 3};

// find - 返回第一个匹配
auto it = ums.find(3);

// count - 返回匹配数量
size_t n = ums.count(3);  // 3

// equal_range - 获取所有相同元素
auto range = ums.equal_range(3);
for (auto i = range.first; i != range.second; ++i) {
    std::cout << *i << " ";  // 3 3 3
}
```

---

## 时间复杂度

| 操作 | 平均 | 最坏 |
|------|------|------|
| insert | O(1) | O(n) |
| erase | O(1) | O(n) |
| find | O(1) | O(n) |
| count | O(1) | O(n) |

---

## 使用场景

### 1. 允许重复的快速查找集合

```cpp
std::unordered_multiset<int> ums;
ums.insert(1);
ums.insert(1);
ums.insert(1);

// 快速查询出现次数
std::cout << ums.count(1);  // 3
```

### 2. 词频统计（无序）

```cpp
std::unordered_multiset<std::string> words;
std::string word;
while (std::cin >> word) {
    words.insert(word);
}

// 统计某个词出现的次数
std::cout << words.count("hello");
```

### 3. 去重后统计

```cpp
std::vector<int> data = {1, 2, 2, 3, 3, 3, 4};
std::unordered_multiset<int> ums(data.begin(), data.end());

// 快速查询每个元素出现的次数
for (int x : {1, 2, 3, 4}) {
    std::cout << x << ": " << ums.count(x) << std::endl;
}
```

---

## 注意事项

### 1. 无序性

```cpp
std::unordered_multiset<int> ums = {3, 1, 4, 1, 5};

// 遍历顺序不确定
for (int x : ums) {
    std::cout << x << " ";  // 顺序不确定
}
```

### 2. 迭代器失效

```cpp
std::unordered_multiset<int> ums = {1, 2, 3};
auto it = ums.find(2);

ums.insert(4);  // 可能导致 rehash
// it 可能失效
```

### 3. 哈希冲突

```cpp
std::unordered_multiset<int> ums;

// 哈希冲突会导致性能下降
// 最坏情况：所有元素哈希到同一桶，O(n)
```

---

## 常见问题

### Q1: unordered_multiset 和 multiset 的区别？

| 特性 | std::multiset | std::unordered_multiset |
|------|---------------|------------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找 | O(log n) | O(1)平均 |
| 遍历顺序 | 有序 | 无序 |

### Q2: 何时使用 unordered_multiset？

✅ **适合**：
- 需要快速查找（O(1)）
- 允许重复元素
- 不需要有序
- 统计频率

❌ **不适合**：
- 需要有序 → multiset
- 需要范围查询 → multiset
- 哈希函数质量差 → multiset

### Q3: 如何获取所有相同元素？

```cpp
std::unordered_multiset<int> ums = {1, 2, 2, 3, 3, 3};

auto range = ums.equal_range(3);
for (auto it = range.first; it != range.second; ++it) {
    std::cout << *it << " ";  // 3 3 3
}
```

---

## 总结

### 何时使用 std::unordered_multiset

✅ **适合**：
- 需要快速查找重复元素
- 不需要有序
- 频率统计
- 缓存实现

❌ **不适合**：
- 需要有序 → multiset
- 需要范围查询 → multiset
- 内存受限 → multiset

---

## 参考文档
- [cppreference - std::unordered_multiset](https://en.cppreference.com/w/cpp/container/unordered_multiset)
