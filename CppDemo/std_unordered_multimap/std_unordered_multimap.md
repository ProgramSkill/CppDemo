# std::unordered_multimap 详细解析

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

`std::unordered_multimap`是基于**哈希表**的关联容器，允许**重复键**且**无序**存储键值对。

### 定义位置

```cpp
#include <unordered_map>
```

### 模板声明

```cpp
template<class Key, class T, class Hash = std::hash<Key>,
         class KeyEqual = std::equal_to<Key>,
         class Allocator = std::allocator<std::pair<const Key, T>>>
class unordered_multimap;
```

---

## 核心特性

| 特性 | std::multimap | std::unordered_multimap |
|------|---------------|------------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找效率 | O(log n) | **O(1)平均** |
| 允许重复键 | ✅ | ✅ |
| operator[] | ❌ | ❌ |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造
std::unordered_multimap<std::string, int> umm1;

// 2. 从范围构造
std::vector<std::pair<std::string, int>> v = {{"a", 1}, {"a", 2}};
std::unordered_multimap<std::string, int> umm2(v.begin(), v.end());

// 3. 初始化列表构造
std::unordered_multimap<std::string, int> umm3 = {{"key", 1}, {"key", 2}};
```

### 插入操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `insert(const pair<K,V>&)` | O(1)平均 | 插入键值对 |
| `emplace(Args&&...)` | O(1)平均 | 原位构造插入 |

```cpp
std::unordered_multimap<std::string, int> umm;

// insert - 总是成功
umm.insert({"key", 1});
umm.insert({"key", 2});
umm.insert({"key", 3});

// emplace - 原位构造
umm.emplace("other", 10);
```

### 删除操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `erase(const Key&)` | O(n)平均 | 删除所有该键 |
| `erase(iterator)` | O(1)平均 | 删除单个元素 |

```cpp
std::unordered_multimap<std::string, int> umm = {{"a", 1}, {"a", 2}, {"b", 3}};

// erase(key) - 删除所有该键
size_t count = umm.erase("a");  // 删除2个

// erase(iterator) - 删除单个
auto it = umm.find("a");
if (it != umm.end()) {
    umm.erase(it);  // 只删除一个
}
```

### 查询操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `find(const Key&)` | O(1)平均 | 查找键 |
| `count(const Key&)` | O(1)平均 | 计数 |
| `equal_range(const Key&)` | O(n)平均 | 返回所有相同键的元素 |

```cpp
std::unordered_multimap<std::string, int> umm = {{"a", 1}, {"a", 2}, {"b", 3}};

// find - 返回第一个匹配
auto it = umm.find("a");

// count - 返回匹配数量
size_t n = umm.count("a");  // 2

// equal_range - 获取所有相同键的元素
auto range = umm.equal_range("a");
for (auto i = range.first; i != range.second; ++i) {
    std::cout << i->second << " ";  // 1 2
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

### 1. 一对多映射关系

```cpp
std::unordered_multimap<std::string, std::string> phonebook;

phonebook.insert({"Alice", "123-456-7890"});
phonebook.insert({"Alice", "987-654-3210"});  // 多个号码

// 查询Alice的所有号码
auto range = phonebook.equal_range("Alice");
for (auto it = range.first; it != range.second; ++it) {
    std::cout << it->second << std::endl;
}
```

### 2. 快速查找（无序）

```cpp
std::unordered_multimap<int, std::string> index;

index.emplace(1, "tag1");
index.emplace(2, "tag2");
index.emplace(1, "tag3");  // 相同键

// 快速查询
std::cout << index.count(1);  // 2
```

### 3. 多值索引

```cpp
std::unordered_multimap<std::string, int> tags;

tags.emplace("important", 1);
tags.emplace("important", 2);
tags.emplace("important", 3);

// 获取所有标记为"important"的项
auto range = tags.equal_range("important");
for (auto it = range.first; it != range.second; ++it) {
    std::cout << it->second << " ";  // 1 2 3
}
```

---

## 注意事项

### 1. 无operator[]

```cpp
std::unordered_multimap<std::string, int> umm;

// ❌ 编译错误：unordered_multimap 不支持 operator[]
// umm["key"] = 5;

// ✅ 使用 insert 或 emplace
umm.insert({"key", 5});
```

### 2. 无序性

```cpp
std::unordered_multimap<std::string, int> umm = {{"c", 3}, {"a", 1}, {"b", 2}};

// 遍历顺序不确定
for (auto& [k, v] : umm) {
    std::cout << k << ":" << v << " ";  // 顺序不确定
}
```

### 3. 迭代器失效

```cpp
std::unordered_multimap<std::string, int> umm = {{"a", 1}, {"b", 2}};
auto it = umm.find("a");

umm.insert({"c", 3});  // 可能导致 rehash
// it 可能失效
```

---

## 常见问题

### Q1: unordered_multimap 和 multimap 的区别？

| 特性 | std::multimap | std::unordered_multimap |
|------|---------------|------------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找 | O(log n) | O(1)平均 |
| 遍历顺序 | 有序 | 无序 |

### Q2: 何时使用 unordered_multimap？

✅ **适合**：
- 需要快速查找（O(1)）
- 允许重复键
- 不需要有序
- 一对多映射

❌ **不适合**：
- 需要有序 → multimap
- 需要范围查询 → multimap
- 哈希函数质量差 → multimap

### Q3: 如何获取所有相同键的值？

```cpp
std::unordered_multimap<std::string, int> umm = {{"a", 1}, {"a", 2}, {"a", 3}};

auto range = umm.equal_range("a");
for (auto it = range.first; it != range.second; ++it) {
    std::cout << it->second << " ";  // 1 2 3
}
```

---

## 总结

### 何时使用 std::unordered_multimap

✅ **适合**：
- 需要快速查找重复键
- 不需要有序
- 一对多映射关系
- 多值索引

❌ **不适合**：
- 需要有序 → multimap
- 需要范围查询 → multimap
- 内存受限 → multimap

---

## 参考文档
- [cppreference - std::unordered_multimap](https://en.cppreference.com/w/cpp/container/unordered_multimap)
