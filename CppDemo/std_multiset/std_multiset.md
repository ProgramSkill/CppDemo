# std::multiset 详细解析

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

`std::multiset`是允许**重复元素**的**有序**集合，基于红黑树实现。

### 定义位置

```cpp
#include <set>
```

### 模板声明

```cpp
template<class T, class Compare = std::less<T>,
         class Allocator = std::allocator<T>>
class multiset;
```

- **T**: 元素类型
- **Compare**: 比较函数
- **Allocator**: 内存分配器

### 为什么选择 std::multiset？

```
┌──────────────────────────────────────────────┐
│        📦 std::multiset 的优势                │
├──────────────────────────────────────────────┤
│ ✅ 自动排序：元素自动按顺序存储               │
│ ✅ 允许重复：支持多个相同元素                 │
│ ✅ 有序遍历：遍历时自动有序                   │
│ ✅ 范围查询：支持lower_bound/upper_bound     │
│ ✅ 稳定迭代：迭代器相对稳定                   │
└──────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | std::set | std::multiset |
|------|----------|---------------|
| 元素唯一 | ✅ | ❌ 允许重复 |
| 自动排序 | ✅ | ✅ |
| 插入结果 | 返回pair<it, bool> | 返回iterator |
| erase(value) | 删除单个元素 | 删除**所有**该值 |
| 底层实现 | 红黑树 | 红黑树 |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造
std::multiset<int> ms1;

// 2. 从范围构造
std::vector<int> v = {1, 2, 2, 3, 3, 3};
std::multiset<int> ms2(v.begin(), v.end());

// 3. 初始化列表构造
std::multiset<int> ms3 = {1, 2, 2, 3, 3, 3};

// 4. 自定义比较器
std::multiset<int, std::greater<int>> ms4;  // 降序

// 5. 拷贝构造
std::multiset<int> ms5(ms3);
```

### 元素访问

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `find(val)` | O(log n) | 查找元素 |
| `count(val)` | O(log n + count) | 计数 |
| `lower_bound(val)` | O(log n) | 第一个>=val的位置 |
| `upper_bound(val)` | O(log n) | 第一个>val的位置 |
| `equal_range(val)` | O(log n) | 返回所有相同元素范围 |

```cpp
std::multiset<int> ms = {1, 2, 2, 3, 3, 3};

// find - 返回第一个匹配
auto it = ms.find(3);

// count - 返回匹配数量
size_t n = ms.count(3);  // 3

// lower_bound - 第一个>=3的位置
auto lower = ms.lower_bound(3);

// upper_bound - 第一个>3的位置
auto upper = ms.upper_bound(3);

// equal_range - 获取所有相同元素
auto range = ms.equal_range(3);
for (auto i = range.first; i != range.second; ++i) {
    std::cout << *i << " ";  // 3 3 3
}
```

### 修改操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `insert(val)` | O(log n) | 插入元素，总是成功 |
| `emplace(args)` | O(log n) | 原位构造插入 |
| `erase(val)` | O(log n + count) | 删除所有该值 |
| `erase(iterator)` | O(log n) | 删除单个元素 |

```cpp
std::multiset<int> ms;

// insert - 总是成功，返回iterator
auto it = ms.insert(5);
ms.insert(5);
ms.insert(5);  // 三个5

// emplace - 原位构造
ms.emplace(10);

// erase(value) - 删除所有该值
size_t count = ms.erase(5);  // 删除3个5，返回3

// erase(iterator) - 删除单个
auto it2 = ms.find(10);
if (it2 != ms.end()) {
    ms.erase(it2);  // 只删除一个
}
```

---

## 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| insert | **O(log n)** |
| erase(value) | **O(log n + count)** |
| erase(iterator) | **O(log n)** |
| find | **O(log n)** |
| count | **O(log n + count)** |
| lower_bound | **O(log n)** |
| upper_bound | **O(log n)** |

---

## 使用场景

### 1. 允许重复的有序集合

```cpp
std::multiset<int> scores;
scores.insert(90);
scores.insert(85);
scores.insert(90);  // 允许重复

// 遍历（自动有序）
for (int score : scores) {
    std::cout << score << " ";  // 85 90 90
}
```

### 2. 任务优先级队列（同优先级按时间）

```cpp
struct Task {
    int priority;
    long timestamp;

    bool operator<(const Task& other) const {
        if (priority != other.priority) {
            return priority > other.priority;  // 高优先级先
        }
        return timestamp < other.timestamp;    // 同优先级按时间
    }
};

std::multiset<Task> tasks;
tasks.emplace(1, 100);
tasks.emplace(1, 101);  // 相同优先级，按时间排序
```

### 3. 统计出现次数

```cpp
std::multiset<std::string> words;
words.insert("hello");
words.insert("hello");
words.insert("hello");
words.insert("world");

// 统计"hello"出现次数
std::cout << words.count("hello");  // 3

// 获取所有"hello"
auto range = words.equal_range("hello");
for (auto it = range.first; it != range.second; ++it) {
    std::cout << *it << " ";
}
```

### 4. 范围查询

```cpp
std::multiset<int> ms = {1, 2, 2, 3, 3, 3, 4, 5};

// 查找所有在[2, 4)范围内的元素
auto lower = ms.lower_bound(2);
auto upper = ms.upper_bound(3);

for (auto it = lower; it != upper; ++it) {
    std::cout << *it << " ";  // 2 2 3 3 3
}
```

---

## 注意事项

### 1. insert 返回 iterator 而非 pair

```cpp
std::multiset<int> ms;

// set: 返回 pair<iterator, bool>
// std::set<int> s;
// auto [it, inserted] = s.insert(5);

// multiset: 总是返回 iterator
auto it = ms.insert(5);  // 总是成功
ms.insert(5);            // 允许重复
```

### 2. erase(value) 删除所有该值

```cpp
std::multiset<int> ms = {1, 2, 2, 3, 3, 3};

// erase(value) - 删除所有该值
size_t count = ms.erase(3);  // 删除3个3，返回3

// 如果只想删除一个，使用迭代器
auto it = ms.find(2);
if (it != ms.end()) {
    ms.erase(it);  // 只删除一个2
}
```

### 3. 有序性

```cpp
std::multiset<int> ms = {5, 1, 3, 2, 4};

// 遍历自动有序
for (int x : ms) {
    std::cout << x << " ";  // 1 2 3 4 5
}
```

### 4. 自定义比较器

```cpp
// 降序排列
std::multiset<int, std::greater<int>> ms_desc;
ms_desc.insert(3);
ms_desc.insert(1);
ms_desc.insert(2);

for (int x : ms_desc) {
    std::cout << x << " ";  // 3 2 1
}
```

---

## 常见问题

### Q1: multiset 和 set 的区别？

| 特性 | std::set | std::multiset |
|------|----------|---------------|
| 元素唯一 | ✅ | ❌ 允许重复 |
| 自动排序 | ✅ | ✅ |
| insert返回 | pair<it, bool> | iterator |
| erase(value) | 删除单个 | 删除所有 |

### Q2: 何时使用 multiset？

✅ **适合**：
- 需要允许重复的有序集合
- 需要统计元素出现次数
- 需要范围查询
- 需要自动排序

❌ **不适合**：
- 元素必须唯一 → 使用 set
- 不需要有序 → 使用 unordered_multiset
- 需要快速查找 → 使用 unordered_multiset

### Q3: 如何获取所有相同元素？

```cpp
std::multiset<int> ms = {1, 2, 2, 3, 3, 3};

// 方法1：使用 equal_range
auto range = ms.equal_range(3);
for (auto it = range.first; it != range.second; ++it) {
    std::cout << *it << " ";  // 3 3 3
}

// 方法2：使用 find 和 count
auto it = ms.find(3);
for (int i = 0; i < ms.count(3); ++i) {
    std::cout << *it << " ";
    ++it;
}
```

### Q4: 如何按自定义顺序排序？

```cpp
// 降序
std::multiset<int, std::greater<int>> ms_desc;

// 自定义比较器
struct CustomCompare {
    bool operator()(int a, int b) const {
        return a > b;  // 降序
    }
};
std::multiset<int, CustomCompare> ms_custom;

// Lambda比较器（C++11）
auto cmp = [](int a, int b) { return a > b; };
std::multiset<int, decltype(cmp)> ms_lambda(cmp);
```

---

## 总结

### 何时使用 std::multiset

✅ **适合**：
- 需要允许重复的有序集合
- 需要统计元素出现次数
- 需要范围查询
- 需要自动排序

❌ **不适合**：
- 元素必须唯一 → 使用 set
- 不需要有序 → 使用 unordered_multiset
- 需要快速查找 → 使用 unordered_multiset

### 最佳实践

1. **使用 equal_range()** 获取所有相同元素
2. **记住 insert 返回 iterator** 而非 pair
3. **小心 erase(value)** 会删除所有该值
4. **利用自动排序** 进行有序遍历
5. **使用范围查询** 进行高效的区间操作

---

## 参考文档
- [cppreference - std::multiset](https://en.cppreference.com/w/cpp/container/multiset)
