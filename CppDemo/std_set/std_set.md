# std::set 详细解析

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

`std::set`是基于**红黑树**的关联容器，存储**唯一**的**有序**元素，自动去重且自动排序。

### 定义位置

```cpp
#include <set>
```

### 模板声明

```cpp
template<class T, class Compare = std::less<T>,
         class Allocator = std::allocator<T>>
class set;
```

- **T**: 元素类型
- **Compare**: 比较函数
- **Allocator**: 内存分配器

### 为什么选择 std::set？

```
┌──────────────────────────────────────────────┐
│        📦 std::set 的优势                     │
├──────────────────────────────────────────────┤
│ ✅ 元素唯一：自动去重                         │
│ ✅ 自动排序：元素自动按顺序存储               │
│ ✅ 快速查找：O(log n)时间复杂度               │
│ ✅ 范围查询：支持lower_bound/upper_bound     │
│ ✅ 迭代器稳定：删除不影响其他迭代器           │
└──────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | std::set | std::unordered_set | std::vector |
|------|----------|-------------------|-------------|
| 底层实现 | 红黑树 | 哈希表 | 动态数组 |
| 元素唯一 | ✅ | ✅ | ❌ |
| 有序 | ✅ | ❌ | ❌ |
| 查找 | O(log n) | O(1)平均 | O(n) |
| 插入 | O(log n) | O(1)平均 | O(1)尾部 |
| 范围查询 | ✅ | ❌ | ❌ |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造
std::set<int> s1;

// 2. 从范围构造
std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};
std::set<int> s2(v.begin(), v.end());  // {1, 2, 3, 4, 5, 6, 9}

// 3. 初始化列表构造
std::set<int> s3 = {5, 2, 8, 1, 9};

// 4. 自定义比较器
std::set<int, std::greater<int>> s4;  // 降序

// 5. 拷贝构造
std::set<int> s5(s3);
```

### 元素访问

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `find(val)` | O(log n) | 查找元素 |
| `count(val)` | O(log n) | 计数（0或1） |
| `contains(val)` | O(log n) | 检查是否存在 (C++20) |
| `lower_bound(val)` | O(log n) | 第一个>=val的位置 |
| `upper_bound(val)` | O(log n) | 第一个>val的位置 |
| `equal_range(val)` | O(log n) | 返回[lower_bound, upper_bound) |

```cpp
std::set<int> s = {10, 20, 30, 40, 50};

// find - 查找元素
auto it = s.find(30);
if (it != s.end()) {
    std::cout << *it;  // 30
}

// count - 计数（0或1）
size_t n = s.count(30);  // 1
size_t m = s.count(100); // 0

// lower_bound/upper_bound - 范围查询
auto lb = s.lower_bound(25);  // 指向30
auto ub = s.upper_bound(30);  // 指向40

// equal_range - 获取范围
auto range = s.equal_range(30);  // [30, 40)
```

### 修改操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `insert(val)` | O(log n) | 插入元素 |
| `emplace(args)` | O(log n) | 原位构造插入 |
| `erase(val)` | O(log n) | 删除元素 |
| `erase(iterator)` | O(log n) | 删除迭代器指向的元素 |
| `clear()` | O(n) | 清空所有元素 |

```cpp
std::set<int> s;

// insert - 返回pair<iterator, bool>
auto result = s.insert(10);
if (result.second) {
    std::cout << "插入成功";
}

// 重复插入无效
s.insert(10);  // 返回false，不插入

// emplace - 原位构造
s.emplace(20);

// erase - 删除
s.erase(10);

// clear - 清空
s.clear();
```

---

## 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| insert | **O(log n)** |
| erase | **O(log n)** |
| find | **O(log n)** |
| count | **O(log n)** |
| lower_bound | **O(log n)** |
| upper_bound | **O(log n)** |

---

## 使用场景

### 1. 去重并保持顺序

```cpp
std::vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};

// 快速去重
std::set<int> s(vec.begin(), vec.end());  // {1, 2, 3, 4, 5, 6, 9}

// 遍历有序结果
for (int x : s) {
    std::cout << x << " ";
}
```

### 2. 快速成员检查

```cpp
std::set<std::string> allowed_users = {"alice", "bob", "charlie"};

std::string user = "bob";
if (allowed_users.find(user) != allowed_users.end()) {
    std::cout << "用户被允许";
}

// 或使用 count()
if (allowed_users.count(user)) {
    std::cout << "用户被允许";
}
```

### 3. 范围查询

```cpp
std::set<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

// 查找所有在[4, 7)范围内的元素
auto lower = nums.lower_bound(4);  // 指向4
auto upper = nums.upper_bound(6);  // 指向7

for (auto it = lower; it != upper; ++it) {
    std::cout << *it << " ";  // 4 5 6
}
```

### 4. 集合运算

```cpp
std::set<int> a = {1, 2, 3, 4, 5};
std::set<int> b = {3, 4, 5, 6, 7};

// 并集
std::set<int> union_set;
std::set_union(a.begin(), a.end(), b.begin(), b.end(),
               std::inserter(union_set, union_set.begin()));
// {1, 2, 3, 4, 5, 6, 7}

// 交集
std::set<int> intersection;
std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(intersection, intersection.begin()));
// {3, 4, 5}

// 差集
std::set<int> difference;
std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                    std::inserter(difference, difference.begin()));
// {1, 2}
```

---

## 注意事项

### 1. 元素不可修改

```cpp
std::set<int> s = {1, 2, 3};

// ❌ 编译错误：不能修改元素
// *s.begin() = 10;

// ✅ 需要删除后重新插入
s.erase(1);
s.insert(10);
```

### 2. 自定义比较器

```cpp
// 降序排列
std::set<int, std::greater<int>> s_desc = {5, 2, 8, 1, 9};
// {9, 8, 5, 2, 1}

// 自定义比较器
struct CustomCompare {
    bool operator()(int a, int b) const {
        return std::abs(a) < std::abs(b);  // 按绝对值排序
    }
};

std::set<int, CustomCompare> s_custom;
s_custom.insert(-5);
s_custom.insert(3);
s_custom.insert(-3);  // {3, -3, -5}
```

### 3. 有序性保证

```cpp
std::set<int> s = {5, 1, 3, 2, 4};

// 遍历自动有序
for (int x : s) {
    std::cout << x << " ";  // 1 2 3 4 5
}
```

---

## 常见问题

### Q1: set 和 unordered_set 的区别？

| 特性 | std::set | std::unordered_set |
|------|----------|-------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找 | O(log n) | O(1)平均 |
| 遍历顺序 | 有序 | 无序 |
| 范围查询 | ✅ | ❌ |

```cpp
// set - 有序
std::set<int> s = {5, 2, 8, 1, 9};
for (int x : s) {
    std::cout << x << " ";  // 1 2 5 8 9
}

// unordered_set - 无序
std::unordered_set<int> us = {5, 2, 8, 1, 9};
for (int x : us) {
    std::cout << x << " ";  // 顺序不确定
}
```

### Q2: set 和 multiset 的区别？

| 特性 | std::set | std::multiset |
|------|----------|---------------|
| 元素唯一 | ✅ | ❌ 允许重复 |
| 自动排序 | ✅ | ✅ |
| insert返回 | pair<it, bool> | iterator |
| erase(value) | 删除单个 | 删除所有 |

```cpp
// set - 元素唯一
std::set<int> s;
s.insert(5);
s.insert(5);  // 只有一个5

// multiset - 允许重复
std::multiset<int> ms;
ms.insert(5);
ms.insert(5);  // 两个5
```

### Q3: 何时使用 set？

✅ **适合**：
- 需要元素唯一
- 需要保持有序
- 频繁查找
- 需要范围查询

❌ **不适合**：
- 不需要有序 → 使用 unordered_set
- 需要重复元素 → 使用 multiset
- 需要快速查找且不需要有序 → 使用 unordered_set

### Q4: 如何高效地批量插入？

```cpp
std::set<int> s;

// ❌ 低效：逐个插入
for (int i = 0; i < 1000; ++i) {
    s.insert(i);
}

// ✅ 高效：使用初始化列表或范围构造
std::vector<int> data;
for (int i = 0; i < 1000; ++i) {
    data.push_back(i);
}
std::set<int> s2(data.begin(), data.end());
```

---

## 总结

### 何时使用 std::set

✅ **适合**：
- 需要元素唯一且有序
- 频繁查找和范围查询
- 需要集合运算
- 需要有序遍历

❌ **不适合**：
- 不需要有序 → 使用 unordered_set
- 需要重复元素 → 使用 multiset
- 需要快速查找 → 使用 unordered_set

### 最佳实践

1. **优先使用 find()** 而非 count() 进行查询
2. **使用 lower_bound/upper_bound** 进行范围查询
3. **自定义比较器** 实现自定义排序
4. **利用有序性** 进行有序遍历
5. **使用集合算法** 进行集合运算

---

## 参考文档
- [cppreference - std::set](https://en.cppreference.com/w/cpp/container/set)
