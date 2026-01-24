# std::multimap 详细解析

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

`std::multimap`是允许**重复键**的**有序**键值对容器，基于红黑树实现。

### 定义位置

```cpp
#include <map>
```

### 模板声明

```cpp
template<class Key, class T, class Compare = std::less<Key>,
         class Allocator = std::allocator<std::pair<const Key, T>>>
class multimap;
```

- **Key**: 键类型
- **T**: 值类型
- **Compare**: 比较函数
- **Allocator**: 内存分配器

### 为什么选择 std::multimap？

```
┌──────────────────────────────────────────────┐
│        📦 std::multimap 的优势                │
├──────────────────────────────────────────────┤
│ ✅ 自动排序：键自动按顺序存储                 │
│ ✅ 允许重复键：支持一对多映射                 │
│ ✅ 有序遍历：遍历时自动有序                   │
│ ✅ 范围查询：支持lower_bound/upper_bound     │
│ ✅ 稳定迭代：迭代器相对稳定                   │
└──────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | std::map | std::multimap |
|------|----------|---------------|
| 键唯一 | ✅ | ❌ 允许重复 |
| operator[] | ✅ | ❌ |
| 自动排序 | ✅ | ✅ |
| erase(key) | 删除单个键 | 删除**所有**该键 |
| 底层实现 | 红黑树 | 红黑树 |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造
std::multimap<std::string, int> mm1;

// 2. 从范围构造
std::vector<std::pair<std::string, int>> v = {{"a", 1}, {"a", 2}};
std::multimap<std::string, int> mm2(v.begin(), v.end());

// 3. 初始化列表构造
std::multimap<std::string, int> mm3 = {{"key1", 1}, {"key1", 2}};

// 4. 自定义比较器
std::multimap<std::string, int, std::greater<std::string>> mm4;

// 5. 拷贝构造
std::multimap<std::string, int> mm5(mm3);
```

### 元素访问

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `find(key)` | O(log n) | 查找键 |
| `count(key)` | O(log n + count) | 计数 |
| `lower_bound(key)` | O(log n) | 第一个>=key的位置 |
| `upper_bound(key)` | O(log n) | 第一个>key的位置 |
| `equal_range(key)` | O(log n) | 返回所有相同键的范围 |

```cpp
std::multimap<std::string, int> mm = {{"a", 1}, {"a", 2}, {"b", 3}};

// find - 返回第一个匹配
auto it = mm.find("a");

// count - 返回匹配数量
size_t n = mm.count("a");  // 2

// lower_bound - 第一个>=key的位置
auto lower = mm.lower_bound("a");

// upper_bound - 第一个>key的位置
auto upper = mm.upper_bound("a");

// equal_range - 获取所有相同键的元素
auto range = mm.equal_range("a");
for (auto i = range.first; i != range.second; ++i) {
    std::cout << i->second << " ";  // 1 2
}
```

### 修改操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `insert(pair)` | O(log n) | 插入键值对，总是成功 |
| `emplace(key, val)` | O(log n) | 原位构造插入 |
| `erase(key)` | O(log n + count) | 删除所有该键 |
| `erase(iterator)` | O(log n) | 删除单个元素 |

```cpp
std::multimap<std::string, int> mm;

// insert - 总是成功，返回iterator
auto it = mm.insert({"key", 1});
mm.insert({"key", 2});
mm.insert({"key", 3});

// emplace - 原位构造
mm.emplace("other", 10);

// erase(key) - 删除所有该键
size_t count = mm.erase("key");  // 删除3个

// erase(iterator) - 删除单个
auto it2 = mm.find("other");
if (it2 != mm.end()) {
    mm.erase(it2);  // 只删除一个
}
```

---

## 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| insert | **O(log n)** |
| erase(key) | **O(log n + count)** |
| erase(iterator) | **O(log n)** |
| find | **O(log n)** |
| count | **O(log n + count)** |
| lower_bound | **O(log n)** |
| upper_bound | **O(log n)** |

---

## 使用场景

### 1. 一对多映射关系

```cpp
std::multimap<std::string, std::string> phonebook;

// 添加多个号码
phonebook.insert({"Alice", "123-456-7890"});
phonebook.insert({"Alice", "987-654-3210"});
phonebook.insert({"Bob", "555-1234"});

// 查询Alice的所有号码
auto range = phonebook.equal_range("Alice");
for (auto it = range.first; it != range.second; ++it) {
    std::cout << it->second << std::endl;
}
```

### 2. 多值索引

```cpp
std::multimap<std::string, int> index;

// 添加标签索引
index.emplace("important", 1);
index.emplace("important", 2);
index.emplace("important", 3);
index.emplace("urgent", 4);

// 获取所有标记为"important"的项
auto range = index.equal_range("important");
for (auto it = range.first; it != range.second; ++it) {
    std::cout << "Item: " << it->second << std::endl;
}
```

### 3. 时间线/日志

```cpp
std::multimap<long, std::string> timeline;

// 添加事件（可能同时发生）
timeline.emplace(1000, "event1");
timeline.emplace(1000, "event2");
timeline.emplace(2000, "event3");

// 按时间顺序遍历
for (const auto& [time, event] : timeline) {
    std::cout << "Time " << time << ": " << event << std::endl;
}
```

### 4. 学生成绩管理

```cpp
std::multimap<std::string, double> grades;

// 添加学生成绩
grades.emplace("Alice", 95.5);
grades.emplace("Bob", 87.3);
grades.emplace("Alice", 92.0);  // Alice的另一次成绩

// 查询Alice的所有成绩
auto range = grades.equal_range("Alice");
double sum = 0;
int count = 0;
for (auto it = range.first; it != range.second; ++it) {
    sum += it->second;
    count++;
}
double average = sum / count;
```

---

## 注意事项

### 1. 无 operator[]

```cpp
std::multimap<std::string, int> mm;

// ❌ 编译错误：multimap 不支持 operator[]
// mm["key"] = 5;

// ✅ 使用 insert 或 emplace
mm.insert({"key", 5});
mm.emplace("key", 5);
```

### 2. insert 返回 iterator 而非 pair

```cpp
std::multimap<std::string, int> mm;

// map: 返回 pair<iterator, bool>
// std::map<std::string, int> m;
// auto [it, inserted] = m.insert({"key", 1});

// multimap: 总是返回 iterator
auto it = mm.insert({"key", 1});  // 总是成功
mm.insert({"key", 2});            // 允许重复键
```

### 3. erase(key) 删除所有该键

```cpp
std::multimap<std::string, int> mm = {{"a", 1}, {"a", 2}, {"b", 3}};

// erase(key) - 删除所有该键
size_t count = mm.erase("a");  // 删除2个，返回2

// 如果只想删除一个，使用迭代器
auto it = mm.find("a");
if (it != mm.end()) {
    mm.erase(it);  // 只删除一个
}
```

### 4. 有序性

```cpp
std::multimap<std::string, int> mm = {{"c", 3}, {"a", 1}, {"b", 2}};

// 遍历自动按键有序
for (const auto& [key, val] : mm) {
    std::cout << key << ": " << val << std::endl;  // a:1, b:2, c:3
}
```

---

## 常见问题

### Q1: multimap 和 map 的区别？

| 特性 | std::map | std::multimap |
|------|----------|---------------|
| 键唯一 | ✅ | ❌ 允许重复 |
| operator[] | ✅ | ❌ |
| 自动排序 | ✅ | ✅ |
| insert返回 | pair<it, bool> | iterator |
| erase(key) | 删除单个键 | 删除所有键 |

### Q2: 何时使用 multimap？

✅ **适合**：
- 一对多映射关系
- 需要按键排序
- 需要存储多个相同键的条目
- 需要范围查询

❌ **不适合**：
- 键必须唯一 → 使用 map
- 不需要有序 → 使用 unordered_multimap
- 需要快速查找 → 使用 unordered_multimap

### Q3: 如何获取所有相同键的值？

```cpp
std::multimap<std::string, int> mm = {{"a", 1}, {"a", 2}, {"a", 3}};

// 方法1：使用 equal_range
auto range = mm.equal_range("a");
for (auto it = range.first; it != range.second; ++it) {
    std::cout << it->second << " ";  // 1 2 3
}

// 方法2：使用 find 和 count
auto it = mm.find("a");
for (int i = 0; i < mm.count("a"); ++i) {
    std::cout << it->second << " ";
    ++it;
}
```

### Q4: 如何按自定义顺序排序？

```cpp
// 降序排列
std::multimap<std::string, int, std::greater<std::string>> mm_desc;

// 自定义比较器
struct CustomCompare {
    bool operator()(const std::string& a, const std::string& b) const {
        return a > b;  // 降序
    }
};
std::multimap<std::string, int, CustomCompare> mm_custom;

// Lambda比较器（C++11）
auto cmp = [](const std::string& a, const std::string& b) { return a > b; };
std::multimap<std::string, int, decltype(cmp)> mm_lambda(cmp);
```

---

## 总结

### 何时使用 std::multimap

✅ **适合**：
- 一对多映射关系
- 需要按键排序
- 需要存储多个相同键的条目
- 需要范围查询

❌ **不适合**：
- 键必须唯一 → 使用 map
- 不需要有序 → 使用 unordered_multimap
- 需要快速查找 → 使用 unordered_multimap

### 最佳实践

1. **使用 equal_range()** 获取所有相同键的元素
2. **记住无 operator[]** 使用 insert/emplace
3. **小心 erase(key)** 会删除所有该键
4. **利用自动排序** 进行有序遍历
5. **使用范围查询** 进行高效的区间操作

---

## 参考文档
- [cppreference - std::multimap](https://en.cppreference.com/w/cpp/container/multimap)
