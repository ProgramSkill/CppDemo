# std::map 详细解析

## 目录

1. [概述](#概述)
2. [核心特性](#核心特性)
3. [成员函数详解](#成员函数详解)
4. [使用场景](#使用场景)
5. [注意事项](#注意事项)
6. [常见问题](#常见问题)

---

## 概述

`std::map` 是基于**红黑树**的关联容器，存储**键值对**（key-value pairs），键**唯一**且**自动排序**。

### 定义位置

```cpp
#include <map>
```

### 模板声明

```cpp
template<class Key, class T, class Compare = std::less<Key>,
         class Allocator = std::allocator<std::pair<const Key, T>>>
class map;
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 键值对 | 每个元素是pair<const Key, Value> |
| 键唯一 | 自动去重，重复键会替换值 |
| 自动排序 | 按键排序（默认升序） |
| 底层实现 | 红黑树 |
| 查找效率 | O(log n) |

## 基本用法

```cpp
// 初始化
map<string, int> m;

// 插入
m["apple"] = 5;        // operator[]
m.insert({"banana", 3});
m.emplace("cherry", 8);  // C++11

// 访问
int x = m["apple"];     // 如果不存在会创建默认值
int y = m.at("apple");  // 不存在抛出异常

// 检查存在
if (m.find("apple") != m.end()) {
    cout << "存在";
}
```

## 元素访问

| 方式 | 键不存在时 | 备注 |
|------|-----------|------|
| `m[key]` | 创建默认值并返回引用 | 可修改值 |
| `m.at(key)` | 抛出out_of_range | 只读或可写 |
| `m.find(key)` | 返回end() | 不创建元素 |

```cpp
map<string, int> m;

// m["apple"] - 存在则返回，不存在则创建
int count = m["apple"];  // 如果不存在，m["apple"]=0
// 现在m中有{"apple": 0}

// m.at() - 安全访问
try {
    int value = m.at("banana");  // 不存在会抛异常
} catch (const out_of_range& e) {
    cerr << "键不存在";
}
```

## 修改操作

```cpp
map<string, int> m;

// 插入或更新
m["key"] = value;  // 键存在则更新，不存在则插入

// insert - 键存在则失败
auto result = m.insert({"key", 100});
// result.second == false 表示键已存在

// insert_or_assign (C++17) - 总是成功
m.insert_or_assign("key", 200);  // 存在则更新，不存在则插入

// emplace - 原位构造
m.emplace("key", 300);
```

## 使用场景

```cpp
// 1. 字典/词频统计
map<string, int> word_count;
string word;
while (cin >> word) {
    word_count[word]++;  // 优雅
}

// 2. 配置存储
map<string, string> config;
config["host"] = "localhost";
config["port"] = "8080";
cout << config["host"] << ":" << config["port"];

// 3. 分组数据
map<int, vector<string>> groups;
groups[1].push_back("apple");
groups[1].push_back("banana");  // 组1

// 4. 有序键访问
map<int, string> m = {{3, "C"}, {1, "A"}, {2, "B"}};
for (auto& [k, v] : m) {
    cout << k << ":" << v << endl;  // 按k排序输出
}
```

## 自定义比较器

```cpp
// 降序
map<int, string, greater<int>> m;

// 自定义排序
struct Person {
    string name;
    int age;
};
auto cmp = [](const Person& a, const Person& b) {
    return a.age < b.age;
};
map<Person, int, decltype(cmp)> m(cmp);
```

## 何时使用map

✅ **适合**：
- 需要键值对存储
- 键唯一且需要有序
- 频繁按键查找
- 一对一映射关系

❌ **不适合**：
- 不需要有序 → unordered_map更快
- 需要重复键 → multimap
- 键是整数且范围小 → 考虑vector

## 成员函数详解

### 构造函数

| 函数 | 说明 |
|------|------|
| `map()` | 默认构造，空容器 |
| `map(const map&)` | 拷贝构造 |
| `map(map&&)` | 移动构造 (C++11) |
| `map(std::initializer_list)` | 初始化列表构造 (C++11) |

```cpp
// 1. 默认构造
std::map<string, int> m1;

// 2. 初始化列表
std::map<string, int> m2 = {{"apple", 5}, {"banana", 3}};

// 3. 拷贝构造
std::map<string, int> m3(m2);

// 4. 移动构造
std::map<string, int> m4(std::move(m2));
```

### 查询操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `find(key)` | O(log n) | 查找键，返回迭代器或end() |
| `count(key)` | O(log n) | 计数（0或1） |
| `contains(key)` | O(log n) | 检查是否存在 (C++20) |
| `lower_bound(key)` | O(log n) | 返回>=key的第一个元素 |
| `upper_bound(key)` | O(log n) | 返回>key的第一个元素 |
| `equal_range(key)` | O(log n) | 返回[lower_bound, upper_bound) |

```cpp
std::map<string, int> m = {{"apple", 5}, {"banana", 3}, {"cherry", 8}};

// find
auto it = m.find("apple");
if (it != m.end()) {
    cout << it->second;  // 5
}

// count
if (m.count("apple")) {
    cout << "存在";
}

// lower_bound / upper_bound
auto lower = m.lower_bound("b");  // 指向 banana
auto upper = m.upper_bound("b");  // 指向 cherry
```

### 插入/删除操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `insert(pair)` | O(log n) | 插入，键存在则失败 |
| `insert_or_assign(key, value)` | O(log n) | 插入或更新 (C++17) |
| `emplace(args)` | O(log n) | 原位构造插入 |
| `erase(key)` | O(log n) | 删除键 |
| `erase(iterator)` | O(log n) | 删除迭代器指向的元素 |
| `clear()` | O(n) | 清空所有元素 |

```cpp
std::map<string, int> m;

// insert
m.insert({"apple", 5});
auto result = m.insert({"apple", 10});
// result.second == false，因为键已存在

// insert_or_assign (C++17)
m.insert_or_assign("apple", 10);  // 更新为10

// emplace
m.emplace("banana", 3);

// erase
m.erase("apple");
```

---

## 注意事项

### 1. operator[] 的陷阱

```cpp
std::map<string, int> m;

// ❌ 危险：访问不存在的键会创建默认值
int x = m["nonexistent"];  // 现在m中有{"nonexistent": 0}

// ✅ 安全：使用find或at
if (m.find("key") != m.end()) {
    int y = m.at("key");
}
```

### 2. 键的不可修改性

```cpp
std::map<string, int> m = {{"apple", 5}};

// ❌ 错误：不能修改键
// m.begin()->first = "orange";  // 编译错误

// ✅ 可以修改值
m.begin()->second = 10;
```

### 3. 自定义比较器的const性

```cpp
auto cmp = [](const int& a, const int& b) {
    return a > b;  // 必须定义严格弱序
};

std::map<int, string, decltype(cmp)> m(cmp);
```

### 4. 迭代器稳定性

```cpp
std::map<int, string> m = {{1, "a"}, {2, "b"}, {3, "c"}};

auto it = m.find(2);
m.erase(1);  // ✅ it 仍然有效
m.erase(2);  // ❌ it 失效
```

---

## 常见问题

### Q1: map 和 unordered_map 的区别？

| 特性 | std::map | std::unordered_map |
|------|----------|-------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 查找复杂度 | O(log n) | O(1) 平均 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 内存开销 | 较小 | 较大 |
| 迭代器稳定 | ✅ 删除时稳定 | ❌ 可能失效 |

```cpp
// map - 有序
std::map<int, string> m = {{3, "C"}, {1, "A"}, {2, "B"}};
for (auto& [k, v] : m) {
    cout << k << ":" << v;  // 1:A, 2:B, 3:C
}

// unordered_map - 无序
std::unordered_map<int, string> um = {{3, "C"}, {1, "A"}, {2, "B"}};
for (auto& [k, v] : um) {
    cout << k << ":" << v;  // 顺序不确定
}
```

### Q2: map 和 multimap 的区别？

| 特性 | std::map | std::multimap |
|------|----------|---------------|
| 键唯一性 | ✅ 唯一 | ❌ 可重复 |
| operator[] | ✅ 支持 | ❌ 不支持 |
| 查找 | 返回单个值 | 返回范围 |

```cpp
// map - 键唯一
std::map<string, int> m;
m["apple"] = 5;
m["apple"] = 10;  // 覆盖

// multimap - 键可重复
std::multimap<string, int> mm;
mm.insert({"apple", 5});
mm.insert({"apple", 10});  // 两个都存在

// 查询所有值
auto range = mm.equal_range("apple");
for (auto it = range.first; it != range.second; ++it) {
    cout << it->second;  // 5, 10
}
```

### Q3: 何时使用 map？

✅ **适合**：
- 需要键值对存储且键唯一
- 需要按键有序遍历
- 频繁按键查找
- 键的类型支持比较

❌ **不适合**：
- 不需要有序 → unordered_map
- 需要重复键 → multimap
- 键是复杂对象且比较困难 → unordered_map

### Q4: 如何高效地批量插入？

```cpp
std::map<int, string> m;

// ❌ 低效：逐个插入
for (int i = 0; i < 1000; ++i) {
    m.insert({i, "value"});
}

// ✅ 高效：使用初始化列表或预构造
std::vector<std::pair<int, string>> data;
for (int i = 0; i < 1000; ++i) {
    data.push_back({i, "value"});
}
m.insert(data.begin(), data.end());
```

---

## 参考文档
- [cppreference - std::map](https://en.cppreference.com/w/cpp/container/map)
- [cppreference - std::multimap](https://en.cppreference.com/w/cpp/container/multimap)
- [cppreference - std::unordered_map](https://en.cppreference.com/w/cpp/container/unordered_map)
