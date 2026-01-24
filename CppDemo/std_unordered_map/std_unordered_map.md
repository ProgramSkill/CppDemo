# std::unordered_map 详细解析

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

`std::unordered_map`是基于**哈希表**的键值对容器，键唯一且**无序**存储。

### 定义位置

```cpp
#include <unordered_map>
```

### 模板声明

```cpp
template<class Key, class T, class Hash = std::hash<Key>,
         class KeyEqual = std::equal_to<Key>,
         class Allocator = std::allocator<std::pair<const Key, T>>>
class unordered_map;
```

- **Key**: 键类型
- **T**: 值类型
- **Hash**: 哈希函数
- **KeyEqual**: 键比较函数

### 为什么选择 std::unordered_map？

```
┌──────────────────────────────────────────────┐
│      📦 std::unordered_map 的优势             │
├──────────────────────────────────────────────┤
│ ✅ 快速查找：O(1)平均时间复杂度               │
│ ✅ 支持operator[]：直观的键值访问             │
│ ✅ 无序存储：不需要排序开销                   │
│ ✅ 灵活哈希：支持自定义哈希函数               │
│ ✅ 高效缓存：适合频繁查询场景                 │
└──────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | std::map | std::unordered_map |
|------|---------|--------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找效率 | O(log n) | **O(1)平均** |
| operator[] | ✅ | ✅ |
| 键唯一 | ✅ | ✅ |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造
std::unordered_map<std::string, int> um1;

// 2. 从范围构造
std::vector<std::pair<std::string, int>> v = {{"a", 1}, {"b", 2}};
std::unordered_map<std::string, int> um2(v.begin(), v.end());

// 3. 初始化列表构造
std::unordered_map<std::string, int> um3 = {{"key1", 1}, {"key2", 2}};

// 4. 拷贝构造
std::unordered_map<std::string, int> um4(um3);

// 5. 自定义哈希函数
auto hash_fn = [](const std::string& s) { return std::hash<std::string>{}(s); };
std::unordered_map<std::string, int, decltype(hash_fn)> um5(0, hash_fn);
```

### 元素访问

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `operator[]` | O(1)平均 | 访问或插入 |
| `at(key)` | O(1)平均 | 安全访问（范围检查） |
| `find(key)` | O(1)平均 | 查找键 |
| `count(key)` | O(1)平均 | 计数（0或1） |

```cpp
std::unordered_map<std::string, int> um = {{"a", 1}, {"b", 2}};

// operator[] - 访问或插入
int val = um["a"];           // 1
um["c"] = 3;                 // 插入新键

// at() - 安全访问
int val2 = um.at("a");       // 1
// um.at("x");               // 抛出out_of_range

// find() - 查找
auto it = um.find("a");
if (it != um.end()) {
    std::cout << it->second;  // 1
}

// count() - 计数
if (um.count("a")) {
    std::cout << "Found";
}
```

### 修改操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `insert(pair)` | O(1)平均 | 插入键值对 |
| `emplace(key, val)` | O(1)平均 | 原位构造插入 |
| `erase(key)` | O(1)平均 | 删除键 |
| `clear()` | O(n) | 清空所有元素 |

```cpp
std::unordered_map<std::string, int> um;

// insert - 返回pair<iterator, bool>
auto [it, inserted] = um.insert({"key", 1});
if (inserted) {
    std::cout << "Inserted";
}

// emplace - 原位构造
um.emplace("key2", 2);

// erase - 删除
um.erase("key");

// clear - 清空
um.clear();
```

---

## 时间复杂度

| 操作 | 平均 | 最坏 |
|------|------|------|
| insert | O(1) | O(n) |
| erase | O(1) | O(n) |
| find | O(1) | O(n) |
| count | O(1) | O(n) |
| operator[] | O(1) | O(n) |

---

## 使用场景

### 1. 缓存实现

```cpp
std::unordered_map<std::string, int> cache;

int get_value(const std::string& key) {
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;  // 缓存命中 O(1)
    }

    // 计算值
    int value = compute(key);
    cache[key] = value;     // 缓存结果
    return value;
}
```

### 2. 频率统计

```cpp
std::unordered_map<std::string, int> freq;

std::string word;
while (std::cin >> word) {
    freq[word]++;  // 非常简洁
}

// 输出频率
for (const auto& [word, count] : freq) {
    std::cout << word << ": " << count << std::endl;
}
```

### 3. 图的邻接表

```cpp
std::unordered_map<int, std::vector<int>> graph;

// 添加边
graph[1].push_back(2);
graph[1].push_back(3);
graph[2].push_back(3);

// 遍历邻接表
for (const auto& [node, neighbors] : graph) {
    std::cout << "Node " << node << ": ";
    for (int neighbor : neighbors) {
        std::cout << neighbor << " ";
    }
    std::cout << std::endl;
}
```

### 4. 配置字典

```cpp
std::unordered_map<std::string, std::string> config;

config["debug"] = "true";
config["log_level"] = "info";
config["port"] = "8080";

// 读取配置
std::string debug_mode = config["debug"];
std::string port = config.at("port");
```

### 5. 去重和计数

```cpp
std::vector<int> nums = {1, 2, 2, 3, 3, 3, 4};
std::unordered_map<int, int> count_map;

for (int num : nums) {
    count_map[num]++;
}

// 输出不重复的元素及其计数
for (const auto& [num, count] : count_map) {
    std::cout << num << " appears " << count << " times" << std::endl;
}
```

---

## 注意事项

### 1. operator[] 会插入

```cpp
std::unordered_map<std::string, int> um;

// ⚠️ 访问不存在的键会插入默认值
int val = um["nonexistent"];  // 插入 {"nonexistent", 0}

// ✅ 使用 find() 或 count() 避免插入
if (um.count("key")) {
    int val = um["key"];
}
```

### 2. 无序性

```cpp
std::unordered_map<std::string, int> um = {{"c", 3}, {"a", 1}, {"b", 2}};

// 遍历顺序不确定
for (const auto& [key, val] : um) {
    std::cout << key << ": " << val << std::endl;  // 顺序不确定
}

// 如果需要有序，使用 map
```

### 3. 哈希冲突

```cpp
std::unordered_map<int, std::string> um;

// 哈希冲突会导致性能下降
// 最坏情况：所有键哈希到同一桶，O(n)

// 可以检查负载因子
std::cout << um.load_factor();      // 当前负载因子
std::cout << um.max_load_factor();  // 最大负载因子
```

### 4. 自定义类型的哈希

```cpp
struct Point {
    int x, y;
};

// 需要定义哈希函数
struct PointHash {
    size_t operator()(const Point& p) const {
        return std::hash<int>{}(p.x) ^ (std::hash<int>{}(p.y) << 1);
    }
};

// 需要定义相等比较
struct PointEqual {
    bool operator()(const Point& a, const Point& b) const {
        return a.x == b.x && a.y == b.y;
    }
};

std::unordered_map<Point, std::string, PointHash, PointEqual> um;
```

---

## 常见问题

### Q1: unordered_map 和 map 的区别？

| 特性 | std::map | std::unordered_map |
|------|----------|-------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找 | O(log n) | O(1)平均 |
| 遍历顺序 | 有序 | 无序 |
| 范围查询 | ✅ | ❌ |

### Q2: 何时使用 unordered_map？

✅ **适合**：
- 需要最快查找
- 不需要有序键
- 键是简单类型（int、string等）
- 缓存、字典实现
- 频率统计

❌ **不适合**：
- 需要有序 → 使用 map
- 需要范围查询 → 使用 map
- 需要遍历时有序 → 使用 map

### Q3: 如何避免 operator[] 插入新键？

```cpp
std::unordered_map<std::string, int> um = {{"a", 1}};

// ❌ 会插入新键
// int val = um["b"];

// ✅ 方法1：使用 find()
auto it = um.find("b");
if (it != um.end()) {
    int val = it->second;
}

// ✅ 方法2：使用 count()
if (um.count("b")) {
    int val = um["b"];
}

// ✅ 方法3：使用 at()
try {
    int val = um.at("b");
} catch (const std::out_of_range&) {
    // 键不存在
}
```

### Q4: 如何自定义哈希函数？

```cpp
// 方法1：定义哈希函数类
struct StringHash {
    size_t operator()(const std::string& s) const {
        size_t hash = 0;
        for (char c : s) {
            hash = hash * 31 + c;
        }
        return hash;
    }
};

std::unordered_map<std::string, int, StringHash> um;

// 方法2：使用Lambda（C++11）
auto hash_fn = [](const std::string& s) {
    return std::hash<std::string>{}(s);
};
std::unordered_map<std::string, int, decltype(hash_fn)> um2(0, hash_fn);
```

---

## 总结

### 何时使用 std::unordered_map

✅ **适合**：
- 需要快速查找（O(1)）
- 不需要有序
- 键是简单类型
- 缓存、字典、计数

❌ **不适合**：
- 需要有序 → 使用 map
- 需要范围查询 → 使用 map
- 需要遍历时有序 → 使用 map

### 最佳实践

1. **优先使用 find()** 而非 operator[] 进行查询
2. **使用 at()** 进行安全访问
3. **自定义哈希函数** 对于复杂类型
4. **监控负载因子** 避免过多哈希冲突
5. **考虑 map** 如果需要有序或范围查询

---

## 参考文档
- [cppreference - std::unordered_map](https://en.cppreference.com/w/cpp/container/unordered_map)
