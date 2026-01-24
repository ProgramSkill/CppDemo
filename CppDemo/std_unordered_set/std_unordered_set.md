# std::unordered_set 详细解析

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

`std::unordered_set`是基于**哈希表**的关联容器，存储唯一的**无序**元素。

### 定义位置

```cpp
#include <unordered_set>
```

### 模板声明

```cpp
template<class Key, class Hash = std::hash<Key>,
         class KeyEqual = std::equal_to<Key>,
         class Allocator = std::allocator<Key>>
class unordered_set;
```

- **Key**: 元素类型
- **Hash**: 哈希函数
- **KeyEqual**: 相等比较函数

### 为什么选择 std::unordered_set？

```
┌──────────────────────────────────────────────┐
│       📦 std::unordered_set 的优势            │
├──────────────────────────────────────────────┤
│ ✅ 快速查找：O(1)平均时间复杂度               │
│ ✅ 无序存储：不需要排序开销                   │
│ ✅ 元素唯一：自动去重                         │
│ ✅ 灵活哈希：支持自定义哈希函数               │
│ ✅ 高效操作：插入删除都是O(1)平均             │
└──────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | std::set | std::unordered_set |
|------|---------|---------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找效率 | O(log n) | **O(1)平均** |
| 插入效率 | O(log n) | **O(1)平均** |
| 迭代器稳定 | ✅ | ❌ rehash时失效 |

---

## 成员函数详解

### 构造函数

```cpp
// 1. 默认构造
std::unordered_set<int> us1;

// 2. 从范围构造
std::vector<int> v = {1, 2, 2, 3, 3, 3};
std::unordered_set<int> us2(v.begin(), v.end());

// 3. 初始化列表构造
std::unordered_set<int> us3 = {1, 2, 3, 4, 5};

// 4. 拷贝构造
std::unordered_set<int> us4(us3);

// 5. 自定义哈希函数
auto hash_fn = [](int x) { return std::hash<int>{}(x); };
std::unordered_set<int, decltype(hash_fn)> us5(0, hash_fn);
```

### 元素访问

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `find(key)` | O(1)平均 | 查找元素 |
| `count(key)` | O(1)平均 | 计数（0或1） |
| `contains(key)` | O(1)平均 | 检查是否存在 (C++20) |

```cpp
std::unordered_set<int> us = {1, 2, 3, 4, 5};

// find - 查找
auto it = us.find(3);
if (it != us.end()) {
    std::cout << "Found: " << *it << std::endl;
}

// count - 计数
if (us.count(3)) {
    std::cout << "Element exists" << std::endl;
}

// contains - 检查存在 (C++20)
if (us.contains(3)) {
    std::cout << "Element exists" << std::endl;
}
```

### 修改操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `insert(val)` | O(1)平均 | 插入元素 |
| `emplace(args)` | O(1)平均 | 原位构造插入 |
| `erase(key)` | O(1)平均 | 删除元素 |
| `clear()` | O(n) | 清空所有元素 |

```cpp
std::unordered_set<int> us;

// insert - 返回pair<iterator, bool>
auto [it, inserted] = us.insert(5);
if (inserted) {
    std::cout << "Inserted" << std::endl;
}

// emplace - 原位构造
us.emplace(10);

// erase - 删除
us.erase(5);

// clear - 清空
us.clear();
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

### 1. 快速查找（不需要顺序）

```cpp
std::unordered_set<int> cache = {1, 2, 3, 4, 5};

// 快速查找
if (cache.find(3) != cache.end()) {
    std::cout << "缓存命中" << std::endl;
}

// 或使用count
if (cache.count(3)) {
    std::cout << "缓存命中" << std::endl;
}
```

### 2. 去重（不需要保持顺序）

```cpp
std::vector<int> vec = {1, 2, 2, 3, 3, 3, 4, 5, 5};

// 快速去重
std::unordered_set<int> unique(vec.begin(), vec.end());

// 输出去重后的元素
for (int x : unique) {
    std::cout << x << " ";
}
```

### 3. 集合运算（无序）

```cpp
std::unordered_set<int> a = {1, 2, 3, 4};
std::unordered_set<int> b = {3, 4, 5, 6};

// 判断交集
bool has_common = false;
for (int x : a) {
    if (b.count(x)) {
        has_common = true;
        std::cout << "Common element: " << x << std::endl;
    }
}

// 计算并集
std::unordered_set<int> union_set(a.begin(), a.end());
union_set.insert(b.begin(), b.end());
```

### 4. 频率统计（去重计数）

```cpp
std::vector<std::string> words = {"apple", "banana", "apple", "cherry", "banana", "apple"};

// 统计不同单词数
std::unordered_set<std::string> unique_words(words.begin(), words.end());
std::cout << "Unique words: " << unique_words.size() << std::endl;  // 3
```

---

## 注意事项

### 1. 无序性

```cpp
std::unordered_set<int> us = {5, 1, 3, 2, 4};

// 遍历顺序不确定
for (int x : us) {
    std::cout << x << " ";  // 顺序不确定
}

// 如果需要有序，使用 set
```

### 2. 迭代器失效

```cpp
std::unordered_set<int> us = {1, 2, 3};
auto it = us.find(2);

// 插入可能导致rehash，迭代器失效
us.insert(4);
// it 可能失效
```

### 3. 哈希冲突

```cpp
std::unordered_set<int> us;

// 哈希冲突会导致性能下降
// 最坏情况：所有元素哈希到同一桶，O(n)

// 可以检查负载因子
std::cout << us.load_factor();      // 当前负载因子
std::cout << us.max_load_factor();  // 最大负载因子

// 手动调整
us.rehash(20);    // 设置桶数量
us.reserve(100);  // 预留空间
```

### 4. 自定义类型的哈希

```cpp
struct Person {
    std::string name;
    int age;

    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }
};

// 需要定义哈希函数
struct PersonHash {
    size_t operator()(const Person& p) const {
        return std::hash<std::string>{}(p.name) ^ (std::hash<int>{}(p.age) << 1);
    }
};

std::unordered_set<Person, PersonHash> people;
```

---

## 常见问题

### Q1: unordered_set 和 set 的区别？

| 特性 | std::set | std::unordered_set |
|------|----------|-------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找 | O(log n) | O(1)平均 |
| 遍历顺序 | 有序 | 无序 |
| 范围查询 | ✅ | ❌ |

### Q2: 何时使用 unordered_set？

✅ **适合**：
- 需要快速查找（O(1)）
- 不需要有序
- 有好的哈希函数
- 需要频繁插入删除
- 去重操作

❌ **不适合**：
- 需要有序 → 使用 set
- 需要范围查询 → 使用 set
- 哈希函数质量差 → 使用 set
- 内存受限 → set开销更小

### Q3: 如何自定义哈希函数？

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

std::unordered_set<std::string, StringHash> us;

// 方法2：使用Lambda（C++11）
auto hash_fn = [](const std::string& s) {
    return std::hash<std::string>{}(s);
};
std::unordered_set<std::string, decltype(hash_fn)> us2(0, hash_fn);
```

### Q4: 如何检查和优化哈希性能？

```cpp
std::unordered_set<int> us = {1, 2, 3, 4, 5};

// 检查哈希信息
std::cout << "Bucket count: " << us.bucket_count() << std::endl;
std::cout << "Load factor: " << us.load_factor() << std::endl;
std::cout << "Max load factor: " << us.max_load_factor() << std::endl;

// 查看每个桶的大小
for (size_t i = 0; i < us.bucket_count(); ++i) {
    std::cout << "Bucket " << i << ": " << us.bucket_size(i) << std::endl;
}

// 优化：预留空间
us.reserve(1000);  // 预留足够空间避免频繁rehash
```

---

## 总结

### 何时使用 std::unordered_set

✅ **适合**：
- 需要快速查找（O(1)）
- 不需要有序
- 去重操作
- 集合运算
- 缓存实现

❌ **不适合**：
- 需要有序 → 使用 set
- 需要范围查询 → 使用 set
- 需要遍历时有序 → 使用 set

### 最佳实践

1. **优先使用 find()** 而非 count() 进行查询
2. **使用 contains()** (C++20) 进行存在性检查
3. **自定义哈希函数** 对于复杂类型
4. **监控负载因子** 避免过多哈希冲突
5. **预留空间** 使用 reserve() 避免频繁rehash

---

## 参考文档
- [cppreference - std::unordered_set](https://en.cppreference.com/w/cpp/container/unordered_set)
