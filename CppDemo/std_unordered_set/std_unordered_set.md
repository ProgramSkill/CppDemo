# std::unordered_set 详细解析

## 概述

`std::unordered_set`是基于哈希表的关联容器，存储唯一的**无序**元素。

```cpp
#include <unordered_set>
```

## 核心特性

| 特性 | std::set | std::unordered_set |
|------|---------|---------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找效率 | O(log n) | **O(1)平均** |
| 插入效率 | O(log n) | **O(1)平均** |
| 迭代器 | 稳定 | 可能因rehash失效 |

## 性能对比

```
┌─────────────────────────────────────────────────┐
│            查找性能对比（n=1,000,000）          │
├─────────────────────────────────────────────────┤
│ std::set:        O(log n) ≈ 20次比较           │
│ std::unordered_set: O(1) ≈ 1次哈希+查找        │
│                                                 │
│ 速度差异：unordered_set 通常快 5-10 倍         │
└─────────────────────────────────────────────────┘
```

## 哈希策略

```cpp
unordered_set<int> us = {1, 2, 3, 4, 5};

// 哈希策略
cout << "bucket_count: " << us.bucket_count();  // 桶数量
cout << "max_bucket_count: " << us.max_bucket_count();
cout << "load_factor: " << us.load_factor();    // 负载因子
cout << "max_load_factor: " << us.max_load_factor();

// 桶信息
for (size_t i = 0; i < us.bucket_count(); ++i) {
    cout << "bucket " << i << ": " << us.bucket_size(i);
}

// 手动调整
us.rehash(20);    // 设置桶数量
us.reserve(100);  // 预留空间
```

## 自定义哈希

```cpp
// 自定义类型作为键
struct Person {
    string name;
    int age;

    bool operator==(const Person& other) const {
        return name == other.name && age == other.age;
    }
};

// 自定义哈希函数
struct PersonHash {
    size_t operator()(const Person& p) const {
        return hash<string>()(p.name) ^ hash<int>()(p.age);
    }
};

unordered_set<Person, PersonHash> people;
```

## 使用场景

```cpp
// 1. 快速查找（不需要顺序）
unordered_set<int> cache;
if (cache.find(x) != cache.end()) {
    cout << "缓存命中";
}

// 2. 去重（不需要保持顺序）
vector<int> vec = {1, 2, 2, 3, 3, 3};
unordered_set<int> unique(vec.begin(), vec.end());

// 3. 集合运算（无序）
unordered_set<int> a = {1, 2, 3};
unordered_set<int> b = {3, 4, 5};

// 判断交集
bool has_common = false;
for (int x : a) {
    if (b.count(x)) {
        has_common = true;
        break;
    }
}
```

## 何时使用unordered_set

✅ **适合**：
- 需要快速查找（O(1)）
- 不需要有序
- 有好的哈希函数
- 需要频繁插入删除

❌ **不适合**：
- 需要有序 → set
- 需要范围查询 → set
- 哈希函数质量差 → set
- 内存受限 → set开销更小

## 参考文档
- [cppreference - std::unordered_set](https://en.cppreference.com/w/cpp/container/unordered_set)
