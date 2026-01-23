# std::unordered_map 详细解析

## 概述

`std::unordered_map`是基于哈希表的键值对容器，键唯一且无序。

```cpp
#include <unordered_map>
```

## 核心特性

| 特性 | std::map | std::unordered_map |
|------|---------|--------------------|
| 底层实现 | 红黑树 | 哈希表 |
| 有序性 | ✅ 有序 | ❌ 无序 |
| 查找效率 | O(log n) | **O(1)平均** |
| operator[] | ✅ | ✅ |

## 性能优势

```cpp
// 性能测试：查找1000000次
map<int, int> m;
unordered_map<int, int> um;

// map: ~20次比较 × 1000000 = 慢
// unordered_map: ~1次哈希 × 1000000 = 快5-10倍
```

## 使用场景

```cpp
// 1. 缓存实现
unordered_map<string, int> cache;

int get(string key) {
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;  // 缓存命中 O(1)
    }
    int value = compute(key);
    cache[key] = value;  // 缓存结果
    return value;
}

// 2. 频率统计
unordered_map<string, int> freq;
string word;
while (cin >> word) {
    freq[word]++;  // 非常简洁
}

// 3. 图的邻接表
unordered_map<int, vector<int>> graph;
graph[1].push_back(2);
graph[1].push_back(3);  // 节点1的邻居

// 4. 配置字典
unordered_map<string, string> config;
config["debug"] = "true";
cout << config["debug"];
```

## 何时使用unordered_map

✅ **适合**：
- 需要最快查找
- 不需要有序键
- 键是简单类型（int, string等）
- 缓存、字典实现

❌ **不适合**：
- 需要有序 → map
- 需要范围查询 → map
- 需要遍历时有序 → map

## 参考文档
- [cppreference - std::unordered_map](https://en.cppreference.com/w/cpp/container/unordered_map)
