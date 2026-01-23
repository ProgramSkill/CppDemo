# std::multimap 详细解析

## 概述

`std::multimap`允许**重复键**的有序键值对容器。

```cpp
#include <map>
```

## 核心特性

| 特性 | std::map | std::multimap |
|------|----------|---------------|
| 键唯一 | ✅ | ❌ 允许重复 |
| operator[] | ✅ | ❌ |
| 自动排序 | ✅ | ✅ |
| erase(key) | 删除单个键 | 删除**所有**该键 |

## 基本用法

```cpp
multimap<string, int> mm;

// 允许重复键
mm.insert({"key", 1});
mm.insert({"key", 2});
mm.insert({"key", 3});
// mm.count("key") == 3

// 无operator[]
// mm["key"] = 5;  // ❌ 编译错误
```

## 关键操作

### 插入
```cpp
multimap<string, int> mm;

// insert - 总是成功
mm.insert({"apple", 5});
mm.insert({"apple", 2});  // 允许重复
mm.emplace("banana", 3);
```

### 删除
```cpp
multimap<string, int> mm = {{"a", 1}, {"a", 2}, {"b", 3}};

// erase(key) - 删除所有该键
size_t count = mm.erase("a");  // 删除2个

// erase(iterator) - 删除单个
auto it = mm.find("a");
if (it != mm.end()) {
    mm.erase(it);  // 只删除一个
}
```

### 查找和访问
```cpp
multimap<int, string> mm = {{1, "one-a"}, {1, "one-b"}, {2, "two"}};

// find - 返回第一个匹配
auto it = mm.find(1);

// count - 返回匹配数量
size_t n = mm.count(1);  // 2

// equal_range - 获取所有相同键的元素
auto range = mm.equal_range(1);
for (auto i = range.first; i != range.second; ++i) {
    cout << i->second << " ";  // one-a one-b
}
```

## 使用场景

```cpp
// 1. 一对多关系
multimap<string, string> phonebook;

phonebook.insert({"Alice", "123-456-7890"});
phonebook.insert({"Alice", "987-654-3210"});  // 多个号码

// 查询Alice的所有号码
auto range = phonebook.equal_range("Alice");
for (auto it = range.first; it != range.second; ++it) {
    cout << it->second << endl;
}

// 2. 多值索引
multimap<string, int> index;

index.emplace("tag1", 1);
index.emplace("tag2", 2);
index.emplace("tag1", 3);  // 相同标签

// 3. 时间线/日志
multimap<time_t, string> timeline;
timeline.emplace(now(), "event1");
timeline.emplace(now(), "event2");  // 相同时间
```

## 何时使用multimap

✅ **适合**：
- 一对多映射关系
- 需要按键排序
- 需要存储多个相同键的条目

❌ **不适合**：
- 键必须唯一 → map
- 不需要有序 → unordered_multimap

## 参考文档
- [cppreference - std::multimap](https://en.cppreference.com/w/cpp/container/multimap)
