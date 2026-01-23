# std::multiset 详细解析

## 概述

`std::multiset`允许**重复元素**的有序集合，基于红黑树实现。

```cpp
#include <set>
```

## 核心特性

| 特性 | std::set | std::multiset |
|------|----------|---------------|
| 元素唯一 | ✅ | ❌ 允许重复 |
| 自动排序 | ✅ | ✅ |
| 插入结果 | 返回pair<it, bool> | 返回iterator |
| erase(value) | 删除单个元素 | 删除**所有**该值 |

## 基本用法

```cpp
multiset<int> ms;

// 允许重复
ms.insert(5);
ms.insert(5);
ms.insert(5);
// ms.size() == 3

// count返回实际数量
size_t n = ms.count(5);  // 3
```

## 关键操作

### 插入
```cpp
multiset<int> ms;

// insert - 总是成功，返回iterator
auto it = ms.insert(5);  // 返回指向插入元素的迭代器
ms.insert(5);
ms.insert(5);  // 三个5

// insert_range
int arr[] = {1, 2, 2, 3};
ms.insert(ms.begin(), ms.end(), arr, arr+4);
```

### 删除
```cpp
multiset<int> ms = {1, 2, 2, 3, 3, 3};

// erase(value) - 删除所有该值
size_t count = ms.erase(3);  // 删除3个3，返回3

// erase(iterator) - 删除单个
auto it = ms.find(2);
if (it != ms.end()) {
    ms.erase(it);  // 只删除一个2
}
```

### 查找
```cpp
multiset<int> ms = {1, 2, 2, 3, 3, 3};

// find - 返回第一个匹配
auto it = ms.find(3);  // 指向第一个3

// equal_range - 获取所有相同元素
auto range = ms.equal_range(3);
for (auto i = range.first; i != range.second; ++i) {
    cout << *i << " ";  // 输出：3 3 3
}
```

## 使用场景

```cpp
// 1. 允许重复的有序集合
multiset<int> scores;
scores.insert(90);
scores.insert(85);
scores.insert(90);  // 允许重复

// 2. 任务优先级队列（同优先级按时间）
multiset<pair<int, time_t>> tasks;
tasks.emplace(1, now());
tasks.emplace(1, now() + 1);  // 相同优先级

// 3. 统计出现次数
multiset<string> words;
words.insert("hello");
words.insert("hello");
words.insert("hello");
cout << words.count("hello");  // 3
```

## 何时使用multiset

✅ **适合**：
- 需要允许重复的有序集合
- 需要统计元素出现次数
- 相同值的元素需要区分

❌ **不适合**：
- 元素必须唯一 → set
- 不需要有序 → unordered_multiset

## 参考文档
- [cppreference - std::multiset](https://en.cppreference.com/w/cpp/container/multiset)
