# std::list 详细解析

## 概述

`std::list`是C++标准库的双向链表容器，支持任意位置的快速插入删除。

```cpp
#include <list>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 双向链表 | 每个节点包含前驱和后继指针 |
| 任意位置操作 | 插入删除O(1) |
| 迭代器稳定 | 插入删除不使迭代器失效 |
| 不支持随机访问 | 无operator[] |

## 成员函数

### 元素访问
```cpp
list<int> l = {10, 20, 30};

l.front();  // 10 - 访问首元素
l.back();   // 30 - 访问尾元素
// l[2];    // ❌ 不支持随机访问
```

### 修改操作
```cpp
list<int> l;

// 双端操作
l.push_front(1);
l.push_back(2);
l.pop_front();
l.pop_back();

// 任意位置插入
auto it = l.begin();
l.insert(it, 99);        // 在it前插入
l.insert(it, 3, 100);    // 插入3个100

// 任意位置删除
l.erase(it);              // 删除it位置
l.erase(l.begin(), l.end()); // 删除范围

// 链表特有操作
l.remove(100);            // 删除所有等于100的元素
l.remove_if([](int n) { return n % 2 == 0; }); // 删除偶数
l.unique();               // 删除连续重复元素
l.reverse();              // 反转链表
l.sort();                 // 排序（链表专用，比通用算法优）
```

### 特殊操作（splice）
```cpp
list<int> l1 = {1, 2, 3};
list<int> l2 = {4, 5, 6};

auto it = l1.begin();
l1.splice(it, l2);  // 将l2所有元素移到l1的it前
// l1: {4, 5, 6, 1, 2, 3}
// l2: 空

l1.splice(it, l2, l2.begin(), l2.end());  // 移动范围
```

## 时间复杂度

| 操作 | 复杂度 |
|------|--------|
| 访问首尾 | O(1) |
| 任意位置插入/删除 | O(1) |
| 访问中间元素 | O(n) - 需遍历 |
| 排序（专用） | O(n log n) |
| size() | O(1) - C++11起 |

## 使用场景

```cpp
// 1. 频繁在中间插入删除
list<int> l = {1, 2, 3, 4, 5};
auto it = l.begin();
advance(it, 2);
l.insert(it, 99);  // O(1)，比vector快

// 2. 需要稳定的迭代器
for (auto it = l.begin(); it != l.end(); ++it) {
    l.insert(it, *it * 2);  // it仍然有效
}

// 3. 实现LRU缓存
list<int> cache;
int access(int page) {
    cache.remove(page);  // O(1)
    cache.push_front(page);  // O(1)
    if (cache.size() > CAPACITY) {
        cache.pop_back();
    }
    return page;
}
```

## 何时使用list

✅ **适合**：
- 频繁在中间插入删除
- 需要稳定的迭代器
- 实现LRU等缓存策略
- 不需要随机访问

❌ **不适合**：
- 需要随机访问 → 用vector/deque
- 数据量小 → vector可能更快（缓存友好）
- 主要在头尾操作 → deque更优

## 参考文档
- [cppreference - std::list](https://en.cppreference.com/w/cpp/container/list)
