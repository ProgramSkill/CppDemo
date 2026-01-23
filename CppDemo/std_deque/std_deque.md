# std::deque 详细解析

## 概述

`std::deque`（双端队列，double-ended queue）是C++标准库的序列容器，支持在两端快速插入删除。

```cpp
#include <deque>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 双端操作 | O(1)时间在头部和尾部插入删除 |
| 随机访问 | 支持operator[]，O(1)时间 |
| 内存结构 | 分段连续内存（多个固定大小数组） |
| 自动扩容 | 无需手动管理内存 |

## 与vector对比

| 特性 | std::vector | std::deque |
|------|-------------|------------|
| 随机访问 | O(1) | O(1) |
| 尾部插入/删除 | O(1)* | O(1) |
| 头部插入/删除 | O(n) | **O(1)** |
| 中间插入/删除 | O(n) | O(n) |
| 内存结构 | 单一连续块 | 分段连续 |
| 扩容 | 需要整体拷贝 | 无需拷贝 |

*摊销常数时间

## 成员函数

### 元素访问
```cpp
deque<int> d = {10, 20, 30, 40, 50};

d[2];        // 30，无边界检查
d.at(2);     // 30，有边界检查
d.front();   // 10
d.back();    // 50
d.data();    // 指向底层数组
```

### 修改操作
```cpp
deque<int> d;

// 双端插入
d.push_back(1);   // 尾部插入
d.push_front(2);  // 头部插入

// 双端删除
d.pop_back();     // 尾部删除
d.pop_front();    // 头部删除

// 原位构造（C++11）
d.emplace_back(3);
d.emplace_front(4);
d.emplace(d.begin() + 1, 5);  // 位置插入
```

## 时间复杂度

| 操作 | 复杂度 |
|------|--------|
| 随机访问 | O(1) |
| 头/尾插入 | O(1) |
| 头/尾删除 | O(1) |
| 中间插入/删除 | O(n) |

## 使用场景

```cpp
// 1. 滑动窗口
deque<int> window;
for (int i = 0; i < n; ++i) {
    // 移除超出窗口的元素
    while (!window.empty() && window.front() <= i - k) {
        window.pop_front();
    }
    // 移除较小元素
    while (!window.empty() && data[window.back()] < data[i]) {
        window.pop_back();
    }
    window.push_back(i);
}

// 2. 需要频繁头尾操作
deque<int> dq;
dq.push_front(1);  // O(1)
dq.push_back(2);   // O(1)
dq.pop_front();    // O(1)
```

## 何时使用deque

✅ **适合**：
- 需要在两端频繁插入删除
- 实现队列或双端队列
- 滑动窗口算法
- 不希望vector扩容时的拷贝开销

❌ **不适合**：
- 主要在中间操作 → 用list
- 需要最极致的随机访问性能 → 用vector
- 数据量很小 → vector可能更简单

## 参考文档
- [cppreference - std::deque](https://en.cppreference.com/w/cpp/container/deque)
