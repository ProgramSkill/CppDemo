# std::priority_queue 详细解析

## 概述

`std::priority_queue`是优先队列容器适配器，默认为**最大堆**。

```cpp
#include <queue>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 堆结构 | 默认最大堆 |
| 默认底层 | vector + 算法 |
| 无迭代器 | 不支持遍历 |
| 优先级访问 | 只能访问top |

## 基本用法

```cpp
// 最大堆（默认）
priority_queue<int> pq;
pq.push(3);
pq.push(1);
pq.push(4);
cout << pq.top();  // 4（最大）
pq.pop();
cout << pq.top();  // 3

// 最小堆
priority_queue<int, vector<int>, greater<int>> min_pq;
min_pq.push(3);
min_pq.push(1);
min_pq.push(4);
cout << min_pq.top();  // 1（最小）
```

## 自定义比较

```cpp
// 最大堆（默认）
auto cmp1 = less<int>();
priority_queue<int, vector<int>, decltype(cmp1)> pq1(cmp1);

// 最小堆
auto cmp2 = greater<int>();
priority_queue<int, vector<int>, decltype(cmp2)> pq2(cmp2);

// 自定义类型
struct Task {
    int priority;
    string name;
    bool operator<(const Task& other) const {
        return priority < other.priority;  // 优先级高的在前
    }
};
priority_queue<Task> tasks;
```

## 使用场景

```cpp
// 1. Top K问题
vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6};
int k = 3;

// 用最小堆找最大的k个
priority_queue<int, vector<int>, greater<int>> pq;
for (int num : nums) {
    pq.push(num);
    if (pq.size() > k) pq.pop();
}
// pq中是最大的3个

// 2. 任务调度
struct Task {
    int priority;
    function<void()> work;
    bool operator<(const Task& other) const {
        return priority < other.priority;
    }
};
priority_queue<Task> scheduler;
scheduler.emplace(1, []{ cout << "low"; });
scheduler.emplace(3, []{ cout << "high"; });
while (!scheduler.empty()) {
    auto task = scheduler.top(); scheduler.pop();
    task.work();
}

// 3. 哈夫曼编码
using Node = pair<int, char>;
auto cmp = [](Node& a, Node& b) { return a.first < b.first; };
priority_queue<Node, vector<Node>, decltype(cmp)> pq(cmp);
```

## 复杂度

| 操作 | 复杂度 |
|------|--------|
| push | O(log n) |
| pop | O(log n) |
| top | O(1) |

## 何时使用priority_queue

✅ **适合**：
- 需要按优先级处理
- Top K问题
- 任务调度
- 堆排序

## 参考文档
- [cppreference - std::priority_queue](https://en.cppreference.com/w/cpp/container/priority_queue)
