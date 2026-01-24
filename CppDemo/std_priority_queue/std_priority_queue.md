# std::priority_queue 详细解析

## 目录

1. [概述](#概述)
2. [核心特性](#核心特性)
3. [成员函数详解](#成员函数详解)
4. [时间复杂度](#时间复杂度)
5. [自定义比较](#自定义比较)
6. [使用场景](#使用场景)
7. [注意事项](#注意事项)
8. [常见问题](#常见问题)

---

## 概述

`std::priority_queue`是**优先队列容器适配器**，基于堆数据结构实现，默认为**最大堆**。

### 定义位置

```cpp
#include <queue>
```

### 模板声明

```cpp
template<class T, class Container = std::vector<T>,
         class Compare = std::less<T>>
class priority_queue;
```

- **T**: 元素类型
- **Container**: 底层容器，默认为 `std::vector<T>`
- **Compare**: 比较函数，默认为 `std::less<T>`（最大堆）

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 堆结构 | 基于二叉堆实现 |
| 默认最大堆 | 最大元素优先出队 |
| 底层容器 | std::vector |
| 无迭代器 | 不支持遍历 |
| 优先级访问 | 只能访问优先级最高的元素 |

---

## 成员函数详解

### 构造函数

| 函数 | 说明 |
|------|------|
| `priority_queue()` | 默认构造，空队列 |
| `priority_queue(const Compare&)` | 指定比较器 |
| `priority_queue(const priority_queue&)` | 拷贝构造 |
| `priority_queue(priority_queue&&)` | 移动构造 (C++11) |

```cpp
// 1. 默认构造（最大堆）
std::priority_queue<int> pq1;

// 2. 最小堆
std::priority_queue<int, std::vector<int>, std::greater<int>> pq2;

// 3. 自定义比较器
auto cmp = [](int a, int b) { return a > b; };
std::priority_queue<int, std::vector<int>, decltype(cmp)> pq3(cmp);

// 4. 从容器构造
std::vector<int> v = {3, 1, 4, 1, 5};
std::priority_queue<int> pq4(v.begin(), v.end());
```

### 修改操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `push(const T&)` | O(log n) | 插入元素 |
| `push(T&&)` | O(log n) | 移动插入元素 |
| `emplace(Args&&...)` | O(log n) | 原位构造插入 |
| `pop()` | O(log n) | 删除优先级最高的元素 |
| `swap(priority_queue&)` | O(1) | 交换内容 |

```cpp
std::priority_queue<int> pq;

// push - 插入
pq.push(3);
pq.push(1);
pq.push(4);

// emplace - 原位构造
pq.emplace(2);

// pop - 删除最大元素
pq.pop();

// swap - 交换
std::priority_queue<int> other;
pq.swap(other);
```

### 查询操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `top()` | O(1) | 返回优先级最高的元素 |
| `empty()` | O(1) | 检查是否为空 |
| `size()` | O(1) | 返回元素数量 |

```cpp
std::priority_queue<int> pq;
pq.push(10);
pq.push(20);
pq.push(5);

// top - 访问最大元素
int max_val = pq.top();  // 20

// empty - 检查是否为空
if (!pq.empty()) {
    std::cout << "Not empty" << std::endl;
}

// size - 获取大小
std::cout << "Size: " << pq.size();  // 3
```

---

## 时间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| push | **O(log n)** | 插入并调整堆 |
| pop | **O(log n)** | 删除并调整堆 |
| top | **O(1)** | 常数时间 |
| empty | **O(1)** | 常数时间 |
| size | **O(1)** | 常数时间 |

---

## 自定义比较

### 最大堆 vs 最小堆

```cpp
// 最大堆（默认）
std::priority_queue<int> max_pq;
max_pq.push(3);
max_pq.push(1);
max_pq.push(4);
std::cout << max_pq.top();  // 4

// 最小堆
std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
min_pq.push(3);
min_pq.push(1);
min_pq.push(4);
std::cout << min_pq.top();  // 1
```

### 自定义类型

```cpp
struct Task {
    int priority;
    std::string name;

    // 定义比较操作符
    bool operator<(const Task& other) const {
        return priority < other.priority;  // 优先级高的在前
    }
};

std::priority_queue<Task> tasks;
tasks.emplace(1, "low");
tasks.emplace(3, "high");
tasks.emplace(2, "medium");

while (!tasks.empty()) {
    std::cout << tasks.top().name << " ";  // high medium low
    tasks.pop();
}
```

### Lambda 比较器

```cpp
auto cmp = [](const std::pair<int, std::string>& a,
              const std::pair<int, std::string>& b) {
    return a.first < b.first;  // 按第一个元素比较
};

std::priority_queue<std::pair<int, std::string>,
                    std::vector<std::pair<int, std::string>>,
                    decltype(cmp)> pq(cmp);

pq.emplace(3, "C");
pq.emplace(1, "A");
pq.emplace(2, "B");

while (!pq.empty()) {
    std::cout << pq.top().second << " ";  // C B A
    pq.pop();
}
```

---

## 使用场景

### 1. Top K 问题

```cpp
// 找最大的 k 个元素
std::vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6};
int k = 3;

// 用最小堆维护最大的 k 个
std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
for (int num : nums) {
    pq.push(num);
    if (pq.size() > k) pq.pop();
}

// pq 中是最大的 3 个：9, 6, 5
```

### 2. 任务调度

```cpp
struct Task {
    int priority;
    std::string name;

    bool operator<(const Task& other) const {
        return priority < other.priority;
    }
};

std::priority_queue<Task> scheduler;
scheduler.emplace(1, "low priority");
scheduler.emplace(3, "high priority");
scheduler.emplace(2, "medium priority");

while (!scheduler.empty()) {
    Task task = scheduler.top();
    scheduler.pop();
    std::cout << "Executing: " << task.name << std::endl;
}
// 输出：high priority, medium priority, low priority
```

### 3. 哈夫曼编码

```cpp
struct Node {
    int freq;
    char ch;

    bool operator>(const Node& other) const {
        return freq > other.freq;
    }
};

std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
pq.emplace(5, 'a');
pq.emplace(9, 'b');
pq.emplace(12, 'c');

// 构建哈夫曼树...
```

### 4. Dijkstra 算法

```cpp
struct Edge {
    int dist;
    int node;

    bool operator>(const Edge& other) const {
        return dist > other.dist;
    }
};

std::priority_queue<Edge, std::vector<Edge>, std::greater<Edge>> pq;
// 使用最小堆找最短路径
```

---

## 注意事项

### 1. 访问空队列是未定义行为

```cpp
std::priority_queue<int> pq;

// ❌ 未定义行为
// int x = pq.top();

// ✅ 先检查
if (!pq.empty()) {
    int x = pq.top();
}
```

### 2. pop() 不返回值

```cpp
std::priority_queue<int> pq;
pq.push(10);

// ❌ 错误：pop() 返回 void
// int x = pq.pop();

// ✅ 正确：先 top 再 pop
int x = pq.top();
pq.pop();
```

### 3. 不支持迭代器

```cpp
std::priority_queue<int> pq;
// ... 添加元素 ...

// ❌ 没有迭代器
// for (auto it = pq.begin(); it != pq.end(); ++it) {}

// ✅ 只能逐个 pop 遍历
while (!pq.empty()) {
    std::cout << pq.top() << " ";
    pq.pop();
}
```

### 4. 比较器必须定义严格弱序

```cpp
// ❌ 错误：不是严格弱序
auto bad_cmp = [](int a, int b) { return a <= b; };

// ✅ 正确：严格弱序
auto good_cmp = [](int a, int b) { return a < b; };
```

---

## 常见问题

### Q1: priority_queue 和 queue 的区别？

| 特性 | std::queue | std::priority_queue |
|------|-----------|-------------------|
| 顺序 | FIFO | 优先级排序 |
| 出队顺序 | 先进先出 | 优先级最高先出 |
| 底层容器 | deque/list | vector + heap |
| 用途 | 任务队列 | 优先队列 |

### Q2: 如何实现最小堆？

```cpp
// 方法1：使用 greater
std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;

// 方法2：自定义比较器
auto cmp = [](int a, int b) { return a > b; };
std::priority_queue<int, std::vector<int>, decltype(cmp)> min_pq(cmp);
```

### Q3: 如何修改已有元素的优先级？

```cpp
// priority_queue 不支持直接修改
// 解决方案：删除后重新插入
std::priority_queue<int> pq;
pq.push(5);

// 要修改 5 为 10
// 1. 创建新队列
std::priority_queue<int> new_pq;
while (!pq.empty()) {
    int val = pq.top();
    pq.pop();
    if (val == 5) val = 10;
    new_pq.push(val);
}
pq = new_pq;
```

### Q4: 如何遍历 priority_queue？

```cpp
std::priority_queue<int> pq;
pq.push(3);
pq.push(1);
pq.push(4);

// 方法1：清空式遍历（会修改队列）
while (!pq.empty()) {
    std::cout << pq.top() << " ";
    pq.pop();
}

// 方法2：副本遍历
std::priority_queue<int> temp = pq;
while (!temp.empty()) {
    std::cout << temp.top() << " ";
    temp.pop();
}
```

---

## 总结

### 何时使用 std::priority_queue

✅ **适合**：
- 需要按优先级处理元素
- Top K 问题
- 任务调度
- Dijkstra/Prim 算法
- 堆排序

❌ **不适合**：
- FIFO 语义 → 使用 queue
- 需要随机访问 → 使用 vector
- 需要修改优先级 → 考虑其他数据结构

### 最佳实践

1. **总是检查 empty()** 在调用 top() 或 pop() 前
2. **使用 emplace()** 而非 push() 以避免临时对象
3. **选择合适的比较器** 根据需求（最大堆/最小堆）
4. **记住 pop() 不返回值** 需要先 top() 再 pop()

---

## 参考文档
- [cppreference - std::priority_queue](https://en.cppreference.com/w/cpp/container/priority_queue)
- [cppreference - std::less](https://en.cppreference.com/w/cpp/utility/functional/less)
- [cppreference - std::greater](https://en.cppreference.com/w/cpp/utility/functional/greater)
