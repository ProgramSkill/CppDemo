# std::queue 详细解析

## 目录

1. [概述](#概述)
2. [核心特性](#核心特性)
3. [底层容器](#底层容器)
4. [成员函数详解](#成员函数详解)
5. [使用场景](#使用场景)
6. [代码示例](#代码示例)
7. [性能考虑](#性能考虑)
8. [注意事项](#注意事项)
9. [常见问题](#常见问题)

---

## 概述

`std::queue` 是 C++ 标准库提供的**容器适配器**（Container Adapter），它实现了**先进先出**（FIFO - First In, First Out）的数据结构。

### 定义位置

```cpp
#include <queue>
```

### 模板声明

```cpp
template <class T, class Container = deque<T>>
class queue;
```

- **T**: 元素类型
- **Container**: 底层容器类型，默认为 `std::deque<T>`

---

## 核心特性

### 1. FIFO 语义
- **插入**：只能在容器尾部（back）进行
- **删除**：只能在容器头部（front）进行
- **访问**：只能访问队头和队尾元素

### 2. 限制访问
- ❌ 不提供迭代器（no iterators）
- ❌ 不支持随机访问
- ❌ 不能遍历所有元素

### 3. 操作保证
- 所有操作的时间复杂度都是常数时间 O(1)
- 异常安全保证：不抛出异常的操作提供 `noexcept` 保证

### 4. 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| 构造 | O(n) 或 O(1)* |
| 析构 | O(n) |
| `push()` / `emplace()` | **O(1)** |
| `pop()` | **O(1)** |
| `front()` / `back()` | **O(1)** |
| `size()` | **O(1)** |
| `empty()` | **O(1)** |
| `swap()` | **O(1)** |

\* 取决于底层容器的构造方式

---

## 底层容器

### 可用的底层容器

`std::queue` 可以使用满足以下条件的容器：

1. **必须支持的操作**：
   - `front()`
   - `back()`
   - `push_back()`
   - `pop_front()`
   - `size()`
   - `empty()`

2. **标准库容器选项**：

   | 容器 | 是否支持 | 说明 |
   |------|---------|------|
   | `std::deque` | ✅ | 默认选择，性能平衡 |
   | `std::list` | ✅ | 频繁插入删除时更优 |
   | `std::vector` | ❌ | 缺少 `pop_front()` |

### 底层容器示例

```cpp
// 使用 deque（默认）
std::queue<int> q1;

// 显式指定 deque
std::queue<int, std::deque<int>> q2;

// 使用 list
std::queue<int, std::list<int>> q3;

// 编译错误：vector 不支持
// std::queue<int, std::vector<int>> q4;
```

---

## 成员函数详解

### 构造函数

| 函数 | 说明 |
|------|------|
| `queue()` | 默认构造函数 |
| `explicit queue(const Container&)` | 以容器拷贝构造 |
| `explicit queue(Container&&)` | 以容器移动构造 (C++11) |
| `queue(const queue&)` | 拷贝构造函数 |
| `queue(queue&&)` | 移动构造函数 (C++11) |

```cpp
// 1. 默认构造函数
std::queue<int> q1;

// 2. 以容器拷贝构造
std::deque<int> deq = {1, 2, 3, 4, 5};
std::queue<int> q2(deq);  // q2 包含 {1, 2, 3, 4, 5}

// 3. 以容器移动构造
std::deque<int> deq2 = {10, 20, 30};
std::queue<int> q3(std::move(deq2));  // q3 包含 {10, 20, 30}，deq2 现为空

// 4. 拷贝构造函数
std::queue<int> q4(q2);  // q4 是 q2 的副本，包含 {1, 2, 3, 4, 5}

// 5. 移动构造函数
std::queue<int> q5(std::move(q3));  // q5 夺取 q3 的资源，q3 现为空
```

#### 构造函数区别说明

**`std::queue<int> q2(deq);` vs `std::queue<int> q4(q2);`**

| 特性 | `q2(deq)` | `q4(q2)` |
|------|-----------|----------|
| 参数类型 | `std::deque<int>`（容器） | `std::queue<int>`（队列） |
| 调用构造函数 | `explicit queue(const Container&)` | `queue(const queue&)` |
| 用途 | 从底层容器构造 queue | 复制已有的 queue |
| explicit 关键字 | ✅ 是 | ❌ 否 |
| 拷贝初始化语法 | `❌ std::queue<int> q = deq;` | `✅ std::queue<int> q = q2;` |

**示例**：
```cpp
std::deque<int> deq = {1, 2, 3};
std::queue<int> q1;

// 以容器构造 - explicit，不能用 =
// std::queue<int> q2 = deq;  // ❌ 编译错误
std::queue<int> q2(deq);      // ✅ 必须用括号

// 拷贝构造 - 非 explicit，可以用 =
std::queue<int> q3 = q1;      // ✅ 可以用 =
std::queue<int> q4(q1);       // ✅ 也可以用括号
```


### 元素访问

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `front()` | O(1) | 访问队头元素 |
| `back()` | O(1) | 访问队尾元素 |

```cpp
std::queue<int> q;
q.push(10);
q.push(20);
q.push(30);

std::cout << q.front();  // 10
std::cout << q.back();   // 30
```

### 修改操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `push(const value_type&)` | O(1) | 在队尾插入元素（拷贝） |
| `push(value_type&&)` | O(1) | 在队尾插入元素（移动，C++11） |
| `emplace(Args&&...)` | O(1) | 就地构造元素（C++11） |
| `pop()` | O(1) | 删除队头元素 |
| `swap(queue& other)` | O(1) | 交换内容（C++11） |

```cpp
std::queue<std::string> q;

// push - 拷贝/移动
std::string str = "hello";
q.push(str);           // 拷贝
q.push("world");       // 移动

// emplace - 就地构造
q.emplace("direct");   // 更高效，避免临时对象

// pop - 删除队头
q.pop();

// swap - 交换两个队列的内容
std::queue<std::string> q1, q2;
q1.push("one");
q1.push("two");
q2.push("three");

q1.swap(q2);  // O(1) 操作，交换内容
// q1 现在包含 {"three"}
// q2 现在包含 {"one", "two"}
```

### 容量查询

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `empty()` | O(1) | 检查是否为空 |
| `size()` | O(1) | 返回元素数量 |

```cpp
std::queue<int> q;
if (q.empty()) {
    std::cout << "Queue is empty";
}
std::cout << "Size: " << q.size();
```

### 比较运算符

| 运算符 | 说明 |
|--------|------|
| `operator==` | 相等比较 |
| `operator!=` | 不等比较 |
| `operator<` | 小于比较 |
| `operator<=` | 小于等于 |
| `operator>` | 大于比较 |
| `operator>=` | 大于等于 |

**注意**：要求两个队列的元素类型和底层容器类型都相同。

---

## 使用场景

### 1. 广度优先搜索（BFS）

```cpp
void bfs(int start) {
    std::queue<int> q;
    std::vector<bool> visited(n, false);

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        // 处理当前节点
        process(current);

        // 访问邻居
        for (int neighbor : getNeighbors(current)) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

### 2. 任务调度

```cpp
class TaskScheduler {
    std::queue<Task> taskQueue;

public:
    void addTask(const Task& task) {
        taskQueue.push(task);
    }

    void process() {
        while (!taskQueue.empty()) {
            Task task = taskQueue.front();
            taskQueue.pop();

            execute(task);
        }
    }
};
```

### 3. 层序遍历

```cpp
void levelOrder(TreeNode* root) {
    if (!root) return;

    std::queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();

        std::cout << node->val << " ";

        if (node->left) q.push(node->left);
        if (node->right) q.push(node->right);
    }
}
```

### 4. 缓冲区管理

```cpp
class DataBuffer {
    std::queue<Data> buffer;
    size_t maxSize;

public:
    DataBuffer(size_t size) : maxSize(size) {}

    void add(const Data& data) {
        if (buffer.size() >= maxSize) {
            buffer.pop();  // 移除最旧的数据
        }
        buffer.push(data);
    }
};
```

---

## 代码示例

### 基本使用

```cpp
#include <iostream>
#include <queue>

int main() {
    std::queue<int> q;

    // 入队
    for (int i = 1; i <= 5; ++i) {
        q.push(i);
    }

    // 出队并处理
    while (!q.empty()) {
        std::cout << q.front() << " ";
        q.pop();
    }
    // 输出: 1 2 3 4 5

    return 0;
}
```

### 自定义类型

```cpp
struct Message {
    int id;
    std::string content;
    int priority;

    Message(int i, const std::string& c, int p)
        : id(i), content(c), priority(p) {}
};

std::queue<Message> messageQueue;

// 添加消息
messageQueue.emplace(1, "Hello", 5);
messageQueue.emplace(2, "World", 3);

// 处理消息
while (!messageQueue.empty()) {
    Message msg = messageQueue.front();
    std::cout << "ID:" << msg.id
              << " Content:" << msg.content
              << " Priority:" << msg.priority << std::endl;
    messageQueue.pop();
}
```

### 交换队列内容

```cpp
std::queue<int> q1, q2;

q1.push(1);
q1.push(2);
q1.push(3);

q2.push(100);
q2.push(200);

// 交换内容 - O(1) 操作
q1.swap(q2);

// q1 现在包含 {100, 200}
// q2 现在包含 {1, 2, 3}
```

---

## 性能考虑

### 1. 底层容器选择

**std::deque（默认）**
- ✅ 最佳全能选择
- ✅ 内存连续性好
- ✅ 中间元素删除不影响两端
- ✅ 缓存友好

**std::list**
- ✅ 频繁插入删除更优
- ❌ 节点内存开销大
- ❌ 缓存不友好

### 2. emplace vs push

```cpp
std::queue<std::pair<int, std::string>> q;

// push - 创建临时对象
q.push(std::make_pair(1, "test"));

// emplace - 就地构造，更高效
q.emplace(1, "test");
```

**推荐**：优先使用 `emplace()`

### 3. 移动语义（C++11）

```cpp
std::queue<std::string> q1;
q1.push("Hello");
q1.push("World");

// 移动构造 - O(1)
std::queue<std::string> q2 = std::move(q1);

// q1 现在为空
// q2 拥有原 q1 的内容
```

---

## 注意事项

### 1. 空队列访问

```cpp
std::queue<int> q;

// 危险！未定义行为
int x = q.front();  // ❌ UB
q.pop();            // ❌ UB

// 正确做法
if (!q.empty()) {
    int x = q.front();  // ✅
    q.pop();            // ✅
}
```

### 2. front() 返回引用

```cpp
std::queue<std::string> q;
q.push("hello");

// front() 返回引用，可以修改
q.front() += " world";  // ✅ 修改队头元素

std::cout << q.front();  // "hello world"
```

### 3. 不提供迭代器

```cpp
std::queue<int> q;
// ... 添加元素 ...

// ❌ 没有迭代器
// for (auto it = q.begin(); it != q.end(); ++it) { }

// ✅ 只能逐个 pop 遍历（会清空队列）
while (!q.empty()) {
    std::cout << q.front() << " ";
    q.pop();
}

// ✅ 或者创建副本遍历
std::queue<int> temp = q;
while (!temp.empty()) {
    std::cout << temp.front() << " ";
    temp.pop();
}
```

### 4. pop() 不返回值

```cpp
std::queue<int> q;
// ...

// ❌ pop() 不返回值
// int x = q.pop();

// ✅ 正确做法
int x = q.front();
q.pop();
```

**设计原因**：如果 `pop()` 返回值且拷贝/移动构造函数抛出异常，元素已被删除但无法获取，导致数据丢失。

---

## 常见问题

### Q1: 为什么不能用 vector 作为底层容器？

**A**: `std::vector` 不提供 `pop_front()` 操作，因为删除首元素需要移动所有剩余元素，时间复杂度为 O(n)，不符合 queue 的 O(1) 要求。

### Q2: queue 和 deque 的区别？

| 特性 | std::queue | std::deque |
|------|-----------|-----------|
| 类型 | 容器适配器 | 容器 |
| 访问限制 | 只能访问首尾 | 可随机访问 |
| 迭代器 | 无 | 有 |
| 使用场景 | FIFO 语义 | 双端队列 |

### Q3: queue 和 priority_queue 的区别？

| 特性 | std::queue | std::priority_queue |
|------|-----------|-------------------|
| 顺序 | FIFO | 优先级排序 |
| 出队顺序 | 先进先出 | 优先级最高先出 |
| 底层容器 | deque/list | vector + heap |
| 用途 | 任务队列 | 优先队列 |

### Q4: 如何遍历 queue 的元素？

```cpp
// 方法 1：清空式遍历
while (!q.empty()) {
    process(q.front());
    q.pop();
}

// 方法 2：副本遍历
std::queue<int> temp = q;
while (!temp.empty()) {
    process(temp.front());
    temp.pop();
}

// 方法 3：使用底层容器（不推荐）
// 直接访问 q.* 不被允许
```

### Q5: 如何反转 queue？

```cpp
std::queue<int> reverseQueue(std::queue<int> q) {
    std::stack<int> s;

    // 转移到 stack
    while (!q.empty()) {
        s.push(q.front());
        q.pop();
    }

    // 转回 queue（已反转）
    while (!s.empty()) {
        q.push(s.top());
        s.pop();
    }

    return q;
}
```

---

## 总结

### 何时使用 std::queue

✅ **适合**：
- 需要 FIFO 语义
- 只需访问首尾元素
- 不需要遍历
- 需要常数时间操作

❌ **不适合**：
- 需要随机访问
- 需要在中间插入/删除
- 需要遍历所有元素

### 最佳实践

1. 使用 `emplace()` 代替 `push()` 提高性能
2. 访问前检查 `empty()`
3. 选择合适的底层容器
4. 利用移动语义避免拷贝
5. 理解 pop() 不返回值的设计

---

## 参考资料

- [C++ Reference - std::queue](https://en.cppreference.com/w/cpp/container/queue)
- [C++ Standard Library - Container Adapters](https://en.cppreference.com/w/cpp/container)
