# std::stack 详细解析

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

`std::stack`是LIFO（后进先出）的**容器适配器**，提供受限的容器接口，只允许在顶部进行操作。

### 定义位置

```cpp
#include <stack>
```

### 模板声明

```cpp
template<class T, class Container = std::deque<T>>
class stack;
```

- **T**: 元素类型
- **Container**: 底层容器，默认为 `std::deque<T>`

### 为什么选择 std::stack？

```
┌──────────────────────────────────────────────┐
│         📦 std::stack 的优势                  │
├──────────────────────────────────────────────┤
│ ✅ LIFO语义：清晰表达后进先出的意图           │
│ ✅ 接口简洁：只暴露必要的操作                 │
│ ✅ 灵活底层：可选择不同的底层容器             │
│ ✅ 高效操作：O(1) 时间的push/pop              │
│ ✅ 类型安全：编译时检查                       │
└──────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | 说明 |
|------|------|
| LIFO | 后进先出，只能访问顶部 |
| 容器适配器 | 基于其他容器实现 |
| 默认底层 | std::deque |
| 无迭代器 | 不支持遍历 |
| 操作限制 | 只能push/pop/top |

---

## 成员函数详解

### 构造函数

| 函数 | 说明 |
|------|------|
| `stack()` | 默认构造，空栈 |
| `stack(const stack&)` | 拷贝构造 |
| `stack(stack&&)` | 移动构造 (C++11) |

```cpp
// 1. 默认构造
std::stack<int> s1;

// 2. 拷贝构造
std::stack<int> s2(s1);

// 3. 移动构造
std::stack<int> s3(std::move(s1));

// 4. 使用自定义底层容器
std::stack<int, std::vector<int>> s4;
std::stack<int, std::list<int>> s5;
```

### 修改操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `push(const T&)` | O(1)* | 压栈（拷贝） |
| `push(T&&)` | O(1)* | 压栈（移动） |
| `emplace(Args&&...)` | O(1)* | 原位构造压栈 |
| `pop()` | O(1)* | 弹栈 |
| `swap(stack&)` | O(1) | 交换内容 |

```cpp
std::stack<int> s;

// push - 压栈
s.push(1);       // 拷贝
s.push(2);

// emplace - 原位构造
s.emplace(3);

// pop - 弹栈
s.pop();         // 删除栈顶

// swap - 交换
std::stack<int> other;
s.swap(other);
```

### 查询操作

| 函数 | 复杂度 | 说明 |
|------|--------|------|
| `top()` | O(1) | 返回栈顶元素的引用 |
| `empty()` | O(1) | 检查是否为空 |
| `size()` | O(1) | 返回元素数量 |

```cpp
std::stack<int> s;
s.push(10);
s.push(20);
s.push(30);

// top - 访问栈顶
int x = s.top();  // 30

// empty - 检查是否为空
if (!s.empty()) {
    std::cout << "Stack is not empty" << std::endl;
}

// size - 获取大小
std::cout << "Size: " << s.size();  // 3
```

---

## 时间复杂度

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| push | **O(1)** | 摊销常数时间 |
| pop | **O(1)** | 常数时间 |
| top | **O(1)** | 常数时间 |
| empty | **O(1)** | 常数时间 |
| size | **O(1)** | 常数时间 |

---

## 底层容器选择

```cpp
// 默认：deque（两端操作高效）
std::stack<int> s1;

// vector（尾部操作高效）
std::stack<int, std::vector<int>> s2;

// list（任意位置操作高效）
std::stack<int, std::list<int>> s3;
```

| 底层容器 | 优点 | 缺点 |
|---------|------|------|
| deque | 两端操作高效 | 内存分段 |
| vector | 缓存友好 | 扩容时拷贝 |
| list | 任意位置操作 | 缓存不友好 |

---

## 使用场景

### 1. 撤销/重做操作

```cpp
std::stack<Action> undo_stack;
std::stack<Action> redo_stack;

void do_action(const Action& a) {
    a.execute();
    undo_stack.push(a);
    while (!redo_stack.empty()) redo_stack.pop();  // 清空redo
}

void undo() {
    if (!undo_stack.empty()) {
        Action a = undo_stack.top();
        undo_stack.pop();
        a.revert();
        redo_stack.push(a);
    }
}
```

### 2. DFS遍历

```cpp
std::stack<Node*> dfs;
dfs.push(root);
while (!dfs.empty()) {
    Node* node = dfs.top();
    dfs.pop();
    visit(node);
    for (Node* child : node->children) {
        dfs.push(child);
    }
}
```

### 3. 括号匹配

```cpp
bool is_balanced(const std::string& expr) {
    std::stack<char> s;
    for (char c : expr) {
        if (c == '(' || c == '[' || c == '{') {
            s.push(c);
        } else if (c == ')' || c == ']' || c == '}') {
            if (s.empty()) return false;
            char open = s.top();
            s.pop();
            if ((c == ')' && open != '(') ||
                (c == ']' && open != '[') ||
                (c == '}' && open != '{')) {
                return false;
            }
        }
    }
    return s.empty();
}
```

### 4. 表达式求值

```cpp
// 后缀表达式求值
int evaluate_postfix(const std::vector<std::string>& tokens) {
    std::stack<int> s;
    for (const auto& token : tokens) {
        if (token == "+" || token == "-" || token == "*" || token == "/") {
            int b = s.top(); s.pop();
            int a = s.top(); s.pop();
            if (token == "+") s.push(a + b);
            else if (token == "-") s.push(a - b);
            else if (token == "*") s.push(a * b);
            else if (token == "/") s.push(a / b);
        } else {
            s.push(std::stoi(token));
        }
    }
    return s.top();
}
```

---

## 注意事项

### 1. 访问空栈是未定义行为

```cpp
std::stack<int> s;

// ❌ 未定义行为
// int x = s.top();  // 栈为空

// ✅ 先检查
if (!s.empty()) {
    int x = s.top();
}
```

### 2. pop() 不返回值

```cpp
std::stack<int> s;
s.push(10);

// ❌ 错误：pop() 返回 void
// int x = s.pop();

// ✅ 正确：先top再pop
int x = s.top();
s.pop();
```

### 3. 底层容器的选择影响性能

```cpp
// deque - 平衡性能（默认）
std::stack<int> s1;

// vector - 缓存友好，但扩容时拷贝
std::stack<int, std::vector<int>> s2;

// list - 任意位置操作，但缓存不友好
std::stack<int, std::list<int>> s3;
```

---

## 常见问题

### Q1: stack 和 queue 的区别？

| 特性 | std::stack | std::queue |
|------|-----------|-----------|
| 顺序 | LIFO | FIFO |
| 操作 | push/pop/top | push/pop/front/back |
| 使用场景 | 撤销、DFS | 任务队列、BFS |

### Q2: 何时使用 stack？

✅ **适合**：
- LIFO 语义
- 撤销/重做操作
- DFS 遍历
- 括号匹配
- 表达式求值
- 函数调用栈模拟

❌ **不适合**：
- FIFO 语义 → 使用 queue
- 需要随机访问 → 使用 vector/deque
- 需要遍历 → 使用 vector/list

### Q3: 如何遍历 stack？

```cpp
std::stack<int> s;
s.push(1);
s.push(2);
s.push(3);

// ❌ stack 不支持迭代器
// for (auto it = s.begin(); it != s.end(); ++it) {}

// ✅ 方法1：逐个弹出（会修改栈）
while (!s.empty()) {
    std::cout << s.top() << " ";
    s.pop();
}

// ✅ 方法2：拷贝后遍历
std::stack<int> temp = s;
while (!temp.empty()) {
    std::cout << temp.top() << " ";
    temp.pop();
}
```

### Q4: stack 的内存开销？

```cpp
std::stack<int> s;
// 内存开销 = 底层容器的开销
// 默认 deque：通常比 vector 多一些指针开销
// 但对于大多数应用来说可以忽略
```

---

## 总结

### 最佳实践

1. **总是检查 empty()** 在调用 top() 或 pop() 前
2. **使用 emplace()** 而非 push() 以避免临时对象
3. **选择合适的底层容器** 根据使用场景
4. **记住 pop() 不返回值** 需要先 top() 再 pop()

---

## 参考文档
- [cppreference - std::stack](https://en.cppreference.com/w/cpp/container/stack)
- [cppreference - std::queue](https://en.cppreference.com/w/cpp/container/queue)
