# std::stack 详细解析

## 概述

`std::stack`是LIFO（后进先出）的容器适配器。

```cpp
#include <stack>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| LIFO | 后进先出 |
| 默认底层 | deque |
| 无迭代器 | 不支持遍历 |
| 操作限制 | 只能访问顶部 |

## 成员函数

```cpp
stack<int> s;

// 压栈
s.push(1);       // 拷贝
s.emplace(2);    // 原位构造

// 访问栈顶
int top = s.top();  // 2

// 弹栈
s.pop();          // 删除栈顶

// 容量
s.empty();        // 是否为空
s.size();         // 栈大小
```

## 使用场景

```cpp
// 1. 函数调用栈模拟
stack<int> call_stack;
void func() {
    call_stack.push(__LINE__);
    // ...
    call_stack.pop();
}

// 2. 撤销操作
stack<Action> undo_stack;
void do_action(Action a) {
    a.execute();
    undo_stack.push(a);
}
void undo() {
    if (!undo_stack.empty()) {
        undo_stack.top().revert();
        undo_stack.pop();
    }
}

// 3. DFS遍历
stack<Node> dfs;
dfs.push(root);
while (!dfs.empty()) {
    Node node = dfs.top(); dfs.pop();
    visit(node);
    for (child : node.children) {
        dfs.push(child);
    }
}

// 4. 括号匹配
stack<char> brackets;
for (char c : expression) {
    if (c == '(') brackets.push(c);
    else if (c == ')') {
        if (brackets.empty()) return false;
        brackets.pop();
    }
}
return brackets.empty();
```

## 自定义底层容器

```cpp
// 使用list作为底层（频繁插入删除）
stack<int, list<int>> s;

// 使用vector作为底层（可能更快）
stack<int, vector<int>> s;
```

## 何时使用stack

✅ **适合**：
- LIFO语义
- 撤销/回溯操作
- DFS遍历
- 括号匹配

## 参考文档
- [cppreference - std::stack](https://en.cppreference.com/w/cpp/container/stack)
