# Recursion (递归)

递归是一种函数调用自身来解决问题的编程技术。

## 核心概念

### 1. 递归三要素

```
1. Base Case (基本情况)    - 递归停止的条件
2. Recursive Step (递归步骤) - 将问题分解为更小的子问题
3. Progress (进展)         - 每次递归都要向基本情况靠近
```

### 2. 递归执行过程

```
递归调用分为两个阶段：
1. Descending Phase (下降阶段) - 函数不断调用自身，压入调用栈
2. Ascending Phase (上升阶段)  - 到达基本情况后，逐层返回结果
```

---

## 示例文件

### 基础递归

| 文件 | 类型 | 说明 |
|------|------|------|
| `factorial.cpp` | 单分支线性递归 | n! = n * (n-1)! |
| `fibonacci.cpp` | 双分支递归 | fib(n) = fib(n-1) + fib(n-2) |
| `binary_search.cpp` | 分治递归 | 每次缩小一半搜索范围 |
| `tree_traversal.cpp` | 树递归 | 前序、中序、后序、层序遍历 |

### 进阶递归

| 文件 | 类型 | 说明 |
|------|------|------|
| `memoization.cpp` | 记忆化递归 | 缓存结果避免重复计算 |
| `backtracking.cpp` | 回溯算法 | 全排列、组合问题 |
| `tail_recursion.cpp` | 尾递归 | 可被编译器优化的递归形式 |
| `recursion_to_iteration.cpp` | 递归转迭代 | 用栈模拟递归调用 |

---

## 递归 vs 迭代

| 特性 | 递归 | 迭代 |
|------|------|------|
| 代码简洁性 | 通常更简洁 | 可能更冗长 |
| 内存使用 | 使用调用栈，可能栈溢出 | 内存使用固定 |
| 性能 | 函数调用有开销 | 通常更快 |
| 适用场景 | 树、图、分治问题 | 简单循环问题 |

---

## 常见递归模式

### 1. 线性递归 (Linear Recursion)
```cpp
// 每次只调用自身一次
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

### 2. 二叉递归 (Binary Recursion)
```cpp
// 每次调用自身两次
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}
```

### 3. 尾递归 (Tail Recursion)
```cpp
// 递归调用是函数的最后一个操作
int factorial_tail(int n, int acc = 1) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);
}
```

### 4. 相互递归 (Mutual Recursion)
```cpp
// 两个函数互相调用
bool isEven(int n);
bool isOdd(int n) {
    if (n == 0) return false;
    return isEven(n - 1);
}
bool isEven(int n) {
    if (n == 0) return true;
    return isOdd(n - 1);
}
```

---

## 调试递归的技巧

1. **打印调用信息** - 输出每次递归的参数和返回值
2. **画递归树** - 可视化递归调用过程
3. **检查基本情况** - 确保递归能够终止
4. **检查进展** - 确保每次递归都在向基本情况靠近
