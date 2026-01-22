# Recursion (递归) - 从入门到精通

递归是一种函数调用自身来解决问题的编程技术。

---

## 目录

1. [什么是递归](#1-什么是递归)
2. [为什么需要递归](#2-为什么需要递归)
3. [递归的工作原理 - 调用栈](#3-递归的工作原理---调用栈)
4. [递归三要素](#4-递归三要素)
5. [如何思考递归问题](#5-如何思考递归问题)
6. [递归的类型](#6-递归的类型)
7. [递归 vs 迭代](#7-递归-vs-迭代)
8. [常见陷阱与错误](#8-常见陷阱与错误)
9. [递归优化技巧](#9-递归优化技巧)
10. [实际应用场景](#10-实际应用场景)
11. [示例文件](#11-示例文件)
12. [学习路线](#12-学习路线)

---

## 1. 什么是递归

### 定义

递归 = 函数直接或间接调用自身

```cpp
void recursion() {
    // ... 做一些事情
    recursion();  // 调用自身
}
```

### 生活中的递归

```
俄罗斯套娃：打开一个娃娃，里面还有一个更小的娃娃
镜子对镜子：两面镜子相对，产生无限反射
文件夹结构：文件夹里有文件夹，文件夹里还有文件夹
```

### 数学中的递归定义

```
阶乘：n! = n × (n-1)!，其中 0! = 1
斐波那契：F(n) = F(n-1) + F(n-2)，其中 F(0)=0, F(1)=1
```

---

## 2. 为什么需要递归

### 递归的优势

```
1. 代码简洁 - 复杂问题用几行代码就能解决
2. 自然表达 - 某些问题天然具有递归结构（树、图）
3. 分治思想 - 将大问题分解为小问题
```

### 适合递归的问题特征

```
✓ 问题可以分解为相同类型的子问题
✓ 子问题的规模比原问题小
✓ 存在最小规模的问题可以直接求解（基本情况）
✓ 子问题的解可以合并为原问题的解
```

### 经典递归问题

```
- 阶乘、斐波那契数列
- 二分查找
- 树的遍历（前序、中序、后序）
- 归并排序、快速排序
- 汉诺塔
- 迷宫求解
- 全排列、组合
```

---

## 3. 递归的工作原理 - 调用栈

### 什么是调用栈

每次函数调用都会在内存中创建一个"栈帧"(Stack Frame)，包含：
- 函数参数
- 局部变量
- 返回地址

```
┌─────────────────────────────────────┐
│            内存布局                  │
├─────────────────────────────────────┤
│  栈 (Stack) ↓                       │  <- 函数调用在这里
│  ...                                │
│  堆 (Heap) ↑                        │
│  全局/静态变量                       │
│  代码段                             │
└─────────────────────────────────────┘
```

### factorial(4) 的调用栈变化

```
调用阶段 (Descending Phase):
─────────────────────────────────────────────────────────────
Step 1:  │ factorial(4)        │  等待 factorial(3) 的结果
         └────────────────────┘
─────────────────────────────────────────────────────────────
Step 2:  │ factorial(3)        │  等待 factorial(2) 的结果
         │ factorial(4)        │  等待中...
         └────────────────────┘
─────────────────────────────────────────────────────────────
Step 3:  │ factorial(2)        │  等待 factorial(1) 的结果
         │ factorial(3)        │  等待中...
         │ factorial(4)        │  等待中...
         └────────────────────┘
─────────────────────────────────────────────────────────────
Step 4:  │ factorial(1)        │  基本情况！返回 1
         │ factorial(2)        │  等待中...
         │ factorial(3)        │  等待中...
         │ factorial(4)        │  等待中...
         └────────────────────┘

返回阶段 (Ascending Phase):
─────────────────────────────────────────────────────────────
Step 5:  │ factorial(2)        │  收到 1，计算 2×1=2，返回 2
         │ factorial(3)        │  等待中...
         │ factorial(4)        │  等待中...
         └────────────────────┘
─────────────────────────────────────────────────────────────
Step 6:  │ factorial(3)        │  收到 2，计算 3×2=6，返回 6
         │ factorial(4)        │  等待中...
         └────────────────────┘
─────────────────────────────────────────────────────────────
Step 7:  │ factorial(4)        │  收到 6，计算 4×6=24，返回 24
         └────────────────────┘
─────────────────────────────────────────────────────────────
最终结果: 24
```

### 栈溢出 (Stack Overflow)

```
当递归太深时，栈空间耗尽：

│ factorial(10000)     │
│ factorial(9999)      │
│ factorial(9998)      │
│ ...                  │  <- 栈空间有限！
│ factorial(2)         │
│ factorial(1)         │
└──────────────────────┘
        💥 Stack Overflow!

解决方案：
1. 确保有正确的基本情况
2. 使用尾递归（编译器可优化）
3. 改用迭代
4. 增加栈大小（不推荐）
```

---

## 4. 递归三要素

### 要素一：基本情况 (Base Case)

```cpp
// 递归必须有终止条件！
int factorial(int n) {
    if (n <= 1) return 1;  // ← 基本情况：n=0 或 n=1 时停止
    return n * factorial(n - 1);
}
```

**常见基本情况：**
```
- 数值：n == 0, n == 1, n < 0
- 数组：数组为空, 只有一个元素
- 链表：节点为 nullptr
- 树：节点为 nullptr
- 字符串：字符串为空
```

### 要素二：递归步骤 (Recursive Step)

```cpp
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);  // ← 递归步骤：调用自身
}
```

**递归步骤的关键：**
```
1. 将问题分解为更小的子问题
2. 子问题与原问题结构相同
3. 调用自身解决子问题
4. 合并子问题的解
```

### 要素三：进展 (Progress)

```cpp
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);  // ← n-1 比 n 更接近基本情况
}
```

**确保进展：**
```
✓ factorial(n) 调用 factorial(n-1)  → n 在减小
✓ binarySearch(left, right) 调用时 right-left 在减小
✓ 树遍历时节点在向叶子移动

✗ 错误示例：factorial(n) 调用 factorial(n)  → 无限循环！
```

---

## 5. 如何思考递归问题

### 递归思维三步法

```
Step 1: 定义函数功能
        "这个函数要做什么？" - 不要想怎么做，先想做什么

Step 2: 找到基本情况
        "最简单的情况是什么？可以直接给出答案"

Step 3: 找到递归关系
        "假设子问题已经解决，如何用它解决当前问题？"
```

### 示例：计算数组元素之和

```
Step 1: 定义函数功能
        sum(arr, n) = 返回数组前 n 个元素的和

Step 2: 找到基本情况
        sum(arr, 0) = 0  (没有元素，和为0)

Step 3: 找到递归关系
        sum(arr, n) = arr[n-1] + sum(arr, n-1)
        "前n个元素的和 = 第n个元素 + 前n-1个元素的和"
```

```cpp
int sum(int arr[], int n) {
    if (n == 0) return 0;                    // 基本情况
    return arr[n - 1] + sum(arr, n - 1);     // 递归关系
}
```

### 信任递归 (Leap of Faith)

```
关键心态：假设递归调用会正确工作！

不要试图在脑中展开所有递归调用，
只需要：
1. 确保基本情况正确
2. 确保递归关系正确
3. 确保每次调用都在向基本情况靠近

然后相信递归会正确工作。
```

---

## 6. 递归的类型

### 6.1 线性递归 (Linear Recursion)

每次只调用自身一次，形成线性调用链。

```cpp
// 阶乘
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

```
调用链：factorial(5) → factorial(4) → factorial(3) → factorial(2) → factorial(1)
时间复杂度：O(n)
空间复杂度：O(n) - 调用栈深度
```

### 6.2 二叉递归 (Binary Recursion)

每次调用自身两次，形成二叉树结构。

```cpp
// 斐波那契
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}
```

```
调用树：
                    fib(5)
                   /      \
              fib(4)        fib(3)
             /      \       /      \
        fib(3)    fib(2)  fib(2)   fib(1)
        /    \
    fib(2) fib(1)

时间复杂度：O(2^n) - 指数级！
空间复杂度：O(n) - 调用栈最大深度
```

### 6.3 尾递归 (Tail Recursion)

递归调用是函数的最后一个操作。

```cpp
// 普通递归 - 不是尾递归
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);  // 还要乘以 n
}

// 尾递归
int factorial_tail(int n, int acc = 1) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);  // 直接返回递归结果
}
```

```
尾递归优势：
- 编译器可以优化为循环（尾调用优化 TCO）
- 不会增加调用栈深度
- 避免栈溢出
```

### 6.4 相互递归 (Mutual Recursion)

两个或多个函数互相调用。

```cpp
bool isEven(int n);
bool isOdd(int n);

bool isEven(int n) {
    if (n == 0) return true;
    return isOdd(n - 1);
}

bool isOdd(int n) {
    if (n == 0) return false;
    return isEven(n - 1);
}
```

```
调用链：isEven(4) → isOdd(3) → isEven(2) → isOdd(1) → isEven(0) → true
```

### 6.5 嵌套递归 (Nested Recursion)

递归调用的参数本身也是递归调用。

```cpp
// Ackermann 函数
int ackermann(int m, int n) {
    if (m == 0) return n + 1;
    if (n == 0) return ackermann(m - 1, 1);
    return ackermann(m - 1, ackermann(m, n - 1));  // 嵌套递归
}
```

---

## 7. 递归 vs 迭代

### 对比表

| 特性 | 递归 | 迭代 |
|------|------|------|
| 代码简洁性 | 通常更简洁、更直观 | 可能更冗长 |
| 内存使用 | 使用调用栈，O(n) 空间 | 通常 O(1) 空间 |
| 性能 | 函数调用有开销 | 通常更快 |
| 栈溢出风险 | 深度递归可能溢出 | 无此风险 |
| 适用场景 | 树、图、分治、回溯 | 简单循环、线性处理 |
| 可读性 | 对递归结构问题更清晰 | 对简单循环更清晰 |

### 同一问题的两种实现

```cpp
// 递归版本
int factorial_recursive(int n) {
    if (n <= 1) return 1;
    return n * factorial_recursive(n - 1);
}

// 迭代版本
int factorial_iterative(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
```

### 何时选择递归

```
✓ 问题具有天然的递归结构（树、图）
✓ 使用分治策略
✓ 需要回溯（如全排列、迷宫）
✓ 代码可读性比性能更重要
```

### 何时选择迭代

```
✓ 简单的线性处理
✓ 性能关键的代码
✓ 递归深度可能很大
✓ 尾递归且编译器不支持 TCO
```

---

## 8. 常见陷阱与错误

### 陷阱 1：缺少基本情况

```cpp
// ❌ 错误：没有基本情况，无限递归
int factorial(int n) {
    return n * factorial(n - 1);
}

// ✓ 正确：有基本情况
int factorial(int n) {
    if (n <= 1) return 1;  // 基本情况
    return n * factorial(n - 1);
}
```

### 陷阱 2：基本情况不完整

```cpp
// ❌ 错误：没有处理负数
int factorial(int n) {
    if (n == 0) return 1;
    return n * factorial(n - 1);  // factorial(-1) 会无限递归
}

// ✓ 正确：处理所有边界情况
int factorial(int n) {
    if (n < 0) return -1;  // 错误处理
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
```

### 陷阱 3：没有向基本情况靠近

```cpp
// ❌ 错误：参数没有变化
int infinite(int n) {
    if (n == 0) return 0;
    return infinite(n);  // n 没有减小！
}

// ❌ 错误：参数在增大
int wrong(int n) {
    if (n == 0) return 0;
    return wrong(n + 1);  // n 在增大，永远到不了 0
}
```

### 陷阱 4：重复计算

```cpp
// ❌ 低效：大量重复计算
int fib(int n) {
    if (n <= 1) return n;
    return fib(n-1) + fib(n-2);  // fib(n-2) 会被计算多次
}

// ✓ 高效：使用记忆化
unordered_map<int, int> memo;
int fib_memo(int n) {
    if (n <= 1) return n;
    if (memo.count(n)) return memo[n];
    return memo[n] = fib_memo(n-1) + fib_memo(n-2);
}
```

### 陷阱 5：栈溢出

```cpp
// ❌ 危险：深度递归
int sum(int n) {
    if (n == 0) return 0;
    return n + sum(n - 1);  // sum(100000) 会栈溢出
}

// ✓ 安全：使用迭代或尾递归
int sum_iterative(int n) {
    int result = 0;
    for (int i = 1; i <= n; i++) result += i;
    return result;
}
```

---

## 9. 递归优化技巧

### 9.1 记忆化 (Memoization)

缓存已计算的结果，避免重复计算。

```cpp
unordered_map<int, long long> memo;

long long fib(int n) {
    if (n <= 1) return n;
    if (memo.count(n)) return memo[n];  // 查缓存
    return memo[n] = fib(n-1) + fib(n-2);  // 存缓存
}
```

```
优化效果：
- 时间复杂度：O(2^n) → O(n)
- 空间复杂度：O(n)（缓存） + O(n)（调用栈）
```

### 9.2 尾递归优化

将递归改写为尾递归形式。

```cpp
// 普通递归：需要保存中间状态
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// 尾递归：使用累加器
int factorial_tail(int n, int acc = 1) {
    if (n <= 1) return acc;
    return factorial_tail(n - 1, n * acc);
}
```

### 9.3 转换为迭代

用显式栈模拟递归调用。

```cpp
// 递归版本
void preorder(Node* node) {
    if (!node) return;
    cout << node->val;
    preorder(node->left);
    preorder(node->right);
}

// 迭代版本
void preorder_iterative(Node* root) {
    stack<Node*> s;
    s.push(root);
    while (!s.empty()) {
        Node* node = s.top(); s.pop();
        if (!node) continue;
        cout << node->val;
        s.push(node->right);  // 先压右，后压左
        s.push(node->left);
    }
}
```

### 9.4 减少递归深度

使用分治减少递归深度。

```cpp
// 线性递归：深度 O(n)
int sum(int arr[], int n) {
    if (n == 0) return 0;
    return arr[n-1] + sum(arr, n-1);
}

// 分治递归：深度 O(log n)
int sum_divide(int arr[], int left, int right) {
    if (left > right) return 0;
    if (left == right) return arr[left];
    int mid = (left + right) / 2;
    return sum_divide(arr, left, mid) + sum_divide(arr, mid+1, right);
}
```

---

## 10. 实际应用场景

### 10.1 数据结构操作

```
- 链表反转、合并
- 二叉树遍历、搜索、插入、删除
- 图的 DFS 遍历
- 堆操作
```

### 10.2 算法

```
- 分治算法：归并排序、快速排序、二分查找
- 回溯算法：N皇后、数独、全排列、子集
- 动态规划：自顶向下的记忆化搜索
```

### 10.3 实际问题

```
- 文件系统遍历（递归访问目录）
- JSON/XML 解析（递归结构）
- 编译器（递归下降解析）
- 数学计算（阶乘、斐波那契、幂运算）
- 游戏 AI（博弈树搜索）
```

---

## 11. 示例文件

### 基础递归

| 文件 | 类型 | 说明 |
|------|------|------|
| `01_factorial.cpp` | 单分支线性递归 | n! = n * (n-1)! |
| `02_fibonacci.cpp` | 双分支递归 | fib(n) = fib(n-1) + fib(n-2) |
| `03_binary_search.cpp` | 分治递归 | 每次缩小一半搜索范围 |
| `04_tree_traversal.cpp` | 树递归 | 前序、中序、后序、层序遍历 |

### 进阶递归

| 文件 | 类型 | 说明 |
|------|------|------|
| `05_memoization.cpp` | 记忆化递归 | 缓存结果避免重复计算 |
| `06_backtracking.cpp` | 回溯算法 | 全排列、组合问题 |
| `07_tail_recursion.cpp` | 尾递归 | 可被编译器优化的递归形式 |
| `08_recursion_to_iteration.cpp` | 递归转迭代 | 用栈模拟递归调用 |

---

## 12. 学习路线

```
入门阶段
├── 理解递归概念和调用栈
├── 01_factorial.cpp - 最简单的递归
└── 02_fibonacci.cpp - 理解双分支递归

基础阶段
├── 03_binary_search.cpp - 分治思想
├── 04_tree_traversal.cpp - 树的递归操作
└── 练习：链表反转、数组求和

进阶阶段
├── 05_memoization.cpp - 优化重复计算
├── 06_backtracking.cpp - 回溯算法
└── 练习：N皇后、子集、组合

精通阶段
├── 07_tail_recursion.cpp - 尾递归优化
├── 08_recursion_to_iteration.cpp - 递归转迭代
└── 练习：复杂 DP、图算法
```

---

## 调试递归的技巧

### 1. 打印调用信息

```cpp
int factorial(int n, int depth = 0) {
    string indent(depth * 2, ' ');
    cout << indent << "factorial(" << n << ") called" << endl;

    if (n <= 1) {
        cout << indent << "factorial(" << n << ") returns 1" << endl;
        return 1;
    }

    int result = n * factorial(n - 1, depth + 1);
    cout << indent << "factorial(" << n << ") returns " << result << endl;
    return result;
}
```

### 2. 画递归树

```
手动画出递归调用的树形结构，帮助理解执行过程。
```

### 3. 小规模测试

```
先用小数据测试：factorial(3) 而不是 factorial(100)
```

### 4. 验证三要素

```
□ 基本情况是否正确？
□ 递归步骤是否正确？
□ 是否在向基本情况靠近？
```

---

## 总结

```
递归的本质：
1. 将大问题分解为相同类型的小问题
2. 解决最小的问题（基本情况）
3. 合并小问题的解得到大问题的解

记住：
- 相信递归会正确工作（Leap of Faith）
- 确保三要素：基本情况、递归步骤、进展
- 注意优化：记忆化、尾递归、转迭代
```
