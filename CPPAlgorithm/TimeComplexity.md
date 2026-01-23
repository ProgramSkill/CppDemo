# 时间复杂度 (Time Complexity) - 算法效率的度量

时间复杂度是衡量算法效率的重要指标，它描述了算法运行时间随输入规模增长的变化趋势。

---

## 目录

1. [什么是时间复杂度](#1-什么是时间复杂度)
2. [大O表示法](#2-大o表示法)
3. [常见复杂度类型](#3-常见复杂度类型)
4. [如何分析时间复杂度](#4-如何分析时间复杂度)
5. [复杂度对比表格](#5-复杂度对比表格)
6. [实际案例分析](#6-实际案例分析)
7. [空间复杂度简介](#7-空间复杂度简介)
8. [最佳实践](#8-最佳实践)
9. [常见误区](#9-常见误区)
10. [练习题](#10-练习题)

---

## 1. 什么是时间复杂度

### 定义

**时间复杂度** = 算法执行时间与输入规模之间的数学关系

```
不是计算具体的运行时间（秒、毫秒）
而是计算运算次数随数据量增长的"趋势"
```

### 为什么需要时间复杂度

```cpp
// 同一个算法在不同电脑上运行时间不同
int sum(int n) {
    int total = 0;
    for (int i = 1; i <= n; i++) {
        total += i;
    }
    return total;
}

// 在超级计算机上运行：n=1000000 耗时 0.001秒
// 在旧电脑上运行：       n=1000000 耗时 0.1秒

// 但无论什么电脑：
// n = 1000    → 运行 1000 次循环
// n = 1000000 → 运行 1000000 次循环
// 时间增长趋势相同 → 都是 O(n)
```

### 核心思想

```
关注点：
✓ 运算次数如何随 n 增长
✓ 增长趋势和数量级

忽略点：
✗ 具体的秒数
✗ 硬件差异
✗ 编程语言差异
✗ 常数系数（2n 和 100n 都是 O(n)）
```

---

## 2. 大O表示法

### 符号含义

```
O(·) = Upper Bound（上界）

O(f(n)) 表示：
"在最坏情况下，运行时间不会超过 f(n) 的某个常数倍"
```

### 数学定义

```
T(n) = O(f(n))

如果存在正常数 c 和 n₀，使得对所有 n ≥ n₀：
    T(n) ≤ c × f(n)

则称 T(n) = O(f(n))
```

### 实际理解

```cpp
// 代码分析
for (int i = 0; i < n; i++) {
    // 循环体执行 n 次
    cout << i;
}
// 运算次数：T(n) = n
// 时间复杂度：O(n)

for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        // 内层循环执行 n×n 次
        cout << i << j;
    }
}
// 运算次数：T(n) = n²
// 时间复杂度：O(n²)
```

### 大O的计算规则

#### 规则 1：忽略常数系数

```cpp
// 这些都是 O(n)
T(n) = 3n      → O(n)
T(n) = 100n    → O(n)
T(n) = 0.5n    → O(n)

// 为什么？
// 当 n 很大时，3n 和 100n 的增长趋势相同
```

#### 规则 2：只保留最高阶项

```cpp
T(n) = n² + 3n + 100  → O(n²)
// 因为 n² 增长得最快，主导了整体复杂度

T(n) = n³ + 1000n²    → O(n³)
// n³ 比 n² 增长快得多

// 示例
for (int i = 0; i < n; i++) {          // O(n)
    for (int j = 0; j < n; j++) {      // O(n²)
        cout << i << j;                // O(n²)
    }
}
for (int k = 0; k < 1000; k++) {       // O(1) - 常数
    cout << k;
}
// 总复杂度：O(n²) + O(1) = O(n²)
```

#### 规则 3：加法法则（顺序执行）

```cpp
// 两个顺序执行的循环
for (int i = 0; i < n; i++) {      // O(n)
    cout << i;
}
for (int j = 0; j < n; j++) {      // O(n)
    cout << j;
}

// 总复杂度：O(n) + O(n) = O(2n) = O(n)
```

#### 规则 4：乘法法则（嵌套循环）

```cpp
// 嵌套循环
for (int i = 0; i < n; i++) {          // 外层：n 次
    for (int j = 0; j < n; j++) {      // 内层：n 次
        cout << i << j;                // 总计：n × n = n² 次
    }
}

// 总复杂度：O(n) × O(n) = O(n²)
```

---

## 3. 常见复杂度类型

### 按性能排序（从快到慢）

```
O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!)
```

### 详细解析

#### O(1) - 常数时间

```cpp
// 数组随机访问
int getElement(int arr[], int index) {
    return arr[index];  // ✅ O(1) - 一步到位
}

// queue 操作
std::queue<int> q;
q.push(10);     // ✅ O(1)
q.pop();        // ✅ O(1)
q.front();      // ✅ O(1)

// 哈希表查找（平均情况）
unordered_map<int, string> map;
map[5] = "hello";   // ✅ O(1)
string s = map[5];  // ✅ O(1)
```

**特点**：无论数据量多大，执行时间不变

#### O(log n) - 对数时间

```cpp
// 二分查找
int binarySearch(int arr[], int n, int target) {
    int left = 0, right = n - 1;

    while (left <= right) {
        int mid = (left + right) / 2;

        if (arr[mid] == target)
            return mid;

        if (arr[mid] < target)
            left = mid + 1;    // 搜索范围缩小一半
        else
            right = mid - 1;   // 搜索范围缩小一半
    }

    return -1;
}
```

**为什么是 log n？**

```
n = 1000    → 最多查找 10 次     (2¹⁰ = 1024)
n = 1000000 → 最多查找 20 次     (2²⁰ ≈ 1000000)
n = 10⁹     → 最多查找 30 次     (2³⁰ ≈ 10⁹)

公式：查找次数 = log₂n
```

**特点**：每一步将问题规模缩小一半

#### O(n) - 线性时间

```cpp
// 线性搜索
int linearSearch(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {  // ✅ O(n) - 遍历所有元素
        if (arr[i] == target)
            return i;
    }
    return -1;
}

// 数组求和
int sum(int arr[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {  // ✅ O(n)
        total += arr[i];
    }
    return total;
}

// 遍历链表
while (head != nullptr) {  // ✅ O(n)
    cout << head->val;
    head = head->next;
}
```

**特点**：需要遍历所有数据一次

#### O(n log n) - 线性对数时间

```cpp
// 归并排序
void mergeSort(int arr[], int left, int right) {
    if (left >= right) return;

    int mid = (left + right) / 2;

    mergeSort(arr, left, mid);      // 递归：log n 层
    mergeSort(arr, mid + 1, right);

    merge(arr, left, mid, right);   // 合并：每层 O(n)
}

// 总复杂度：O(log n) × O(n) = O(n log n)
```

**常见场景**：
- 高效排序算法（归并排序、快速排序、堆排序）
- 某些树操作

#### O(n²) - 平方时间

```cpp
// 冒泡排序
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n - 1 - i; j++) {
        if (arr[j] > arr[j + 1])
            swap(arr[j], arr[j + 1]);
    }
}

// 查找数组中所有重复元素
for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
        if (arr[i] == arr[j])
            cout << "Duplicate: " << arr[i];
    }
}
```

**特点**：嵌套循环，两层都是 O(n)

#### O(2ⁿ) - 指数时间

```cpp
// 递归斐波那契（未优化）
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);  // ✅ O(2ⁿ) - 每次调用产生两次新调用
}

// 调用树：
//                    fib(5)
//                   /      \
//              fib(4)        fib(3)
//             /      \       /      \
//        fib(3)    fib(2)  fib(2)   fib(1)
//        /    \
//    fib(2) fib(1)
```

**为什么这么慢？**

```
n = 10  → 1024 次调用
n = 20  → 1,048,576 次调用
n = 30  → 1,073,741,824 次调用（超过10亿次！）
```

**特点**：每次递归调用次数翻倍，增长极快

#### O(n!) - 阶乘时间

```cpp
// 生成全排列
void permutations(string str, int l, int r) {
    if (l == r) {
        cout << str << endl;
        return;
    }

    for (int i = l; i <= r; i++) {
        swap(str[l], str[i]);
        permutations(str, l + 1, r);  // ✅ O(n!)
        swap(str[l], str[i]);  // 回溯
    }
}
```

**为什么最慢？**

```
n = 5  → 120 种排列
n = 10 → 3,628,800 种排列
n = 15 → 1,307,674,368,000 种排列（超过万亿！）
```

**特点**：通常出现在暴力穷举问题

---

## 4. 如何分析时间复杂度

### 分析步骤

```
Step 1: 识别基本操作
        什么操作被重复执行？（赋值、比较、循环等）

Step 2: 计算操作次数
        随输入规模 n，操作执行多少次？

Step 3: 应用大O规则
        忽略常数、保留最高阶项
```

### 实例分析

#### 示例 1：简单循环

```cpp
void printNumbers(int n) {
    for (int i = 0; i < n; i++) {
        cout << i << endl;  // 基本操作：执行 n 次
    }
}

// 分析：
// 循环执行 n 次
// 每次执行 1 次输出
// 总操作：n × 1 = n
// 复杂度：O(n)
```

#### 示例 2：嵌套循环

```cpp
void printPairs(int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << i << "," << j << endl;  // 基本操作
        }
    }
}

// 分析：
// 外层循环：n 次
// 内层循环：n 次
// 总操作：n × n = n²
// 复杂度：O(n²)
```

#### 示例 3：条件循环

```cpp
void printHalf(int n) {
    for (int i = 0; i < n; i += 2) {  // 每次增加 2
        cout << i << endl;
    }
}

// 分析：
// 循环执行 n/2 次
// 总操作：n/2
// 复杂度：O(n/2) = O(n)  (忽略常数)
```

#### 示例 4：对数循环

```cpp
void powersOfTwo(int n) {
    for (int i = 1; i < n; i *= 2) {  // 每次翻倍
        cout << i << endl;
    }
}

// 分析：
// i 的值：1, 2, 4, 8, 16, ..., 2^k < n
// 循环次数：k = log₂n
// 复杂度：O(log n)
```

#### 示例 5：复杂嵌套

```cpp
void complexFunction(int n) {
    for (int i = 0; i < n; i++) {           // O(n)
        for (int j = 0; j < n; j++) {       // O(n²)
            cout << i << j;
        }
    }

    for (int k = 0; k < n; k++) {           // O(n)
        cout << k;
    }
}

// 分析：
// 第一部分：O(n²)
// 第二部分：O(n)
// 总复杂度：O(n²) + O(n) = O(n²)  (保留最高阶)
```

#### 示例 6：递归分析

```cpp
int factorial(int n) {
    if (n <= 1) return 1;                // O(1) - 基本情况
    return n * factorial(n - 1);         // 递归调用
}

// 分析：
// 递归深度：n 层
// 每层操作：O(1)
// 总复杂度：n × O(1) = O(n)

// 递推公式：
// T(n) = T(n-1) + O(1)
// T(n) = O(n)
```

#### 示例 7：二分递归

```cpp
int fib(int n) {
    if (n <= 1) return n;                // O(1)
    return fib(n-1) + fib(n-2);          // 两次递归调用
}

// 分析：
// 每次调用产生 2 个新调用
// 调用树是满二叉树
// 总复杂度：O(2ⁿ)

// 递推公式：
// T(n) = T(n-1) + T(n-2) + O(1)
// T(n) = O(2ⁿ)
```

---

## 5. 复杂度对比表格

### 增长速度对比

假设每次操作 1 纳秒：

| n | O(1) | O(log n) | O(n) | O(n log n) | O(n²) | O(2ⁿ) |
|---|------|----------|------|------------|-------|-------|
| 10 | 1 ns | 3 ns | 10 ns | 33 ns | 100 ns | 1,024 ns |
| 100 | 1 ns | 7 ns | 100 ns | 664 ns | 10 μs | 10¹⁸ 年 |
| 1,000 | 1 ns | 10 ns | 1 μs | 10 μs | 1 ms | 宇宙毁灭 |
| 10,000 | 1 ns | 13 ns | 10 μs | 130 μs | 100 ms | 不可能 |
| 100,000 | 1 ns | 17 ns | 100 μs | 1.7 ms | 10 秒 | 不可能 |
| 1,000,000 | 1 ns | 20 ns | 1 ms | 20 ms | 16.7 分钟 | 不可能 |

**注**：μs = 微秒，ms = 毫秒

### 性能分级

| 复杂度 | 性能评价 | 实际应用 |
|--------|---------|---------|
| O(1) | ⭐⭐⭐⭐⭐ 完美 | 数组访问、哈希表、栈/队列操作 |
| O(log n) | ⭐⭐⭐⭐⭐ 优秀 | 二分查找、平衡树操作 |
| O(n) | ⭐⭐⭐⭐ 很好 | 线性搜索、简单遍历 |
| O(n log n) | ⭐⭐⭐ 良好 | 高效排序算法 |
| O(n²) | ⭐⭐ 一般 | 简单排序、小数据集 |
| O(2ⁿ) | ⭐ 差 | 需要优化（如记忆化） |
| O(n!) | ⭐ 极差 | 仅用于极小数据集 |

### 数据规模参考

```
O(1), O(log n), O(n), O(n log n)    → 可处理 10⁸ 以上数据
O(n²)                                → 可处理 10⁴ 左右数据
O(n³)                                → 可处理 10² 左右数据
O(2ⁿ), O(n!)                         → 仅能处理 n ≤ 30
```

---

## 6. 实际案例分析

### 案例 1：两数之和

```cpp
// 方法 1：暴力搜索 - O(n²)
vector<int> twoSum_brute(vector<int>& nums, int target) {
    for (int i = 0; i < nums.size(); i++) {
        for (int j = i + 1; j < nums.size(); j++) {
            if (nums[i] + nums[j] == target)
                return {i, j};
        }
    }
    return {};
}

// 方法 2：哈希表 - O(n)
vector<int> twoSum_hash(vector<int>& nums, int target) {
    unordered_map<int, int> map;
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (map.count(complement))
            return {map[complement], i};
        map[nums[i]] = i;
    }
    return {};
}

// 性能对比：
// n = 10,000
// 方法1：100,000,000 次操作
// 方法2：10,000 次操作
// 方法2快 10,000 倍！
```

### 案例 2：斐波那契数列

```cpp
// 方法 1：朴素递归 - O(2ⁿ)
int fib_recursive(int n) {
    if (n <= 1) return n;
    return fib_recursive(n - 1) + fib_recursive(n - 2);
}

// 方法 2：记忆化 - O(n)
unordered_map<int, int> memo;
int fib_memo(int n) {
    if (n <= 1) return n;
    if (memo.count(n)) return memo[n];
    return memo[n] = fib_memo(n - 1) + fib_memo(n - 2);
}

// 方法 3：迭代 - O(n)
int fib_iterative(int n) {
    if (n <= 1) return n;
    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// 性能对比（n = 40）：
// 方法1：超过 1 秒
// 方法2：< 1 毫秒
// 方法3：< 1 微秒
```

### 案例 3：排序算法对比

```cpp
// 冒泡排序 - O(n²)
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// 归并排序 - O(n log n)
void mergeSort(int arr[], int left, int right) {
    if (left >= right) return;
    int mid = (left + right) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// 性能对比：
// n = 100,000
// 冒泡排序：~100 亿次操作 → 几秒到几分钟
// 归并排序：~170 万次操作 → 几毫秒
// 相差约 1000 倍！
```

### 案例 4：查找算法

```cpp
// 线性搜索 - O(n)
int linearSearch(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

// 二分查找 - O(log n)
int binarySearch(int arr[], int n, int target) {
    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// 性能对比：
// n = 1,000,000
// 线性搜索（平均）：500,000 次比较
// 二分查找：20 次比较
// 二分查找快 25,000 倍！
```

---

## 7. 空间复杂度简介

### 定义

**空间复杂度** = 算法运行时占用的内存空间随输入规模增长的变化趋势

### 常见空间复杂度

```cpp
// O(1) - 常数空间
int sum(int arr[], int n) {
    int total = 0;  // 只用 1 个变量
    for (int i = 0; i < n; i++) {
        total += arr[i];
    }
    return total;
}

// O(n) - 线性空间
int* copyArray(int arr[], int n) {
    int* copy = new int[n];  // 分配 n 个空间
    for (int i = 0; i < n; i++) {
        copy[i] = arr[i];
    }
    return copy;
}

// O(n) - 递归栈空间
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);  // 递归深度：n
}

// O(n) - 哈希表
vector<int> findDuplicates(vector<int>& nums) {
    unordered_map<int, int> count;  // 最多存储 n 个元素
    vector<int> result;
    for (int num : nums) {
        if (++count[num] == 2)
            result.push_back(num);
    }
    return result;
}
```

### 空间 vs 时间权衡

```cpp
// 方法 1：时间优先 - O(n) 时间, O(1) 空间
int fib_iterative(int n) {
    if (n <= 1) return n;
    int prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        int curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }
    return prev1;
}

// 方法 2：空间优先 - O(n) 时间, O(n) 空间（便于查询）
unordered_map<int, int> fibCache;
int fib_memo(int n) {
    if (n <= 1) return n;
    if (fibCache.count(n)) return fibCache[n];
    return fibCache[n] = fib_memo(n - 1) + fib_memo(n - 2);
}
```

---

## 8. 最佳实践

### 优化策略

#### 1. 选择合适的算法

```cpp
// ❌ 对已排序数组用线性搜索
int index = linearSearch(sortedArray, n, target);  // O(n)

// ✅ 对已排序数组用二分查找
int index = binarySearch(sortedArray, n, target);  // O(log n)
```

#### 2. 使用哈希表加速查找

```cpp
// ❌ O(n²) - 每次都遍历
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        if (data[j].id == queryId) ...
    }
}

// ✅ O(n) - 预处理建立哈希表
unordered_map<int, Data> dataMap;
for (int i = 0; i < n; i++) {
    dataMap[data[i].id] = data[i];
}
// 查找：O(1)
Data result = dataMap[queryId];
```

#### 3. 避免嵌套循环

```cpp
// ❌ O(n²) - 嵌套循环
for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
        if (arr[i] + arr[j] == target) ...
    }
}

// ✅ O(n) - 双指针
sort(arr, arr + n);  // O(n log n)
int left = 0, right = n - 1;
while (left < right) {
    int sum = arr[left] + arr[right];
    if (sum == target) ...
    else if (sum < target) left++;
    else right--;
}
```

#### 4. 记忆化递归

```cpp
// ❌ O(2ⁿ) - 重复计算
int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

// ✅ O(n) - 记忆化
unordered_map<int, int> memo;
int fib(int n) {
    if (n <= 1) return n;
    if (memo.count(n)) return memo[n];
    return memo[n] = fib(n - 1) + fib(n - 2);
}
```

### 何时优化

```
不需要优化：
✓ 数据规模小（n < 100）
✓ 代码已经够快（< 1 秒）
✓ 过早优化是万恶之源

需要优化：
✓ 处理大数据集（n > 100,000）
✓ 实时性要求高
✓ 成为性能瓶颈
✓ 复杂度 > O(n²)
```

---

## 9. 常见误区

### 误区 1：混淆最坏、平均、最好情况

```cpp
// 快速排序的复杂度
// 最好情况（每次都均匀分割）：O(n log n)
// 平均情况：O(n log n)
// 最坏情况（已排序数组）：O(n²)

// 大O通常指最坏情况
// 也可以用 θ 表示平均情况
```

### 误区 2：忽略常数因子

```cpp
// 虽然都是 O(n)，但实际性能可能有差异
// 方法 1：遍历 1 次
int sum1 = accumulate(arr, arr + n, 0);  // 实际更快

// 方法 2：遍历 10 次
for (int k = 0; k < 10; k++) {
    for (int i = 0; i < n; i++) {
        total += arr[i];
    }
}

// 都是 O(n)，但方法1快 10 倍
```

### 误区 3：误认为递归总是 O(log n)

```cpp
// ❌ 错误理解
// "递归是 log n" → 不一定！

// 正确分析：
// factorial(n) → O(n)  (线性递归)
// binarySearch() → O(log n)  (每次问题减半)
// fib(n) → O(2ⁿ)  (指数递归！)
```

### 误区 4：忽略空间复杂度

```cpp
// 虽然时间复杂度是 O(n)，但空间可能是 O(n)
void reverse(vector<int>& nums) {
    vector<int> copy(nums.rbegin(), nums.rend());  // O(n) 空间
    nums = copy;
}

// 优化：O(1) 空间
void reverse_inplace(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    while (left < right) {
        swap(nums[left++], nums[right--]);
    }
}
```

---

## 10. 练习题

### 基础题

分析以下代码的时间复杂度：

```cpp
// 题 1
for (int i = 0; i < n; i++) {
    cout << i;
}
// 答案：O(n)

// 题 2
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
        cout << i << j;
    }
}
// 答案：O(n²)

// 题 3
for (int i = 0; i < n; i *= 2) {
    cout << i;
}
// 答案：O(log n)

// 题 4
for (int i = 0; i < n; i++) {
    for (int j = 0; j < 10; j++) {
        cout << i << j;
    }
}
// 答案：O(n) - 内层循环是常数

// 题 5
int i = 1;
while (i < n) {
    i *= 3;
}
// 答案：O(log n) - 以 3 为底
```

### 进阶题

```cpp
// 题 6
for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
        cout << i << j;
    }
}
// 答案：O(n²)
// 分析：1 + 2 + 3 + ... + n = n(n+1)/2

// 题 7
for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
        cout << i << j;
    }
}
// 答案：O(n²)
// 分析：0 + 1 + 2 + ... + (n-1) = n(n-1)/2

// 题 8
void recursive(int n) {
    if (n <= 1) return;
    for (int i = 0; i < n; i++) {
        cout << i;
    }
    recursive(n / 2);
}
// 答案：O(n)
// 分析：n + n/2 + n/4 + ... = 2n = O(n)

// 题 9
void recursive2(int n) {
    if (n <= 1) return;
    recursive2(n / 2);
    recursive2(n / 2);
}
// 答案：O(n)
// 分析：类似归并树的节点数

// 题 10
for (int i = 0; i < n; i++) {
    for (int j = 1; j < n; j *= 2) {
        cout << i << j;
    }
}
// 答案：O(n log n)
// 分析：外层 n 次，内层 log n 次
```

### 挑战题

```cpp
// 题 11：分析复杂度
void strange(int n) {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j += i) {
            cout << i << j;
        }
    }
}
// 答案：O(n log n)
// 分析：n/1 + n/2 + n/3 + ... + n/n = n × (1 + 1/2 + 1/3 + ... + 1/n)
//       = n × H(n) = n × log n = O(n log n)

// 题 12：递归复杂度
int func(int n) {
    if (n <= 1) return 1;
    return func(n/2) + func(n/2);
}
// 答案：O(n)
// 分析：类似完全二叉树，每层都是 O(n)，共 log n 层
//       但实际上重复计算了，总节点数是 2n-1
```

---

## 总结

### 关键要点

```
1. 时间复杂度关注趋势，不是绝对时间
2. 大O表示最坏情况的上界
3. 常见复杂度：O(1) < O(log n) < O(n) < O(n²) < O(2ⁿ)
4. 嵌套循环用乘法，顺序执行用加法
5. 选择算法时优先考虑更低的复杂度
```

### 分析技巧

```
✓ 数循环层数
✓ 识别递归深度和分支数
✓ 应用大O简化规则
✓ 画递归树辅助分析
✓ 记住常见算法的复杂度
```

### 实用建议

```
✓ 优先选择 O(n log n) 或更好的算法
✓ 小数据集（n < 100）简单算法即可
✓ 大数据集必须优化算法复杂度
✓ 用哈希表加速查找
✓ 避免嵌套循环
✓ 记忆化优化递归
```

---

## 参考资料

- 《算法导论》(Introduction to Algorithms) - 第3章
- 《算法》(Algorithms) - Sedgewick
- [Big O Cheat Sheet](https://www.bigocheatsheet.com/)
- [VisuAlgo - 算法可视化](https://visualgo.net/)
