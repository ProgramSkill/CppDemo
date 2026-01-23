# std::set 详细解析

## 概述

`std::set`是基于红黑树的关联容器，存储**唯一**的**有序**元素。

```cpp
#include <set>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 元素唯一 | 自动去重，重复插入无效 |
| 自动排序 | 按键值排序（默认升序） |
| 底层实现 | 红黑树（平衡BST） |
| 查找效率 | O(log n) |

## 与其他容器对比

| 特性 | std::set | std::unordered_set | std::vector |
|------|----------|-------------------|-------------|
| 元素唯一 | ✅ | ✅ | ❌ |
| 有序 | ✅ | ❌ | ❌ |
| 查找 | O(log n) | O(1)平均 | O(n) |
| 插入 | O(log n) | O(1)平均 | O(1)尾部 |

## 成员函数

### 插入操作
```cpp
set<int> s;

// 插入（返回pair<iterator, bool>）
auto result = s.insert(10);  // { iterator, success }
if (result.second) {
    cout << "插入成功";
}

// 重复插入无效
s.insert(10);
s.insert(10);  // 只有一个10
```

### 删除操作
```cpp
set<int> s = {1, 2, 3, 4, 5};

// 按值删除（删除所有该值，set中只有一个）
size_t count = s.erase(3);  // 返回删除数量

// 按位置删除
auto it = s.find(2);
if (it != s.end()) {
    s.erase(it);
}

// 范围删除
s.erase(s.begin(), s.end());  // 清空
```

### 查找操作
```cpp
set<int> s = {10, 20, 30, 40, 50};

// find - 查找元素
auto it = s.find(30);
if (it != s.end()) {
    cout << *it;  // 30
}

// count - 计数（0或1）
size_t n = s.count(30);  // 1
size_t m = s.count(100); // 0

// lower_bound/upper_bound（有序访问）
auto lb = s.lower_bound(25);  // 第一个>=25的元素
auto ub = s.upper_bound(30);  // 第一个>30的元素

// equal_range
auto range = s.equal_range(30);  // [lower_bound, upper_bound)
```

## 使用场景

```cpp
// 1. 去重并保持顺序
vector<int> vec = {3, 1, 4, 1, 5, 9, 2, 6};
set<int> s(vec.begin(), vec.end());  // {1, 2, 3, 4, 5, 6, 9}

// 2. 快速查找成员
if (s.find(5) != s.end()) {
    cout << "5在集合中";
}

// 3. 范围查询（利用有序性）
set<int> nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
auto range = nums.equal_range(5);  // [5, 6)
// 可找到[4, 7)区间的所有元素

// 4. 集合运算
set<int> a = {1, 2, 3};
set<int> b = {3, 4, 5};

set<int> result;
set_union(a.begin(), a.end(), b.begin(), b.end(),
           inserter(result, result.begin()));  // 并集
```

## 自定义比较器

```cpp
// 降序
set<int, greater<int>> s = {1, 2, 3, 4, 5};

// 自定义排序
auto cmp = [](int a, int b) { return abs(a) < abs(b); };
set<int, decltype(cmp)> s(cmp);
s.insert(-5);
s.insert(3);
s.insert(-3);  // 按绝对值排序
```

## 何时使用set

✅ **适合**：
- 需要元素唯一
- 需要保持有序
- 频繁查找
- 需要范围查询

❌ **不适合**：
- 不需要有序 → unordered_set更快
- 需要重复元素 → multiset
- 需要随机访问 → vector

## 参考文档
- [cppreference - std::set](https://en.cppreference.com/w/cpp/container/set)
