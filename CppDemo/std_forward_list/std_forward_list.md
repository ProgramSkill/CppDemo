# std::forward_list 详细解析

## 概述

`std::forward_list`是C++11引入的单向链表容器，省去了后向指针，内存开销比`std::list`更小。

```cpp
#include <forward_list>
```

## 核心特性

| 特性 | std::list | std::forward_list |
|------|-----------|-------------------|
| 链表类型 | 双向 | 单向 |
| 每节点指针 | 2个（prev+next） | 1个（next） |
| 内存开销 | 较大 | **小50%** |
| 支持操作 | 完整 | 受限（无size()等） |

## 成员函数

### 元素访问
```cpp
forward_list<int> fl = {10, 20, 30};

fl.front();  // 10 - 只能访问首元素
// fl.back(); // ❌ 不支持
// fl[2];    // ❌ 不支持随机访问
```

### 修改操作
```cpp
forward_list<int> fl;

// 只能在头部操作
fl.push_front(1);
fl.pop_front();
// push_back/pope_back ❌ 不支持

// 位置操作（在指定位置之后）
auto it = fl.begin();
fl.insert_after(it, 99);     // 在it后插入
fl.emplace_after(it, 100);   // 在it后原位构造
fl.erase_after(it);          // 删除it后的元素
fl.erase_after(it, fl.end()); // 删除范围
```

### 特殊操作
```cpp
forward_list<int> fl = {1, 2, 3, 2, 4};

fl.remove(2);                // 删除所有2
fl.remove_if([](int n) { return n % 2 == 0; }); // 删除偶数
fl.unique();                 // 删除连续重复
fl.reverse();                // 反转
fl.sort();                   // 排序

// before_begin() - 返回首前迭代器（特殊）
auto it = fl.before_begin();  // 指向首元素之前
fl.insert_after(it, 0);      // 可在首元素前插入
```

## 时间复杂度

| 操作 | 复杂度 |
|------|--------|
| 访问首元素 | O(1) |
| 头部插入/删除 | O(1) |
| 任意位置插入/删除 | O(1) |
| 访问中间元素 | O(n) |
| size() | ❌ 无此函数 |

## 使用场景

```cpp
// 1. 哈希表桶实现（内存敏感场景）
forward_list<pair<int, string>> buckets[10];

void insert(int key, string value) {
    size_t index = hash(key) % 10;
    buckets[index].emplace_front(key, value);
}

// 2. 前端插入频繁的场景
forward_list<int> fl;
for (int i = 0; i < 1000; ++i) {
    fl.push_front(i);  // O(1)，比list更快
}
```

## 何时使用forward_list

✅ **适合**：
- 内存受限环境
- 只需单向遍历
- 主要在头部操作
- 实现哈希表桶

❌ **不适合**：
- 需要反向遍历 → 用list
- 需要size() → 自己维护或用list
- 需要在尾部操作 → 用list
- 需要快速获取大小 → 用list

## 参考文档
- [cppreference - std::forward_list](https://en.cppreference.com/w/cpp/container/forward_list)
