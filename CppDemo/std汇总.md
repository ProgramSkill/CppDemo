# C++ 标准库(std)组件汇总

本文档汇总了项目中涉及的所有C++标准库组件，按类别分类整理。

## 目录
- [序列容器](#序列容器)
- [关联容器](#关联容器)
- [无序容器](#无序容器)
- [容器适配器](#容器适配器)
- [工具类型](#工具类型)
- [视图类型](#视图类型)

---

## 序列容器

序列容器按照严格的线性顺序存储元素。

| 组件 | 头文件 | C++版本 | 描述 | 特点 | 适用场景 |
|------|--------|---------|------|------|----------|
| [**std::vector**](./std_vector/std_vector.md) | `<vector>` | C++98 | 动态数组 | 连续内存、动态扩容、随机访问O(1) | 需要频繁随机访问、尾部插入删除 |
| [**std::array**](./std_array/std_array.md) | `<array>` | C++11 | 固定大小数组 | 编译期确定大小、栈上分配、零开销 | 大小固定、需要数组语义 |
| [**std::deque**](./std_deque/std_deque.md) | `<deque>` | C++98 | 双端队列 | 两端插入删除O(1)、随机访问O(1) | 需要两端操作的场景 |
| [**std::list**](./std_list/std_list.md) | `<list>` | C++98 | 双向链表 | 任意位置插入删除O(1)、不支持随机访问 | 频繁中间插入删除、不需要随机访问 |
| [**std::forward_list**](./std_forward_list/std_forward_list.md) | `<forward_list>` | C++11 | 单向链表 | 内存占用更小、只能单向遍历 | 内存敏感、只需单向遍历 |

---

## 关联容器

关联容器实现了排序的数据结构，通常基于红黑树实现。

| 组件 | 头文件 | C++版本 | 描述 | 特点 | 适用场景 |
|------|--------|---------|------|------|----------|
| [**std::set**](./std_set/std_set.md) | `<set>` | C++98 | 有序集合 | 元素唯一、自动排序、查找O(log n) | 需要有序且唯一的元素集合 |
| [**std::multiset**](./std_multiset/std_multiset.md) | `<set>` | C++98 | 有序多重集合 | 允许重复元素、自动排序 | 需要有序且允许重复的元素 |
| [**std::map**](./std_map/std_map.md) | `<map>` | C++98 | 有序键值对 | 键唯一、按键排序、查找O(log n) | 需要有序的键值映射 |
| [**std::multimap**](./std_multimap/std_multimap.md) | `<map>` | C++98 | 有序多重键值对 | 允许键重复、按键排序 | 一个键对应多个值的场景 |

---

## 无序容器

无序容器基于哈希表实现，提供平均O(1)的查找性能。

| 组件 | 头文件 | C++版本 | 描述 | 特点 | 适用场景 |
|------|--------|---------|------|------|----------|
| [**std::unordered_set**](./std_unordered_set/std_unordered_set.md) | `<unordered_set>` | C++11 | 无序集合 | 元素唯一、哈希存储、平均查找O(1) | 不需要排序、追求查找性能 |
| [**std::unordered_multiset**](./std_unordered_multiset/std_unordered_multiset.md) | `<unordered_set>` | C++11 | 无序多重集合 | 允许重复、哈希存储 | 不需要排序、允许重复元素 |
| [**std::unordered_map**](./std_unordered_map/std_unordered_map.md) | `<unordered_map>` | C++11 | 无序键值对 | 键唯一、哈希存储、平均查找O(1) | 不需要排序的键值映射 |
| [**std::unordered_multimap**](./std_unordered_multimap/std_unordered_multimap.md) | `<unordered_map>` | C++11 | 无序多重键值对 | 允许键重复、哈希存储 | 一键多值且不需要排序 |

---

## 容器适配器

容器适配器提供了不同的接口来访问底层容器。

| 组件 | 头文件 | C++版本 | 描述 | 底层容器 | 适用场景 |
|------|--------|---------|------|----------|----------|
| [**std::stack**](./std_stack/std_stack.md) | `<stack>` | C++98 | 栈(LIFO) | deque/vector/list | 后进先出的数据结构 |
| [**std::queue**](./std_queue/std_queue.md) | `<queue>` | C++98 | 队列(FIFO) | deque/list | 先进先出的数据结构 |
| [**std::priority_queue**](./std_priority_queue/std_priority_queue.md) | `<queue>` | C++98 | 优先队列 | vector/deque | 需要按优先级处理元素 |

---

## 工具类型

工具类型提供了各种通用的辅助功能。

| 组件 | 头文件 | C++版本 | 描述 | 特点 | 适用场景 |
|------|--------|---------|------|------|----------|
| [**std::optional**](./std_optional/std_optional.md) | `<optional>` | C++17 | 可选值 | 可能有值或无值、避免空指针 | 表示可能不存在的值 |
| [**std::variant**](./std_variant/std_variant.md) | `<variant>` | C++17 | 类型安全的联合体 | 可存储多种类型之一、类型安全 | 需要存储多种可能类型的值 |
| [**std::any**](./std_any/std_any.md) | `<any>` | C++17 | 类型擦除容器 | 可存储任意类型、运行时类型检查 | 需要存储未知类型的值 |
| [**std::tuple**](./std_tuple/std_tuple.md) | `<tuple>` | C++11 | 元组 | 固定大小的异构容器 | 返回多个值、临时组合数据 |
| [**std::bitset**](./std_bitset/std_bitset.md) | `<bitset>` | C++98 | 位集合 | 固定大小的位序列、位操作 | 需要高效的位操作 |
| [**std::index_sequence**](./std_index_sequence/std_index_sequence.md) | `<utility>` | C++14 | 编译期整数序列 | 模板元编程、参数包展开 | 编译期计算、参数包操作 |

---

## 视图类型

视图类型提供了对现有数据的非拥有式访问。

| 组件 | 头文件 | C++版本 | 描述 | 特点 | 适用场景 |
|------|--------|---------|------|------|----------|
| [**std::span**](./std_span/std_span.md) | `<span>` | C++20 | 连续序列视图 | 非拥有、轻量级、统一接口 | 传递数组/容器的引用视图 |
| [**std::string_view**](./std_string_view/std_string_view.md) | `<string_view>` | C++17 | 字符串视图 | 非拥有、避免拷贝、只读 | 传递字符串而不拷贝 |

---

## 快速参考指南

### 按需求选择容器

| 需求 | 推荐组件 | 原因 |
|------|----------|------|
| 需要随机访问 | `std::vector`, `std::array`, `std::deque` | O(1)随机访问性能 |
| 频繁插入/删除(任意位置) | `std::list`, `std::forward_list` | O(1)插入删除 |
| 只在两端操作 | `std::deque`, `std::queue` | 两端O(1)操作 |
| 需要排序 | `std::set`, `std::map` | 自动维护有序 |
| 需要快速查找 | `std::unordered_set`, `std::unordered_map` | 平均O(1)查找 |
| 后进先出(LIFO) | `std::stack` | 栈语义 |
| 先进先出(FIFO) | `std::queue` | 队列语义 |
| 按优先级处理 | `std::priority_queue` | 自动排序 |
| 可能无值的情况 | `std::optional` | 类型安全的可选值 |
| 多种类型之一 | `std::variant` | 类型安全的联合体 |
| 传递数组不拷贝 | `std::span` | 非拥有视图 |
| 传递字符串不拷贝 | `std::string_view` | 字符串视图 |

### 性能对比表

| 容器 | 随机访问 | 头部插入 | 尾部插入 | 中间插入 | 查找 | 内存布局 |
|------|----------|----------|----------|----------|------|----------|
| **vector** | O(1) | O(n) | O(1)* | O(n) | O(n) | 连续 |
| **array** | O(1) | - | - | - | O(n) | 连续 |
| **deque** | O(1) | O(1) | O(1) | O(n) | O(n) | 分段连续 |
| **list** | O(n) | O(1) | O(1) | O(1) | O(n) | 非连续 |
| **forward_list** | O(n) | O(1) | O(n) | O(1) | O(n) | 非连续 |
| **set/map** | - | O(log n) | O(log n) | O(log n) | O(log n) | 树结构 |
| **unordered_set/map** | - | O(1)* | O(1)* | O(1)* | O(1)* | 哈希表 |

*注：平均情况下的时间复杂度，最坏情况可能不同

### C++版本总结

| C++版本 | 新增组件 |
|---------|----------|
| **C++98** | vector, array(C风格), deque, list, set, multiset, map, multimap, stack, queue, priority_queue, bitset |
| **C++11** | array, forward_list, unordered_set, unordered_multiset, unordered_map, unordered_multimap, tuple |
| **C++14** | index_sequence |
| **C++17** | optional, variant, any, string_view |
| **C++20** | span |

---

## 总结

本文档汇总了C++标准库中最常用的26个组件，涵盖了：
- **5个序列容器**：提供线性存储
- **4个关联容器**：提供有序的键值存储
- **4个无序容器**：提供基于哈希的快速查找
- **3个容器适配器**：提供特定的数据结构语义
- **6个工具类型**：提供类型安全和编译期计算
- **2个视图类型**：提供非拥有式的数据访问

选择合适的容器和工具类型可以显著提升代码的性能和可维护性。建议根据具体需求参考快速参考指南和性能对比表进行选择。

---

