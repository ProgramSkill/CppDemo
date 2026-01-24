# C++ 技术术语中英对照表

## 模板编程 (Template Programming)

| English Term | 中文 | Description |
|-------------|------|-------------|
| Non-Type Template Parameters | 非类型模板参数 | Template parameters other than types |
| Non-Type Template Arguments | 非类型模板实参 | Actual values passed to non-type template parameters |
| Template specialization | 模板特化 | Providing specific implementations for certain types or values |
| Template instantiation | 模板实例化 | Process where compiler generates concrete code from template |
| Compile-time | 编译期 | Determined during compilation rather than runtime |
| if constexpr | 编译期 if | C++17 feature for compile-time conditional branching |
| Parameter pack | 参数包 | C++11 feature for accepting arbitrary number of parameters |
| Fold expression | 折叠表达式 | C++17 feature for operations on parameter packs |
| SFINAE | 替换失败不是错误 | Substitution Failure Is Not An Error |
| Variadic template | 可变参数模板 | Template accepting variable number of arguments |

## 类与对象 (Classes and Objects)

| English Term | 中文 | Description |
|-------------|------|-------------|
| Static member | 静态成员 | Members shared among all instances of a class |
| Virtual function | 虚函数 | Function that can be overridden in derived classes |
| Pure virtual function | 纯虚函数 | Virtual function with no implementation (= 0) |
| Abstract class | 抽象类 | Class with at least one pure virtual function |
| Constructor | 构造函数 | Special function to initialize objects |
| Destructor | 析构函数 | Special function to clean up objects |
| Copy constructor | 拷贝构造函数 | Constructor that creates copy of object |
| Move constructor | 移动构造函数 | Constructor that transfers resources (C++11) |
| Rule of Three/Five | 三/五法则 | Guidelines for special member functions |

## 内存管理 (Memory Management)

| English Term | 中文 | Description |
|-------------|------|-------------|
| Smart pointer | 智能指针 | RAII wrapper for automatic memory management |
| unique_ptr | 独占指针 | Smart pointer with exclusive ownership |
| shared_ptr | 共享指针 | Smart pointer with shared ownership |
| weak_ptr | 弱引用指针 | Non-owning reference to shared_ptr |
| RAII | 资源获取即初始化 | Resource Acquisition Is Initialization |
| Memory leak | 内存泄漏 | Failure to deallocate memory |
| Dangling pointer | 悬空指针 | Pointer to deallocated memory |

## 容器与迭代器 (Containers and Iterators)

| English Term | 中文 | Description |
|-------------|------|-------------|
| Sequence container | 序列容器 | Container with linear ordering (vector, list, deque) |
| Associative container | 关联容器 | Container with key-based access (map, set) |
| Iterator | 迭代器 | Object for traversing containers |
| Random access iterator | 随机访问迭代器 | Iterator supporting arbitrary jumps |
| Bidirectional iterator | 双向迭代器 | Iterator supporting forward/backward movement |
| Forward iterator | 前向迭代器 | Iterator supporting forward movement only |

## 现代C++特性 (Modern C++ Features)

| English Term | 中文 | Description |
|-------------|------|-------------|
| Lambda expression | Lambda表达式 | Anonymous function object (C++11) |
| Move semantics | 移动语义 | Efficient resource transfer (C++11) |
| Perfect forwarding | 完美转发 | Preserving value category in forwarding |
| Type deduction | 类型推导 | Automatic type determination (auto, decltype) |
| Range-based for loop | 基于范围的for循环 | Simplified iteration syntax (C++11) |
| Structured binding | 结构化绑定 | Decomposing objects into variables (C++17) |
| std::optional | 可选值 | Container for optional values (C++17) |
| std::variant | 变体类型 | Type-safe union (C++17) |
| std::any | 任意类型 | Type-erased container (C++17) |

## 并发编程 (Concurrency)

| English Term | 中文 | Description |
|-------------|------|-------------|
| Thread | 线程 | Unit of execution |
| Mutex | 互斥量 | Mutual exclusion lock |
| Condition variable | 条件变量 | Synchronization primitive |
| Atomic operation | 原子操作 | Indivisible operation |
| Race condition | 竞态条件 | Concurrent access causing undefined behavior |
| Deadlock | 死锁 | Circular wait for resources |

---

## 使用说明

- 本表涵盖现代 C++ (C++11/14/17/20) 的核心术语
- 适用于学习和开发参考
- 建议结合实际代码示例理解概念