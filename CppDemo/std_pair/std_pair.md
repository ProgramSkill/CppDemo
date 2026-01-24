# std::pair - C++ 标准库配对容器

## 概述

`std::pair` 是 C++ 标准库提供的一个模板类，用于将两个可能不同类型的值组合成一个单元。它定义在 `<utility>` 头文件中，是 C++98 标准的一部分。

### 基本特性

| 特性 | 说明 |
|------|------|
| **头文件** | `<utility>` |
| **命名空间** | `std` |
| **C++版本** | C++98 |
| **类型** | 模板类 `template<class T1, class T2> struct pair` |
| **成员** | `first` (第一个元素), `second` (第二个元素) |
| **用途** | 存储两个值的组合、函数返回多个值、作为关联容器的元素类型 |

### 为什么使用 std::pair？

1. **函数返回多个值**：在 C++17 之前，pair 是返回两个值的常用方式
2. **关联容器的元素**：`std::map` 和 `std::multimap` 的元素类型是 `pair<const Key, Value>`
3. **简单的数据组合**：无需定义专门的结构体即可组合两个值
4. **算法返回值**：许多 STL 算法返回 pair，如 `std::minmax_element`

---

## 基本语法

### 模板定义

```cpp
template<class T1, class T2>
struct pair {
    T1 first;   // 第一个元素
    T2 second;  // 第二个元素
};
```

---

## 创建和初始化

### 1. 构造函数方式

```cpp
// 默认构造
std::pair<int, std::string> p1;

// 值初始化
std::pair<int, std::string> p2(1, "hello");

// 拷贝构造
std::pair<int, std::string> p3(p2);

// 列表初始化 (C++11)
std::pair<int, std::string> p4{42, "world"};
```

### 2. make_pair 辅助函数

`std::make_pair` 可以自动推导类型，使代码更简洁：

```cpp
// 自动类型推导
auto p1 = std::make_pair(42, 3.14);        // pair<int, double>
auto p2 = std::make_pair("hello", 100);    // pair<const char*, int>
auto p3 = std::make_pair(std::string("hi"), 'A');  // pair<string, char>
```

### 3. 统一初始化 (C++11)

```cpp
// 使用花括号初始化
std::pair<int, double> p1 = {1, 2.5};
std::pair<std::string, int> p2 = {"Alice", 30};

// 函数返回值
return {true, 42};  // 返回 pair<bool, int>
```

### 创建方式对比

| 方式 | 优点 | 缺点 | 推荐场景 |
|------|------|------|----------|
| **构造函数** | 类型明确 | 需要显式指定类型 | 类型复杂或需要明确类型时 |
| **make_pair** | 自动类型推导 | 可能推导出非预期类型 | 类型简单且明确时 |
| **列表初始化** | 语法简洁 | C++11+ | 返回值、临时对象 |

---

## 访问和操作

### 1. 访问元素

pair 的成员是公开的，可以直接访问：

```cpp
std::pair<int, std::string> p(42, "hello");

// 直接访问
int num = p.first;           // 访问第一个元素
std::string str = p.second;  // 访问第二个元素

// 修改元素
p.first = 100;
p.second = "world";
```

### 2. 结构化绑定 (C++17)

C++17 引入的结构化绑定让代码更简洁：

```cpp
std::pair<bool, double> result = safeDivide(10.0, 2.0);

// 传统方式
if (result.first) {
    std::cout << result.second << std::endl;
}

// 结构化绑定
auto [success, value] = safeDivide(10.0, 2.0);
if (success) {
    std::cout << value << std::endl;
}
```

### 3. 比较操作

pair 支持所有比较运算符，按字典序比较：

```cpp
std::pair<int, int> a(1, 2);
std::pair<int, int> b(1, 3);
std::pair<int, int> c(2, 1);

// 比较规则：先比较 first，如果相等再比较 second
a < b   // true  (first 相等，second: 2 < 3)
a < c   // true  (first: 1 < 2)
a == b  // false
```

### 比较运算符表

| 运算符 | 说明 | 比较规则 |
|--------|------|----------|
| `==` | 相等 | first 和 second 都相等 |
| `!=` | 不等 | first 或 second 不相等 |
| `<` | 小于 | 先比较 first，相等则比较 second |
| `<=` | 小于等于 | 同上 |
| `>` | 大于 | 同上 |
| `>=` | 大于等于 | 同上 |

---

## 典型应用场景

### 1. 函数返回多个值

在 C++17 之前，pair 是返回两个值的标准方式：

```cpp
// 返回操作结果和状态
std::pair<bool, double> safeDivide(double a, double b) {
    if (b == 0) {
        return {false, 0.0};  // 失败
    }
    return {true, a / b};     // 成功
}

// 使用
auto [success, result] = safeDivide(10.0, 2.0);
if (success) {
    std::cout << "结果: " << result << std::endl;
}
```

**参考代码**：`std_pair.cpp:8-13`

### 2. 查找最小值和最大值

```cpp
std::pair<int, int> findMinMax(const std::vector<int>& nums) {
    if (nums.empty()) return {0, 0};

    int minVal = nums[0];
    int maxVal = nums[0];

    for (int num : nums) {
        if (num < minVal) minVal = num;
        if (num > maxVal) maxVal = num;
    }

    return {minVal, maxVal};
}

// 使用
std::vector<int> numbers = {5, 2, 9, 1, 7, 3};
auto [minNum, maxNum] = findMinMax(numbers);
std::cout << "最小值: " << minNum << ", 最大值: " << maxNum << std::endl;
```

**参考代码**：`std_pair.cpp:16-28`

### 3. 与 std::map 配合使用

`std::map` 的元素类型是 `pair<const Key, Value>`：

```cpp
std::map<std::string, int> ages;

// 插入元素
ages.insert(std::make_pair("Alice", 30));
ages.insert({"Bob", 25});

// 遍历 map (每个元素是 pair)
for (const auto& person : ages) {
    std::cout << person.first << " 的年龄是 " << person.second << std::endl;
}

// C++17 结构化绑定
for (const auto& [name, age] : ages) {
    std::cout << name << ": " << age << " 岁" << std::endl;
}
```

**参考代码**：`std_pair.cpp:62-78`

### 4. 存储不同类型的组合

pair 可以组合任意两种类型：

```cpp
// 学生姓名和成绩列表
std::pair<std::string, std::vector<int>> studentScores("张三", {85, 90, 92});

std::cout << studentScores.first << " 的成绩: ";
for (int score : studentScores.second) {
    std::cout << score << " ";
}
```

**参考代码**：`std_pair.cpp:91-97`

### 应用场景总结

| 场景 | 示例 | 优势 |
|------|------|------|
| **函数返回多值** | 返回成功标志和结果 | 避免使用输出参数 |
| **算法返回值** | `std::minmax_element` | 标准库统一接口 |
| **关联容器元素** | `std::map` 的键值对 | 自然的键值表示 |
| **临时数据组合** | 组合两个相关数据 | 无需定义结构体 |

---

## 注意事项和最佳实践

### 1. 语义清晰性问题

**问题**：`first` 和 `second` 缺乏语义，代码可读性差

```cpp
// 不好：语义不清
std::pair<std::string, int> person("Alice", 30);
std::cout << person.first << " is " << person.second << " years old" << std::endl;

// 更好：使用结构化绑定
auto [name, age] = person;
std::cout << name << " is " << age << " years old" << std::endl;

// 最好：对于复杂数据，定义结构体
struct Person {
    std::string name;
    int age;
};
```

**建议**：
- 简单临时组合使用 pair
- 需要多次使用或超过2个字段时，定义结构体
- C++17+ 优先使用结构化绑定提高可读性

### 2. make_pair 的类型推导陷阱

```cpp
// 陷阱：推导为 const char* 而非 string
auto p1 = std::make_pair("hello", 42);  // pair<const char*, int>

// 解决方案1：显式指定类型
std::pair<std::string, int> p2 = std::make_pair("hello", 42);

// 解决方案2：使用 string 字面量
auto p3 = std::make_pair(std::string("hello"), 42);

// 解决方案3：使用列表初始化
std::pair<std::string, int> p4{"hello", 42};
```

### 3. pair vs tuple vs struct

选择合适的数据结构：

| 特性 | pair | tuple | struct |
|------|------|-------|--------|
| **元素数量** | 固定2个 | 任意数量 | 任意数量 |
| **成员名称** | first, second | 无名称（索引访问） | 自定义名称 |
| **语义清晰** | ❌ 差 | ❌ 差 | ✅ 好 |
| **使用场景** | 临时组合2个值 | 临时组合多个值 | 长期使用的数据结构 |
| **C++版本** | C++98 | C++11 | 所有版本 |

**选择建议**：
- **pair**：临时组合2个值，如函数返回值
- **tuple**：临时组合3个或更多值
- **struct**：需要多次使用、需要清晰语义的数据结构

### 4. 性能考虑

```cpp
// pair 是轻量级的，通常按值传递
std::pair<int, int> getCoordinates() {
    return {10, 20};  // 高效，可能被优化为 RVO
}

// 对于大对象，考虑使用引用
std::pair<const std::string&, const std::vector<int>&>
getLargeData(const std::string& str, const std::vector<int>& vec) {
    return {str, vec};  // 返回引用，避免拷贝
}
```

**性能提示**：
- 小对象（如基本类型）：按值传递
- 大对象：考虑使用引用或移动语义
- C++17+ 结构化绑定不会产生额外拷贝

---

## 常用成员函数和操作

### 成员函数表

| 函数 | 说明 | 示例 |
|------|------|------|
| `pair()` | 默认构造函数 | `std::pair<int, int> p;` |
| `pair(const T1& x, const T2& y)` | 值构造函数 | `std::pair<int, int> p(1, 2);` |
| `pair(const pair& p)` | 拷贝构造函数 | `std::pair<int, int> p2(p1);` |
| `pair(pair&& p)` | 移动构造函数 (C++11) | `std::pair<int, int> p2(std::move(p1));` |
| `operator=` | 赋值运算符 | `p1 = p2;` |
| `swap(pair& p)` | 交换两个 pair | `p1.swap(p2);` |

### 非成员函数

| 函数 | 说明 | 示例 |
|------|------|------|
| `make_pair(T1, T2)` | 创建 pair，自动推导类型 | `auto p = std::make_pair(1, 2);` |
| `swap(pair&, pair&)` | 交换两个 pair | `std::swap(p1, p2);` |
| `get<N>(pair)` | 获取第 N 个元素 (C++11) | `std::get<0>(p);` |

### 比较运算符

所有比较运算符都支持：`==`, `!=`, `<`, `<=`, `>`, `>=`

---

## 总结

### 核心要点

1. **定义**：`std::pair` 是存储两个值的轻量级模板类
2. **头文件**：`<utility>`
3. **成员访问**：通过 `first` 和 `second` 访问元素
4. **创建方式**：构造函数、`make_pair`、列表初始化
5. **C++17 特性**：结构化绑定提高代码可读性

### 使用建议

| 场景 | 建议 |
|------|------|
| **临时组合2个值** | ✅ 使用 pair |
| **函数返回2个值** | ✅ 使用 pair（C++17前）或结构化绑定（C++17+） |
| **需要3个或更多值** | ❌ 使用 tuple 或 struct |
| **需要清晰语义** | ❌ 定义 struct |
| **长期使用的数据结构** | ❌ 定义 struct |

### 优缺点

**优点**：
- ✅ 轻量级，零开销
- ✅ 标准库广泛使用（如 map）
- ✅ 支持比较操作
- ✅ C++98 即可用

**缺点**：
- ❌ 只能存储2个元素
- ❌ `first`/`second` 缺乏语义
- ❌ 不适合复杂数据结构

### 相关组件

- [**std::tuple**](../std_tuple/std_tuple.md)：存储任意数量的元素
- [**std::map**](../std_map/std_map.md)：使用 pair 作为元素类型
- **struct**：自定义数据结构

---

## 参考资料

- **示例代码**：[std_pair.cpp](./std_pair.cpp)
- **C++ Reference**：https://en.cppreference.com/w/cpp/utility/pair
- **相关文档**：[std组件汇总](../std汇总.md)

