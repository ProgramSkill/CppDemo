# std::tuple 详细解析

## 概述

`std::tuple`是C++11引入的固定大小异构集合，可以存储不同类型的元素。

```cpp
#include <tuple>
```

## 核心特性

| 特性 | std::pair | std::tuple |
|------|----------|------------|
| 元素数量 | 2个 | 任意个 |
| 类型 | 可不同 | 可不同 |
| 大小 | 编译时确定 | 编译时确定 |

## 基本用法

```cpp
// 创建
tuple<int, string, double> t1(42, "hello", 3.14);
tuple<int, string, double> t2 = make_tuple(42, "hello", 3.14);
auto t3 = make_tuple(42, "hello", 3.14);  // C++11自动推导

// 访问
int x = get<0>(t1);      // 42
string s = get<1>(t1);   // "hello"
double d = get<2>(t1);   // 3.14

// 按类型访问
int y = get<int>(t1);    // 42
```

## 结构化绑定（C++17）

```cpp
tuple<string, int, double> student("Alice", 25, 95.5);

// C++17之前
string name = get<0>(student);
int age = get<1>(student);
double score = get<2>(student);

// C++17结构化绑定
auto [name, age, score] = student;
cout << name << " " << age << " " << score;
```

## tie和ignore

```cpp
tuple<int, int, int> t(1, 2, 3);

int a, b, c;
tie(a, b, c) = t;  // 解包

tie(a, ignore, c) = t;  // 忽略中间值
```

## tuple_cat（C++17）

```cpp
tuple<int, char> t1(1, 'a');
tuple<double, string> t2(3.14, "hello");

auto t3 = tuple_cat(t1, t2);
// 类型: tuple<int, char, double, string>
```

## 使用场景

```cpp
// 1. 多返回值
tuple<int, int, int> divide(int a, int b) {
    return make_tuple(a / b, a % b, a * b);
}
auto [quotient, remainder, product] = divide(17, 5);

// 2. 函数参数打包
template<typename... Args>
void print_all(Args... args) {
    tuple<Args...> t(args...);
    // 处理...
}

// 3. 多键比较
struct Person {
    string name;
    int age;
    bool operator<(const Person& other) const {
        return tie(age, name) < tie(other.age, other.name);
    }
};
```

## 参考文档
- [cppreference - std::tuple](https://en.cppreference.com/w/cpp/utility/tuple)
