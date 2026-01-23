# std::map 详细解析

## 概述

`std::map`是基于红黑树的关联容器，存储**键值对**，键**唯一**且**有序**。

```cpp
#include <map>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 键值对 | 每个元素是pair<const Key, Value> |
| 键唯一 | 自动去重，重复键会替换值 |
| 自动排序 | 按键排序（默认升序） |
| 底层实现 | 红黑树 |
| 查找效率 | O(log n) |

## 基本用法

```cpp
// 初始化
map<string, int> m;

// 插入
m["apple"] = 5;        // operator[]
m.insert({"banana", 3});
m.emplace("cherry", 8);  // C++11

// 访问
int x = m["apple"];     // 如果不存在会创建默认值
int y = m.at("apple");  // 不存在抛出异常

// 检查存在
if (m.find("apple") != m.end()) {
    cout << "存在";
}
```

## 元素访问

| 方式 | 键不存在时 | 备注 |
|------|-----------|------|
| `m[key]` | 创建默认值并返回引用 | 可修改值 |
| `m.at(key)` | 抛出out_of_range | 只读或可写 |
| `m.find(key)` | 返回end() | 不创建元素 |

```cpp
map<string, int> m;

// m["apple"] - 存在则返回，不存在则创建
int count = m["apple"];  // 如果不存在，m["apple"]=0
// 现在m中有{"apple": 0}

// m.at() - 安全访问
try {
    int value = m.at("banana");  // 不存在会抛异常
} catch (const out_of_range& e) {
    cerr << "键不存在";
}
```

## 修改操作

```cpp
map<string, int> m;

// 插入或更新
m["key"] = value;  // 键存在则更新，不存在则插入

// insert - 键存在则失败
auto result = m.insert({"key", 100});
// result.second == false 表示键已存在

// insert_or_assign (C++17) - 总是成功
m.insert_or_assign("key", 200);  // 存在则更新，不存在则插入

// emplace - 原位构造
m.emplace("key", 300);
```

## 使用场景

```cpp
// 1. 字典/词频统计
map<string, int> word_count;
string word;
while (cin >> word) {
    word_count[word]++;  // 优雅
}

// 2. 配置存储
map<string, string> config;
config["host"] = "localhost";
config["port"] = "8080";
cout << config["host"] << ":" << config["port"];

// 3. 分组数据
map<int, vector<string>> groups;
groups[1].push_back("apple");
groups[1].push_back("banana");  // 组1

// 4. 有序键访问
map<int, string> m = {{3, "C"}, {1, "A"}, {2, "B"}};
for (auto& [k, v] : m) {
    cout << k << ":" << v << endl;  // 按k排序输出
}
```

## 自定义比较器

```cpp
// 降序
map<int, string, greater<int>> m;

// 自定义排序
struct Person {
    string name;
    int age;
};
auto cmp = [](const Person& a, const Person& b) {
    return a.age < b.age;
};
map<Person, int, decltype(cmp)> m(cmp);
```

## 何时使用map

✅ **适合**：
- 需要键值对存储
- 键唯一且需要有序
- 频繁按键查找
- 一对一映射关系

❌ **不适合**：
- 不需要有序 → unordered_map更快
- 需要重复键 → multimap
- 键是整数且范围小 → 考虑vector

## 参考文档
- [cppreference - std::map](https://en.cppreference.com/w/cpp/container/map)
