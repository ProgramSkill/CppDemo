# std::string_view 详细解析

## 概述

`std::string_view`是C++17引入的非拥有字符串引用。

```cpp
#include <string_view>
```

## 核心特性

| 特性 | std::string | std::string_view |
|------|-------------|------------------|
| 拥有内存 | ✅ | ❌ |
| 拷贝开销 | O(n) | **O(1)** |
| 修改 | ✅ | ❌ 只读 |
| C++17 | ✅ | ✅ |

## 基本用法

```cpp
string str = "hello world";
string_view sv = str;  // 零拷贝

cout << sv;
cout << sv.size();
sv = sv.substr(0, 5);  // 子串，零拷贝
```

## 使用场景

```cpp
// 1. 函数参数（避免拷贝）
void print(string_view sv);
print("literal");   // OK
print(string("temp"));  // OK
string s = "test"; print(s);  // OK

// 2. 字符串分割
vector<string_view> split(string_view str, char delim) {
    vector<string_view> result;
    size_t start = 0;
    for (size_t i = 0; i <= str.size(); ++i) {
        if (i == str.size() || str[i] == delim) {
            result.push_back(str.substr(start, i - start));
            start = i + 1;
        }
    }
    return result;
}
```

## 注意事项

⚠️ **生命周期**：
```cpp
// 危险！
string_view sv = string("temp");
// sv悬空了
```

## 参考文档
- [cppreference - std::string_view](https://en.cppreference.com/w/cpp/string/basic_string_view)
