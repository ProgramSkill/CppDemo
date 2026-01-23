# std::bitset 详细解析

## 概述

`std::bitset`是固定大小的位集合，用于位操作。

```cpp
#include <bitset>
```

## 核心特性

| 特性 | 说明 |
|------|------|
| 固定大小 | 编译时确定 |
| 位操作 | 高效位操作 |
| 内存紧凑 | 每个元素1bit |

## 基本用法

```cpp
bitset<8> b(0b10101010);

cout << b;          // 10101010
cout << b[2];       // 0
cout << b.count();  // 4（1的个数）
cout << b.size();   // 8
```

## 位操作

```cpp
bitset<8> b;

// 设置
b.set();        // 全1
b.set(2);       // 设置位2
b.reset(2);     // 清除位2
b.flip(2);      // 翻转位2

// 测试
bool x = b.test(3);
bool y = b[3];

// 全局操作
b.all();    // 是否全1
b.any();    // 是否有1
b.none();   // 是否全0
```

## 运算

```cpp
bitset<8> b1(0b1100);
bitset<8> b2(0b1010);

b1 & b2;   // 1000（与）
b1 | b2;   // 1110（或）
b1 ^ b2;   // 0110（异或）
~b1;       // 0011（非）
b1 << 2;   // 110000（左移）
b1 >> 1;   // 0110（右移）
```

## 使用场景

```cpp
// 1. 标志位
enum Flag { READ = 0, WRITE = 1, EXECUTE = 2 };
bitset<3> flags;
flags[READ] = true;

// 2. 子集
bitset<5> s1(0b10101);  // {0, 2, 4}
bitset<5> s2(0b00101);  // {0, 2}
bitset<5> inter = s1 & s2;  // 交集

// 3. 筛法
bitset<N+1> isPrime;
isPrime.set();  // 全设为true
isPrime[0] = isPrime[1] = false;
for (int i = 2; i*i <= N; ++i) {
    if (isPrime[i]) {
        for (int j = i*i; j <= N; j += i) {
            isPrime[j] = false;
        }
    }
}
```

## 参考文档
- [cppreference - std::bitset](https://en.cppreference.com/w/cpp/utility/bitset)
