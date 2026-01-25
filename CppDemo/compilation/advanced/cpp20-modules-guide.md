# C++20 Modules Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Modules vs Headers](#modules-vs-headers)
3. [Basic Module Syntax](#basic-module-syntax)
4. [Module Compilation](#module-compilation)
5. [Migration Strategy](#migration-strategy)
6. [Toolchain Support](#toolchain-support)
7. [Best Practices](#best-practices)

---

## Introduction

C++20 introduces modules as a modern alternative to header files, addressing long-standing compilation issues.

### Problems with Headers

- Slow compilation (repeated parsing)
- Macro pollution
- Order-dependent includes
- Fragile include guards

### Module Benefits

✅ Faster compilation (parsed once)
✅ No macro leakage
✅ Order-independent imports
✅ Better encapsulation
✅ Smaller binary size

---

## Modules vs Headers

### Traditional Headers

```cpp
// math.h
#ifndef MATH_H
#define MATH_H

int add(int a, int b);
int multiply(int a, int b);

#endif

// math.cpp
#include "math.h"
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }

// main.cpp
#include "math.h"
int main() {
    return add(1, 2);
}
```

### With Modules

```cpp
// math.cppm (module interface)
export module math;

export int add(int a, int b) { return a + b; }
export int multiply(int a, int b) { return a * b; }

// main.cpp
import math;
int main() {
    return add(1, 2);
}
```

---

## Basic Module Syntax

### Module Declaration

```cpp
// Simple module
export module mymodule;

export int getValue() { return 42; }
```

### Module Interface vs Implementation

**Interface (.cppm):**
```cpp
export module math;

export int add(int a, int b);
export int multiply(int a, int b);
```

**Implementation (.cpp):**
```cpp
module math;

int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
```

### Importing Modules

```cpp
import math;           // Import module
import std;            // Import standard library (C++23)

int main() {
    return add(1, 2);
}
```

### Module Partitions

```cpp
// math-impl.cppm (partition)
export module math:impl;
int helper() { return 0; }

// math.cppm (primary interface)
export module math;
import :impl;
export int add(int a, int b);
```

---

## Module Compilation

### Compilation Order

Modules must be compiled in dependency order:

```bash
# 1. Compile module interface first
g++ -std=c++20 -fmodules-ts -c math.cppm -o math.o

# 2. Compile main program
g++ -std=c++20 -fmodules-ts -c main.cpp -o main.o

# 3. Link
g++ math.o main.o -o app
```

### CMake Support (3.28+)

```cmake
cmake_minimum_required(VERSION 3.28)
project(ModuleDemo CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_executable(app
    main.cpp
    math.cppm
)
```

---

## Migration Strategy

### Gradual Migration

1. **Start with new code** - Use modules for new features
2. **Keep headers** - Maintain backward compatibility
3. **Wrap headers** - Create module wrappers

---

## Toolchain Support

### Compiler Support (2024)

| Compiler | Version | Support Level |
|----------|---------|---------------|
| **GCC** | 11+ | Experimental |
| **Clang** | 16+ | Good |
| **MSVC** | VS 2019+ | Good |

### Build System Support

- **CMake**: 3.28+ (native support)
- **Ninja**: Full support
- **MSBuild**: Full support

---

## Best Practices

### 1. Module Naming

```cpp
// Good - descriptive names
export module company.product.feature;

// Bad - generic names
export module utils;
```

### 2. Minimize Exports

```cpp
// Only export public API
export module math;

export int add(int a, int b);  // Public
int helper();                   // Private (not exported)
```

### 3. Use Module Partitions

Split large modules into partitions for better organization.

---

## Conclusion

C++20 modules represent the future of C++ compilation:

- **Faster builds** - Significant compilation speedup
- **Better encapsulation** - No macro leakage
- **Modern design** - Cleaner than headers

**Current status**: Toolchain support is improving but not yet universal. Consider for new projects.

