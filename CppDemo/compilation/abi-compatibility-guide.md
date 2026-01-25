# ABI Compatibility Guide

## Table of Contents
1. [Introduction](#introduction)
2. [What is ABI?](#what-is-abi)
3. [ABI vs API](#abi-vs-api)
4. [Symbol Visibility](#symbol-visibility)
5. [ABI Breaking Changes](#abi-breaking-changes)
6. [Version Management](#version-management)
7. [Best Practices](#best-practices)

---

## Introduction

Application Binary Interface (ABI) defines how compiled code interacts at the binary level. ABI compatibility is crucial for library updates without recompilation.

---

## What is ABI?

ABI specifies:
- Function calling conventions
- Data structure layout
- Name mangling rules
- Virtual table layout
- Exception handling

### Why ABI Matters

**ABI compatible**: Update library without recompiling clients
**ABI incompatible**: Must recompile all dependent code

---

## ABI vs API

### API (Application Programming Interface)

Source-level interface - what you see in code:

```cpp
// API
class Calculator {
public:
    int add(int a, int b);
    int multiply(int a, int b);
};
```

### ABI (Application Binary Interface)

Binary-level interface - how it's compiled:

- Function name mangling: `_ZN10Calculator3addEii`
- Memory layout of `Calculator` class
- Virtual table structure
- Calling convention

### Example

```cpp
// Version 1.0 - API and ABI
class Widget {
    int value;
public:
    int getValue();
};

// Version 2.0 - API compatible, ABI incompatible
class Widget {
    int value;
    int newField;  // ❌ Changes memory layout
public:
    int getValue();
};
```

---

## Symbol Visibility

### Controlling Symbol Export

**GCC/Clang:**
```cpp
// Export symbol
__attribute__((visibility("default")))
void publicFunction();

// Hide symbol
__attribute__((visibility("hidden")))
void internalFunction();
```

**MSVC:**
```cpp
// Export from DLL
__declspec(dllexport) void publicFunction();

// Import from DLL
__declspec(dllimport) void publicFunction();
```

### Cross-Platform Macro

```cpp
#ifdef _WIN32
  #ifdef BUILDING_DLL
    #define API __declspec(dllexport)
  #else
    #define API __declspec(dllimport)
  #endif
#else
  #define API __attribute__((visibility("default")))
#endif

API void publicFunction();
```

---

## ABI Breaking Changes

### What Breaks ABI

❌ **Adding/removing class members**
```cpp
class Widget {
    int value;
    int newMember;  // ❌ Breaks ABI
};
```

❌ **Changing member order**
```cpp
class Widget {
    int b;  // ❌ Swapped order
    int a;
};
```

❌ **Adding virtual functions**
```cpp
class Base {
    virtual void foo();
    virtual void bar();  // ❌ Changes vtable
};
```

❌ **Changing function signatures**
```cpp
// Old
void process(int x);

// New
void process(int x, int y = 0);  // ❌ Different mangled name
```

### What Preserves ABI

✅ **Adding new functions**
```cpp
class Widget {
    int getValue();
    int getDoubleValue();  // ✅ OK - new function
};
```

✅ **Adding non-virtual overloads**
```cpp
void process(int x);
void process(double x);  // ✅ OK - different signature
```

---

## Version Management

### Semantic Versioning for ABI

**MAJOR.MINOR.PATCH**

- **MAJOR**: ABI-breaking changes
- **MINOR**: ABI-compatible additions
- **PATCH**: Bug fixes (ABI-compatible)

### Library Versioning (Linux)

```bash
# Create versioned shared library
g++ -shared -fPIC -Wl,-soname,libmylib.so.1 \
    mylib.cpp -o libmylib.so.1.2.3

# Create symlinks
ln -s libmylib.so.1.2.3 libmylib.so.1
ln -s libmylib.so.1 libmylib.so
```

**Usage:**
- `libmylib.so` - Development (latest)
- `libmylib.so.1` - Runtime (ABI version)
- `libmylib.so.1.2.3` - Specific version

---

## Best Practices

### 1. Use PIMPL Pattern

Hide implementation details to maintain ABI stability:

```cpp
// widget.h
class Widget {
public:
    Widget();
    ~Widget();
    void doSomething();
private:
    class Impl;
    Impl* pImpl;  // Pointer to implementation
};

// widget.cpp
class Widget::Impl {
    int value;
    std::string data;  // Can change without breaking ABI
};
```

### 2. Version Your Namespaces

```cpp
namespace mylib {
inline namespace v1 {
    class Widget { /*...*/ };
}
}

// Later version
namespace mylib {
inline namespace v2 {
    class Widget { /*...*/ };  // Different ABI
}
}
```

### 3. Document ABI Guarantees

Clearly state your ABI policy:
- Which versions maintain ABI compatibility
- When ABI breaks are allowed
- Migration guides for breaking changes

### 4. Test ABI Compatibility

Use tools to detect ABI changes:
```bash
# Linux - check ABI with abi-compliance-checker
abi-compliance-checker -lib mylib \
  -old libmylib-1.0.so -new libmylib-2.0.so
```

---

## Conclusion

ABI compatibility is critical for library maintenance:

- **Understand what breaks ABI** - Class layout, vtables, signatures
- **Use PIMPL** - Hide implementation details
- **Version carefully** - Follow semantic versioning
- **Test changes** - Use ABI checking tools

Key takeaway: ABI stability enables seamless library updates without recompiling dependent code.

