# C++ Compiler Internals Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Compilation Pipeline](#compilation-pipeline)
3. [Template Compilation](#template-compilation)
4. [Symbol Management](#symbol-management)
5. [Name Mangling](#name-mangling)
6. [One Definition Rule (ODR)](#one-definition-rule-odr)
7. [Linker Symbol Resolution](#linker-symbol-resolution)
8. [Understanding Compilation Errors](#understanding-compilation-errors)

---

## Introduction

Understanding compiler internals helps you write better code, debug complex issues, and optimize compilation times. This guide explores how C++ compilers transform source code into executable binaries.

---

## Compilation Pipeline

### The Four Stages Revisited

```
Source Code (.cpp)
    ↓
[Preprocessor] → Expanded source (.i)
    ↓
[Compiler Frontend] → Assembly (.s)
    ↓
[Assembler] → Object file (.o)
    ↓
[Linker] → Executable
```

### Compiler Frontend Stages

**1. Lexical Analysis (Lexer)**
- Converts source code into tokens
- Identifies keywords, identifiers, operators, literals

```cpp
int x = 42;
// Tokens: [int] [x] [=] [42] [;]
```

**2. Syntax Analysis (Parser)**
- Builds Abstract Syntax Tree (AST)
- Checks grammar rules

```
Assignment
├── Variable: x
└── Literal: 42
```

**3. Semantic Analysis**
- Type checking
- Symbol table construction
- Scope resolution

**4. Intermediate Representation (IR)**
- Platform-independent representation
- LLVM IR, GCC GIMPLE

**5. Optimization**
- Dead code elimination
- Constant folding
- Inlining

**6. Code Generation**
- Converts IR to assembly
- Register allocation

---

## Template Compilation

### Template Instantiation Model

Templates are compiled in two phases:

**Phase 1: Template Definition**
- Syntax checking
- Name lookup for non-dependent names

**Phase 2: Template Instantiation**
- Occurs when template is used
- Full type checking
- Code generation

### Example

```cpp
// Template definition (Phase 1)
template<typename T>
T add(T a, T b) {
    return a + b;
}

// Template instantiation (Phase 2)
int result = add(5, 3);        // Instantiates add<int>
double d = add(1.5, 2.5);      // Instantiates add<double>
```

### Implicit vs Explicit Instantiation

**Implicit instantiation:**
```cpp
template<typename T>
class Vector { /*...*/ };

Vector<int> v;  // Compiler generates Vector<int> code
```

**Explicit instantiation:**
```cpp
// In header
template<typename T>
class Vector { /*...*/ };

// In .cpp file
template class Vector<int>;  // Force instantiation
```

### Extern Template (C++11)

Prevents implicit instantiation to reduce compilation time and binary size.

```cpp
// header.h
template<typename T>
class MyClass { /*...*/ };

extern template class MyClass<int>;  // Don't instantiate here

// source1.cpp
#include "header.h"
template class MyClass<int>;  // Explicit instantiation

// source2.cpp
#include "header.h"
MyClass<int> obj;  // Uses instantiation from source1.cpp
```

**Benefits:**
- Reduces compilation time
- Smaller object files
- Single instantiation point

---

## Symbol Management

### Symbol Table

The compiler maintains a symbol table tracking:
- Function names and signatures
- Variable names and types
- Class/struct definitions
- Template instantiations

### Symbol Types

```bash
# View symbols in object file
nm myfile.o

# Symbol types:
# T - Text (code) section
# D - Initialized data
# B - Uninitialized data (BSS)
# U - Undefined (external reference)
# W - Weak symbol
```

---

## Name Mangling

### Why Name Mangling?

C++ supports function overloading, namespaces, and templates. The linker needs unique names for each symbol.

```cpp
// Source code
int add(int a, int b);
double add(double a, double b);

// Mangled names (GCC/Clang)
_Z3addii      // add(int, int)
_Z3adddd      // add(double, double)
```

### Mangling Examples

```cpp
namespace math {
    class Calculator {
        int add(int a, int b);
    };
}

// Mangled: _ZN4math10Calculator3addEii
// Breakdown:
// _Z - prefix
// N - nested name
// 4math - namespace "math" (4 chars)
// 10Calculator - class "Calculator" (10 chars)
// 3add - function "add" (3 chars)
// E - end of nested name
// ii - parameters (int, int)
```

### Demangling Tools

```bash
# GCC/Clang
c++filt _Z3addii
# Output: add(int, int)

# View demangled symbols
nm -C myfile.o

# MSVC
undname ?add@@YAHHH@Z
```

### extern "C" Linkage

Prevents name mangling for C compatibility:

```cpp
extern "C" {
    int add(int a, int b);  // No mangling: add
}

// C++ function
int multiply(int a, int b);  // Mangled: _Z8multiplyii
```

---

## One Definition Rule (ODR)

### The Rule

Every entity (function, variable, class) must have exactly one definition in the entire program.

### ODR Violations

**Violation: Multiple definitions**
```cpp
// file1.cpp
int globalVar = 10;

// file2.cpp
int globalVar = 20;  // ❌ ODR violation
// Linker error: multiple definition of 'globalVar'
```

**Correct: Use extern**
```cpp
// header.h
extern int globalVar;

// file1.cpp
int globalVar = 10;  // Definition

// file2.cpp
extern int globalVar;  // Declaration only
```

### Inline Functions and ODR

Inline functions are exempt from ODR (can be defined in headers):

```cpp
// header.h
inline int add(int a, int b) {
    return a + b;  // OK in header
}
```

### Templates and ODR

Templates are also exempt (definitions in headers):

```cpp
// header.h
template<typename T>
T max(T a, T b) {
    return a > b ? a : b;  // OK in header
}
```

---

## Linker Symbol Resolution

### How the Linker Works

1. **Collect symbols** from all object files
2. **Resolve references** - match undefined symbols to definitions
3. **Relocate addresses** - assign final memory addresses
4. **Generate executable**

### Symbol Resolution Process

```cpp
// file1.cpp
void helper();  // Declaration (undefined symbol)

int main() {
    helper();   // Reference to undefined symbol
    return 0;
}

// file2.cpp
void helper() {  // Definition
    // implementation
}
```

**Linker process:**
1. file1.o has undefined symbol: `helper`
2. file2.o has defined symbol: `helper`
3. Linker matches them → success

### Common Linker Errors

**Undefined reference:**
```
undefined reference to `function_name'
```
**Cause:** Function declared but not defined

**Multiple definition:**
```
multiple definition of `variable_name'
```
**Cause:** ODR violation

---

## Understanding Compilation Errors

### Template Error Messages

Template errors can be verbose due to instantiation chains:

```cpp
template<typename T>
void process(T value) {
    value.nonexistent();  // Error
}

int main() {
    process(42);
}
```

**Error message:**
```
error: 'int' has no member named 'nonexistent'
  in instantiation of function template 'process<int>'
  requested here: process(42);
```

### Reading Error Messages

**Top-down approach:**
1. Find the actual error location
2. Trace back through instantiation chain
3. Identify root cause

### Common Error Patterns

**Missing semicolon:**
```
expected ';' after class definition
```

**Template syntax:**
```
expected primary-expression before '>' token
```

**Linker errors:**
```
undefined reference to 'vtable for ClassName'
```
**Cause:** Missing virtual function implementation

---

## Conclusion

Understanding compiler internals helps you:

- **Write better code** - Understand cost of language features
- **Debug faster** - Interpret error messages correctly
- **Optimize builds** - Use extern template, explicit instantiation
- **Avoid pitfalls** - Understand ODR, name mangling, symbol resolution

Key takeaways:
- Templates are instantiated on-demand
- Name mangling enables C++ features
- ODR prevents duplicate definitions
- Linker resolves symbols across translation units

