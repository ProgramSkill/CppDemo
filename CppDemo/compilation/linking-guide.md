# C++ Linking Guide: Static vs Dynamic

## Table of Contents
1. [Introduction](#introduction)
2. [What is Linking?](#what-is-linking)
3. [Static Linking](#static-linking)
4. [Dynamic Linking](#dynamic-linking)
5. [Comparison](#comparison)
6. [Library Creation](#library-creation)
7. [Common Issues](#common-issues)
8. [Best Practices](#best-practices)

---

## Introduction

Linking is the final stage of compilation that combines object files and libraries into an executable. Understanding the difference between static and dynamic linking is crucial for managing dependencies, binary size, and deployment.

---

## What is Linking?

The linker performs three main tasks:

1. **Symbol Resolution**: Matches function calls to their definitions
2. **Relocation**: Assigns final memory addresses to code and data
3. **Library Integration**: Incorporates library code into the executable

---

## Static Linking

Static linking copies library code directly into the executable at compile time.

### How It Works

```
Object Files + Static Libraries (.a, .lib)
              ↓
          Linker
              ↓
    Standalone Executable
```

### Creating a Static Library

**Linux/macOS:**
```bash
# Compile source files
g++ -c utils.cpp -o utils.o
g++ -c helper.cpp -o helper.o

# Create static library
ar rcs libmylib.a utils.o helper.o

# Link with static library
g++ main.cpp -L. -lmylib -o app
```

**Windows (MSVC):**
```cmd
cl /c utils.cpp helper.cpp
lib /OUT:mylib.lib utils.obj helper.obj
cl main.cpp mylib.lib
```

### Advantages

✅ **No runtime dependencies** - executable is self-contained
✅ **Faster startup** - no dynamic loading overhead
✅ **Version control** - no DLL hell issues
✅ **Easier deployment** - single file distribution

### Disadvantages

❌ **Larger executables** - library code duplicated in each program
❌ **No updates** - must recompile to update library
❌ **Memory waste** - multiple processes can't share library code
❌ **Longer link times** - especially for large libraries

---

## Dynamic Linking

Dynamic linking loads library code at runtime, keeping executables small and libraries shared.

### How It Works

```
Executable + Shared Libraries (.so, .dll, .dylib)
              ↓
      Runtime Loader
              ↓
    Program in Memory
```

### Creating a Shared Library

**Linux:**
```bash
# Create shared library
g++ -shared -fPIC utils.cpp helper.cpp -o libmylib.so

# Link with shared library
g++ main.cpp -L. -lmylib -o app

# Run (need library in path)
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./app
```

**Windows:**
```cmd
# Create DLL
cl /LD utils.cpp helper.cpp /Fe:mylib.dll

# Link with DLL
cl main.cpp mylib.lib
```

### Advantages

✅ **Smaller executables** - library code not embedded
✅ **Shared memory** - multiple processes share one library copy
✅ **Easy updates** - update library without recompiling programs
✅ **Plugin architecture** - load libraries dynamically at runtime

### Disadvantages

❌ **Runtime dependencies** - requires correct library versions
❌ **DLL hell** - version conflicts between applications
❌ **Slower startup** - dynamic loading overhead
❌ **Deployment complexity** - must distribute libraries

---

## Comparison

| Aspect | Static Linking | Dynamic Linking |
|--------|----------------|-----------------|
| **Binary Size** | Large | Small |
| **Memory Usage** | High (duplicated) | Low (shared) |
| **Startup Time** | Fast | Slower |
| **Dependencies** | None | Required at runtime |
| **Updates** | Recompile needed | Replace library file |
| **Deployment** | Simple | Complex |
| **Best For** | Embedded, standalone apps | Desktop, server apps |

---

## Library Creation

### Complete Example

**Library header** (`mylib.h`):
```cpp
#ifndef MYLIB_H
#define MYLIB_H

#ifdef _WIN32
  #ifdef BUILDING_DLL
    #define API __declspec(dllexport)
  #else
    #define API __declspec(dllimport)
  #endif
#else
  #define API
#endif

API int add(int a, int b);
API int multiply(int a, int b);

#endif
```

**Library implementation** (`mylib.cpp`):
```cpp
#include "mylib.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}
```

**Build commands:**
```bash
# Static library
g++ -c mylib.cpp -o mylib.o
ar rcs libmylib.a mylib.o

# Shared library
g++ -shared -fPIC mylib.cpp -o libmylib.so
```

---

## Common Issues

### Issue 1: Undefined Reference

**Error:**
```
undefined reference to `function_name'
```

**Solutions:**
```bash
# Check library order (dependencies last)
g++ main.cpp -lmylib -lpthread

# Specify library path
g++ main.cpp -L/path/to/libs -lmylib
```

### Issue 2: Library Not Found at Runtime

**Error:**
```
error while loading shared libraries: libmylib.so: cannot open shared object file
```

**Solutions:**
```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH

# Install to system path
sudo cp libmylib.so /usr/local/lib
sudo ldconfig

# Use RPATH
g++ main.cpp -L. -lmylib -Wl,-rpath,'$ORIGIN'
```

### Issue 3: Symbol Conflicts

Multiple definitions of the same symbol.

**Solution:**
```bash
# Use namespaces in code
# Or use static linking with --whole-archive carefully
```

---

## Best Practices

### 1. Choose Based on Use Case

**Use static linking for:**
- Embedded systems
- Standalone tools
- Simple deployment requirements

**Use dynamic linking for:**
- Desktop applications
- Server software
- Plugin systems

### 2. Version Your Libraries

```bash
# Linux shared library versioning
g++ -shared -fPIC -Wl,-soname,libmylib.so.1 mylib.cpp -o libmylib.so.1.0.0
ln -s libmylib.so.1.0.0 libmylib.so.1
ln -s libmylib.so.1 libmylib.so
```

### 3. Use RPATH for Portability

```bash
# Embed library search path
g++ main.cpp -L. -lmylib -Wl,-rpath,'$ORIGIN/lib'
```

### 4. Check Dependencies

```bash
# Linux
ldd ./app

# macOS
otool -L ./app

# Windows
dumpbin /DEPENDENTS app.exe
```

### 5. Consider Hybrid Approach

```bash
# Static link custom libraries, dynamic link system libraries
g++ main.cpp -static-libstdc++ -lmylib -o app
```

---

## Conclusion

Understanding linking is essential for C++ development. Key takeaways:

- **Static linking**: Self-contained, larger binaries, simpler deployment
- **Dynamic linking**: Smaller binaries, shared libraries, complex deployment
- **Choose wisely**: Based on deployment requirements and use case
- **Version carefully**: Avoid DLL hell with proper versioning
- **Test thoroughly**: Verify dependencies on target systems

The right linking strategy depends on your project's requirements, deployment environment, and maintenance considerations.

