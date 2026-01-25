# Windows DLL Development Guide

## Table of Contents
1. [Introduction](#introduction)
2. [DLL Basics](#dll-basics)
3. [Creating a DLL](#creating-a-dll)
4. [Using a DLL](#using-a-dll)
5. [Export/Import Mechanisms](#exportimport-mechanisms)
6. [DLL Entry Point](#dll-entry-point)
7. [Static vs Dynamic Loading](#static-vs-dynamic-loading)
8. [Best Practices](#best-practices)

---

## Introduction

Dynamic Link Libraries (DLLs) are Windows' implementation of shared libraries. This guide covers DLL creation, usage, and Windows-specific mechanisms like `__declspec(dllexport/dllimport)`.

**Key Concepts:**
- DLL exports functions/classes for use by other programs
- Import libraries (.lib) link against DLLs at compile time
- DLLs can be loaded statically (at startup) or dynamically (at runtime)

---

## DLL Basics

### What is a DLL?

A DLL is a library that contains code and data that can be used by multiple programs simultaneously.

**Benefits:**
- ✅ Code reuse across applications
- ✅ Smaller executable sizes
- ✅ Easy updates (replace DLL without recompiling)
- ✅ Memory sharing between processes

**Components:**
- `.dll` file - The actual library
- `.lib` file - Import library for linking
- `.h` file - Header with declarations

### DLL vs Static Library

| Aspect | DLL | Static Library (.lib) |
|--------|-----|----------------------|
| **Linking** | Runtime | Compile time |
| **Size** | Smaller executables | Larger executables |
| **Updates** | Replace DLL | Recompile needed |
| **Dependencies** | Requires DLL at runtime | Self-contained |
| **Memory** | Shared across processes | Duplicated per process |

---

## Creating a DLL

### Step 1: Define API Macro

**mylib.h:**

```cpp
#ifndef MYLIB_H
#define MYLIB_H

// Define export/import macro
#ifdef MYLIB_EXPORTS
    #define MYLIB_API __declspec(dllexport)
#else
    #define MYLIB_API __declspec(dllimport)
#endif

// Export functions
MYLIB_API int add(int a, int b);
MYLIB_API int multiply(int a, int b);

// Export class
class MYLIB_API Calculator {
public:
    int add(int a, int b);
    int subtract(int a, int b);
};

#endif
```

### Step 2: Implement Functions

**mylib.cpp:**

```cpp
#include "mylib.h"

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int Calculator::add(int a, int b) {
    return a + b;
}

int Calculator::subtract(int a, int b) {
    return a - b;
}
```

---

### Step 3: Build the DLL

```cmd
REM Compile with MYLIB_EXPORTS defined
cl /LD /DMYLIB_EXPORTS mylib.cpp /Fe:mylib.dll

REM This creates:
REM - mylib.dll (the library)
REM - mylib.lib (import library)
REM - mylib.exp (export file)
```

**Compiler flags:**
- `/LD` - Create DLL
- `/DMYLIB_EXPORTS` - Define macro for dllexport

---

## Using a DLL

### Static Loading (Implicit Linking)

Link against the import library at compile time.

**main.cpp:**

```cpp
#include "mylib.h"
#include <iostream>

int main() {
    // Use exported functions
    std::cout << "5 + 3 = " << add(5, 3) << std::endl;

    // Use exported class
    Calculator calc;
    std::cout << "10 - 4 = " << calc.subtract(10, 4) << std::endl;

    return 0;
}
```

**Build:**

```cmd
REM Compile and link with import library
cl main.cpp mylib.lib

REM Run (mylib.dll must be in PATH or same directory)
main.exe
```

---

## Export/Import Mechanisms

### __declspec(dllexport/dllimport)

Windows-specific keywords for controlling symbol visibility.

```cpp
// When building DLL
__declspec(dllexport) int add(int a, int b);

// When using DLL
__declspec(dllimport) int add(int a, int b);
```

### Why Use Both?

**dllexport:**
- Tells compiler to export symbol from DLL
- Creates entry in export table
- Generates import library (.lib)

**dllimport:**
- Optimizes function calls (direct vs indirect)
- Reduces code size
- Improves performance

### Module Definition File (.def)

Alternative to __declspec for exporting functions.

**mylib.def:**

```
LIBRARY mylib
EXPORTS
    add
    multiply
```

**Build with .def:**

```cmd
cl /LD mylib.cpp /DEF:mylib.def
```

---

## DLL Entry Point

### DllMain Function

Optional entry point called when DLL is loaded/unloaded.

```cpp
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD  ul_reason_for_call,
                      LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        // DLL loaded by process
        // Initialize resources
        break;
    case DLL_THREAD_ATTACH:
        // New thread created
        break;
    case DLL_THREAD_DETACH:
        // Thread exiting
        break;
    case DLL_PROCESS_DETACH:
        // DLL unloaded by process
        // Cleanup resources
        break;
    }
    return TRUE;
}
```

### When to Use DllMain

**Use for:**
- Resource initialization
- Thread-local storage setup
- Cleanup on unload

**Avoid:**
- Heavy computation
- Loading other DLLs
- Synchronization (can cause deadlocks)

---

## Static vs Dynamic Loading

### Static Loading (Load-Time)

DLL loaded automatically when program starts.

**Advantages:**
- ✅ Simple to use
- ✅ Automatic error handling
- ✅ Type-safe function calls

**Disadvantages:**
- ❌ Program won't start if DLL missing
- ❌ All DLLs loaded at startup

### Dynamic Loading (Run-Time)

Load DLL explicitly using Windows API.

```cpp
#include <windows.h>
#include <iostream>

int main() {
    // Load DLL at runtime
    HMODULE hDll = LoadLibrary("mylib.dll");
    if (!hDll) {
        std::cerr << "Failed to load DLL" << std::endl;
        return 1;
    }

    // Get function pointer
    typedef int (*AddFunc)(int, int);
    AddFunc add = (AddFunc)GetProcAddress(hDll, "add");

    if (!add) {
        std::cerr << "Failed to find function" << std::endl;
        FreeLibrary(hDll);
        return 1;
    }

    // Use function
    std::cout << "5 + 3 = " << add(5, 3) << std::endl;

    // Unload DLL
    FreeLibrary(hDll);
    return 0;
}
```

**Advantages:**
- ✅ Program runs even if DLL missing
- ✅ Load DLLs on demand
- ✅ Plugin architecture support

**Disadvantages:**
- ❌ More complex code
- ❌ No type safety
- ❌ Manual error handling

---

## Best Practices

### 1. Use Export/Import Macro Pattern

```cpp
#ifdef MYLIB_EXPORTS
    #define MYLIB_API __declspec(dllexport)
#else
    #define MYLIB_API __declspec(dllimport)
#endif
```

### 2. Version Your DLLs

Include version information in DLL name or use versioning APIs.

```cpp
// mylib_v1.dll, mylib_v2.dll
// Or use GetFileVersionInfo API
```

### 3. Minimize DllMain Code

Keep DllMain simple to avoid loader lock issues.

### 4. Handle Missing DLLs Gracefully

```cpp
HMODULE hDll = LoadLibrary("optional.dll");
if (!hDll) {
    // Provide fallback or disable feature
    std::cout << "Optional feature not available" << std::endl;
}
```

### 5. Use Delay Loading

Load DLLs only when functions are first called.

```cmd
REM Link with delay load
cl main.cpp mylib.lib /DELAYLOAD:mylib.dll delayimp.lib
```

### 6. Check DLL Dependencies

```cmd
REM Use dumpbin to check dependencies
dumpbin /DEPENDENTS myapp.exe
```

---

## Conclusion

Windows DLL development requires understanding export/import mechanisms and loading strategies. Key takeaways:

- **Use __declspec(dllexport/dllimport)** - Standard Windows approach
- **Choose loading strategy** - Static for simplicity, dynamic for flexibility
- **Keep DllMain minimal** - Avoid complex initialization
- **Version your DLLs** - Prevent compatibility issues
- **Handle missing DLLs** - Graceful degradation

Understanding DLL mechanics enables building modular, maintainable Windows applications.

