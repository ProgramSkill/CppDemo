# Precompiled Headers (PCH) Guide

## Table of Contents
1. [Introduction](#introduction)
2. [What are Precompiled Headers?](#what-are-precompiled-headers)
3. [Benefits and Trade-offs](#benefits-and-trade-offs)
4. [GCC/Clang PCH](#gccclang-pch)
5. [MSVC PCH](#msvc-pch)
6. [CMake Integration](#cmake-integration)
7. [Best Practices](#best-practices)
8. [Common Issues](#common-issues)

---

## Introduction

Precompiled headers (PCH) are a compilation optimization technique that speeds up build times by pre-processing frequently used header files once and reusing the result.

**Typical speedup:** 30-70% reduction in compilation time for large projects.

---

## What are Precompiled Headers?

PCH works by:
1. Compiling commonly used headers once into a binary format
2. Reusing this precompiled data for subsequent compilations
3. Skipping the parsing and preprocessing of these headers

### Typical PCH Contents

```cpp
// pch.h or stdafx.h
#pragma once

// Standard library headers
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// Third-party libraries
#include <boost/algorithm/string.hpp>
#include <fmt/format.h>

// Project-wide headers
#include "common/types.h"
#include "common/utils.h"
```

---

## Benefits and Trade-offs

### Benefits

✅ **Faster compilation** - 30-70% speedup for large projects
✅ **Reduced redundancy** - Parse common headers once
✅ **Better IDE performance** - Faster IntelliSense/code completion
✅ **Consistent includes** - Standardized header inclusion

### Trade-offs

❌ **Initial overhead** - First compilation is slower
❌ **Disk space** - PCH files can be large (100MB+)
❌ **Maintenance** - Must keep PCH up to date
❌ **Rebuild triggers** - Changing PCH forces full rebuild

---

## GCC/Clang PCH

### Creating PCH

**Step 1: Create header file** (`pch.h`):
```cpp
#pragma once
#include <iostream>
#include <vector>
#include <string>
```

**Step 2: Precompile the header:**
```bash
# GCC
g++ -std=c++17 pch.h

# Clang
clang++ -std=c++17 pch.h
```

This generates `pch.h.gch` (GCC) or `pch.h.pch` (Clang).

**Step 3: Use in source files:**
```cpp
// main.cpp
#include "pch.h"  // Must be first include

int main() {
    std::vector<std::string> data = {"hello", "world"};
    return 0;
}
```

**Step 4: Compile:**
```bash
g++ -std=c++17 main.cpp -o app
# Automatically uses pch.h.gch if present
```

---

## MSVC PCH

### Creating PCH

**Step 1: Create header** (`pch.h`):
```cpp
#pragma once
#include <iostream>
#include <vector>
#include <string>
```

**Step 2: Create implementation** (`pch.cpp`):
```cpp
#include "pch.h"
// Empty file, just includes the header
```

**Step 3: Configure project:**
```cmd
# Compile PCH
cl /c /Ycpch.h pch.cpp

# Compile source files using PCH
cl /c /Yupch.h main.cpp

# Link
cl main.obj pch.obj
```

### Visual Studio Settings

- Right-click `pch.cpp` → Properties → Precompiled Headers → **Create (/Yc)**
- Right-click project → Properties → Precompiled Headers → **Use (/Yu)**

---

## CMake Integration

### Modern CMake (3.16+)

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyProject)

add_executable(myapp main.cpp utils.cpp)

# Enable PCH
target_precompile_headers(myapp PRIVATE pch.h)
```

### With Custom PCH Header

```cmake
# Create PCH header content
target_precompile_headers(myapp PRIVATE
    <iostream>
    <vector>
    <string>
    <memory>
)
```

### Reusing PCH Across Targets

```cmake
add_library(common_pch INTERFACE)
target_precompile_headers(common_pch INTERFACE pch.h)

add_executable(app1 main1.cpp)
target_link_libraries(app1 PRIVATE common_pch)

add_executable(app2 main2.cpp)
target_link_libraries(app2 PRIVATE common_pch)
```

---

## Best Practices

### 1. Include Only Stable Headers

**Good:**
```cpp
// pch.h
#include <iostream>      // Standard library - stable
#include <vector>        // Standard library - stable
#include "config.h"      // Rarely changes
```

**Bad:**
```cpp
// pch.h
#include "feature.h"     // Changes frequently
#include "temp.h"        // Temporary code
```

### 2. PCH Must Be First Include

```cpp
// main.cpp
#include "pch.h"  // MUST be first
#include "other.h"
```

### 3. Keep PCH Small

Only include headers used by 70%+ of source files.

### 4. One PCH Per Project

Avoid multiple PCH files - adds complexity.

---

## Common Issues

### Issue 1: PCH Not Found

**Error:**
```
fatal error: pch.h: No such file or directory
```

**Solution:**
```bash
# Ensure PCH is in include path
g++ -I. -std=c++17 pch.h
g++ -I. -std=c++17 main.cpp -o app
```

### Issue 2: PCH Out of Date

**Error:**
```
error: pch.h.gch: created with different compiler version
```

**Solution:**
```bash
# Rebuild PCH
rm pch.h.gch
g++ -std=c++17 pch.h
```

### Issue 3: Full Rebuild on PCH Change

Changing PCH forces recompilation of all files.

**Mitigation:** Keep PCH stable, only include rarely-changing headers.

---

## Conclusion

Precompiled headers significantly speed up compilation for large projects. Key takeaways:

- **Use for stable headers** - Standard library, third-party libs
- **Keep PCH small** - Only frequently used headers
- **PCH must be first** - Include before any other headers
- **CMake 3.16+** - Use `target_precompile_headers()` for easy setup
- **Trade-off** - Faster builds vs rebuild cost when PCH changes

For projects with 50+ source files, PCH typically provides 30-70% compilation speedup.

