# C++ Compiler Optimization Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Optimization Levels](#optimization-levels)
3. [Common Optimization Flags](#common-optimization-flags)
4. [Optimization Techniques](#optimization-techniques)
5. [Profile-Guided Optimization](#profile-guided-optimization)
6. [Link-Time Optimization](#link-time-optimization)
7. [Platform-Specific Optimizations](#platform-specific-optimizations)
8. [Measuring Performance](#measuring-performance)
9. [Best Practices](#best-practices)

---

## Introduction

Compiler optimization transforms your code to run faster and use less memory while preserving the original behavior. Understanding optimization options helps you balance compilation time, binary size, and runtime performance.

**Key Trade-offs:**
- Compilation time vs runtime performance
- Binary size vs execution speed
- Debuggability vs optimization level

---

## Optimization Levels

### GCC/Clang Optimization Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| **-O0** | No optimization (default) | Development, debugging |
| **-O1** | Basic optimization | Quick builds with some optimization |
| **-O2** | Moderate optimization | Production builds (recommended) |
| **-O3** | Aggressive optimization | Performance-critical code |
| **-Os** | Optimize for size | Embedded systems, small binaries |
| **-Ofast** | -O3 + fast math | Scientific computing (may break standards) |
| **-Og** | Optimize for debugging | Debug builds with some optimization |

### Examples

```bash
# No optimization - fastest compilation
g++ -O0 main.cpp -o app

# Standard production build
g++ -O2 main.cpp -o app

# Maximum performance
g++ -O3 main.cpp -o app

# Smallest binary
g++ -Os main.cpp -o app
```

---

## Common Optimization Flags

### Inlining Control

```bash
# Enable function inlining
g++ -O2 -finline-functions main.cpp

# Aggressive inlining
g++ -O3 -finline-limit=1000 main.cpp

# Disable inlining
g++ -O2 -fno-inline main.cpp
```

### Loop Optimizations

```bash
# Enable loop unrolling
g++ -O2 -funroll-loops main.cpp

# Vectorization (SIMD)
g++ -O3 -ftree-vectorize -march=native main.cpp
```

### Architecture-Specific

```bash
# Optimize for current CPU
g++ -O3 -march=native main.cpp

# Target specific architecture
g++ -O3 -march=skylake main.cpp
g++ -O3 -march=armv8-a main.cpp
```

---

## Optimization Techniques

### 1. Function Inlining

Replaces function calls with the function body to eliminate call overhead.

```cpp
// Before optimization
inline int add(int a, int b) { return a + b; }
int result = add(5, 3);

// After inlining
int result = 5 + 3;
```

### 2. Loop Unrolling

Reduces loop overhead by executing multiple iterations per loop cycle.

```cpp
// Original
for (int i = 0; i < 4; i++) {
    array[i] = i * 2;
}

// Unrolled
array[0] = 0 * 2;
array[1] = 1 * 2;
array[2] = 2 * 2;
array[3] = 3 * 2;
```

### 3. Dead Code Elimination

Removes code that doesn't affect program output.

```cpp
int unused = 42;  // Removed if never used
if (false) {      // Entire block removed
    doSomething();
}
```

---

## Profile-Guided Optimization

PGO uses runtime profiling data to guide optimization decisions.

### Step 1: Build with Instrumentation

```bash
g++ -O2 -fprofile-generate main.cpp -o app
```

### Step 2: Run with Representative Data

```bash
./app < typical_input.txt
# Generates .gcda files
```

### Step 3: Rebuild with Profile Data

```bash
g++ -O2 -fprofile-use main.cpp -o app_optimized
```

### Benefits

- Better branch prediction
- Improved code layout
- Optimized hot paths
- 10-30% performance improvement typical

---

## Link-Time Optimization

LTO performs optimization across translation units during linking.

### Enable LTO

```bash
# GCC/Clang
g++ -O2 -flto main.cpp utils.cpp -o app

# With parallel LTO
g++ -O2 -flto=4 main.cpp utils.cpp -o app
```

### Benefits

- Cross-file inlining
- Better dead code elimination
- Smaller binaries
- 5-15% performance improvement

---

## Platform-Specific Optimizations

### x86/x86-64

```bash
# Use SSE/AVX instructions
g++ -O3 -msse4.2 -mavx2 main.cpp

# Optimize for specific CPU
g++ -O3 -march=haswell main.cpp
g++ -O3 -march=znver2 main.cpp  # AMD Zen 2
```

### ARM

```bash
# ARM64 with NEON
g++ -O3 -march=armv8-a+simd main.cpp

# ARM Cortex-A53
g++ -O3 -mcpu=cortex-a53 main.cpp
```

### RISC-V

```bash
g++ -O3 -march=rv64gc main.cpp
```

---

## Measuring Performance

### Using time Command

```bash
time ./app
# Output: real 0m1.234s
```

### Using perf (Linux)

```bash
perf stat ./app
# Shows CPU cycles, cache misses, etc.
```

### Comparing Binaries

```bash
# Check binary size
ls -lh app_O2 app_O3

# Compare execution time
hyperfine './app_O2' './app_O3'
```

---

## Best Practices

### 1. Start with -O2

```bash
# Good default for production
g++ -O2 -Wall -Wextra main.cpp -o app
```

### 2. Use -O3 Carefully

```bash
# Only for performance-critical code
# Test thoroughly - may increase binary size
g++ -O3 main.cpp -o app
```

### 3. Enable Warnings

```bash
# Catch potential issues
g++ -O2 -Wall -Wextra -Werror main.cpp -o app
```

### 4. Profile Before Optimizing

```bash
# Find bottlenecks first
perf record ./app
perf report
```

### 5. Separate Debug and Release Builds

```bash
# Debug build - no optimization
g++ -O0 -g -DDEBUG main.cpp -o app_debug

# Release build - optimized
g++ -O2 -DNDEBUG main.cpp -o app_release
```

### 6. Consider LTO for Final Builds

```bash
# Production build with LTO
g++ -O2 -flto main.cpp utils.cpp -o app
```

### 7. Test Optimized Code Thoroughly

Optimization can expose bugs:
- Race conditions
- Undefined behavior
- Floating-point precision issues

---

## Conclusion

Compiler optimization is a powerful tool for improving performance. Key takeaways:

- **Use -O2 for production** - best balance of speed and safety
- **Profile before optimizing** - measure to find real bottlenecks
- **Test thoroughly** - optimization can expose hidden bugs
- **Consider PGO and LTO** - for maximum performance gains
- **Match target architecture** - use -march=native when appropriate

Remember: premature optimization is the root of all evil. Write clear code first, then optimize based on profiling data.

