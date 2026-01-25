# C++ Dependency & Package Management Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Why Dependency Management Matters](#why-dependency-management-matters)
3. [Conan Package Manager](#conan-package-manager)
4. [vcpkg Package Manager](#vcpkg-package-manager)
5. [CMake FetchContent](#cmake-fetchcontent)
6. [Git Submodules](#git-submodules)
7. [Comparison](#comparison)
8. [Best Practices](#best-practices)

---

## Introduction

Managing third-party dependencies is crucial for modern C++ projects. This guide covers the main tools and strategies for dependency management.

---

## Why Dependency Management Matters

### Problems Without Dependency Management

- Manual library downloads
- Version conflicts
- Platform-specific builds
- Difficult updates
- No reproducible builds

### Benefits of Dependency Management

✅ **Automated downloads** - Fetch libraries automatically
✅ **Version control** - Specify exact versions
✅ **Reproducible builds** - Same dependencies everywhere
✅ **Easy updates** - Update with single command
✅ **Conflict resolution** - Handle version conflicts

---

## Conan Package Manager

Conan is a decentralized C++ package manager with a large repository.

### Installation

```bash
pip install conan
conan --version
```

### Basic Usage

**Step 1: Create conanfile.txt**
```ini
[requires]
fmt/9.1.0
spdlog/1.11.0

[generators]
CMakeDeps
CMakeToolchain
```

**Step 2: Install dependencies**
```bash
conan install . --build=missing
```

**Step 3: Use in CMakeLists.txt**
```cmake
cmake_minimum_required(VERSION 3.15)
project(MyApp)

find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

add_executable(myapp main.cpp)
target_link_libraries(myapp fmt::fmt spdlog::spdlog)
```

### Advanced Conan Features

**Conanfile.py (Python-based):**
```python
from conan import ConanFile

class MyAppConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    requires = "fmt/9.1.0", "spdlog/1.11.0"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        if self.settings.os == "Windows":
            self.requires("winsdk/10.0")
```

**Version ranges:**
```ini
[requires]
fmt/[>=9.0.0 <10.0.0]
```

---

## vcpkg Package Manager

vcpkg is Microsoft's cross-platform package manager, integrated with Visual Studio.

### Installation

```bash
# Clone repository
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Bootstrap
./bootstrap-vcpkg.sh  # Linux/macOS
.\bootstrap-vcpkg.bat  # Windows
```

### Basic Usage

```bash
# Install packages
./vcpkg install fmt spdlog

# List installed packages
./vcpkg list

# Search packages
./vcpkg search boost
```

### CMake Integration

```bash
# Method 1: Toolchain file
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ..

# Method 2: Manifest mode (vcpkg.json)
```

**vcpkg.json:**
```json
{
  "dependencies": [
    "fmt",
    "spdlog"
  ]
}
```

**CMakeLists.txt:**
```cmake
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

target_link_libraries(myapp PRIVATE fmt::fmt spdlog::spdlog)
```

---

## CMake FetchContent

FetchContent downloads dependencies at configure time (CMake 3.11+).

### Basic Usage

```cmake
include(FetchContent)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        9.1.0
)

FetchContent_MakeAvailable(fmt)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE fmt::fmt)
```

### Multiple Dependencies

```cmake
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.2
)

FetchContent_MakeAvailable(googletest json)
```

### Advantages

✅ No external tools required
✅ Integrated with CMake
✅ Source-based (always builds from source)

### Disadvantages

❌ Slower configure time
❌ No binary caching
❌ Downloads every time

---

## Git Submodules

Git submodules embed external repositories as subdirectories.

### Adding Submodules

```bash
# Add submodule
git submodule add https://github.com/fmtlib/fmt.git external/fmt

# Initialize and update
git submodule update --init --recursive
```

### Using in CMake

```cmake
add_subdirectory(external/fmt)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE fmt::fmt)
```

### Advantages

✅ Version controlled with your project
✅ No external tools needed
✅ Works offline after initial clone

### Disadvantages

❌ Complex Git workflow
❌ Easy to forget updates
❌ Nested submodules can be tricky

---

## Comparison

| Feature | Conan | vcpkg | FetchContent | Submodules |
|---------|-------|-------|--------------|------------|
| **Binary caching** | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| **Setup complexity** | Medium | Low | Very Low | Low |
| **Build speed** | Fast | Fast | Slow | Medium |
| **Version control** | External | External | External | In repo |
| **Offline support** | Cache only | Cache only | ❌ No | ✅ Yes |
| **Cross-platform** | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Good |
| **Package count** | ~1000 | ~2000 | Any Git | Any Git |

---

## Best Practices

### 1. Choose the Right Tool

**Use Conan for:**
- Large projects with many dependencies
- Need for binary caching
- Cross-platform development

**Use vcpkg for:**
- Windows-focused projects
- Visual Studio integration
- Microsoft ecosystem

**Use FetchContent for:**
- Small projects
- Few dependencies
- Simple setup

**Use Submodules for:**
- Full control over dependencies
- Offline development
- Vendoring dependencies

### 2. Lock Dependency Versions

```ini
# Conan - use exact versions
[requires]
fmt/9.1.0
spdlog/1.11.0
```

```json
// vcpkg - use baseline
{
  "dependencies": ["fmt", "spdlog"],
  "builtin-baseline": "a1b2c3d4..."
}
```

### 3. Document Dependencies

Create a README documenting:
- Required dependencies
- Installation instructions
- Version requirements

### 4. CI/CD Integration

```yaml
# GitHub Actions example
- name: Install Conan
  run: pip install conan

- name: Install dependencies
  run: conan install . --build=missing

- name: Build
  run: cmake --build build
```

### 5. Avoid Mixing Tools

Stick to one dependency manager per project to avoid conflicts.

---

## Conclusion

Modern C++ projects need robust dependency management. Key takeaways:

- **Conan** - Best for large, cross-platform projects
- **vcpkg** - Best for Windows and Visual Studio
- **FetchContent** - Best for simple projects
- **Submodules** - Best for full control

Choose based on your project needs, team expertise, and ecosystem.

