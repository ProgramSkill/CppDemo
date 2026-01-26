# 跨平台 C++ 兼容性指南
# Cross-Platform C++ Compatibility Guide

## 目录 (Table of Contents)

1. [平台差异概述](#平台差异概述-platform-differences-overview)
2. [完全兼容的特性](#完全兼容的特性-fully-compatible-features)
3. [不兼容的特性](#不兼容的特性-incompatible-features)
4. [文件系统差异](#文件系统差异-file-system-differences)
5. [进程和线程](#进程和线程-processes-and-threads)
6. [网络编程](#网络编程-network-programming)
7. [处理平台差异的方法](#处理平台差异的方法-handling-platform-differences)
8. [最佳实践](#最佳实践-best-practices)

---

## 平台差异概述 (Platform Differences Overview)

跨平台开发的核心挑战是处理不同操作系统之间的差异。本文档详细说明哪些特性可以跨平台使用，哪些需要特殊处理。

**主要平台对比：**
- **Windows** - 使用 Windows API，路径分隔符 `\`
- **Linux** - 使用 POSIX API，路径分隔符 `/`
- **macOS** - 基于 BSD，使用 POSIX API，路径分隔符 `/`

---

## 完全兼容的特性 (Fully Compatible Features)

以下特性在 Windows、Linux、macOS 上完全兼容，可以直接使用，无需修改。

### ✅ 标准 C++ 语言特性

**所有 C++ 标准特性都是跨平台的：**

| 特性类别 | 说明 | 示例 |
|---------|------|------|
| **基本语法** | 类、函数、模板、继承等 | `class MyClass { };` |
| **STL 容器** | vector, map, set, list 等 | `std::vector<int> v;` |
| **STL 算法** | sort, find, transform 等 | `std::sort(v.begin(), v.end());` |
| **智能指针** | unique_ptr, shared_ptr | `std::unique_ptr<int> p;` |
| **Lambda 表达式** | C++11 及以后 | `[](int x) { return x * 2; }` |
| **异常处理** | try-catch | `try { } catch(...) { }` |
| **RTTI** | typeid, dynamic_cast | `typeid(obj).name()` |

**示例代码（完全跨平台）：**

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>

class DataProcessor {
private:
    std::vector<int> data;

public:
    void addData(int value) {
        data.push_back(value);
    }

    void process() {
        // 使用 STL 算法
        std::sort(data.begin(), data.end());

        // 使用 Lambda
        std::for_each(data.begin(), data.end(),
            [](int n) { std::cout << n << " "; });
    }
};

int main() {
    // 使用智能指针
    auto processor = std::make_unique<DataProcessor>();
    processor->addData(5);
    processor->addData(2);
    processor->process();

    return 0;
}
// 这段代码在所有平台上都能正常编译运行
```

---

### ✅ 标准 C++ 库

**完全跨平台的标准库：**

| 库 | 功能 | 兼容性 |
|----|------|--------|
| `<iostream>` | 输入输出流 | ✅ 完全兼容 |
| `<fstream>` | 文件流 | ✅ 完全兼容 |
| `<string>` | 字符串处理 | ✅ 完全兼容 |
| `<vector>` | 动态数组 | ✅ 完全兼容 |
| `<map>` | 关联容器 | ✅ 完全兼容 |
| `<algorithm>` | 算法库 | ✅ 完全兼容 |
| `<thread>` | C++11 线程 | ✅ 完全兼容 |
| `<mutex>` | 互斥锁 | ✅ 完全兼容 |
| `<chrono>` | 时间库 | ✅ 完全兼容 |
| `<filesystem>` | C++17 文件系统 | ✅ 完全兼容 |
| `<regex>` | 正则表达式 | ✅ 完全兼容 |

**示例：使用标准库（跨平台）：**

```cpp
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>

int main() {
    // 文件操作（跨平台）
    std::ofstream file("data.txt");
    file << "Hello, Cross-Platform!" << std::endl;
    file.close();

    // 时间操作（跨平台）
    auto start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto end = std::chrono::steady_clock::now();

    std::cout << "Elapsed: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    return 0;
}
```

---

### ✅ C++17 文件系统库

**`<filesystem>` 是跨平台的救星：**

```cpp
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    // 自动处理路径分隔符差异
    fs::path dataPath = "data/config.txt";  // 在所有平台上都正确

    // 创建目录（跨平台）
    fs::create_directories("output/logs");

    // 遍历目录（跨平台）
    for (const auto& entry : fs::directory_iterator(".")) {
        std::cout << entry.path() << std::endl;
    }

    // 检查文件是否存在（跨平台）
    if (fs::exists("data.txt")) {
        std::cout << "File size: " << fs::file_size("data.txt") << " bytes" << std::endl;
    }

    return 0;
}
```

**优势：**
- ✅ 自动处理路径分隔符（`/` vs `\`）
- ✅ 统一的文件操作接口
- ✅ 无需平台特定代码

---

## 不兼容的特性 (Incompatible Features)

以下特性在不同平台上有显著差异，需要特殊处理或使用条件编译。

### ❌ 系统 API

**Windows API vs POSIX API：**

| 功能 | Windows | Linux/macOS | 兼容性 |
|------|---------|-------------|--------|
| **进程创建** | `CreateProcess()` | `fork()` + `exec()` | ❌ 完全不同 |
| **线程创建** | `CreateThread()` | `pthread_create()` | ❌ 完全不同 |
| **文件操作** | `CreateFile()` | `open()` | ❌ 完全不同 |
| **内存映射** | `CreateFileMapping()` | `mmap()` | ❌ 完全不同 |
| **套接字** | Winsock | BSD Sockets | ⚠️ 类似但有差异 |
| **动态库加载** | `LoadLibrary()` | `dlopen()` | ❌ 完全不同 |
| **睡眠函数** | `Sleep(ms)` | `sleep(s)` / `usleep(us)` | ❌ 参数单位不同 |

**示例：睡眠函数的平台差异**

```cpp
// ❌ 错误：直接使用平台特定 API
#ifdef _WIN32
    Sleep(1000);  // Windows: 毫秒
#else
    sleep(1);     // Linux: 秒
#endif

// ✅ 正确：使用 C++11 标准库
#include <thread>
#include <chrono>
std::this_thread::sleep_for(std::chrono::seconds(1));  // 跨平台
```

---

### ❌ 文件路径

**路径分隔符差异：**

| 平台 | 路径分隔符 | 示例 |
|------|-----------|------|
| **Windows** | `\` (反斜杠) | `C:\Users\data\file.txt` |
| **Linux/macOS** | `/` (正斜杠) | `/home/user/data/file.txt` |

**错误示例：**

```cpp
// ❌ 硬编码路径分隔符
std::string path = "data\\config.txt";  // 只在 Windows 上工作

// ❌ 使用平台特定路径
#ifdef _WIN32
    std::string path = "C:\\Program Files\\MyApp\\data.txt";
#else
    std::string path = "/usr/local/share/myapp/data.txt";
#endif
```

**正确示例：**

```cpp
// ✅ 使用 C++17 filesystem
#include <filesystem>
namespace fs = std::filesystem;

fs::path dataPath = fs::current_path() / "data" / "config.txt";
// 自动使用正确的分隔符

// ✅ 或者使用正斜杠（Windows 也支持）
std::string path = "data/config.txt";  // 在所有平台上都工作
```

---

### ❌ 动态库

**动态库文件扩展名和加载方式：**

| 平台 | 文件扩展名 | 加载函数 |
|------|-----------|---------|
| **Windows** | `.dll` | `LoadLibrary()` / `GetProcAddress()` |
| **Linux** | `.so` | `dlopen()` / `dlsym()` |
| **macOS** | `.dylib` | `dlopen()` / `dlsym()` |

**示例：跨平台动态库加载**

```cpp
#ifdef _WIN32
    #include <windows.h>
    HMODULE lib = LoadLibrary("mylib.dll");
    auto func = (FuncType)GetProcAddress(lib, "myFunction");
    FreeLibrary(lib);
#else
    #include <dlfcn.h>
    void* lib = dlopen("libmylib.so", RTLD_LAZY);
    auto func = (FuncType)dlsym(lib, "myFunction");
    dlclose(lib);
#endif
```

---

### ❌ 网络编程

**Winsock vs BSD Sockets：**

| 功能 | Windows (Winsock) | Linux/macOS (BSD Sockets) |
|------|------------------|--------------------------|
| **初始化** | 需要 `WSAStartup()` | 不需要初始化 |
| **清理** | 需要 `WSACleanup()` | 不需要清理 |
| **头文件** | `<winsock2.h>` | `<sys/socket.h>`, `<netinet/in.h>` |
| **套接字类型** | `SOCKET` (unsigned int) | `int` |
| **错误处理** | `WSAGetLastError()` | `errno` |
| **关闭套接字** | `closesocket()` | `close()` |
| **无效套接字** | `INVALID_SOCKET` | `-1` |

**错误示例：**

```cpp
// ❌ 只在 Windows 上工作
#include <winsock2.h>
SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
closesocket(sock);

// ❌ 只在 Linux 上工作
#include <sys/socket.h>
int sock = socket(AF_INET, SOCK_STREAM, 0);
close(sock);
```

**正确示例（跨平台）：**

```cpp
#ifdef _WIN32
    #include <winsock2.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET SocketType;
    #define CLOSE_SOCKET closesocket
    #define SOCKET_ERROR_CODE WSAGetLastError()
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <unistd.h>
    typedef int SocketType;
    #define CLOSE_SOCKET close
    #define SOCKET_ERROR_CODE errno
    #define INVALID_SOCKET -1
#endif

class NetworkManager {
public:
    NetworkManager() {
#ifdef _WIN32
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    }

    ~NetworkManager() {
#ifdef _WIN32
        WSACleanup();
#endif
    }

    SocketType createSocket() {
        return socket(AF_INET, SOCK_STREAM, 0);
    }

    void closeSocket(SocketType sock) {
        CLOSE_SOCKET(sock);
    }
};
```

---

### ❌ 进程和线程

**进程创建差异：**

| 功能 | Windows | Linux/macOS |
|------|---------|-------------|
| **进程创建** | `CreateProcess()` | `fork()` + `exec()` |
| **进程 ID** | `DWORD` | `pid_t` |
| **等待进程** | `WaitForSingleObject()` | `waitpid()` |
| **线程创建** | `CreateThread()` | `pthread_create()` |

**错误示例：**

```cpp
// ❌ Windows 特定
#include <windows.h>
STARTUPINFO si;
PROCESS_INFORMATION pi;
CreateProcess(NULL, "app.exe", NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi);

// ❌ Linux 特定
#include <unistd.h>
pid_t pid = fork();
if (pid == 0) {
    execl("/bin/app", "app", NULL);
}
```

**正确示例（使用 C++11 标准库）：**

```cpp
#include <thread>
#include <iostream>

void workerFunction() {
    std::cout << "Worker thread running" << std::endl;
}

int main() {
    // 创建线程（跨平台）
    std::thread worker(workerFunction);

    // 等待线程完成
    worker.join();

    return 0;
}
// 这段代码在所有平台上都能正常工作
```

---

### ❌ 控制台颜色输出

**ANSI 转义码支持差异：**

| 平台 | ANSI 转义码支持 | 说明 |
|------|---------------|------|
| **Linux/macOS** | ✅ 原生支持 | 直接使用 `\033[31m` 等 |
| **Windows 10+** | ✅ 需要启用 | 需要启用虚拟终端处理 |
| **Windows 7/8** | ❌ 不支持 | 需要使用 Windows API |

**跨平台控制台颜色示例：**

```cpp
#ifdef _WIN32
    #include <windows.h>

    void enableANSI() {
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD dwMode = 0;
        GetConsoleMode(hOut, &dwMode);
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, dwMode);
    }
#else
    void enableANSI() {
        // Linux/macOS 原生支持，无需操作
    }
#endif

int main() {
    enableANSI();

    // 使用 ANSI 转义码（跨平台）
    std::cout << "\033[31m红色文本\033[0m" << std::endl;
    std::cout << "\033[32m绿色文本\033[0m" << std::endl;
    std::cout << "\033[34m蓝色文本\033[0m" << std::endl;

    return 0;
}
```

---

## 文件系统差异 (File System Differences)

### 路径分隔符

**问题：**
- Windows 使用反斜杠 `\`
- Linux/macOS 使用正斜杠 `/`

**解决方案：**

```cpp
// ✅ 方案 1：使用 C++17 filesystem（推荐）
#include <filesystem>
namespace fs = std::filesystem;

fs::path configPath = fs::current_path() / "config" / "settings.ini";
// 自动使用正确的分隔符

// ✅ 方案 2：使用正斜杠（Windows 也支持）
std::string path = "data/config.txt";  // 在所有平台上都工作

// ❌ 错误：硬编码反斜杠
std::string path = "data\\config.txt";  // 只在 Windows 上工作
```

### 行尾符差异

**问题：**
- Windows: `\r\n` (CRLF)
- Linux/macOS: `\n` (LF)

**解决方案：**

```cpp
// 使用文本模式打开文件，自动处理行尾符
std::ifstream file("data.txt");  // 文本模式，自动转换

// 或者使用二进制模式并手动处理
std::ifstream file("data.txt", std::ios::binary);
std::string line;
while (std::getline(file, line)) {
    // 移除可能的 \r
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}
```

---

## 处理平台差异的方法 (Handling Platform Differences)

### 1. 优先使用标准 C++ 库

**原则：能用标准库就不用平台特定 API**

```cpp
// ✅ 优先选择
#include <thread>          // 线程
#include <mutex>           // 互斥锁
#include <chrono>          // 时间
#include <filesystem>      // 文件系统
#include <random>          // 随机数

// ❌ 避免直接使用
// Windows API, POSIX API
```

**优势：**
- ✅ 代码可移植性强
- ✅ 无需条件编译
- ✅ 编译器优化更好
- ✅ 维护成本低

---

### 2. 使用条件编译

**平台检测宏：**

```cpp
// 检测操作系统
#ifdef _WIN32
    // Windows (32位和64位)
#elif __linux__
    // Linux
#elif __APPLE__
    // macOS
#elif __unix__
    // Unix-like 系统
#endif

// 检测编译器
#ifdef _MSC_VER
    // Microsoft Visual C++
#elif __GNUC__
    // GCC
#elif __clang__
    // Clang
#endif

// 检测架构
#if defined(_M_X64) || defined(__x86_64__)
    // 64位 x86
#elif defined(_M_ARM64) || defined(__aarch64__)
    // 64位 ARM
#endif
```

**条件编译示例：**

```cpp
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
    #define PATH_SEPARATOR '\\'
#else
    #define EXPORT __attribute__((visibility("default")))
    #define PATH_SEPARATOR '/'
#endif

class EXPORT MyClass {
    // ...
};
```

---

### 3. 使用抽象层

**创建平台抽象接口：**

```cpp
// Platform.h
class Platform {
public:
    static void sleep(int milliseconds);
    static std::string getExecutablePath();
    static void setConsoleColor(int color);
};

// Platform_Windows.cpp
#ifdef _WIN32
void Platform::sleep(int ms) {
    Sleep(ms);
}

std::string Platform::getExecutablePath() {
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    return std::string(buffer);
}
#endif

// Platform_Linux.cpp
#ifdef __linux__
void Platform::sleep(int ms) {
    usleep(ms * 1000);
}

std::string Platform::getExecutablePath() {
    char buffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1) {
        buffer[len] = '\0';
        return std::string(buffer);
    }
    return "";
}
#endif

// 使用时无需关心平台
int main() {
    Platform::sleep(1000);  // 跨平台
    std::cout << "Executable: " << Platform::getExecutablePath() << std::endl;
    return 0;
}
```

---

### 4. 使用跨平台库

**推荐的跨平台库：**

| 库 | 功能 | 优势 |
|----|------|------|
| **Boost** | 通用工具库 | 功能丰富，准标准库 |
| **Qt** | GUI + 工具库 | 完整的应用框架 |
| **POCO** | 网络 + 工具 | 轻量级，易学 |
| **SDL2** | 多媒体 | 游戏开发 |
| **SFML** | 多媒体 | 简单易用 |
| **wxWidgets** | GUI | 原生外观 |

**示例：使用 Boost.Filesystem（C++17 之前）：**

```cpp
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

int main() {
    fs::path dataPath = fs::current_path() / "data" / "config.txt";

    if (fs::exists(dataPath)) {
        std::cout << "File size: " << fs::file_size(dataPath) << std::endl;
    }

    // 遍历目录
    for (const auto& entry : fs::directory_iterator(".")) {
        std::cout << entry.path() << std::endl;
    }

    return 0;
}
// 完全跨平台，无需条件编译
```

---

## 最佳实践 (Best Practices)

### 1. 从一开始就考虑跨平台

**不要等到移植时才处理平台差异**

```cpp
// ❌ 错误：先写 Windows 代码，后期再移植
#include <windows.h>
Sleep(1000);

// ✅ 正确：从一开始就使用跨平台代码
#include <thread>
#include <chrono>
std::this_thread::sleep_for(std::chrono::seconds(1));
```

---

### 2. 最小化平台特定代码

**将平台特定代码隔离到单独的文件或函数中**

```
project/
├── src/
│   ├── main.cpp           # 跨平台代码
│   ├── platform/
│   │   ├── platform.h     # 平台抽象接口
│   │   ├── windows.cpp    # Windows 实现
│   │   └── linux.cpp      # Linux 实现
```

---

### 3. 在所有目标平台上测试

**不要假设代码在其他平台上能正常工作**

- ✅ 使用 CI/CD 在多个平台上自动测试
- ✅ 定期在真实硬件上测试
- ✅ 测试边界情况和错误处理

---

### 4. 使用统一的构建系统

**推荐使用 CMake 进行跨平台构建**

```cmake
cmake_minimum_required(VERSION 3.17)
project(MyApp)

set(CMAKE_CXX_STANDARD 17)

# 跨平台源文件
add_executable(myapp
    src/main.cpp
    src/core.cpp
)

# 平台特定源文件
if(WIN32)
    target_sources(myapp PRIVATE src/platform/windows.cpp)
elseif(UNIX)
    target_sources(myapp PRIVATE src/platform/linux.cpp)
endif()

# 平台特定链接库
if(WIN32)
    target_link_libraries(myapp ws2_32)
elseif(UNIX)
    target_link_libraries(myapp pthread)
endif()
```

---

### 5. 文档化平台差异

**记录已知的平台差异和解决方案**

```markdown
## 已知平台差异

### 文件路径
- Windows: 使用 `\` 分隔符
- Linux/macOS: 使用 `/` 分隔符
- 解决方案: 使用 `std::filesystem::path`

### 网络初始化
- Windows: 需要调用 `WSAStartup()`
- Linux/macOS: 不需要初始化
- 解决方案: 使用 NetworkManager 类封装
```

---

## 总结 (Summary)

### 跨平台开发的核心原则

**✅ 完全兼容的特性（优先使用）：**
- 标准 C++ 语言特性（类、模板、STL）
- C++ 标准库（iostream、vector、thread、mutex）
- C++17 filesystem 库（路径处理）
- C++11 线程库（thread、mutex、chrono）

**❌ 不兼容的特性（需要特殊处理）：**
- 系统 API（Windows API vs POSIX）
- 文件路径分隔符（`\` vs `/`）
- 动态库加载（LoadLibrary vs dlopen）
- 网络编程（Winsock vs BSD Sockets）
- 进程创建（CreateProcess vs fork/exec）

### 处理平台差异的策略

1. **优先使用标准 C++ 库** - 最简单、最可靠
2. **使用条件编译** - 当必须使用平台特定 API 时
3. **创建抽象层** - 隔离平台差异
4. **使用跨平台库** - Boost、Qt、POCO 等

### 最佳实践总结

- ✅ 从项目开始就考虑跨平台
- ✅ 最小化平台特定代码
- ✅ 在所有目标平台上测试
- ✅ 使用统一的构建系统（CMake）
- ✅ 文档化已知的平台差异

### 推荐阅读

- [C++ 标准库参考](https://en.cppreference.com/)
- [CMake 文档](https://cmake.org/documentation/)
- [Boost 库](https://www.boost.org/)
- [跨平台编译工具链三元组系统详解](./triplet-system-explained.md)

---

**记住：能用标准 C++ 就不用平台特定 API！**

