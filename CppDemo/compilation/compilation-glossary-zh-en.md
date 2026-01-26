# C++ 编译相关术语中英文对照表
# C++ Compilation Terminology Reference (Chinese-English)

本文档提供 C++ 编译、构建和开发相关的中英文术语对照表，方便查阅和学习。

This document provides a Chinese-English glossary of C++ compilation, build, and development terminology for easy reference and learning.

---

## 目录 (Table of Contents)

1. [基础编译术语](#基础编译术语-basic-compilation-terms)
2. [构建系统](#构建系统-build-systems)
3. [编译器选项](#编译器选项-compiler-options)
4. [链接相关](#链接相关-linking-terms)
5. [优化相关](#优化相关-optimization-terms)
6. [调试相关](#调试相关-debugging-terms)
7. [平台与架构](#平台与架构-platforms-and-architectures)
8. [库相关](#库相关-library-terms)
9. [文件类型](#文件类型-file-types)
10. [交叉编译](#交叉编译-cross-compilation)

---

## 基础编译术语 (Basic Compilation Terms)

| 中文 | 英文 | 说明 |
|------|------|------|
| 编译器 | Compiler | 将源代码转换为目标代码的程序 |
| 预处理器 | Preprocessor | 处理 #include、#define 等指令 |
| 汇编器 | Assembler | 将汇编代码转换为机器码 |
| 链接器 | Linker | 将目标文件组合成可执行文件 |
| 源文件 | Source File | 包含源代码的文件 (.cpp, .c) |
| 头文件 | Header File | 包含声明的文件 (.h, .hpp) |
| 目标文件 | Object File | 编译后的二进制文件 (.o, .obj) |
| 可执行文件 | Executable | 可直接运行的程序文件 (.exe, 无扩展名) |
| 编译单元 | Translation Unit | 单个源文件及其包含的所有头文件 |
| 预编译 | Precompilation | 预处理阶段的处理 |
| 编译 | Compilation | 将源代码转换为目标代码 |
| 链接 | Linking | 将目标文件组合成最终程序 |
| 构建 | Build | 完整的编译和链接过程 |
| 重新构建 | Rebuild | 清理后重新构建所有文件 |
| 增量编译 | Incremental Compilation | 只编译修改过的文件 |
| 符号 | Symbol | 函数名、变量名等标识符 |
| 符号表 | Symbol Table | 存储符号信息的数据结构 |
| 名称修饰 | Name Mangling | C++ 编译器对符号名的编码 |
| 宏 | Macro | 预处理器定义的文本替换 |
| 预处理指令 | Preprocessor Directive | #include、#define 等指令 |

---

## 构建系统 (Build Systems)

| 中文 | 英文 | 说明 |
|------|------|------|
| 构建系统 | Build System | 自动化编译过程的工具 |
| 构建工具 | Build Tool | 执行构建任务的程序 |
| 构建脚本 | Build Script | 定义构建过程的脚本文件 |
| 构建目标 | Build Target | 构建系统要生成的输出 |
| 依赖关系 | Dependency | 文件或目标之间的依赖 |
| 构建配置 | Build Configuration | 调试版、发布版等配置 |
| 项目文件 | Project File | 定义项目结构的文件 |
| 解决方案 | Solution | Visual Studio 中的项目集合 |
| 工作区 | Workspace | 项目的工作目录 |
| 生成器 | Generator | CMake 中生成构建文件的工具 |
| 工具链 | Toolchain | 编译器、链接器等工具的集合 |
| 工具集 | Toolset | Visual Studio 中的编译器版本 |
| 属性表 | Property Sheet | Visual Studio 中的可重用配置 |
| 批量构建 | Batch Build | 同时构建多个配置 |
| 并行构建 | Parallel Build | 使用多核并行编译 |
| 清理 | Clean | 删除构建生成的文件 |
| 配置管理器 | Configuration Manager | Visual Studio 的配置管理工具 |

---

## 编译器选项 (Compiler Options)

| 中文 | 英文 | 说明 |
|------|------|------|
| 编译器标志 | Compiler Flag | 传递给编译器的选项 |
| 优化级别 | Optimization Level | 代码优化程度 (-O0, -O2, -O3) |
| 警告级别 | Warning Level | 编译器警告的详细程度 |
| 调试信息 | Debug Information | 用于调试的符号信息 |
| 预处理器定义 | Preprocessor Definition | 编译时定义的宏 (-D) |
| 包含路径 | Include Path | 头文件搜索路径 (-I) |
| 库路径 | Library Path | 库文件搜索路径 (-L) |
| 链接库 | Link Library | 要链接的库 (-l) |
| 标准版本 | Standard Version | C++ 标准 (C++11, C++17, C++20) |
| 运行时库 | Runtime Library | 程序运行时依赖的库 (/MD, /MT) |
| 字符集 | Character Set | Unicode 或多字节字符集 |
| 异常处理 | Exception Handling | 启用/禁用异常处理 |
| 运行时检查 | Runtime Check | 运行时错误检查 |
| 内联 | Inlining | 函数内联优化 |
| 位置无关代码 | Position Independent Code | PIC, 用于共享库 (-fPIC) |
| 警告即错误 | Treat Warnings as Errors | 将警告视为错误 (-Werror, /WX) |
| 详细输出 | Verbose Output | 显示详细编译信息 (-v) |

---

## 链接相关 (Linking Terms)

| 中文 | 英文 | 说明 |
|------|------|------|
| 静态链接 | Static Linking | 将库代码复制到可执行文件 |
| 动态链接 | Dynamic Linking | 运行时加载共享库 |
| 静态库 | Static Library | 编译时链接的库 (.a, .lib) |
| 动态库 | Dynamic Library | 运行时加载的库 (.so, .dll) |
| 共享库 | Shared Library | Linux 下的动态库 (.so) |
| 导入库 | Import Library | Windows DLL 的链接库 (.lib) |
| 导出 | Export | 从 DLL 导出符号 |
| 导入 | Import | 从 DLL 导入符号 |
| 符号解析 | Symbol Resolution | 链接器查找符号定义 |
| 未定义引用 | Undefined Reference | 链接时找不到符号定义 |
| 重复定义 | Multiple Definition | 同一符号有多个定义 |
| 弱符号 | Weak Symbol | 可被覆盖的符号 |
| 强符号 | Strong Symbol | 不可被覆盖的符号 |
| 链接时优化 | Link-Time Optimization | LTO, 跨编译单元优化 |
| 全程序优化 | Whole Program Optimization | MSVC 的 LTO |
| 延迟加载 | Delay Loading | 延迟加载 DLL |
| 重定位 | Relocation | 调整代码和数据地址 |
| 入口点 | Entry Point | 程序开始执行的位置 |

---

## 优化相关 (Optimization Terms)

| 中文 | 英文 | 说明 |
|------|------|------|
| 优化 | Optimization | 提高代码性能的技术 |
| 循环展开 | Loop Unrolling | 减少循环开销的优化 |
| 函数内联 | Function Inlining | 将函数调用替换为函数体 |
| 常量折叠 | Constant Folding | 编译时计算常量表达式 |
| 死代码消除 | Dead Code Elimination | 删除永不执行的代码 |
| 公共子表达式消除 | Common Subexpression Elimination | CSE, 避免重复计算 |
| 尾调用优化 | Tail Call Optimization | 优化尾递归函数 |
| 向量化 | Vectorization | 使用 SIMD 指令优化 |
| 自动向量化 | Auto-Vectorization | 编译器自动向量化 |
| 循环向量化 | Loop Vectorization | 对循环进行向量化 |
| 内存对齐 | Memory Alignment | 数据按边界对齐 |
| 缓存优化 | Cache Optimization | 提高缓存命中率 |
| 分支预测 | Branch Prediction | CPU 预测分支走向 |
| 代码大小优化 | Size Optimization | 减小代码体积 (-Os) |
| 速度优化 | Speed Optimization | 提高执行速度 (-O2, -O3) |
| 激进优化 | Aggressive Optimization | 可能影响精度的优化 (-Ofast) |
| 调试优化 | Debug Optimization | 保留调试信息的优化 (-Og) |

---

## 调试相关 (Debugging Terms)

| 中文 | 英文 | 说明 |
|------|------|------|
| 调试器 | Debugger | 用于调试程序的工具 (GDB, LLDB) |
| 断点 | Breakpoint | 程序暂停执行的位置 |
| 单步执行 | Step | 逐行执行代码 |
| 单步进入 | Step Into | 进入函数内部 |
| 单步跳过 | Step Over | 跳过函数调用 |
| 单步跳出 | Step Out | 跳出当前函数 |
| 继续执行 | Continue | 继续运行到下一个断点 |
| 监视点 | Watchpoint | 监视变量值变化 |
| 调用栈 | Call Stack | 函数调用的层次结构 |
| 栈帧 | Stack Frame | 单个函数调用的栈信息 |
| 回溯 | Backtrace | 显示调用栈 |
| 核心转储 | Core Dump | 程序崩溃时的内存快照 |
| 符号文件 | Symbol File | 包含调试符号的文件 (.pdb, .dSYM) |
| 调试符号 | Debug Symbol | 变量名、函数名等调试信息 |
| 远程调试 | Remote Debugging | 在远程设备上调试 |
| 附加进程 | Attach to Process | 调试正在运行的进程 |
| 条件断点 | Conditional Breakpoint | 满足条件时才触发的断点 |
| 内存泄漏 | Memory Leak | 未释放的内存 |
| 地址消毒器 | Address Sanitizer | ASan, 检测内存错误 |
| 未定义行为消毒器 | Undefined Behavior Sanitizer | UBSan, 检测未定义行为 |

---

## 平台与架构 (Platforms and Architectures)

| 中文 | 英文 | 说明 |
|------|------|------|
| 平台 | Platform | 操作系统和硬件的组合 |
| 架构 | Architecture | CPU 指令集架构 |
| 目标平台 | Target Platform | 程序运行的平台 |
| 主机平台 | Host Platform | 编译程序的平台 |
| 交叉编译 | Cross-Compilation | 在一个平台上为另一个平台编译 |
| 本地编译 | Native Compilation | 为当前平台编译 |
| x86 | x86 | 32 位 Intel/AMD 架构 |
| x64 | x64 / x86-64 / AMD64 | 64 位 Intel/AMD 架构 |
| ARM | ARM | ARM 32 位架构 |
| ARM64 | ARM64 / AArch64 | ARM 64 位架构 |
| RISC-V | RISC-V | 开源 RISC 架构 |
| 大端序 | Big-Endian | 高位字节在前 |
| 小端序 | Little-Endian | 低位字节在前 |
| 字节序 | Endianness | 多字节数据的存储顺序 |
| 位数 | Bit Width | 32 位或 64 位系统 |
| 指令集 | Instruction Set | CPU 支持的指令集合 |
| SIMD | SIMD | 单指令多数据并行 |
| SSE | SSE | x86 的 SIMD 扩展 |
| AVX | AVX | 高级向量扩展 |
| NEON | NEON | ARM 的 SIMD 扩展 |

---

## 库相关 (Library Terms)

| 中文 | 英文 | 说明 |
|------|------|------|
| 库 | Library | 可重用的代码集合 |
| 标准库 | Standard Library | C++ 标准库 (STL) |
| 第三方库 | Third-Party Library | 外部开发的库 |
| 系统库 | System Library | 操作系统提供的库 |
| 运行时库 | Runtime Library | 程序运行时需要的库 |
| C 运行时 | C Runtime | CRT, C 标准库实现 |
| C++ 运行时 | C++ Runtime | C++ 标准库实现 |
| 头文件库 | Header-Only Library | 只有头文件的库 |
| 包管理器 | Package Manager | 管理依赖库的工具 (vcpkg, Conan) |
| 依赖项 | Dependency | 项目依赖的外部库 |
| 传递依赖 | Transitive Dependency | 依赖项的依赖项 |
| 库搜索路径 | Library Search Path | 链接器查找库的路径 |
| 运行时路径 | Runtime Path | 运行时查找动态库的路径 (RPATH) |
| 导出符号 | Exported Symbol | 库对外公开的符号 |
| 隐藏符号 | Hidden Symbol | 库内部使用的符号 |
| API | API | 应用程序编程接口 |
| ABI | ABI | 应用程序二进制接口 |
| 版本兼容性 | Version Compatibility | 不同版本间的兼容性 |

---

## 文件类型 (File Types)

| 中文 | 英文 | 说明 |
|------|------|------|
| 源文件 | Source File | C++ 源代码文件 (.cpp, .cc, .cxx) |
| 头文件 | Header File | 声明文件 (.h, .hpp, .hxx) |
| 目标文件 | Object File | 编译后的二进制 (.o, .obj) |
| 静态库文件 | Static Library File | Linux: .a, Windows: .lib |
| 动态库文件 | Dynamic Library File | Linux: .so, Windows: .dll, macOS: .dylib |
| 可执行文件 | Executable File | Windows: .exe, Linux/macOS: 无扩展名 |
| 预编译头 | Precompiled Header | .pch, .gch |
| 汇编文件 | Assembly File | .s, .asm |
| 模块文件 | Module File | C++20 模块 (.ixx, .cppm) |
| 资源文件 | Resource File | Windows 资源 (.rc) |
| 定义文件 | Definition File | 模块定义 (.def) |
| 清单文件 | Manifest File | Windows 清单 (.manifest) |
| 符号文件 | Symbol File | Windows: .pdb, macOS: .dSYM |
| 导入库 | Import Library | Windows DLL 的 .lib 文件 |
| 导出文件 | Export File | .exp |
| 中间文件 | Intermediate File | 编译过程中的临时文件 |
| 依赖文件 | Dependency File | Make 依赖关系 (.d) |

---

## 交叉编译 (Cross-Compilation)

| 中文 | 英文 | 说明 |
|------|------|------|
| 交叉编译 | Cross-Compilation | 在一个平台上为另一个平台编译 |
| 交叉编译器 | Cross-Compiler | 生成其他平台代码的编译器 |
| 目标三元组 | Target Triplet | 描述目标平台的字符串 (arch-vendor-os) |
| 工具链文件 | Toolchain File | CMake 交叉编译配置文件 |
| 系统根目录 | Sysroot | 目标系统的根文件系统 |
| 目标架构 | Target Architecture | 编译目标的 CPU 架构 |
| 主机架构 | Host Architecture | 编译器运行的 CPU 架构 |
| 远程调试 | Remote Debugging | 在目标设备上调试程序 |
| 调试服务器 | Debug Server | 目标设备上的调试代理 (gdbserver) |
| 仿真器 | Emulator | 模拟目标平台的软件 (QEMU) |
| 板级支持包 | Board Support Package | BSP, 硬件相关的支持代码 |
| 裸机编程 | Bare-Metal Programming | 无操作系统的嵌入式编程 |
| 嵌入式系统 | Embedded System | 专用功能的计算机系统 |
| 物联网 | Internet of Things | IoT, 互联网连接的设备 |
| 树莓派 | Raspberry Pi | 流行的单板计算机 |
| 启动加载程序 | Bootloader | 系统启动时的初始化程序 |

---

## 结语 (Conclusion)

本术语表涵盖了 C++ 编译、构建和开发过程中的常用术语。建议配合其他编译文档一起使用，以便更好地理解和掌握 C++ 开发工具链。

This glossary covers common terminology used in C++ compilation, build, and development processes. It is recommended to use this reference alongside other compilation documentation for better understanding and mastery of the C++ development toolchain.

