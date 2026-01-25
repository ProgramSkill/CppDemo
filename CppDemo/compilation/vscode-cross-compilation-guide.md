# VS Code C++ Cross-Compilation Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Toolchain Setup](#toolchain-setup)
4. [tasks.json for Cross-Compilation](#tasksjson-for-cross-compilation)
5. [c_cpp_properties.json Configuration](#c_cpp_propertiesjson-configuration)
6. [Remote Debugging](#remote-debugging)
7. [Common Target Platforms](#common-target-platforms)
8. [CMake Integration](#cmake-integration)
9. [Best Practices](#best-practices)

---

## Introduction

Cross-compilation in VS Code allows you to build C++ applications for different target platforms (ARM, RISC-V, embedded systems) while developing on your host machine (x86-64).

**Use Cases:**
- Embedded systems development
- IoT device programming
- Raspberry Pi applications
- Android native development
- Building for different architectures

**Key Concepts:**
- **Host**: Your development machine (e.g., x86-64 Linux)
- **Target**: The platform where code will run (e.g., ARM Cortex-A)
- **Toolchain**: Cross-compiler and tools for target platform

---

## Prerequisites

### Required VS Code Extensions

1. **C/C++ Extension Pack** by Microsoft
2. **CMake Tools** (optional, for CMake projects)
3. **Remote - SSH** (for remote debugging)

### Install Cross-Compilation Toolchain

**For ARM (Linux host):**
```bash
# ARM 32-bit
sudo apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf

# ARM 64-bit (aarch64)
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

**For Raspberry Pi:**
```bash
# Download official toolchain
git clone https://github.com/raspberrypi/tools.git
export PATH=$PATH:~/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64/bin
```

**For Windows (using WSL):**
```bash
# Install WSL and Ubuntu
wsl --install

# Inside WSL, install toolchain
sudo apt update
sudo apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
```

### Verify Toolchain Installation

```bash
# Check ARM compiler
arm-linux-gnueabihf-g++ --version

# Check aarch64 compiler
aarch64-linux-gnu-g++ --version
```

---

## Toolchain Setup

### Understanding Toolchain Triplets

Cross-compilation toolchains use triplet naming: `<arch>-<vendor>-<os>-<abi>`

**Common triplets:**
- `arm-linux-gnueabihf` - ARM 32-bit, hard-float
- `aarch64-linux-gnu` - ARM 64-bit
- `arm-none-eabi` - ARM bare-metal (no OS)
- `riscv64-unknown-linux-gnu` - RISC-V 64-bit

### Setting Up Environment Variables

Create a shell script to set toolchain paths:

**toolchain-env.sh:**
```bash
#!/bin/bash
export CROSS_COMPILE=arm-linux-gnueabihf-
export CC=${CROSS_COMPILE}gcc
export CXX=${CROSS_COMPILE}g++
export AR=${CROSS_COMPILE}ar
export AS=${CROSS_COMPILE}as
export LD=${CROSS_COMPILE}ld
export STRIP=${CROSS_COMPILE}strip

# Sysroot for target libraries
export SYSROOT=/usr/arm-linux-gnueabihf
```

**Usage:**
```bash
source toolchain-env.sh
```

---

## tasks.json for Cross-Compilation

### Basic Cross-Compilation Task

**.vscode/tasks.json:**

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "cross-compile-arm",
            "type": "shell",
            "command": "arm-linux-gnueabihf-g++",
            "args": [
                "-g",
                "-std=c++17",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}.arm"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$gcc"]
        }
    ]
}
```

### Multi-Target Build Configuration

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build-arm32",
            "type": "shell",
            "command": "arm-linux-gnueabihf-g++",
            "args": [
                "-g",
                "-std=c++17",
                "src/*.cpp",
                "-I", "include",
                "-o",
                "build/arm32/app"
            ],
            "group": "build"
        },
        {
            "label": "build-arm64",
            "type": "shell",
            "command": "aarch64-linux-gnu-g++",
            "args": [
                "-g",
                "-std=c++17",
                "src/*.cpp",
                "-I", "include",
                "-o",
                "build/arm64/app"
            ],
            "group": "build"
        },
        {
            "label": "build-all-targets",
            "dependsOn": ["build-arm32", "build-arm64"],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

---

## c_cpp_properties.json Configuration

### Basic Configuration for ARM

**.vscode/c_cpp_properties.json:**

```json
{
    "configurations": [
        {
            "name": "ARM Cross-Compile",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/arm-linux-gnueabihf/include",
                "/usr/arm-linux-gnueabihf/include/c++/9"
            ],
            "defines": ["__ARM_ARCH_7A__"],
            "compilerPath": "/usr/bin/arm-linux-gnueabihf-g++",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-arm"
        }
    ],
    "version": 4
}
```

### Multi-Target Configuration

```json
{
    "configurations": [
        {
            "name": "ARM32",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/arm-linux-gnueabihf/include"
            ],
            "compilerPath": "/usr/bin/arm-linux-gnueabihf-g++",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-arm"
        },
        {
            "name": "ARM64",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/aarch64-linux-gnu/include"
            ],
            "compilerPath": "/usr/bin/aarch64-linux-gnu-g++",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-arm64"
        },
        {
            "name": "Host (x86-64)",
            "includePath": ["${workspaceFolder}/**"],
            "compilerPath": "/usr/bin/g++",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```

**Switch configurations:** Use status bar or `Ctrl+Shift+P` → "C/C++: Select a Configuration"

---

## Remote Debugging

### Setup gdbserver on Target Device

**On target device (e.g., Raspberry Pi):**

```bash
# Install gdbserver
sudo apt install gdbserver

# Run your application with gdbserver
gdbserver :2345 ./app
```

### Configure launch.json for Remote Debugging

**.vscode/launch.json:**

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Remote Debug (ARM)",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/arm32/app",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/arm-linux-gnueabihf-gdb",
            "miDebuggerServerAddress": "192.168.1.100:2345",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "cross-compile-arm"
        }
    ]
}
```

### Remote Debugging Workflow

1. **Build on host:**
   ```bash
   arm-linux-gnueabihf-g++ -g main.cpp -o app
   ```

2. **Copy to target:**
   ```bash
   scp app pi@192.168.1.100:~/
   ```

3. **Start gdbserver on target:**
   ```bash
   ssh pi@192.168.1.100
   gdbserver :2345 ./app
   ```

4. **Start debugging in VS Code:**
   - Press `F5`
   - Debugger connects to remote gdbserver

---

## Common Target Platforms

### Raspberry Pi (ARM32)

**Toolchain:**
```bash
sudo apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
```

**tasks.json:**
```json
{
    "label": "build-raspberry-pi",
    "command": "arm-linux-gnueabihf-g++",
    "args": [
        "-g",
        "-std=c++17",
        "-march=armv7-a",
        "-mfpu=neon-vfpv4",
        "${file}",
        "-o",
        "${fileBasenameNoExtension}.pi"
    ]
}
```

### Raspberry Pi 4 (ARM64)

**Toolchain:**
```bash
sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

**tasks.json:**
```json
{
    "label": "build-raspberry-pi-4",
    "command": "aarch64-linux-gnu-g++",
    "args": [
        "-g",
        "-std=c++17",
        "-march=armv8-a",
        "${file}",
        "-o",
        "${fileBasenameNoExtension}.pi4"
    ]
}
```

### RISC-V

**Toolchain:**
```bash
sudo apt install gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

**tasks.json:**
```json
{
    "label": "build-riscv",
    "command": "riscv64-linux-gnu-g++",
    "args": [
        "-g",
        "-std=c++17",
        "${file}",
        "-o",
        "${fileBasenameNoExtension}.riscv"
    ]
}
```

---

## CMake Integration

### CMake Toolchain File

Create a toolchain file for cross-compilation:

**arm-toolchain.cmake:**

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Specify the cross compiler
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

# Where to look for libraries and headers
set(CMAKE_FIND_ROOT_PATH /usr/arm-linux-gnueabihf)

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

### VS Code CMake Settings

**.vscode/settings.json:**

```json
{
    "cmake.configureSettings": {
        "CMAKE_TOOLCHAIN_FILE": "${workspaceFolder}/arm-toolchain.cmake"
    },
    "cmake.buildDirectory": "${workspaceFolder}/build-arm"
}
```

### Using CMake Tools Extension

1. **Install CMake Tools** extension
2. **Configure:**
   - `Ctrl+Shift+P` → "CMake: Configure"
   - Select toolchain file when prompted
3. **Build:**
   - `Ctrl+Shift+P` → "CMake: Build"
   - Or press `F7`

---

## Best Practices

### 1. Organize Build Outputs by Target

```
project/
├── build-x86/
├── build-arm32/
├── build-arm64/
└── build-riscv/
```

### 2. Use Separate Configuration Files

Create target-specific configuration files:

```
.vscode/
├── tasks-arm.json
├── tasks-x86.json
└── launch-arm.json
```

### 3. Automate Deployment

Create a task to build and deploy:

```json
{
    "label": "build-and-deploy",
    "type": "shell",
    "command": "arm-linux-gnueabihf-g++ ${file} -o app && scp app pi@192.168.1.100:~/"
}
```

### 4. Test on Target Hardware

Always test cross-compiled binaries on actual target hardware:
- Emulators (QEMU) are useful but not perfect
- Real hardware may expose timing, hardware-specific issues

### 5. Use Sysroot for Dependencies

When linking against target libraries:

```bash
arm-linux-gnueabihf-g++ main.cpp \
    --sysroot=/usr/arm-linux-gnueabihf \
    -L/usr/arm-linux-gnueabihf/lib \
    -o app
```

### 6. Version Control Configuration

**Commit:**
- `.vscode/tasks.json`
- `.vscode/c_cpp_properties.json`
- Toolchain files (*.cmake)

**Ignore:**
- `build-*/` directories
- Compiled binaries

---

## Conclusion

VS Code provides excellent support for cross-compilation workflows. Key takeaways:

- **Configure toolchains properly** - Use correct triplets and paths
- **Use tasks.json** - Automate cross-compilation builds
- **Configure IntelliSense** - Set correct compiler paths for each target
- **Remote debugging** - Use gdbserver for on-target debugging
- **CMake integration** - Use toolchain files for complex projects

With proper configuration, VS Code enables efficient cross-platform C++ development from a single development environment.

