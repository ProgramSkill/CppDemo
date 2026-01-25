# VS Code C++ Development Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Extensions](#setup-and-extensions)
3. [Project Structure](#project-structure)
4. [tasks.json - Build Configuration](#tasksjson---build-configuration)
5. [launch.json - Debug Configuration](#launchjson---debug-configuration)
6. [c_cpp_properties.json - IntelliSense](#c_cpp_propertiesjson---intellisense)
7. [Cross-Platform Configuration](#cross-platform-configuration)
8. [Debugging Features](#debugging-features)
9. [Best Practices](#best-practices)

---

## Introduction

Visual Studio Code is a lightweight, cross-platform code editor with powerful C++ development capabilities. This guide covers setting up VS Code for C++ compilation and debugging across Windows, Linux, and macOS.

**Key Features:**
- IntelliSense code completion
- Integrated debugging with breakpoints
- Task automation for building
- Cross-platform support
- Git integration

---

## Setup and Extensions

### Required Extensions

Install the C/C++ extension pack from Microsoft:

1. Open VS Code
2. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on macOS)
3. Search for "C/C++"
4. Install **C/C++ Extension Pack** by Microsoft

**Included extensions:**
- C/C++ (IntelliSense, debugging, code browsing)
- C/C++ Themes
- CMake Tools (optional, for CMake projects)

### Required Compilers

**Windows:**
- MSVC (Visual Studio Build Tools)
- MinGW-w64 (GCC for Windows)
- Clang for Windows

**Linux:**
```bash
sudo apt install build-essential gdb
```

**macOS:**
```bash
xcode-select --install
```

### Verify Installation

```bash
# Check compiler
g++ --version
clang++ --version

# Check debugger
gdb --version      # Linux
lldb --version     # macOS
```

---

## Project Structure

### Typical VS Code C++ Project

```
MyProject/
├── .vscode/
│   ├── tasks.json           # Build tasks
│   ├── launch.json          # Debug configuration
│   └── c_cpp_properties.json # IntelliSense settings
├── src/
│   ├── main.cpp
│   └── utils.cpp
├── include/
│   └── utils.h
└── build/                   # Output directory
```

### Creating .vscode Directory

```bash
mkdir .vscode
cd .vscode
```

VS Code will create these files automatically when you:
- Configure a build task (tasks.json)
- Start debugging (launch.json)
- Configure IntelliSense (c_cpp_properties.json)

---

## tasks.json - Build Configuration

### What is tasks.json?

Defines build tasks that compile your C++ code. Access via `Terminal > Configure Tasks`.

### Basic tasks.json (GCC/Clang)

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "type": "shell",
            "command": "g++",
            "args": [
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
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

### tasks.json for Multiple Files

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build project",
            "type": "shell",
            "command": "g++",
            "args": [
                "-g",
                "-std=c++17",
                "src/*.cpp",
                "-I", "include",
                "-o",
                "build/app"
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

### Key Properties

| Property | Description |
|----------|-------------|
| `label` | Task name shown in UI |
| `command` | Compiler executable (g++, clang++, cl) |
| `args` | Compiler arguments |
| `group.kind` | Task type (build, test) |
| `isDefault` | Default task for `Ctrl+Shift+B` |
| `problemMatcher` | Parse compiler errors |

---

## launch.json - Debug Configuration

### What is launch.json?

Defines how to launch and debug your program. Access via `Run > Add Configuration`.

### Basic launch.json (GCC/Clang - Linux/macOS)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug C++",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "preLaunchTask": "build"
        }
    ]
}
```

### launch.json for Windows (MSVC)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug C++ (MSVC)",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}.exe",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "console": "integratedTerminal",
            "preLaunchTask": "build"
        }
    ]
}
```

### Key Properties

| Property | Description |
|----------|-------------|
| `name` | Configuration name in debug dropdown |
| `type` | Debugger type (cppdbg for GDB/LLDB, cppvsdbg for MSVC) |
| `program` | Path to executable |
| `args` | Command-line arguments |
| `preLaunchTask` | Task to run before debugging (usually "build") |
| `MIMode` | Debugger (gdb or lldb) |
| `stopAtEntry` | Break at main() entry |

---

## c_cpp_properties.json - IntelliSense

### What is c_cpp_properties.json?

Configures IntelliSense for code completion and error detection. Access via `Ctrl+Shift+P` → "C/C++: Edit Configurations (JSON)".

### Basic c_cpp_properties.json

```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/include",
                "/usr/local/include"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```

### Windows Configuration

```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "defines": ["_DEBUG", "UNICODE"],
            "compilerPath": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.30.30705/bin/Hostx64/x64/cl.exe",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "windows-msvc-x64"
        }
    ],
    "version": 4
}
```

### Key Properties

| Property | Description |
|----------|-------------|
| `includePath` | Directories to search for headers |
| `defines` | Preprocessor definitions |
| `compilerPath` | Path to compiler (for IntelliSense) |
| `cppStandard` | C++ standard (c++11, c++14, c++17, c++20) |
| `intelliSenseMode` | Platform and compiler mode |

---

## Cross-Platform Configuration

### Multi-Platform tasks.json

Use platform-specific configurations in a single file:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "type": "shell",
            "command": "g++",
            "args": [
                "-g",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}"
            ],
            "windows": {
                "command": "cl",
                "args": [
                    "/Zi",
                    "/EHsc",
                    "${file}",
                    "/Fe:",
                    "${fileDirname}/${fileBasenameNoExtension}.exe"
                ]
            },
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

### Multi-Platform launch.json

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug C++",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "MIMode": "gdb",
            "linux": {
                "MIMode": "gdb"
            },
            "osx": {
                "MIMode": "lldb"
            },
            "windows": {
                "type": "cppvsdbg",
                "program": "${fileDirname}/${fileBasenameNoExtension}.exe"
            },
            "preLaunchTask": "build"
        }
    ]
}
```

---

## Debugging Features

### Setting Breakpoints

- Click left margin next to line number
- Or press `F9` on current line
- Red dot indicates breakpoint

### Debug Controls

| Shortcut | Action |
|----------|--------|
| `F5` | Start debugging |
| `F10` | Step over |
| `F11` | Step into |
| `Shift+F11` | Step out |
| `F5` (while debugging) | Continue |
| `Shift+F5` | Stop debugging |

### Watch Variables

1. Right-click variable → **Add to Watch**
2. Or use **Watch** panel in debug sidebar
3. Hover over variables to see values

### Debug Console

Execute expressions while debugging:

```cpp
// In debug console
variable_name
myObject.method()
sizeof(array)
```

### Conditional Breakpoints

Right-click breakpoint → **Edit Breakpoint** → Add condition:

```cpp
i == 10
ptr != nullptr
count > 100
```

### Data Breakpoints

Break when variable value changes (Windows only with MSVC debugger).

---

## Best Practices

### 1. Use Workspace Settings

Create `.vscode/settings.json` for project-specific settings:

```json
{
    "files.associations": {
        "*.h": "cpp",
        "*.tpp": "cpp"
    },
    "C_Cpp.default.cppStandard": "c++17",
    "C_Cpp.default.compilerPath": "/usr/bin/g++",
    "editor.formatOnSave": true
}
```

### 2. Organize Build Outputs

```json
{
    "args": [
        "-g",
        "src/*.cpp",
        "-I", "include",
        "-o", "build/app"
    ]
}
```

### 3. Use Variables

VS Code provides useful variables:

| Variable | Description |
|----------|-------------|
| `${workspaceFolder}` | Project root directory |
| `${file}` | Current file path |
| `${fileBasename}` | Current file name |
| `${fileDirname}` | Current file directory |
| `${fileBasenameNoExtension}` | File name without extension |

### 4. Multiple Build Configurations

Create separate tasks for Debug and Release:

```json
{
    "tasks": [
        {
            "label": "build-debug",
            "command": "g++",
            "args": ["-g", "-O0", "${file}", "-o", "build/debug/app"]
        },
        {
            "label": "build-release",
            "command": "g++",
            "args": ["-O2", "${file}", "-o", "build/release/app"]
        }
    ]
}
```

### 5. Use CMake Extension

For larger projects, use CMake Tools extension:

1. Install **CMake Tools** extension
2. Create `CMakeLists.txt`
3. Use `Ctrl+Shift+P` → "CMake: Configure"
4. Build with `Ctrl+Shift+P` → "CMake: Build"

---

## Conclusion

VS Code provides a powerful, lightweight C++ development environment. Key takeaways:

- **Install C/C++ Extension Pack** - Essential for C++ development
- **Configure tasks.json** - Automate compilation
- **Configure launch.json** - Enable debugging
- **Use IntelliSense** - Code completion and error detection
- **Cross-platform support** - Same configuration works on Windows/Linux/macOS

VS Code's flexibility and extensive extension ecosystem make it an excellent choice for C++ development across all platforms.

