# Backspace 按键修复

本文档记录 ConPTY 终端控件 Backspace 按键工作不正常的问题分析和修复方案。

## 问题现象

按下 Backspace 键时，终端没有正确删除字符，或者行为异常。

## 问题原因

### BS 字符 vs DEL 字符

终端中有两个与"退格"相关的控制字符：

| 字符 | ASCII 码 | 名称 | 传统用途 |
|------|----------|------|----------|
| `\b` | 0x08 | BS (Backspace) | 光标左移一格（不删除字符） |
| `\x7f` | 0x7F | DEL (Delete) | 删除光标前的字符 |

### 历史背景

- **早期终端**：BS (0x08) 只是移动光标，不删除字符
- **现代终端**：大多数使用 DEL (0x7F) 作为退格键的输入
- **Windows ConPTY**：期望接收 DEL 字符来执行退格删除操作

### 原始代码问题

```csharp
case Keys.Back:
    SendInput("\b");  // 发送 BS (0x08)
    e.Handled = true;
    break;
```

发送 `\b` (0x08) 在某些终端中只会移动光标，不会删除字符。

## 修复方案

```csharp
case Keys.Back:
    SendInput("\x7f"); // DEL 字符，大多数终端使用这个作为退格
    e.Handled = true;
    break;
```

## 对比说明

```
修复前：按下 Backspace
    │
    ▼
SendInput("\b")  ─── 发送 0x08 (BS)
    │
    ▼
终端收到 BS
    │
    ▼
光标左移（可能不删除字符）


修复后：按下 Backspace
    │
    ▼
SendInput("\x7f")  ─── 发送 0x7F (DEL)
    │
    ▼
终端收到 DEL
    │
    ▼
删除光标前的字符 ✓
```

## 相关知识

### 终端控制字符表

| 字符 | 十六进制 | 名称 | 功能 |
|------|----------|------|------|
| `\x00` | 0x00 | NUL | 空字符 |
| `\x03` | 0x03 | ETX | Ctrl+C，中断 |
| `\x04` | 0x04 | EOT | Ctrl+D，EOF |
| `\x08` | 0x08 | BS | 退格（光标左移） |
| `\x09` | 0x09 | HT | Tab |
| `\x0A` | 0x0A | LF | 换行 |
| `\x0D` | 0x0D | CR | 回车 |
| `\x1B` | 0x1B | ESC | 转义序列开始 |
| `\x7F` | 0x7F | DEL | 删除 |

### 不同系统的退格键行为

| 系统/终端 | Backspace 发送 |
|-----------|----------------|
| Linux/Unix 终端 | DEL (0x7F) |
| Windows CMD | BS (0x08) |
| Windows ConPTY | DEL (0x7F) |
| macOS Terminal | DEL (0x7F) |
| PuTTY (默认) | DEL (0x7F) |

## 总结

- **问题**：Backspace 发送 BS (0x08) 而不是 DEL (0x7F)
- **原因**：现代终端期望 DEL 字符来执行退格删除
- **修复**：将 `\b` 改为 `\x7f`
