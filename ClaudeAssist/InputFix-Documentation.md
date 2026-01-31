# ConPTY 终端输入问题修复

本文档记录 ConPTY 终端控件键盘输入无响应问题的原因分析和修复方案。

## 问题现象

- 终端光标正常闪烁（说明控件有焦点）
- 按键没有任何反应
- 无法输入任何字符

## 问题原因

### 根本原因：FileStream 的 using 语句导致管道句柄被关闭

原始代码：

```csharp
public void SendInput(string text)
{
    if (!_isRunning || _pipeInputWrite == null || _pipeInputWrite.IsInvalid) return;

    try
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        using var stream = new FileStream(_pipeInputWrite, FileAccess.Write, 4096, false);
        stream.Write(bytes, 0, bytes.Length);
        stream.Flush();
    }
    catch (Exception ex)
    {
        Debug.WriteLine($"SendInput error: {ex.Message}");
    }
}
```

### 问题分析

1. `using var stream = new FileStream(...)` 创建一个新的 FileStream
2. 当 `using` 块结束时，FileStream 被 Dispose
3. **FileStream.Dispose() 会关闭底层的 SafeFileHandle**
4. 第一次调用 `SendInput` 后，`_pipeInputWrite` 句柄就被关闭了
5. 后续所有输入都无法发送

### 流程图

```
第一次按键
    │
    ▼
SendInput("a")
    │
    ▼
new FileStream(_pipeInputWrite, ...)
    │
    ▼
stream.Write() ─── 成功写入
    │
    ▼
using 块结束
    │
    ▼
stream.Dispose()
    │
    ▼
_pipeInputWrite 句柄被关闭 ◄── 问题根源！
    │
    ▼
第二次按键
    │
    ▼
SendInput("b")
    │
    ▼
_pipeInputWrite.IsInvalid == true
    │
    ▼
直接 return，输入被忽略
```

## 修复方案

### 1. 添加持久的输入流字段

```csharp
#region 字段

private IntPtr _hPC = IntPtr.Zero;
private SafeFileHandle? _pipeInputRead;
private SafeFileHandle? _pipeInputWrite;
private SafeFileHandle? _pipeOutputRead;
private SafeFileHandle? _pipeOutputWrite;
private FileStream? _inputStream; // 新增：持久的输入流

// ... 其他字段
```

### 2. 在创建进程后初始化输入流

```csharp
private void CreatePseudoConsoleAndProcess()
{
    // ... 创建进程代码 ...

    // 关闭不需要的句柄
    _pipeInputRead?.Close();
    _pipeInputRead = null;
    _pipeOutputWrite?.Close();
    _pipeOutputWrite = null;

    // 新增：创建持久的输入流
    _inputStream = new FileStream(_pipeInputWrite!, FileAccess.Write, 4096, false);
}
```

### 3. 修改 SendInput 使用持久输入流

```csharp
public void SendInput(string text)
{
    if (!_isRunning || _inputStream == null) return;

    try
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        _inputStream.Write(bytes, 0, bytes.Length);
        _inputStream.Flush();
    }
    catch (Exception ex)
    {
        Debug.WriteLine($"SendInput error: {ex.Message}");
    }
}
```

### 4. 在清理时关闭输入流

```csharp
private void Cleanup()
{
    _isRunning = false;

    // 新增：关闭输入流
    try
    {
        _inputStream?.Close();
        _inputStream?.Dispose();
    }
    catch { }
    _inputStream = null;

    // ... 其他清理代码 ...
}
```

## 修复后的流程

```
第一次按键
    │
    ▼
SendInput("a")
    │
    ▼
_inputStream.Write() ─── 成功写入
    │
    ▼
_inputStream.Flush()
    │
    ▼
_inputStream 保持打开状态 ◄── 关键！
    │
    ▼
第二次按键
    │
    ▼
SendInput("b")
    │
    ▼
_inputStream.Write() ─── 成功写入
    │
    ▼
继续正常工作...
```

## 关键教训

1. **FileStream 会接管 SafeFileHandle 的生命周期** - 当 FileStream 被 Dispose 时，底层句柄也会被关闭
2. **对于需要持续使用的管道，应该创建持久的 FileStream** - 而不是每次操作都创建新的
3. **using 语句虽然方便，但要注意资源的生命周期** - 特别是涉及到句柄共享的情况

## 其他相关修复

在修复输入问题的同时，还进行了以下改进：

### 1. 支持 Unicode 字符输入

```csharp
// 修复前：只支持 ASCII
if (e.KeyChar >= 32 && e.KeyChar < 127)

// 修复后：支持所有可打印字符（包括中文）
if (e.KeyChar >= 32)
```

### 2. 添加焦点事件处理

```csharp
protected override void OnGotFocus(EventArgs e)
{
    base.OnGotFocus(e);
    _cursorVisible = true;
    Invalidate();
}

protected override void OnLostFocus(EventArgs e)
{
    base.OnLostFocus(e);
    _cursorVisible = false;
    Invalidate();
}
```

### 3. 设置 TabStop 属性

```csharp
public ConPtyTerminal()
{
    // ...
    TabStop = true; // 允许 Tab 键聚焦
    // ...
}
```
