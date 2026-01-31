# 智能 Ctrl+C 功能实现文档

## 概述

在 Web 终端中实现智能的 Ctrl+C 功能，使其行为与原生终端（如 VSCode 终端）保持一致：
- **有选中文本时**：执行复制操作
- **无选中文本时**：发送中断信号（SIGINT）

## 问题背景

在标准的 Web 终端实现中，键盘事件会直接传递给后端终端进程，这导致：
1. 用户选中文本后按 Ctrl+C 会发送中断信号，而不是复制文本
2. 无法区分用户的意图是复制还是中断
3. 用户体验与桌面应用不一致

## 解决方案

### 核心：智能 Ctrl+C 处理

```javascript
// 添加键盘事件处理，支持智能 Ctrl+C
terminalInstance.textarea.addEventListener('keydown', (e) => {
    // Ctrl+C 或 Cmd+C 智能处理
    if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        // 检查是否有选中文本
        const selection = terminalInstance.getSelection();
        if (selection) {
            // 有选中文本，执行复制
            e.preventDefault();
            navigator.clipboard.writeText(selection).then(() => {
                console.log('Text copied to clipboard');
            }).catch(err => {
                console.error('Failed to copy text:', err);
            });
            return;
        }
        // 没有选中文本，让 Ctrl+C 传递给终端（发送中断信号）
    }
});
```

## 技术实现细节

### 1. 事件处理流程

```
用户按下 Ctrl+C
    ↓
检查是否有选中文本 (terminalInstance.getSelection())
    ↓
有选中文本？
    ├─ 是 → 阻止默认事件 (e.preventDefault())
    │      ├─ 复制到剪贴板 (navigator.clipboard.writeText())
    │      └─ 结束处理
    └─ 否 → 让事件继续传递给终端
           ├─ 通过 WebSocket 发送给后端
           └─ 后端发送 Ctrl+C 字符给终端进程
```

### 2. 关键 API 说明

#### `terminalInstance.getSelection()`
- **作用**：获取当前终端中选中的文本
- **返回值**：选中的文本字符串，无选中时返回空字符串
- **用途**：判断用户是否想要复制文本

#### `navigator.clipboard.writeText()`
- **作用**：将文本写入系统剪贴板
- **参数**：要复制的文本
- **返回**：Promise，成功时 resolve，失败时 reject
- **兼容性**：需要 HTTPS 环境或 localhost

#### `e.preventDefault()`
- **作用**：阻止事件的默认行为
- **用途**：防止 Ctrl+C 被发送给终端进程

### 3. 字符编码处理

当 Ctrl+C 需要发送中断信号时，事件会通过 `terminalInstance.onData()` 正常传递：

```javascript
terminalInstance.onData((data) => {
    if (wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send(JSON.stringify({ type: 'input', data }));
    }
});
```

Ctrl+C 在终端中对应的字符是 `\x03`（ETX - End of Text），这会触发 SIGINT 信号。

## 兼容性考虑

### 1. 浏览器支持
- **Chrome**：完全支持
- **Firefox**：支持，需要用户授权剪贴板访问
- **Safari**：支持，需要 HTTPS
- **Edge**：完全支持

### 2. 安全限制
- 剪贴板访问需要用户上下文（用户交互触发）
- 必须在 HTTPS 环境下或 localhost 才能访问剪贴板
- 某些浏览器可能需要用户授权

## 完整实现代码

```javascript
// 在终端初始化后添加以下代码

terminalInstance.onData((data) => {
    if (wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send(JSON.stringify({ type: 'input', data }));
    }
});

// 智能Ctrl+C处理
terminalInstance.textarea.addEventListener('keydown', (e) => {
    // Ctrl+C 或 Cmd+C 智能处理
    if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        // 检查是否有选中文本
        const selection = terminalInstance.getSelection();
        if (selection) {
            // 有选中文本，执行复制
            e.preventDefault();
            navigator.clipboard.writeText(selection).then(() => {
                console.log('Text copied to clipboard');
            }).catch(err => {
                console.error('Failed to copy text:', err);
            });
            return;
        }
        // 没有选中文本，让 Ctrl+C 传递给终端（发送中断信号）
    }
});
```

## 测试用例

### 1. 复制功能测试
- [ ] 选中任意文本
- [ ] 按 Ctrl+C
- [ ] 在其他应用中粘贴，验证文本是否正确复制

### 2. 中断信号测试
- [ ] 运行一个长时间执行的命令（如 `ping -t localhost`）
- [ ] 确保没有选中文本
- [ ] 按 Ctrl+C
- [ ] 验证命令是否被中断

## 为什么不实现右键复制？

右键复制在 Web 终端中存在以下挑战：

1. **事件冲突**：右键事件可能与浏览器的默认上下文菜单冲突
2. **用户体验**：不同用户对右键行为的期望不同
3. **实现复杂性**：需要处理各种边界情况和兼容性问题
4. **移动端适配**：移动设备的长按行为与桌面端不同

因此，专注于实现 Ctrl+C 的智能处理是最实用和可靠的解决方案。

## 扩展功能建议

### 1. 粘贴支持
```javascript
// 添加粘贴支持
terminalInstance.textarea.addEventListener('paste', (e) => {
    e.preventDefault();
    const text = e.clipboardData.getData('text');
    if (wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send(JSON.stringify({ type: 'input', data: text }));
    }
});
```

### 2. 全选快捷键
```javascript
// 支持 Ctrl+A 全选
terminalInstance.textarea.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault();
        terminalInstance.selectAll();
    }
});
```

## 总结

通过拦截键盘事件并智能判断用户意图，我们成功实现了与原生终端一致的 Ctrl+C 行为。这个解决方案：

1. **简单可靠**：只处理 Ctrl+C，避免复杂的右键逻辑
2. **用户体验佳**：行为符合用户在 VSCode 等编辑器中的习惯
3. **功能完整**：保持复制和中断信号两种功能
4. **兼容性好**：支持主流浏览器

这种实现方式专注于核心功能，提供了最佳的投入产出比，是 Web 终端项目的理想选择。
