# ConPTY 终端文本选择与复制功能实现

本文档详细说明 `ConPtyTerminal` 控件中文本选择和复制功能的实现原理。

## 目录

1. [功能概述](#功能概述)
2. [数据结构](#数据结构)
3. [坐标转换](#坐标转换)
4. [鼠标事件处理](#鼠标事件处理)
5. [选择高亮绘制](#选择高亮绘制)
6. [文本提取与复制](#文本提取与复制)
7. [键盘快捷键](#键盘快捷键)
8. [右键上下文菜单](#右键上下文菜单)

---

## 功能概述

终端文本选择功能允许用户：

- **鼠标拖拽选择** - 按住左键拖动选择任意范围的文本
- **双击选择单词** - 双击自动选择光标处的完整单词
- **全选** - 使用 `Ctrl+Shift+A` 选择所有文本
- **复制** - 使用 `Ctrl+C` 或 `Ctrl+Shift+C` 复制选中文本
- **右键菜单** - 提供复制、粘贴、全选等操作

---

## 数据结构

### 选择状态字段

```csharp
// 文本选择
private bool _isSelecting = false;           // 是否正在进行选择操作
private Point _selectionStart = Point.Empty; // 选择起始位置（列, 行）
private Point _selectionEnd = Point.Empty;   // 选择结束位置（列, 行）
private bool _hasSelection = false;          // 是否有有效选择
private Color _selectionColor = Color.FromArgb(100, 0, 122, 204); // 选择高亮颜色（半透明蓝色）
```

### 关键点

- `Point.X` 表示**列**（字符位置）
- `Point.Y` 表示**行**（行号，对应 `_lines` 列表索引）
- 选择范围可以是反向的（结束位置在起始位置之前），需要在使用时规范化

---

## 坐标转换

### 屏幕坐标到单元格坐标

将鼠标点击的像素位置转换为终端的行列位置：

```csharp
private Point ScreenToCell(Point screenPoint)
{
    // 计算可见行数和起始行
    int visibleRows = Height / _charHeight;
    int startLine = Math.Max(0, _lines.Count - visibleRows - _scrollOffset);

    // 像素位置转换为列号（考虑左边距 2 像素）
    int col = Math.Max(0, (screenPoint.X - 2) / _charWidth);
    
    // 像素位置转换为行号（加上滚动偏移）
    int row = startLine + screenPoint.Y / _charHeight;

    return new Point(col, row);
}
```

### 转换原理

```
屏幕坐标 (像素)          单元格坐标 (字符)
┌─────────────────┐      ┌─────────────────┐
│ (50, 30)        │  =>  │ (col=5, row=2)  │
│                 │      │                 │
│ _charWidth = 9  │      │ 每个字符 9 像素宽 │
│ _charHeight = 18│      │ 每行 18 像素高   │
└─────────────────┘      └─────────────────┘
```

---

## 鼠标事件处理

### 1. 鼠标按下 (OnMouseDown)

```csharp
protected override void OnMouseDown(MouseEventArgs e)
{
    base.OnMouseDown(e);
    Focus(); // 确保控件获得焦点

    if (e.Button == MouseButtons.Left)
    {
        // 开始选择
        _isSelecting = true;
        _selectionStart = ScreenToCell(e.Location);
        _selectionEnd = _selectionStart;
        _hasSelection = false;
        Invalidate();
    }
    else if (e.Button == MouseButtons.Right)
    {
        // 显示右键菜单
        ShowContextMenu(e.Location);
    }
}
```

### 2. 鼠标移动 (OnMouseMove)

```csharp
protected override void OnMouseMove(MouseEventArgs e)
{
    base.OnMouseMove(e);

    if (_isSelecting && e.Button == MouseButtons.Left)
    {
        // 更新选择结束位置
        _selectionEnd = ScreenToCell(e.Location);
        _hasSelection = true;
        Invalidate(); // 重绘以显示选择高亮
    }
}
```

### 3. 鼠标释放 (OnMouseUp)

```csharp
protected override void OnMouseUp(MouseEventArgs e)
{
    base.OnMouseUp(e);

    if (e.Button == MouseButtons.Left && _isSelecting)
    {
        _isSelecting = false;
        _selectionEnd = ScreenToCell(e.Location);

        // 检查是否有有效选择（起始和结束位置不同）
        if (_selectionStart != _selectionEnd)
        {
            _hasSelection = true;
        }
        Invalidate();
    }
}
```

### 4. 双击选择单词 (OnMouseDoubleClick)

```csharp
protected override void OnMouseDoubleClick(MouseEventArgs e)
{
    base.OnMouseDoubleClick(e);

    if (e.Button == MouseButtons.Left)
    {
        var cell = ScreenToCell(e.Location);
        SelectWord(cell);
    }
}

private void SelectWord(Point cell)
{
    lock (_buffer)
    {
        if (cell.Y < 0 || cell.Y >= _lines.Count) return;

        string line = _lines[cell.Y];
        if (cell.X < 0 || cell.X >= line.Length) return;

        // 向左查找单词边界
        int start = cell.X;
        while (start > 0 && IsWordChar(line[start - 1]))
            start--;

        // 向右查找单词边界
        int end = cell.X;
        while (end < line.Length - 1 && IsWordChar(line[end + 1]))
            end++;

        _selectionStart = new Point(start, cell.Y);
        _selectionEnd = new Point(end + 1, cell.Y);
        _hasSelection = true;
        Invalidate();
    }
}

private bool IsWordChar(char c)
{
    return char.IsLetterOrDigit(c) || c == '_' || c == '-';
}
```

---

## 选择高亮绘制

在 `OnPaint` 方法中绘制选择高亮：

```csharp
protected override void OnPaint(PaintEventArgs e)
{
    // ... 基础绘制代码 ...

    lock (_buffer)
    {
        int visibleRows = Height / _charHeight;
        int startLine = Math.Max(0, _lines.Count - visibleRows - _scrollOffset);
        int endLine = Math.Min(_lines.Count, startLine + visibleRows + 1);

        // 规范化选择范围（确保 start 在 end 之前）
        var selStart = _selectionStart;
        var selEnd = _selectionEnd;
        if (_hasSelection && (selStart.Y > selEnd.Y || 
            (selStart.Y == selEnd.Y && selStart.X > selEnd.X)))
        {
            (selStart, selEnd) = (selEnd, selStart);
        }

        using var brush = new SolidBrush(_foregroundColor);
        using var selectionBrush = new SolidBrush(_selectionColor);

        for (int i = startLine; i < endLine; i++)
        {
            int screenY = (i - startLine) * _charHeight;
            string line = i < _lines.Count ? _lines[i] : string.Empty;

            // 绘制选择高亮
            if (_hasSelection && i >= selStart.Y && i <= selEnd.Y)
            {
                // 计算当前行的选择范围
                int selStartCol = (i == selStart.Y) ? selStart.X : 0;
                int selEndCol = (i == selEnd.Y) ? selEnd.X : line.Length;

                selStartCol = Math.Max(0, selStartCol);
                selEndCol = Math.Max(selStartCol, Math.Min(selEndCol, 
                    Math.Max(line.Length, _columns)));

                if (selEndCol > selStartCol)
                {
                    int selX = 2 + selStartCol * _charWidth;
                    int selWidth = (selEndCol - selStartCol) * _charWidth;
                    g.FillRectangle(selectionBrush, selX, screenY, selWidth, _charHeight);
                }
            }

            // 绘制文本（在高亮之上）
            g.DrawString(line, _terminalFont, brush, 2, screenY, 
                StringFormat.GenericTypographic);
        }
    }
}
```

### 多行选择示意图

```
行 0: [Hello World]           <- 不在选择范围
行 1: [This is |selected|]    <- selStart.Y = 1, selStart.X = 8
行 2: [|entire line selected|] <- 整行选中
行 3: [|partial| text here]   <- selEnd.Y = 3, selEnd.X = 7
行 4: [Not selected]          <- 不在选择范围
```

---

## 文本提取与复制

### 获取选中文本

```csharp
public string GetSelectedText()
{
    if (!_hasSelection) return string.Empty;

    lock (_buffer)
    {
        var start = _selectionStart;
        var end = _selectionEnd;

        // 规范化：确保 start 在 end 之前
        if (start.Y > end.Y || (start.Y == end.Y && start.X > end.X))
        {
            (start, end) = (end, start);
        }

        var sb = new StringBuilder();

        for (int row = start.Y; row <= end.Y && row < _lines.Count; row++)
        {
            if (row < 0) continue;

            string line = row < _lines.Count ? _lines[row] : string.Empty;

            // 确定当前行的选择列范围
            int startCol = (row == start.Y) ? start.X : 0;
            int endCol = (row == end.Y) ? end.X : line.Length;

            // 边界检查
            startCol = Math.Max(0, Math.Min(startCol, line.Length));
            endCol = Math.Max(0, Math.Min(endCol, line.Length));

            if (startCol < endCol)
            {
                sb.Append(line.Substring(startCol, endCol - startCol));
            }

            // 非最后一行添加换行符
            if (row < end.Y)
            {
                sb.AppendLine();
            }
        }

        return sb.ToString();
    }
}
```

### 复制到剪贴板

```csharp
public void CopySelection()
{
    string text = GetSelectedText();
    if (!string.IsNullOrEmpty(text))
    {
        try
        {
            Clipboard.SetText(text);
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Copy failed: {ex.Message}");
        }
    }
}
```

### 全选

```csharp
public void SelectAll()
{
    lock (_buffer)
    {
        if (_lines.Count == 0) return;

        _selectionStart = new Point(0, 0);
        int lastLine = _lines.Count - 1;
        _selectionEnd = new Point(_lines[lastLine].Length, lastLine);
        _hasSelection = true;
    }
    Invalidate();
}
```

### 清除选择

```csharp
public void ClearSelection()
{
    _hasSelection = false;
    _isSelecting = false;
    _selectionStart = Point.Empty;
    _selectionEnd = Point.Empty;
    Invalidate();
}
```

---

## 键盘快捷键

在 `OnKeyDown` 中处理复制相关快捷键：

```csharp
protected override void OnKeyDown(KeyEventArgs e)
{
    base.OnKeyDown(e);

    if (!_isRunning) return;

    // Ctrl+Shift+C: 复制选中文本
    if (e.Control && e.Shift && e.KeyCode == Keys.C)
    {
        CopySelection();
        e.Handled = true;
        return;
    }

    // Ctrl+C: 智能处理
    // - 有选中文本时复制
    // - 无选中文本时发送中断信号 (SIGINT)
    if (e.Control && e.KeyCode == Keys.C && !e.Shift)
    {
        if (_hasSelection)
        {
            CopySelection();
        }
        else
        {
            SendInput("\x03"); // 发送 Ctrl+C 中断信号
        }
        e.Handled = true;
        return;
    }

    // Ctrl+Shift+V 或 Ctrl+V: 粘贴
    if (e.Control && e.KeyCode == Keys.V)
    {
        if (Clipboard.ContainsText())
        {
            SendInput(Clipboard.GetText());
        }
        e.Handled = true;
        return;
    }

    // Ctrl+Shift+A: 全选
    if (e.Control && e.Shift && e.KeyCode == Keys.A)
    {
        SelectAll();
        e.Handled = true;
        return;
    }

    // ... 其他按键处理 ...
}
```

### 快捷键列表

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+C` | 有选择时复制，无选择时发送中断信号 |
| `Ctrl+Shift+C` | 复制选中文本 |
| `Ctrl+V` | 粘贴 |
| `Ctrl+Shift+A` | 全选 |

---

## 右键上下文菜单

```csharp
private void ShowContextMenu(Point location)
{
    var menu = new ContextMenuStrip();
    menu.Renderer = new ModernMenuRenderer();

    var copyItem = new ToolStripMenuItem("复制", null, (s, e) => CopySelection())
    {
        Enabled = _hasSelection,
        ShortcutKeyDisplayString = "Ctrl+Shift+C"
    };

    var pasteItem = new ToolStripMenuItem("粘贴", null, (s, e) =>
    {
        if (Clipboard.ContainsText())
        {
            SendInput(Clipboard.GetText());
        }
    })
    {
        Enabled = Clipboard.ContainsText(),
        ShortcutKeyDisplayString = "Ctrl+V"
    };

    var selectAllItem = new ToolStripMenuItem("全选", null, (s, e) => SelectAll())
    {
        ShortcutKeyDisplayString = "Ctrl+Shift+A"
    };

    var clearItem = new ToolStripMenuItem("清屏", null, (s, e) =>
    {
        Clear();
        SendInput("cls\r");
    });

    menu.Items.Add(copyItem);
    menu.Items.Add(pasteItem);
    menu.Items.Add(new ToolStripSeparator());
    menu.Items.Add(selectAllItem);
    menu.Items.Add(new ToolStripSeparator());
    menu.Items.Add(clearItem);

    menu.Show(this, location);
}
```

---

## 总结

文本选择和复制功能的实现要点：

1. **坐标系统** - 使用 `Point(列, 行)` 表示选择范围
2. **坐标转换** - `ScreenToCell()` 将像素坐标转换为字符坐标
3. **鼠标事件** - 处理 `MouseDown`、`MouseMove`、`MouseUp` 实现拖拽选择
4. **选择规范化** - 始终确保 start 在 end 之前
5. **高亮绘制** - 在 `OnPaint` 中绘制半透明选择背景
6. **文本提取** - 遍历选择范围内的行，提取对应列的字符
7. **剪贴板操作** - 使用 `Clipboard.SetText()` 复制文本
