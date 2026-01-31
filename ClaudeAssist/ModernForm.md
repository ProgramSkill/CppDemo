# ModernForm 现代风格窗体

仿 DevExpress V25.2 风格的自定义 WinForms 窗体基类。

## 功能特性

### 外观
- 无边框窗体设计
- 深色主题配色
- 自定义标题栏（32px 高度）
- 细边框装饰

### 标题栏按钮
| 按钮 | 功能 | 悬停效果 |
|------|------|----------|
| 最小化 | 最小化窗口 | 灰色高亮 |
| 最大化/还原 | 切换最大化状态 | 灰色高亮 |
| 关闭 | 关闭窗口 | 红色高亮 |

### 交互功能
- 拖动标题栏移动窗口
- 双击标题栏切换最大化
- 拖动边框调整窗口大小
- 四角和四边调整大小光标

## 颜色配置

```csharp
// 标题栏颜色
_titleBarColor = Color.FromArgb(30, 30, 30);

// 窗体背景色
_backgroundColor = Color.FromArgb(37, 37, 38);

// 边框颜色
_borderColor = Color.FromArgb(60, 60, 60);

// 文字颜色
_textColor = Color.FromArgb(241, 241, 241);

// 按钮悬停色
_buttonHoverColor = Color.FromArgb(62, 62, 64);

// 关闭按钮悬停色
_closeButtonHoverColor = Color.FromArgb(232, 17, 35);
```

## 使用方法

### 基本用法

让窗体继承 `ModernForm` 即可：

```csharp
public partial class Form1 : ModernForm
{
    public Form1()
    {
        InitializeComponent();
    }
}
```

### 常量配置

```csharp
TITLE_HEIGHT = 32;    // 标题栏高度
BORDER_RADIUS = 8;    // 边框圆角（预留）
RESIZE_BORDER = 6;    // 调整大小边框宽度
BUTTON_WIDTH = 46;    // 标题栏按钮宽度
```

## 注意事项

1. 窗体内容区域从 `Padding.Top`（32px）开始
2. 添加控件时注意避开标题栏区域
3. 最大化时使用工作区域，不覆盖任务栏

## 代码解析

### 1. 类定义与常量

```csharp
public class ModernForm : Form
{
    private const int TITLE_HEIGHT = 32;      // 标题栏高度
    private const int BORDER_RADIUS = 8;      // 圆角半径（预留）
    private const int RESIZE_BORDER = 6;      // 边框调整区域宽度
    private const int BUTTON_WIDTH = 46;      // 窗口按钮宽度
}
```

继承自 `System.Windows.Forms.Form`，定义了控制窗体外观的核心常量。

### 2. 状态字段

```csharp
private bool _isMaximized = false;        // 最大化状态
private Point _dragStart;                  // 拖动起始点
private bool _isDragging = false;          // 是否正在拖动
private Rectangle _restoreBounds;          // 还原时的窗口位置
private int _hoverButton = -1;             // 悬停按钮索引 (-1=无, 0=最小化, 1=最大化, 2=关闭)
```

用于跟踪窗口状态和鼠标交互。

### 3. 初始化方法

```csharp
private void InitializeModernStyle()
{
    FormBorderStyle = FormBorderStyle.None;  // 移除系统边框
    BackColor = _backgroundColor;             // 设置背景色
    DoubleBuffered = true;                    // 启用双缓冲防止闪烁
    SetStyle(ControlStyles.ResizeRedraw, true); // 调整大小时重绘
    Padding = new Padding(0, TITLE_HEIGHT, 0, 0); // 为标题栏预留空间
}
```

关键设置：`FormBorderStyle.None` 移除系统边框，`DoubleBuffered` 防止绘制闪烁。

### 4. 绘制方法

```csharp
protected override void OnPaint(PaintEventArgs e)
{
    base.OnPaint(e);
    var g = e.Graphics;
    g.SmoothingMode = SmoothingMode.AntiAlias;  // 抗锯齿
    DrawTitleBar(g);   // 绘制标题栏
    DrawBorder(g);     // 绘制边框
}
```

重写 `OnPaint` 实现自定义绘制，`SmoothingMode.AntiAlias` 使图形边缘平滑。

### 5. 标题栏绘制

```csharp
private void DrawTitleBar(Graphics g)
{
    // 标题栏背景
    using var titleBrush = new SolidBrush(_titleBarColor);
    g.FillRectangle(titleBrush, 0, 0, Width, TITLE_HEIGHT);

    // 标题文字
    using var textBrush = new SolidBrush(_textColor);
    using var font = new Font("Segoe UI", 9f);
    var textSize = g.MeasureString(Text, font);
    g.DrawString(Text, font, textBrush, 12, (TITLE_HEIGHT - textSize.Height) / 2);

    DrawWindowButtons(g);  // 绘制窗口按钮
}
```

使用 `using var` 确保 GDI+ 资源正确释放，`MeasureString` 计算文字尺寸实现垂直居中。

### 6. 窗口按钮绘制

```csharp
private void DrawWindowButtons(Graphics g)
{
    int buttonY = 0;
    int buttonHeight = TITLE_HEIGHT;

    // 关闭按钮 (最右侧)
    int closeX = Width - BUTTON_WIDTH;
    if (_hoverButton == 2)
    {
        using var brush = new SolidBrush(_closeButtonHoverColor);
        g.FillRectangle(brush, closeX, buttonY, BUTTON_WIDTH, buttonHeight);
    }
    DrawCloseIcon(g, closeX, buttonY, BUTTON_WIDTH, buttonHeight);

    // 最大化按钮
    int maxX = Width - BUTTON_WIDTH * 2;
    // ... 类似逻辑

    // 最小化按钮
    int minX = Width - BUTTON_WIDTH * 3;
    // ... 类似逻辑
}
```

按钮从右到左排列：关闭 → 最大化 → 最小化。悬停时绘制背景色。

### 7. 按钮图标绘制

```csharp
// 关闭图标 (X)
private void DrawCloseIcon(Graphics g, int x, int y, int w, int h)
{
    using var pen = new Pen(_textColor, 1f);
    int cx = x + w / 2;  // 中心X
    int cy = y + h / 2;  // 中心Y
    int size = 5;
    g.DrawLine(pen, cx - size, cy - size, cx + size, cy + size);
    g.DrawLine(pen, cx + size, cy - size, cx - size, cy + size);
}

// 最大化图标 (□ 或还原图标)
private void DrawMaximizeIcon(Graphics g, int x, int y, int w, int h)
{
    using var pen = new Pen(_textColor, 1f);
    int cx = x + w / 2, cy = y + h / 2, size = 5;
    if (_isMaximized)
    {
        // 还原图标：两个重叠的矩形
        g.DrawRectangle(pen, cx - size + 2, cy - size, size * 2 - 2, size * 2 - 2);
        g.DrawRectangle(pen, cx - size, cy - size + 2, size * 2 - 2, size * 2 - 2);
    }
    else
    {
        g.DrawRectangle(pen, cx - size, cy - size, size * 2, size * 2);
    }
}

// 最小化图标 (-)
private void DrawMinimizeIcon(Graphics g, int x, int y, int w, int h)
{
    using var pen = new Pen(_textColor, 1f);
    int cx = x + w / 2, cy = y + h / 2;
    g.DrawLine(pen, cx - 5, cy, cx + 5, cy);
}
```

使用简单的线条和矩形绘制 Windows 风格的窗口控制图标。

### 8. 鼠标移动处理

```csharp
protected override void OnMouseMove(MouseEventArgs e)
{
    base.OnMouseMove(e);

    // 处理窗口拖动
    if (_isDragging)
    {
        Point currentScreen = PointToScreen(e.Location);
        Location = new Point(
            currentScreen.X - _dragStart.X,
            currentScreen.Y - _dragStart.Y);
        return;
    }

    // 检测按钮悬停，触发局部重绘
    int oldHover = _hoverButton;
    _hoverButton = GetButtonAtPoint(e.Location);
    if (oldHover != _hoverButton)
        Invalidate(new Rectangle(Width - BUTTON_WIDTH * 3, 0, BUTTON_WIDTH * 3, TITLE_HEIGHT));

    // 设置调整大小光标
    Cursor = GetResizeCursor(e.Location);
}
```

`PointToScreen` 将客户端坐标转换为屏幕坐标，实现窗口拖动。`Invalidate` 局部重绘提高性能。

### 9. 鼠标按下与点击

```csharp
protected override void OnMouseDown(MouseEventArgs e)
{
    base.OnMouseDown(e);
    if (e.Button == MouseButtons.Left)
    {
        int button = GetButtonAtPoint(e.Location);
        if (button >= 0) return;  // 点击按钮区域，不启动拖动

        if (e.Y < TITLE_HEIGHT)
        {
            _isDragging = true;
            _dragStart = e.Location;
        }
        else
        {
            HandleResize(e.Location);  // 处理边框调整
        }
    }
}

protected override void OnMouseClick(MouseEventArgs e)
{
    base.OnMouseClick(e);
    int button = GetButtonAtPoint(e.Location);
    switch (button)
    {
        case 0: WindowState = FormWindowState.Minimized; break;
        case 1: ToggleMaximize(); break;
        case 2: Close(); break;
    }
}
```

区分标题栏拖动和按钮点击，避免冲突。

### 10. 最大化切换

```csharp
private void ToggleMaximize()
{
    if (_isMaximized)
    {
        _isMaximized = false;
        Bounds = _restoreBounds;  // 还原到之前的位置和大小
    }
    else
    {
        _restoreBounds = Bounds;  // 保存当前位置和大小
        _isMaximized = true;
        var screen = Screen.FromControl(this).WorkingArea;  // 获取工作区（排除任务栏）
        Bounds = screen;
    }
    Invalidate();
}
```

使用 `Screen.FromControl(this).WorkingArea` 获取当前屏幕工作区，最大化时不覆盖任务栏。

### 11. 调整大小光标

```csharp
private Cursor GetResizeCursor(Point p)
{
    if (_isMaximized) return Cursors.Default;

    bool left = p.X < RESIZE_BORDER;
    bool right = p.X > Width - RESIZE_BORDER;
    bool top = p.Y < RESIZE_BORDER;
    bool bottom = p.Y > Height - RESIZE_BORDER;

    if ((left && top) || (right && bottom)) return Cursors.SizeNWSE;  // ↖↘
    if ((right && top) || (left && bottom)) return Cursors.SizeNESW;  // ↗↙
    if (left || right) return Cursors.SizeWE;   // ↔
    if (top || bottom) return Cursors.SizeNS;   // ↕

    return Cursors.Default;
}
```

根据鼠标位置返回对应的调整大小光标。

### 12. Win32 API 调整大小

```csharp
private void HandleResize(Point p)
{
    if (_isMaximized) return;

    // Windows 消息常量
    const int HTLEFT = 10, HTRIGHT = 11, HTTOP = 12, HTBOTTOM = 15;
    const int HTTOPLEFT = 13, HTTOPRIGHT = 14;
    const int HTBOTTOMLEFT = 16, HTBOTTOMRIGHT = 17;

    // 根据鼠标位置确定调整方向
    bool left = p.X < RESIZE_BORDER;
    bool right = p.X > Width - RESIZE_BORDER;
    bool top = p.Y < RESIZE_BORDER;
    bool bottom = p.Y > Height - RESIZE_BORDER;

    int hit = 0;
    if (left && top) hit = HTTOPLEFT;
    else if (right && top) hit = HTTOPRIGHT;
    else if (left && bottom) hit = HTBOTTOMLEFT;
    else if (right && bottom) hit = HTBOTTOMRIGHT;
    else if (left) hit = HTLEFT;
    else if (right) hit = HTRIGHT;
    else if (top) hit = HTTOP;
    else if (bottom) hit = HTBOTTOM;

    if (hit != 0)
    {
        ReleaseCapture();
        SendMessage(Handle, 0x112, (IntPtr)(0xF000 + hit), IntPtr.Zero);
    }
}
```

### 13. Win32 API 声明

```csharp
[System.Runtime.InteropServices.DllImport("user32.dll")]
private static extern bool ReleaseCapture();

[System.Runtime.InteropServices.DllImport("user32.dll")]
private static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, IntPtr lParam);
```

- `ReleaseCapture()`: 释放鼠标捕获，允许系统接管
- `SendMessage()`: 发送 `WM_SYSCOMMAND` (0x112) 消息
- `0xF000 + hit`: `SC_SIZE` 基值加上方向码，触发系统调整大小行为

这种方式利用 Windows 原生调整大小逻辑，比纯托管代码实现更流畅。
