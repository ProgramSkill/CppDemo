# DevExpress v25.2 风格多标签页控件实现文档

## 概述

本文档介绍如何在 WinForms 中实现一个类似 **DevExpress v25.2** 风格的现代化多标签页控件系统。

## 核心组件

### 1. ModernTabControl - 标签栏控件

`ModernTabControl` 是核心控件，负责绘制和管理标签页。

**文件位置**: `ClaudeAssist/ModernTabControl.cs`

#### 主要特性

| 特性 | 说明 |
|------|------|
| **现代化扁平设计** | 深色主题，符合 DevExpress v25.2 视觉风格 |
| **彩色活动指示器** | 活动标签顶部显示彩色指示线（默认蓝色 #007ACC） |
| **标签关闭按钮** | 悬停时显示，支持点击关闭 |
| **标签拖拽重排序** | 支持鼠标拖拽调整标签顺序 |
| **新增标签按钮** | 右侧 (+) 按钮快速创建标签 |
| **滚动按钮** | 标签溢出时自动显示左右滚动按钮 |
| **右键上下文菜单** | 支持关闭/关闭其他/关闭全部/复制标签 |

#### 颜色主题（DevExpress v25.2 风格）

```csharp
TabBarColor = Color.FromArgb(45, 45, 48);           // 标签栏背景
TabBarInactiveColor = Color.FromArgb(30, 30, 30);   // 非活动区域背景
ActiveTabIndicatorColor = Color.FromArgb(0, 122, 204); // 活动指示器（蓝色）
TabHoverColor = Color.FromArgb(62, 62, 64);         // 悬停背景
TabActiveColor = Color.FromArgb(37, 37, 38);        // 活动标签背景
TabTextColor = Color.FromArgb(241, 241, 241);       // 文字颜色
TabInactiveTextColor = Color.FromArgb(150, 150, 150); // 非活动文字
CloseButtonHoverColor = Color.FromArgb(200, 200, 200);
CloseButtonPressedColor = Color.FromArgb(232, 17, 35); // 关闭按钮红色
```

#### 关键尺寸常量

```csharp
TAB_HEIGHT = 36;              // 标签栏高度
TAB_MIN_WIDTH = 140;          // 标签最小宽度
TAB_MAX_WIDTH = 200;          // 标签最大宽度
TAB_PADDING = 8;              // 内边距
CLOSE_BUTTON_SIZE = 14;       // 关闭按钮尺寸
NEW_TAB_BUTTON_WIDTH = 36;    // 新建标签按钮宽度
SCROLL_BUTTON_WIDTH = 24;     // 滚动按钮宽度
ICON_SIZE = 16;               // 图标尺寸
```

### 2. ModernTabContainer - 标签容器

`ModernTabContainer` 是组合控件，包含 `ModernTabControl`（标签栏）和 `Panel`（内容区域）。

**文件位置**: `ClaudeAssist/ModernTabContainer.cs`

#### 职责

- 管理标签栏和内容面板的布局
- 自动切换选中标签对应的内容
- 提供统一的 API 操作标签

### 3. TabItem - 标签页数据类

表示单个标签页的数据结构。

```csharp
public class TabItem
{
    public string Title { get; set; }           // 标签标题
    public Image? Icon { get; set; }            // 标签图标
    public Control? Content { get; set; }       // 关联的内容控件
    public object? Tag { get; set; }            // 自定义数据
    internal Rectangle Bounds { get; set; }     // 绘制边界
    internal Rectangle CloseButtonBounds { get; set; } // 关闭按钮边界
}
```

## 实现细节

### 绘制流程

1. **OnPaint** 方法按以下顺序绘制：
   - 标签栏背景（深色）
   - 底部边框线
   - 各个标签页（根据状态不同样式）
   - 滚动按钮（如需要）
   - 新建标签按钮 (+)

2. **DrawTab** 绘制单个标签：
   - 背景色（活动/悬停/普通）
   - 顶部活动指示器（仅活动标签）
   - 图标
   - 标题文字
   - 关闭按钮（悬停时）
   - 右侧分隔线

### 鼠标交互处理

#### 状态管理

```csharp
private int _hoverIndex = -1;              // 当前悬停的标签索引
private int _pressedIndex = -1;            // 当前按下的标签索引
private int _closeButtonHoverIndex = -1;   // 关闭按钮悬停索引
private bool _isDragging = false;          // 是否正在拖拽
private int _dragStartIndex = -1;          // 拖拽起始索引
```

#### 事件处理流程

1. **MouseDown**: 
   - 检测是否点击新建按钮、滚动按钮
   - 检测是否点击标签或关闭按钮
   - 记录拖拽起始位置

2. **MouseMove**: 
   - 更新悬停状态
   - 处理拖拽重排序
   - 设置鼠标光标

3. **MouseUp**: 
   - 执行点击操作
   - 结束拖拽

### 标签拖拽重排序

```csharp
private void HandleTabDrag(Point location)
{
    // 检查拖拽距离
    if (Math.Abs(location.X - _dragStartPoint.X) < 5)
        return;

    _isDragging = true;

    // 找到鼠标下的标签
    for (int i = 0; i < _tabs.Count; i++)
    {
        if (i == _dragStartIndex) continue;
        
        if (GetTabBounds(i).Contains(location))
        {
            // 移动标签位置
            MoveTab(_dragStartIndex, i);
            _dragStartIndex = i;
            break;
        }
    }
}
```

### 滚动处理

当标签总宽度超过容器宽度时：

1. 计算 `_scrollOffset` 偏移量
2. 显示左右滚动按钮
3. 点击滚动按钮调整偏移量
4. 重新计算标签绘制位置

```csharp
private void ScrollRight()
{
    int maxOffset = Math.Max(0, GetTotalTabsWidth() - GetTabContentWidth());
    _scrollOffset = Math.Min(maxOffset, _scrollOffset + TAB_MIN_WIDTH);
    UpdateTabBounds();
    Invalidate();
}
```

### 上下文菜单

右键点击标签时显示：

- 关闭 (Ctrl+W)
- 关闭其他标签页
- 关闭所有标签页
- 分隔线
- 复制标签页

使用自定义 `ModernMenuRenderer` 实现深色菜单样式。

## 使用方式

### 基本用法

```csharp
// 创建容器
var tabContainer = new ModernTabContainer
{
    Dock = DockStyle.Fill
};

// 添加标签页
tabContainer.AddTab("主页", icon, contentPanel);
tabContainer.AddTab("设置", icon, settingsPanel);

// 订阅事件
tabContainer.TabSelected += (s, e) => 
    Console.WriteLine($"选中: {e.Tab?.Title}");

tabContainer.NewTabButtonClick += (s, e) => 
    tabContainer.AddTab("新标签");
```

### 事件列表

| 事件 | 参数 | 说明 |
|------|------|------|
| `TabSelected` | `TabEventArgs` | 标签被选中时触发 |
| `TabClosing` | `TabCancelEventArgs` | 标签关闭前触发（可取消） |
| `TabClosed` | `TabCancelEventArgs` | 标签关闭后触发 |
| `NewTabButtonClick` | `TabEventArgs` | 点击 (+) 按钮时触发 |
| `TabMoved` | `TabMovedEventArgs` | 标签重排序后触发 |
| `TabMouseClick` | `TabMouseEventArgs` | 鼠标点击标签时触发 |
| `TabMouseDoubleClick` | `TabMouseEventArgs` | 鼠标双击标签时触发 |

### 属性配置

```csharp
// 外观
TabBar.TabBarColor = Color.FromArgb(45, 45, 48);
TabBar.ActiveTabIndicatorColor = Color.FromArgb(0, 122, 204);

// 行为
TabBar.AllowTabReorder = true;      // 允许拖拽重排序
TabBar.ShowCloseButton = true;      // 显示关闭按钮
TabBar.ShowNewTabButton = true;     // 显示新建按钮
TabBar.AllowContextMenu = true;     // 允许右键菜单
```

## 演示程序

**文件位置**: `ClaudeAssist/TabDemoForm.cs`

演示程序展示：

1. 4 个预设标签页（主页、设置、工具箱、文档）
2. 彩色图标（使用颜色方块作为图标）
3. 每个标签页包含：
   - 标题区域
   - 功能说明文本
   - "添加新标签" 按钮
   - "关闭当前标签" 按钮
   - 示例数据表格（DataGridView）

运行演示：

```bash
cd ClaudeAssist
dotnet run
```

## 技术要点

### 1. 双缓冲绘制

```csharp
SetStyle(ControlStyles.UserPaint |
         ControlStyles.AllPaintingInWmPaint |
         ControlStyles.OptimizedDoubleBuffer |
         ControlStyles.ResizeRedraw, true);
```

### 2. 平滑文字渲染

```csharp
g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;
```

### 3. 抗锯齿绘制

```csharp
g.SmoothingMode = SmoothingMode.AntiAlias;
```

### 4. 局部重绘优化

使用 `Invalidate(Rectangle)` 只重绘需要更新的区域，提高性能。

## 扩展建议

1. **添加动画效果**: 标签切换时添加淡入淡出动画
2. **支持垂直标签**: 实现左侧/右侧垂直排列的标签栏
3. **添加快捷键**: Ctrl+Tab 切换标签，Ctrl+W 关闭标签
4. **标签分组**: 支持将相关标签分组显示
5. **标签预览**: 鼠标悬停时显示标签内容预览
6. **持久化**: 保存/恢复标签页状态

## 参考

- DevExpress v25.2 Visual Style Guide
- Windows 11 Design Principles
- Material Design Tab Guidelines
