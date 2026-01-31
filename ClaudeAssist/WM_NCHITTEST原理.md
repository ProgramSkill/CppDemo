# WM_NCHITTEST 原理详解

## 什么是 WM_NCHITTEST

`WM_NCHITTEST` 是 Windows 消息系统的核心机制之一：

- **WM** - Windows Message（Windows 消息）
- **NC** - Non-Client（非客户区）
- **HIT TEST** - 命中测试

用于**告诉系统鼠标当前位于窗口的哪个区域**。

## 窗口区域划分

```
┌─────────────────────────────────────────┐
│  非客户区 (Non-Client Area)              │
│  ┌─────────────────────────────────┐    │
│  │ 标题栏 (HTCAPTION)              │    │
│  ├─────────────────────────────────┤    │
│  │                                 │    │
│  │   客户区 (Client Area)          │    │
│  │   HTCLIENT                      │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│  边框区域                                │
└─────────────────────────────────────────┘
```

## 工作流程

```
用户移动/点击鼠标
        ↓
Windows 发送 WM_NCHITTEST 消息
        ↓
程序返回一个区域代码 (如 HTRIGHT = 11)
        ↓
Windows 知道"鼠标在右边框上"
        ↓
Windows 自动发送 WM_NCLBUTTONDOWN (非客户区左键按下)
        ↓
Windows 进入调整大小模式，自动处理：
    - 捕获鼠标 (SetCapture)
    - 显示调整大小光标
    - 追踪鼠标移动
    - 实时调整窗口尺寸
    - 松开鼠标后结束
```

## 返回值完整表格

| 返回值 | 常量名 | 含义 | Windows 自动行为 |
|--------|--------|------|------------------|
| 0 | `HTERROR` | 错误 | 播放错误提示音 |
| 1 | `HTCLIENT` | 客户区 | 正常鼠标事件 |
| 2 | `HTCAPTION` | 标题栏 | **拖动移动窗口** |
| 3 | `HTSYSMENU` | 系统菜单 | 显示系统菜单 |
| 4 | `HTGROWBOX` | 大小调整框 | 同 HTSIZE |
| 5 | `HTMENU` | 菜单栏 | 激活菜单 |
| 6 | `HTHSCROLL` | 水平滚动条 | 滚动操作 |
| 7 | `HTVSCROLL` | 垂直滚动条 | 滚动操作 |
| 8 | `HTMINBUTTON` | 最小化按钮 | 最小化窗口 |
| 9 | `HTMAXBUTTON` | 最大化按钮 | 最大化/还原窗口 |
| 10 | `HTLEFT` | 左边框 | **向左调整宽度** ↔ |
| 11 | `HTRIGHT` | 右边框 | **向右调整宽度** ↔ |
| 12 | `HTTOP` | 上边框 | **向上调整高度** ↕ |
| 13 | `HTTOPLEFT` | 左上角 | **对角调整** ↖↘ |
| 14 | `HTTOPRIGHT` | 右上角 | **对角调整** ↗↙ |
| 15 | `HTBOTTOM` | 下边框 | **向下调整高度** ↕ |
| 16 | `HTBOTTOMLEFT` | 左下角 | **对角调整** ↙↗ |
| 17 | `HTBOTTOMRIGHT` | 右下角 | **对角调整** ↘↖ |
| 18 | `HTBORDER` | 不可调整边框 | 无操作 |
| 19 | `HTOBJECT` | 对象 | - |
| 20 | `HTCLOSE` | 关闭按钮 | 关闭窗口 |
| 21 | `HTHELP` | 帮助按钮 | 进入帮助模式 |
| -1 | `HTNOWHERE` | 不在窗口上 | 忽略 |
| -2 | `HTTRANSPARENT` | 透明区域 | 穿透到下层窗口 |

## 代码实现示例

```csharp
protected override void WndProc(ref Message m)
{
    base.WndProc(ref m);

    if (m.Msg == 0x84) // WM_NCHITTEST
    {
        // 从 LParam 解析屏幕坐标
        // 低16位 = X坐标，高16位 = Y坐标
        Point p = new Point(m.LParam.ToInt32() & 0xFFFF, m.LParam.ToInt32() >> 16);
        p = PointToClient(p); // 转换为客户端坐标

        if (_isMaximized) return; // 最大化时不允许调整大小
        
        // 检测鼠标是否在边缘区域（6像素内）
        bool left = p.X < RESIZE_BORDER;
        bool right = p.X > Width - RESIZE_BORDER;
        bool top = p.Y < RESIZE_BORDER;
        bool bottom = p.Y > Height - RESIZE_BORDER;
        
        // 返回对应的区域代码
        if (left && top) m.Result = (IntPtr)13;        // HTTOPLEFT
        else if (right && top) m.Result = (IntPtr)14;  // HTTOPRIGHT
        else if (left && bottom) m.Result = (IntPtr)16; // HTBOTTOMLEFT
        else if (right && bottom) m.Result = (IntPtr)17; // HTBOTTOMRIGHT
        else if (left) m.Result = (IntPtr)10;          // HTLEFT
        else if (right) m.Result = (IntPtr)11;         // HTRIGHT
        else if (top) m.Result = (IntPtr)12;           // HTTOP
        else if (bottom) m.Result = (IntPtr)15;        // HTBOTTOM
        else if (p.Y < TITLE_HEIGHT) m.Result = (IntPtr)2; // HTCAPTION (拖动)
    }
}
```

## 为什么需要 WM_NCHITTEST

当窗体设置 `FormBorderStyle.None`（无边框）时：
- Windows 不知道哪里是标题栏
- Windows 不知道哪里是边框
- 窗口无法拖动和调整大小

通过响应 `WM_NCHITTEST`，你可以：
1. **自定义标题栏区域** → 让自绘的标题栏支持拖动
2. **自定义边框区域** → 让无边框窗口支持调整大小
3. **完全由系统处理拖拽逻辑** → 不需要自己写复杂的拖拽代码

## 你不需要写的代码

使用 `WM_NCHITTEST` 后，以下操作全部由 Windows 内核自动处理：

- ❌ 鼠标捕获 (`SetCapture`)
- ❌ 鼠标移动追踪
- ❌ 计算新的窗口尺寸
- ❌ 调用 `SetWindowPos` 改变窗口大小
- ❌ 处理鼠标释放

## 总结

`WM_NCHITTEST` 是**声明式**的编程方式：
- 你只需告诉 Windows **鼠标在哪**
- Windows 负责**所有行为**

这是 Windows 窗口系统设计的精妙之处——通过简单的消息返回值，就能获得完整的窗口管理功能。
