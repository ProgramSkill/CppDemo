using System.ComponentModel;
using System.Drawing.Drawing2D;

namespace ClaudeAssist
{
    /// <summary>
    /// DevExpress v25.2 风格的多标签页控件
    /// 特性：
    /// - 现代化扁平设计
    /// - 标签关闭按钮
    /// - 标签拖拽重排序
    /// - 彩色活动标签指示器
    /// - 新增标签按钮 (+)
    /// - 右键上下文菜单
    /// - 平滑动画效果
    /// </summary>
    public class ModernTabControl : Control
    {
        #region 字段

        private List<TabItem> _tabs = new();
        private int _selectedIndex = -1;
        private int _hoverIndex = -1;
        private int _pressedIndex = -1;
        private int _closeButtonHoverIndex = -1;
        private Rectangle _newTabButtonRect;
        private bool _newTabButtonHover = false;
        private bool _newTabButtonPressed = false;
        private int _scrollOffset = 0;
        private bool _isDragging = false;
        private int _dragStartIndex = -1;
        private int _dragCurrentIndex = -1;
        private Point _dragStartPoint;
        private Point _dragCurrentPoint;

        // 滚动按钮
        private Rectangle _scrollLeftRect;
        private Rectangle _scrollRightRect;
        private bool _scrollLeftHover = false;
        private bool _scrollRightHover = false;
        private bool _canScrollLeft = false;
        private bool _canScrollRight = false;

        // 常量
        private const int TAB_HEIGHT = 36;
        private const int TAB_MIN_WIDTH = 140;
        private const int TAB_MAX_WIDTH = 200;
        private const int TAB_PADDING = 8;
        private const int CLOSE_BUTTON_SIZE = 14;
        private const int NEW_TAB_BUTTON_WIDTH = 36;
        private const int SCROLL_BUTTON_WIDTH = 24;
        private const int ICON_SIZE = 16;

        #endregion

        #region 属性

        [Browsable(false)]
        public List<TabItem> Tabs => _tabs;

        [Browsable(false)]
        [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
        public int SelectedIndex
        {
            get => _selectedIndex;
            set
            {
                if (_selectedIndex != value && value >= -1 && value < _tabs.Count)
                {
                    int oldIndex = _selectedIndex;
                    _selectedIndex = value;
                    OnSelectedIndexChanged(oldIndex, value);
                    Invalidate();
                }
            }
        }

        public TabItem? SelectedTab => _selectedIndex >= 0 && _selectedIndex < _tabs.Count ? _tabs[_selectedIndex] : null;

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "45, 45, 48")]
        public Color TabBarColor { get; set; } = Color.FromArgb(45, 45, 48);

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "30, 30, 30")]
        public Color TabBarInactiveColor { get; set; } = Color.FromArgb(30, 30, 30);

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "0, 122, 204")]
        public Color ActiveTabIndicatorColor { get; set; } = Color.FromArgb(0, 122, 204);

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "62, 62, 64")]
        public Color TabHoverColor { get; set; } = Color.FromArgb(62, 62, 64);

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "37, 37, 38")]
        public Color TabActiveColor { get; set; } = Color.FromArgb(37, 37, 38);

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "241, 241, 241")]
        public Color TabTextColor { get; set; } = Color.FromArgb(241, 241, 241);

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "150, 150, 150")]
        public Color TabInactiveTextColor { get; set; } = Color.FromArgb(150, 150, 150);

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "200, 200, 200")]
        public Color CloseButtonHoverColor { get; set; } = Color.FromArgb(200, 200, 200);

        [Category("Appearance")]
        [DefaultValue(typeof(Color), "232, 17, 35")]
        public Color CloseButtonPressedColor { get; set; } = Color.FromArgb(232, 17, 35);

        [Category("Behavior")]
        [DefaultValue(true)]
        public bool AllowTabReorder { get; set; } = true;

        [Category("Behavior")]
        [DefaultValue(true)]
        public bool ShowCloseButton { get; set; } = true;

        [Category("Behavior")]
        [DefaultValue(true)]
        public bool ShowNewTabButton { get; set; } = true;

        [Category("Behavior")]
        [DefaultValue(true)]
        public bool AllowContextMenu { get; set; } = true;

        #endregion

        #region 事件

        public event EventHandler<TabEventArgs>? TabSelected;
        public event EventHandler<TabCancelEventArgs>? TabClosing;
        public event EventHandler<TabCancelEventArgs>? TabClosed;
        public event EventHandler<TabEventArgs>? NewTabButtonClick;
        public event EventHandler<TabMovedEventArgs>? TabMoved;
        public event EventHandler<TabMouseEventArgs>? TabMouseClick;
        public event EventHandler<TabMouseEventArgs>? TabMouseDoubleClick;

        #endregion

        #region 构造函数

        public ModernTabControl()
        {
            SetStyle(ControlStyles.UserPaint |
                    ControlStyles.AllPaintingInWmPaint |
                    ControlStyles.OptimizedDoubleBuffer |
                    ControlStyles.ResizeRedraw, true);

            Height = TAB_HEIGHT;
            BackColor = TabBarInactiveColor;
        }

        #endregion

        #region 公共方法

        public TabItem AddTab(string title, Control? content = null)
        {
            return AddTab(title, null, content);
        }

        public TabItem AddTab(string title, Image? icon, Control? content = null)
        {
            var tab = new TabItem(title, icon, content)
            {
                Bounds = CalculateTabBounds(_tabs.Count)
            };
            _tabs.Add(tab);

            if (_selectedIndex == -1)
                SelectedIndex = 0;

            UpdateTabBounds();
            Invalidate();
            return tab;
        }

        public void RemoveTab(int index)
        {
            if (index < 0 || index >= _tabs.Count) return;

            var tab = _tabs[index];
            var e = new TabCancelEventArgs(tab, index, false);
            TabClosing?.Invoke(this, e);

            if (e.Cancel) return;

            _tabs.RemoveAt(index);

            if (_selectedIndex == index)
            {
                _selectedIndex = Math.Min(index, _tabs.Count - 1);
                if (_selectedIndex >= 0)
                    TabSelected?.Invoke(this, new TabEventArgs(_tabs[_selectedIndex], _selectedIndex));
            }
            else if (_selectedIndex > index)
            {
                _selectedIndex--;
            }

            UpdateTabBounds();
            Invalidate();
            TabClosed?.Invoke(this, new TabCancelEventArgs(tab, index, false));
        }

        public void RemoveTab(TabItem tab)
        {
            int index = _tabs.IndexOf(tab);
            if (index >= 0) RemoveTab(index);
        }

        public void ClearTabs()
        {
            for (int i = _tabs.Count - 1; i >= 0; i--)
            {
                var e = new TabCancelEventArgs(_tabs[i], i, false);
                TabClosing?.Invoke(this, e);
                if (!e.Cancel)
                {
                    var tab = _tabs[i];
                    _tabs.RemoveAt(i);
                    TabClosed?.Invoke(this, new TabCancelEventArgs(tab, i, false));
                }
            }
            _selectedIndex = -1;
            UpdateTabBounds();
            Invalidate();
        }

        public void SelectTab(int index)
        {
            SelectedIndex = index;
        }

        public void MoveTab(int fromIndex, int toIndex)
        {
            if (fromIndex < 0 || fromIndex >= _tabs.Count || toIndex < 0 || toIndex >= _tabs.Count)
                return;

            if (fromIndex == toIndex) return;

            var tab = _tabs[fromIndex];
            _tabs.RemoveAt(fromIndex);
            _tabs.Insert(toIndex, tab);

            if (_selectedIndex == fromIndex)
                _selectedIndex = toIndex;
            else if (_selectedIndex > fromIndex && _selectedIndex <= toIndex)
                _selectedIndex--;
            else if (_selectedIndex < fromIndex && _selectedIndex >= toIndex)
                _selectedIndex++;

            UpdateTabBounds();
            Invalidate();
            TabMoved?.Invoke(this, new TabMovedEventArgs(tab, fromIndex, toIndex));
        }

        #endregion

        #region 绘制

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;

            // 绘制标签栏背景
            using (var brush = new SolidBrush(TabBarInactiveColor))
            {
                g.FillRectangle(brush, ClientRectangle);
            }

            // 绘制底部边框线
            using (var pen = new Pen(Color.FromArgb(60, 60, 60), 1))
            {
                g.DrawLine(pen, 0, Height - 1, Width, Height - 1);
            }

            // 设置裁剪区域（排除滚动按钮和新标签按钮区域）
            int contentWidth = GetTabContentWidth();
            var clipRect = new Rectangle(
                _canScrollLeft ? SCROLL_BUTTON_WIDTH : 0,
                0,
                contentWidth,
                Height);

            // 绘制标签
            for (int i = 0; i < _tabs.Count; i++)
            {
                var tab = _tabs[i];
                var bounds = GetTabBounds(i);

                // 检查标签是否在可见区域
                if (bounds.Right < clipRect.Left || bounds.Left > clipRect.Right)
                    continue;

                bool isActive = i == _selectedIndex;
                bool isHover = i == _hoverIndex && !_isDragging;
                bool isPressed = i == _pressedIndex && !_isDragging;

                DrawTab(g, tab, bounds, isActive, isHover, isPressed);
            }

            // 绘制滚动按钮
            if (_canScrollLeft)
                DrawScrollLeftButton(g);
            if (_canScrollRight)
                DrawScrollRightButton(g);

            // 绘制新标签按钮
            if (ShowNewTabButton)
                DrawNewTabButton(g);
        }

        private void DrawTab(Graphics g, TabItem tab, Rectangle bounds, bool isActive, bool isHover, bool isPressed)
        {
            // 绘制标签背景
            Color bgColor = isActive ? TabActiveColor : (isPressed ? TabHoverColor : (isHover ? TabHoverColor : Color.Transparent));

            if (bgColor != Color.Transparent)
            {
                using var brush = new SolidBrush(bgColor);
                g.FillRectangle(brush, bounds);
            }

            // 绘制活动标签指示器（顶部彩色线条）
            if (isActive)
            {
                using var brush = new SolidBrush(ActiveTabIndicatorColor);
                g.FillRectangle(brush, bounds.Left, 0, bounds.Width, 2);
            }

            // 绘制关闭按钮
            if (ShowCloseButton && (isActive || isHover))
            {
                DrawCloseButton(g, tab, bounds);
            }

            // 绘制图标
            int currentX = bounds.Left + TAB_PADDING;
            if (tab.Icon != null)
            {
                var iconRect = new Rectangle(currentX, (Height - ICON_SIZE) / 2, ICON_SIZE, ICON_SIZE);
                g.DrawImage(tab.Icon, iconRect);
                currentX += ICON_SIZE + 4;
            }

            // 绘制标题
            int textWidth = bounds.Width - (currentX - bounds.Left) - TAB_PADDING;
            if (ShowCloseButton && (isActive || isHover))
                textWidth -= CLOSE_BUTTON_SIZE + TAB_PADDING;

            var textRect = new Rectangle(currentX, 0, Math.Max(0, textWidth), Height);
            using (var brush = new SolidBrush(isActive ? TabTextColor : TabInactiveTextColor))
            using (var font = new Font("Segoe UI", 9f))
            {
                var format = new StringFormat
                {
                    Alignment = StringAlignment.Near,
                    LineAlignment = StringAlignment.Center,
                    Trimming = StringTrimming.EllipsisCharacter
                };
                g.DrawString(tab.Title, font, brush, textRect, format);
            }

            // 绘制右侧分隔线（非活动标签）
            if (!isActive && bounds.Right < Width - (_canScrollRight ? SCROLL_BUTTON_WIDTH : 0) - (ShowNewTabButton ? NEW_TAB_BUTTON_WIDTH : 0))
            {
                using var pen = new Pen(Color.FromArgb(60, 60, 60), 1);
                g.DrawLine(pen, bounds.Right - 1, 8, bounds.Right - 1, Height - 8);
            }

            tab.Bounds = bounds;
        }

        private void DrawCloseButton(Graphics g, TabItem tab, Rectangle tabBounds)
        {
            int closeX = tabBounds.Right - CLOSE_BUTTON_SIZE - TAB_PADDING;
            int closeY = (Height - CLOSE_BUTTON_SIZE) / 2;
            var closeRect = new Rectangle(closeX, closeY, CLOSE_BUTTON_SIZE, CLOSE_BUTTON_SIZE);

            bool isHover = _tabs.IndexOf(tab) == _closeButtonHoverIndex;

            if (isHover)
            {
                Color bgColor = (_pressedIndex >= 0 && _tabs.IndexOf(tab) == _pressedIndex) ? CloseButtonPressedColor : CloseButtonHoverColor;
                using var brush = new SolidBrush(bgColor);
                g.FillEllipse(brush, closeRect);
            }

            using var pen = new Pen(isHover ? Color.White : TabInactiveTextColor, 1.5f);
            int padding = 4;
            g.DrawLine(pen, closeRect.Left + padding, closeRect.Top + padding, closeRect.Right - padding, closeRect.Bottom - padding);
            g.DrawLine(pen, closeRect.Right - padding, closeRect.Top + padding, closeRect.Left + padding, closeRect.Bottom - padding);

            tab.CloseButtonBounds = closeRect;
        }

        private void DrawNewTabButton(Graphics g)
        {
            Color bgColor = _newTabButtonPressed ? TabHoverColor : (_newTabButtonHover ? TabHoverColor : Color.Transparent);

            if (bgColor != Color.Transparent)
            {
                using var brush = new SolidBrush(bgColor);
                g.FillRectangle(brush, _newTabButtonRect);
            }

            using var pen = new Pen(_newTabButtonHover ? TabTextColor : TabInactiveTextColor, 1.5f);
            int cx = _newTabButtonRect.Left + _newTabButtonRect.Width / 2;
            int cy = _newTabButtonRect.Top + _newTabButtonRect.Height / 2;
            int size = 6;

            g.DrawLine(pen, cx - size, cy, cx + size, cy);
            g.DrawLine(pen, cx, cy - size, cx, cy + size);
        }

        private void DrawScrollLeftButton(Graphics g)
        {
            Color bgColor = _scrollLeftHover ? TabHoverColor : Color.Transparent;
            if (bgColor != Color.Transparent)
            {
                using var brush = new SolidBrush(bgColor);
                g.FillRectangle(brush, _scrollLeftRect);
            }

            using var pen = new Pen(_canScrollLeft ? TabTextColor : TabInactiveTextColor, 1.5f);
            int cx = _scrollLeftRect.Left + _scrollLeftRect.Width / 2;
            int cy = _scrollLeftRect.Top + _scrollLeftRect.Height / 2;
            int size = 4;

            g.DrawLine(pen, cx + size, cy - size, cx - size, cy);
            g.DrawLine(pen, cx - size, cy, cx + size, cy + size);
        }

        private void DrawScrollRightButton(Graphics g)
        {
            Color bgColor = _scrollRightHover ? TabHoverColor : Color.Transparent;
            if (bgColor != Color.Transparent)
            {
                using var brush = new SolidBrush(bgColor);
                g.FillRectangle(brush, _scrollRightRect);
            }

            using var pen = new Pen(_canScrollRight ? TabTextColor : TabInactiveTextColor, 1.5f);
            int cx = _scrollRightRect.Left + _scrollRightRect.Width / 2;
            int cy = _scrollRightRect.Top + _scrollRightRect.Height / 2;
            int size = 4;

            g.DrawLine(pen, cx - size, cy - size, cx + size, cy);
            g.DrawLine(pen, cx + size, cy, cx - size, cy + size);
        }

        #endregion

        #region 鼠标处理

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            _dragCurrentPoint = e.Location;

            // 处理拖拽
            if (_isDragging && AllowTabReorder)
            {
                HandleTabDrag(e.Location);
                return;
            }

            // 检查新标签按钮悬停
            bool newTabHover = ShowNewTabButton && _newTabButtonRect.Contains(e.Location);
            if (newTabHover != _newTabButtonHover)
            {
                _newTabButtonHover = newTabHover;
                Invalidate(_newTabButtonRect);
            }

            // 检查滚动按钮悬停
            bool scrollLeftHover = _canScrollLeft && _scrollLeftRect.Contains(e.Location);
            if (scrollLeftHover != _scrollLeftHover)
            {
                _scrollLeftHover = scrollLeftHover;
                Invalidate(_scrollLeftRect);
            }

            bool scrollRightHover = _canScrollRight && _scrollRightRect.Contains(e.Location);
            if (scrollRightHover != _scrollRightHover)
            {
                _scrollRightHover = scrollRightHover;
                Invalidate(_scrollRightRect);
            }

            // 检查标签悬停
            int oldHover = _hoverIndex;
            int oldCloseHover = _closeButtonHoverIndex;
            _hoverIndex = -1;
            _closeButtonHoverIndex = -1;

            for (int i = 0; i < _tabs.Count; i++)
            {
                var bounds = GetTabBounds(i);
                if (bounds.Contains(e.Location))
                {
                    _hoverIndex = i;

                    // 检查关闭按钮悬停
                    if (ShowCloseButton)
                    {
                        if (_tabs[i].CloseButtonBounds.Contains(e.Location))
                        {
                            _closeButtonHoverIndex = i;
                        }
                    }
                    break;
                }
            }

            if (oldHover != _hoverIndex || oldCloseHover != _closeButtonHoverIndex)
                Invalidate();
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);

            if (e.Button != MouseButtons.Left) return;

            // 检查新标签按钮
            if (ShowNewTabButton && _newTabButtonRect.Contains(e.Location))
            {
                _newTabButtonPressed = true;
                Invalidate(_newTabButtonRect);
                return;
            }

            // 检查滚动按钮
            if (_canScrollLeft && _scrollLeftRect.Contains(e.Location))
            {
                ScrollLeft();
                return;
            }

            if (_canScrollRight && _scrollRightRect.Contains(e.Location))
            {
                ScrollRight();
                return;
            }

            // 检查标签点击
            for (int i = 0; i < _tabs.Count; i++)
            {
                var bounds = GetTabBounds(i);
                if (bounds.Contains(e.Location))
                {
                    // 检查关闭按钮点击
                    if (ShowCloseButton && _tabs[i].CloseButtonBounds.Contains(e.Location))
                    {
                        _pressedIndex = i;
                        Invalidate();
                        return;
                    }

                    // 开始拖拽或选择标签
                    _pressedIndex = i;
                    _dragStartPoint = e.Location;
                    _dragStartIndex = i;

                    if (i != _selectedIndex)
                        SelectedIndex = i;

                    Invalidate();
                    return;
                }
            }
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);

            // 处理新标签按钮
            if (_newTabButtonPressed)
            {
                _newTabButtonPressed = false;
                if (_newTabButtonHover && _newTabButtonRect.Contains(e.Location))
                {
                    NewTabButtonClick?.Invoke(this, new TabEventArgs(null, -1));
                }
                Invalidate(_newTabButtonRect);
            }

            // 处理关闭按钮点击
            if (_pressedIndex >= 0 && ShowCloseButton && _closeButtonHoverIndex == _pressedIndex)
            {
                var tab = _tabs[_pressedIndex];
                if (tab.CloseButtonBounds.Contains(e.Location))
                {
                    RemoveTab(_pressedIndex);
                    _pressedIndex = -1;
                    return;
                }
            }

            // 结束拖拽
            if (_isDragging && AllowTabReorder)
            {
                EndTabDrag();
            }

            _pressedIndex = -1;
            _dragStartIndex = -1;
            Invalidate();
        }

        protected override void OnMouseLeave(EventArgs e)
        {
            base.OnMouseLeave(e);

            _hoverIndex = -1;
            _closeButtonHoverIndex = -1;
            _newTabButtonHover = false;
            _scrollLeftHover = false;
            _scrollRightHover = false;
            Invalidate();
        }

        protected override void OnMouseClick(MouseEventArgs e)
        {
            base.OnMouseClick(e);

            for (int i = 0; i < _tabs.Count; i++)
            {
                if (GetTabBounds(i).Contains(e.Location))
                {
                    TabMouseClick?.Invoke(this, new TabMouseEventArgs(_tabs[i], i, e.Button, e.Location));
                    break;
                }
            }

            // 右键菜单
            if (e.Button == MouseButtons.Right && AllowContextMenu)
            {
                for (int i = 0; i < _tabs.Count; i++)
                {
                    if (GetTabBounds(i).Contains(e.Location))
                    {
                        ShowTabContextMenu(_tabs[i], i, e.Location);
                        break;
                    }
                }
            }
        }

        protected override void OnMouseDoubleClick(MouseEventArgs e)
        {
            base.OnMouseDoubleClick(e);

            for (int i = 0; i < _tabs.Count; i++)
            {
                if (GetTabBounds(i).Contains(e.Location))
                {
                    TabMouseDoubleClick?.Invoke(this, new TabMouseEventArgs(_tabs[i], i, e.Button, e.Location));
                    break;
                }
            }
        }

        #endregion

        #region 拖拽重排序

        private void HandleTabDrag(Point location)
        {
            if (Math.Abs(location.X - _dragStartPoint.X) < 5 &&
                Math.Abs(location.Y - _dragStartPoint.Y) < 5)
                return;

            _isDragging = true;

            // 找到鼠标下的标签位置
            for (int i = 0; i < _tabs.Count; i++)
            {
                if (i == _dragStartIndex) continue;

                var bounds = GetTabBounds(i);
                if (bounds.Contains(location))
                {
                    // 交换标签位置
                    if (_dragCurrentIndex != i)
                    {
                        _dragCurrentIndex = i;
                        MoveTab(_dragStartIndex, i);
                        _dragStartIndex = i;
                    }
                    break;
                }
            }
        }

        private void EndTabDrag()
        {
            _isDragging = false;
            _dragCurrentIndex = -1;
        }

        #endregion

        #region 滚动

        private void ScrollLeft()
        {
            _scrollOffset = Math.Max(0, _scrollOffset - TAB_MIN_WIDTH);
            UpdateTabBounds();
            Invalidate();
        }

        private void ScrollRight()
        {
            int maxOffset = Math.Max(0, GetTotalTabsWidth() - GetTabContentWidth());
            _scrollOffset = Math.Min(maxOffset, _scrollOffset + TAB_MIN_WIDTH);
            UpdateTabBounds();
            Invalidate();
        }

        #endregion

        #region 上下文菜单

        private void ShowTabContextMenu(TabItem tab, int index, Point location)
        {
            var menu = new ContextMenuStrip();
            menu.Renderer = new ModernMenuRenderer();

            var itemClose = new ToolStripMenuItem("关闭") { ShortcutKeyDisplayString = "Ctrl+W" };
            itemClose.Click += (s, e) => RemoveTab(index);

            var itemCloseOthers = new ToolStripMenuItem("关闭其他标签页");
            itemCloseOthers.Click += (s, e) =>
            {
                for (int i = _tabs.Count - 1; i >= 0; i--)
                {
                    if (i != index) RemoveTab(i);
                }
            };

            var itemCloseAll = new ToolStripMenuItem("关闭所有标签页");
            itemCloseAll.Click += (s, e) => ClearTabs();

            var itemDuplicate = new ToolStripMenuItem("复制标签页");
            itemDuplicate.Click += (s, e) =>
            {
                AddTab(tab.Title + " (副本)", tab.Icon);
            };

            menu.Items.Add(itemClose);
            menu.Items.Add(itemCloseOthers);
            menu.Items.Add(itemCloseAll);
            menu.Items.Add(new ToolStripSeparator());
            menu.Items.Add(itemDuplicate);

            menu.Show(this, location);
        }

        #endregion

        #region 辅助方法

        private Rectangle CalculateTabBounds(int index)
        {
            int x = (_canScrollLeft ? SCROLL_BUTTON_WIDTH : 0) + index * CalculateTabWidth() - _scrollOffset;
            int width = CalculateTabWidth();
            return new Rectangle(x, 0, width, Height);
        }

        private Rectangle GetTabBounds(int index)
        {
            if (index < 0 || index >= _tabs.Count)
                return Rectangle.Empty;
            return CalculateTabBounds(index);
        }

        private int CalculateTabWidth()
        {
            int availableWidth = GetTabContentWidth();
            if (_tabs.Count == 0) return TAB_MIN_WIDTH;

            int width = availableWidth / _tabs.Count;
            return Math.Max(TAB_MIN_WIDTH, Math.Min(TAB_MAX_WIDTH, width));
        }

        private int GetTabContentWidth()
        {
            int width = Width;
            if (_canScrollLeft) width -= SCROLL_BUTTON_WIDTH;
            if (_canScrollRight) width -= SCROLL_BUTTON_WIDTH;
            if (ShowNewTabButton) width -= NEW_TAB_BUTTON_WIDTH;
            return Math.Max(0, width);
        }

        private int GetTotalTabsWidth()
        {
            return _tabs.Count * CalculateTabWidth();
        }

        private void UpdateTabBounds()
        {
            // 计算是否需要滚动
            int totalWidth = GetTotalTabsWidth();
            int contentWidth = Width - (ShowNewTabButton ? NEW_TAB_BUTTON_WIDTH : 0);

            _canScrollLeft = _scrollOffset > 0;
            _canScrollRight = totalWidth > contentWidth && _scrollOffset < totalWidth - contentWidth + SCROLL_BUTTON_WIDTH;

            // 更新滚动按钮区域
            _scrollLeftRect = new Rectangle(0, 0, SCROLL_BUTTON_WIDTH, Height);
            _scrollRightRect = new Rectangle(Width - (ShowNewTabButton ? NEW_TAB_BUTTON_WIDTH : 0) - SCROLL_BUTTON_WIDTH, 0, SCROLL_BUTTON_WIDTH, Height);

            // 更新新标签按钮区域
            _newTabButtonRect = new Rectangle(Width - NEW_TAB_BUTTON_WIDTH, 0, NEW_TAB_BUTTON_WIDTH, Height);
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            UpdateTabBounds();
        }

        private void OnSelectedIndexChanged(int oldIndex, int newIndex)
        {
            if (newIndex >= 0 && newIndex < _tabs.Count)
            {
                TabSelected?.Invoke(this, new TabEventArgs(_tabs[newIndex], newIndex));
            }
        }

        #endregion
    }

    #region 支持类

    public class TabItem
    {
        public string Title { get; set; }
        public Image? Icon { get; set; }
        public Control? Content { get; set; }
        public object? Tag { get; set; }
        internal Rectangle Bounds { get; set; }
        internal Rectangle CloseButtonBounds { get; set; }

        public TabItem(string title, Image? icon = null, Control? content = null)
        {
            Title = title;
            Icon = icon;
            Content = content;
        }
    }

    public class TabEventArgs : EventArgs
    {
        public TabItem? Tab { get; }
        public int Index { get; }

        public TabEventArgs(TabItem? tab, int index)
        {
            Tab = tab;
            Index = index;
        }
    }

    public class TabCancelEventArgs : CancelEventArgs
    {
        public TabItem Tab { get; }
        public int Index { get; }

        public TabCancelEventArgs(TabItem tab, int index, bool cancel) : base(cancel)
        {
            Tab = tab;
            Index = index;
        }
    }

    public class TabMovedEventArgs : EventArgs
    {
        public TabItem Tab { get; }
        public int FromIndex { get; }
        public int ToIndex { get; }

        public TabMovedEventArgs(TabItem tab, int fromIndex, int toIndex)
        {
            Tab = tab;
            FromIndex = fromIndex;
            ToIndex = toIndex;
        }
    }

    public class TabMouseEventArgs : EventArgs
    {
        public TabItem Tab { get; }
        public int Index { get; }
        public MouseButtons Button { get; }
        public Point Location { get; }

        public TabMouseEventArgs(TabItem tab, int index, MouseButtons button, Point location)
        {
            Tab = tab;
            Index = index;
            Button = button;
            Location = location;
        }
    }

    public class ModernMenuRenderer : ToolStripProfessionalRenderer
    {
        public ModernMenuRenderer() : base(new ModernColorTable()) { }

        protected override void OnRenderMenuItemBackground(ToolStripItemRenderEventArgs e)
        {
            if (e.Item.Selected)
            {
                using var brush = new SolidBrush(Color.FromArgb(62, 62, 64));
                e.Graphics.FillRectangle(brush, e.Item.ContentRectangle);
            }
        }
    }

    public class ModernColorTable : ProfessionalColorTable
    {
        public override Color ToolStripDropDownBackground => Color.FromArgb(45, 45, 48);
        public override Color MenuBorder => Color.FromArgb(60, 60, 60);
        public override Color MenuItemBorder => Color.Transparent;
        public override Color MenuItemSelected => Color.FromArgb(62, 62, 64);
        public override Color MenuItemSelectedGradientBegin => Color.FromArgb(62, 62, 64);
        public override Color MenuItemSelectedGradientEnd => Color.FromArgb(62, 62, 64);
        public override Color SeparatorDark => Color.FromArgb(60, 60, 60);
        public override Color SeparatorLight => Color.FromArgb(60, 60, 60);
    }

    #endregion
}
