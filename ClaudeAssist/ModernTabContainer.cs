using System.ComponentModel;

namespace ClaudeAssist
{
    /// <summary>
    /// 完整的标签页容器控件，包含标签栏和内容区域
    /// </summary>
    public class ModernTabContainer : Control
    {
        private ModernTabControl _tabControl;
        private Panel _contentPanel;
        private int _tabBarHeight = 36;

        [Category("Behavior")]
        [DefaultValue(36)]
        public int TabBarHeight
        {
            get => _tabBarHeight;
            set
            {
                _tabBarHeight = value;
                UpdateLayout();
            }
        }

        [Browsable(false)]
        public ModernTabControl TabBar => _tabControl;

        [Browsable(false)]
        public Panel ContentPanel => _contentPanel;

        public event EventHandler<TabEventArgs>? TabSelected
        {
            add => _tabControl.TabSelected += value;
            remove => _tabControl.TabSelected -= value;
        }

        public event EventHandler<TabCancelEventArgs>? TabClosing
        {
            add => _tabControl.TabClosing += value;
            remove => _tabControl.TabClosing -= value;
        }

        public event EventHandler<TabCancelEventArgs>? TabClosed
        {
            add => _tabControl.TabClosed += value;
            remove => _tabControl.TabClosed -= value;
        }

        public event EventHandler<TabEventArgs>? NewTabButtonClick
        {
            add => _tabControl.NewTabButtonClick += value;
            remove => _tabControl.NewTabButtonClick -= value;
        }

        public event EventHandler<TabMovedEventArgs>? TabMoved
        {
            add => _tabControl.TabMoved += value;
            remove => _tabControl.TabMoved -= value;
        }

        public ModernTabContainer()
        {
            SetStyle(ControlStyles.UserPaint |
                    ControlStyles.AllPaintingInWmPaint |
                    ControlStyles.OptimizedDoubleBuffer |
                    ControlStyles.ResizeRedraw, true);

            // 创建标签栏
            _tabControl = new ModernTabControl
            {
                Dock = DockStyle.None,
                Height = _tabBarHeight
            };
            _tabControl.TabSelected += OnTabSelected;
            Controls.Add(_tabControl);

            // 创建内容面板
            _contentPanel = new Panel
            {
                Dock = DockStyle.None,
                BackColor = Color.FromArgb(37, 37, 38),
                BorderStyle = BorderStyle.None
            };
            Controls.Add(_contentPanel);

            UpdateLayout();
        }

        private void OnTabSelected(object? sender, TabEventArgs e)
        {
            UpdateContentPanel();
        }

        private void UpdateLayout()
        {
            if (_tabControl == null || _contentPanel == null) return;

            _tabControl.Height = _tabBarHeight;
            _tabControl.Width = Width;
            _tabControl.Location = new Point(0, 0);

            _contentPanel.Location = new Point(0, _tabBarHeight);
            _contentPanel.Size = new Size(Width, Math.Max(0, Height - _tabBarHeight));
        }

        private void UpdateContentPanel()
        {
            _contentPanel.Controls.Clear();

            var selectedTab = _tabControl.SelectedTab;
            if (selectedTab?.Content != null)
            {
                selectedTab.Content.Dock = DockStyle.Fill;
                _contentPanel.Controls.Add(selectedTab.Content);
            }
        }

        public TabItem AddTab(string title, Control? content = null)
        {
            return _tabControl.AddTab(title, content);
        }

        public TabItem AddTab(string title, Image? icon, Control? content = null)
        {
            return _tabControl.AddTab(title, icon, content);
        }

        public void RemoveTab(int index)
        {
            _tabControl.RemoveTab(index);
        }

        public void RemoveTab(TabItem tab)
        {
            _tabControl.RemoveTab(tab);
        }

        public void ClearTabs()
        {
            _tabControl.ClearTabs();
        }

        public void SelectTab(int index)
        {
            _tabControl.SelectTab(index);
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            UpdateLayout();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

            // 绘制内容区域背景
            using var brush = new SolidBrush(_contentPanel.BackColor);
            e.Graphics.FillRectangle(brush, _contentPanel.Bounds);
        }
    }
}
