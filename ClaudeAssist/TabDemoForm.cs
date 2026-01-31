namespace ClaudeAssist
{
    /// <summary>
    /// DevExpress v25.2 风格多标签页演示窗体
    /// </summary>
    public class TabDemoForm : ModernForm
    {
        private ModernTabContainer _tabContainer;
        private int _tabCounter = 1;
        private ImageList _imageList;

        public TabDemoForm()
        {
            Text = "ModernTabControl Demo - DevExpress v25.2 Style";
            Size = new Size(1200, 800);

            InitializeComponents();
            InitializeDemoTabs();
        }

        private void InitializeComponents()
        {
            // 创建图标列表
            _imageList = new ImageList();
            _imageList.ColorDepth = ColorDepth.Depth32Bit;
            _imageList.ImageSize = new Size(16, 16);

            // 添加简单的图标颜色块
            _imageList.Images.Add(CreateColorIcon(Color.FromArgb(0, 122, 204)));    // 蓝色 - 文件
            _imageList.Images.Add(CreateColorIcon(Color.FromArgb(0, 168, 107)));    // 绿色 - 设置
            _imageList.Images.Add(CreateColorIcon(Color.FromArgb(255, 127, 39)));   // 橙色 - 工具
            _imageList.Images.Add(CreateColorIcon(Color.FromArgb(128, 100, 162)));  // 紫色 - 帮助
            _imageList.Images.Add(CreateColorIcon(Color.FromArgb(192, 80, 77)));    // 红色 - 警告

            // 创建标签容器
            _tabContainer = new ModernTabContainer
            {
                Dock = DockStyle.Fill,
                TabBarHeight = 36
            };

            // 自定义标签栏颜色主题（DevExpress v25.2 风格）
            _tabContainer.TabBar.TabBarColor = Color.FromArgb(45, 45, 48);
            _tabContainer.TabBar.TabBarInactiveColor = Color.FromArgb(30, 30, 30);
            _tabContainer.TabBar.ActiveTabIndicatorColor = Color.FromArgb(0, 122, 204);
            _tabContainer.TabBar.TabHoverColor = Color.FromArgb(62, 62, 64);
            _tabContainer.TabBar.TabActiveColor = Color.FromArgb(37, 37, 38);
            _tabContainer.TabBar.TabTextColor = Color.FromArgb(241, 241, 241);
            _tabContainer.TabBar.TabInactiveTextColor = Color.FromArgb(150, 150, 150);
            _tabContainer.TabBar.CloseButtonHoverColor = Color.FromArgb(200, 200, 200);
            _tabContainer.TabBar.CloseButtonPressedColor = Color.FromArgb(232, 17, 35);

            // 订阅事件
            _tabContainer.NewTabButtonClick += OnNewTabButtonClick;
            _tabContainer.TabSelected += OnTabSelected;
            _tabContainer.TabClosing += OnTabClosing;
            _tabContainer.TabClosed += OnTabClosed;
            _tabContainer.TabMoved += OnTabMoved;

            Controls.Add(_tabContainer);
        }

        private void InitializeDemoTabs()
        {
            // 添加一些示例标签页
            AddNewTab("主页", 0);
            AddNewTab("设置", 1);
            AddNewTab("工具箱", 2);
            AddNewTab("文档", 3);
        }

        private void AddNewTab(string title = "", int iconIndex = -1)
        {
            if (string.IsNullOrEmpty(title))
            {
                title = $"新标签 {_tabCounter}";
            }

            _tabCounter++;

            // 创建内容面板
            var contentPanel = CreateContentPanel(title);

            // 添加标签
            Image? icon = iconIndex >= 0 ? _imageList.Images[iconIndex] : null;
            _tabContainer.AddTab(title, icon, contentPanel);
        }

        private Panel CreateContentPanel(string title)
        {
            var panel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(37, 37, 38),
                Padding = new Padding(20)
            };

            // 标题标签
            var lblTitle = new Label
            {
                Text = title,
                Font = new Font("Segoe UI Light", 32, FontStyle.Regular),
                ForeColor = Color.FromArgb(241, 241, 241),
                AutoSize = true,
                Location = new Point(40, 40)
            };
            panel.Controls.Add(lblTitle);

            // 信息标签
            var lblInfo = new Label
            {
                Text = $"这是 \"{title}\" 标签页的内容区域。\n\n" +
                       "功能特性：\n" +
                       "• 点击标签切换内容\n" +
                       "• 悬停显示关闭按钮\n" +
                       "• 拖拽标签重排序\n" +
                       "• 右键上下文菜单\n" +
                       "• + 按钮新建标签\n" +
                       "• 滚动按钮处理溢出",
                Font = new Font("Segoe UI", 10, FontStyle.Regular),
                ForeColor = Color.FromArgb(150, 150, 150),
                AutoSize = true,
                Location = new Point(40, 100)
            };
            panel.Controls.Add(lblInfo);

            // 添加一些示例按钮
            var btnAddTab = new Button
            {
                Text = "添加新标签",
                Location = new Point(40, 250),
                Size = new Size(120, 32),
                FlatStyle = FlatStyle.Flat,
                BackColor = Color.FromArgb(0, 122, 204),
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 9),
                Cursor = Cursors.Hand
            };
            btnAddTab.FlatAppearance.BorderSize = 0;
            btnAddTab.Click += (s, e) => AddNewTab();
            panel.Controls.Add(btnAddTab);

            var btnCloseTab = new Button
            {
                Text = "关闭当前标签",
                Location = new Point(180, 250),
                Size = new Size(120, 32),
                FlatStyle = FlatStyle.Flat,
                BackColor = Color.FromArgb(62, 62, 64),
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 9),
                Cursor = Cursors.Hand
            };
            btnCloseTab.FlatAppearance.BorderSize = 0;
            btnCloseTab.Click += (s, e) =>
            {
                if (_tabContainer.TabBar.Tabs.Count > 1)
                {
                    int currentIndex = _tabContainer.TabBar.SelectedIndex;
                    if (currentIndex >= 0)
                        _tabContainer.RemoveTab(currentIndex);
                }
            };
            panel.Controls.Add(btnCloseTab);

            // 添加示例数据网格
            var dataGrid = CreateSampleDataGrid();
            dataGrid.Location = new Point(40, 320);
            dataGrid.Size = new Size(600, 250);
            panel.Controls.Add(dataGrid);

            return panel;
        }

        private DataGridView CreateSampleDataGrid()
        {
            var grid = new DataGridView
            {
                BackgroundColor = Color.FromArgb(45, 45, 48),
                BorderStyle = BorderStyle.None,
                ColumnHeadersDefaultCellStyle = new DataGridViewCellStyle
                {
                    BackColor = Color.FromArgb(62, 62, 64),
                    ForeColor = Color.FromArgb(241, 241, 241),
                    Font = new Font("Segoe UI", 9),
                    Alignment = DataGridViewContentAlignment.MiddleLeft
                },
                DefaultCellStyle = new DataGridViewCellStyle
                {
                    BackColor = Color.FromArgb(45, 45, 48),
                    ForeColor = Color.FromArgb(241, 241, 241),
                    Font = new Font("Segoe UI", 9),
                    SelectionBackColor = Color.FromArgb(0, 122, 204),
                    SelectionForeColor = Color.White
                },
                GridColor = Color.FromArgb(60, 60, 60),
                ColumnHeadersBorderStyle = DataGridViewHeaderBorderStyle.None,
                CellBorderStyle = DataGridViewCellBorderStyle.SingleHorizontal,
                EnableHeadersVisualStyles = false,
                RowHeadersVisible = false,
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect,
                ReadOnly = true,
                AllowUserToAddRows = false,
                AllowUserToDeleteRows = false
            };

            grid.Columns.Add("ID", "编号");
            grid.Columns.Add("Name", "名称");
            grid.Columns.Add("Status", "状态");
            grid.Columns.Add("Date", "日期");

            // 添加示例数据
            grid.Rows.Add("001", "项目 Alpha", "进行中", DateTime.Now.ToString("yyyy-MM-dd"));
            grid.Rows.Add("002", "项目 Beta", "已完成", DateTime.Now.AddDays(-5).ToString("yyyy-MM-dd"));
            grid.Rows.Add("003", "项目 Gamma", "待开始", DateTime.Now.AddDays(3).ToString("yyyy-MM-dd"));
            grid.Rows.Add("004", "项目 Delta", "进行中", DateTime.Now.ToString("yyyy-MM-dd"));
            grid.Rows.Add("005", "项目 Epsilon", "暂停", DateTime.Now.AddDays(-2).ToString("yyyy-MM-dd"));

            return grid;
        }

        private Bitmap CreateColorIcon(Color color)
        {
            var bmp = new Bitmap(16, 16);
            using (var g = Graphics.FromImage(bmp))
            {
                using var brush = new SolidBrush(color);
                g.FillRectangle(brush, 2, 2, 12, 12);
                using var pen = new Pen(Color.FromArgb(100, 100, 100), 1);
                g.DrawRectangle(pen, 2, 2, 11, 11);
            }
            return bmp;
        }

        #region 事件处理

        private void OnNewTabButtonClick(object? sender, TabEventArgs e)
        {
            AddNewTab();
        }

        private void OnTabSelected(object? sender, TabEventArgs e)
        {
            // 标签选中时的处理
            System.Diagnostics.Debug.WriteLine($"Selected tab: {e.Tab?.Title} (Index: {e.Index})");
        }

        private void OnTabClosing(object? sender, TabCancelEventArgs e)
        {
            // 标签关闭前的处理（可以取消关闭）
            System.Diagnostics.Debug.WriteLine($"Closing tab: {e.Tab.Title} (Index: {e.Index})");
        }

        private void OnTabClosed(object? sender, TabCancelEventArgs e)
        {
            // 标签关闭后的处理
            System.Diagnostics.Debug.WriteLine($"Closed tab: {e.Tab.Title} (Index: {e.Index})");
        }

        private void OnTabMoved(object? sender, TabMovedEventArgs e)
        {
            // 标签移动后的处理
            System.Diagnostics.Debug.WriteLine($"Moved tab: {e.Tab.Title} from {e.FromIndex} to {e.ToIndex}");
        }
        #endregion
    }
}
