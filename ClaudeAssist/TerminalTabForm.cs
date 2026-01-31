namespace ClaudeAssist
{
    /// <summary>
    /// 多标签页终端窗口 - 每个标签页是一个 ConPTY 终端
    /// </summary>
    public class TerminalTabForm : ModernForm
    {
        private ModernTabContainer _tabContainer;
        private int _terminalCounter = 1;
        private ImageList _imageList;

        public TerminalTabForm()
        {
            Text = "Terminal - Multi-Tab ConPTY";
            Size = new Size(1200, 800);
            MinimumSize = new Size(800, 600);

            InitializeComponents();
            CreateNewTerminalTab();
        }

        private void InitializeComponents()
        {
            // 创建图标列表
            _imageList = new ImageList
            {
                ColorDepth = ColorDepth.Depth32Bit,
                ImageSize = new Size(16, 16)
            };

            // 终端图标（绿色方块表示活动终端）
            _imageList.Images.Add("terminal", CreateTerminalIcon(Color.FromArgb(13, 188, 121)));
            _imageList.Images.Add("terminal_inactive", CreateTerminalIcon(Color.FromArgb(100, 100, 100)));

            // 创建标签容器
            _tabContainer = new ModernTabContainer
            {
                Dock = DockStyle.Fill,
                TabBarHeight = 36
            };

            // 设置深色主题
            _tabContainer.TabBar.TabBarColor = Color.FromArgb(45, 45, 48);
            _tabContainer.TabBar.TabBarInactiveColor = Color.FromArgb(30, 30, 30);
            _tabContainer.TabBar.ActiveTabIndicatorColor = Color.FromArgb(13, 188, 121); // 绿色指示器
            _tabContainer.TabBar.TabHoverColor = Color.FromArgb(62, 62, 64);
            _tabContainer.TabBar.TabActiveColor = Color.FromArgb(37, 37, 38);
            _tabContainer.TabBar.TabTextColor = Color.FromArgb(241, 241, 241);
            _tabContainer.TabBar.TabInactiveTextColor = Color.FromArgb(150, 150, 150);
            _tabContainer.TabBar.CloseButtonHoverColor = Color.FromArgb(200, 200, 200);
            _tabContainer.TabBar.CloseButtonPressedColor = Color.FromArgb(232, 17, 35);

            // 订阅事件
            _tabContainer.NewTabButtonClick += OnNewTabButtonClick;
            _tabContainer.TabClosing += OnTabClosing;
            _tabContainer.TabClosed += OnTabClosed;
            _tabContainer.TabSelected += OnTabSelected;

            Controls.Add(_tabContainer);

            // 设置键盘快捷键
            KeyPreview = true;
        }

        private void CreateNewTerminalTab(string? shellPath = null, string? workingDirectory = null)
        {
            string title = $"Terminal {_terminalCounter}";
            _terminalCounter++;

            // 创建终端控件
            var terminal = new ConPtyTerminal
            {
                Dock = DockStyle.Fill,
                ShellPath = shellPath ?? GetDefaultShell(),
                WorkingDirectory = workingDirectory ?? Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
            };

            // 订阅终端事件
            terminal.ProcessExited += OnTerminalProcessExited;

            // 创建容器面板
            var panel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.FromArgb(30, 30, 30),
                Padding = new Padding(0)
            };
            panel.Controls.Add(terminal);

            // 点击面板时聚焦终端
            panel.Click += (s, ev) => terminal.Focus();
            panel.GotFocus += (s, ev) => terminal.Focus();

            // 添加标签页
            var tab = _tabContainer.AddTab(title, _imageList.Images["terminal"], panel);
            tab.Tag = terminal;

            // 启动终端
            terminal.Start();

            // 选中新标签页
            _tabContainer.SelectTab(_tabContainer.TabBar.Tabs.Count - 1);

            // 聚焦终端
            terminal.Focus();
        }

        private string GetDefaultShell()
        {
            // 优先使用 PowerShell Core，然后是 Windows PowerShell，最后是 cmd
            string[] shells = new[]
            {
                @"C:\Program Files\PowerShell\7\pwsh.exe",
                @"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
                "cmd.exe"
            };

            foreach (var shell in shells)
            {
                if (File.Exists(shell) || shell == "cmd.exe")
                {
                    return shell;
                }
            }

            return "cmd.exe";
        }

        private Bitmap CreateTerminalIcon(Color color)
        {
            var bmp = new Bitmap(16, 16);
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;

                // 绘制终端图标（简化的命令行符号 >_）
                using var brush = new SolidBrush(color);
                using var pen = new Pen(color, 2f);

                // 外框
                g.DrawRectangle(pen, 2, 2, 11, 11);

                // > 符号
                g.DrawLine(pen, 4, 6, 7, 8);
                g.DrawLine(pen, 7, 8, 4, 10);

                // _ 光标
                g.DrawLine(pen, 9, 10, 12, 10);
            }
            return bmp;
        }

        #region 事件处理

        private void OnNewTabButtonClick(object? sender, TabEventArgs e)
        {
            CreateNewTerminalTab();
        }

        private void OnTabClosing(object? sender, TabCancelEventArgs e)
        {
            // 获取终端控件
            if (e.Tab.Tag is ConPtyTerminal terminal && terminal.IsRunning)
            {
                // 可以在这里询问用户是否确认关闭正在运行的终端
                // 目前直接允许关闭
            }
        }

        private void OnTabClosed(object? sender, TabCancelEventArgs e)
        {
            // 停止并释放终端
            if (e.Tab.Tag is ConPtyTerminal terminal)
            {
                terminal.Stop();
                terminal.Dispose();
            }

            // 如果没有标签页了，创建一个新的
            if (_tabContainer.TabBar.Tabs.Count == 0)
            {
                CreateNewTerminalTab();
            }
        }

        private void OnTabSelected(object? sender, TabEventArgs e)
        {
            // 聚焦选中的终端
            if (e.Tab?.Tag is ConPtyTerminal terminal)
            {
                BeginInvoke(() => terminal.Focus());
            }
        }

        private void OnTerminalProcessExited(object? sender, EventArgs e)
        {
            if (sender is ConPtyTerminal terminal)
            {
                // 找到对应的标签页并更新图标
                foreach (var tab in _tabContainer.TabBar.Tabs)
                {
                    if (tab.Tag == terminal)
                    {
                        tab.Icon = _imageList.Images["terminal_inactive"];
                        _tabContainer.TabBar.Invalidate();
                        break;
                    }
                }
            }
        }

        #endregion

        #region 键盘快捷键

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            // Ctrl+Shift+T: 新建标签页
            if (keyData == (Keys.Control | Keys.Shift | Keys.T))
            {
                CreateNewTerminalTab();
                return true;
            }

            // Ctrl+W: 关闭当前标签页
            if (keyData == (Keys.Control | Keys.W))
            {
                int currentIndex = _tabContainer.TabBar.SelectedIndex;
                if (currentIndex >= 0)
                {
                    _tabContainer.RemoveTab(currentIndex);
                }
                return true;
            }

            // Ctrl+Tab: 切换到下一个标签页
            if (keyData == (Keys.Control | Keys.Tab))
            {
                int count = _tabContainer.TabBar.Tabs.Count;
                if (count > 1)
                {
                    int next = (_tabContainer.TabBar.SelectedIndex + 1) % count;
                    _tabContainer.SelectTab(next);
                }
                return true;
            }

            // Ctrl+Shift+Tab: 切换到上一个标签页
            if (keyData == (Keys.Control | Keys.Shift | Keys.Tab))
            {
                int count = _tabContainer.TabBar.Tabs.Count;
                if (count > 1)
                {
                    int prev = (_tabContainer.TabBar.SelectedIndex - 1 + count) % count;
                    _tabContainer.SelectTab(prev);
                }
                return true;
            }

            // Ctrl+1-9: 切换到指定标签页
            if (keyData >= (Keys.Control | Keys.D1) && keyData <= (Keys.Control | Keys.D9))
            {
                int index = (keyData & Keys.KeyCode) - Keys.D1;
                if (index < _tabContainer.TabBar.Tabs.Count)
                {
                    _tabContainer.SelectTab(index);
                }
                return true;
            }

            return base.ProcessCmdKey(ref msg, keyData);
        }

        #endregion

        #region 菜单

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);

            // 创建菜单栏
            var menuStrip = new MenuStrip
            {
                BackColor = Color.FromArgb(45, 45, 48),
                ForeColor = Color.FromArgb(241, 241, 241),
                Renderer = new ModernMenuRenderer()
            };

            // 文件菜单
            var fileMenu = new ToolStripMenuItem("文件(&F)");
            fileMenu.DropDownItems.Add(new ToolStripMenuItem("新建终端(&N)", null, (s, ev) => CreateNewTerminalTab())
            {
                ShortcutKeyDisplayString = "Ctrl+Shift+T"
            });
            fileMenu.DropDownItems.Add(new ToolStripMenuItem("新建 PowerShell", null, (s, ev) =>
                CreateNewTerminalTab(@"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")));
            fileMenu.DropDownItems.Add(new ToolStripMenuItem("新建 CMD", null, (s, ev) =>
                CreateNewTerminalTab("cmd.exe")));
            fileMenu.DropDownItems.Add(new ToolStripSeparator());
            fileMenu.DropDownItems.Add(new ToolStripMenuItem("关闭标签页(&C)", null, (s, ev) =>
            {
                int idx = _tabContainer.TabBar.SelectedIndex;
                if (idx >= 0) _tabContainer.RemoveTab(idx);
            })
            {
                ShortcutKeyDisplayString = "Ctrl+W"
            });
            fileMenu.DropDownItems.Add(new ToolStripSeparator());
            fileMenu.DropDownItems.Add(new ToolStripMenuItem("退出(&X)", null, (s, ev) => Close())
            {
                ShortcutKeyDisplayString = "Alt+F4"
            });

            // 编辑菜单
            var editMenu = new ToolStripMenuItem("编辑(&E)");
            editMenu.DropDownItems.Add(new ToolStripMenuItem("复制(&C)", null, (s, ev) =>
            {
                // TODO: 实现选择复制
            })
            {
                ShortcutKeyDisplayString = "Ctrl+Shift+C"
            });
            editMenu.DropDownItems.Add(new ToolStripMenuItem("粘贴(&V)", null, (s, ev) =>
            {
                if (_tabContainer.TabBar.SelectedTab?.Tag is ConPtyTerminal terminal)
                {
                    if (Clipboard.ContainsText())
                    {
                        terminal.SendInput(Clipboard.GetText());
                    }
                }
            })
            {
                ShortcutKeyDisplayString = "Ctrl+V"
            });
            editMenu.DropDownItems.Add(new ToolStripSeparator());
            editMenu.DropDownItems.Add(new ToolStripMenuItem("清屏(&L)", null, (s, ev) =>
            {
                if (_tabContainer.TabBar.SelectedTab?.Tag is ConPtyTerminal terminal)
                {
                    terminal.Clear();
                    terminal.SendInput("cls\r");
                }
            }));

            // 视图菜单
            var viewMenu = new ToolStripMenuItem("视图(&V)");
            viewMenu.DropDownItems.Add(new ToolStripMenuItem("下一个标签页", null, (s, ev) =>
            {
                int count = _tabContainer.TabBar.Tabs.Count;
                if (count > 1)
                {
                    int next = (_tabContainer.TabBar.SelectedIndex + 1) % count;
                    _tabContainer.SelectTab(next);
                }
            })
            {
                ShortcutKeyDisplayString = "Ctrl+Tab"
            });
            viewMenu.DropDownItems.Add(new ToolStripMenuItem("上一个标签页", null, (s, ev) =>
            {
                int count = _tabContainer.TabBar.Tabs.Count;
                if (count > 1)
                {
                    int prev = (_tabContainer.TabBar.SelectedIndex - 1 + count) % count;
                    _tabContainer.SelectTab(prev);
                }
            })
            {
                ShortcutKeyDisplayString = "Ctrl+Shift+Tab"
            });

            // 帮助菜单
            var helpMenu = new ToolStripMenuItem("帮助(&H)");
            helpMenu.DropDownItems.Add(new ToolStripMenuItem("快捷键", null, (s, ev) =>
            {
                MessageBox.Show(
                    "快捷键列表:\n\n" +
                    "Ctrl+Shift+T  新建终端标签页\n" +
                    "Ctrl+W        关闭当前标签页\n" +
                    "Ctrl+Tab      切换到下一个标签页\n" +
                    "Ctrl+Shift+Tab 切换到上一个标签页\n" +
                    "Ctrl+1-9      切换到指定标签页\n" +
                    "Ctrl+C        中断当前命令\n" +
                    "Ctrl+V        粘贴\n",
                    "快捷键",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
            }));
            helpMenu.DropDownItems.Add(new ToolStripSeparator());
            helpMenu.DropDownItems.Add(new ToolStripMenuItem("关于", null, (s, ev) =>
            {
                MessageBox.Show(
                    "Multi-Tab Terminal\n\n" +
                    "基于 Windows ConPTY API 的多标签页终端\n" +
                    "使用 ModernTabControl 组件\n\n" +
                    "ClaudeAssist Project",
                    "关于",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
            }));

            menuStrip.Items.Add(fileMenu);
            menuStrip.Items.Add(editMenu);
            menuStrip.Items.Add(viewMenu);
            menuStrip.Items.Add(helpMenu);

            Controls.Add(menuStrip);
            menuStrip.BringToFront();

            // 调整标签容器位置
            _tabContainer.Padding = new Padding(0, menuStrip.Height, 0, 0);
        }

        #endregion

        #region 资源释放

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);

            // 关闭所有终端
            foreach (var tab in _tabContainer.TabBar.Tabs.ToList())
            {
                if (tab.Tag is ConPtyTerminal terminal)
                {
                    terminal.Stop();
                    terminal.Dispose();
                }
            }
        }

        #endregion
    }
}
