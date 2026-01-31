using System.Drawing.Drawing2D;

namespace ClaudeAssist
{
    public class ModernForm : Form
    {
        private const int TITLE_HEIGHT = 32;
        private const int BORDER_RADIUS = 8;
        private const int RESIZE_BORDER = 6;
        private const int BUTTON_WIDTH = 46;

        private bool _isMaximized = false;
        private Point _dragStart;
        private bool _isDragging = false;
        private Rectangle _restoreBounds;

        // 按钮悬停状态
        private int _hoverButton = -1; // -1=无, 0=最小化, 1=最大化, 2=关闭

        // 颜色主题 (DevExpress V25.2 风格)
        private Color _titleBarColor = Color.FromArgb(30, 30, 30);
        private Color _titleBarActiveColor = Color.FromArgb(45, 45, 48);
        private Color _backgroundColor = Color.FromArgb(37, 37, 38);
        private Color _borderColor = Color.FromArgb(60, 60, 60);
        private Color _textColor = Color.FromArgb(241, 241, 241);
        private Color _buttonHoverColor = Color.FromArgb(62, 62, 64);
        private Color _closeButtonHoverColor = Color.FromArgb(232, 17, 35);

        public ModernForm()
        {
            InitializeModernStyle();
        }

        private void InitializeModernStyle()
        {
            FormBorderStyle = FormBorderStyle.None;
            BackColor = _backgroundColor;
            DoubleBuffered = true;
            SetStyle(ControlStyles.ResizeRedraw, true);
            Padding = new Padding(0, TITLE_HEIGHT, 0, 0);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;

            DrawTitleBar(g);
            DrawBorder(g);
        }

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

            // 绘制窗口按钮
            DrawWindowButtons(g);
        }

        private void DrawWindowButtons(Graphics g)
        {
            int buttonY = 0;
            int buttonHeight = TITLE_HEIGHT;

            // 关闭按钮
            int closeX = Width - BUTTON_WIDTH;
            if (_hoverButton == 2)
            {
                using var brush = new SolidBrush(_closeButtonHoverColor);
                g.FillRectangle(brush, closeX, buttonY, BUTTON_WIDTH, buttonHeight);
            }
            DrawCloseIcon(g, closeX, buttonY, BUTTON_WIDTH, buttonHeight);

            // 最大化按钮
            int maxX = Width - BUTTON_WIDTH * 2;
            if (_hoverButton == 1)
            {
                using var brush = new SolidBrush(_buttonHoverColor);
                g.FillRectangle(brush, maxX, buttonY, BUTTON_WIDTH, buttonHeight);
            }
            DrawMaximizeIcon(g, maxX, buttonY, BUTTON_WIDTH, buttonHeight);

            // 最小化按钮
            int minX = Width - BUTTON_WIDTH * 3;
            if (_hoverButton == 0)
            {
                using var brush = new SolidBrush(_buttonHoverColor);
                g.FillRectangle(brush, minX, buttonY, BUTTON_WIDTH, buttonHeight);
            }
            DrawMinimizeIcon(g, minX, buttonY, BUTTON_WIDTH, buttonHeight);
        }

        private void DrawCloseIcon(Graphics g, int x, int y, int w, int h)
        {
            using var pen = new Pen(_textColor, 1f);
            int cx = x + w / 2;
            int cy = y + h / 2;
            int size = 5;
            g.DrawLine(pen, cx - size, cy - size, cx + size, cy + size);
            g.DrawLine(pen, cx + size, cy - size, cx - size, cy + size);
        }

        private void DrawMaximizeIcon(Graphics g, int x, int y, int w, int h)
        {
            using var pen = new Pen(_textColor, 1f);
            int cx = x + w / 2;
            int cy = y + h / 2;
            int size = 5;
            if (_isMaximized)
            {
                // 还原图标
                g.DrawRectangle(pen, cx - size + 2, cy - size, size * 2 - 2, size * 2 - 2);
                g.DrawRectangle(pen, cx - size, cy - size + 2, size * 2 - 2, size * 2 - 2);
            }
            else
            {
                g.DrawRectangle(pen, cx - size, cy - size, size * 2, size * 2);
            }
        }

        private void DrawMinimizeIcon(Graphics g, int x, int y, int w, int h)
        {
            using var pen = new Pen(_textColor, 1f);
            int cx = x + w / 2;
            int cy = y + h / 2;
            g.DrawLine(pen, cx - 5, cy, cx + 5, cy);
        }

        private void DrawBorder(Graphics g)
        {
            using var pen = new Pen(_borderColor, 1f);
            g.DrawRectangle(pen, 0, 0, Width - 1, Height - 1);
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            if (_isDragging)
            {
                Point currentScreen = PointToScreen(e.Location);
                Location = new Point(
                    currentScreen.X - _dragStart.X,
                    currentScreen.Y - _dragStart.Y);
                return;
            }

            // 检测按钮悬停
            int oldHover = _hoverButton;
            _hoverButton = GetButtonAtPoint(e.Location);
            if (oldHover != _hoverButton)
                Invalidate(new Rectangle(Width - BUTTON_WIDTH * 3, 0, BUTTON_WIDTH * 3, TITLE_HEIGHT));

            // 设置调整大小光标
            Cursor = GetResizeCursor(e.Location);
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);

            if (e.Button == MouseButtons.Left)
            {
                int button = GetButtonAtPoint(e.Location);
                if (button >= 0)
                    return;

                if (e.Y < TITLE_HEIGHT)
                {
                    _isDragging = true;
                    _dragStart = e.Location;
                }
                else
                {
                    // 调整大小由 WndProc 处理
                }
            }
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            _isDragging = false;
        }

        protected override void OnMouseClick(MouseEventArgs e)
        {
            base.OnMouseClick(e);

            int button = GetButtonAtPoint(e.Location);
            switch (button)
            {
                case 0: // 最小化
                    WindowState = FormWindowState.Minimized;
                    break;
                case 1: // 最大化/还原
                    ToggleMaximize();
                    break;
                case 2: // 关闭
                    Close();
                    break;
            }
        }

        protected override void OnMouseDoubleClick(MouseEventArgs e)
        {
            base.OnMouseDoubleClick(e);
            if (e.Y < TITLE_HEIGHT && GetButtonAtPoint(e.Location) < 0)
            {
                ToggleMaximize();
            }
        }

        protected override void OnMouseLeave(EventArgs e)
        {
            base.OnMouseLeave(e);
            if (_hoverButton >= 0)
            {
                _hoverButton = -1;
                Invalidate(new Rectangle(Width - BUTTON_WIDTH * 3, 0, BUTTON_WIDTH * 3, TITLE_HEIGHT));
            }
        }

        private int GetButtonAtPoint(Point p)
        {
            if (p.Y > TITLE_HEIGHT) return -1;

            if (p.X >= Width - BUTTON_WIDTH) return 2; // 关闭
            if (p.X >= Width - BUTTON_WIDTH * 2) return 1; // 最大化
            if (p.X >= Width - BUTTON_WIDTH * 3) return 0; // 最小化

            return -1;
        }

        private void ToggleMaximize()
        {
            if (_isMaximized)
            {
                _isMaximized = false;
                Bounds = _restoreBounds;
            }
            else
            {
                _restoreBounds = Bounds;
                _isMaximized = true;
                var screen = Screen.FromControl(this).WorkingArea;
                Bounds = screen;
            }
            Invalidate();
        }

        private Cursor GetResizeCursor(Point p)
        {
            if (_isMaximized) return Cursors.Default;

            bool left = p.X < RESIZE_BORDER;
            bool right = p.X > Width - RESIZE_BORDER;
            bool top = p.Y < RESIZE_BORDER;
            bool bottom = p.Y > Height - RESIZE_BORDER;

            if ((left && top) || (right && bottom)) return Cursors.SizeNWSE;
            if ((right && top) || (left && bottom)) return Cursors.SizeNESW;
            if (left || right) return Cursors.SizeWE;
            if (top || bottom) return Cursors.SizeNS;

            return Cursors.Default;
        }

        ///
        protected override void WndProc(ref Message m)
        {
            base.WndProc(ref m);

            /*
            WM_NCHITTEST 是以下内容的缩写：
            WM - Windows Message NC - Non-Client（非客户区） HIT - Hit（命中/点击检测） TEST - Test（测试）
            完整含义： Windows Message Non-Client Hit Test
            解释：
            Windows Message: Windows系统的消息机制
            Non-Client: 指窗口的非客户区域（标题栏、边框、菜单栏等），区别于客户区域（Client Area，即程序内容显示区域）
            Hit Test: 命中测试，检测鼠标点击或悬停的位置
            为什么叫"非客户区"？ 在Windows窗口架构中：
            客户区(Client Area): 程序可以自由绘制内容的区域
            非客户区(Non-Client Area): 系统负责管理的区域（标题栏、边框、滚动条等）
            所以 WM_NCHITTEST 就是"Windows消息：非客户区命中测试"，用来确定鼠标在窗口非客户区的具体位置。

            屏幕坐标 (Screen Coordinates):
            以整个屏幕左上角为原点 (0,0)
            WM_NCHITTEST 消息中的坐标就是屏幕坐标
            不受窗口位置影响

            客户端坐标 (Client Coordinates):
            以窗口客户区左上角为原点 (0,0)
            受窗口位置影响
            用于窗口内部绘制和交互
             */
            if (m.Msg == 0x84) // WM_NCHITTEST
            {
                /*
                LParam 的结构：
                LParam 是一个32位整数
                低16位 (0-15位)：存储 X 坐标
                高16位 (16-31位)：存储 Y 坐标
                 */
                Point p = new Point(m.LParam.ToInt32() & 0xFFFF, m.LParam.ToInt32() >> 16);
                p = PointToClient(p);//将屏幕坐标转换为客户端坐标

                if (_isMaximized) return;
                
                bool left = p.X < RESIZE_BORDER;
                bool right = p.X > Width - RESIZE_BORDER;
                bool top = p.Y < RESIZE_BORDER;
                bool bottom = p.Y > Height - RESIZE_BORDER;

                //HTLEFT(10) 到 HTBOTTOMRIGHT(17) 实现八方向调整大小
                if (left && top) m.Result = (IntPtr)13; // HTTOPLEFT 左上角	对角调整 ↖↘
                else if (right && top) m.Result = (IntPtr)14; // HTTOPRIGHT
                else if (left && bottom) m.Result = (IntPtr)16; // HTBOTTOMLEFT
                else if (right && bottom) m.Result = (IntPtr)17; // HTBOTTOMRIGHT
                else if (left) m.Result = (IntPtr)10; // HTLEFT
                else if (right) m.Result = (IntPtr)11; // HTRIGHT
                else if (top) m.Result = (IntPtr)12; // HTTOP
                else if (bottom) m.Result = (IntPtr)15; // HTBOTTOM
                else if (p.Y < TITLE_HEIGHT) m.Result = (IntPtr)2; // HTCAPTION 拖动移动窗口


                /*
                 WM_NCHITTEST 返回值完整表格
                返回值	常量名	含义	Windows 自动行为
                0	HTERROR	错误	播放错误提示音
                1	HTCLIENT	客户区	正常鼠标事件
                2	HTCAPTION	标题栏	拖动移动窗口
                3	HTSYSMENU	系统菜单	显示系统菜单
                4	HTGROWBOX	大小调整框	同 HTSIZE
                5	HTMENU	菜单栏	激活菜单
                6	HTHSCROLL	水平滚动条	滚动操作
                7	HTVSCROLL	垂直滚动条	滚动操作
                8	HTMINBUTTON	最小化按钮	最小化窗口
                9	HTMAXBUTTON	最大化按钮	最大化/还原窗口
                10	HTLEFT	左边框	向左调整宽度 ↔
                11	HTRIGHT	右边框	向右调整宽度 ↔
                12	HTTOP	上边框	向上调整高度 ↕
                13	HTTOPLEFT	左上角	对角调整 ↖↘
                14	HTTOPRIGHT	右上角	对角调整 ↗↙
                15	HTBOTTOM	下边框	向下调整高度 ↕
                16	HTBOTTOMLEFT	左下角	对角调整 ↙↗
                17	HTBOTTOMRIGHT	右下角	对角调整 ↘↖
                18	HTBORDER	不可调整边框	无操作
                19	HTOBJECT	对象	-
                20	HTCLOSE	关闭按钮	关闭窗口
                21	HTHELP	帮助按钮	进入帮助模式
                -1	HTNOWHERE	不在窗口上	忽略
                -2	HTTRANSPARENT	透明区域	穿透到下层窗口
                 */
            }
        }

        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern bool ReleaseCapture();

        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, IntPtr lParam);
    }
}
