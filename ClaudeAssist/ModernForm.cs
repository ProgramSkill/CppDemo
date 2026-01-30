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
                    // 处理窗口调整大小
                    HandleResize(e.Location);
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

        private void HandleResize(Point p)
        {
            if (_isMaximized) return;

            const int HTLEFT = 10, HTRIGHT = 11, HTTOP = 12, HTBOTTOM = 15;
            const int HTTOPLEFT = 13, HTTOPRIGHT = 14, HTBOTTOMLEFT = 16, HTBOTTOMRIGHT = 17;

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

        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern bool ReleaseCapture();

        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern IntPtr SendMessage(IntPtr hWnd, int Msg, IntPtr wParam, IntPtr lParam);
    }
}
