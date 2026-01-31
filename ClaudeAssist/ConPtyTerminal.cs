using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace ClaudeAssist
{
    /// <summary>
    /// ConPTY 终端控件 - 使用 Windows Pseudo Console API
    /// </summary>
    public class ConPtyTerminal : Control
    {
        #region Native API

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern int CreatePseudoConsole(
            COORD size,
            SafeFileHandle hInput,
            SafeFileHandle hOutput,
            uint dwFlags,
            out IntPtr phPC);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern int ResizePseudoConsole(IntPtr hPC, COORD size);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern void ClosePseudoConsole(IntPtr hPC);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CreatePipe(
            out SafeFileHandle hReadPipe,
            out SafeFileHandle hWritePipe,
            IntPtr lpPipeAttributes,
            uint nSize);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool InitializeProcThreadAttributeList(
            IntPtr lpAttributeList,
            int dwAttributeCount,
            int dwFlags,
            ref IntPtr lpSize);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool UpdateProcThreadAttribute(
            IntPtr lpAttributeList,
            uint dwFlags,
            IntPtr Attribute,
            IntPtr lpValue,
            IntPtr cbSize,
            IntPtr lpPreviousValue,
            IntPtr lpReturnSize);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern void DeleteProcThreadAttributeList(IntPtr lpAttributeList);

        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern bool CreateProcessW(
            string? lpApplicationName,
            string lpCommandLine,
            IntPtr lpProcessAttributes,
            IntPtr lpThreadAttributes,
            bool bInheritHandles,
            uint dwCreationFlags,
            IntPtr lpEnvironment,
            string? lpCurrentDirectory,
            ref STARTUPINFOEX lpStartupInfo,
            out PROCESS_INFORMATION lpProcessInformation);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr hObject);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool TerminateProcess(IntPtr hProcess, uint uExitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern uint WaitForSingleObject(IntPtr hHandle, uint dwMilliseconds);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetExitCodeProcess(IntPtr hProcess, out uint lpExitCode);

        private const uint EXTENDED_STARTUPINFO_PRESENT = 0x00080000;
        private const int PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016;
        private const uint STILL_ACTIVE = 259;

        [StructLayout(LayoutKind.Sequential)]
        private struct COORD
        {
            public short X;
            public short Y;

            public COORD(short x, short y)
            {
                X = x;
                Y = y;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct STARTUPINFO
        {
            public int cb;
            public IntPtr lpReserved;
            public IntPtr lpDesktop;
            public IntPtr lpTitle;
            public int dwX;
            public int dwY;
            public int dwXSize;
            public int dwYSize;
            public int dwXCountChars;
            public int dwYCountChars;
            public int dwFillAttribute;
            public int dwFlags;
            public short wShowWindow;
            public short cbReserved2;
            public IntPtr lpReserved2;
            public IntPtr hStdInput;
            public IntPtr hStdOutput;
            public IntPtr hStdError;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct STARTUPINFOEX
        {
            public STARTUPINFO StartupInfo;
            public IntPtr lpAttributeList;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct PROCESS_INFORMATION
        {
            public IntPtr hProcess;
            public IntPtr hThread;
            public int dwProcessId;
            public int dwThreadId;
        }

        #endregion

        #region 字段

        private IntPtr _hPC = IntPtr.Zero;
        private SafeFileHandle? _pipeInputRead;
        private SafeFileHandle? _pipeInputWrite;
        private SafeFileHandle? _pipeOutputRead;
        private SafeFileHandle? _pipeOutputWrite;
        private FileStream? _inputStream; // 持久的输入流
        private IntPtr _hProcess = IntPtr.Zero;
        private IntPtr _hThread = IntPtr.Zero;
        private int _processId;
        private IntPtr _attributeList = IntPtr.Zero;

        private Thread? _readThread;
        private volatile bool _isRunning;
        private StringBuilder _buffer = new();
        private List<string> _lines = new();
        private int _scrollOffset = 0;
        private int _cursorX = 0;
        private int _cursorY = 0;
        private bool _cursorVisible = true;
        private System.Windows.Forms.Timer? _cursorTimer;

        // 终端尺寸（字符数）
        private int _columns = 120;
        private int _rows = 30;

        // 字体设置
        private Font _terminalFont;
        private int _charWidth;
        private int _charHeight;

        // 颜色
        private Color _backgroundColor = Color.FromArgb(30, 30, 30);
        private Color _foregroundColor = Color.FromArgb(204, 204, 204);

        // ANSI 颜色表
        private readonly Color[] _ansiColors = new Color[]
        {
            Color.FromArgb(0, 0, 0),       // 0 - Black
            Color.FromArgb(205, 49, 49),   // 1 - Red
            Color.FromArgb(13, 188, 121),  // 2 - Green
            Color.FromArgb(229, 229, 16),  // 3 - Yellow
            Color.FromArgb(36, 114, 200),  // 4 - Blue
            Color.FromArgb(188, 63, 188),  // 5 - Magenta
            Color.FromArgb(17, 168, 205),  // 6 - Cyan
            Color.FromArgb(229, 229, 229), // 7 - White
            Color.FromArgb(102, 102, 102), // 8 - Bright Black
            Color.FromArgb(241, 76, 76),   // 9 - Bright Red
            Color.FromArgb(35, 209, 139),  // 10 - Bright Green
            Color.FromArgb(245, 245, 67),  // 11 - Bright Yellow
            Color.FromArgb(59, 142, 234),  // 12 - Bright Blue
            Color.FromArgb(214, 112, 214), // 13 - Bright Magenta
            Color.FromArgb(41, 184, 219),  // 14 - Bright Cyan
            Color.FromArgb(255, 255, 255)  // 15 - Bright White
        };

        private Color _currentForeground;
        private Color _currentBackground;

        // 文本选择
        private bool _isSelecting = false;
        private Point _selectionStart = Point.Empty;
        private Point _selectionEnd = Point.Empty;
        private bool _hasSelection = false;
        private Color _selectionColor = Color.FromArgb(100, 0, 122, 204);

        #endregion

        #region 属性

        [Category("Terminal")]
        [DefaultValue("cmd.exe")]
        public string ShellPath { get; set; } = "cmd.exe";

        [Category("Terminal")]
        [DesignerSerializationVisibility(DesignerSerializationVisibility.Hidden)]
        public string? WorkingDirectory { get; set; }

        [Category("Terminal")]
        [DefaultValue(120)]
        public int Columns
        {
            get => _columns;
            set
            {
                _columns = Math.Max(20, value);
                ResizeTerminal();
            }
        }

        [Category("Terminal")]
        [DefaultValue(30)]
        public int Rows
        {
            get => _rows;
            set
            {
                _rows = Math.Max(5, value);
                ResizeTerminal();
            }
        }

        [Browsable(false)]
        public bool IsRunning => _isRunning;

        [Browsable(false)]
        public int ProcessId => _processId;

        #endregion

        #region 事件

        public event EventHandler? ProcessExited;
        public event EventHandler<string>? DataReceived;

        #endregion

        #region 构造函数

        public ConPtyTerminal()
        {
            SetStyle(ControlStyles.UserPaint |
                    ControlStyles.AllPaintingInWmPaint |
                    ControlStyles.OptimizedDoubleBuffer |
                    ControlStyles.ResizeRedraw |
                    ControlStyles.Selectable, true);

            _terminalFont = new Font("Cascadia Mono", 11f, FontStyle.Regular);
            _currentForeground = _foregroundColor;
            _currentBackground = _backgroundColor;

            BackColor = _backgroundColor;
            ForeColor = _foregroundColor;
            TabStop = true; // 允许Tab键聚焦

            CalculateCharSize();
            InitializeCursorTimer();
        }

        protected override void OnGotFocus(EventArgs e)
        {
            base.OnGotFocus(e);
            _cursorVisible = true;
            Invalidate();
        }

        protected override void OnLostFocus(EventArgs e)
        {
            base.OnLostFocus(e);
            _cursorVisible = false;
            Invalidate();
        }

        #endregion

        #region 公共方法

        public void Start()
        {
            if (_isRunning) return;

            try
            {
                CreatePipes();
                CreatePseudoConsoleAndProcess();
                StartReadThread();
                _isRunning = true;
            }
            catch (Exception ex)
            {
                AppendText($"\r\n[错误] 启动终端失败: {ex.Message}\r\n");
                Cleanup();
            }
        }

        public void Stop()
        {
            if (!_isRunning) return;

            _isRunning = false;

            if (_hProcess != IntPtr.Zero)
            {
                TerminateProcess(_hProcess, 0);
            }

            Cleanup();
        }

        public void SendInput(string text)
        {
            if (!_isRunning || _inputStream == null) return;

            try
            {
                var bytes = Encoding.UTF8.GetBytes(text);
                _inputStream.Write(bytes, 0, bytes.Length);
                _inputStream.Flush();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"SendInput error: {ex.Message}");
            }
        }

        public void SendKey(Keys key)
        {
            string? sequence = key switch
            {
                Keys.Enter => "\r",
                Keys.Back => "\b",
                Keys.Tab => "\t",
                Keys.Escape => "\x1b",
                Keys.Up => "\x1b[A",
                Keys.Down => "\x1b[B",
                Keys.Right => "\x1b[C",
                Keys.Left => "\x1b[D",
                Keys.Home => "\x1b[H",
                Keys.End => "\x1b[F",
                Keys.Insert => "\x1b[2~",
                Keys.Delete => "\x1b[3~",
                Keys.PageUp => "\x1b[5~",
                Keys.PageDown => "\x1b[6~",
                Keys.F1 => "\x1bOP",
                Keys.F2 => "\x1bOQ",
                Keys.F3 => "\x1bOR",
                Keys.F4 => "\x1bOS",
                Keys.F5 => "\x1b[15~",
                Keys.F6 => "\x1b[17~",
                Keys.F7 => "\x1b[18~",
                Keys.F8 => "\x1b[19~",
                Keys.F9 => "\x1b[20~",
                Keys.F10 => "\x1b[21~",
                Keys.F11 => "\x1b[23~",
                Keys.F12 => "\x1b[24~",
                _ => null
            };

            if (sequence != null)
            {
                SendInput(sequence);
            }
        }

        public void Clear()
        {
            lock (_buffer)
            {
                _buffer.Clear();
                _lines.Clear();
                _cursorX = 0;
                _cursorY = 0;
                _scrollOffset = 0;
            }
            ClearSelection();
            Invalidate();
        }

        public string GetSelectedText()
        {
            if (!_hasSelection) return string.Empty;

            lock (_buffer)
            {
                var start = _selectionStart;
                var end = _selectionEnd;

                // 确保 start 在 end 之前
                if (start.Y > end.Y || (start.Y == end.Y && start.X > end.X))
                {
                    (start, end) = (end, start);
                }

                var sb = new StringBuilder();

                for (int row = start.Y; row <= end.Y && row < _lines.Count; row++)
                {
                    if (row < 0) continue;

                    string line = row < _lines.Count ? _lines[row] : string.Empty;

                    int startCol = (row == start.Y) ? start.X : 0;
                    int endCol = (row == end.Y) ? end.X : line.Length;

                    startCol = Math.Max(0, Math.Min(startCol, line.Length));
                    endCol = Math.Max(0, Math.Min(endCol, line.Length));

                    if (startCol < endCol)
                    {
                        sb.Append(line.Substring(startCol, endCol - startCol));
                    }

                    if (row < end.Y)
                    {
                        sb.AppendLine();
                    }
                }

                return sb.ToString();
            }
        }

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

        public void ClearSelection()
        {
            _hasSelection = false;
            _isSelecting = false;
            _selectionStart = Point.Empty;
            _selectionEnd = Point.Empty;
            Invalidate();
        }

        #endregion

        #region ConPTY 实现

        private void CreatePipes()
        {
            if (!CreatePipe(out _pipeInputRead!, out _pipeInputWrite!, IntPtr.Zero, 0))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "CreatePipe for input failed");

            if (!CreatePipe(out _pipeOutputRead!, out _pipeOutputWrite!, IntPtr.Zero, 0))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "CreatePipe for output failed");
        }

        private void CreatePseudoConsoleAndProcess()
        {
            var size = new COORD((short)_columns, (short)_rows);

            int hr = CreatePseudoConsole(size, _pipeInputRead!, _pipeOutputWrite!, 0, out _hPC);
            if (hr != 0)
                throw new Win32Exception(hr, "CreatePseudoConsole failed");

            // 初始化进程属性列表
            IntPtr listSize = IntPtr.Zero;
            InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref listSize);

            _attributeList = Marshal.AllocHGlobal(listSize);
            if (!InitializeProcThreadAttributeList(_attributeList, 1, 0, ref listSize))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "InitializeProcThreadAttributeList failed");

            // 设置伪控制台属性
            if (!UpdateProcThreadAttribute(
                _attributeList,
                0,
                (IntPtr)PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                _hPC,
                (IntPtr)IntPtr.Size,
                IntPtr.Zero,
                IntPtr.Zero))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "UpdateProcThreadAttribute failed");

            // 创建进程
            var startupInfo = new STARTUPINFOEX
            {
                StartupInfo = new STARTUPINFO { cb = Marshal.SizeOf<STARTUPINFOEX>() },
                lpAttributeList = _attributeList
            };

            string workDir = WorkingDirectory ?? Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

            if (!CreateProcessW(
                null,
                ShellPath,
                IntPtr.Zero,
                IntPtr.Zero,
                false,
                EXTENDED_STARTUPINFO_PRESENT,
                IntPtr.Zero,
                workDir,
                ref startupInfo,
                out var processInfo))
                throw new Win32Exception(Marshal.GetLastWin32Error(), "CreateProcess failed");

            _hProcess = processInfo.hProcess;
            _hThread = processInfo.hThread;
            _processId = processInfo.dwProcessId;

            // 关闭不需要的句柄
            _pipeInputRead?.Close();
            _pipeInputRead = null;
            _pipeOutputWrite?.Close();
            _pipeOutputWrite = null;

            // 创建持久的输入流
            _inputStream = new FileStream(_pipeInputWrite!, FileAccess.Write, 4096, false);
        }

        private void StartReadThread()
        {
            _readThread = new Thread(ReadThreadProc)
            {
                IsBackground = true,
                Name = "ConPTY Read Thread"
            };
            _readThread.Start();
        }

        private void ReadThreadProc()
        {
            var buffer = new byte[4096];

            try
            {
                using var stream = new FileStream(_pipeOutputRead!, FileAccess.Read, 4096, false);

                while (_isRunning)
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead == 0)
                    {
                        // 进程已退出
                        break;
                    }

                    string text = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                    ProcessOutput(text);
                }
            }
            catch (Exception ex)
            {
                if (_isRunning)
                {
                    Debug.WriteLine($"Read thread error: {ex.Message}");
                }
            }

            // 检查进程是否已退出
            if (_hProcess != IntPtr.Zero)
            {
                if (GetExitCodeProcess(_hProcess, out uint exitCode) && exitCode != STILL_ACTIVE)
                {
                    BeginInvoke(() =>
                    {
                        AppendText($"\r\n[进程已退出，退出码: {exitCode}]\r\n");
                        _isRunning = false;
                        ProcessExited?.Invoke(this, EventArgs.Empty);
                    });
                }
            }
        }

        private void ProcessOutput(string text)
        {
            DataReceived?.Invoke(this, text);

            lock (_buffer)
            {
                // ANSI 转义序列处理
                int i = 0;
                while (i < text.Length)
                {
                    if (text[i] == '\x1b' && i + 1 < text.Length)
                    {
                        char nextChar = text[i + 1];

                        if (nextChar == '[')
                        {
                            // CSI 序列 (ESC [)
                            int start = i + 2;
                            int end = start;
                            while (end < text.Length && !char.IsLetter(text[end]) && text[end] != '@')
                            {
                                end++;
                            }

                            if (end < text.Length)
                            {
                                string seq = text.Substring(start, end - start);
                                char cmd = text[end];
                                ProcessCSI(seq, cmd);
                                i = end + 1;
                                continue;
                            }
                        }
                        else if (nextChar == ']')
                        {
                            // OSC 序列 (ESC ]) - 用于设置窗口标题等
                            // 格式: ESC ] Ps ; Pt BEL 或 ESC ] Ps ; Pt ESC \
                            int start = i + 2;
                            int end = start;

                            // 查找序列结束符 (BEL=0x07 或 ESC \)
                            while (end < text.Length)
                            {
                                if (text[end] == '\x07') // BEL
                                {
                                    break;
                                }
                                if (text[end] == '\x1b' && end + 1 < text.Length && text[end + 1] == '\\')
                                {
                                    end++; // 跳过 ESC \
                                    break;
                                }
                                end++;
                            }

                            // 跳过整个 OSC 序列（不显示）
                            i = end + 1;
                            continue;
                        }
                        else if (nextChar == '(' || nextChar == ')' || nextChar == '*' || nextChar == '+')
                        {
                            // 字符集选择序列，跳过
                            i += 3;
                            continue;
                        }
                        else if (nextChar == '=' || nextChar == '>')
                        {
                            // 键盘模式序列，跳过
                            i += 2;
                            continue;
                        }
                    }
                    
                    if (text[i] == '\r')
                    {
                        _cursorX = 0;
                    }
                    else if (text[i] == '\n')
                    {
                        _cursorY++;
                        EnsureLine(_cursorY);
                    }
                    else if (text[i] == '\b')
                    {
                        if (_cursorX > 0) _cursorX--;
                    }
                    else if (text[i] == '\t')
                    {
                        _cursorX = ((_cursorX / 8) + 1) * 8;
                    }
                    else if (text[i] >= 32)
                    {
                        EnsureLine(_cursorY);
                        SetChar(_cursorY, _cursorX, text[i]);
                        _cursorX++;
                        if (_cursorX >= _columns)
                        {
                            _cursorX = 0;
                            _cursorY++;
                            EnsureLine(_cursorY);
                        }
                    }

                    i++;
                }
            }

            BeginInvoke(Invalidate);
        }

        private void ProcessCSI(string seq, char cmd)
        {
            var parts = seq.Split(';');
            int[] nums = parts.Select(p => int.TryParse(p, out int n) ? n : 0).ToArray();

            switch (cmd)
            {
                case 'H': // 光标位置
                case 'f':
                    _cursorY = (nums.Length > 0 && nums[0] > 0 ? nums[0] : 1) - 1;
                    _cursorX = (nums.Length > 1 && nums[1] > 0 ? nums[1] : 1) - 1;
                    break;

                case 'A': // 光标上移
                    _cursorY = Math.Max(0, _cursorY - (nums.Length > 0 && nums[0] > 0 ? nums[0] : 1));
                    break;

                case 'B': // 光标下移
                    _cursorY += nums.Length > 0 && nums[0] > 0 ? nums[0] : 1;
                    break;

                case 'C': // 光标右移
                    _cursorX += nums.Length > 0 && nums[0] > 0 ? nums[0] : 1;
                    break;

                case 'D': // 光标左移
                    _cursorX = Math.Max(0, _cursorX - (nums.Length > 0 && nums[0] > 0 ? nums[0] : 1));
                    break;

                case 'J': // 清屏
                    int mode = nums.Length > 0 ? nums[0] : 0;
                    if (mode == 2 || mode == 3)
                    {
                        _lines.Clear();
                        _cursorX = 0;
                        _cursorY = 0;
                    }
                    break;

                case 'K': // 清行
                    EnsureLine(_cursorY);
                    if (_cursorY < _lines.Count && _cursorX < _lines[_cursorY].Length)
                    {
                        _lines[_cursorY] = _lines[_cursorY].Substring(0, _cursorX);
                    }
                    break;

                case 'm': // SGR - 设置图形渲染
                    foreach (int n in nums)
                    {
                        ProcessSGR(n);
                    }
                    break;
            }
        }

        private void ProcessSGR(int code)
        {
            switch (code)
            {
                case 0: // 重置
                    _currentForeground = _foregroundColor;
                    _currentBackground = _backgroundColor;
                    break;

                case >= 30 and <= 37: // 前景色
                    _currentForeground = _ansiColors[code - 30];
                    break;

                case 39: // 默认前景色
                    _currentForeground = _foregroundColor;
                    break;

                case >= 40 and <= 47: // 背景色
                    _currentBackground = _ansiColors[code - 40];
                    break;

                case 49: // 默认背景色
                    _currentBackground = _backgroundColor;
                    break;

                case >= 90 and <= 97: // 亮前景色
                    _currentForeground = _ansiColors[code - 90 + 8];
                    break;

                case >= 100 and <= 107: // 亮背景色
                    _currentBackground = _ansiColors[code - 100 + 8];
                    break;
            }
        }

        private void EnsureLine(int lineIndex)
        {
            while (_lines.Count <= lineIndex)
            {
                _lines.Add(string.Empty);
            }
        }

        private void SetChar(int line, int col, char c)
        {
            EnsureLine(line);
            string currentLine = _lines[line];

            if (col >= currentLine.Length)
            {
                currentLine = currentLine.PadRight(col + 1);
            }

            var chars = currentLine.ToCharArray();
            if (col < chars.Length)
            {
                chars[col] = c;
            }
            _lines[line] = new string(chars);
        }

        private void ResizeTerminal()
        {
            if (_hPC != IntPtr.Zero)
            {
                var size = new COORD((short)_columns, (short)_rows);
                ResizePseudoConsole(_hPC, size);
            }
        }

        private void Cleanup()
        {
            _isRunning = false;

            // 关闭输入流
            try
            {
                _inputStream?.Close();
                _inputStream?.Dispose();
            }
            catch { }
            _inputStream = null;

            if (_attributeList != IntPtr.Zero)
            {
                DeleteProcThreadAttributeList(_attributeList);
                Marshal.FreeHGlobal(_attributeList);
                _attributeList = IntPtr.Zero;
            }

            if (_hPC != IntPtr.Zero)
            {
                ClosePseudoConsole(_hPC);
                _hPC = IntPtr.Zero;
            }

            if (_hThread != IntPtr.Zero)
            {
                CloseHandle(_hThread);
                _hThread = IntPtr.Zero;
            }

            if (_hProcess != IntPtr.Zero)
            {
                CloseHandle(_hProcess);
                _hProcess = IntPtr.Zero;
            }

            _pipeInputRead?.Dispose();
            _pipeInputWrite?.Dispose();
            _pipeOutputRead?.Dispose();
            _pipeOutputWrite?.Dispose();

            _pipeInputRead = null;
            _pipeInputWrite = null;
            _pipeOutputRead = null;
            _pipeOutputWrite = null;
        }

        #endregion

        #region 绘制

        private void CalculateCharSize()
        {
            // 使用 TextRenderer 计算字符尺寸，更精确
            _charWidth = TextRenderer.MeasureText("M", _terminalFont, Size.Empty, 
                TextFormatFlags.NoPadding | TextFormatFlags.NoPrefix).Width;
            _charHeight = TextRenderer.MeasureText("M", _terminalFont, Size.Empty,
                TextFormatFlags.NoPadding | TextFormatFlags.NoPrefix).Height;
        }

        private int GetCharWidth(char c)
        {
            // 中文、日文、韩文等双宽度字符
            if (IsWideChar(c))
            {
                return _charWidth * 2;
            }
            return _charWidth;
        }

        private bool IsWideChar(char c)
        {
            // CJK 字符范围（中文、日文、韩文）
            return (c >= 0x4E00 && c <= 0x9FFF) ||   // CJK 统一表意文字
                   (c >= 0x3400 && c <= 0x4DBF) ||   // CJK 扩展 A
                   (c >= 0x20000 && c <= 0x2A6DF) || // CJK 扩展 B
                   (c >= 0x2A700 && c <= 0x2B73F) || // CJK 扩展 C
                   (c >= 0x2B740 && c <= 0x2B81F) || // CJK 扩展 D
                   (c >= 0xF900 && c <= 0xFAFF) ||   // CJK 兼容表意文字
                   (c >= 0x3000 && c <= 0x303F) ||   // CJK 标点符号
                   (c >= 0xFF00 && c <= 0xFFEF) ||   // 全角字符
                   (c >= 0x3040 && c <= 0x309F) ||   // 平假名
                   (c >= 0x30A0 && c <= 0x30FF) ||   // 片假名
                   (c >= 0xAC00 && c <= 0xD7AF);     // 韩文音节
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            var g = e.Graphics;

            g.Clear(_backgroundColor);
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;

            lock (_buffer)
            {
                int visibleRows = Height / _charHeight;
                int startLine = Math.Max(0, _lines.Count - visibleRows - _scrollOffset);
                int endLine = Math.Min(_lines.Count, startLine + visibleRows + 1);

                // 获取规范化的选择范围
                var selStart = _selectionStart;
                var selEnd = _selectionEnd;
                if (_hasSelection && (selStart.Y > selEnd.Y || (selStart.Y == selEnd.Y && selStart.X > selEnd.X)))
                {
                    (selStart, selEnd) = (selEnd, selStart);
                }

                var textFlags = TextFormatFlags.NoPadding | TextFormatFlags.NoPrefix | TextFormatFlags.SingleLine;

                for (int i = startLine; i < endLine; i++)
                {
                    int screenY = (i - startLine) * _charHeight;
                    string line = i < _lines.Count ? _lines[i] : string.Empty;

                    // 逐字符绘制，正确处理双宽度字符
                    int x = 2;
                    for (int col = 0; col < line.Length; col++)
                    {
                        char c = line[col];
                        int charW = GetCharWidth(c);

                        // 绘制选择高亮
                        if (_hasSelection && i >= selStart.Y && i <= selEnd.Y)
                        {
                            int selStartCol = (i == selStart.Y) ? selStart.X : 0;
                            int selEndCol = (i == selEnd.Y) ? selEnd.X : line.Length;

                            if (col >= selStartCol && col < selEndCol)
                            {
                                using var selBrush = new SolidBrush(_selectionColor);
                                g.FillRectangle(selBrush, x, screenY, charW, _charHeight);
                            }
                        }

                        // 绘制字符
                        TextRenderer.DrawText(g, c.ToString(), _terminalFont,
                            new Rectangle(x, screenY, charW, _charHeight),
                            _foregroundColor, textFlags);

                        x += charW;
                    }
                }

                // 绘制光标
                if (_cursorVisible && _isRunning && !_hasSelection)
                {
                    int cursorScreenY = _cursorY - startLine;
                    if (cursorScreenY >= 0 && cursorScreenY < visibleRows)
                    {
                        // 计算光标X位置（考虑双宽度字符）
                        int cursorX = 2;
                        if (_cursorY < _lines.Count)
                        {
                            string cursorLine = _lines[_cursorY];
                            for (int col = 0; col < _cursorX && col < cursorLine.Length; col++)
                            {
                                cursorX += GetCharWidth(cursorLine[col]);
                            }
                            // 如果光标超出行末，按单宽度计算
                            if (_cursorX > cursorLine.Length)
                            {
                                cursorX += (_cursorX - cursorLine.Length) * _charWidth;
                            }
                        }
                        else
                        {
                            cursorX = 2 + _cursorX * _charWidth;
                        }

                        int y = cursorScreenY * _charHeight;
                        using var cursorBrush = new SolidBrush(Color.FromArgb(200, _foregroundColor));
                        g.FillRectangle(cursorBrush, cursorX, y, _charWidth, _charHeight);
                    }
                }
            }
        }

        private void InitializeCursorTimer()
        {
            _cursorTimer = new System.Windows.Forms.Timer
            {
                Interval = 500
            };
            _cursorTimer.Tick += (s, e) =>
            {
                _cursorVisible = !_cursorVisible;
                Invalidate();
            };
            _cursorTimer.Start();
        }

        #endregion

        #region 键盘处理

        protected override bool IsInputKey(Keys keyData)
        {
            switch (keyData)
            {
                case Keys.Up:
                case Keys.Down:
                case Keys.Left:
                case Keys.Right:
                case Keys.Tab:
                case Keys.Enter:
                    return true;
            }
            return base.IsInputKey(keyData);
        }

        protected override void OnKeyDown(KeyEventArgs e)
        {
            base.OnKeyDown(e);

            // 任何按键都清除选择状态（除了复制快捷键）
            if (!(e.Control && (e.KeyCode == Keys.C || e.KeyCode == Keys.A)))
            {
                if (_hasSelection)
                {
                    ClearSelection();
                }
            }

            if (!_isRunning) return;

            // Ctrl+Shift+C: 复制选中文本
            if (e.Control && e.Shift && e.KeyCode == Keys.C)
            {
                CopySelection();
                e.Handled = true;
                return;
            }

            // Ctrl+C: 发送中断信号（如果没有选中文本）
            if (e.Control && e.KeyCode == Keys.C && !e.Shift)
            {
                if (_hasSelection)
                {
                    CopySelection();
                }
                else
                {
                    SendInput("\x03");
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

            // 特殊键
            switch (e.KeyCode)
            {
                case Keys.Up:
                case Keys.Down:
                case Keys.Left:
                case Keys.Right:
                case Keys.Home:
                case Keys.End:
                case Keys.Insert:
                case Keys.Delete:
                case Keys.PageUp:
                case Keys.PageDown:
                case Keys.F1:
                case Keys.F2:
                case Keys.F3:
                case Keys.F4:
                case Keys.F5:
                case Keys.F6:
                case Keys.F7:
                case Keys.F8:
                case Keys.F9:
                case Keys.F10:
                case Keys.F11:
                case Keys.F12:
                case Keys.Escape:
                    SendKey(e.KeyCode);
                    e.Handled = true;
                    break;

                case Keys.Enter:
                    SendInput("\r");
                    e.Handled = true;
                    break;

                case Keys.Back:
                    SendInput("\x7f"); // DEL 字符，大多数终端使用这个作为退格
                    e.Handled = true;
                    break;

                case Keys.Tab:
                    SendInput("\t");
                    e.Handled = true;
                    break;
            }
        }

        protected override void OnKeyPress(KeyPressEventArgs e)
        {
            base.OnKeyPress(e);

            if (!_isRunning) return;

            // 普通字符输入（包括中文等Unicode字符）
            // 排除控制字符（0-31）但允许所有可打印字符
            if (e.KeyChar >= 32)
            {
                SendInput(e.KeyChar.ToString());
                e.Handled = true;
            }
        }

        #endregion

        #region 鼠标选择和滚轮

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            Focus();

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
                // 右键菜单
                ShowContextMenu(e.Location);
            }
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);

            if (_isSelecting && e.Button == MouseButtons.Left)
            {
                _selectionEnd = ScreenToCell(e.Location);
                _hasSelection = true;
                Invalidate();
            }
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);

            if (e.Button == MouseButtons.Left && _isSelecting)
            {
                _isSelecting = false;
                _selectionEnd = ScreenToCell(e.Location);

                // 检查是否有有效选择
                if (_selectionStart != _selectionEnd)
                {
                    _hasSelection = true;
                }
                Invalidate();
            }
        }

        protected override void OnMouseDoubleClick(MouseEventArgs e)
        {
            base.OnMouseDoubleClick(e);

            if (e.Button == MouseButtons.Left)
            {
                // 双击选择单词
                var cell = ScreenToCell(e.Location);
                SelectWord(cell);
            }
        }

        protected override void OnMouseWheel(MouseEventArgs e)
        {
            base.OnMouseWheel(e);

            int delta = e.Delta > 0 ? 3 : -3;
            lock (_buffer)
            {
                _scrollOffset = Math.Max(0, Math.Min(_lines.Count - 1, _scrollOffset + delta));
            }
            Invalidate();
        }

        private Point ScreenToCell(Point screenPoint)
        {
            int visibleRows = Height / _charHeight;
            int startLine = Math.Max(0, _lines.Count - visibleRows - _scrollOffset);

            int col = Math.Max(0, (screenPoint.X - 2) / _charWidth);
            int row = startLine + screenPoint.Y / _charHeight;

            return new Point(col, row);
        }

        private void SelectWord(Point cell)
        {
            lock (_buffer)
            {
                if (cell.Y < 0 || cell.Y >= _lines.Count) return;

                string line = _lines[cell.Y];
                if (cell.X < 0 || cell.X >= line.Length) return;

                // 找到单词边界
                int start = cell.X;
                int end = cell.X;

                while (start > 0 && IsWordChar(line[start - 1]))
                    start--;

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

        #endregion

        #region 辅助方法

        private void AppendText(string text)
        {
            lock (_buffer)
            {
                foreach (char c in text)
                {
                    if (c == '\r')
                    {
                        _cursorX = 0;
                    }
                    else if (c == '\n')
                    {
                        _cursorY++;
                        EnsureLine(_cursorY);
                    }
                    else if (c >= 32)
                    {
                        EnsureLine(_cursorY);
                        SetChar(_cursorY, _cursorX, c);
                        _cursorX++;
                    }
                }
            }
            Invalidate();
        }

        #endregion

        #region 资源释放

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                Stop();
                _cursorTimer?.Dispose();
                _terminalFont?.Dispose();
            }
            base.Dispose(disposing);
        }

        #endregion
    }
}
