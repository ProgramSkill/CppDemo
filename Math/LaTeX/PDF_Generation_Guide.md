# Markdown 转 PDF 命令指南

## 前置条件

1. 安装 [pandoc](https://pandoc.org/)
2. 安装 [MiKTeX](https://miktex.org/) (包含 XeLaTeX 引擎)

## XeLaTeX 路径

在 Windows 上，XeLaTeX 通常安装在：

```
C:\Users\Jason\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe
```

## 完整命令

```bash
cd "C:\Users\Jason\source\repos\ProgramSkill\CppDemo\Math\Grade7Math\Grade7B" && pandoc "Chapter10_SystemsOfBinaryLinearEquations.md" -o "Chapter10_SystemsOfBinaryLinearEquations_Compact.pdf" --pdf-engine="C:\Users\Jason\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" -V CJKmainfont="Microsoft YaHei" -V fontsize=10pt -V geometry:margin=1cm --number-sections
```

---

## 命令结构

```
pandoc [输入文件] -o [输出文件] [选项...]
```

---

## 参数详解

| 参数 | 说明 |
|------|------|
| `pandoc` | 调用 pandoc 程序 |
| `"Chapter10_SystemsOfBinaryLinearEquations.md"` | 输入的 Markdown 文件 |
| `-o "Chapter10_SystemsOfBinaryLinearEquations_Compact.pdf"` | 输出 PDF 文件名 |
| `--pdf-engine="..."` | 指定 PDF 生成引擎 (XeLaTeX) |
| `-V CJKmainfont="Microsoft YaHei"` | 中文字体设为微软雅黑 |
| `-V fontsize=10pt` | 字体大小 10pt (紧凑) |
| `-V geometry:margin=1cm` | 页边距 1cm (紧凑) |
| `--number-sections` | 章节自动编号 |

---

## 各参数详细说明

### `--pdf-engine`

指定 LaTeX 引擎。可选值：

| 引擎 | 说明 |
|------|------|
| `xelatex` | 支持 Unicode 和中文，推荐 |
| `lualatex` | 类似 XeLaTeX |
| `pdflatex` | 不直接支持中文 |

### `-V` (Variable)

设置 LaTeX 模板变量。格式：`-V key=value`

#### 字体相关

```bash
-V CJKmainfont="Microsoft YaHei"    # 中文字体
-V CJKmainfont="SimSun"             # 宋体
-V mainfont="Times New Roman"       # 英文字体
```

#### 字号相关

```bash
-V fontsize=10pt    # 紧凑
-V fontsize=11pt    # 标准
-V fontsize=12pt    # 较大
```

#### 页面边距

```bash
-V geometry:margin=1cm     # 紧凑 (四周 1cm)
-V geometry:margin=2cm     # 标准
-V geometry:margin=3cm     # 宽松
```

单独设置各边：

```bash
-V geometry:top=1cm -V geometry:bottom=1.5cm -V geometry:left=1cm -V geometry:right=1cm
```

#### 纸张大小

```bash
-V geometry:paperwidth=210mm -V geometry:paperheight=297mm    # A4
-V geometry:paperwidth=216mm -V geometry:paperheight=279mm    # Letter
```

---

## 其他常用选项

### 目录相关

```bash
--toc                    # 生成目录
--toc-depth=2            # 目录深度 (默认 3)
```

### 编号相关

```bash
--number-sections        # 章节编号
--number-offsets=1       # 编号起始值
```

### 链接颜色

```bash
-V colorlinks=true       # 启用彩色链接
-V linkcolor=blue        # 内部链接颜色
-V urlcolor=blue         # URL 链接颜色
```

---

## 常用组合示例

### 紧凑型 (小字 + 小边距)

```bash
pandoc "input.md" -o "output.pdf" --pdf-engine="C:\Users\Jason\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" -V CJKmainfont="Microsoft YaHei" -V fontsize=10pt -V geometry:margin=1cm
```

### 标准型

```bash
pandoc "input.md" -o "output.pdf" --pdf-engine="C:\Users\Jason\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" -V CJKmainfont="Microsoft YaHei" -V fontsize=11pt -V geometry:margin=2cm --toc
```

### 带目录和链接

```bash
pandoc "input.md" -o "output.pdf" --pdf-engine="C:\Users\Jason\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" -V CJKmainfont="Microsoft YaHei" -V fontsize=11pt -V geometry:margin=2cm --toc --number-sections -V colorlinks=true -V linkcolor=blue
```

---

## 故障排除

### 问题：`xelatex not found`

**原因**：pandoc 找不到 xelatex.exe

**解决**：使用完整路径

```bash
--pdf-engine="C:\Users\Jason\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
```

### 问题：中文显示为方块或乱码

**原因**：未指定中文字体或字体不存在

**解决**：指定系统可用字体

```bash
-V CJKmainfont="Microsoft YaHei"    # 微软雅黑
-V CJKmainfont="SimSun"             # 宋体
-V CJKmainfont="SimHei"             # 黑体
```

### 问题：`permission denied`

**原因**：PDF 文件正在被其他程序打开（如 PDF 阅读器）

**解决**：关闭 PDF 阅读器后重试

### 问题：特殊字符警告

```
WARNING: Missing character: There is no ✓ (U+2713) in font
```

**原因**：Latin Math 字体不支持某些 Unicode 符号

**影响**：仅影响特殊符号，数学公式和主要内容正常

---

## 可用的中文字体

| 字体名称 | 说明 |
|----------|------|
| `Microsoft YaHei` | 微软雅黑 - 无衬线，现代 |
| `SimSun` | 宋体 - 衬线，正式 |
| `SimHei` | 黑体 - 无衬线 |
| `KaiTi` | 楷体 - 手写风格 |
| `FangSong` | 仿宋 - 类似宋体 |

---

## 快捷脚本 (可选)

### Windows 批处理文件 (.bat)

```batch
@echo off
set INPUT=%1
set OUTPUT=%~n1.pdf
set XELATEX=C:\Users\Jason\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe

pandoc "%INPUT%" -o "%OUTPUT%" --pdf-engine="%XELATEX%" -V CJKmainfont="Microsoft YaHei" -V fontsize=10pt -V geometry:margin=1cm --number-sections

echo PDF generated: %OUTPUT%
```

使用方式：
```
md2pdf.bat Chapter10_SystemsOfBinaryLinearEquations.md
```
