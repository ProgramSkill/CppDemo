# LaTeX 介绍

> 一份从入门到精通的 LaTeX 文档编写指南

---

## 目录

| 章节 | 内容 |
|------|------|
| 一 | LaTeX 简介 |
| 二 | 安装与配置 |
| 三 | 基础语法 |
| 四 | 数学公式 |
| 五 | 图形与表格 |
| 六 | 进阶功能 |
| 七 | 常用宏包 |
| 八 | 实战技巧 |
| 九、常见问题 |
| 十、资源推荐 |

---

## 一、LaTeX 简介

### 1.1 什么是 LaTeX？

> **LaTeX** 是一个高质量的排版系统，由 Leslie Lamport 在 1984 年创建，基于 Donald Knuth 开发的 TeX 排版系统。

### 1.2 LaTeX 名称的含义

#### 缩写由来

| 部分 | 来源 | 含义 |
|------|------|------|
| **La** | Leslie Lamport | 创建者的姓氏 |
| **TeX** | τ + ε + χ | 希腊字母组合，源自希腊词 "τέχνη" (techne) |

```
LaTeX = Lamport TeX
```

**TeX** 的含义：
- 希腊词 **τέχνη** (techne) = 艺术、技艺、工艺
- 由 Donald Knuth 命名，代表"排版艺术"

#### 正确发音

| 写法 | 发音 | 说明 |
|------|------|------|
| **LaTeX** | /ˈlɑːtɛx/ 或 /ˈleɪtɛx/ | "La" 发音如 "large" 或 "late" |
| | | "TeX" 发音如 "loch" 中的 "ch" |
| **错误发音** | /ˈleɪtɛks/ | ❌ 不要读成 "Lateks"（像 Latex 橡胶） |

#### 正确书写

```
✓ 正确：LaTeX
✗ 错误：Latex、LATEX、latex
```

- **L** 和 **T** 大写，**a** 和 **e** 小写
- **X** 必须大写（代表希腊字母 χ）

### 1.3 LaTeX vs Word

| 特性 | LaTeX | Word |
|------|--------|------|
| **学习曲线** | 较陡，需要学习语法 | 平缓，所见即所得 |
| **文档稳定性** | 格式与内容分离 | 格式与内容混合 |
| | 长文档更稳定 | 大文档易崩溃 |
| **数学公式** | 极其强大，可排版复杂公式 | 公式编辑器功能有限 |
| | 天然支持 | 需要插件或特殊输入 |
| **参考文献** | 自动管理，格式统一 | 手动管理，易出错 |
| | BibTeX 支持 | 需手动管理 |
| **可移植性 | 文本文件，跨平台 | 二进制格式，版本相关 |
| | 源码管理友好 | 不适合版本控制 |
| **免费开源** | 完全免费 | 付费软件 |

### 1.4 为什么选择 LaTeX？

| 适用场景 | 说明 |
|----------|------|
| **学术论文** | 各大期刊、会议论文模板 |
| | 格式统一，自动生成参考文献 |
| **数学文档** | 数学公式排版美观 |
| | 支持复杂公式、算法伪代码 |
| **技术文档** | API 文档、手册 |
| | 支持代码高亮、交叉引用 |
| **书籍出版** | 专业出版级排版 |
| | 页码、目录、索引自动化 |
| | 版本控制友好 |

---

## 二、安装与配置

### 2.1 TeX 发行版

| 发行版 | 平台 | 特点 |
|--------|------|------|
| **TeX Live** | Windows/Linux/macOS | 最全的 TeX 发行版 |
| | | 包含几乎所有宏包 |
| **MiKTeX** | Windows | 安装小，自动安装宏包 |
| | | 适合 Windows 用户 |
| | | 存在安全漏洞修复机制 |
| **MacTeX** | macOS | macOS 标准 TeX 发行版 |
| | | 集成编辑器 TeXShop |

### 2.2 在线 LaTeX 编辑器

| 编辑器 | 特点 |
|--------|------|
| **Overleaf** | 在线协作，实时预览 |
| | 大量模板 |
| **Papeira** | 跨平台，支持本地/云端 |
| | 同步 GitHub |
| **Authorea** | 在线编辑器，支持多种期刊模板 |
| **ShareLaTeX** | 可在线编辑和协作 |

### 2.3 基础文档结构

```latex
\documentclass{article}      % 文档类型

\usepackage{ctex}           % 中文支持（使用 xeLaTeX 编译）

\title{我的第一个LaTeX文档}  % 标题
\author{张三}               % 作者
\date{\today}             % 日期

\begin{document}           % 文档开始

\maketitle                 % 生成标题

这里是文档内容...

\end{document}             % 文档结束
```

---

## 三、基础语法

### 3.1 文档类型

```latex
% 文章
\documentclass{article}

% 报告
\ocumentclass{report}

% 书籍
\documentclass{book}

% 演示文稿
\documentclass{beamer}

% 中文支持
\documentclass{ctexart}    % 中文文章
\documentclass{ctexbook}    % 中文书籍
```

### 3.2 章节结构

```latex
\section{一级章节}
\subsection{二级章节}
\subsubsection{三级章节}
\paragraph{段落}
```

### 3.3 文本格式

```latex
\textbf{粗体}
\textit{斜体}
\underline{下划线}
\texttt{等宽字体}

% 字体大小
\tiny
\scriptsize
\footnotesize
\small
\normalsize
\large
\Large
\Huge
```

### 3.4 列表

#### 无序列表

```latex
\begin{itemize}
  \item 第一项
  \item 第二项
  \begin{itemize}
    \item 嵌套列表
    \item 嵌套列表
  \end{itemize}
\end{itemize}
```

#### 有序列表

```latex
\begin{enumerate}
  \item 第一项
  \item 第二项
\end{enumerate}
```

#### 描述列表

```latex
\begin{description}
  \item[标签] 描述内容
  \item[标签2] 描述内容2
\end{description}
```

---

## 四、数学公式

### 4.1 行内公式与独立公式

```latex
% 行内公式（同一行）
这是行内公式 $E = mc^2$ 的例子。

% 独立公式（单独一行）
$$
E = mc^2
$$
% 带编号的公式
$$
a^2 + b^2 = c^2 \label{eq:pythagorean}
```

### 4.2 基础数学符号

```latex
% 运算符
$a + b$    $a - b$    $a \times b$    $a \div b$

% 关系符号
$a = b$    $a \neq b$    $a \approx b$    $a > b$
$a \geq b$    $a \leq b$

% 集合符号
$a \in A$    $a \notin B$    $A \subset B$    $A \cup B$
$A \cap B$    $A \setminus B$    $A^c$

% 箭头
$\leftarrow$    $\rightarrow$    $\leftrightarrow$
$\Leftarrow$    $\Rightarrow$    $\Leftrightarrow$

% 希腊字母
$\alpha$ $\beta$ $\gamma$ $\delta$ $\theta$ $\lambda $\mu$
$\pi$ $\phi$ $\omega$ $\rho$ $\sigma

% 上标下标
$x^2$    $x_i$    $x_{i+1}$    $x^{-1}$
```

### 4.3 分数与根式

```latex
% 分数
$\frac{a}{b}$              % 简单分数
$\frac{a+b}{c+d}$          % 复杂分数

% 连分数
$cfrac{a}{b}$             % 连分数（更紧凑）

% 根式
$\sqrt{x}$                % 平方根
$\sqrt[n]{x}$             % n 次方根
```

### 4.4 矩阵

```latex
% 括号矩阵
$\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}$

% 方括号矩阵
$\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}$

% 大矩阵
$\left[
\begin{matrix}
a & b & c \\
d & e & f \\
g & h & i
\end{matrix}
\right]$
```

### 4.5 多行公式

```latex
% 方程组
$$
\begin{cases}
x + y = 5 \\
x - y = 3
\end{cases}
$$

% 对齐
$$
\begin{aligned}
a &= b + c \\
  &= d + e \\
  &= f
\end{aligned}
$$

% 分段函数
$$
f(x) =
\begin{cases}
x + 1 & x < 0 \\
2x & x \geq 0
\end{cases}
$$
```

---

## 五、图形与表格

### 5.1 TikZ 绘图

```latex
\usepackage{tikz}

% 简单图形
\begin{tikzpicture}
  \draw (0,0) -- (4,0) -- (4,3) -- cycle;
  \node at (2,1.5) {三角形};
\end{tikzpicture}
```

### 5.2 表格

#### 基础表格

```latex
\begin{tabular}{|c|c|c|}
\hline
姓名 & 年龄 & 成绩 \\
\hline
张三 & 18 & 95 \\
\hline
李四 & 19 & 98 \\
\hline
\end{tabular}
```

#### 复杂表格

```latex
\begin{table}[htbp]
\caption{学生成绩表}
\centering
\begin{tabular}{|l|c|c|c|}
\hline
姓名 & 语文 & 数学 & 英语 & 总分 \\
\hline
\hline
张三 & 85 & 92 & 88 & 265 \\
\hline
李四 & 90 & 95 & 87 & 272 \\
\hline
\end{tabular}
\label{tab:scores}
\end{table}
```

---

## 六、进阶功能

### 6.1 交叉引用

```latex
% 标签
\section{引言}\label{sec:intro}

% 引用
见第\ref{sec:intro}节

% 引用公式
从公式\eqref{eq:pythagorean}可知...

% 引用表格
如表\ref{tab:scores}所示...

% 引用页面
如第\pageref{sec:intro}页所述...
```

### 6.2 参考文献

```latex
\bibliographystyle{plain}
\bibliography{ref}

@book{lamport1994,
  title={LaTeX: A Document Preparation System},
  author={Lamport, Leslie},
  year={1994},
  publisher={Addison-Wesley}
}

@article{einstein1905,
  title={Zur Elektrodynamik bewegter Körper},
  author={Einstein, Albert},
  journal={Annalen der Physik},
  volume={322},
  number={10},
  pages={891-921},
  year={1905}
}
```

### 6.3 自定义命令

```latex
% 简单命令
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}

% 带参数的命令
\newcommand{\vect}[1]{\mathbf{#1}}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}

% 使用
$\vect{v}$, $\norm{v}$
```

### 6.4 定理环境

```latex
\usepackage{amsthm}

\newtheorem{theorem}{定理}
\newtheorem{lemma}{引理}
\newtheorem{corollary}{推论}
\newtheorem{definition}{定义}

\begin{theorem}[费马]
对于 $n > 2$，方程 $x^n + y^n = z^n$ 没有正整数解。
\end{theorem}
```

---

## 七、常用宏包

### 7.1 中文支持

```latex
% ctex（推荐）
\usepackage{ctex}              % xeLaTeX 编译
% \documentclass{ctexart}

% CTeX
\usepackage{CJK}
```

### 7.2 数学宏包

```latex
\usepackage{amsmath}           % AMS 数学
\usepackage{amssymb}           % AMS 符号
\usepackage{amsthm}            % AMS 定理环境
\usepackage{mathtools}         % 扩展功能

% 使用示例
\usepackage{mathtools}
\DeclareMathOperator{\diag}{diag}
```

### 7.3 图形宏包

```latex
\usepackage{tikz}             % 绘图
\usepackage{pgfplots}         % 函数图
\pgfplotsset{compat=1.18}

% 函数图示例
\begin{tikzpicture}
  \begin{axis}
    \addplot[domain=0:4] {x^2};
  \end{axis}
\end{tikzpicture}
```

### 7.4 代码高亮

```latex
\usepackage{listings}

\lstsetlanguage{C++}
\lstset{
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    commentstyle=\color{green!60!black},
    stringstyle=\color{orange},
    frame=single,
    breaklines=true
}

\begin{lstlisting}[language=C++]
#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
\end{lstlisting}
```

---

## 八、实战技巧

### 8.1 文档编译

#### 编译方式

| 编译器 | 特点 | 适用 |
|--------|------|------|
| **pdfLaTeX** | 最常用，支持中文 | 日常使用 |
| **XeLaTeX** | 支持 Unicode 字体 | 中文文档 |
| **LuaLaTeX** | 可编程 | 高级用户 |

#### 编译命令

```bash
# 编译一次
xelatex filename.tex

# 完整编译（处理交叉引用）
xelatex filename.tex
bibtex filename
xelatex filename.tex
xelatex filename.tex
```

### 8.2 中文文档模板

```latex
\documentclass[UTF8]{ctexart}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{xcolor}

\hypersetup{
    colorlinks=true,
    linkcolor=blue
}

\title{\LaTeX 入门}
\author{张三}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\section{引言}
这里是内容...
\end{document}
```

### 8.3 常见错误与解决

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `! Undefined control sequence` | 命令拼写错误 | 检查拼写 |
| `! Missing $ inserted` | 数学模式错误 | 检查 `$ ... $` 配对 |
| `! LaTeX Error: File ended` | 环境未闭合 | 检查 `\begin{...}` 和 `\end{...}` 配对 |
| 中文乱码 | 编译器不支持中文 | 使用 xeLuaTeX + ctex |
| 表格错位 | 表格行末 `\\` 漏加 | 检查每行末尾 |
| 图文不居中 | 忘要额外包 | 使用 `\usepackage{float}` |

### 8.4 编译优化

```latex
% 超时解决方案
% 使用 \usepackage{etex}
% 临时禁用某些功能
```

---

## 九、常见问题

### Q1: LaTeX 和 Word 哪个更好？

**答**：取决于使用场景。

| 场景 | 推荐 |
|------|------|
| 数学论文 | LaTeX |
| 中文论文 | LaTeX (ctex) |
| 简短文档 | Word |
| 需要版本控制 | LaTeX |
| 协作文档 | Word 或 Overleaf |

### Q2: 如何学习 LaTeX？

**答**：

```
阶段1：基础语法（1-2周）
├─ 文档结构
├─ 文本格式
├─ 列表
└─ 简单数学公式

阶段2：数学公式（2-4周）
├─ 复杂公式
├─ 多行公式
├─ 矩阵
└─ 定理环境

阶段3：高级功能（按需学习）
├─ 图形 (TikZ)
├─ 表格
├─ 参考文献
└─ 自定义命令
```

### Q3: Overleaf vs 本地编辑器？

| 特性 | Overleaf | 本地编辑器 |
|------|---------|------------|
| 优点 | 在线、协作、实时预览 | 完全控制、无网络依赖 |
| 缺点 | 需要网络 | 需要自己配置 |

### Q4: 如何处理中文？

**答**：使用中文支持方案：

```latex
% 方案1：ctex（推荐）
\documentclass{ctexart}
\usepackage{ctex}

% 方案2：xeLaTeX + xeCJK
\documentclass{article}
\usepackage{xeCJK}
\setCJKmainfont{SimSun}
```

---

## 十、资源推荐

### 10.1 在线资源

| 资源 | 链接 |
|------|------|
| LaTeX 官方文档 | https://ctan.org/ |
| lshort.pdf（一份简短介绍） | https://ctan.org/tex-archive/info/lshort/lshort.pdf |
| lshort-zh-cn.pdf（中文版） | https://ctan.org/pkg/lshort-zh-cn/ |
| Overleaf 指南 | https://www.overleaf.com/learn |
| TeX Stack Exchange | https://tex.stackexchange.com/ |

### 10.2 推荐书籍

| 书名 | 作者 | 适用阶段 |
|------|------|----------|
| 《lshort》 | Leslie Lamport | 入门必读 |
| 《LaTeX 入门》 | 刘海洋 | 入门教程 |
| 《LaTeX 入门与进阶》 | 陈志旭、赵旭 |
| 《LaTeX 完全学习手册》 | 胡伟 |
| 《LaTeX 数学排版》 | 罗铁 |
| 《LaTeX 图形排版》 | 张凯 |

### 10.3 模板资源

- **CTEX 文档类**：中文论文标准模板
- **IEEE 模板**：期刊论文模板
- **Thesis 模板**：学位论文模板
- **Beamer 模板**：演示文稿模板

---

## 十一、快速参考

### 11.1 基础模板

```latex
\documentclass{ctexart}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}

\title{标题}
\author{作者}
\date{\today}

\begin{document}
\maketitle
\tableofcontents

\section{第一节}
内容...

\subsection{小节}
内容...

\subsection{数学公式示例}
$$
E = mc^2
$$

\section{参考文献}
% 参考文献...

\end{document}
```

### 11.2 数学公式速查

```latex
% 希腊字母
$\alpha \beta \gamma \delta \epsilon \zeta \eta \theta$
$\iota \kappa \lambda \mu \nu \xi \pi \rho \sigma \tau \upsilon \phi \chi \psi \omega$

% 关系符号
\leq \geq \neq \approx \equiv \sim \propto

% 箭头
\rightarrow \leftarrow \leftrightarrow
\Rightarrow \Leftarrow \Leftrightarrow

% 集合
\in \notin \subset \subseteq \cup \cap \setminus
\emptyset \varnothing \infty

% 运算
\times \div \pm \mp \cdot *

% 特殊符号
\forall \exists \therefore \because \nabla \partial \infty
```

### 11.3 常用命令

```latex
% 文字效果
\textbf{粗体} \textit{斜体} \underline{下划线}
\emph{强调} \texttt{等宽}

% 空间
\quad    空格
\qquad  两个空格
\!      负空间
\hspace{1em} 指定宽度

% 分数
\frac{a}{b}              简单分数
\dfrac{a}{b}             显示模式分数

% 括号
\left( \right)           自动调整大小
\left[ \right]           方括号
\left\{ \right\}           花括号
```

---

## 十二、实战示例

### 12.1 数学试卷模板

```latex
\documentclass[UTF8]{ctexart}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{ctex}

\geometry{a4paper, margin=2.5cm}

\title{初二数学期末试卷}
\author{教务处}
\date{2025年1月}

\begin{document}
\maketitle

\section{一、选择题（每题3分，共30分）}

1. 下列各数中，最大的数是\hfill $\root[3]{-8}$ \hfill

A. $-8$  B. $-3$  C. $0$  D. $3$

...

\section{二、填空题（每题4分，共20分）}

1. 若 $|x| = 3$，则 $x = \hfill

...

\section{三、解答题（共50分）}

1. 解方程：$2x^2 - 3x - 2 = 0$

\noindent \textbf{解：}

$$
\begin{aligned}
2x^2 - 3x - 2 = 0 \\
(2x + 1)(x - 2) = 0
\end{aligned}
$$

$\therefore x_1 = -\frac{1}{2}, x_2 = 2$

...

\end{document}
```

### 12.2 代码文档模板

```latex
\documentclass{article}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}

\lstsetlanguage{C++}
\lstset{
    basicstyle=\ttfamily\small,
    keywordstyle=\color{blue},
    stringstyle=\color{green!60!black},
    commentstyle=\color{gray},
    frame=single,
    breaklines=true,
    numbers=left,
    numberstyle=\tiny\color{gray},
    captionpos=b
}

\lstdefinelanguage{C++}

\title{C++ 代码文档模板}
\author{开发团队}
\date{\today}

\begin{document}
\makettitle

\section{代码示例}

\section{基础语法}

\begin{lstlisting}[caption={Hello World}]
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
\end{lstlisting}

\section{函数定义}

\begin{lstlisting}[caption={函数模板}]
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
\end{lstlisting}

\end{document}
```

---

## 十三、总结

### 13.1 学习路径

```
第一阶段：基础语法（1个月）
├─ 文档结构
├─ 文本格式
├─ 列表
└─ 简单公式

第二阶段：数学排版（1个月）
├─ 复杂公式
├─ 定理环境
└─ 矩阵与符号

第三阶段：进阶功能（按需）
├─ 图形 (TikZ)
├─ 参考文献
└─ 自定义命令

第四阶段：专业应用
├─ 学位论文
├→ 期刊论文
└→ 技术文档
```

### 13.2 核心要点

1. **内容为王** | **Content is King**：LaTeX 只是工具，内容质量最重要
2. **渐进学习** | **Learn Progressively**：从简单开始，逐步深入
3. **实践为主** | **Practice More**：多写多练，积累经验
4. **善用资源** | **Use Resources**：充分利用模板和社区

---

**最后更新**：2025年1月

**声明**：本文档仅供参考，具体配置请根据实际环境和需求调整。
