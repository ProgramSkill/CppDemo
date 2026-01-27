# Basic I/O 术语中英文对照表
# Basic I/O Terminology - Chinese-English Glossary

本文档提供 Basic I/O 示例中所有重要术语的中英文对照。
This document provides Chinese-English translation for all important terms in Basic I/O examples.

---

## 📚 基础概念 / Basic Concepts

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 输入/输出 | Input/Output (I/O) | 数据进出微控制器的接口 |
| 端口 | Port | 一组引脚，如 P0, P1, P2, P3 |
| 引脚 | Pin | 单个物理连接点 |
| 位 | Bit | 最小数据单位，0 或 1 |
| 字节 | Byte | 8 位数据 |
| 特殊功能寄存器 | Special Function Register (SFR) | 控制硬件的寄存器 |
| 位寻址 | Bit-addressable | 可单独访问每个位的特性 |

---

## 🔌 端口与引脚 / Ports and Pins

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| P0 口 | Port 0 (P0) | 8 位双向 I/O 端口 |
| P1 口 | Port 1 (P1) | 8 位准双向 I/O 端口 |
| P2 口 | Port 2 (P2) | 8 位准双向 I/O 端口 |
| P3 口 | Port 3 (P3) | 8 位准双向 I/O 端口，具有复用功能 |
| 准双向端口 | Quasi-bidirectional port | 8051 特有的端口类型 |
| 双向端口 | Bidirectional port | 可同时输入和输出 |
| 位操作 | Bit manipulation | 对单个位进行操作 |
| 端口操作 | Port operation | 对整个端口进行操作 |

---

## 💡 LED 相关术语 / LED Related Terms

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 发光二极管 | Light Emitting Diode (LED) | 发光元件 |
| 低电平有效 | Active low | 0 = 开启，1 = 关闭 |
| 高电平有效 | Active high | 1 = 开启，0 = 关闭 |
| 灌电流 | Current sink | 电流流入端口（低电平） |
| 拉电流 | Current source | 电流流出端口（高电平） |
| 阳极 | Anode | LED 正极 |
| 阴极 | Cathode | LED 负极 |
| 电阻 | Resistor | 限流元件 |
| 限流电阻 | Current limiting resistor | 保护 LED 的电阻 |
| 闪烁 | Blink/Flash | 周期性亮灭 |
| 跑马灯 | Running LED/Chaser | LED 依次点亮的效果 |

---

## ⌨️ 编程术语 / Programming Terms

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 头文件 | Header file | .h 文件，包含声明 |
| 寄存器定义 | Register definition | SFR 的定义 |
| 单个位声明 | Single bit declaration | sbit 关键字 |
| 主函数 | Main function | 程序入口点 |
| 无限循环 | Infinite loop | while(1) 或 for(;;) |
| 延时函数 | Delay function | 产生时间延迟 |
| 嵌套循环 | Nested loop | 循环内套循环 |
| 条件语句 | Conditional statement | if-else 语句 |
| 赋值 | Assignment | 给变量赋值 |
| 取反 | Toggle/Invert | 0 变 1，1 变 0 |
| 位移 | Shift operation | << 或 >> |

---

## 🕐 时间与定时 / Timing

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 毫秒 | Millisecond (ms) | 千分之一秒 |
| 微秒 | Microsecond (µs) | 百万分之一秒 |
| 机器周期 | Machine cycle | 8051 的基本时间单位 |
| 晶振频率 | Crystal frequency | 时钟频率，如 12MHz |
| 软件延时 | Software delay | 用循环实现的延时 |
| 硬件定时器 | Hardware timer | 精确定时的硬件模块 |
| 时间精度 | Timing precision | 延时的准确性 |
| 校准 | Calibration | 调整延时以匹配实际时间 |

---

## 🔧 硬件连接 / Hardware Connections

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 电路图 | Circuit diagram | 连接示意图 |
| 万能板 | Breadboard | 用于原型开发 |
| 跳线 | Jumper wire | 连接导线 |
| 电源 | Power supply (VCC) | 正电源，通常 5V |
| 地 | Ground (GND) | 零电位参考 |
| 极性 | Polarity | 元件的方向性 |
| 正极 | Positive terminal | 高电位端 |
| 负极 | Negative terminal | 低电位端 |
| 负载 | Load | 消耗功率的元件 |

---

## ⚡ 电源与电流 / Power and Current

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 电流 | Current (I) | 单位：安培(A)、毫安(mA)、微安(µA) |
| 电压 | Voltage (V) | 单位：伏特(V) |
| 功率 | Power (P) | 单位：瓦特(W) |
| 电流限制 | Current limit | 最大允许电流 |
| 额定电流 | Rated current | 正常工作电流 |
| 峰值电流 | Peak current | 瞬间最大电流 |
| 总电流 | Total current | 所有引脚电流之和 |
| 电源电流 | Supply current | 芯片消耗的总电流 |

---

## 🐛 调试与故障排除 / Debugging and Troubleshooting

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 调试 | Debugging | 查找和修复错误 |
| 故障排除 | Troubleshooting | 解决问题的过程 |
| 预期行为 | Expected behavior | 程序应有的表现 |
| 实际行为 | Actual behavior | 程序实际的表现 |
| 极性错误 | Wrong polarity | LED 接反了 |
| 连接错误 | Wrong connection | 接线错误 |
| 编译错误 | Compilation error | 代码语法错误 |
| 烧录 | Programming/Flash | 将代码写入芯片 |
| 万用表 | Multimeter | 测量电压、电流的工具 |
| 示波器 | Oscilloscope | 查看波形的工具 |

---

## 📊 代码模式与变化 / Code Patterns and Variations

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 多路 LED | Multiple LEDs | 控制多个 LED |
| 跑马灯 | Running LED/Chaser | LED 依次点亮 |
| 摩斯码 | Morse code | 用长短信号表示字符 |
| SOS 求救信号 | SOS distress signal | 求救信号：··· --- ··· |
| 点 | Dot | 短信号 |
| 划 | Dash | 长信号 |
| 速度控制 | Speed control | 改变闪烁速度 |
| 花样 | Pattern | LED 显示的模式 |
| 移位 | Shift | 位向左或向右移动 |

---

## 🔬 位操作详解 / Bit Operations Details

| 中文 | English | 代码示例 / Code Example |
|------|---------|------------------------|
| 置位 | Set bit | `P1 \|= (1 << 0)` 或 `P1_0 = 1` |
| 清零 | Clear bit | `P1 &= ~(1 << 0)` 或 `P1_0 = 0` |
| 取反 | Toggle bit | `P1 ^= (1 << 0)` 或 `P1_0 = ~P1_0` |
| 测试位 | Test bit | `if(P1_0)` |
| 按位或 | Bitwise OR | `\|` |
| 按位与 | Bitwise AND | `&` |
| 按位异或 | Bitwise XOR | `^` |
| 按位取反 | Bitwise NOT | `~` |
| 左移 | Left shift | `<<` |
| 右移 | Right shift | `>>` |

---

## 📖 C 语言关键字 / C Language Keywords

| 中文 | English | 用途 / Usage |
|------|---------|-------------|
| include | Include | 包含头文件 |
| define | Define | 定义宏 |
| sbit | Single bit | 声明位变量 |
| void | Void | 无返回值 |
| unsigned | Unsigned | 无符号数 |
| char | Char | 字符型（8位） |
| int | Int | 整型（16位） |
| if | If | 条件判断 |
| else | Else | 否则分支 |
| while | While | 当型循环 |
| for | For | 计数循环 |
| return | Return | 返回 |
| volatile | Volatile | 防止编译器优化 |

---

## 🎯 编程最佳实践 / Programming Best Practices

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 代码注释 | Code comments | 解释代码的文本 |
| 函数原型 | Function prototype | 函数声明 |
| 参数 | Parameter | 函数输入 |
| 返回值 | Return value | 函数输出 |
| 局部变量 | Local variable | 函数内部变量 |
| 全局变量 | Global variable | 整个程序可访问的变量 |
| 命名规范 | Naming convention | 变量命名规则 |
| 代码缩进 | Code indentation | 代码层级结构 |
| 模块化 | Modularity | 将代码分成模块 |
| 可读性 | Readability | 代码易读性 |

---

## 🔍 特殊寄存器位 / Special Register Bits

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| P0^0 - P0^7 | Port 0 bits | P0 口的 8 个位 |
| P1^0 - P1^7 | Port 1 bits | P1 口的 8 个位 |
| P2^0 - P2^7 | Port 2 bits | P2 口的 8 个位 |
| P3^0 - P3^7 | Port 3 bits | P3 口的 8 个位 |
| P1_0 | P1 bit 0 | P1.0 的另一种写法 |
| 位掩码 | Bit mask | 用于位操作的掩码值 |

---

## 📐 计算公式 / Calculation Formulas

| 中文 | English | 公式 / Formula |
|------|---------|----------------|
| LED 电流计算 | LED current calculation | I = (VCC - VLED) / R |
| 欧姆定律 | Ohm's law | V = I × R |
| 功率计算 | Power calculation | P = V × I |
| 机器周期 | Machine cycle | T = 12 / 晶振频率 |
| 延时估算 | Delay estimation | 时间 ≈ 循环次数 × 机器周期 |

---

## 🎓 学习路径 / Learning Path

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 基础 I/O | Basic I/O | 端口输入输出 |
| 定时器 | Timers | 精确定时 |
| 中断 | Interrupts | 事件驱动 |
| 串口通信 | Serial communication | 数据传输 |
| 高级应用 | Advanced applications | 综合运用 |

---

## 💬 常用短语 / Common Phrases

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 活动低 | Active low | 低电平激活 |
| 活动高 | Active high | 高电平激活 |
| 额定值 | Rated value | 标准工作值 |
| 安全值 | Safe value | 不会损坏元件的值 |
| 典型值 | Typical value | 常用值 |
| 最大值 | Maximum value | 上限 |
| 最小值 | Minimum value | 下限 |
| 推荐值 | Recommended value | 建议使用的值 |

---

## 📏 测量单位 / Measurement Units

| 中文 | English | 符号 / Symbol |
|------|---------|---------------|
| 伏特 | Volt | V |
| 安培 | Ampere | A |
| 毫安 | Milliampere | mA |
| 微安 | Microampere | µA |
| 欧姆 | Ohm | Ω |
| 千欧 | Kilo-ohm | kΩ |
| 兆欧 | Mega-ohm | MΩ |
| 赫兹 | Hertz | Hz |
| 千赫 | Kilohertz | kHz |
| 兆赫 | Megahertz | MHz |
| 秒 | Second | s |
| 毫秒 | Millisecond | ms |
| 微秒 | Microsecond | µs |
| 瓦特 | Watt | W |
| 毫瓦 | Milliwatt | mW |

---

## 🔧 工具与设备 / Tools and Equipment

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 编译器 | Compiler | 将代码转换为机器码 |
| 仿真器 | Simulator | 模拟芯片运行 |
| 烧录器 | Programmer | 将程序写入芯片 |
| 开发板 | Development board | 用于学习和开发 |
| 面包板 | Breadboard | 无需焊接的连接板 |
| 万用表 | Multimeter | 测量电压、电流、电阻 |
| 逻辑分析仪 | Logic analyzer | 查看数字信号 |
| 示波器 | Oscilloscope | 查看模拟信号 |

---

## 📝 文档类型 / Document Types

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 数据手册 | Datasheet | 芯片的技术规格 |
| 参考手册 | Reference manual | 详细功能说明 |
| 应用笔记 | Application note | 实际应用指南 |
| 用户指南 | User guide | 使用说明 |
| 教程 | Tutorial | 学习材料 |
| 示例代码 | Example code | 参考程序 |
| 技术文档 | Technical documentation | 技术说明 |

---

## 🚀 常用开发工具 / Common Development Tools

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| Keil C51 | Keil C51 | 商业 8051 编译器 |
| SDCC | SDCC | 开源 8051 编译器 |
| Proteus | Proteus | 电路仿真软件 |
| USBasp | USBasp | USB 烧录器 |
| ISP 编程器 | ISP programmer | 在系统编程器 |

---

## 📊 LED 常见颜色 / Common LED Colors

| 中文 | English | 典型电压降 / Typical Vf |
|------|---------|----------------------|
| 红色 LED | Red LED | 1.8V - 2.2V |
| 绿色 LED | Green LED | 1.9V - 2.4V |
| 黄色 LED | Yellow LED | 2.0V - 2.4V |
| 蓝色 LED | Blue LED | 2.8V - 3.3V |
| 白色 LED | White LED | 2.8V - 3.3V |

---

## 🎯 编程技巧 / Programming Tips

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 宏定义 | Macro definition | 使用 #define 定义常量 |
| 位操作 | Bit operation | 高效的位控制 |
| 函数封装 | Function encapsulation | 将功能封装为函数 |
| 代码复用 | Code reuse | 避免重复代码 |
| 注释清晰 | Clear comments | 便于理解 |
| 模块化设计 | Modular design | 分层设计 |

---

## 🔍 常见错误类型 / Common Error Types

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 语法错误 | Syntax error | 代码不符合语法 |
| 逻辑错误 | Logic error | 程序逻辑不对 |
| 运行时错误 | Runtime error | 运行时出错 |
| 编译错误 | Compilation error | 编译失败 |
| 链接错误 | Link error | 链接失败 |
| 警告 | Warning | 非致命问题，但需注意 |

---

## 📚 学习建议 / Learning Tips

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 从简单开始 | Start simple | 先学基础 |
| 动手实践 | Hands-on practice | 实际操作 |
| 理论结合 | Theory with practice | 理论与实践结合 |
| 阅读代码 | Read code | 阅读他人代码 |
| 修改实验 | Modify and experiment | 尝试修改 |
| 记录笔记 | Take notes | 做好笔记 |
| 问问题 | Ask questions | 不懂就问 |

---

## 📞 获取帮助 / Getting Help

| 中文 | English | 说明 / Notes |
|------|---------|--------------|
| 论坛 | Forum | 在线社区 |
| 文档 | Documentation | 技术资料 |
| 数据手册 | Datasheet | 芯片规格书 |
| 参考设计 | Reference design | 设计参考 |
| 示例代码 | Example code | 代码示例 |
| 技术支持 | Technical support | 专业帮助 |

---

## 💡 记忆口诀 / Memory Aids

### LED 连接记忆
- **低电平有效强**：Active low 拉电流强（20mA），推电流弱（60µA）
- **长正短负**：LED 长脚接正极，短脚接负极

### 位操作记忆
- **或置位**：OR 用于置 1（Set bit）
- **与清零**：AND 用于清 0（Clear bit）
- **异或取反**：XOR 用于取反（Toggle bit）

### 延时计算
- **12MHz 1 微秒**：12MHz 晶振，1 个机器周期 = 1µs
- **循环约 10**：每次循环约 10 个机器周期

---

**说明：** 本词汇表涵盖了 Basic I/O 示例中的所有关键术语。建议初学者在学习过程中随时查阅。

**Note:** This glossary covers all key terms in Basic I/O examples. Beginners are encouraged to refer to it frequently during learning.
