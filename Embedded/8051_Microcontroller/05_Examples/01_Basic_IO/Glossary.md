# Basic I/O Terminology Glossary
# Basic I/O 术语中英文对照表

本文档提供 Basic I/O 示例中所有重要术语的英中对照。
This document provides English-Chinese translation for all important terms in Basic I/O examples.

---

## 📚 Basic Concepts / 基础概念

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Input/Output (I/O) | 输入/输出 | 数据进出微控制器的接口 |
| Port | 端口 | 一组引脚，如 P0, P1, P2, P3 |
| Pin | 引脚 | 单个物理连接点 |
| Bit | 位 | 最小数据单位，0 或 1 |
| Byte | 字节 | 8 位数据 |
| Special Function Register (SFR) | 特殊功能寄存器 | 控制硬件的寄存器 |
| Bit-addressable | 位寻址 | 可单独访问每个位的特性 |

---

## 🔌 Ports and Pins / 端口与引脚

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Port 0 (P0) | P0 口 | 8 位双向 I/O 端口 |
| Port 1 (P1) | P1 口 | 8 位准双向 I/O 端口 |
| Port 2 (P2) | P2 口 | 8 位准双向 I/O 端口 |
| Port 3 (P3) | P3 口 | 8 位准双向 I/O 端口，具有复用功能 |
| Quasi-bidirectional port | 准双向端口 | 8051 特有的端口类型 |
| Bidirectional port | 双向端口 | 可同时输入和输出 |
| Bit manipulation | 位操作 | 对单个位进行操作 |
| Port operation | 端口操作 | 对整个端口进行操作 |

---

## 💡 LED Related Terms / LED 相关术语

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Light Emitting Diode (LED) | 发光二极管 | 发光元件 |
| Active low | 低电平有效 | 0 = 开启，1 = 关闭 |
| Active high | 高电平有效 | 1 = 开启，0 = 关闭 |
| Current sink | 灌电流 | 电流从外部流入端口（低电平吸入电流） |
| Current source | 拉电流 | 电流从端口流出到外部（高电平输出电流） |
| Anode | 阳极 | LED 正极 |
| Cathode | 阴极 | LED 负极 |
| Resistor | 电阻 | 限流元件 |
| Current limiting resistor | 限流电阻 | 保护 LED 的电阻 |
| Blink/Flash | 闪烁 | 周期性亮灭 |
| Running LED/Chaser | 跑马灯 | LED 依次点亮的效果 |

---

## 🔘 Button Related Terms / 按钮相关术语

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Button | 按钮 | 输入控制元件 |
| Push button | 按键 | 瞬时接触开关 |
| Normally Open (NO) | 常开 | 未按下时断开，按下时导通 |
| Normally Closed (NC) | 常闭 | 未按下时导通，按下时断开 |
| Momentary switch | 瞬动开关 | 按下时导通，松开后自动复位 |
| Latch switch | 自锁开关 | 第一次按下锁定，再次按下释放 |
| Pull-up resistor | 上拉电阻 | 将引脚拉至高电平的电阻 |
| Pull-down resistor | 下拉电阻 | 将引脚拉至低电平的电阻 |
| Active low | 低电平有效 | 按下时引脚为低电平 |
| Active high | 高电平有效 | 按下时引脚为高电平 |
| Switch bounce | 开关抖动 | 机械触点接触时的抖动现象 |
| Debouncing | 去抖动 | 消除开关抖动影响 |
| Debounce delay | 去抖动延时 | 软件去抖动的延时时间 |
| Edge detection | 边沿检测 | 检测信号变化沿 |
| Press detection | 按下检测 | 检测按钮被按下 |
| Release detection | 释放检测 | 检测按钮被释放 |
| Long press | 长按 | 按钮持续按下较长时间 |
| Short press | 短按 | 按钮快速按下并释放 |
| Single click | 单击 | 按钮按下一次 |
| Double click | 双击 | 按钮快速按两次 |
| Polling | 轮询 | 循环检测输入状态 |
| Interrupt | 中断 | 事件触发的响应机制 |
| Floating pin | 悬空引脚 | 未连接的引脚，状态不确定 |
| Weak pull-up | 弱上拉 | 内部小电流上拉，典型8051约60µA |
| Strong pull-up | 强上拉 | 外部大电流上拉，典型8051可达mA级 |
| EMI | 电磁干扰 | Electromagnetic Interference，外部电磁噪声对电路的干扰 |
| Noise immunity | 抗干扰性 | 抵抗噪声的能力 |
| Fail-safe design | 故障安全设计 | 故障时自动进入安全状态 |
| Contact bounce | 触点抖动 | 开关触点的机械抖动 |
| Bounce period | 抖动周期 | 开关抖动持续的时间（10-50ms） |

---

## ⌨️ Programming Terms / 编程术语

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Header file | 头文件 | .h 文件，包含声明 |
| Register definition | 寄存器定义 | SFR 的定义 |
| Single bit declaration | 单个位声明 | sbit 关键字 |
| Main function | 主函数 | 程序入口点 |
| Infinite loop | 无限循环 | while(1) 或 for(;;) |
| Delay function | 延时函数 | 产生时间延迟 |
| Nested loop | 嵌套循环 | 循环内套循环 |
| Conditional statement | 条件语句 | if-else 语句 |
| Assignment | 赋值 | 给变量赋值 |
| Toggle/Invert | 取反 | 0 变 1，1 变 0 |
| Shift operation | 位移 | << 或 >> |
| State machine | 状态机 | 按状态转换的程序结构 |
| Switch statement | Switch语句 | 多分支选择语句 |

---

## 🕐 Timing / 时间与定时

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Millisecond (ms) | 毫秒 | 千分之一秒 |
| Microsecond (µs) | 微秒 | 百万分之一秒 |
| Machine cycle | 机器周期 | 8051 的基本时间单位 |
| Crystal frequency | 晶振频率 | 时钟频率，如 12MHz |
| Software delay | 软件延时 | 用循环实现的延时 |
| Hardware timer | 硬件定时器 | 精确定时的硬件模块 |
| Timing precision | 时间精度 | 延时的准确性 |
| Calibration | 校准 | 调整延时以匹配实际时间 |

---

## 🔧 Hardware Connections / 硬件连接

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Circuit diagram | 电路图 | 连接示意图 |
| Breadboard | 面包板 | 用于原型开发 |
| Jumper wire | 跳线 | 连接导线 |
| Power supply (VCC) | 电源 | 正电源，通常 5V |
| Ground (GND) | 地 | 零电位参考 |
| Polarity | 极性 | 元件的方向性 |
| Positive terminal | 正极 | 高电位端 |
| Negative terminal | 负极 | 低电位端 |
| Load | 负载 | 消耗功率的元件 |

---

## ⚡ Power and Current / 电源与电流

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Current (I) | 电流 | 单位：安培(A)、毫安(mA)、微安(µA) |
| Voltage (V) | 电压 | 单位：伏特(V) |
| Power (P) | 功率 | 单位：瓦特(W) |
| Current limit | 电流限制 | 最大允许电流 |
| Rated current | 额定电流 | 正常工作电流 |
| Peak current | 峰值电流 | 瞬间最大电流 |
| Total current | 总电流 | 所有引脚电流之和 |
| Supply current | 电源电流 | 芯片消耗的总电流 |

---

## 🐛 Debugging and Troubleshooting / 调试与故障排除

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Debugging | 调试 | 查找和修复错误 |
| Troubleshooting | 故障排除 | 解决问题的过程 |
| Expected behavior | 预期行为 | 程序应有的表现 |
| Actual behavior | 实际行为 | 程序实际的表现 |
| Wrong polarity | 极性错误 | LED 接反了 |
| Wrong connection | 连接错误 | 接线错误 |
| Compilation error | 编译错误 | 代码语法错误 |
| Programming/Flashing | 烧录 | 将代码写入芯片 |
| Multimeter | 万用表 | 测量电压、电流、电阻的工具 |
| Oscilloscope | 示波器 | 查看波形的工具 |

---

## 📊 Code Patterns and Variations / 代码模式与变化

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Multiple LEDs | 多路 LED | 控制多个 LED |
| Running LED/Chaser | 跑马灯 | LED 依次点亮 |
| Morse code | 摩斯码 | 用长短信号表示字符 |
| SOS distress signal | SOS 求救信号 | 求救信号：··· --- ··· |
| Dot | 点 | 短信号 |
| Dash | 划 | 长信号 |
| Speed control | 速度控制 | 改变闪烁速度 |
| Pattern | 花样 | LED 显示的模式 |
| Shift | 移位 | 位向左或向右移动 |

---

## 🔬 Bit Operations Details / 位操作详解

| English | 中文 | 代码示例 / Code Example |
|---------|------|------------------------|
| Set bit | 置位 | `P1 |= (1 << 0)` 或 `P1_0 = 1` |
| Clear bit | 清零 | `P1 &= ~(1 << 0)` 或 `P1_0 = 0` |
| Toggle bit | 取反 | `P1 ^= (1 << 0)` 或 `P1_0 = ~P1_0` |
| Test bit | 测试位 | `if(P1_0)` |
| Bitwise OR | 按位或 | `|` |
| Bitwise AND | 按位与 | `&` |
| Bitwise XOR | 按位异或 | `^` |
| Bitwise NOT | 按位取反 | `~` |
| Left shift | 左移 | `<<` |
| Right shift | 右移 | `>>` |

---

## 📖 C Language Keywords / C 语言关键字

| English | 中文 | 用途 / Usage |
|---------|------|-------------|
| Include | 包含 | 包含头文件 |
| Define | 定义 | 定义宏 |
| sbit | 单个位 | 声明位变量 |
| void | 空 | 无返回值 |
| unsigned | 无符号 | 无符号数 |
| char | 字符型 | 字符型（8位） |
| int | 整型 | 整型（16位） |
| if | 如果 | 条件判断 |
| else | 否则 | 否则分支 |
| while | 当...时 | 当型循环 |
| for | 循环 | 计数循环 |
| return | 返回 | 返回 |
| volatile | 易失性 | 防止编译器优化 |

---

## 🎯 Programming Best Practices / 编程最佳实践

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Code comments | 代码注释 | 解释代码的文本 |
| Function prototype | 函数原型 | 函数声明 |
| Parameter | 参数 | 函数输入 |
| Return value | 返回值 | 函数输出 |
| Local variable | 局部变量 | 函数内部变量 |
| Global variable | 全局变量 | 整个程序可访问的变量 |
| Naming convention | 命名规范 | 变量命名规则 |
| Code indentation | 代码缩进 | 代码层级结构 |
| Modularity | 模块化 | 将代码分成模块 |
| Readability | 可读性 | 代码易读性 |

---

## 🔍 Special Register Bits / 特殊寄存器位

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Port 0 bits | P0^0 - P0^7 | P0 口的 8 个位 |
| Port 1 bits | P1^0 - P1^7 | P1 口的 8 个位 |
| Port 2 bits | P2^0 - P2^7 | P2 口的 8 个位 |
| Port 3 bits | P3^0 - P3^7 | P3 口的 8 个位 |
| P1 bit 0 | P1_0 | P1.0 的另一种写法 |
| Bit mask | 位掩码 | 用于位操作的掩码值 |

---

## 📐 Calculation Formulas / 计算公式

| English | 中文 | 公式 / Formula |
|---------|------|----------------|
| LED current calculation | LED 电流计算 | I = (VCC - VLED) / R |
| Ohm's law | 欧姆定律 | V = I × R |
| Power calculation | 功率计算 | P = V × I |
| Machine cycle | 机器周期 | T = 12 / 晶振频率 |
| Delay estimation | 延时估算 | 时间 ≈ 循环次数 × 机器周期 |

---

## 🎓 Learning Path / 学习路径

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Basic I/O | 基础 I/O | 端口输入输出 |
| Timers | 定时器 | 精确定时 |
| Interrupts | 中断 | 事件驱动 |
| Serial communication | 串口通信 | 数据传输 |
| Advanced applications | 高级应用 | 综合运用 |

---

## 💬 Common Phrases / 常用短语

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Active low | 低电平有效 | 低电平激活 |
| Active high | 高电平有效 | 高电平激活 |
| Rated value | 额定值 | 正常工作时的标准值 |
| Safe value | 安全值 | 不会损坏元件的安全范围 |
| Typical value | 典型值 | 典型工况下的代表值 |
| Maximum value | 最大值 | 允许的最大极限值 |
| Minimum value | 最小值 | 允许的最小极限值 |
| Recommended value | 推荐值 | 建议的最佳工作值 |

---

## 📏 Measurement Units / 测量单位

| English | 中文 | 符号 / Symbol |
|---------|------|---------------|
| Volt | 伏特 | V |
| Ampere | 安培 | A |
| Milliampere | 毫安 | mA |
| Microampere | 微安 | µA |
| Ohm | 欧姆 | Ω |
| Kilo-ohm | 千欧 | kΩ |
| Mega-ohm | 兆欧 | MΩ |
| Hertz | 赫兹 | Hz |
| Kilohertz | 千赫 | kHz |
| Megahertz | 兆赫 | MHz |
| Second | 秒 | s |
| Millisecond | 毫秒 | ms |
| Microsecond | 微秒 | µs |
| Watt | 瓦特 | W |
| Milliwatt | 毫瓦 | mW |

---

## 🔧 Tools and Equipment / 工具与设备

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Compiler | 编译器 | 将代码转换为机器码 |
| Simulator | 仿真器 | 模拟芯片运行 |
| Programmer | 烧录器 | 将程序写入芯片 |
| Development board | 开发板 | 用于学习和开发 |
| Breadboard | 面包板 | 无需焊接的连接板 |
| Multimeter | 万用表 | 测量电压、电流、电阻 |
| Logic analyzer | 逻辑分析仪 | 查看数字信号 |
| Oscilloscope | 示波器 | 查看模拟信号 |

---

## 📝 Document Types / 文档类型

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Datasheet | 数据手册 | 芯片的技术规格 |
| Reference manual | 参考手册 | 详细功能说明 |
| Application note | 应用笔记 | 实际应用指南 |
| User guide | 用户指南 | 使用说明 |
| Tutorial | 教程 | 学习材料 |
| Example code | 示例代码 | 参考程序 |
| Technical documentation | 技术文档 | 技术说明 |

---

## 🚀 Common Development Tools / 常用开发工具

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Keil C51 | Keil C51 编译器 | 商业 8051 编译器 |
| SDCC | SDCC | 开源 8051 编译器 |
| Proteus | Proteus | 电路仿真软件 |
| USBasp | USBasp | USB 烧录器 |
| ISP programmer | ISP 编程器 | 在系统编程器 |

---

## 📊 Common LED Colors / LED 常见颜色

| English | 中文 | 典型电压降 / Typical Vf |
|---------|------|----------------------|
| Red LED | 红色 LED | 1.8V - 2.2V |
| Green LED | 绿色 LED | 1.9V - 2.4V |
| Yellow LED | 黄色 LED | 2.0V - 2.4V |
| Blue LED | 蓝色 LED | 2.8V - 3.3V |
| White LED | 白色 LED | 2.8V - 3.3V |

---

## 🎯 Programming Tips / 编程技巧

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Macro definition | 宏定义 | 使用 #define 定义常量 |
| Bit operation | 位操作 | 高效的位控制 |
| Function encapsulation | 函数封装 | 将功能封装为函数 |
| Code reuse | 代码复用 | 避免重复代码 |
| Clear comments | 注释清晰 | 便于理解 |
| Modular design | 模块化设计 | 分层设计 |

---

## 🔍 Common Error Types / 常见错误类型

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Syntax error | 语法错误 | 代码不符合语法 |
| Logic error | 逻辑错误 | 程序逻辑不对 |
| Runtime error | 运行时错误 | 运行时出错 |
| Compilation error | 编译错误 | 编译失败 |
| Link error | 链接错误 | 链接失败 |
| Warning | 警告 | 非致命问题，但需注意 |

---

## 📚 Learning Tips / 学习建议

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Start simple | 从简单开始 | 先学基础 |
| Hands-on practice | 动手实践 | 实际操作 |
| Theory with practice | 理论结合 | 理论与实践结合 |
| Read code | 阅读代码 | 阅读他人代码 |
| Modify and experiment | 修改实验 | 尝试修改 |
| Take notes | 记录笔记 | 做好笔记 |
| Ask questions | 问问题 | 不懂就问 |

---

## 📞 Getting Help / 获取帮助

| English | 中文 | 说明 / Notes |
|---------|------|--------------|
| Forum | 论坛 | 在线社区 |
| Documentation | 文档 | 技术资料 |
| Datasheet | 数据手册 | 芯片规格书 |
| Reference design | 参考设计 | 设计参考 |
| Example code | 示例代码 | 代码示例 |
| Technical support | 技术支持 | 专业帮助 |

---

## 💡 Memory Aids / 记忆口诀

### LED Connection Memory / LED 连接记忆
- **Sinking is strong, sourcing is weak**：灌电流强，拉电流弱（典型8051：20mA vs 60µA）
- **Long positive short negative**：LED 长脚接正极，短脚接负极

### Button Connection Memory / 按钮连接记忆
- **Pull-up button reads low**：上拉电阻按钮按下时读低电平
- **Pull-down button reads high**：下拉电阻按钮按下时读高电平
- **10kΩ is the sweet spot**：10kΩ 是平衡功耗和抗干扰的最佳值
- **Debounce 50ms**：软件去抖动延时约 20–50ms（常用50ms）

### Bit Operation Memory / 位操作记忆
- **OR to set**：OR 用于置 1（Set bit）
- **AND to clear**：AND 用于清 0（Clear bit）
- **XOR to toggle**：XOR 用于取反（Toggle bit）

### Delay Calculation / 延时计算
- **12MHz 1 microsecond**：12MHz 晶振，1 个机器周期 = 1µs
- **Loop about 10 cycles**：每次循环约 10 个机器周期

---

**说明：** 本词汇表涵盖了 Basic I/O 示例中的所有关键术语，按英中对照排列。建议初学者在学习过程中随时查阅。

**Note:** This glossary covers all key terms in Basic I/O examples, arranged in English-Chinese order. Beginners are encouraged to refer to it frequently during learning.
