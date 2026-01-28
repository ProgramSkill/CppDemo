# 8051 Hardware Architecture Glossary
# 8051 硬件架构词汇表

This glossary provides categorized definitions of key terms used in 8051 microcontroller hardware architecture documentation.

本词汇表按分类提供8051单片机硬件架构文档中使用的关键术语定义。

---

## 1. Special Function Registers (SFR)
## 1. 特殊功能寄存器

| Register | 中文名称 | Address | Description |
|----------|---------|---------|-------------|
| ACC | 累加器 | E0H | Primary 8-bit register for arithmetic/logic operations. Bit-addressable. <br> 用于算术和逻辑运算的主要8位寄存器。可位寻址。 |
| B | B寄存器 | F0H | 8-bit register for multiplication and division. Bit-addressable. <br> 用于乘法和除法运算的8位寄存器。可位寻址。 |
| PSW | 程序状态字 | D0H | Program Status Word containing status flags (CY, AC, F0, RS1, RS0, OV, P). Bit-addressable. <br> 包含状态标志的程序状态字。可位寻址。 |
| SP | 堆栈指针 | 81H | Stack Pointer, points to top of stack in internal RAM. Default: 07H. <br> 指向内部RAM堆栈顶部的指针。默认值：07H。 |
| DPTR | 数据指针 | 82H-83H | 16-bit Data Pointer (DPH:DPL) for external memory access. <br> 用于外部存储器访问的16位数据指针。 |
| DPL | 数据指针低字节 | 82H | Low byte of DPTR. <br> DPTR的低字节。 |
| DPH | 数据指针高字节 | 83H | High byte of DPTR. <br> DPTR的高字节。 |
| P0-P3 | 端口0-3 | 80H-B0H | 8-bit I/O ports. P0 and P2 also used for external memory addressing. <br> 8位I/O端口。P0和P2也用于外部存储器寻址。 |
| IE | 中断使能寄存器 | A8H | Interrupt Enable register, controls individual interrupt enables. Bit-addressable. <br> 中断使能寄存器，控制各个中断的使能。可位寻址。 |
| IP | 中断优先级寄存器 | B8H | Interrupt Priority register, sets priority levels. Bit-addressable. <br> 中断优先级寄存器，设置优先级。可位寻址。 |
| TCON | 定时器控制寄存器 | 88H | Timer Control register, controls timer/counter operation. Bit-addressable. <br> 定时器控制寄存器，控制定时器/计数器操作。可位寻址。 |
| TMOD | 定时器模式寄存器 | 89H | Timer Mode register, sets operating modes for Timer 0 and Timer 1. <br> 定时器模式寄存器，设置定时器0和1的操作模式。 |
| TL0 | 定时器0低字节 | 8AH | Timer 0 Low byte. <br> 定时器0低字节。 |
| TH0 | 定时器0高字节 | 8CH | Timer 0 High byte. <br> 定时器0高字节。 |
| TL1 | 定时器1低字节 | 8BH | Timer 1 Low byte. <br> 定时器1低字节。 |
| TH1 | 定时器1高字节 | 8DH | Timer 1 High byte. <br> 定时器1高字节。 |
| SCON | 串行控制寄存器 | 98H | Serial Control register, controls serial port operation. Bit-addressable. <br> 串行控制寄存器，控制串口操作。可位寻址。 |
| SBUF | 串行缓冲器 | 99H | Serial Buffer, used for serial data transmission/reception. <br> 串行缓冲器，用于串行数据发送/接收。 |
| PCON | 电源控制寄存器 | 87H | Power Control register, controls power-saving modes and baud rate. <br> 电源控制寄存器，控制省电模式和波特率。 |

---

## 2. Program Status Word (PSW) Flags
## 2. 程序状态字标志位

| Flag | 中文名称 | Bit Position | Description |
|------|---------|--------------|-------------|
| CY (C) | 进位标志 | PSW.7 | Carry flag, set when arithmetic operation generates carry/borrow. <br> 进位标志，算术运算产生进位/借位时置位。 |
| AC | 辅助进位标志 | PSW.6 | Auxiliary Carry, set when carry from bit 3 to bit 4 occurs. <br> 辅助进位，从第3位到第4位产生进位时置位。 |
| F0 | 用户标志0 | PSW.5 | User-defined flag bit 0. <br> 用户自定义标志位0。 |
| RS1 | 寄存器组选择位1 | PSW.4 | Register bank select bit 1 (with RS0, selects bank 0-3). <br> 寄存器组选择位1（与RS0配合选择组0-3）。 |
| RS0 | 寄存器组选择位0 | PSW.3 | Register bank select bit 0 (with RS1, selects bank 0-3). <br> 寄存器组选择位0（与RS1配合选择组0-3）。 |
| OV | 溢出标志 | PSW.2 | Overflow flag, set when signed arithmetic overflow occurs. <br> 溢出标志，有符号算术溢出时置位。 |
| P | 奇偶校验标志 | PSW.0 | Parity flag, set if ACC contains odd number of 1s. <br> 奇偶校验标志，累加器含奇数个1时置位。 |

---

## 3. Control Signals
## 3. 控制信号

| Signal | 中文名称 | Description |
|--------|---------|-------------|
| ALE | 地址锁存使能 | Address Latch Enable, latches low address byte from P0 during external memory access. <br> 地址锁存使能，在外部存储器访问时从P0锁存低地址字节。 |
| PSEN | 程序存储器使能 | Program Store Enable, reads from external code memory (MOVC). <br> 程序存储器使能，从外部代码存储器读取（MOVC）。 |
| RD | 读信号 | Read signal, asserted when reading external data memory (MOVX A, @DPTR). <br> 读信号，从外部数据存储器读取时有效（MOVX A, @DPTR）。 |
| WR | 写信号 | Write signal, asserted when writing to external data memory (MOVX @DPTR, A). <br> 写信号，向外部数据存储器写入时有效（MOVX @DPTR, A）。 |
| EA | 外部访问使能 | External Access enable, determines internal/external code memory usage. <br> 外部访问使能，决定使用内部/外部代码存储器。 |

---

## 4. Memory & Addressing
## 4. 存储器与寻址

| Term | 中文名称 | Description |
|------|---------|-------------|
| Internal RAM | 内部RAM | On-chip RAM (128/256 bytes), includes register banks and bit-addressable area. <br> 片内RAM（128/256字节），包括寄存器组和可位寻址区。 |
| External RAM | 外部RAM | Off-chip data memory (up to 64KB), accessed via MOVX instructions. <br> 片外数据存储器（最多64KB），通过MOVX指令访问。 |
| Code Memory | 代码存储器 | ROM/Flash memory storing program code, accessed via MOVC instructions. <br> 存储程序代码的ROM/Flash，通过MOVC指令访问。 |
| SFR | 特殊功能寄存器 | Special Function Registers (80H-FFH), control peripherals and CPU. <br> 特殊功能寄存器（80H-FFH），控制外设和CPU。 |
| Bit-Addressable | 可位寻址 | Memory/registers where individual bits can be directly accessed. <br> 可直接访问单个位的存储器/寄存器。 |
| Register Bank | 寄存器组 | One of four sets of 8 registers (R0-R7), selected by RS1:RS0. <br> 四组8个寄存器（R0-R7）之一，由RS1:RS0选择。 |
| Stack | 堆栈 | LIFO memory structure in internal RAM for return addresses and data. <br> 内部RAM中的后进先出结构，存储返回地址和数据。 |
| Direct Addressing | 直接寻址 | Access memory using 8-bit address (e.g., MOV A, 30H). <br> 使用8位地址访问存储器（如MOV A, 30H）。 |
| Indirect Addressing | 间接寻址 | Access memory through register pointer (e.g., MOV A, @R0). <br> 通过寄存器指针访问存储器（如MOV A, @R0）。 |
| Immediate Addressing | 立即寻址 | Operand is constant in instruction (e.g., MOV A, #55H). <br> 操作数是指令中的常数（如MOV A, #55H）。 |
| Indexed Addressing | 变址寻址 | Access using base + offset (e.g., MOVC A, @A+DPTR). <br> 使用基址+偏移访问（如MOVC A, @A+DPTR）。 |

---

## 5. Interrupt Related
## 5. 中断相关

| Term | 中文名称 | Description |
|------|---------|-------------|
| ISR | 中断服务程序 | Interrupt Service Routine, code executed when interrupt occurs. <br> 中断发生时执行的代码。 |
| EA | 全局中断使能 | Enable All interrupts bit (IE.7), master interrupt enable. <br> 全局中断使能位（IE.7），主中断使能。 |
| ET0 | 定时器0中断使能 | Timer 0 interrupt enable bit (IE.1). <br> 定时器0中断使能位（IE.1）。 |
| ET1 | 定时器1中断使能 | Timer 1 interrupt enable bit (IE.3). <br> 定时器1中断使能位（IE.3）。 |
| EX0 | 外部中断0使能 | External interrupt 0 enable bit (IE.0). <br> 外部中断0使能位（IE.0）。 |
| EX1 | 外部中断1使能 | External interrupt 1 enable bit (IE.2). <br> 外部中断1使能位（IE.2）。 |
| ES | 串口中断使能 | Serial port interrupt enable bit (IE.4). <br> 串口中断使能位（IE.4）。 |
| PT0 | 定时器0优先级 | Timer 0 priority bit (IP.1). <br> 定时器0优先级位（IP.1）。 |
| PT1 | 定时器1优先级 | Timer 1 priority bit (IP.3). <br> 定时器1优先级位（IP.3）。 |
| PX0 | 外部中断0优先级 | External interrupt 0 priority bit (IP.0). <br> 外部中断0优先级位（IP.0）。 |
| PX1 | 外部中断1优先级 | External interrupt 1 priority bit (IP.2). <br> 外部中断1优先级位（IP.2）。 |
| PS | 串口优先级 | Serial port priority bit (IP.4). <br> 串口优先级位（IP.4）。 |
| RETI | 中断返回指令 | Return from interrupt instruction, restores PC and clears interrupt flag. <br> 中断返回指令，恢复PC并清除中断标志。 |

---

## 6. Timer/Counter Related
## 6. 定时器/计数器相关

| Term | 中文名称 | Description |
|------|---------|-------------|
| Timer Mode 0 | 定时器模式0 | 13-bit timer/counter mode. <br> 13位定时器/计数器模式。 |
| Timer Mode 1 | 定时器模式1 | 16-bit timer/counter mode. <br> 16位定时器/计数器模式。 |
| Timer Mode 2 | 定时器模式2 | 8-bit auto-reload timer/counter mode. <br> 8位自动重装定时器/计数器模式。 |
| Timer Mode 3 | 定时器模式3 | Split timer mode (Timer 0 only). <br> 分离定时器模式（仅定时器0）。 |
| TR0 | 定时器0运行控制 | Timer 0 run control bit (TCON.4), starts/stops Timer 0. <br> 定时器0运行控制位（TCON.4），启动/停止定时器0。 |
| TR1 | 定时器1运行控制 | Timer 1 run control bit (TCON.6), starts/stops Timer 1. <br> 定时器1运行控制位（TCON.6），启动/停止定时器1。 |
| TF0 | 定时器0溢出标志 | Timer 0 overflow flag (TCON.5), set when Timer 0 overflows. <br> 定时器0溢出标志（TCON.5），定时器0溢出时置位。 |
| TF1 | 定时器1溢出标志 | Timer 1 overflow flag (TCON.7), set when Timer 1 overflows. <br> 定时器1溢出标志（TCON.7），定时器1溢出时置位。 |
| GATE | 门控位 | Gate control bit in TMOD, enables external control of timer. <br> TMOD中的门控位，使能定时器的外部控制。 |
| C/T | 计数器/定时器选择 | Counter/Timer select bit in TMOD (0=timer, 1=counter). <br> TMOD中的计数器/定时器选择位（0=定时器，1=计数器）。 |
| Machine Cycle | 机器周期 | Basic timing unit, equals 12 oscillator periods in standard 8051. <br> 基本时序单位，标准8051中等于12个振荡器周期。 |

---

## 7. Serial Communication
## 7. 串行通信

| Term | 中文名称 | Description |
|------|---------|-------------|
| Serial Mode 0 | 串行模式0 | Shift register mode, fixed baud rate. <br> 移位寄存器模式，固定波特率。 |
| Serial Mode 1 | 串行模式1 | 8-bit UART mode, variable baud rate. <br> 8位UART模式，可变波特率。 |
| Serial Mode 2 | 串行模式2 | 9-bit UART mode, fixed baud rate. <br> 9位UART模式，固定波特率。 |
| Serial Mode 3 | 串行模式3 | 9-bit UART mode, variable baud rate. <br> 9位UART模式，可变波特率。 |
| TI | 发送中断标志 | Transmit interrupt flag (SCON.1), set when transmission complete. <br> 发送中断标志（SCON.1），发送完成时置位。 |
| RI | 接收中断标志 | Receive interrupt flag (SCON.0), set when reception complete. <br> 接收中断标志（SCON.0），接收完成时置位。 |
| REN | 接收使能 | Receive enable bit (SCON.4), enables serial reception. <br> 接收使能位（SCON.4），使能串行接收。 |
| SM0, SM1 | 串行模式选择 | Serial mode select bits (SCON.7, SCON.6), select serial mode 0-3. <br> 串行模式选择位（SCON.7, SCON.6），选择串行模式0-3。 |
| TB8 | 发送位8 | Transmit bit 8 (SCON.3), 9th data bit in modes 2 and 3. <br> 发送位8（SCON.3），模式2和3中的第9个数据位。 |
| RB8 | 接收位8 | Receive bit 8 (SCON.2), 9th data bit in modes 2 and 3. <br> 接收位8（SCON.2），模式2和3中的第9个数据位。 |
| Baud Rate | 波特率 | Data transmission rate in bits per second (bps). <br> 数据传输速率，单位为每秒位数（bps）。 |
| SMOD | 波特率倍增位 | Baud rate doubler bit (PCON.7), doubles serial baud rate. <br> 波特率倍增位（PCON.7），使串口波特率加倍。 |

---

## 8. Instructions - Data Transfer
## 8. 指令 - 数据传送

| Instruction | 中文名称 | Description |
|-------------|---------|-------------|
| MOV | 传送 | Move data from source to destination. <br> 将数据从源传送到目的地。 |
| MOVC | 代码存储器读取 | Move data from code memory (ROM/Flash) to ACC. <br> 从代码存储器读取数据到累加器。 |
| MOVX | 外部存储器访问 | Move data between ACC and external data memory. <br> 在累加器和外部数据存储器之间传送数据。 |
| PUSH | 入栈 | Push register onto stack. <br> 将寄存器压入堆栈。 |
| POP | 出栈 | Pop top of stack into register. <br> 将栈顶弹出到寄存器。 |
| XCH | 交换 | Exchange ACC with specified operand. <br> 将累加器与指定操作数交换。 |
| XCHD | 半字节交换 | Exchange low nibble of ACC with memory. <br> 交换累加器低半字节与存储器。 |
| SWAP | 半字节互换 | Swap high and low nibbles of ACC. <br> 交换累加器的高低半字节。 |

---

## 9. Instructions - Arithmetic
## 9. 指令 - 算术运算

| Instruction | 中文名称 | Description |
|-------------|---------|-------------|
| ADD | 加法 | Add source operand to ACC. <br> 将源操作数加到累加器。 |
| ADDC | 带进位加法 | Add source and carry flag to ACC. <br> 将源操作数和进位标志加到累加器。 |
| SUBB | 带借位减法 | Subtract source and carry from ACC. <br> 从累加器减去源操作数和进位。 |
| INC | 加1 | Increment operand by 1. <br> 将操作数加1。 |
| DEC | 减1 | Decrement operand by 1. <br> 将操作数减1。 |
| MUL | 乘法 | Multiply ACC by B register (result in B:ACC). <br> 用B寄存器乘累加器（结果在B:ACC）。 |
| DIV | 除法 | Divide ACC by B register (quotient in ACC, remainder in B). <br> 用B寄存器除累加器（商在ACC，余数在B）。 |
| DA | 十进制调整 | Decimal adjust ACC for BCD addition. <br> 为BCD加法调整累加器。 |

---

## 10. Instructions - Logic Operations
## 10. 指令 - 逻辑运算

| Instruction | 中文名称 | Description |
|-------------|---------|-------------|
| ANL | 逻辑与 | Perform bitwise AND operation. <br> 执行按位与运算。 |
| ORL | 逻辑或 | Perform bitwise OR operation. <br> 执行按位或运算。 |
| XRL | 逻辑异或 | Perform bitwise XOR operation. <br> 执行按位异或运算。 |
| CLR | 清零 | Clear ACC or bit to 0. <br> 将累加器或位清零。 |
| CPL | 取反 | Complement (invert) ACC or bit. <br> 对累加器或位取反。 |
| RL | 左循环移位 | Rotate ACC left by one bit. <br> 将累加器左循环移位一位。 |
| RLC | 带进位左循环移位 | Rotate ACC left through carry flag. <br> 通过进位标志将累加器左循环移位。 |
| RR | 右循环移位 | Rotate ACC right by one bit. <br> 将累加器右循环移位一位。 |
| RRC | 带进位右循环移位 | Rotate ACC right through carry flag. <br> 通过进位标志将累加器右循环移位。 |

---

## 11. Instructions - Control Transfer
## 11. 指令 - 控制转移

| Instruction | 中文名称 | Description |
|-------------|---------|-------------|
| LJMP | 长跳转 | Long jump to any address in 64KB space. <br> 跳转到64KB空间的任何地址。 |
| AJMP | 绝对跳转 | Absolute jump within 2KB range. <br> 在2KB范围内绝对跳转。 |
| SJMP | 短跳转 | Short jump within -128 to +127 bytes. <br> 在-128到+127字节范围内跳转。 |
| JMP | 间接跳转 | Indirect jump using @A+DPTR. <br> 使用@A+DPTR间接跳转。 |
| JZ | 为零则跳转 | Jump if ACC is zero. <br> 如果累加器为零则跳转。 |
| JNZ | 不为零则跳转 | Jump if ACC is not zero. <br> 如果累加器不为零则跳转。 |
| JC | 有进位则跳转 | Jump if carry flag is set. <br> 如果进位标志置位则跳转。 |
| JNC | 无进位则跳转 | Jump if carry flag is cleared. <br> 如果进位标志清零则跳转。 |
| JB | 位为1则跳转 | Jump if specified bit is set. <br> 如果指定位为1则跳转。 |
| JNB | 位为0则跳转 | Jump if specified bit is cleared. <br> 如果指定位为0则跳转。 |
| JBC | 位为1则跳转并清零 | Jump if bit is set, then clear the bit. <br> 如果位为1则跳转并清零该位。 |
| CJNE | 比较不等则跳转 | Compare operands and jump if not equal. <br> 比较操作数，不相等则跳转。 |
| DJNZ | 减1不为零则跳转 | Decrement and jump if not zero. <br> 减1后不为零则跳转。 |
| LCALL | 长调用 | Long call to any address in 64KB space. <br> 调用64KB空间的任何地址。 |
| ACALL | 绝对调用 | Absolute call within 2KB range. <br> 在2KB范围内绝对调用。 |
| RET | 返回 | Return from subroutine. <br> 从子程序返回。 |
| RETI | 中断返回 | Return from interrupt. <br> 从中断返回。 |
| NOP | 空操作 | No operation, consume one machine cycle. <br> 空操作，消耗一个机器周期。 |

---

## 12. Instructions - Bit Manipulation
## 12. 指令 - 位操作

| Instruction | 中文名称 | Description |
|-------------|---------|-------------|
| SETB | 置位 | Set specified bit to 1. <br> 将指定位置1。 |
| CLR | 清位 | Clear specified bit to 0. <br> 将指定位清零。 |
| CPL | 位取反 | Complement (invert) specified bit. <br> 对指定位取反。 |
| ANL | 位与 | AND bit with carry or direct bit. <br> 位与进位或直接位进行与运算。 |
| ORL | 位或 | OR bit with carry or direct bit. <br> 位与进位或直接位进行或运算。 |
| MOV | 位传送 | Move bit to/from carry flag. <br> 位传送到/从进位标志。 |

---

## 13. General Concepts
## 13. 通用概念

| Term | 中文名称 | Description |
|------|---------|-------------|
| BCD | 二进制编码十进制 | Binary Coded Decimal, each decimal digit encoded as 4 bits. <br> 二进制编码的十进制，每个十进制数字编码为4位。 |
| Hexadecimal | 十六进制 | Base-16 number system (0-9, A-F), commonly used in 8051 (e.g., FFH). <br> 16进制数系统（0-9, A-F），在8051中常用（如FFH）。 |
| Nibble | 半字节 | 4 bits, half of a byte (0-15 or 0-F). <br> 4位，半个字节（0-15或0-F）。 |
| LSB | 最低有效位 | Least Significant Bit, rightmost bit (2^0). <br> 最低有效位，最右边的位（2^0）。 |
| MSB | 最高有效位 | Most Significant Bit, leftmost bit (2^7 for 8-bit, 2^15 for 16-bit). <br> 最高有效位，最左边的位（8位为2^7，16位为2^15）。 |
| Polling | 轮询 | Repeatedly checking flag/status bit to detect event. <br> 反复检查标志/状态位以检测事件。 |
| Reset | 复位 | Initialize microcontroller to known state, set registers to defaults. <br> 将单片机初始化到已知状态，设置寄存器为默认值。 |
| Volatile | 易失性 | Memory that loses contents when power removed (e.g., RAM). <br> 断电时丢失内容的存储器（如RAM）。 |
| Non-Volatile | 非易失性 | Memory that retains contents when power removed (e.g., ROM, Flash). <br> 断电时保留内容的存储器（如ROM, Flash）。 |
| Oscillator | 振荡器 | External crystal/clock providing timing signals. <br> 提供时序信号的外部晶振/时钟。 |
| Watchdog Timer | 看门狗定时器 | Timer that resets system if not refreshed, recovers from malfunctions. <br> 如不刷新就复位系统的定时器，从故障中恢复。 |
| PC | 程序计数器 | Program Counter, 16-bit register holding next instruction address. <br> 程序计数器，保存下一条指令地址的16位寄存器。 |

---

## 14. Usage Notes
## 14. 使用说明

| Topic | 说明 | Description |
|-------|------|-------------|
| Address Notation | 地址表示法 | Addresses written in hexadecimal with 'H' suffix (e.g., 80H, FFH). <br> 地址用十六进制表示，带'H'后缀（如80H, FFH）。 |
| Bit Addressing | 位寻址 | Access bits using register.bit notation (e.g., PSW.7 for CY flag). <br> 使用寄存器.位表示法访问位（如PSW.7表示CY标志）。 |
| Register Banks | 寄存器组 | Active bank determined by RS1:RS0 (00=Bank0, 01=Bank1, 10=Bank2, 11=Bank3). <br> 活动寄存器组由RS1:RS0确定（00=组0, 01=组1, 10=组2, 11=组3）。 |
| Immediate Values | 立即数 | Immediate data prefixed with '#' (e.g., MOV A, #55H). <br> 立即数前缀为'#'（如MOV A, #55H）。 |
| Indirect Addressing | 间接寻址符号 | '@' indicates indirect addressing through register (e.g., MOV A, @R0). <br> '@'表示通过寄存器间接寻址（如MOV A, @R0）。 |

---

## 15. Cross-References
## 15. 交叉引用

For detailed information about specific topics, refer to:

有关特定主题的详细信息，请参阅：

| Topic | 主题 | Document |
|-------|------|----------|
| Data Pointer | 数据指针 | `DPTR_Data_Pointer.md` |
| Interrupts | 中断系统 | `Interrupt_IE_IP.md` |
| Power Control | 电源控制 | `PCON_Power_Control.md` |
| PSW, ACC, B | 程序状态字与寄存器 | `PSW_ACC_B_Registers.md` |
| Serial Port | 串行端口 | `Serial_Port_SCON_SBUF.md` |
| Stack Pointer | 堆栈指针 | `SP_Stack_Pointer.md` |
| Timers | 定时器/计数器 | `Timer_Registers.md` |
| Overview | 硬件架构概述 | `README.md` |

---

**Document Version**: 2.0 (Reorganized by Category)
**Last Updated**: 2026-01-28
**Reference**: 8051 Microcontroller Hardware Architecture Documentation
