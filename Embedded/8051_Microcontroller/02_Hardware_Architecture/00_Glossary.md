# 8051 Hardware Architecture Glossary
# 8051 硬件架构词汇表

This glossary provides definitions and explanations of key terms, abbreviations, and concepts used in 8051 microcontroller hardware architecture documentation.

本词汇表提供8051单片机硬件架构文档中使用的关键术语、缩写和概念的定义与解释。

---

## A

### ACC (Accumulator)
**中文**: 累加器
**定义**: The primary 8-bit register used for arithmetic and logical operations. Located at address E0H, it is bit-addressable.
**定义**: 用于算术和逻辑运算的主要8位寄存器。位于地址E0H，可位寻址。

### ALE (Address Latch Enable)
**中文**: 地址锁存使能
**定义**: Control signal used to latch the lower 8 bits of address from Port 0 during external memory access.
**定义**: 在外部存储器访问期间，用于从端口0锁存地址低8位的控制信号。

### Addressing Mode
**中文**: 寻址方式
**定义**: The method by which an instruction accesses operands. 8051 supports register, direct, indirect, immediate, and indexed addressing modes.
**定义**: 指令访问操作数的方法。8051支持寄存器、直接、间接、立即数和变址寻址方式。

---

## B

### B Register
**中文**: B寄存器
**定义**: 8-bit register used primarily for multiplication and division operations. Located at address F0H, it is bit-addressable.
**定义**: 主要用于乘法和除法运算的8位寄存器。位于地址F0H，可位寻址。

### Bit-Addressable
**中文**: 可位寻址
**定义**: Memory locations or registers where individual bits can be directly accessed and manipulated using bit instructions.
**定义**: 可以使用位指令直接访问和操作单个位的内存位置或寄存器。

### Baud Rate
**中文**: 波特率
**定义**: The rate at which data is transmitted in serial communication, measured in bits per second (bps).
**定义**: 串行通信中数据传输的速率，以每秒位数(bps)为单位测量。

---

## C

### Carry Flag (C)
**中文**: 进位标志
**定义**: Bit 7 of PSW register. Set when an arithmetic operation generates a carry or borrow.
**定义**: PSW寄存器的第7位。当算术运算产生进位或借位时置位。

### CY (Carry)
**中文**: 进位位
**定义**: Same as Carry Flag. Can be used as a 1-bit accumulator for boolean operations.
**定义**: 与进位标志相同。可用作布尔运算的1位累加器。

---

## D

### DPTR (Data Pointer)
**中文**: 数据指针
**定义**: 16-bit register composed of DPH (high byte) and DPL (low byte), used for accessing external memory and code memory.
**定义**: 由DPH(高字节)和DPL(低字节)组成的16位寄存器，用于访问外部存储器和代码存储器。

### DPH (Data Pointer High)
**中文**: 数据指针高字节
**定义**: High byte of DPTR register, located at address 83H.
**定义**: DPTR寄存器的高字节，位于地址83H。

### DPL (Data Pointer Low)
**中文**: 数据指针低字节
**定义**: Low byte of DPTR register, located at address 82H.
**定义**: DPTR寄存器的低字节，位于地址82H。

---

## E

### EA (Enable All Interrupts)
**中文**: 全局中断使能
**定义**: Bit 7 of IE register. When set, enables all interrupts; when cleared, disables all interrupts.
**定义**: IE寄存器的第7位。置位时使能所有中断；清零时禁止所有中断。

### External Memory
**中文**: 外部存储器
**定义**: Memory located outside the 8051 chip, accessed via MOVX instructions using Port 0 and Port 2.
**定义**: 位于8051芯片外部的存储器，通过MOVX指令使用端口0和端口2访问。

---

## F

### Flag
**中文**: 标志位
**定义**: A bit in the PSW register that indicates the status of the last operation (e.g., Carry, Auxiliary Carry, Overflow, Parity).
**定义**: PSW寄存器中的位，指示上次操作的状态(如进位、辅助进位、溢出、奇偶校验)。

---

## G

### GF0, GF1 (General Purpose Flags)
**中文**: 通用标志位
**定义**: Bits 3 and 2 of PCON register, available for user-defined purposes.
**定义**: PCON寄存器的第3位和第2位，可用于用户自定义目的。

---

## I

### IE (Interrupt Enable)
**中文**: 中断使能寄存器
**定义**: Special Function Register at address A8H that controls the enabling/disabling of individual interrupts.
**定义**: 位于地址A8H的特殊功能寄存器，控制各个中断的使能/禁止。

### IP (Interrupt Priority)
**中文**: 中断优先级寄存器
**定义**: Special Function Register at address B8H that sets the priority level of each interrupt source.
**定义**: 位于地址B8H的特殊功能寄存器，设置每个中断源的优先级。

### ISR (Interrupt Service Routine)
**中文**: 中断服务程序
**定义**: A subroutine that executes when an interrupt occurs, handling the interrupt event.
**定义**: 当中断发生时执行的子程序，处理中断事件。

### Internal Memory
**中文**: 内部存储器
**定义**: Memory located within the 8051 chip, including RAM (128/256 bytes) and SFR (Special Function Registers).
**定义**: 位于8051芯片内部的存储器，包括RAM(128/256字节)和SFR(特殊功能寄存器)。

---

## L

### LSB (Least Significant Bit)
**中文**: 最低有效位
**定义**: The rightmost bit in a binary number, representing the smallest value (2^0).
**定义**: 二进制数中最右边的位，表示最小值(2^0)。

---

## M

### Machine Cycle
**中文**: 机器周期
**定义**: The basic unit of time for instruction execution, equal to 12 oscillator periods in standard 8051.
**定义**: 指令执行的基本时间单位，在标准8051中等于12个振荡器周期。

### MOVC
**中文**: 代码存储器读取指令
**定义**: Instruction to read data from code memory (ROM/Flash). Format: MOVC A, @A+DPTR or MOVC A, @A+PC.
**定义**: 从代码存储器(ROM/Flash)读取数据的指令。格式：MOVC A, @A+DPTR 或 MOVC A, @A+PC。

### MOVX
**中文**: 外部存储器访问指令
**定义**: Instruction to read from or write to external data memory. Format: MOVX A, @DPTR or MOVX @DPTR, A.
**定义**: 从外部数据存储器读取或写入的指令。格式：MOVX A, @DPTR 或 MOVX @DPTR, A。

### MSB (Most Significant Bit)
**中文**: 最高有效位
**定义**: The leftmost bit in a binary number, representing the largest value (2^7 for 8-bit, 2^15 for 16-bit).
**定义**: 二进制数中最左边的位，表示最大值(8位为2^7，16位为2^15)。

---

## O

### OV (Overflow Flag)
**中文**: 溢出标志
**定义**: Bit 2 of PSW register. Set when a signed arithmetic operation produces a result outside the valid range.
**定义**: PSW寄存器的第2位。当有符号算术运算产生超出有效范围的结果时置位。

### Oscillator
**中文**: 振荡器
**定义**: External crystal or clock source that provides timing signals for the microcontroller.
**定义**: 为单片机提供时序信号的外部晶振或时钟源。

---

## P

### P (Parity Flag)
**中文**: 奇偶校验标志
**定义**: Bit 0 of PSW register. Set if the accumulator contains an odd number of 1s, cleared if even.
**定义**: PSW寄存器的第0位。如果累加器包含奇数个1则置位，偶数个则清零。

### PC (Program Counter)
**中文**: 程序计数器
**定义**: 16-bit register that holds the address of the next instruction to be executed.
**定义**: 保存下一条要执行指令地址的16位寄存器。

### PCON (Power Control Register)
**中文**: 电源控制寄存器
**定义**: Special Function Register at address 87H that controls power-saving modes and serial baud rate doubling.
**定义**: 位于地址87H的特殊功能寄存器，控制省电模式和串口波特率倍增。

### PSW (Program Status Word)
**中文**: 程序状态字
**定义**: Special Function Register at address D0H containing status flags (CY, AC, F0, RS1, RS0, OV, P).
**定义**: 位于地址D0H的特殊功能寄存器，包含状态标志(CY, AC, F0, RS1, RS0, OV, P)。

### PSEN (Program Store Enable)
**中文**: 程序存储器使能
**定义**: Control signal used to read data from external code memory during MOVC instructions.
**定义**: 在MOVC指令期间用于从外部代码存储器读取数据的控制信号。

---

## R

### RD (Read)
**中文**: 读信号
**定义**: Control signal asserted when reading from external data memory during MOVX A, @DPTR instruction.
**定义**: 在MOVX A, @DPTR指令期间从外部数据存储器读取时有效的控制信号。

### Register Bank
**中文**: 寄存器组
**定义**: One of four sets of 8 general-purpose registers (R0-R7) in internal RAM, selected by RS1 and RS0 bits in PSW.
**定义**: 内部RAM中四组8个通用寄存器(R0-R7)之一，由PSW中的RS1和RS0位选择。

### RETI
**中文**: 中断返回指令
**定义**: Instruction used to return from an interrupt service routine, restoring PC and clearing interrupt-in-progress flag.
**定义**: 用于从中断服务程序返回的指令，恢复PC并清除中断进行标志。

### RS0, RS1 (Register Bank Select)
**中文**: 寄存器组选择位
**定义**: Bits 3 and 4 of PSW register that select which of the four register banks (0-3) is currently active.
**定义**: PSW寄存器的第3位和第4位，选择四个寄存器组(0-3)中哪一个当前有效。

---

## S

### SBUF (Serial Buffer)
**中文**: 串行缓冲器
**定义**: Special Function Register at address 99H used for serial data transmission and reception. Physically two separate registers.
**定义**: 位于地址99H的特殊功能寄存器，用于串行数据发送和接收。物理上是两个独立的寄存器。

### SCON (Serial Control)
**中文**: 串行控制寄存器
**定义**: Special Function Register at address 98H that controls serial port operation mode, enables, and status flags.
**定义**: 位于地址98H的特殊功能寄存器，控制串口操作模式、使能和状态标志。

### SFR (Special Function Register)
**中文**: 特殊功能寄存器
**定义**: Registers in the address range 80H-FFH that control and monitor peripheral functions and CPU operations.
**定义**: 地址范围80H-FFH中的寄存器，控制和监视外设功能和CPU操作。

### SP (Stack Pointer)
**中文**: 堆栈指针
**定义**: 8-bit Special Function Register at address 81H that points to the top of the stack in internal RAM. Default value is 07H.
**定义**: 位于地址81H的8位特殊功能寄存器，指向内部RAM中堆栈的顶部。默认值为07H。

### Stack
**中文**: 堆栈
**定义**: Last-In-First-Out (LIFO) memory structure in internal RAM used for storing return addresses and temporary data.
**定义**: 内部RAM中的后进先出(LIFO)内存结构，用于存储返回地址和临时数据。

---

## T

### TCON (Timer Control)
**中文**: 定时器控制寄存器
**定义**: Special Function Register at address 88H that controls timer/counter operation and external interrupt flags.
**定义**: 位于地址88H的特殊功能寄存器，控制定时器/计数器操作和外部中断标志。

### TH0, TH1 (Timer High Byte)
**中文**: 定时器高字节
**定义**: High byte registers for Timer 0 (8CH) and Timer 1 (8DH), storing the upper 8 bits of the timer count.
**定义**: 定时器0(8CH)和定时器1(8DH)的高字节寄存器，存储定时器计数的高8位。

### TL0, TL1 (Timer Low Byte)
**中文**: 定时器低字节
**定义**: Low byte registers for Timer 0 (8AH) and Timer 1 (8BH), storing the lower 8 bits of the timer count.
**定义**: 定时器0(8AH)和定时器1(8BH)的低字节寄存器，存储定时器计数的低8位。

### TMOD (Timer Mode)
**中文**: 定时器模式寄存器
**定义**: Special Function Register at address 89H that sets the operating mode for Timer 0 and Timer 1.
**定义**: 位于地址89H的特殊功能寄存器，设置定时器0和定时器1的操作模式。

### Timer/Counter
**中文**: 定时器/计数器
**定义**: Hardware module that can count machine cycles (timer mode) or external events (counter mode).
**定义**: 可以计数机器周期(定时器模式)或外部事件(计数器模式)的硬件模块。

---

## W

### WR (Write)
**中文**: 写信号
**定义**: Control signal asserted when writing to external data memory during MOVX @DPTR, A instruction.
**定义**: 在MOVX @DPTR, A指令期间向外部数据存储器写入时有效的控制信号。

### Watchdog Timer
**中文**: 看门狗定时器
**定义**: Timer that resets the system if not periodically refreshed, used to recover from software malfunctions.
**定义**: 如果不定期刷新就会复位系统的定时器，用于从软件故障中恢复。

---

## Common Instruction Mnemonics
## 常用指令助记符

### ACALL (Absolute Call)
**中文**: 绝对调用
**定义**: Call subroutine within 2KB address range from current PC.
**定义**: 在当前PC的2KB地址范围内调用子程序。

### ADD (Add)
**中文**: 加法
**定义**: Add source operand to accumulator.
**定义**: 将源操作数加到累加器。

### AJMP (Absolute Jump)
**中文**: 绝对跳转
**定义**: Jump to address within 2KB range from current PC.
**定义**: 跳转到当前PC的2KB范围内的地址。

### ANL (AND Logic)
**中文**: 逻辑与
**定义**: Perform bitwise AND operation.
**定义**: 执行按位与运算。

### CJNE (Compare and Jump if Not Equal)
**中文**: 比较不等则跳转
**定义**: Compare two operands and jump if they are not equal.
**定义**: 比较两个操作数，如果不相等则跳转。

### CLR (Clear)
**中文**: 清零
**定义**: Clear accumulator or bit to 0.
**定义**: 将累加器或位清零。

### CPL (Complement)
**中文**: 取反
**定义**: Complement (invert) accumulator or bit.
**定义**: 对累加器或位取反。

### DA (Decimal Adjust)
**中文**: 十进制调整
**定义**: Adjust accumulator for BCD addition.
**定义**: 为BCD加法调整累加器。

### DEC (Decrement)
**中文**: 减1
**定义**: Decrement operand by 1.
**定义**: 将操作数减1。

### DIV (Divide)
**中文**: 除法
**定义**: Divide accumulator by B register.
**定义**: 用B寄存器除累加器。

### DJNZ (Decrement and Jump if Not Zero)
**中文**: 减1不为零则跳转
**定义**: Decrement operand and jump if result is not zero.
**定义**: 将操作数减1，如果结果不为零则跳转。

### INC (Increment)
**中文**: 加1
**定义**: Increment operand by 1.
**定义**: 将操作数加1。

### JB (Jump if Bit set)
**中文**: 位为1则跳转
**定义**: Jump if specified bit is set to 1.
**定义**: 如果指定位为1则跳转。

### JC (Jump if Carry)
**中文**: 有进位则跳转
**定义**: Jump if carry flag is set.
**定义**: 如果进位标志置位则跳转。

### JMP (Jump)
**中文**: 跳转
**定义**: Unconditional jump to specified address.
**定义**: 无条件跳转到指定地址。

### JNB (Jump if Bit Not set)
**中文**: 位为0则跳转
**定义**: Jump if specified bit is cleared to 0.
**定义**: 如果指定位为0则跳转。

### JNC (Jump if No Carry)
**中文**: 无进位则跳转
**定义**: Jump if carry flag is cleared.
**定义**: 如果进位标志清零则跳转。

### JNZ (Jump if Not Zero)
**中文**: 不为零则跳转
**定义**: Jump if accumulator is not zero.
**定义**: 如果累加器不为零则跳转。

### JZ (Jump if Zero)
**中文**: 为零则跳转
**定义**: Jump if accumulator is zero.
**定义**: 如果累加器为零则跳转。

### LCALL (Long Call)
**中文**: 长调用
**定义**: Call subroutine at any address in 64KB memory space.
**定义**: 在64KB内存空间的任何地址调用子程序。

### LJMP (Long Jump)
**中文**: 长跳转
**定义**: Jump to any address in 64KB memory space.
**定义**: 跳转到64KB内存空间的任何地址。

### MOV (Move)
**中文**: 传送
**定义**: Copy data from source to destination.
**定义**: 将数据从源复制到目的地。

### MUL (Multiply)
**中文**: 乘法
**定义**: Multiply accumulator by B register.
**定义**: 用B寄存器乘累加器。

### NOP (No Operation)
**中文**: 空操作
**定义**: Do nothing, consume one machine cycle.
**定义**: 什么都不做，消耗一个机器周期。

### ORL (OR Logic)
**中文**: 逻辑或
**定义**: Perform bitwise OR operation.
**定义**: 执行按位或运算。

### POP
**中文**: 出栈
**定义**: Pop top of stack into specified register.
**定义**: 将栈顶弹出到指定寄存器。

### PUSH
**中文**: 入栈
**定义**: Push specified register onto stack.
**定义**: 将指定寄存器压入栈。

### RET (Return)
**中文**: 返回
**定义**: Return from subroutine.
**定义**: 从子程序返回。

### RL (Rotate Left)
**中文**: 左循环移位
**定义**: Rotate accumulator left by one bit.
**定义**: 将累加器左循环移位一位。

### RLC (Rotate Left through Carry)
**中文**: 带进位左循环移位
**定义**: Rotate accumulator left through carry flag.
**定义**: 通过进位标志将累加器左循环移位。

### RR (Rotate Right)
**中文**: 右循环移位
**定义**: Rotate accumulator right by one bit.
**定义**: 将累加器右循环移位一位。

### RRC (Rotate Right through Carry)
**中文**: 带进位右循环移位
**定义**: Rotate accumulator right through carry flag.
**定义**: 通过进位标志将累加器右循环移位。

### SETB (Set Bit)
**中文**: 置位
**定义**: Set specified bit to 1.
**定义**: 将指定位置1。

### SJMP (Short Jump)
**中文**: 短跳转
**定义**: Jump to address within -128 to +127 bytes from current PC.
**定义**: 跳转到当前PC的-128到+127字节范围内的地址。

### SUBB (Subtract with Borrow)
**中文**: 带借位减法
**定义**: Subtract source and carry from accumulator.
**定义**: 从累加器中减去源操作数和进位。

### SWAP
**中文**: 半字节交换
**定义**: Swap high and low nibbles of accumulator.
**定义**: 交换累加器的高低半字节。

### XCH (Exchange)
**中文**: 交换
**定义**: Exchange accumulator with specified operand.
**定义**: 将累加器与指定操作数交换。

### XRL (Exclusive OR Logic)
**中文**: 逻辑异或
**定义**: Perform bitwise XOR operation.
**定义**: 执行按位异或运算。

---

## Additional Terms
## 附加术语

### BCD (Binary Coded Decimal)
**中文**: 二进制编码的十进制
**定义**: Number representation where each decimal digit is encoded as a 4-bit binary number.
**定义**: 每个十进制数字编码为4位二进制数的数字表示法。

### Hexadecimal
**中文**: 十六进制
**定义**: Base-16 number system using digits 0-9 and letters A-F. Commonly used in 8051 programming (e.g., FFH, A0H).
**定义**: 使用数字0-9和字母A-F的16进制数系统。在8051编程中常用(如FFH, A0H)。

### Nibble
**中文**: 半字节
**定义**: 4 bits, or half of a byte. Can represent values 0-15 (0-F in hexadecimal).
**定义**: 4位，或半个字节。可以表示值0-15(十六进制0-F)。

### Polling
**中文**: 轮询
**定义**: Technique of repeatedly checking a flag or status bit to detect an event.
**定义**: 反复检查标志或状态位以检测事件的技术。

### Reset
**中文**: 复位
**定义**: Initialization of the microcontroller to a known state, setting all registers to default values.
**定义**: 将单片机初始化到已知状态，将所有寄存器设置为默认值。

### Volatile
**中文**: 易失性
**定义**: Memory that loses its contents when power is removed (e.g., RAM).
**定义**: 断电时丢失内容的存储器(如RAM)。

### Non-Volatile
**中文**: 非易失性
**定义**: Memory that retains its contents when power is removed (e.g., ROM, Flash).
**定义**: 断电时保留内容的存储器(如ROM, Flash)。

---

## Usage Notes
## 使用说明

1. **Address Notation**: Addresses are typically written in hexadecimal with 'H' suffix (e.g., 80H, FFH).
   **地址表示法**: 地址通常用十六进制表示，带'H'后缀(如80H, FFH)。

2. **Bit Addressing**: Some registers can be accessed bit-by-bit using bit addresses (e.g., PSW.7 for CY flag).
   **位寻址**: 某些寄存器可以使用位地址逐位访问(如PSW.7表示CY标志)。

3. **Register Banks**: The active register bank is determined by RS1:RS0 bits in PSW (00=Bank0, 01=Bank1, 10=Bank2, 11=Bank3).
   **寄存器组**: 活动寄存器组由PSW中的RS1:RS0位确定(00=组0, 01=组1, 10=组2, 11=组3)。

4. **Immediate Values**: Immediate data is prefixed with '#' symbol (e.g., MOV A, #55H).
   **立即数**: 立即数前缀为'#'符号(如MOV A, #55H)。

5. **Indirect Addressing**: '@' symbol indicates indirect addressing through a register (e.g., MOV A, @R0).
   **间接寻址**: '@'符号表示通过寄存器间接寻址(如MOV A, @R0)。

---

## Cross-References
## 交叉引用

For detailed information about specific topics, refer to:
有关特定主题的详细信息，请参阅：

- **DPTR Register**: See `DPTR_Data_Pointer.md`
- **Interrupts**: See `Interrupt_IE_IP.md`
- **Power Control**: See `PCON_Power_Control.md`
- **PSW, ACC, B Registers**: See `PSW_ACC_B_Registers.md`
- **Serial Port**: See `Serial_Port_SCON_SBUF.md`
- **Stack Pointer**: See `SP_Stack_Pointer.md`
- **Timers**: See `Timer_Registers.md`
- **Overview**: See `README.md`

---

**Document Version**: 1.0
**Last Updated**: 2026-01-28
**Reference**: 8051 Microcontroller Hardware Architecture Documentation
