# STM32 Peripherals Glossary
# STM32 外设词汇表

This glossary provides categorized definitions of key terms used in STM32 peripheral documentation.

本词汇表按分类提供STM32外设文档中使用的关键术语定义。

---

## 1. GPIO - General Purpose Input/Output
## 1. GPIO - 通用输入输出

| Term | 中文名称 | Description |
|------|---------|-------------|
| GPIO | 通用输入输出 | General Purpose Input/Output pins for digital I/O operations. <br> 用于数字输入输出操作的通用引脚。 |
| Pin | 引脚 | Individual I/O connection point on the microcontroller. <br> 微控制器上的单个I/O连接点。 |
| Port | 端口 | Group of 16 GPIO pins (e.g., GPIOA, GPIOB). <br> 16个GPIO引脚的组（如GPIOA、GPIOB）。 |
| Push-Pull | 推挽输出 | Output mode that can actively drive both HIGH and LOW. <br> 可主动驱动高电平和低电平的输出模式。 |
| Open-Drain | 开漏输出 | Output mode that can only actively drive LOW, HIGH is floating. <br> 只能主动驱动低电平的输出模式，高电平为浮空。 |
| Pull-up | 上拉 | Internal resistor that pulls pin to HIGH when not driven. <br> 当引脚未被驱动时将其拉至高电平的内部电阻。 |
| Pull-down | 下拉 | Internal resistor that pulls pin to LOW when not driven. <br> 当引脚未被驱动时将其拉至低电平的内部电阻。 |
| Floating | 浮空 | Input mode with no pull-up or pull-down resistor. <br> 无上拉或下拉电阻的输入模式。 |
| Alternate Function (AF) | 复用功能 | GPIO pin used by peripheral (UART, SPI, etc.). <br> GPIO引脚被外设使用（UART、SPI等）。 |
| Analog Mode | 模拟模式 | GPIO pin connected to ADC/DAC, digital I/O disabled. <br> GPIO引脚连接到ADC/DAC，数字I/O禁用。 |

---

## 2. Clock and Reset
## 2. 时钟与复位

| Term | 中文名称 | Description |
|------|---------|-------------|
| RCC | 复位和时钟控制 | Reset and Clock Control peripheral. <br> 复位和时钟控制外设。 |
| HSE | 高速外部时钟 | High-Speed External oscillator (4-26 MHz crystal). <br> 高速外部振荡器（4-26 MHz晶振）。 |
| HSI | 高速内部时钟 | High-Speed Internal RC oscillator (16 MHz). <br> 高速内部RC振荡器（16 MHz）。 |
| LSE | 低速外部时钟 | Low-Speed External oscillator (32.768 kHz crystal for RTC). <br> 低速外部振荡器（32.768 kHz晶振用于RTC）。 |
| LSI | 低速内部时钟 | Low-Speed Internal RC oscillator (~32 kHz). <br> 低速内部RC振荡器（约32 kHz）。 |
| PLL | 锁相环 | Phase-Locked Loop for frequency multiplication. <br> 用于倍频的锁相环。 |
| SYSCLK | 系统时钟 | System clock frequency (main CPU clock). <br> 系统时钟频率（主CPU时钟）。 |
| AHB | 高级高性能总线 | Advanced High-performance Bus. <br> 高级高性能总线。 |
| APB1 | 外设总线1 | Advanced Peripheral Bus 1 (low-speed peripherals). <br> 高级外设总线1（低速外设）。 |
| APB2 | 外设总线2 | Advanced Peripheral Bus 2 (high-speed peripherals). <br> 高级外设总线2（高速外设）。 |
| Prescaler | 预分频器 | Divider that reduces clock frequency. <br> 降低时钟频率的分频器。 |

---

## 3. UART/USART - Serial Communication
## 3. UART/USART - 串行通信

| Term | 中文名称 | Description |
|------|---------|-------------|
| UART | 通用异步收发器 | Universal Asynchronous Receiver-Transmitter. <br> 通用异步收发器。 |
| USART | 通用同步异步收发器 | Universal Synchronous/Asynchronous Receiver-Transmitter. <br> 通用同步/异步收发器。 |
| Baud Rate | 波特率 | Data transmission rate in bits per second (bps). <br> 数据传输速率，单位为每秒位数（bps）。 |
| TX | 发送 | Transmit data pin. <br> 发送数据引脚。 |
| RX | 接收 | Receive data pin. <br> 接收数据引脚。 |
| Start Bit | 起始位 | Bit that signals the beginning of data transmission. <br> 标志数据传输开始的位。 |
| Stop Bit | 停止位 | Bit(s) that signal the end of data transmission. <br> 标志数据传输结束的位。 |
| Parity | 奇偶校验 | Error detection bit (even, odd, or none). <br> 错误检测位（偶校验、奇校验或无）。 |
| Oversampling | 过采样 | Sampling data multiple times per bit period. <br> 每个位周期多次采样数据。 |
| RXNE | 接收非空 | Receive data register Not Empty flag. <br> 接收数据寄存器非空标志。 |
| TXE | 发送空 | Transmit data register Empty flag. <br> 发送数据寄存器空标志。 |
| TC | 传输完成 | Transmission Complete flag. <br> 传输完成标志。 |

---

## 4. SPI - Serial Peripheral Interface
## 4. SPI - 串行外设接口

| Term | 中文名称 | Description |
|------|---------|-------------|
| SPI | 串行外设接口 | Serial Peripheral Interface, synchronous serial protocol. <br> 串行外设接口，同步串行协议。 |
| MOSI | 主出从入 | Master Out Slave In data line. <br> 主设备输出从设备输入数据线。 |
| MISO | 主入从出 | Master In Slave Out data line. <br> 主设备输入从设备输出数据线。 |
| SCK | 串行时钟 | Serial Clock signal. <br> 串行时钟信号。 |
| NSS (CS) | 片选 | Slave Select / Chip Select signal. <br> 从设备选择/片选信号。 |
| CPOL | 时钟极性 | Clock Polarity (idle state of clock). <br> 时钟极性（时钟空闲状态）。 |
| CPHA | 时钟相位 | Clock Phase (data sampling edge). <br> 时钟相位（数据采样边沿）。 |
| Full-Duplex | 全双工 | Simultaneous bidirectional data transfer. <br> 同时双向数据传输。 |
| Half-Duplex | 半双工 | Bidirectional data transfer, one direction at a time. <br> 双向数据传输，一次一个方向。 |
| Simplex | 单工 | Unidirectional data transfer only. <br> 仅单向数据传输。 |

---

## 5. I2C - Inter-Integrated Circuit
## 5. I2C - I2C总线

| Term | 中文名称 | Description |
|------|---------|-------------|
| I2C | I2C总线 | Inter-Integrated Circuit, two-wire serial protocol. <br> 集成电路间总线，双线串行协议。 |
| SDA | 串行数据 | Serial Data line (bidirectional). <br> 串行数据线（双向）。 |
| SCL | 串行时钟 | Serial Clock line. <br> 串行时钟线。 |
| Master | 主设备 | Device that initiates communication and generates clock. <br> 发起通信并产生时钟的设备。 |
| Slave | 从设备 | Device that responds to master requests. <br> 响应主设备请求的设备。 |
| 7-bit Address | 7位地址 | Standard I2C device addressing (0-127). <br> 标准I2C设备寻址（0-127）。 |
| 10-bit Address | 10位地址 | Extended I2C device addressing. <br> 扩展I2C设备寻址。 |
| ACK | 应答 | Acknowledge bit sent by receiver. <br> 接收方发送的应答位。 |
| NACK | 非应答 | Not Acknowledge, signals end of transfer or error. <br> 非应答，表示传输结束或错误。 |
| START Condition | 起始条件 | SDA falls while SCL is HIGH. <br> SCL为高时SDA下降。 |
| STOP Condition | 停止条件 | SDA rises while SCL is HIGH. <br> SCL为高时SDA上升。 |
| Clock Stretching | 时钟延展 | Slave holds SCL LOW to slow down master. <br> 从设备保持SCL为低以减慢主设备速度。 |

---

## 6. Timer - Timers and PWM
## 6. Timer - 定时器与PWM

| Term | 中文名称 | Description |
|------|---------|-------------|
| TIM | 定时器 | Timer peripheral for timing and PWM generation. <br> 用于定时和PWM生成的定时器外设。 |
| Basic Timer | 基本定时器 | Simple up-counter timer (TIM6, TIM7). <br> 简单的向上计数定时器（TIM6、TIM7）。 |
| General-Purpose Timer | 通用定时器 | Timer with input capture, output compare, PWM (TIM2-TIM5). <br> 具有输入捕获、输出比较、PWM功能的定时器（TIM2-TIM5）。 |
| Advanced Timer | 高级定时器 | Timer with complementary outputs, dead-time (TIM1, TIM8). <br> 具有互补输出、死区时间的定时器（TIM1、TIM8）。 |
| Counter | 计数器 | Register that increments/decrements with clock. <br> 随时钟递增/递减的寄存器。 |
| Prescaler | 预分频器 | Divides timer clock frequency. <br> 分频定时器时钟频率。 |
| Auto-Reload Register (ARR) | 自动重装载寄存器 | Defines timer period/overflow value. <br> 定义定时器周期/溢出值。 |
| PWM | 脉宽调制 | Pulse Width Modulation for analog-like output. <br> 用于类模拟输出的脉宽调制。 |
| Duty Cycle | 占空比 | Percentage of time signal is HIGH in PWM. <br> PWM信号为高电平的时间百分比。 |
| Input Capture | 输入捕获 | Captures timer value on external event. <br> 在外部事件时捕获定时器值。 |
| Output Compare | 输出比较 | Generates output when counter matches compare value. <br> 当计数器匹配比较值时产生输出。 |
| Update Event | 更新事件 | Timer overflow/underflow event. <br> 定时器溢出/下溢事件。 |
| Encoder Mode | 编码器模式 | Reads quadrature encoder signals. <br> 读取正交编码器信号。 |

---

## 7. ADC/DAC - Analog Conversion
## 7. ADC/DAC - 模拟转换

| Term | 中文名称 | Description |
|------|---------|-------------|
| ADC | 模数转换器 | Analog-to-Digital Converter. <br> 模数转换器。 |
| DAC | 数模转换器 | Digital-to-Analog Converter. <br> 数模转换器。 |
| Resolution | 分辨率 | Number of bits in conversion (8-bit, 10-bit, 12-bit). <br> 转换的位数（8位、10位、12位）。 |
| Sampling Rate | 采样率 | Number of conversions per second. <br> 每秒转换次数。 |
| Conversion Time | 转换时间 | Time required to complete one conversion. <br> 完成一次转换所需的时间。 |
| Single Conversion | 单次转换 | One-time ADC conversion. <br> 一次性ADC转换。 |
| Continuous Conversion | 连续转换 | Automatic repeated conversions. <br> 自动重复转换。 |
| Scan Mode | 扫描模式 | Convert multiple channels sequentially. <br> 顺序转换多个通道。 |
| Injected Channel | 注入通道 | High-priority ADC channel that can interrupt regular conversion. <br> 可中断常规转换的高优先级ADC通道。 |
| Regular Channel | 常规通道 | Standard ADC conversion channel. <br> 标准ADC转换通道。 |
| VREF | 参考电压 | Reference voltage for ADC/DAC (typically 3.3V). <br> ADC/DAC的参考电压（通常为3.3V）。 |
| EOC | 转换结束 | End Of Conversion flag. <br> 转换结束标志。 |

---

## 8. DMA - Direct Memory Access
## 8. DMA - 直接内存访问

| Term | 中文名称 | Description |
|------|---------|-------------|
| DMA | 直接内存访问 | Direct Memory Access for data transfer without CPU. <br> 无需CPU参与的直接内存访问数据传输。 |
| Channel | 通道 | Independent DMA data path (DMA1/DMA2 have multiple channels). <br> 独立的DMA数据路径（DMA1/DMA2有多个通道）。 |
| Stream | 数据流 | DMA data path in advanced DMA controllers (STM32F4/F7). <br> 高级DMA控制器中的数据路径（STM32F4/F7）。 |
| Memory-to-Memory | 内存到内存 | DMA transfer between two memory locations. <br> 两个内存位置之间的DMA传输。 |
| Peripheral-to-Memory | 外设到内存 | DMA transfer from peripheral to memory. <br> 从外设到内存的DMA传输。 |
| Memory-to-Peripheral | 内存到外设 | DMA transfer from memory to peripheral. <br> 从内存到外设的DMA传输。 |
| Circular Mode | 循环模式 | DMA automatically restarts after completion. <br> DMA完成后自动重启。 |
| Normal Mode | 正常模式 | DMA stops after one transfer. <br> 一次传输后DMA停止。 |
| Priority | 优先级 | DMA channel priority (Low, Medium, High, Very High). <br> DMA通道优先级（低、中、高、超高）。 |
| Transfer Complete | 传输完成 | DMA transfer finished interrupt. <br> DMA传输完成中断。 |
| Half Transfer | 半传输 | DMA half-way completion interrupt. <br> DMA传输过半中断。 |

---

## 9. RTC - Real-Time Clock
## 9. RTC - 实时时钟

| Term | 中文名称 | Description |
|------|---------|-------------|
| RTC | 实时时钟 | Real-Time Clock for timekeeping. <br> 用于计时的实时时钟。 |
| Calendar | 日历 | Date and time tracking (year, month, day, hour, minute, second). <br> 日期和时间跟踪（年、月、日、时、分、秒）。 |
| Alarm | 闹钟 | Programmable time-based interrupt. <br> 可编程的基于时间的中断。 |
| Wakeup Timer | 唤醒定时器 | Periodic wakeup from low-power modes. <br> 从低功耗模式周期性唤醒。 |
| Backup Domain | 备份域 | RTC registers powered by VBAT, retain data during power-off. <br> 由VBAT供电的RTC寄存器，断电时保留数据。 |
| VBAT | 备用电池 | Backup battery voltage for RTC. <br> RTC的备用电池电压。 |
| Timestamp | 时间戳 | Captures time on external event. <br> 在外部事件时捕获时间。 |
| Subsecond | 亚秒 | Fractional second precision. <br> 秒的小数精度。 |

---

## 10. Watchdog - System Reliability
## 10. Watchdog - 系统可靠性

| Term | 中文名称 | Description |
|------|---------|-------------|
| IWDG | 独立看门狗 | Independent Watchdog, runs on separate LSI clock. <br> 独立看门狗，运行在独立的LSI时钟上。 |
| WWDG | 窗口看门狗 | Window Watchdog, must be refreshed within time window. <br> 窗口看门狗，必须在时间窗口内刷新。 |
| Timeout | 超时 | Period after which watchdog resets system if not refreshed. <br> 如果不刷新，看门狗复位系统的周期。 |
| Refresh | 刷新 | Resetting watchdog counter to prevent system reset. <br> 重置看门狗计数器以防止系统复位。 |
| Window | 窗口 | Time range within which WWDG must be refreshed. <br> WWDG必须刷新的时间范围。 |
| Early Wakeup Interrupt | 早期唤醒中断 | WWDG interrupt before timeout. <br> WWDG超时前的中断。 |

---

## 11. CAN - Controller Area Network
## 11. CAN - 控制器局域网

| Term | 中文名称 | Description |
|------|---------|-------------|
| CAN | 控制器局域网 | Controller Area Network, automotive/industrial bus. <br> 控制器局域网，汽车/工业总线。 |
| CAN_H | CAN高电平线 | CAN High signal line. <br> CAN高电平信号线。 |
| CAN_L | CAN低电平线 | CAN Low signal line. <br> CAN低电平信号线。 |
| Bit Rate | 位速率 | CAN communication speed (e.g., 125 kbps, 500 kbps, 1 Mbps). <br> CAN通信速度（如125 kbps、500 kbps、1 Mbps）。 |
| Standard Frame | 标准帧 | CAN frame with 11-bit identifier. <br> 具有11位标识符的CAN帧。 |
| Extended Frame | 扩展帧 | CAN frame with 29-bit identifier. <br> 具有29位标识符的CAN帧。 |
| Identifier (ID) | 标识符 | Message priority and address. <br> 消息优先级和地址。 |
| Data Frame | 数据帧 | CAN frame carrying data (0-8 bytes). <br> 携带数据的CAN帧（0-8字节）。 |
| Remote Frame | 远程帧 | Request for data from another node. <br> 从另一个节点请求数据。 |
| Error Frame | 错误帧 | Signals transmission error. <br> 信号传输错误。 |
| Filter | 过滤器 | Accepts only specific CAN messages. <br> 仅接受特定的CAN消息。 |
| Mailbox | 邮箱 | CAN transmit/receive buffer. <br> CAN发送/接收缓冲区。 |
| Arbitration | 仲裁 | Priority-based bus access mechanism. <br> 基于优先级的总线访问机制。 |

---

## 12. USB - Universal Serial Bus
## 12. USB - 通用串行总线

| Term | 中文名称 | Description |
|------|---------|-------------|
| USB | 通用串行总线 | Universal Serial Bus for high-speed communication. <br> 用于高速通信的通用串行总线。 |
| USB Device | USB设备 | STM32 acts as USB peripheral. <br> STM32作为USB外设。 |
| USB Host | USB主机 | STM32 controls USB devices. <br> STM32控制USB设备。 |
| USB OTG | USB On-The-Go | Can switch between Host and Device modes. <br> 可在主机和设备模式之间切换。 |
| Endpoint | 端点 | Unidirectional data channel in USB. <br> USB中的单向数据通道。 |
| CDC | 通信设备类 | Communication Device Class (Virtual COM Port). <br> 通信设备类（虚拟串口）。 |
| HID | 人机接口设备 | Human Interface Device (keyboard, mouse). <br> 人机接口设备（键盘、鼠标）。 |
| MSC | 大容量存储类 | Mass Storage Class (USB flash drive). <br> 大容量存储类（U盘）。 |
| Enumeration | 枚举 | USB device identification and configuration process. <br> USB设备识别和配置过程。 |
| Descriptor | 描述符 | Data structure describing USB device capabilities. <br> 描述USB设备能力的数据结构。 |

---

## 13. EXTI - External Interrupt
## 13. EXTI - 外部中断

| Term | 中文名称 | Description |
|------|---------|-------------|
| EXTI | 外部中断 | External Interrupt/Event controller. <br> 外部中断/事件控制器。 |
| Interrupt Line | 中断线 | EXTI line connected to GPIO pin. <br> 连接到GPIO引脚的EXTI线。 |
| Rising Edge | 上升沿 | Signal transition from LOW to HIGH. <br> 信号从低到高的转换。 |
| Falling Edge | 下降沿 | Signal transition from HIGH to LOW. <br> 信号从高到低的转换。 |
| Rising/Falling Edge | 双边沿 | Trigger on both edges. <br> 在两个边沿都触发。 |
| Event Mode | 事件模式 | Generates event without CPU interrupt. <br> 产生事件但不触发CPU中断。 |
| Interrupt Mode | 中断模式 | Generates CPU interrupt. <br> 产生CPU中断。 |
| Pending Register | 挂起寄存器 | Indicates which EXTI line triggered. <br> 指示哪条EXTI线触发。 |
| NVIC | 嵌套向量中断控制器 | Nested Vectored Interrupt Controller. <br> 嵌套向量中断控制器。 |

---

## 14. Power Management
## 14. 电源管理

| Term | 中文名称 | Description |
|------|---------|-------------|
| Run Mode | 运行模式 | Normal operation mode, all peripherals active. <br> 正常运行模式，所有外设活动。 |
| Sleep Mode | 睡眠模式 | CPU stopped, peripherals running. <br> CPU停止，外设运行。 |
| Stop Mode | 停止模式 | All clocks stopped, SRAM and registers retained. <br> 所有时钟停止，SRAM和寄存器保留。 |
| Standby Mode | 待机模式 | Lowest power mode, only backup domain active. <br> 最低功耗模式，仅备份域活动。 |
| WFI | 等待中断 | Wait For Interrupt instruction. <br> 等待中断指令。 |
| WFE | 等待事件 | Wait For Event instruction. <br> 等待事件指令。 |
| Wakeup Source | 唤醒源 | Event that exits low-power mode (interrupt, RTC, etc.). <br> 退出低功耗模式的事件（中断、RTC等）。 |
| VDD | 主电源 | Main power supply voltage (typically 3.3V). <br> 主电源电压（通常为3.3V）。 |
| VDDA | 模拟电源 | Analog power supply for ADC/DAC. <br> ADC/DAC的模拟电源。 |
| Brown-out Reset | 欠压复位 | Reset when voltage drops below threshold. <br> 电压低于阈值时复位。 |

---

## 15. Flash and EEPROM
## 15. Flash与EEPROM

| Term | 中文名称 | Description |
|------|---------|-------------|
| Flash Memory | Flash存储器 | Non-volatile memory for program code and data. <br> 用于程序代码和数据的非易失性存储器。 |
| EEPROM | 电可擦除只读存储器 | Electrically Erasable Programmable Read-Only Memory. <br> 电可擦除可编程只读存储器。 |
| Page | 页 | Smallest erasable unit in Flash (typically 1-2KB). <br> Flash中最小的可擦除单元（通常为1-2KB）。 |
| Sector | 扇区 | Erasable unit in Flash (STM32F4/F7, varies by size). <br> Flash中的可擦除单元（STM32F4/F7，大小不一）。 |
| Erase | 擦除 | Setting all bits to 1 (0xFF) before programming. <br> 编程前将所有位设置为1（0xFF）。 |
| Program | 编程 | Writing data to Flash memory. <br> 向Flash存储器写入数据。 |
| Write Protection | 写保护 | Prevents accidental modification of Flash. <br> 防止意外修改Flash。 |
| Read Protection | 读保护 | Prevents unauthorized reading of Flash contents. <br> 防止未经授权读取Flash内容。 |
| Bootloader | 引导加载程序 | Program that loads and starts main application. <br> 加载并启动主应用程序的程序。 |
| IAP | 在应用编程 | In-Application Programming, firmware update from application. <br> 在应用编程，从应用程序更新固件。 |

---

## 16. General Terms
## 16. 通用术语

| Term | 中文名称 | Description |
|------|---------|-------------|
| HAL | 硬件抽象层 | Hardware Abstraction Layer, high-level API. <br> 硬件抽象层，高级API。 |
| LL | 底层库 | Low-Layer library, register-level API. <br> 底层库，寄存器级API。 |
| Register | 寄存器 | Hardware memory location for peripheral control. <br> 用于外设控制的硬件内存位置。 |
| Bit | 位 | Single binary digit (0 or 1). <br> 单个二进制数字（0或1）。 |
| Byte | 字节 | 8 bits of data. <br> 8位数据。 |
| Word | 字 | 32 bits (4 bytes) in STM32. <br> STM32中的32位（4字节）。 |
| Half-Word | 半字 | 16 bits (2 bytes). <br> 16位（2字节）。 |
| MSB | 最高有效位 | Most Significant Bit. <br> 最高有效位。 |
| LSB | 最低有效位 | Least Significant Bit. <br> 最低有效位。 |
| Big-Endian | 大端序 | Most significant byte stored first. <br> 最高有效字节先存储。 |
| Little-Endian | 小端序 | Least significant byte stored first (STM32 uses this). <br> 最低有效字节先存储（STM32使用此方式）。 |
| Polling | 轮询 | Repeatedly checking status flag. <br> 重复检查状态标志。 |
| Interrupt | 中断 | Asynchronous event that interrupts normal execution. <br> 中断正常执行的异步事件。 |
| Callback | 回调函数 | Function called when event occurs. <br> 事件发生时调用的函数。 |
| ISR | 中断服务程序 | Interrupt Service Routine. <br> 中断服务程序。 |
| Priority | 优先级 | Determines which interrupt executes first. <br> 决定哪个中断先执行。 |
| Preemption | 抢占 | Higher priority interrupt can interrupt lower priority. <br> 高优先级中断可以中断低优先级。 |
| Timeout | 超时 | Maximum time to wait for operation. <br> 等待操作的最长时间。 |
| Buffer | 缓冲区 | Temporary storage area for data. <br> 数据的临时存储区域。 |
| FIFO | 先进先出 | First In First Out data structure. <br> 先进先出数据结构。 |
| Volatile | 易失性 | Variable that can change unexpectedly (used for registers). <br> 可能意外改变的变量（用于寄存器）。 |

---

## Cross-References / 交叉引用

For detailed information about specific peripherals, refer to:

有关特定外设的详细信息，请参阅：

| Peripheral | 外设 | Directory |
|------------|------|-----------|
| GPIO | 通用输入输出 | `01_GPIO/` |
| UART/USART | 串行通信 | `02_UART_USART/` |
| SPI | 串行外设接口 | `03_SPI/` |
| I2C | I2C总线 | `04_I2C/` |
| Timer | 定时器与PWM | `05_Timer/` |
| ADC/DAC | 模数/数模转换 | `06_ADC_DAC/` |
| DMA | 直接内存访问 | `07_DMA/` |
| RTC | 实时时钟 | `08_RTC/` |
| Watchdog | 看门狗 | `09_Watchdog/` |
| CAN | CAN总线 | `10_CAN/` |
| USB | USB通信 | `11_USB/` |
| EXTI | 外部中断 | `12_EXTI/` |
| Power Management | 电源管理 | `13_Power_Management/` |
| Flash/EEPROM | Flash与EEPROM | `14_Flash_EEPROM/` |

---

**Document Version**: 1.0
**Last Updated**: 2026-01-29
**Reference**: STM32 Peripheral Documentation

