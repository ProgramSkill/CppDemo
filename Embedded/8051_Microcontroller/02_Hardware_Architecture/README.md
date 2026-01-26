# 8051 Hardware Architecture

## Block Diagram Overview

The 8051 microcontroller consists of:
- CPU (Central Processing Unit)
- Memory (RAM and ROM)
- I/O Ports
- Timers/Counters
- Serial Port
- Interrupt System

## Memory Organization

### Program Memory (ROM/EPROM)
- **Size**: 4KB internal (expandable to 64KB external)
- **Address Range**: 0000H to FFFFH
- **Purpose**: Stores program code and constant data

### Data Memory (RAM)
- **Internal RAM**: 128 bytes (00H to 7FH)
  - **Register Banks**: 00H-1FH (4 banks, R0-R7 each)
  - **Bit-Addressable Area**: 20H-2FH (16 bytes = 128 bits)
  - **General Purpose**: 30H-7FH (80 bytes)
- **Special Function Registers (SFR)**: 80H-FFH

### External Memory
- Can expand up to 64KB each for program and data memory

## CPU Architecture

### Registers
- **Accumulator (A)**: 8-bit register for arithmetic operations
- **B Register**: 8-bit register for multiplication/division
- **Program Counter (PC)**: 16-bit register (0000H-FFFFH)
- **Data Pointer (DPTR)**: 16-bit register for external memory access
- **Stack Pointer (SP)**: 8-bit register (default: 07H)
- **Program Status Word (PSW)**: Contains flags (CY, AC, OV, P, etc.)

### Register Banks
Four banks (Bank 0-3), each containing 8 registers (R0-R7)
- Selected using RS0 and RS1 bits in PSW

## I/O Ports

### Port 0 (P0)
- 8-bit open-drain bidirectional port
- Requires external pull-up resistors
- Dual function: I/O or multiplexed address/data bus

### Port 1 (P1)
- 8-bit quasi-bidirectional port
- Internal pull-ups
- General-purpose I/O

### Port 2 (P2)
- 8-bit quasi-bidirectional port
- Dual function: I/O or high-order address bus

### Port 3 (P3)
- 8-bit quasi-bidirectional port
- Alternate functions:
  - P3.0: RXD (Serial input)
  - P3.1: TXD (Serial output)
  - P3.2: INT0 (External interrupt 0)
  - P3.3: INT1 (External interrupt 1)
  - P3.4: T0 (Timer 0 external input)
  - P3.5: T1 (Timer 1 external input)
  - P3.6: WR (External data memory write)
  - P3.7: RD (External data memory read)
