# Assembly Language Programming

## Basic Syntax

### Instruction Format
```assembly
LABEL:  MNEMONIC  OPERAND  ; Comment
```

### Example Program
```assembly
ORG 0000H           ; Origin at 0000H
LJMP MAIN           ; Jump to main program

ORG 0030H           ; Start main program
MAIN:
    MOV P1, #0FFH   ; Set P1 to 0xFF
    MOV P1, #00H    ; Set P1 to 0x00
    SJMP MAIN       ; Loop forever
END
```

## Common Instructions

### Data Transfer
- `MOV A, #data` - Move immediate data to accumulator
- `MOV A, Rn` - Move register to accumulator
- `MOV A, @Ri` - Move indirect RAM to accumulator
- `MOVX A, @DPTR` - Move external RAM to accumulator

### Arithmetic
- `ADD A, #data` - Add immediate to accumulator
- `SUBB A, #data` - Subtract with borrow
- `INC A` - Increment accumulator
- `DEC A` - Decrement accumulator
- `MUL AB` - Multiply A and B
- `DIV AB` - Divide A by B

### Logic
- `ANL A, #data` - AND operation
- `ORL A, #data` - OR operation
- `XRL A, #data` - XOR operation
- `CPL A` - Complement accumulator

### Branch
- `LJMP addr` - Long jump
- `SJMP addr` - Short jump
- `JZ addr` - Jump if zero
- `JNZ addr` - Jump if not zero
- `CJNE A, #data, addr` - Compare and jump if not equal
