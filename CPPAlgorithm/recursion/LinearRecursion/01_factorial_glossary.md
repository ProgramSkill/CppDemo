# Factorial Recursion Glossary / 阶乘递归词汇表

## Recursion Terms / 递归术语

| Term | 术语 | Definition / 定义 |
|------|------|-------------------|
| Recursion | 递归 | A function that calls itself to solve smaller subproblems |
| Base Case | 基准情况 | The condition that stops recursion (e.g., n == 0 or n == 1) |
| Recursive Step | 递归步骤 | The part where function calls itself with a smaller input |
| Recursive Call | 递归调用 | When a function invokes itself (e.g., factorial(n-1)) |
| Linear Recursion | 线性递归 | Recursion with only one recursive call per function |
| Single-Branch | 单分支 | Only one path of recursive calls (not tree-like) |
| Descending Phase | 递推阶段 | Phase where recursive calls go deeper toward base case |
| Ascending Phase | 回归阶段 | Phase where values return back up the call stack |
| Call Stack | 调用栈 | Memory structure storing active function calls |
| Stack Depth | 栈深度 | Number of nested function calls (O(n) for factorial) |

## C++ Syntax / C++语法

| Term | 术语 | Definition / 定义 |
|------|------|-------------------|
| #include | 包含指令 | Preprocessor directive to include header files |
| iostream | 输入输出流 | Standard library for input/output operations |
| namespace | 命名空间 | Container for identifiers to avoid naming conflicts |
| long long | 长整型 | 64-bit integer type for large numbers |
| int | 整型 | 32-bit integer type |
| return | 返回 | Statement to exit function and return a value |
| cout | 标准输出 | Standard output stream for printing |
| endl | 换行符 | End line and flush output buffer |
| main() | 主函数 | Program entry point |

## Algorithm Concepts / 算法概念

| Term | 术语 | Definition / 定义 |
|------|------|-------------------|
| Factorial | 阶乘 | n! = n × (n-1) × ... × 1 |
| Time Complexity | 时间复杂度 | O(n) - n recursive calls |
| Space Complexity | 空间复杂度 | O(n) - call stack depth |
| Error Check | 错误检查 | Validating input before processing |
| Undefined | 未定义 | Invalid operation (e.g., negative factorial) |

## Mathematical Notation / 数学符号

| Symbol | 名称 | Meaning / 含义 |
|--------|------|----------------|
| n! | n的阶乘 | Factorial of n |
| O(n) | 大O表示法 | Linear time/space complexity |
| = | 等于 | Equals |
| × | 乘号 | Multiplication |
