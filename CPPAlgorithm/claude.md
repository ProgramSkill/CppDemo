# CPPAlgorithm Project

## Project Overview

This is a C++ algorithm learning project built with Visual Studio. The project focuses on implementing and understanding fundamental algorithms and data structures through hands-on coding practice.

## Project Structure

```
CPPAlgorithm/
├── recursion/           # Recursion algorithms and examples
│   ├── linear/          # Linear recursion (single-branch)
│   ├── binary/          # Binary recursion (dual-branch)
│   ├── tail/            # Tail recursion
│   └── backtracking/    # Backtracking algorithms
├── CPPAlgorithm.vcxproj # Visual Studio project file
└── x64/                 # Build output directory
```

## Build System

- **IDE**: Visual Studio (Windows)
- **Language**: C++17 or later
- **Project Type**: Console Application

## Coding Conventions

### File Naming
- Use lowercase with underscores: `array_sum.cpp`, `binary_search.cpp`
- Prefix with number for ordered learning: `01_factorial.cpp`, `02_fibonacci.cpp`

### File Encoding
- All `.cpp` and `.h` files must use **UTF-8 with BOM** encoding

### Code Style
- Use descriptive variable names
- Add comments explaining the algorithm logic
- Include time and space complexity analysis in comments
- Provide example inputs/outputs in main() for testing

### Example File Template

```cpp
/*
 * Algorithm: [Name]
 * Type: [Recursion Type / Algorithm Category]
 * Time Complexity: O(?)
 * Space Complexity: O(?)
 *
 * Description:
 * [Brief explanation of what this algorithm does]
 */

#include <iostream>
using namespace std;

// Function implementation
int algorithm(int n) {
    // Base case
    if (n <= 1) return 1;

    // Recursive step
    return n * algorithm(n - 1);
}

int main() {
    // Test cases
    cout << "algorithm(5) = " << algorithm(5) << endl;
    return 0;
}
```

## Algorithm Categories

### Currently Implemented
- **Recursion**: Factorial, Fibonacci, array operations, tree traversals

### Planned Topics
- Sorting algorithms (QuickSort, MergeSort, HeapSort)
- Searching algorithms (Binary Search, DFS, BFS)
- Dynamic Programming
- Graph algorithms
- String algorithms

## Development Guidelines

1. **Focus on clarity**: Code should be educational and easy to understand
2. **Include visualization**: Add debug output to show algorithm execution flow
3. **Test thoroughly**: Include multiple test cases covering edge cases
4. **Document complexity**: Always analyze and document time/space complexity
5. **Compare approaches**: When applicable, show both recursive and iterative solutions

## Common Tasks

- Adding new algorithm implementations
- Optimizing existing algorithms (memoization, tail recursion)
- Converting between recursive and iterative approaches
- Adding test cases and examples
