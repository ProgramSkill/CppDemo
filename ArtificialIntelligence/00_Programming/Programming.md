# Programming Fundamentals for AI: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: Python Basics](#chapter-1-python-basics)
  - [Chapter 2: Data Types and Structures](#chapter-2-data-types-and-structures)
  - [Chapter 3: Control Flow](#chapter-3-control-flow)
  - [Chapter 4: Functions and Modules](#chapter-4-functions-and-modules)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 5: Object-Oriented Programming](#chapter-5-object-oriented-programming)
  - [Chapter 6: NumPy for Numerical Computing](#chapter-6-numpy-for-numerical-computing)
  - [Chapter 7: Pandas for Data Analysis](#chapter-7-pandas-for-data-analysis)
  - [Chapter 8: Data Visualization with Matplotlib](#chapter-8-data-visualization-with-matplotlib)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 9: Debugging and Profiling](#chapter-9-debugging-and-profiling)
  - [Chapter 10: Performance Optimization](#chapter-10-performance-optimization)
  - [Chapter 11: Best Practices for AI Projects](#chapter-11-best-practices-for-ai-projects)
  - [Chapter 12: Project: Building a Data Pipeline](#chapter-12-project-building-a-data-pipeline)

---

## Introduction

Welcome to the comprehensive guide on **Programming Fundamentals for AI**! This tutorial is designed to take you from a complete beginner to an expert practitioner in the programming skills essential for artificial intelligence and machine learning.

### What You'll Learn

| Level | Duration | Topics Covered | Skills Acquired |
|-------|----------|----------------|-----------------|
| **Beginner** | 2-4 weeks | Python basics, Data types, Control flow, Functions | Write basic Python programs, understand core concepts |
| **Intermediate** | 4-8 weeks | OOP, NumPy, Pandas, Matplotlib | Build data processing pipelines, create visualizations |
| **Advanced** | 4-6 weeks | Debugging, Optimization, Best practices | Write production-quality code, optimize performance |

### Prerequisites

- Basic computer literacy
- A computer with Python 3.8+ installed
- Curiosity and willingness to learn!

### Why Python for AI?

Python has become the **de facto standard** for AI and machine learning due to:

| Advantage | Description |
|-----------|-------------|
| **Readability** | Clean syntax that reads like pseudocode |
| **Rich Ecosystem** | NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch |
| **Community Support** | Largest ML community, extensive documentation |
| **Rapid Prototyping** | Quick development cycle for experiments |
| **Integration** | Easy to integrate with C/C++ for performance |

---

## Part I: Beginner Level

### Chapter 1: Python Basics

#### 1.1 Your First Python Program

```python
# Your first Python program
print("Hello, World!")
print("Welcome to AI Programming!")
```

**Understanding the Code**:
- `#` starts a comment (ignored by Python)
- `print()` is a built-in function that displays output
- Strings are enclosed in quotes (`"` or `'`)

#### 1.2 Variables and Assignment

Variables are containers for storing data values:

```python
# Variable assignment
name = "Alice"           # String
age = 25                 # Integer
height = 1.75            # Float
is_student = True        # Boolean

# Multiple assignment
x, y, z = 1, 2, 3

# Print variables
print(f"Name: {name}, Age: {age}")
print(f"Height: {height}m, Student: {is_student}")
```

**Naming Conventions**:
```python
# Good variable names
user_name = "John"       # snake_case (recommended)
learningRate = 0.01      # camelCase (less common in Python)
MAX_ITERATIONS = 1000    # UPPER_CASE for constants

# Bad variable names (avoid)
x = "John"               # Not descriptive
# 1st_name = "John"      # Can't start with number (SyntaxError)
```

#### 1.3 Basic Operators

**Arithmetic Operators**:
```python
a, b = 10, 3

print(f"Addition: {a + b}")        # 13
print(f"Subtraction: {a - b}")     # 7
print(f"Multiplication: {a * b}")  # 30
print(f"Division: {a / b}")        # 3.333...
print(f"Floor Division: {a // b}") # 3
print(f"Modulus: {a % b}")         # 1
print(f"Exponent: {a ** b}")       # 1000
```

**Comparison Operators**:
```python
x, y = 5, 10

print(f"x == y: {x == y}")   # False
print(f"x != y: {x != y}")   # True
print(f"x < y: {x < y}")     # True
print(f"x > y: {x > y}")     # False
```

**Logical Operators**:
```python
a, b = True, False

print(f"a and b: {a and b}")   # False
print(f"a or b: {a or b}")     # True
print(f"not a: {not a}")       # False
```

#### 1.4 Input and Output

```python
# Getting user input
name = input("Enter your name: ")
age = int(input("Enter your age: "))  # Convert to integer

# Formatted output
print(f"Hello, {name}! You are {age} years old.")

# Multiple formatting styles
print("Name: %s, Age: %d" % (name, age))           # Old style
print("Name: {}, Age: {}".format(name, age))       # str.format()
print(f"Name: {name}, Age: {age}")                 # f-strings (recommended)
```

---

### Chapter 2: Data Types and Structures

#### 2.1 Primitive Data Types

```python
# Integer - whole numbers
count = 42
big_number = 1_000_000  # Underscores for readability

# Float - decimal numbers
pi = 3.14159
scientific = 1.5e-10    # Scientific notation

# String - text
message = "Hello, AI!"
multiline = """This is a
multiline string"""

# Boolean - True/False
is_active = True
is_complete = False

# NoneType - absence of value
result = None
```

**Type Checking and Conversion**:
```python
# Check type
print(type(42))        # <class 'int'>
print(type(3.14))      # <class 'float'>
print(type("hello"))   # <class 'str'>

# Type conversion
x = int("42")          # String to int
y = float("3.14")      # String to float
z = str(42)            # Int to string
b = bool(1)            # Int to bool (True)
```

#### 2.2 Lists

Lists are **ordered, mutable** collections:

```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
nested = [[1, 2], [3, 4], [5, 6]]

# Accessing elements (0-indexed)
print(numbers[0])      # 1 (first element)
print(numbers[-1])     # 5 (last element)
print(numbers[1:3])    # [2, 3] (slicing)

# List operations
numbers.append(6)      # Add to end
numbers.insert(0, 0)   # Insert at index
numbers.remove(3)      # Remove value
popped = numbers.pop() # Remove and return last

# List methods
print(len(numbers))    # Length
print(sum(numbers))    # Sum
print(sorted(numbers)) # Sorted copy
```

**List Comprehensions**:
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(1, 4)] for i in range(1, 4)]
```

#### 2.3 Tuples

Tuples are **ordered, immutable** collections:

```python
# Creating tuples
point = (3, 4)
rgb = (255, 128, 0)

# Tuple unpacking
x, y = point
print(f"x={x}, y={y}")

# Use case: multiple return values
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

minimum, maximum, average = get_stats([1, 2, 3, 4, 5])
```

#### 2.4 Dictionaries

Dictionaries are **key-value pairs**:

```python
# Creating dictionaries
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

# Accessing values
print(person["name"])           # Alice
print(person.get("age"))        # 25
print(person.get("job", "N/A")) # N/A (default)

# Modifying dictionaries
person["job"] = "Engineer"      # Add new key-value
person["age"] = 26              # Update value
del person["city"]              # Delete key-value

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
```

#### 2.5 Sets

Sets are **unordered collections of unique elements**:

```python
# Creating sets
numbers = {1, 2, 3, 4, 5}
unique = set([1, 1, 2, 2, 3, 3])  # {1, 2, 3}

# Set operations
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)    # Union: {1, 2, 3, 4, 5, 6}
print(a & b)    # Intersection: {3, 4}
print(a - b)    # Difference: {1, 2}
```

---

### Chapter 3: Control Flow

#### 3.1 Conditional Statements

```python
# Basic if-elif-else
age = 18

if age < 13:
    print("Child")
elif age < 20:
    print("Teenager")
else:
    print("Adult")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"
```

#### 3.2 Loops

**For Loops**:
```python
# Iterate over list
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# Iterate with index
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# Range-based loops
for i in range(5):          # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 10, 2):   # 2, 4, 6, 8
    print(i)
```

**While Loops**:
```python
# Basic while loop
count = 0
while count < 5:
    print(count)
    count += 1

# While with break
while True:
    user_input = input("Enter 'quit' to exit: ")
    if user_input == 'quit':
        break
```

#### 3.3 Exception Handling

```python
# Basic try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int("not a number")
except ValueError:
    print("Invalid number format")
except TypeError:
    print("Type error occurred")

# Try-except-else-finally
try:
    file = open("data.txt", "r")
    data = file.read()
except FileNotFoundError:
    print("File not found")
else:
    print("File read successfully")
finally:
    print("This always executes")
```

---

### Chapter 4: Functions and Modules

#### 4.1 Defining Functions

```python
# Basic function
def greet(name):
    """Greet a person by name."""
    return f"Hello, {name}!"

# Function with default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Function with multiple return values
def calculate_stats(numbers):
    total = sum(numbers)
    count = len(numbers)
    average = total / count
    return total, count, average
```

#### 4.2 *args and **kwargs

```python
# *args - variable positional arguments
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))          # 6
print(sum_all(1, 2, 3, 4, 5))    # 15

# **kwargs - variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25)
```

#### 4.3 Lambda Functions

```python
# Lambda syntax
square = lambda x: x ** 2
add = lambda x, y: x + y

# Common use cases
numbers = [1, 2, 3, 4, 5]

# With map()
squares = list(map(lambda x: x**2, numbers))

# With filter()
evens = list(filter(lambda x: x % 2 == 0, numbers))

# With sorted()
points = [(1, 2), (3, 1), (2, 3)]
sorted_by_y = sorted(points, key=lambda p: p[1])
```

#### 4.4 Modules and Imports

```python
# Importing modules
import math
print(math.sqrt(16))    # 4.0

# Import specific functions
from math import sqrt, pi

# Import with alias
import numpy as np
import pandas as pd
```

#### 4.5 Decorators

```python
# Basic decorator
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "Done"
```

---

## Part II: Intermediate Level

### Chapter 5: Object-Oriented Programming

#### 5.1 Classes and Objects

```python
class Dog:
    """A simple Dog class."""
    
    # Class attribute
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def birthday(self):
        self.age += 1
        return f"{self.name} is now {self.age} years old"

# Creating objects
buddy = Dog("Buddy", 3)
print(buddy.bark())      # Buddy says Woof!
```

#### 5.2 Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Polymorphism
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.speak())
```

#### 5.3 Encapsulation

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner           # Public
        self._balance = balance      # Protected
        self.__pin = "1234"          # Private
    
    @property
    def balance(self):
        return self._balance
    
    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = value
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
```

#### 5.4 Special Methods

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

v1 = Vector(3, 4)
v2 = Vector(1, 2)
print(v1 + v2)      # Vector(4, 6)
```

---

### Chapter 6: NumPy for Numerical Computing

#### 6.1 NumPy Basics

```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.zeros((3, 4))
arr3 = np.ones((2, 3))
arr4 = np.eye(3)
arr5 = np.arange(0, 10, 2)
arr6 = np.linspace(0, 1, 5)
arr7 = np.random.randn(3, 3)

# Array attributes
print(arr1.shape)      # (5,)
print(arr1.dtype)      # int64
print(arr1.ndim)       # 1

# Reshaping
arr = np.arange(12).reshape(3, 4)
```

#### 6.2 Array Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Element-wise operations
print(a + b)        # [ 6  8 10 12]
print(a * b)        # [ 5 12 21 32]
print(a ** 2)       # [ 1  4  9 16]
print(np.sqrt(a))   # [1. 1.41 1.73 2.]

# Aggregations
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(arr))         # 21
print(np.sum(arr, axis=0)) # [5, 7, 9]
print(np.mean(arr))        # 3.5
```

#### 6.3 Indexing and Slicing

```python
arr = np.arange(20).reshape(4, 5)

# Basic indexing
print(arr[0, 0])      # 0
print(arr[2, 3])      # 13

# Slicing
print(arr[0:2, 1:4])  # Rows 0-1, columns 1-3
print(arr[:, 0])      # First column
print(arr[1, :])      # Second row

# Boolean indexing
print(arr[arr > 10])
```

#### 6.4 Broadcasting

```python
# Scalar broadcast
arr = np.array([1, 2, 3])
print(arr + 10)  # [11, 12, 13]

# Row vector broadcast
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = np.array([1, 0, 1])
print(matrix + row)
```

#### 6.5 Linear Algebra

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
print(A @ B)
print(np.dot(A, B))

# Transpose
print(A.T)

# Determinant and Inverse
print(np.linalg.det(A))
print(np.linalg.inv(A))

# Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solving linear systems
b = np.array([1, 2])
x = np.linalg.solve(A, b)
```

---

### Chapter 7: Pandas for Data Analysis

#### 7.1 Series and DataFrames

```python
import pandas as pd
import numpy as np

# Creating a Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Creating a DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['NYC', 'LA', 'Chicago', 'Houston'],
    'Salary': [70000, 80000, 90000, 75000]
})

# Reading/Writing CSV
# df = pd.read_csv('data.csv')
# df.to_csv('output.csv', index=False)
```

#### 7.2 Data Selection and Filtering

```python
# Column selection
print(df['Name'])           # Single column
print(df[['Name', 'Age']])  # Multiple columns

# Row selection
print(df.iloc[0])           # First row by index
print(df.loc[0])            # First row by label

# Boolean filtering
print(df[df['Age'] > 28])
print(df[(df['Age'] > 25) & (df['Salary'] < 85000)])
```

#### 7.3 Data Manipulation

```python
# Adding columns
df['Bonus'] = df['Salary'] * 0.1
df['Tax'] = df['Salary'].apply(lambda x: x * 0.2 if x > 75000 else x * 0.15)

# Renaming columns
df = df.rename(columns={'Name': 'Employee'})

# Sorting
df_sorted = df.sort_values('Salary', ascending=False)

# Groupby
print(df.groupby('City')['Salary'].mean())
```

#### 7.4 Handling Missing Data

```python
# Detecting missing values
print(df.isnull().sum())

# Dropping missing values
df_dropped = df.dropna()

# Filling missing values
df_filled = df.fillna(0)
df_filled = df.fillna(df.mean())
```

#### 7.5 Merging and Joining

```python
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['A', 'B', 'C']})
df2 = pd.DataFrame({'ID': [1, 2, 4], 'Score': [85, 90, 78]})

# Merge
merged = pd.merge(df1, df2, on='ID', how='inner')
merged_left = pd.merge(df1, df2, on='ID', how='left')

# Concatenation
df_concat = pd.concat([df1, df1], ignore_index=True)
```

---

### Chapter 8: Data Visualization with Matplotlib

#### 8.1 Basic Plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Function')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5
plt.scatter(x, y, c='blue', alpha=0.6)
plt.show()
```

#### 8.2 Statistical Plots

```python
# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.show()

# Box plot
data_groups = [np.random.randn(100) + i for i in range(4)]
plt.boxplot(data_groups, labels=['A', 'B', 'C', 'D'])
plt.show()

# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values)
plt.show()
```

#### 8.3 Subplots

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Line Plot')

axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
axes[0, 1].set_title('Scatter Plot')

axes[1, 0].bar(['A', 'B', 'C'], [10, 20, 30])
axes[1, 0].set_title('Bar Chart')

axes[1, 1].pie([30, 25, 20, 15, 10], labels=['A', 'B', 'C', 'D', 'E'])
axes[1, 1].set_title('Pie Chart')

plt.tight_layout()
plt.show()
```

---

## Part III: Advanced Level

### Chapter 9: Debugging and Profiling

#### 9.1 Debugging Techniques

```python
# Using logging module
import logging

logging.basicConfig(level=logging.DEBUG)

def calculate_mean(numbers):
    logging.debug(f"Input: {numbers}")
    if not numbers:
        logging.warning("Empty list")
        return None
    mean = sum(numbers) / len(numbers)
    logging.info(f"Mean: {mean}")
    return mean

# Using pdb
import pdb

def complex_function(data):
    result = []
    for i, item in enumerate(data):
        if i == 5:
            pdb.set_trace()  # Debugger starts here
        result.append(item ** 2)
    return result
```

#### 9.2 Profiling

```python
import cProfile
import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.4f}s")
        return result
    return wrapper

# Profile a function
cProfile.run('function_to_profile()')
```

---

### Chapter 10: Performance Optimization

#### 10.1 Algorithm Optimization

```python
# O(n²) - slow
def find_pairs_slow(numbers, target):
    pairs = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == target:
                pairs.append((numbers[i], numbers[j]))
    return pairs

# O(n) - fast
def find_pairs_fast(numbers, target):
    seen = set()
    pairs = []
    for num in numbers:
        complement = target - num
        if complement in seen:
            pairs.append((complement, num))
        seen.add(num)
    return pairs
```

#### 10.2 NumPy Vectorization

```python
# Slow: Python loop
def normalize_loop(data):
    result = []
    mean = sum(data) / len(data)
    std = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    for x in data:
        result.append((x - mean) / std)
    return result

# Fast: NumPy
def normalize_numpy(data):
    arr = np.array(data)
    return (arr - arr.mean()) / arr.std()
```

#### 10.3 Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Process Pool for CPU-bound tasks
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(cpu_bound_task, data))

# Thread Pool for I/O-bound tasks
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(io_bound_task, urls))
```

---

### Chapter 11: Best Practices for AI Projects

#### 11.1 Project Structure

```
my_ai_project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── tests/
├── configs/
├── requirements.txt
└── README.md
```

#### 11.2 Code Quality

```python
from typing import List, Tuple
import numpy as np

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    epochs: int = 100
) -> Tuple[np.ndarray, List[float]]:
    """
    Train a simple linear model.
    
    Args:
        X: Input features
        y: Target values
        learning_rate: Step size
        epochs: Number of iterations
    
    Returns:
        weights, losses
    """
    pass
```

#### 11.3 Reproducibility

```python
import random
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
```

---

### Chapter 12: Project: Building a Data Pipeline

See the `Python/` subdirectory for a complete data pipeline implementation including:
- Configuration management
- Data loading and validation
- Feature engineering
- Model training and evaluation

---

**Last Updated**: 2024-01-29
