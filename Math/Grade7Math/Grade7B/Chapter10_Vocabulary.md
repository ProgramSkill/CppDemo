# Chapter 10 Systems of Linear Equations in Two Variables
# Vocabulary List (From Beginner to Competition Level)

## Basic Terms

| Term | Definition |
|------|------------|
| Variable | A symbol (usually a letter) representing an unknown value |
| Coefficient | The numerical factor of a term containing a variable |
| Constant | A fixed numerical value that does not change |
| Equation | A mathematical statement showing two expressions are equal |
| Solution | The value(s) that make an equation true |

## Core Concepts

| Term | Definition |
|------|------------|
| Linear equation in one variable | An equation with one unknown where the highest power is 1 |
| Linear equation in two variables | An equation with two unknowns where the highest power of each is 1 |
| System of linear equations | Two or more equations considered together |
| System of linear equations in two variables | Two linear equations with the same two unknowns |

## Solution Methods

| Term | Definition |
|------|------------|
| Elimination | The process of removing one variable from a system |
| Substitution method | Solving by expressing one variable in terms of another and substituting |
| Addition-subtraction method | Solving by adding or subtracting equations to eliminate a variable |
| Back-substitution | Substituting a found value back to find other unknowns |
| Verification | Checking if the solution satisfies all original equations |

## Equation Components

| Term | Definition |
|------|------------|
| Term | A single mathematical expression (number, variable, or product) |
| Degree | The power/exponent of a variable in a term |
| Standard form | The conventional way of writing an equation (ax + by = c) |
| Left-hand side (LHS) | The expression on the left of the equals sign |
| Right-hand side (RHS) | The expression on the right of the equals sign |

## Solution Types

| Term | Definition |
|------|------------|
| Unique solution | Exactly one solution exists |
| No solution | No values satisfy all equations |
| Infinitely many solutions | Unlimited number of solutions exist |
| Integer solution | A solution where all values are integers |
| Positive integer solution | A solution where all values are positive integers |

## Geometric Terms

| Term | Definition |
|------|------------|
| Cartesian coordinate system | A plane with perpendicular x and y axes |
| Straight line | A line extending infinitely in both directions |
| Intersection point | The point where two lines cross |
| Parallel lines | Lines that never intersect |
| Coincident lines | Lines that overlap completely |

## Word Problem Terms

| Term | Definition |
|------|------------|
| Sum | The result of addition |
| Difference | The result of subtraction |
| Distance problem | Problems involving speed, time, and distance |
| Work problem | Problems involving work rate and time |
| Matching problem | Problems about pairing items in correct ratios |
| Efficiency | The rate at which work is completed |

## Mathematical Operations

| Term | Definition |
|------|------------|
| Add | Combine numbers to get a sum |
| Subtract | Find the difference between numbers |
| Multiply | Repeated addition of a number |
| Divide | Split into equal parts |
| Simplify | Reduce an expression to its simplest form |
| Transform | Change the form of an equation |

## Common Phrases

| Phrase | Usage |
|--------|-------|
| According to the problem | Introducing information from the problem statement |
| Let x be... | Defining a variable |
| Substitute into | Replacing a variable with an expression |
| From equation ① | Referencing a numbered equation |
| Therefore | Introducing a conclusion |
| The solution is | Stating the final answer |
| Verify the answer | Checking the solution |
| Clear denominators | Multiplying to eliminate fractions |
| Remove parentheses | Expanding brackets |
| Combine like terms | Simplifying by grouping similar terms |

---

## Key Formulas

### Standard Form of Linear Equation in Two Variables

```
ax + by = c  (where a ≠ 0, b ≠ 0)
```

### System of Linear Equations in Two Variables

```
{  ax + by = c  ①
{  dx + ey = f  ②
```

### Common Word Problem Formulas

| Type | Formula |
|------|---------|
| Distance | Distance = Speed × Time |
| Work | Work = Rate × Time |
| Price | Total = Unit Price × Quantity |
| Profit | Profit = Revenue - Cost |

---

## Advanced Terms (Competition Level)

### Number Theory

| Term | Definition |
|------|------------|
| Greatest Common Divisor (GCD) | The largest positive integer that divides two numbers |
| Least Common Multiple (LCM) | The smallest positive integer divisible by two numbers |
| Coprime / Relatively prime | Two numbers whose GCD is 1 |
| Divisibility | Property of one number dividing another with no remainder |
| Diophantine equation | An equation seeking integer solutions |
| Particular solution | One specific solution to an equation |
| General solution | Formula representing all solutions |

### Algorithms

| Term | Definition |
|------|------------|
| Euclidean algorithm | Method to find GCD by repeated division |
| Extended Euclidean algorithm | Finds GCD and coefficients for Bezout's identity |
| Bezout's identity | ax + by = gcd(a,b) has integer solutions |
| Back-substitution | Working backwards to find coefficients |

### Linear Algebra Terms

| Term | Definition |
|------|------------|
| Determinant | A scalar value computed from a square matrix |
| Cramer's Rule | Method to solve systems using determinants |
| Matrix | A rectangular array of numbers |
| Homogeneous system | System where all constant terms are zero |
| Trivial solution | The zero solution (0, 0) |
| Non-trivial solution | A solution other than (0, 0) |

### Competition Techniques

| Term | Definition |
|------|------------|
| Simon's Favorite Factoring Trick | Adding a constant to factor xy + ax + by |
| Frobenius number | Largest integer not representable as ax + by |
| Chicken McNugget Theorem | g(a,b) = ab - a - b for coprime a, b |
| Vieta's formulas | Relates roots of polynomial to its coefficients |
| Symmetric system | System unchanged when variables are swapped |

### Special Functions

| Term | Definition |
|------|------------|
| Absolute value | Distance from zero, always non-negative |
| Floor function | Greatest integer less than or equal to x |
| Ceiling function | Smallest integer greater than or equal to x |

### System Properties

| Term | Definition |
|------|------------|
| Consistent system | System with at least one solution |
| Inconsistent system | System with no solution |
| Overdetermined system | More equations than unknowns |
| Parameter | A variable representing a family of values |

---

## Competition Formulas

### Cramer's Rule
```
For system: ax + by = c, dx + ey = f

x = (ce - bf) / (ae - bd)
y = (af - cd) / (ae - bd)
```

### Diophantine Equation
```
ax + by = c has integer solutions iff gcd(a,b) | c

General solution:
x = x₀ + (b/d)t
y = y₀ - (a/d)t
where d = gcd(a,b)
```

### Frobenius Number
```
For coprime a, b:
g(a,b) = ab - a - b
```
