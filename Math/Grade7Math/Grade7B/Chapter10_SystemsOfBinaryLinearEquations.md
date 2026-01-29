# Chapter 10: Systems of Linear Equations in Two Variables
## From Beginner to Competition Level

---

# Part I: Foundations (Beginner Level)

## 1.1 Introduction: Why Two Variables?

### 1.1.1 The Limitation of One Variable

In previous studies, we learned to solve problems using one unknown:

**Example**: Xiao Ming bought some apples. If he buys 3 more, he will have 10 apples. How many does he have now?

```
Let x = number of apples Xiao Ming has
Equation: x + 3 = 10
Solution: x = 7
```

This works well for simple problems. But what about this?

**Example**: Xiao Ming has some apples, Xiao Hong has some oranges. Together they have 10 fruits. Xiao Hong has 2 more than Xiao Ming. How many does each have?

Using one variable is possible but awkward:
```
Let x = Xiao Ming's apples
Then Xiao Hong has (x + 2) oranges
Equation: x + (x + 2) = 10
```

### 1.1.2 The Power of Two Variables

Using two variables is more natural:
```
Let x = Xiao Ming's apples
Let y = Xiao Hong's oranges

From "together they have 10": x + y = 10
From "Xiao Hong has 2 more": y = x + 2

This gives us a system of equations!
```

**Key Insight**: Real-world problems often involve multiple unknown quantities. Using multiple variables allows us to model these relationships directly.

---

## 1.2 Linear Equations in Two Variables

### 1.2.1 Definition

> **Definition**: An equation containing **two unknowns** where the **degree of each term containing unknowns is 1** is called a **linear equation in two variables**.

**Standard Form**: $ax + by = c$ where $a \neq 0$, $b \neq 0$

### 1.2.2 Identifying Linear Equations in Two Variables

**Three Essential Conditions**:
1. Exactly two different unknowns
2. Each term with unknowns has degree 1
3. Unknowns are not in denominators or under radicals

| Equation | Linear in Two Variables? | Reason |
|----------|-------------------------|--------|
| $3x + 2y = 5$ | Yes | Two unknowns, both degree 1 |
| $x - y = 7$ | Yes | Two unknowns, both degree 1 |
| $x^2 + y = 4$ | No | $x^2$ has degree 2 |
| $xy = 6$ | No | $xy$ has degree 2 |
| $x + y + z = 10$ | No | Three unknowns |
| $\frac{1}{x} + y = 3$ | No | $x$ is in denominator |
| $\sqrt{x} + y = 2$ | No | $x$ is under radical |

**Special Case**: $x = 5$ can be written as $x + 0y = 5$, so it is technically a linear equation in two variables.

### 1.2.3 Solutions of a Linear Equation in Two Variables

> **Definition**: A pair of values $(x, y)$ that makes the equation true is called a **solution** of the equation.

**Example**: For the equation $x + y = 5$

| x | y | Check: x + y = 5? |
|---|---|-------------------|
| 1 | 4 | 1 + 4 = 5 ✓ |
| 2 | 3 | 2 + 3 = 5 ✓ |
| 0 | 5 | 0 + 5 = 5 ✓ |
| -1 | 6 | -1 + 6 = 5 ✓ |
| 2.5 | 2.5 | 2.5 + 2.5 = 5 ✓ |

**Key Property**: A single linear equation in two variables has **infinitely many solutions**.

**Geometric Interpretation**: The equation $ax + by = c$ represents a **straight line** in the coordinate plane. Every point on this line is a solution.

---

## 1.3 Systems of Linear Equations in Two Variables

### 1.3.1 Why Systems?

Since one equation has infinitely many solutions, we cannot determine unique values for $x$ and $y$. We need **additional constraints**.

**The Classic "Chickens and Rabbits" Problem**:
> A cage contains chickens and rabbits, totaling 10 heads and 28 legs. How many of each?

Let $x$ = number of chickens, $y$ = number of rabbits

- From "10 heads": $x + y = 10$
- From "28 legs" (chickens have 2, rabbits have 4): $2x + 4y = 28$

Together, these form a **system of linear equations in two variables**.

### 1.3.2 Definition

> **Definition**: Two or more linear equations in two variables with the **same unknowns** form a **system of linear equations in two variables**.

**Standard Notation**:
$$\begin{cases} a_1x + b_1y = c_1 \\ a_2x + b_2y = c_2 \end{cases}$$

### 1.3.3 Solution of a System

> **Definition**: A pair of values that satisfies **all equations simultaneously** is called the **solution of the system**.

**Geometric Interpretation**: The solution is the **intersection point** of the two lines.

**Three Possibilities**:

| Geometric Relationship | Algebraic Condition | Number of Solutions |
|-----------------------|---------------------|---------------------|
| Lines intersect | $\frac{a_1}{a_2} \neq \frac{b_1}{b_2}$ | Exactly one |
| Lines are parallel | $\frac{a_1}{a_2} = \frac{b_1}{b_2} \neq \frac{c_1}{c_2}$ | None |
| Lines coincide | $\frac{a_1}{a_2} = \frac{b_1}{b_2} = \frac{c_1}{c_2}$ | Infinitely many |

---

## 1.4 Solving Systems: Substitution Method

### 1.4.1 Core Idea

**Elimination**: Convert a system of two variables into a single equation with one variable.

```
Two variables → One variable → Solve → Back-substitute
```

### 1.4.2 Steps

1. **Isolate** one variable in one equation
2. **Substitute** into the other equation
3. **Solve** the resulting equation
4. **Back-substitute** to find the other variable
5. **Verify** the solution

### 1.4.3 Example

**Solve**:
$$\begin{cases} y = 2x - 1 & \text{①} \\ 3x + 2y = 12 & \text{②} \end{cases}$$

**Solution**:

Step 1: Equation ① already has $y$ isolated.

Step 2: Substitute ① into ②:
$$3x + 2(2x - 1) = 12$$

Step 3: Solve:
$$3x + 4x - 2 = 12$$
$$7x = 14$$
$$x = 2$$

Step 4: Back-substitute into ①:
$$y = 2(2) - 1 = 3$$

Step 5: Verify in ②: $3(2) + 2(3) = 6 + 6 = 12$ ✓

**Answer**: $\begin{cases} x = 2 \\ y = 3 \end{cases}$

### 1.4.4 When to Use Substitution

- One equation already has a variable isolated
- One variable has coefficient 1 or -1
- Avoiding fractions is possible

---

## 1.5 Solving Systems: Elimination Method

### 1.5.1 Core Idea

Add or subtract equations to eliminate one variable.

**Mathematical Basis**: If $A = B$ and $C = D$, then $A \pm C = B \pm D$

### 1.5.2 Steps

1. **Align** coefficients of one variable (multiply if needed)
2. **Add or subtract** equations to eliminate that variable
3. **Solve** for the remaining variable
4. **Back-substitute** to find the other variable
5. **Verify** the solution

### 1.5.3 Example 1: Coefficients Already Aligned

**Solve**:
$$\begin{cases} 3x + 2y = 13 & \text{①} \\ 3x - 2y = 5 & \text{②} \end{cases}$$

**Solution**:

Coefficients of $x$ are both 3. Coefficients of $y$ are 2 and -2 (opposites).

① + ②: $(3x + 3x) + (2y - 2y) = 13 + 5$
$$6x = 18 \Rightarrow x = 3$$

Substitute into ①: $3(3) + 2y = 13 \Rightarrow y = 2$

**Answer**: $\begin{cases} x = 3 \\ y = 2 \end{cases}$

### 1.5.4 Example 2: Coefficients Need Adjustment

**Solve**:
$$\begin{cases} 2x + 3y = 7 & \text{①} \\ 3x + 2y = 8 & \text{②} \end{cases}$$

**Solution**:

To eliminate $x$: multiply ① by 3 and ② by 2

① × 3: $6x + 9y = 21$ ... ③
② × 2: $6x + 4y = 16$ ... ④

③ - ④: $5y = 5 \Rightarrow y = 1$

Substitute into ①: $2x + 3(1) = 7 \Rightarrow x = 2$

**Answer**: $\begin{cases} x = 2 \\ y = 1 \end{cases}$

### 1.5.5 When to Use Elimination

- Coefficients of one variable are equal or opposite
- Coefficients have simple LCM
- Both variables have non-unit coefficients

---

## 1.6 Word Problems with Systems

### 1.6.1 Problem-Solving Framework

```
Real Problem → Set Variables → Find Relationships →
Build System → Solve → Verify → Answer
```

### 1.6.2 Common Problem Types

**Type 1: Sum and Difference**
> Two numbers sum to 50 and differ by 14. Find them.

$$\begin{cases} x + y = 50 \\ x - y = 14 \end{cases}$$

Adding: $2x = 64 \Rightarrow x = 32, y = 18$

**Type 2: Age Problems**
> A father is 30 years older than his son. In 5 years, the father will be 3 times as old as the son. Find their current ages.

Let $f$ = father's age, $s$ = son's age

$$\begin{cases} f = s + 30 \\ f + 5 = 3(s + 5) \end{cases}$$

Substituting: $s + 30 + 5 = 3s + 15 \Rightarrow s = 10, f = 40$

**Type 3: Distance Problems**
> Two cars start 300 km apart and drive toward each other. Car A travels at 60 km/h, Car B at 40 km/h. When do they meet?

Let $t$ = time until meeting

Distance by A: $60t$, Distance by B: $40t$

$60t + 40t = 300 \Rightarrow t = 3$ hours

---

# Part II: Intermediate Level

## 2.1 Systems with Parameters

### 2.1.1 Finding Parameter Values

**Problem**: For what value of $k$ does the system have solution $x = 2, y = 3$?
$$\begin{cases} kx + y = 7 \\ x - ky = -7 \end{cases}$$

**Solution**: Substitute $x = 2, y = 3$:
- From ①: $2k + 3 = 7 \Rightarrow k = 2$
- From ②: $2 - 3k = -7 \Rightarrow k = 3$

Since $k$ must satisfy both, and $2 \neq 3$, **no such $k$ exists**.

### 2.1.2 Conditions for Special Solutions

**Problem**: For what values of $m$ does the system have no solution?
$$\begin{cases} 2x + my = 3 \\ 4x + 6y = 5 \end{cases}$$

**Solution**: No solution when lines are parallel:
$$\frac{2}{4} = \frac{m}{6} \neq \frac{3}{5}$$

From $\frac{2}{4} = \frac{m}{6}$: $m = 3$

Check: $\frac{3}{5} \neq \frac{1}{2}$ ✓

**Answer**: $m = 3$

---

## 2.2 Systems with Three or More Equations

### 2.2.1 Overdetermined Systems

When we have more equations than unknowns, the system may be **inconsistent** (no solution) or **redundant** (some equations are consequences of others).

**Example**: Solve if possible:
$$\begin{cases} x + y = 5 \\ x - y = 1 \\ 2x + y = 8 \end{cases}$$

From ① and ②: $x = 3, y = 2$

Check in ③: $2(3) + 2 = 8$ ✓

The system is consistent; ③ is satisfied by the solution of ① and ②.

### 2.2.2 Systems in Three Variables

**Example**: Solve:
$$\begin{cases} x + y + z = 6 & \text{①} \\ 2x - y + z = 3 & \text{②} \\ x + 2y - z = 5 & \text{③} \end{cases}$$

**Strategy**: Eliminate one variable to get a system in two variables.

① + ③: $2x + 3y = 11$ ... ④
② + ③: $3x + y = 8$ ... ⑤

From ⑤: $y = 8 - 3x$

Substitute into ④: $2x + 3(8 - 3x) = 11$
$2x + 24 - 9x = 11$
$-7x = -13$
$x = \frac{13}{7}$

Continue to find $y = \frac{17}{7}$, $z = \frac{12}{7}$

---

## 2.3 Integer Solutions (Diophantine Equations)

### 2.3.1 Introduction

In many practical problems, we need **integer solutions** (you can't have 2.5 chickens!).

**Definition**: A linear Diophantine equation is an equation of the form $ax + by = c$ where we seek integer solutions.

### 2.3.2 Existence of Integer Solutions

> **Theorem**: The equation $ax + by = c$ has integer solutions if and only if $\gcd(a, b) | c$.

**Examples**:
- $6x + 9y = 12$: $\gcd(6, 9) = 3$, and $3 | 12$ ✓ Solutions exist
- $6x + 9y = 10$: $\gcd(6, 9) = 3$, and $3 \nmid 10$ ✗ No integer solutions

### 2.3.3 Finding All Integer Solutions

**Method**: Find one particular solution, then add the general solution.

**Example**: Find all integer solutions to $3x + 5y = 7$

Step 1: Find one solution by inspection or extended Euclidean algorithm.
By inspection: $x = 4, y = -1$ works: $3(4) + 5(-1) = 12 - 5 = 7$ ✓

Step 2: General solution:
$$x = 4 + 5t, \quad y = -1 - 3t \quad (t \in \mathbb{Z})$$

**Verification**: $3(4 + 5t) + 5(-1 - 3t) = 12 + 15t - 5 - 15t = 7$ ✓

### 2.3.4 Positive Integer Solutions

**Problem**: Find all positive integer solutions to $3x + 5y = 23$

Step 1: General solution (find particular solution first)
$x = 1, y = 4$ works: $3(1) + 5(4) = 23$ ✓

General: $x = 1 + 5t$, $y = 4 - 3t$

Step 2: Apply constraints $x > 0$ and $y > 0$:
- $1 + 5t > 0 \Rightarrow t > -\frac{1}{5} \Rightarrow t \geq 0$
- $4 - 3t > 0 \Rightarrow t < \frac{4}{3} \Rightarrow t \leq 1$

So $t \in \{0, 1\}$

**Solutions**: $(1, 4)$ and $(6, 1)$

---

## 2.4 Special Techniques

### 2.4.1 Symmetric Systems

When a system is symmetric in $x$ and $y$, introduce $s = x + y$ and $p = xy$.

**Example**: Solve:
$$\begin{cases} x + y = 5 \\ xy = 6 \end{cases}$$

$x$ and $y$ are roots of $t^2 - 5t + 6 = 0$

$(t - 2)(t - 3) = 0$

**Solutions**: $(x, y) = (2, 3)$ or $(3, 2)$

### 2.4.2 Ratio Method

When equations involve ratios, use proportionality.

**Example**: If $\frac{x}{2} = \frac{y}{3} = \frac{x + y}{k}$, find $k$.

Let $\frac{x}{2} = \frac{y}{3} = m$

Then $x = 2m$, $y = 3m$, so $x + y = 5m$

$\frac{x + y}{k} = m \Rightarrow \frac{5m}{k} = m \Rightarrow k = 5$

### 2.4.3 Using Determinants (Cramer's Rule)

For the system:
$$\begin{cases} a_1x + b_1y = c_1 \\ a_2x + b_2y = c_2 \end{cases}$$

Define:
$$D = \begin{vmatrix} a_1 & b_1 \\ a_2 & b_2 \end{vmatrix} = a_1b_2 - a_2b_1$$

$$D_x = \begin{vmatrix} c_1 & b_1 \\ c_2 & b_2 \end{vmatrix} = c_1b_2 - c_2b_1$$

$$D_y = \begin{vmatrix} a_1 & c_1 \\ a_2 & c_2 \end{vmatrix} = a_1c_2 - a_2c_1$$

If $D \neq 0$: $x = \frac{D_x}{D}$, $y = \frac{D_y}{D}$

**Example**: Solve $\begin{cases} 2x + 3y = 8 \\ 5x - 2y = 1 \end{cases}$

$D = 2(-2) - 5(3) = -4 - 15 = -19$
$D_x = 8(-2) - 1(3) = -16 - 3 = -19$
$D_y = 2(1) - 5(8) = 2 - 40 = -38$

$x = \frac{-19}{-19} = 1$, $y = \frac{-38}{-19} = 2$

---

# Part III: Advanced Level (Competition Preparation)

## 3.1 The Extended Euclidean Algorithm

### 3.1.1 Bezout's Identity

> **Theorem (Bezout's Identity)**: For any integers $a$ and $b$, there exist integers $x$ and $y$ such that:
> $$ax + by = \gcd(a, b)$$

This is fundamental for solving Diophantine equations systematically.

### 3.1.2 The Algorithm

To find $x, y$ such that $ax + by = \gcd(a, b)$:

**Example**: Find $x, y$ such that $35x + 15y = \gcd(35, 15) = 5$

Step 1: Apply Euclidean algorithm
```
35 = 2 × 15 + 5
15 = 3 × 5 + 0
```
So $\gcd(35, 15) = 5$

Step 2: Back-substitute
```
5 = 35 - 2 × 15
5 = 35 × 1 + 15 × (-2)
```

**Answer**: $x = 1$, $y = -2$

### 3.1.3 Systematic Method (Table Form)

For $\gcd(a, b)$, maintain quotients and back-substitute:

| Step | Equation | Expression for remainder |
|------|----------|-------------------------|
| 1 | $a = q_1 b + r_1$ | $r_1 = a - q_1 b$ |
| 2 | $b = q_2 r_1 + r_2$ | $r_2 = b - q_2 r_1$ |
| ... | ... | ... |

**Example**: Solve $91x + 35y = 7$

Euclidean algorithm:
```
91 = 2 × 35 + 21
35 = 1 × 21 + 14
21 = 1 × 14 + 7
14 = 2 × 7 + 0
```

Back-substitution:
```
7 = 21 - 1 × 14
  = 21 - 1 × (35 - 21)
  = 2 × 21 - 35
  = 2 × (91 - 2 × 35) - 35
  = 2 × 91 - 5 × 35
```

So $x = 2$, $y = -5$ is a particular solution.

General solution: $x = 2 + 5t$, $y = -5 - 13t$ for $t \in \mathbb{Z}$

---

## 3.2 Systems with Absolute Values

### 3.2.1 Basic Approach

When equations contain absolute values, consider cases based on the sign of expressions inside.

**Example**: Solve $|x - 1| + |y - 2| = 0$

Since absolute values are non-negative, and their sum is 0:
$$|x - 1| = 0 \text{ and } |y - 2| = 0$$
$$x = 1, y = 2$$

### 3.2.2 Multiple Cases

**Example**: Solve the system:
$$\begin{cases} |x| + y = 3 \\ x + |y| = 3 \end{cases}$$

**Case 1**: $x \geq 0$, $y \geq 0$
$$\begin{cases} x + y = 3 \\ x + y = 3 \end{cases}$$
Infinitely many solutions on segment: $x + y = 3$, $x \geq 0$, $y \geq 0$

**Case 2**: $x < 0$, $y \geq 0$
$$\begin{cases} -x + y = 3 \\ x + y = 3 \end{cases}$$
Adding: $2y = 6 \Rightarrow y = 3$, $x = 0$
But $x < 0$ required, so no solution in this case.

**Case 3**: $x \geq 0$, $y < 0$
$$\begin{cases} x + y = 3 \\ x - y = 3 \end{cases}$$
Adding: $2x = 6 \Rightarrow x = 3$, $y = 0$
But $y < 0$ required, so no solution.

**Case 4**: $x < 0$, $y < 0$
$$\begin{cases} -x + y = 3 \\ x - y = 3 \end{cases}$$
Adding: $0 = 6$, contradiction. No solution.

**Final Answer**: All points $(x, y)$ with $x + y = 3$, $x \geq 0$, $y \geq 0$

---

## 3.3 Systems with Floor and Ceiling Functions

### 3.3.1 Floor Function Basics

The floor function $\lfloor x \rfloor$ is the greatest integer $\leq x$.

**Property**: $\lfloor x \rfloor \leq x < \lfloor x \rfloor + 1$

### 3.3.2 Example Problem

**Problem**: Find all real solutions to:
$$\begin{cases} \lfloor x \rfloor + y = 2.5 \\ x + \lfloor y \rfloor = 3.5 \end{cases}$$

**Solution**:

Let $\lfloor x \rfloor = m$ and $\lfloor y \rfloor = n$ where $m, n \in \mathbb{Z}$

From the equations:
- $y = 2.5 - m$
- $x = 3.5 - n$

For $\lfloor y \rfloor = n$: $n \leq 2.5 - m < n + 1$
For $\lfloor x \rfloor = m$: $m \leq 3.5 - n < m + 1$

From the first inequality: $1.5 - m \leq n < 2.5 - m$
Since $n$ is an integer: $n = \lceil 1.5 - m \rceil$ or $n = 2 - m$ (when $m$ is integer)

Substituting $n = 2 - m$ into the second inequality:
$m \leq 3.5 - (2 - m) < m + 1$
$m \leq 1.5 + m < m + 1$

The left inequality is always true. The right gives $1.5 < 1$, which is false.

Try $n = 1 - m$ (when $1.5 - m$ is not an integer):
This requires careful case analysis...

After systematic checking: $m = 1, n = 1$ gives $x = 2.5, y = 1.5$

Verify: $\lfloor 2.5 \rfloor + 1.5 = 1 + 1.5 = 2.5$ ✓
$2.5 + \lfloor 1.5 \rfloor = 2.5 + 1 = 3.5$ ✓

---

## 3.4 Parametric Families of Solutions

### 3.4.1 When Parameters Appear in Coefficients

**Problem**: For what values of $a$ does the system have a unique solution?
$$\begin{cases} ax + y = 1 \\ x + ay = 1 \end{cases}$$

**Solution**:

Using Cramer's rule: $D = a \cdot a - 1 \cdot 1 = a^2 - 1$

Unique solution exists when $D \neq 0$:
$$a^2 - 1 \neq 0 \Rightarrow a \neq \pm 1$$

**When $a = 1$**: Both equations become $x + y = 1$ (infinitely many solutions)

**When $a = -1$**:
- Equation 1: $-x + y = 1$
- Equation 2: $x - y = 1$

Adding: $0 = 2$, contradiction (no solution)

### 3.4.2 Finding Solutions in Terms of Parameters

**Problem**: Solve in terms of $k$:
$$\begin{cases} x + ky = k + 1 \\ kx + y = 2k \end{cases}$$

**Solution**:

$D = 1 \cdot 1 - k \cdot k = 1 - k^2$

For $k \neq \pm 1$:
$$x = \frac{(k+1) \cdot 1 - k \cdot 2k}{1 - k^2} = \frac{k + 1 - 2k^2}{1 - k^2} = \frac{-(2k^2 - k - 1)}{1 - k^2} = \frac{-(2k+1)(k-1)}{(1-k)(1+k)}$$
$$x = \frac{(2k+1)(k-1)}{(k-1)(k+1)} = \frac{2k+1}{k+1}$$ (for $k \neq 1$)

Similarly: $y = \frac{k - 1}{k + 1}$ (for $k \neq -1$)

---

## 3.5 Competition Techniques

### 3.5.1 Adding and Subtracting Strategically

**Problem**: If $x + y = 5$ and $x - y = 3$, find $x^2 - y^2$.

**Solution**:
$$x^2 - y^2 = (x+y)(x-y) = 5 \times 3 = 15$$

No need to find $x$ and $y$ individually!

### 3.5.2 Using Symmetry

**Problem**: If $a + b = 7$ and $ab = 12$, find $a^2 + b^2$.

**Solution**:
$$a^2 + b^2 = (a + b)^2 - 2ab = 49 - 24 = 25$$

### 3.5.3 Clever Substitutions

**Problem**: Solve:
$$\begin{cases} \frac{1}{x} + \frac{1}{y} = 5 \\ \frac{1}{x} - \frac{1}{y} = 1 \end{cases}$$

**Solution**: Let $u = \frac{1}{x}$, $v = \frac{1}{y}$

$$\begin{cases} u + v = 5 \\ u - v = 1 \end{cases}$$

Adding: $2u = 6 \Rightarrow u = 3$, so $v = 2$

Therefore: $x = \frac{1}{3}$, $y = \frac{1}{2}$

### 3.5.4 Homogeneous Systems

A system is **homogeneous** if all constant terms are 0:
$$\begin{cases} a_1x + b_1y = 0 \\ a_2x + b_2y = 0 \end{cases}$$

**Key Property**: $(0, 0)$ is always a solution (trivial solution).

Non-trivial solutions exist if and only if $D = a_1b_2 - a_2b_1 = 0$.

---

# Part IV: Competition Level Problems

## 4.1 Classic Competition Problems

### Problem 1 (Chickens and Rabbits - Extended)

> A farmer has chickens, rabbits, and crickets. There are 100 heads and 280 legs total. Chickens have 2 legs, rabbits have 4 legs, and crickets have 6 legs. If the number of crickets equals the number of chickens, how many of each animal are there?

**Solution**:

Let $c$ = chickens, $r$ = rabbits, $k$ = crickets

Given: $k = c$

System:
$$\begin{cases} c + r + k = 100 \\ 2c + 4r + 6k = 280 \\ k = c \end{cases}$$

Substituting $k = c$ into the first two equations:
$$\begin{cases} 2c + r = 100 \\ 8c + 4r = 280 \end{cases}$$

From ①: $r = 100 - 2c$

Substitute into ②: $8c + 4(100 - 2c) = 280$
$8c + 400 - 8c = 280$
$400 = 280$ — Contradiction!

**Answer**: No solution exists. The problem conditions are inconsistent.

---

### Problem 2 (Number Theory)

> Find all pairs of positive integers $(x, y)$ such that $\frac{1}{x} + \frac{1}{y} = \frac{1}{6}$.

**Solution**:

Multiply by $6xy$:
$$6y + 6x = xy$$
$$xy - 6x - 6y = 0$$
$$xy - 6x - 6y + 36 = 36$$
$$(x - 6)(y - 6) = 36$$

Since $x, y > 0$ and we need $(x-6)(y-6) = 36$:

Factor pairs of 36: $(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), (9, 4), (12, 3), (18, 2), (36, 1)$

Also negative pairs: $(-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)$

For positive $x, y$:
- $(x-6, y-6) = (1, 36) \Rightarrow (x, y) = (7, 42)$
- $(x-6, y-6) = (2, 18) \Rightarrow (x, y) = (8, 24)$
- $(x-6, y-6) = (3, 12) \Rightarrow (x, y) = (9, 18)$
- $(x-6, y-6) = (4, 9) \Rightarrow (x, y) = (10, 15)$
- $(x-6, y-6) = (6, 6) \Rightarrow (x, y) = (12, 12)$

And symmetric pairs: $(42, 7), (24, 8), (18, 9), (15, 10)$

For negative factor pairs, we need $x > 0, y > 0$:
- $(x-6, y-6) = (-1, -36) \Rightarrow (x, y) = (5, -30)$ — Invalid
- etc. — All invalid

**Answer**: $(7, 42), (8, 24), (9, 18), (10, 15), (12, 12), (15, 10), (18, 9), (24, 8), (42, 7)$

---

### Problem 3 (Parameter Analysis)

> For what values of $m$ does the system have infinitely many solutions?
> $$\begin{cases} (m-1)x + y = m \\ mx + (m-1)y = 1 \end{cases}$$

**Solution**:

Infinitely many solutions occur when the two equations represent the same line:
$$\frac{m-1}{m} = \frac{1}{m-1} = \frac{m}{1}$$

From $\frac{m-1}{m} = \frac{1}{m-1}$:
$(m-1)^2 = m$
$m^2 - 2m + 1 = m$
$m^2 - 3m + 1 = 0$
$m = \frac{3 \pm \sqrt{5}}{2}$

Check with $\frac{1}{m-1} = \frac{m}{1}$:
$1 = m(m-1) = m^2 - m$

For $m = \frac{3 + \sqrt{5}}{2}$:
$m^2 - m = \frac{(3+\sqrt{5})^2}{4} - \frac{3+\sqrt{5}}{2} = \frac{14 + 6\sqrt{5} - 6 - 2\sqrt{5}}{4} = \frac{8 + 4\sqrt{5}}{4} = 2 + \sqrt{5} \neq 1$

**Answer**: No value of $m$ gives infinitely many solutions.

---

### Problem 4 (Symmetric System)

> Solve the system:
> $$\begin{cases} x + y + z = 6 \\ xy + yz + zx = 11 \\ xyz = 6 \end{cases}$$

**Solution**:

$x, y, z$ are roots of the cubic equation:
$$t^3 - (x+y+z)t^2 + (xy+yz+zx)t - xyz = 0$$
$$t^3 - 6t^2 + 11t - 6 = 0$$

Try $t = 1$: $1 - 6 + 11 - 6 = 0$ ✓

Factor: $(t-1)(t^2 - 5t + 6) = (t-1)(t-2)(t-3) = 0$

**Answer**: $(x, y, z)$ is any permutation of $(1, 2, 3)$

---

### Problem 5 (Competition Classic)

> Find all integer solutions to $x^2 + y^2 = 2xy + 2x + 2y$.

**Solution**:

Rearrange: $x^2 - 2xy + y^2 = 2x + 2y$
$(x - y)^2 = 2(x + y)$

Let $u = x - y$ and $v = x + y$. Then:
$u^2 = 2v$

So $v = \frac{u^2}{2}$, which requires $u$ to be even. Let $u = 2k$.

Then $v = 2k^2$

Solving for $x$ and $y$:
- $x = \frac{u + v}{2} = \frac{2k + 2k^2}{2} = k + k^2 = k(k+1)$
- $y = \frac{v - u}{2} = \frac{2k^2 - 2k}{2} = k^2 - k = k(k-1)$

**Answer**: $(x, y) = (k(k+1), k(k-1))$ for any integer $k$

Examples: $k=0: (0,0)$; $k=1: (2,0)$; $k=2: (6,2)$; $k=-1: (0,2)$

---

### Problem 6 (Chinese Competition Style)

> Given that $x, y$ are positive integers satisfying $3x + 5y = 2023$, find the number of solutions.

**Solution**:

General solution: First find one particular solution.
$3(1) + 5(-1) = -2$, so $3(-1011.5) + 5(606.9)$ doesn't work directly.

Try: $3(6) + 5(1) = 23$, so $3(6 \times 88) + 5(1 \times 88) = 3(528) + 5(88) = 2024$ (close)

Better: $3(1) + 5(404) = 3 + 2020 = 2023$ ✓

Particular solution: $x_0 = 1, y_0 = 404$

General solution: $x = 1 + 5t$, $y = 404 - 3t$

For positive integers:
- $x > 0$: $1 + 5t > 0 \Rightarrow t > -\frac{1}{5} \Rightarrow t \geq 0$
- $y > 0$: $404 - 3t > 0 \Rightarrow t < \frac{404}{3} = 134.\overline{6} \Rightarrow t \leq 134$

**Answer**: 135 solutions (for $t = 0, 1, 2, \ldots, 134$)

---

### Problem 7 (Inequality Constraints)

> Find all pairs of non-negative integers $(x, y)$ satisfying:
> $$2x + 3y \leq 12$$

**Solution**:

For each value of $y$, find valid $x$ values:

- $y = 0$: $2x \leq 12 \Rightarrow x \leq 6$, so $x \in \{0,1,2,3,4,5,6\}$ — 7 pairs
- $y = 1$: $2x \leq 9 \Rightarrow x \leq 4.5$, so $x \in \{0,1,2,3,4\}$ — 5 pairs
- $y = 2$: $2x \leq 6 \Rightarrow x \leq 3$, so $x \in \{0,1,2,3\}$ — 4 pairs
- $y = 3$: $2x \leq 3 \Rightarrow x \leq 1.5$, so $x \in \{0,1\}$ — 2 pairs
- $y = 4$: $2x \leq 0 \Rightarrow x = 0$ — 1 pair

**Answer**: $7 + 5 + 4 + 2 + 1 = 19$ pairs

---

### Problem 8 (National Competition Level)

> If $a + b + c = 0$, simplify:
> $$\frac{a^2}{bc} + \frac{b^2}{ca} + \frac{c^2}{ab}$$

**Solution**:

Since $a + b + c = 0$, we have $a = -(b + c)$

$$\frac{a^2}{bc} + \frac{b^2}{ca} + \frac{c^2}{ab} = \frac{a^3 + b^3 + c^3}{abc}$$

Using the identity: $a^3 + b^3 + c^3 - 3abc = (a+b+c)(a^2+b^2+c^2-ab-bc-ca)$

Since $a + b + c = 0$:
$$a^3 + b^3 + c^3 = 3abc$$

Therefore:
$$\frac{a^3 + b^3 + c^3}{abc} = \frac{3abc}{abc} = 3$$

**Answer**: 3

---

## 4.2 Advanced Techniques

### 4.2.1 The Chicken McNugget Theorem (Frobenius Coin Problem)

> **Theorem**: For coprime positive integers $a$ and $b$, the largest integer that cannot be expressed as $ax + by$ (with non-negative integers $x, y$) is $ab - a - b$.

**Example**: With coins of 3 and 5, what's the largest amount that cannot be made?

$\gcd(3, 5) = 1$, so the answer is $3 \times 5 - 3 - 5 = 7$

Verify: 8 = 3 + 5, 9 = 3×3, 10 = 5×2, 11 = 3 + 3 + 5, ...
But 7 cannot be made with 3s and 5s.

### 4.2.2 Simon's Favorite Factoring Trick

When you have an equation like $xy + ax + by = c$, add $ab$ to both sides:

$$xy + ax + by + ab = c + ab$$
$$(x + b)(y + a) = c + ab$$

This converts a difficult equation into a factoring problem.

**Example**: Find all positive integer solutions to $xy + 2x + 3y = 25$

Add $6$ to both sides:
$$xy + 2x + 3y + 6 = 31$$
$$(x + 3)(y + 2) = 31$$

Since 31 is prime, factor pairs are $(1, 31)$ and $(31, 1)$:
- $(x+3, y+2) = (1, 31) \Rightarrow (x, y) = (-2, 29)$ — Invalid
- $(x+3, y+2) = (31, 1) \Rightarrow (x, y) = (28, -1)$ — Invalid

**Answer**: No positive integer solutions.

---

### 4.2.3 Vieta's Formulas Connection

For a quadratic $t^2 - st + p = 0$ with roots $x$ and $y$:
- $x + y = s$ (sum)
- $xy = p$ (product)

This connects systems involving sum and product to quadratic equations.

---

# Part V: Practice Problems by Level

## 5.1 Basic Level (Foundation)

**Problem 1**: Solve the system:
$$\begin{cases} x + y = 7 \\ x - y = 3 \end{cases}$$

**Problem 2**: Solve using substitution:
$$\begin{cases} y = 3x - 2 \\ 2x + y = 8 \end{cases}$$

**Problem 3**: Solve using elimination:
$$\begin{cases} 2x + 3y = 12 \\ 4x - 3y = 6 \end{cases}$$

**Problem 4**: The sum of two numbers is 20, and their difference is 4. Find the numbers.

**Problem 5**: A father is 24 years older than his son. In 6 years, the father will be twice as old. Find their current ages.

---

## 5.2 Intermediate Level

**Problem 6**: Solve:
$$\begin{cases} 3x + 4y = 10 \\ 5x - 2y = 4 \end{cases}$$

**Problem 7**: Find all positive integer solutions to $2x + 5y = 23$.

**Problem 8**: For what value of $k$ does the system have no solution?
$$\begin{cases} 2x + ky = 3 \\ 6x + 9y = 5 \end{cases}$$

**Problem 9**: Solve:
$$\begin{cases} \frac{x}{2} + \frac{y}{3} = 2 \\ \frac{x}{3} - \frac{y}{2} = -\frac{1}{6} \end{cases}$$

**Problem 10**: If $x + y = 5$ and $xy = 6$, find $x^2 + y^2$.

---

## 5.3 Advanced Level

**Problem 11**: Find all pairs of positive integers $(x, y)$ such that $\frac{1}{x} + \frac{1}{y} = \frac{1}{12}$.

**Problem 12**: Solve:
$$\begin{cases} |x| + y = 5 \\ x + |y| = 5 \end{cases}$$

**Problem 13**: For what values of $a$ does the system have infinitely many solutions?
$$\begin{cases} ax + 2y = a + 2 \\ x + (a-1)y = 2 \end{cases}$$

**Problem 14**: Solve:
$$\begin{cases} x + y + z = 6 \\ x - y + z = 2 \\ x + y - z = 4 \end{cases}$$

**Problem 15**: Find all integer solutions to $x^2 - y^2 = 24$.

---

## 5.4 Competition Level

**Problem 16**: Find the number of positive integer solutions to $5x + 7y = 1000$.

**Problem 17**: If $a + b + c = 0$, find the value of:
$$\frac{a^2}{a^2 - bc} + \frac{b^2}{b^2 - ca} + \frac{c^2}{c^2 - ab}$$

**Problem 18**: Solve:
$$\begin{cases} x + y = xy \\ x + z = 2xz \\ y + z = 3yz \end{cases}$$

**Problem 19**: Find all positive integers $n$ such that $n^2 + 1$ is divisible by $n + 1$.

**Problem 20**: (National Competition) Find all integer solutions to:
$$xy + 3x - 5y = -3$$

---

# Part VI: Answer Key

## Basic Level Answers

**1.** $x = 5, y = 2$

**2.** $x = 2, y = 4$

**3.** $x = 3, y = 2$

**4.** The numbers are 12 and 8.

**5.** Father: 42 years, Son: 18 years.

## Intermediate Level Answers

**6.** $x = 2, y = 1$

**7.** $(4, 3), (9, 1)$

**8.** $k = 3$

**9.** $x = 3, y = 1$

**10.** $x^2 + y^2 = 13$

## Advanced Level Answers

**11.** $(13, 156), (14, 84), (15, 60), (16, 48), (18, 36), (20, 30), (21, 28), (24, 24)$ and symmetric pairs

**12.** All points on the segment $x + y = 5$ where $x \geq 0, y \geq 0$

**13.** $a = 2$

**14.** $x = 3, y = 2, z = 1$

**15.** $(x, y) = (\pm 5, \pm 1), (\pm 7, \pm 5), (\pm 13, \pm 11)$

## Competition Level Answers

**16.** 29 solutions

**17.** 1

**18.** $(x, y, z) = (0, 0, 0)$ or $(2, 2, 1)$

**19.** $n = 1$

**20.** Using Simon's Favorite Factoring Trick: $(x-5)(y+3) = -18$
Solutions: $(x, y) = (6, 15), (4, -21), (7, 6), (3, -12), (11, 0), (-1, -6), (23, -2), (-13, -4)$

---

# Part VII: Summary and Key Formulas

## Key Concepts

1. **Linear equation in two variables**: $ax + by = c$ where $a, b \neq 0$
2. **System of equations**: Two equations with the same unknowns
3. **Solution**: Values satisfying all equations simultaneously

## Solution Methods

| Method | When to Use |
|--------|-------------|
| Substitution | Coefficient of 1 or -1 present |
| Elimination | Equal or opposite coefficients |
| Cramer's Rule | General systems, competition |

## Key Formulas

**Cramer's Rule**:
$$x = \frac{c_1b_2 - c_2b_1}{a_1b_2 - a_2b_1}, \quad y = \frac{a_1c_2 - a_2c_1}{a_1b_2 - a_2b_1}$$

**Diophantine Equation** $ax + by = c$:
- Has integer solutions iff $\gcd(a,b) | c$
- General solution: $x = x_0 + \frac{b}{d}t$, $y = y_0 - \frac{a}{d}t$

**Frobenius Number** (for coprime $a, b$):
$$g(a,b) = ab - a - b$$

## Competition Tips

1. Look for patterns before computing
2. Use algebraic identities to simplify
3. Factor when possible
4. Check for integer constraints early

5. Verify solutions in original equations
6. Consider geometric interpretations

---

## Study Path

```
Beginner → Basic systems → Word problems →
Integer solutions → Parameters → Competition techniques
```

---

**End of Chapter 10**
