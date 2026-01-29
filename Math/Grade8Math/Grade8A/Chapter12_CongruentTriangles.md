# Chapter 12: Congruent Triangles
## From Beginner to Competition Level

---

## Table of Contents

| Part | Level | Content |
|------|-------|---------|
| I | Beginner | What is congruence? Basic concepts and properties |
| II | Beginner | The five congruence criteria (SSS, SAS, ASA, AAS, HL) |
| III | Intermediate | How to write congruence proofs |
| IV | Intermediate | Properties and criteria of angle bisectors |
| V | Advanced | Common auxiliary lines and proof techniques |
| VI | Competition | Competition-level problems with detailed solutions |
| VII | Practice | 30 problems organized by difficulty level |
| VIII | Answers | Complete answer key |
| IX | Summary | Key theorems and study path |
| X | Reference | Essential formulas quick reference table |

---

# Part I: Foundations (Beginner Level)

## 1.1 Introduction: What is Congruence?

### 1.1.1 Congruence in Daily Life

Look around you. Many objects come in identical pairs or sets:
- Two keys made from the same mold
- Two coins of the same denomination
- Stamps of the same design
- Tiles on a floor
- Pages printed from the same template

These objects share a special property: they have **exactly the same shape and size**. In mathematics, we call such figures **congruent**.

### 1.1.2 Definition of Congruent Figures

> **Definition**: Two figures are **congruent** if one can be moved (by sliding, rotating, or flipping) to coincide exactly with the other.

**Key insight**: Congruent figures are identical in every way - same shape, same size, same measurements.

**Notation**: We use the symbol $\cong$ to denote congruence.

### 1.1.3 Why Study Congruent Triangles?

Triangles are special among all polygons because of their **rigidity**:
- Once you fix the three sides of a triangle, its shape is completely determined
- This is why triangles are used in bridges, buildings, and other structures

Understanding congruent triangles allows us to:
1. Prove that two line segments are equal
2. Prove that two angles are equal
3. Solve real-world measurement problems
4. Build a foundation for more advanced geometry

---

## 1.2 Congruent Triangles: Definition and Notation

### 1.2.1 Definition

> **Definition**: Two triangles are **congruent** if all six corresponding parts (three pairs of sides and three pairs of angles) are equal.

In simpler terms: congruent triangles are triangles that are exactly the same.

### 1.2.2 Notation and Corresponding Parts

We write: $\triangle ABC \cong \triangle DEF$

**Important**: The order of vertices tells us which parts correspond!

From $\triangle ABC \cong \triangle DEF$, we know:

| Corresponding Vertices | Corresponding Sides | Corresponding Angles |
|----------------------|--------------------|--------------------|
| A $\leftrightarrow$ D | $AB = DE$ | $\angle A = \angle D$ |
| B $\leftrightarrow$ E | $BC = EF$ | $\angle B = \angle E$ |
| C $\leftrightarrow$ F | $CA = FD$ | $\angle C = \angle F$ |

### 1.2.3 The CPCTC Principle

> **CPCTC**: **C**orresponding **P**arts of **C**ongruent **T**riangles are **C**ongruent.

This principle is the key to using congruent triangles in proofs:
1. First, prove two triangles are congruent
2. Then, conclude that any pair of corresponding parts are equal

**Example**: If we prove $\triangle ABC \cong \triangle DEF$, we can immediately conclude:
- $AB = DE$, $BC = EF$, $CA = FD$ (corresponding sides)
- $\angle A = \angle D$, $\angle B = \angle E$, $\angle C = \angle F$ (corresponding angles)

---

## 1.3 Finding Corresponding Parts

### 1.3.1 Method 1: Read from the Congruence Statement

The easiest method! If told $\triangle ABC \cong \triangle PQR$:
- First letters correspond: A $\leftrightarrow$ P
- Second letters correspond: B $\leftrightarrow$ Q
- Third letters correspond: C $\leftrightarrow$ R

### 1.3.2 Method 2: Match by Position in the Figure

When looking at a diagram:
- Vertices in similar positions correspond
- Look for marked equal sides or angles
- Identify shared (common) elements

### 1.3.3 Method 3: Match by Measurements

- The longest side corresponds to the longest side
- The shortest side corresponds to the shortest side
- The largest angle corresponds to the largest angle
- Equal sides are opposite to equal angles

### 1.3.4 Example

**Problem**: Given $\triangle ABC \cong \triangle DEF$ with $AB = 5$, $BC = 7$, $CA = 6$, and $\angle A = 50°$. Find $DE$, $EF$, $FD$, and $\angle D$.

**Solution**:
From the correspondence A $\leftrightarrow$ D, B $\leftrightarrow$ E, C $\leftrightarrow$ F:
- $DE = AB = 5$
- $EF = BC = 7$
- $FD = CA = 6$
- $\angle D = \angle A = 50°$

---

## 1.4 Properties of Congruent Triangles

### 1.4.1 Three Fundamental Properties

**Property 1 (Reflexive)**: Every triangle is congruent to itself.
$$\triangle ABC \cong \triangle ABC$$

**Property 2 (Symmetric)**: If $\triangle ABC \cong \triangle DEF$, then $\triangle DEF \cong \triangle ABC$.

**Property 3 (Transitive)**: If $\triangle ABC \cong \triangle DEF$ and $\triangle DEF \cong \triangle GHI$, then $\triangle ABC \cong \triangle GHI$.

### 1.4.2 Why These Properties Matter

- **Reflexive**: Useful when a triangle appears in two different contexts
- **Symmetric**: We can write the congruence in either order
- **Transitive**: We can chain congruences together

---

# Part II: Congruence Criteria (Beginner Level)

## 2.1 The Big Question

### 2.1.1 Do We Need All Six Parts?

By definition, congruent triangles have six pairs of equal parts. But do we really need to check all six to prove congruence?

**The answer is NO!** We can prove congruence with just **three** carefully chosen conditions.

### 2.1.2 Which Three Conditions Work?

Not just any three conditions will do. Through mathematical reasoning, we find that certain combinations of three conditions are sufficient:

| Criterion | Conditions | Works? |
|-----------|------------|--------|
| SSS | Three sides | Yes |
| SAS | Two sides + included angle | Yes |
| ASA | Two angles + included side | Yes |
| AAS | Two angles + non-included side | Yes |
| HL | Hypotenuse + leg (right triangles) | Yes |
| SSA | Two sides + non-included angle | **No!** |
| AAA | Three angles | **No!** |

Let's explore each criterion in detail.

---

## 2.2 SSS (Side-Side-Side)

### 2.2.1 The Theorem

> **SSS Criterion**: If three sides of one triangle are equal to three sides of another triangle, then the two triangles are congruent.

### 2.2.2 Why SSS Works

**Intuitive explanation**: Triangles are rigid. If you have three sticks of fixed lengths, there is only one way to connect them into a triangle (ignoring reflections).

**Formal reasoning**: Given three side lengths satisfying the triangle inequality, the triangle is uniquely determined. This is because:
- The three sides fix all three angles (by the Law of Cosines)
- Therefore, all six parts are determined

### 2.2.3 How to Use SSS

**Standard format**:
```
In △ABC and △DEF:
  AB = DE (given/reason)
  BC = EF (given/reason)
  CA = FD (given/reason)
∴ △ABC ≅ △DEF (SSS)
```

### 2.2.4 Example 1

**Problem**: In quadrilateral $ABCD$, $AB = CD$ and $AD = CB$. Prove that $\angle A = \angle C$.

**Solution**:

Draw diagonal $BD$.

In $\triangle ABD$ and $\triangle CDB$:
- $AB = CD$ (given)
- $AD = CB$ (given)
- $BD = DB$ (common side)

$\therefore \triangle ABD \cong \triangle CDB$ (SSS)

$\therefore \angle A = \angle C$ (CPCTC)

**Key technique**: We used the **common side** $BD$ as the third pair of equal sides.

---

## 2.3 SAS (Side-Angle-Side)

### 2.3.1 The Theorem

> **SAS Criterion**: If two sides and the **included angle** of one triangle are equal to two sides and the included angle of another triangle, then the two triangles are congruent.

### 2.3.2 Critical Warning

**The angle MUST be the included angle!**

The included angle is the angle **between** the two sides. If the angle is not between the two sides, SAS does not apply!

```
✓ Correct: AB = DE, ∠A = ∠D, AC = DF  (∠A is between AB and AC)
✗ Wrong:  AB = DE, ∠B = ∠E, AC = DF  (∠B is NOT between AB and AC)
```

### 2.3.3 Why SAS Works

If two sides and the included angle are fixed:
- The positions of two vertices relative to the third are determined
- The third side is therefore determined (connecting those two vertices)
- All three sides are now known, so by SSS, the triangle is unique

### 2.3.4 How to Use SAS

**Standard format**:
```
In △ABC and △DEF:
  AB = DE (given/reason)
  ∠A = ∠D (given/reason)      ← This angle is between AB and AC
  AC = DF (given/reason)
∴ △ABC ≅ △DEF (SAS)
```

### 2.3.5 Example 2

**Problem**: In $\triangle ABC$, $AD$ bisects $\angle BAC$, and $BD = CD$. Prove that $AB = AC$.

**Solution**:

In $\triangle ABD$ and $\triangle ACD$:
- $BD = CD$ (given)
- $\angle BAD = \angle CAD$ (AD bisects $\angle BAC$)
- $AD = AD$ (common side)

$\therefore \triangle ABD \cong \triangle ACD$ (SAS)

$\therefore AB = AC$ (CPCTC)

---

## 2.4 ASA (Angle-Side-Angle)

### 2.4.1 The Theorem

> **ASA Criterion**: If two angles and the **included side** of one triangle are equal to two angles and the included side of another triangle, then the two triangles are congruent.

### 2.4.2 Why ASA Works

If two angles and the included side are fixed:
- The third angle is determined (angle sum = 180°)
- The directions of the other two sides are fixed by the angles
- The lengths of those sides are determined by where they intersect
- Therefore, the triangle is unique

### 2.4.3 How to Use ASA

**Standard format**:
```
In △ABC and △DEF:
  ∠B = ∠E (given/reason)
  BC = EF (given/reason)      ← This side is between ∠B and ∠C
  ∠C = ∠F (given/reason)
∴ △ABC ≅ △DEF (ASA)
```

### 2.4.4 Example 3

**Problem**: Given $\angle B = \angle E$, $BC = EF$, and $\angle C = \angle F$. Prove $\triangle ABC \cong \triangle DEF$.

**Solution**:

In $\triangle ABC$ and $\triangle DEF$:
- $\angle B = \angle E$ (given)
- $BC = EF$ (given)
- $\angle C = \angle F$ (given)

$\therefore \triangle ABC \cong \triangle DEF$ (ASA)

---

## 2.5 AAS (Angle-Angle-Side)

### 2.5.1 The Theorem

> **AAS Criterion**: If two angles and a **non-included side** of one triangle are equal to two angles and the corresponding non-included side of another triangle, then the two triangles are congruent.

### 2.5.2 AAS is a Consequence of ASA

If two angles are equal, the third angles must also be equal (since angle sum = 180°).

So AAS actually gives us three equal angles plus one equal side, which implies ASA.

### 2.5.3 How to Use AAS

**Standard format**:
```
In △ABC and △DEF:
  ∠A = ∠D (given/reason)
  ∠B = ∠E (given/reason)
  BC = EF (given/reason)      ← BC is opposite to ∠A
∴ △ABC ≅ △DEF (AAS)
```

### 2.5.4 Example 4

**Problem**: Given $\angle A = \angle D$, $\angle B = \angle E$, and $AC = DF$. Prove $\triangle ABC \cong \triangle DEF$.

**Solution**:

In $\triangle ABC$ and $\triangle DEF$:
- $\angle A = \angle D$ (given)
- $\angle B = \angle E$ (given)
- $AC = DF$ (given)

$\therefore \triangle ABC \cong \triangle DEF$ (AAS)

---

## 2.6 HL (Hypotenuse-Leg)

### 2.6.1 The Theorem

> **HL Criterion**: If the **hypotenuse** and one **leg** of a right triangle are equal to the hypotenuse and one leg of another right triangle, then the two triangles are congruent.

### 2.6.2 Important Restriction

**HL only works for right triangles!**

Do not use HL for triangles that are not right triangles.

### 2.6.3 Why HL Works

In a right triangle, if the hypotenuse $c$ and one leg $a$ are known:
- The other leg is determined: $b = \sqrt{c^2 - a^2}$ (Pythagorean theorem)
- All three sides are now known
- By SSS, the triangle is unique

### 2.6.4 How to Use HL

**Standard format**:
```
In Rt△ABC and Rt△DEF (∠C = ∠F = 90°):
  AB = DE (hypotenuse)
  AC = DF (leg)
∴ Rt△ABC ≅ Rt△DEF (HL)
```

### 2.6.5 Example 5

**Problem**: In $\triangle ABC$, $BD \perp AC$ at $D$, $CE \perp AB$ at $E$, and $BD = CE$. Prove that $AB = AC$.

**Solution**:

Since $BD \perp AC$ and $CE \perp AB$:
- $\angle BDC = \angle CEB = 90°$

In $\text{Rt}\triangle BDC$ and $\text{Rt}\triangle CEB$:
- $BC = CB$ (common side, hypotenuse)
- $BD = CE$ (given, legs)

$\therefore \text{Rt}\triangle BDC \cong \text{Rt}\triangle CEB$ (HL)

$\therefore \angle DCB = \angle EBC$ (CPCTC)

$\therefore AB = AC$ (sides opposite equal angles)

---

## 2.7 Why SSA and AAA Don't Work

### 2.7.1 The SSA Trap

**SSA (Side-Side-Angle)**: Two sides and a non-included angle.

**Why it fails**: Given two sides and an angle opposite one of them, there may be **two different triangles** possible (the "ambiguous case").

**Counterexample**: Consider $AB = 5$, $BC = 4$, $\angle A = 30°$.

Two different triangles can satisfy these conditions - one acute and one obtuse.

### 2.7.2 The AAA Trap

**AAA (Angle-Angle-Angle)**: Three angles.

**Why it fails**: Triangles with the same angles can have **different sizes**. They are **similar** but not necessarily **congruent**.

**Counterexample**: A small equilateral triangle and a large equilateral triangle both have angles 60°, 60°, 60°, but they are not congruent.

### 2.7.3 Summary Table

| Criterion | Valid? | Reason |
|-----------|--------|--------|
| SSS | Yes | Three sides determine a unique triangle |
| SAS | Yes | Two sides + included angle determine the third side |
| ASA | Yes | Two angles + included side determine the triangle |
| AAS | Yes | Equivalent to ASA (third angle is determined) |
| HL | Yes | For right triangles, Pythagorean theorem determines third side |
| SSA | **No** | Ambiguous case - may give two triangles |
| AAA | **No** | Only determines shape, not size (similarity) |

---

# Part III: Writing Congruence Proofs (Intermediate Level)

## 3.1 The Structure of a Proof

### 3.1.1 Standard Format

A well-written congruence proof has this structure:

```
In △___ and △___:
  [First condition] (reason)
  [Second condition] (reason)
  [Third condition] (reason)
∴ △___ ≅ △___ (criterion)
∴ [conclusion] (CPCTC)
```

### 3.1.2 Example of Good Format

**Problem**: Given $AB = DE$, $\angle B = \angle E$, $BC = EF$. Prove $AC = DF$.

**Solution**:

In $\triangle ABC$ and $\triangle DEF$:
- $AB = DE$ (given)
- $\angle B = \angle E$ (given)
- $BC = EF$ (given)

$\therefore \triangle ABC \cong \triangle DEF$ (SAS)

$\therefore AC = DF$ (CPCTC)

### 3.1.3 Common Reasons in Proofs

| Reason | When to Use |
|--------|-------------|
| given | Information stated in the problem |
| common side | Same segment appears in both triangles |
| vertical angles | Angles formed by intersecting lines |
| alternate interior angles | With parallel lines |
| definition of midpoint | Point divides segment into two equal parts |
| definition of angle bisector | Ray divides angle into two equal parts |
| definition of perpendicular | Lines meet at 90° |

---

## 3.2 Finding the Third Condition

### 3.2.1 The Challenge

Often, a problem gives you two conditions directly. Your job is to find the third condition to complete the proof.

### 3.2.2 Common Sources of the Third Condition

**1. Common Side**
```
If triangles share a side, that side equals itself.
BD = BD (common side)
```

**2. Vertical Angles**
```
When two lines intersect:
∠AEB = ∠CED (vertical angles)
```

**3. Common Angle**
```
If triangles share an angle:
∠A = ∠A (common angle)
```

**4. Supplementary/Complementary Relationships**
```
If ∠1 + ∠3 = 90° and ∠2 + ∠3 = 90°:
Then ∠1 = ∠2
```

## 3.3 Choosing the Right Criterion

### 3.3.1 Decision Flowchart

```
Start
  ↓
Is there a right angle? ─Yes→ Do you have hypotenuse + leg? ─Yes→ Use HL
  │                                      │
  No                                     No
  ↓                                      ↓
Do you have 3 sides? ─Yes→ Use SSS      Check other criteria
  │
  No
  ↓
Do you have 2 sides + included angle? ─Yes→ Use SAS
  │
  No
  ↓
Do you have 2 angles + included side? ─Yes→ Use ASA
  │
  No
  ↓
Do you have 2 angles + any side? ─Yes→ Use AAS
```

### 3.3.2 Quick Selection Guide

| What You Have | Criterion to Use |
|---------------|------------------|
| 3 sides | SSS |
| 2 sides + angle between them | SAS |
| 2 angles + side between them | ASA |
| 2 angles + any side | AAS |
| Right triangle + hypotenuse + leg | HL |

---

# Part IV: Angle Bisector Properties (Intermediate Level)

## 4.1 Definition Review

> **Definition**: An angle bisector is a ray that divides an angle into two equal parts.

If ray $OC$ bisects $\angle AOB$, then $\angle AOC = \angle COB = \frac{1}{2}\angle AOB$.

---

## 4.2 The Angle Bisector Property Theorem

### 4.2.1 Statement

> **Theorem**: A point on the angle bisector is equidistant from the two sides of the angle.

**In symbols**: If $P$ is on the bisector of $\angle AOB$, and $PD \perp OA$, $PE \perp OB$, then $PD = PE$.

### 4.2.2 Proof

**Given**: $OC$ bisects $\angle AOB$, $P$ is on $OC$, $PD \perp OA$ at $D$, $PE \perp OB$ at $E$.

**Prove**: $PD = PE$

**Proof**:

In $\triangle OPD$ and $\triangle OPE$:
- $\angle PDO = \angle PEO = 90°$ (given)
- $\angle POD = \angle POE$ (definition of angle bisector)
- $OP = OP$ (common side)

$\therefore \triangle OPD \cong \triangle OPE$ (AAS)

$\therefore PD = PE$ (CPCTC)

## 4.3 The Converse Theorem

### 4.3.1 Statement

> **Theorem**: A point equidistant from the two sides of an angle lies on the angle bisector.

**In symbols**: If $P$ is inside $\angle AOB$, $PD \perp OA$, $PE \perp OB$, and $PD = PE$, then $P$ lies on the bisector of $\angle AOB$.

### 4.3.2 Proof

**Given**: $P$ is inside $\angle AOB$, $PD \perp OA$ at $D$, $PE \perp OB$ at $E$, $PD = PE$.

**Prove**: $P$ lies on the bisector of $\angle AOB$.

**Proof**:

Draw $OP$.

In $\text{Rt}\triangle OPD$ and $\text{Rt}\triangle OPE$:
- $PD = PE$ (given)
- $OP = OP$ (common side)

$\therefore \text{Rt}\triangle OPD \cong \text{Rt}\triangle OPE$ (HL)

$\therefore \angle POD = \angle POE$ (CPCTC)

$\therefore OP$ bisects $\angle AOB$

$\therefore P$ lies on the bisector of $\angle AOB$

## 4.4 Applications

### 4.4.1 Example 6

**Problem**: $OC$ bisects $\angle AOB$, $P$ is on $OC$, $PD \perp OA$ at $D$, $PE \perp OB$ at $E$. If $PD = 3$ cm, find $PE$.

**Solution**:

Since $OC$ bisects $\angle AOB$ and $P$ is on $OC$:

By the angle bisector property theorem: $PE = PD = 3$ cm.

### 4.4.2 Example 7

**Problem**: In $\triangle ABC$, $AD$ bisects $\angle BAC$, $DE \perp AB$ at $E$, $DF \perp AC$ at $F$. Given $DE = 4$ cm, find $DF$.

**Solution**:

Since $AD$ bisects $\angle BAC$ and $D$ is on $AD$:

By the angle bisector property theorem: $DF = DE = 4$ cm.

---

# Part V: Auxiliary Lines and Proof Techniques (Advanced Level)

## 5.1 Common Auxiliary Lines

### 5.1.1 Connecting Two Points

**When to use**: To create triangles that share a common side.

**Example**: In a quadrilateral, draw a diagonal to create two triangles.

### 5.1.2 Extending a Line Segment

**When to use**: To create vertical angles or use properties of parallel lines.

### 5.1.3 Drawing a Perpendicular

**When to use**: To create right angles, especially when using HL criterion.

### 5.1.4 Drawing a Parallel Line

**When to use**: To create equal angles (alternate interior, corresponding).

## 5.2 Proof Strategies

### 5.2.1 Working Backwards

Start from what you want to prove, and ask: "What would I need to prove this?"

**Example**: To prove $AB = CD$:
- I need two congruent triangles containing $AB$ and $CD$
- Which triangles? $\triangle ABX$ and $\triangle CDY$ for some points $X$, $Y$
- What conditions do I have to prove them congruent?

### 5.2.2 Looking for Hidden Triangles

Sometimes the triangles you need are not obvious. Look for:
- Overlapping triangles
- Triangles formed by auxiliary lines
- Triangles sharing a common vertex

## 5.3 Classic Examples

### 5.3.1 Example 8: Using Common Side

**Problem**: In quadrilateral $ABCD$, $AB = CD$ and $AD = CB$. Prove $\angle A = \angle C$.

**Solution**:

Draw diagonal $BD$.

In $\triangle ABD$ and $\triangle CDB$:
- $AB = CD$ (given)
- $AD = CB$ (given)
- $BD = DB$ (common side)

$\therefore \triangle ABD \cong \triangle CDB$ (SSS)

$\therefore \angle A = \angle C$ (CPCTC)

### 5.3.2 Example 9: Using Vertical Angles

**Problem**: Lines $AC$ and $BD$ intersect at $E$. Given $AE = CE$ and $BE = DE$. Prove $AB = CD$.

**Solution**:

In $\triangle ABE$ and $\triangle CDE$:
- $AE = CE$ (given)
- $\angle AEB = \angle CED$ (vertical angles)
- $BE = DE$ (given)

$\therefore \triangle ABE \cong \triangle CDE$ (SAS)

$\therefore AB = CD$ (CPCTC)

---

# Part VI: Competition Level Problems

## 6.1 Problem 1: Isosceles Triangle with Perpendiculars

**Problem**: In $\triangle ABC$, $AB = AC$. Point $D$ is on $BC$. $DE \perp AB$ at $E$, $DF \perp AC$ at $F$. Prove $DE = DF$.

**Solution**:

Since $AB = AC$, we have $\angle B = \angle C$ (base angles of isosceles triangle).

Since $DE \perp AB$ and $DF \perp AC$:
- $\angle DEB = \angle DFC = 90°$

In $\triangle DEB$ and $\triangle DFC$:
- $\angle DEB = \angle DFC = 90°$ (proven)
- $\angle B = \angle C$ (proven)

$\therefore \angle BDE = \angle CDF$ (angle sum in triangle)

In $\triangle DEB$ and $\triangle DFC$:
- $\angle DEB = \angle DFC$ (proven)
- $\angle B = \angle C$ (proven)
- $BD = CD$? No, this is not given.

**Alternative approach**: Use AAS with the angles we have.

Actually, we need to reconsider. Let's use a different pair of triangles.

In $\text{Rt}\triangle DEB$ and $\text{Rt}\triangle DFC$:
- $\angle B = \angle C$ (base angles)
- $\angle DEB = \angle DFC = 90°$

We need a side. Notice that we don't have $BD = CD$ given.

**Correct approach**:

In $\triangle DEB$ and $\triangle DFC$:
- $\angle DEB = \angle DFC = 90°$
- $\angle B = \angle C$
- $\therefore \angle EDB = \angle FDC$

But we still need a side to prove congruence.

**Key insight**: We need to find equal sides. Since we only have angle information, we cannot directly prove $DE = DF$ without additional conditions.

**If $D$ is the midpoint of $BC$**: Then $BD = CD$, and we can use AAS.

## 6.2 Problem 2: Measuring Across a River

**Problem**: To measure the distance $AB$ across a river, we stand at point $B$ on one bank and:
1. Draw $BF \perp AB$
2. Take points $C$, $D$ on $BF$ such that $CD = BC$
3. Draw $DE \perp BF$ such that $A$, $C$, $E$ are collinear

Prove that $DE = AB$.

**Solution**:

Since $AB \perp BF$ and $DE \perp BF$:
- $\angle ABC = \angle EDC = 90°$

In $\triangle ABC$ and $\triangle EDC$:
- $\angle ABC = \angle EDC = 90°$ (construction)
- $BC = DC$ (construction)
- $\angle ACB = \angle ECD$ (vertical angles)

$\therefore \triangle ABC \cong \triangle EDC$ (ASA)

$\therefore AB = ED$ (CPCTC)

**Practical significance**: This method allows us to measure distances that cannot be measured directly.

## 6.3 Problem 3: Angle Bisector Application

**Problem**: In $\triangle ABC$, $AD$ bisects $\angle BAC$, $DE \perp AB$ at $E$, $DF \perp AC$ at $F$. Prove that $\triangle AEF$ is isosceles.

**Solution**:

Since $AD$ bisects $\angle BAC$, and $DE \perp AB$, $DF \perp AC$:

By the angle bisector property: $DE = DF$

In $\text{Rt}\triangle AED$ and $\text{Rt}\triangle AFD$:
- $DE = DF$ (proven)
- $AD = AD$ (common side)

$\therefore \text{Rt}\triangle AED \cong \text{Rt}\triangle AFD$ (HL)

$\therefore AE = AF$ (CPCTC)

$\therefore \triangle AEF$ is isosceles.

## 6.4 Problem 4: Classic Competition Problem

**Problem**: In $\triangle ABC$, $\angle BAC = 90°$, $AB = AC$. $D$ is a point on line $BC$, $BD \perp DE$ at $D$, $CE \perp DE$ at $E$. Prove $DE = BD + CE$.

**Solution**:

Since $\angle BAC = 90°$ and $AB = AC$:
- $\angle ABC = \angle ACB = 45°$

Since $BD \perp DE$ and $CE \perp DE$:
- $\angle BDA = 90°$ and $\angle CEA = 90°$

In $\triangle ABD$:
- $\angle ABD = 45°$, $\angle ADB = 90°$
- $\therefore \angle BAD = 45°$

In $\triangle ACE$:
- $\angle ACE = 45°$, $\angle AEC = 90°$
- $\therefore \angle CAE = 45°$

Now consider $\triangle ABD$ and $\triangle CAE$:
- $\angle ABD = \angle CAE = 45°$
- $\angle ADB = \angle CEA = 90°$
- $AB = CA$ (given)

$\therefore \triangle ABD \cong \triangle CAE$ (AAS)

$\therefore BD = AE$ and $AD = CE$ (CPCTC)

$\therefore DE = DA + AE = CE + BD$

---

# Part VII: Practice Problems

## 7.1 Basic Level (1-10)

**Problem 1**: Given $\triangle ABC \cong \triangle DEF$, $AB = 5$, $BC = 6$, $CA = 7$. Find $DE$, $EF$, $FD$.

**Problem 2**: Given $\triangle ABC \cong \triangle DEF$, $\angle A = 50°$, $\angle B = 60°$. Find $\angle D$, $\angle E$, $\angle F$.

**Problem 3**: Which criterion would you use? $AB = DE$, $BC = EF$, $CA = FD$.

**Problem 4**: Which criterion would you use? $AB = DE$, $\angle A = \angle D$, $AC = DF$.

**Problem 5**: Which criterion would you use? $\angle A = \angle D$, $AB = DE$, $\angle B = \angle E$.

**Problem 6**: In $\text{Rt}\triangle ABC$ and $\text{Rt}\triangle DEF$, $\angle C = \angle F = 90°$, $AB = DE$, $AC = DF$. Are they congruent? Why?

**Problem 7**: $OC$ bisects $\angle AOB$, $P$ is on $OC$, $PD \perp OA$, $PE \perp OB$. If $PD = 5$, find $PE$.

**Problem 8**: Can we use SSA to prove congruence? Explain.

**Problem 9**: Can we use AAA to prove congruence? Explain.

**Problem 10**: State the CPCTC principle.

## 7.2 Intermediate Level (11-20)

**Problem 11**: In $\triangle ABC$ and $\triangle DEF$, $AB = DE$, $\angle B = \angle E$, $BC = EF$. Prove $\triangle ABC \cong \triangle DEF$.

**Problem 12**: In quadrilateral $ABCD$, $AB = CD$, $BC = DA$. Prove $\angle B = \angle D$.

**Problem 13**: $AD$ bisects $\angle BAC$, $DE \perp AB$ at $E$, $DF \perp AC$ at $F$. Prove $DE = DF$.

**Problem 14**: In $\triangle ABC$, $AB = AC$, $D$ is the midpoint of $BC$. Prove $AD \perp BC$.

**Problem 15**: Given $\angle 1 = \angle 2$, $\angle 3 = \angle 4$, $AC = BD$. Prove $\triangle ABC \cong \triangle BAD$.

**Problem 16**: In $\text{Rt}\triangle ABC$, $\angle C = 90°$, $CD \perp AB$ at $D$. If $AC = 3$, $BC = 4$, find $CD$.

**Problem 17**: In $\triangle ABC$, $\angle B = \angle C$. Prove $AB = AC$.

**Problem 18**: $P$ is inside $\angle AOB$, $PD \perp OA$, $PE \perp OB$, $PD = PE$. Prove $P$ is on the bisector of $\angle AOB$.

**Problem 19**: In $\triangle ABC$, $AB = AC$, $BD \perp AC$ at $D$, $CE \perp AB$ at $E$. Prove $BD = CE$.

**Problem 20**: Lines $AC$ and $BD$ intersect at $O$. $OA = OC$, $OB = OD$. Prove $AB \parallel CD$.

## 7.3 Advanced Level (21-30)

**Problem 21**: In $\triangle ABC$, $AD$ bisects $\angle BAC$, $E$ is on $BC$, $EF \parallel AC$ meets $AD$ at $F$. Prove $AE = EF$.

**Problem 22**: In $\triangle ABC$, $\angle ACB = 90°$, $CD \perp AB$ at $D$. Prove $CD^2 = AD \cdot DB$.

**Problem 23**: In $\triangle ABC$, $AB = AC$, $D$ is on $BC$, $E$ is on $AC$, $F$ is on $AB$, $DE \perp AC$, $DF \perp AB$. If $DE = DF$, prove $D$ is the midpoint of $BC$.

**Problem 24**: In $\triangle ABC$, $\angle BAC = 90°$, $AB = AC$, $D$ is on $BC$, $DE \perp AB$ at $E$, $DF \perp AC$ at $F$. Prove $EF = DE + DF$.

**Problem 25**: In $\triangle ABC$, $AD$ is the angle bisector, $AB > AC$. Prove $BD > CD$.

**Problem 26**: In quadrilateral $ABCD$, $AB = CD$, $AD = BC$, $AC$ and $BD$ intersect at $O$. Prove $OA = OC$ and $OB = OD$.

**Problem 27**: In $\triangle ABC$, $M$ is the midpoint of $BC$, $MD \perp AB$ at $D$, $ME \perp AC$ at $E$. If $AB = AC$, prove $MD = ME$.

**Problem 28**: In $\triangle ABC$, $\angle B = 2\angle C$, $AD$ bisects $\angle BAC$ and meets $BC$ at $D$. Prove $AB + BD = AC$.

**Problem 29**: In $\triangle ABC$, $AB = AC$, $\angle A = 20°$. $D$ is on $AC$ such that $BD$ bisects $\angle ABC$. Find $\angle BDC$.

**Problem 30**: In $\triangle ABC$, $\angle BAC = 90°$, $AB = AC$, $D$ and $E$ are on $BC$, $\angle DAE = 45°$. Prove $DE^2 = BD^2 + CE^2$.

---

# Part VIII: Answer Key

## Basic Level Answers

**1.** $DE = 5$, $EF = 6$, $FD = 7$

**2.** $\angle D = 50°$, $\angle E = 60°$, $\angle F = 70°$

**3.** SSS

**4.** SAS

**5.** ASA

**6.** Yes, by HL (hypotenuse-leg for right triangles)

**7.** $PE = 5$ (angle bisector property)

**8.** No, SSA can give two different triangles (ambiguous case)

**9.** No, AAA only proves similarity, not congruence

**10.** Corresponding Parts of Congruent Triangles are Congruent

## Intermediate Level Answers

**11.** Use SAS: $AB = DE$, $\angle B = \angle E$, $BC = EF$

**12.** Draw diagonal $BD$, use SSS to prove $\triangle ABD \cong \triangle CDB$

**13.** Direct application of angle bisector property theorem

**14.** Use SAS: $BD = CD$, $\angle ADB = \angle ADC$, $AD = AD$. Then $\angle ADB = \angle ADC = 90°$

**15.** Use ASA or AAS depending on the configuration

**16.** $CD = \frac{AC \cdot BC}{AB} = \frac{3 \times 4}{5} = \frac{12}{5} = 2.4$

**17.** Draw altitude from $A$, use AAS to prove the two smaller triangles congruent

**18.** Use HL to prove $\text{Rt}\triangle OPD \cong \text{Rt}\triangle OPE$

**19.** Use AAS: $\angle B = \angle C$, $\angle BDC = \angle CEB = 90°$, $BC = CB$

**20.** Use SAS to prove $\triangle AOB \cong \triangle COD$, then $\angle OAB = \angle OCD$ (alternate interior angles)

## Advanced Level Answers

**21.** Use ASA: $\angle EAF = \angle EFA$ (alternate interior angles), so $AE = EF$

**22.** Use similar triangles: $\triangle ACD \sim \triangle CBD$

**23.** Use congruence to show $BD = CD$

**24.** Rotate $\triangle ADF$ by 90° around $A$ to coincide with $\triangle AEB$

**25.** Construct $E$ on $AB$ such that $AE = AC$, use SAS

**26.** Use SSS twice

**27.** Use AAS: $\angle B = \angle C$, $\angle MDB = \angle MEC = 90°$

**28.** Extend $AB$ to $E$ such that $BE = BD$, prove $\triangle ADE$ isosceles

**29.** $\angle BDC = 30°$

**30.** Use rotation: rotate $\triangle ABD$ by 90° around $A$

---

# Part IX: Summary

## 9.1 Key Theorems

### Congruence Criteria

| Criterion | Full Name | Conditions | Applies To |
|-----------|-----------|------------|------------|
| SSS | Side-Side-Side | 3 sides equal | All triangles |
| SAS | Side-Angle-Side | 2 sides + included angle | All triangles |
| ASA | Angle-Side-Angle | 2 angles + included side | All triangles |
| AAS | Angle-Angle-Side | 2 angles + any side | All triangles |
| HL | Hypotenuse-Leg | Hypotenuse + leg | Right triangles only |

### Invalid Criteria

| Criterion | Why It Fails |
|-----------|--------------|
| SSA | Ambiguous case (two possible triangles) |
| AAA | Only determines shape, not size |

## 9.2 Angle Bisector Theorems

**Property**: A point on the angle bisector is equidistant from the two sides.

**Converse**: A point equidistant from the two sides lies on the angle bisector.

## 9.3 Study Path

```
Beginner:
  Definition of congruence → Corresponding parts → CPCTC principle
  ↓
Intermediate:
  SSS → SAS → ASA → AAS → HL → Why SSA/AAA fail
  ↓
Advanced:
  Writing proofs → Finding conditions → Auxiliary lines
  ↓
Competition:
  Complex proofs → Multiple congruences → Rotation/reflection
```

---

# Part X: Quick Reference

## 10.1 Congruence Criteria at a Glance

| Criterion | What You Need | Remember |
|-----------|---------------|----------|
| SSS | 3 sides | "Three sides determine a triangle" |
| SAS | 2 sides + included angle | "Angle MUST be between the sides" |
| ASA | 2 angles + included side | "Side MUST be between the angles" |
| AAS | 2 angles + any side | "Third angle is automatic" |
| HL | Right angle + hypotenuse + leg | "Only for right triangles" |

## 10.2 Common Proof Elements

| Element | How to Get It |
|---------|---------------|
| Common side | Same segment in both triangles |
| Vertical angles | Two intersecting lines |
| Alternate interior angles | Parallel lines + transversal |
| Right angles | Perpendicular lines |
| Equal angles | Angle bisector, isosceles triangle |

## 10.3 Proof Checklist

Before writing a proof, ask:
1. Which two triangles am I comparing?
2. What criterion will I use?
3. Do I have all three conditions?
4. Are my corresponding parts correct?

---

**End of Chapter 12**

*This document covers congruent triangles from beginner level through National Junior High Math League competition level.*
