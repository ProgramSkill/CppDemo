# Chapter 11: Triangles
## From Beginner to Competition Level

---

## Table of Contents

| Part | Level | Content |
|------|-------|---------|
| I | Beginner | Basic concepts, definitions, sides and angles of triangles |
| II | Intermediate | Triangle inequality, special line segments, area, midsegment |
| III | Advanced | Angle sums, congruence, similarity, Pythagorean theorem, angle bisector theorem |
| IV | Competition | 16 competition problems with detailed solutions (including league-level) |
| V | Practice | 40 problems organized by difficulty level |
| VI | Answers | Complete answer key |
| VII | Summary | Key formulas and study path |
| VIII | Techniques | Essential problem-solving techniques with clear explanations |
| IX | Real Problems | Historical competition problems with detailed solutions |
| X | Reference | Essential formulas quick reference table |

---

# Part I: Foundations (Beginner Level)

## 1.1 Introduction: What is a Triangle?

### 1.1.1 Triangles in Daily Life

Triangles are everywhere in our world:
- The shape of a set square
- The structure of pyramids
- Cable-stayed bridges
- Roof trusses in buildings
- Bicycle frames

**Why are triangles so common in structures?** Because triangles have a unique property called **stability**. We will explore this in detail later.

### 1.1.2 Definition of a Triangle

> **Definition**: A triangle is a figure formed by three line segments connecting three non-collinear points end to end.

**Key Points**:
1. **Three line segments**: A triangle has three sides
2. **Non-collinear points**: The three vertices cannot lie on the same line
3. **Connected end to end**: Each segment's endpoint connects to the next segment's starting point

### 1.1.3 Notation and Terminology

**Vertices**: The three vertices are usually denoted by capital letters A, B, C

**Sides**: The three sides of a triangle
- Side AB (or BA)
- Side BC (or CB)
- Side CA (or AC)
- Also denoted by lowercase letters: side $a$ (opposite to A), side $b$ (opposite to B), side $c$ (opposite to C)

**Angles**: The three interior angles
- $\angle A$ (or $\angle BAC$): the angle at vertex A
- $\angle B$ (or $\angle ABC$): the angle at vertex B
- $\angle C$ (or $\angle ACB$): the angle at vertex C

**Triangle Notation**: $\triangle ABC$ (read as "triangle ABC")

**Note**: The order of vertices can be rearranged. $\triangle ABC$, $\triangle ACB$, and $\triangle BAC$ all represent the same triangle.

---

## 1.2 The Triangle Inequality

### 1.2.1 Exploration Activity

Can any three sticks form a triangle when connected end to end?

**Attempt 1**: 3cm, 4cm, 5cm sticks → Can form a triangle
**Attempt 2**: 2cm, 3cm, 6cm sticks → Cannot form a triangle
**Attempt 3**: 4cm, 4cm, 8cm sticks → Cannot form a triangle

**Why can't some combinations form triangles?**

Analysis:
- Attempt 2: $2 + 3 = 5 < 6$, the two shorter sticks together are shorter than the third
- Attempt 3: $4 + 4 = 8$, the two sticks equal the third, forming a straight line

### 1.2.2 The Triangle Inequality Theorem

> **Theorem**: The sum of any two sides of a triangle is greater than the third side.

**Mathematical Expression**:
If $a$, $b$, $c$ are the three sides of a triangle, then:
$$a + b > c$$
$$a + c > b$$
$$b + c > a$$

**Understanding the Theorem**:
- This theorem reflects the fact that "the shortest distance between two points is a straight line"
- In $\triangle ABC$, going directly from A to B via side AB is shorter than going via A to C to B
- Therefore: $AC + CB > AB$

### 1.2.3 Corollary

> **Corollary**: The difference of any two sides of a triangle is less than the third side.

**Mathematical Expression**:
$$|a - b| < c$$
$$|a - c| < b$$
$$|b - c| < a$$

**Derivation**:
From $a + b > c$, we get $a > c - b$
From $a + c > b$, we get $c > b - a$
Therefore: $c - b < a < c + b$, which means $|a - b| < c$

### 1.2.4 Determining if Three Segments Can Form a Triangle

**Method 1**: Check all three combinations
- Is $a + b > c$?
- Is $a + c > b$?
- Is $b + c > a$?
If all three inequalities hold, the segments can form a triangle.

**Method 2 (Shortcut)**: Only check if the sum of the two shorter sides is greater than the longest side
- Find the longest side
- Calculate the sum of the other two sides
- If sum > longest side, they can form a triangle

**Why Method 2 Works**:
If the sum of the two shorter sides > longest side, then:
- Longest side + any other side > remaining side (since the longest side is already large enough)
- All three inequalities are automatically satisfied

### 1.2.5 Examples

**Example 1**: Can the following sets of line segments form a triangle?
(1) 3cm, 4cm, 5cm
(2) 2cm, 3cm, 6cm
(3) 4cm, 4cm, 8cm
(4) 5cm, 7cm, 10cm

**Solution**:
(1) $3 + 4 = 7 > 5$ **Can form a triangle**

(2) $2 + 3 = 5 < 6$ **Cannot form a triangle**

(3) $4 + 4 = 8 = 8$ **Cannot form a triangle** (equality doesn't count)

(4) $5 + 7 = 12 > 10$ **Can form a triangle**

**Example 2**: If two sides of a triangle have lengths 3 and 5, what is the range of the third side $x$?

**Solution**:
By the triangle inequality:
- Sum of two sides > third side: $3 + 5 > x$, so $x < 8$
- Difference of two sides < third side: $5 - 3 < x$, so $x > 2$

**Answer**: $2 < x < 8$

**Example 3**: If the three sides of a triangle are 3, 4, and $x$, find the range of $x$.

**Solution**:
When $x$ is the longest side: $3 + 4 > x$, so $x < 7$
When $x$ is not the longest side: $x + 3 > 4$, so $x > 1$

**Answer**: $1 < x < 7$

**Note**: We must consider both cases—when $x$ is the longest side and when it isn't!

**Example 4**: The three sides of a triangle are in the ratio 2:3:4, and the perimeter is 18cm. Find the lengths of the three sides.

**Solution**:
Let the three sides be $2k$, $3k$, $4k$

From the perimeter: $2k + 3k + 4k = 18$
$9k = 18$
$k = 2$

The three sides are: $2 \times 2 = 4$cm, $3 \times 2 = 6$cm, $4 \times 2 = 8$cm

**Verification**: $4 + 6 = 10 > 8$ Satisfies the triangle inequality

### 1.2.6 Common Mistakes

**Mistake 1**: Thinking only one inequality needs to be checked
- Correct approach: Check that the sum of any two sides > third side (or use the shortcut)

**Mistake 2**: Thinking "greater than or equal to" is sufficient
- Correct approach: Must be strictly greater! Equality means no triangle can be formed

**Mistake 3**: Forgetting the absolute value
- Wrong: $a - b < c$
- Correct: $|a - b| < c$

---

# Part II: Intermediate Level

## 2.1 Special Line Segments in Triangles

### 2.1.1 Altitude (Height) of a Triangle

> **Definition**: The altitude of a triangle is the perpendicular line segment from a vertex to the line containing the opposite side.

**Key Points**:
1. An altitude is a line segment, not a line
2. An altitude must be perpendicular to the opposite side (or its extension)
3. Every triangle has three altitudes (one from each vertex)

**Position of Altitudes**:
- **Acute triangle**: All three altitudes lie inside the triangle, meeting at an interior point (orthocenter)
- **Right triangle**: The two legs are altitudes; they meet at the right angle vertex (orthocenter is at the vertex)
- **Obtuse triangle**: Two altitudes lie outside the triangle, meeting at an exterior point (orthocenter is outside)

**How to Construct an Altitude**:
1. Draw a perpendicular from a vertex to the opposite side
2. The segment from the vertex to the foot of the perpendicular is the altitude

**Property of Altitudes**:
- The three altitudes (or their extensions) of a triangle meet at a single point
- This point is called the **orthocenter**

### 2.1.2 Median of a Triangle

> **Definition**: A median of a triangle is a line segment joining a vertex to the midpoint of the opposite side.

**Key Points**:
1. A median is a line segment
2. Each vertex corresponds to one median
3. Every triangle has three medians

**Properties of Medians**:
- The three medians of a triangle meet at a single point
- This point is called the **centroid**
- The centroid divides each median in the ratio 2:1 (from vertex to centroid : from centroid to midpoint = 2:1)

**How to Construct a Median**:
1. Find the midpoint of one side
2. Connect the opposite vertex to this midpoint

**Application**: A median divides a triangle into two triangles of equal area.

### 2.1.3 Angle Bisector of a Triangle

> **Definition**: An angle bisector of a triangle is the line segment from a vertex to the opposite side that bisects the interior angle at that vertex.

**Key Points**:
1. An angle bisector is a line segment (not a ray or line)
2. Each angle corresponds to one angle bisector
3. Every triangle has three angle bisectors

**Properties of Angle Bisectors**:
- The three angle bisectors of a triangle meet at a single point
- This point is called the **incenter** (center of the inscribed circle)
- The incenter is equidistant from all three sides of the triangle

**How to Construct an Angle Bisector**:
- Using a protractor: Measure the angle and draw a ray at half the angle
- Using compass and straightedge: Draw an arc centered at the vertex, then draw arcs from the intersection points, and connect the vertex to the new intersection

### 2.1.4 Comparison of the Three Special Segments

| Segment Type | Definition | Property | Point of Concurrency | Location (Acute Triangle) |
|--------------|------------|----------|---------------------|---------------------------|
| Altitude | Perpendicular from vertex to opposite side | Three altitudes meet at one point | Orthocenter | Interior |
| Median | Segment from vertex to midpoint of opposite side | Three medians meet at one point; divides median 2:1 | Centroid | Interior |
| Angle Bisector | Segment that bisects an interior angle | Three angle bisectors meet at one point | Incenter | Interior |

### 2.1.5 Examples

**Example 5**: In $\triangle ABC$, $AD$ is an altitude, $AE$ is a median, and $AF$ is an angle bisector.
(1) If $BD = 3$ and $CD = 5$, find the length of $BC$
(2) If $BE = EC = 4$, find the length of $BC$
(3) If $\angle BAF = \angle CAF = 30°$, find $\angle BAC$

**Solution**:
(1) $AD$ is an altitude, so $D$ is on $BC$
   $BC = BD + CD = 3 + 5 = 8$

(2) $AE$ is a median, so $E$ is the midpoint of $BC$
   $BC = BE + EC = 4 + 4 = 8$

(3) $AF$ is an angle bisector
   $\angle BAC = \angle BAF + \angle CAF = 30° + 30° = 60°$

**Example 6**: In $\triangle ABC$, $AB = AC = 5$ cm, $BC = 6$ cm, and $AD$ is an altitude. Find the length of $BD$.

**Solution**:
Since $AB = AC$ and $AD$ is an altitude,
$BD = CD$ (In an isosceles triangle, the altitude to the base is also the median)
Since $BC = 6$ cm,
$BD = 6 \div 2 = 3$ cm

---

## 2.2 Stability of Triangles

### 2.2.1 Discovering Stability

**Hands-on Experiment**:
1. Nail three sticks together to form a triangle
2. Nail four sticks together to form a quadrilateral
3. Push on each shape and observe which one changes more easily

**Results**:
- Triangle: Does not deform; shape is fixed
- Quadrilateral: Easily deforms

### 2.2.2 The Stability Theorem

> **Theorem**: Triangles have stability (rigidity).

**Understanding Stability**:
- Once the three side lengths of a triangle are fixed, the shape of the triangle is uniquely determined
- This is a consequence of the triangle inequality
- Quadrilaterals, pentagons, etc., do not have stability (their shapes are not uniquely determined by side lengths)

### 2.2.3 Applications of Stability

**In Architecture**:
- Roof trusses: Use triangular stability to support roofs
- Bridge cables: Form triangular structures to distribute weight
- Scaffolding: Use diagonal braces to form triangles

**In Daily Life**:
- Bicycle kickstands: Maintain stability when parked
- Folding chairs: Form triangles when opened
- Set squares: Utilize stability and right angle properties

**Contrast**: Quadrilaterals lack stability, but can be made stable by adding a diagonal (dividing the quadrilateral into two triangles).

**How to Stabilize a Quadrilateral?**
- Add a diagonal: Divides the quadrilateral into two triangles
- Reinforce the corners: Restrict angle deformation

**Example 7**: A quadrilateral wooden frame easily deforms. Add one stick to make it stable.

**Solution**: Connect any diagonal.

**Reason**: After adding a diagonal, the quadrilateral is divided into two triangles. Using the stability of triangles, the entire structure becomes stable.

---

## 2.3 Area of a Triangle

### 2.3.1 Basic Area Formula

> **Formula**: Area of a triangle = $\frac{1}{2} \times \text{base} \times \text{height}$

**Mathematical Expression**:
$$S_{\triangle ABC} = \frac{1}{2} \times a \times h_a = \frac{1}{2} \times b \times h_b = \frac{1}{2} \times c \times h_c$$

where $h_a$, $h_b$, $h_c$ are the altitudes to sides $a$, $b$, $c$ respectively.

**Key Points**:
1. The base and height must be perpendicular to each other
2. Any side can be chosen as the base
3. The height is the perpendicular distance from the opposite vertex to the base

### 2.3.2 Heron's Formula (Competition Level)

> **Heron's Formula**: For a triangle with sides $a$, $b$, $c$ and semi-perimeter $s = \frac{a+b+c}{2}$:
> $$S = \sqrt{s(s-a)(s-b)(s-c)}$$

**Example**: Find the area of a triangle with sides 3, 4, 5.

**Solution**:
$s = \frac{3+4+5}{2} = 6$

$S = \sqrt{6(6-3)(6-4)(6-5)} = \sqrt{6 \times 3 \times 2 \times 1} = \sqrt{36} = 6$

**Verification**: This is a right triangle (3-4-5), so $S = \frac{1}{2} \times 3 \times 4 = 6$ ✓

### 2.3.3 Area Relationships

**Property 1**: A median divides a triangle into two triangles of equal area.

**Property 2**: If two triangles have the same base, their areas are proportional to their heights.

**Property 3**: If two triangles have the same height, their areas are proportional to their bases.

---

## 2.4 Midsegment (Midline) of a Triangle

### 2.4.1 Definition

> **Definition**: A midsegment of a triangle is a line segment connecting the midpoints of two sides.

### 2.4.2 Midsegment Theorem

> **Theorem**: The midsegment of a triangle is parallel to the third side and equals half its length.

**Mathematical Expression**:
If $D$ and $E$ are midpoints of $AB$ and $AC$ respectively, then:
$$DE \parallel BC \quad \text{and} \quad DE = \frac{1}{2}BC$$

### 2.4.3 Proof

```
Given: In △ABC, D is the midpoint of AB, E is the midpoint of AC
Prove: DE ∥ BC and DE = ½BC

Proof: Extend DE to point F such that EF = DE
       Connect CF

       In △ADE and △CFE:
       AE = CE (E is midpoint of AC)
       ∠AED = ∠CEF (vertical angles)
       DE = FE (construction)

       Therefore △ADE ≅ △CFE (SAS)

       So AD = CF and ∠ADE = ∠CFE

       Since AD = CF and AD = BD (D is midpoint)
       We have BD = CF

       Since ∠ADE = ∠CFE (alternate interior angles)
       BD ∥ CF

       Therefore BDFC is a parallelogram
       So DF ∥ BC and DF = BC

       Since DE = ½DF
       DE = ½BC
```

### 2.4.4 Applications

**Example**: In $\triangle ABC$, $D$, $E$, $F$ are midpoints of $AB$, $BC$, $CA$ respectively. If $BC = 10$, find $DF$.

**Solution**:
$DF$ is the midsegment parallel to $BC$
$DF = \frac{1}{2}BC = \frac{1}{2} \times 10 = 5$

---

# Part III: Advanced Level

## 3.1 Interior Angle Sum of a Triangle

### 3.1.1 Exploration Activity

**Hands-on Activity**:
1. Draw any triangle on paper
2. Cut out the three angles
3. Place the three angles together and observe what happens

**Result**: The three angles form exactly a straight angle (180°)!

### 3.1.2 The Interior Angle Sum Theorem

> **Theorem**: The sum of the three interior angles of a triangle equals 180°.

**Mathematical Expression**:
In $\triangle ABC$: $\angle A + \angle B + \angle C = 180°$

### 3.1.3 Proof of the Theorem

**Method 1: Parallel Line Method**

```
Given: △ABC
Prove: ∠A + ∠B + ∠C = 180°

Proof: Draw line MN through point A parallel to BC

      Since MN ∥ BC
      ∠1 = ∠B (alternate interior angles with parallel lines)
      ∠2 = ∠C (alternate interior angles with parallel lines)

      Since ∠1 + ∠BAC + ∠2 = 180° (definition of straight angle)
      Therefore ∠B + ∠BAC + ∠C = 180°

      That is: ∠A + ∠B + ∠C = 180°
```

**Proof Strategy Analysis**:
1. To add three angles together, it's best to have them share a common vertex
2. Drawing a parallel line through one vertex allows us to "transfer" the other two angles
3. Use properties of parallel lines to establish angle relationships

**Method 2: Vertex Angle Assembly**

```
Proof: Draw line DE through point A parallel to BC

      Since DE ∥ BC
      ∠DAB = ∠B (alternate interior angles)
      ∠EAC = ∠C (alternate interior angles)

      Since ∠DAB + ∠BAC + ∠EAC = 180° (straight angle)
      Therefore ∠B + ∠BAC + ∠C = 180°
```

### 3.1.4 Corollaries of the Interior Angle Sum Theorem

**Corollary 1**: The two acute angles of a right triangle are complementary.

**Proof**:
In right triangle $\triangle ABC$ with $\angle C = 90°$
Since $\angle A + \angle B + \angle C = 180°$
$\angle A + \angle B = 180° - 90° = 90°$
Therefore $\angle A + \angle B = 90°$ (complementary)

**Corollary 2**: An exterior angle of a triangle equals the sum of the two non-adjacent interior angles.

**Corollary 3**: An exterior angle of a triangle is greater than either non-adjacent interior angle.

These corollaries are explained in detail in the next section.

### 3.1.5 Examples

**Example 8**: In $\triangle ABC$, $\angle A = 40°$ and $\angle B = 60°$. Find $\angle C$.

**Solution**:
By the interior angle sum theorem:
$\angle A + \angle B + \angle C = 180°$
$40° + 60° + \angle C = 180°$
$\angle C = 180° - 40° - 60° = 80°$

**Example 9**: In right triangle $\triangle ABC$, $\angle C = 90°$ and $\angle A = 35°$. Find $\angle B$.

**Solution**:
In $\text{Rt}\triangle ABC$, $\angle C = 90°$
By Corollary 1: The two acute angles of a right triangle are complementary
$\angle A + \angle B = 90°$
$35° + \angle B = 90°$
$\angle B = 90° - 35° = 55°$

**Example 10**: In $\triangle ABC$, $\angle A = 2\angle B$ and $\angle C = \angle A + \angle B$. Find all three angles.

**Solution**:
Since $\angle A + \angle B + \angle C = 180°$
And $\angle C = \angle A + \angle B$
Therefore $\angle C = 180° \div 2 = 90°$

Since $\angle A = 2\angle B$
And $\angle A + \angle B = 90°$
$2\angle B + \angle B = 90°$
$3\angle B = 90°$
$\angle B = 30°$

$\angle A = 2 \times 30° = 60°$

**Answer**: $\angle A = 60°$, $\angle B = 30°$, $\angle C = 90°$

**Example 11**: In $\triangle ABC$, $\angle A - \angle B = 30°$ and $\angle C = 80°$. Find $\angle A$ and $\angle B$.

**Solution**:
Since $\angle A + \angle B + \angle C = 180°$
$\angle A + \angle B = 180° - 80° = 100°$ ... ①

And $\angle A - \angle B = 30°$ ... ②

Adding ① and ②: $2\angle A = 130°$, so $\angle A = 65°$
Subtracting ② from ①: $2\angle B = 70°$, so $\angle B = 35°$

**Answer**: $\angle A = 65°$, $\angle B = 35°$

### 3.1.6 Problem-Solving Tips

**Finding the third angle given two angles**: Subtract the sum of the known angles from 180°

**Finding one acute angle in a right triangle given the other**: Subtract the known acute angle from 90°

**Finding angles given relationships**: Set up and solve a system of equations

---

## 3.2 Exterior Angles of a Triangle

### 3.2.1 Definition of Exterior Angle

> **Definition**: An exterior angle of a triangle is the angle formed by one side of the triangle and the extension of an adjacent side.

**Key Points**:
1. An exterior angle is supplementary to its adjacent interior angle
2. Each vertex has two exterior angles (which are vertical angles and therefore equal)
3. A triangle has 6 exterior angles in total (2 at each vertex)

**Relationship Between Exterior and Interior Angles**:
- Exterior angle + Adjacent interior angle = 180° (supplementary angles)
- Exterior angle = 180° - Adjacent interior angle

### 3.2.2 Exterior Angle Properties

**Property 1**: An exterior angle of a triangle equals the sum of the two non-adjacent interior angles.

**Proof**:
```
In △ABC, let ∠ACD be an exterior angle

Since ∠ACD + ∠ACB = 180° (supplementary angles)
And ∠A + ∠B + ∠ACB = 180° (interior angle sum theorem)
Therefore ∠ACD = ∠A + ∠B
```

**Property 2**: An exterior angle of a triangle is greater than either non-adjacent interior angle.

**Proof**:
```
Since ∠ACD = ∠A + ∠B (Property 1)
And ∠B > 0°
Therefore ∠ACD > ∠A

Similarly: ∠ACD > ∠B
```

### 3.2.3 Applications of Exterior Angle Properties

**Application Scenarios**:
1. Finding angle measures
2. Proving angle inequalities
3. Calculating sums of multiple exterior angles

**Problem-Solving Strategy**:
- When you see an exterior angle, immediately think: it equals the sum of the two non-adjacent interior angles
- Convert exterior angle problems to interior angle problems

### 3.2.4 Examples

**Example 12**: In $\triangle ABC$, exterior angle $\angle ACD = 100°$ and $\angle A = 40°$. Find $\angle B$.

**Solution**:
By exterior angle Property 1:
Exterior angle = sum of non-adjacent interior angles
$\angle ACD = \angle A + \angle B$
$100° = 40° + \angle B$
$\angle B = 100° - 40° = 60°$

**Example 13**: In $\triangle ABC$, exterior angle $\angle ACD = 120°$ and $\angle B = 50°$. Find $\angle A$ and $\angle C$.

**Solution**:
(1) Finding $\angle A$:
By exterior angle property: $\angle ACD = \angle A + \angle B$
$120° = \angle A + 50°$
$\angle A = 120° - 50° = 70°$

(2) Finding $\angle C$:
$\angle C = 180° - \angle ACD = 180° - 120° = 60°$

Or using interior angle sum: $\angle C = 180° - \angle A - \angle B = 180° - 70° - 50° = 60°$

**Example 14**: In $\triangle ABC$, exterior angle $\angle ACD = 110°$, $\angle B = 40°$, and $AE$ bisects $\angle CAD$. Find $\angle CAE$.

**Solution**:
(1) First find $\angle CAD$:
$\angle CAD = 180° - \angle ACD = 180° - 110° = 70°$

(2) Find $\angle CAE$:
Since $AE$ bisects $\angle CAD$
$\angle CAE = \frac{1}{2}\angle CAD = \frac{1}{2} \times 70° = 35°$

**Example 15**: In $\triangle ABC$, $\angle A = 50°$, $\angle B = 60°$, $CD$ is an extension of $BC$, and $CE$ bisects $\angle ACD$. Find $\angle ACE$.

**Solution**:
(1) First find $\angle ACB$:
$\angle ACB = 180° - \angle A - \angle B = 180° - 50° - 60° = 70°$

(2) Find $\angle ACD$:
$\angle ACD = 180° - \angle ACB = 180° - 70° = 110°$

(3) Find $\angle ACE$:
Since $CE$ bisects $\angle ACD$
$\angle ACE = \frac{1}{2}\angle ACD = \frac{1}{2} \times 110° = 55°$

### 3.2.5 Common Mistakes

**Mistake 1**: Confusing exterior angle with adjacent interior angle
- Wrong: Exterior angle + adjacent interior angle = 90°
- Correct: Exterior angle + adjacent interior angle = 180°

**Mistake 2**: Misapplying exterior angle property
- Wrong: Exterior angle = adjacent interior angle + another interior angle
- Correct: Exterior angle = sum of the two NON-ADJACENT interior angles

---

## 3.3 Classification of Triangles

### 3.3.1 Classification by Angles

**Classification Criterion**: Based on the largest angle of the triangle

| Type | Definition | Angle Characteristics |
|------|------------|----------------------|
| Acute triangle | All three angles are acute | Each angle < 90° |
| Right triangle | One angle is a right angle | One angle = 90°, other two are complementary |
| Obtuse triangle | One angle is obtuse | One angle > 90° |

**Memory Aid**:
- Acute triangle: All three angles are "sharp"
- Right triangle: One angle is exactly 90°
- Obtuse triangle: One angle is greater than 90°

**Determination Method**:
- Look at the largest angle: If largest angle < 90° → Acute
- Look at the largest angle: If largest angle = 90° → Right
- Look at the largest angle: If largest angle > 90° → Obtuse

### 3.3.2 Classification by Sides

**Classification Criterion**: Based on the equality relationships among sides

| Type | Definition | Side Characteristics | Special Properties |
|------|------------|---------------------|-------------------|
| Scalene triangle | All three sides are different | $a \neq b \neq c$ | No special properties |
| Isosceles triangle | Two sides are equal | Two sides equal | Two base angles equal; "three-in-one" property |
| Equilateral triangle | All three sides are equal | $a = b = c$ | All three angles are 60° |

**Special Terminology for Isosceles Triangles**:
- **Legs**: The two equal sides
- **Base**: The third side
- **Vertex angle**: The angle between the two legs
- **Base angles**: The two angles at the base

**Relationship Between Equilateral and Isosceles Triangles**:
- An equilateral triangle is a special case of an isosceles triangle
- An isosceles triangle is not necessarily equilateral

### 3.3.3 Relationship Between the Two Classifications

Classification by angles and by sides are independent:
- An isosceles triangle can be acute, right, or obtuse
- An equilateral triangle is always acute (all angles are 60°)

### 3.3.4 Examples

**Example 16**: The three interior angles of a triangle are in the ratio 1:2:3. Find each angle and determine the type of triangle.

**Solution**:
Let the three angles be $x$, $2x$, $3x$

Since $x + 2x + 3x = 180°$
$6x = 180°$
$x = 30°$

The three angles are: 30°, 60°, 90°

**Answer**: This is a **right triangle**.

**Example 17**: In $\triangle ABC$, $AB = AC$ and $\angle A = 50°$. Determine the type of triangle.

**Solution**:
Since $AB = AC$
$\triangle ABC$ is an **isosceles triangle**

Since $\angle A = 50° < 90°$
$\triangle ABC$ is also an **acute triangle**

**Answer**: This is an isosceles acute triangle.

**Example 18**: One angle of an isosceles triangle is 40°. Find the other two angles.

**Solution**:
Case analysis is required:

**Case 1**: The vertex angle is 40°
Base angles = $(180° - 40°) \div 2 = 70°$
**Answer**: The other two angles are both 70°.

**Case 2**: A base angle is 40°
Vertex angle = $180° - 2 \times 40° = 100°$
**Answer**: The other two angles are 40° and 100°.

**Note**: When dealing with angles in isosceles triangles, always consider whether the given angle is the vertex angle or a base angle!

---

## 3.4 Polygons and Their Angle Sums

### 3.4.1 Definition of Polygons

> **Definition**: A polygon is a closed figure in a plane formed by line segments connected end to end.

**Components**:
- **Sides**: The line segments forming the polygon
- **Vertices**: The common endpoints of adjacent sides
- **Diagonals**: Line segments connecting non-adjacent vertices
- **Interior angles**: Angles formed by adjacent sides
- **Exterior angles**: Angles formed by one side and the extension of an adjacent side

**Naming Convention**:
- Named by number of sides: triangle (3), quadrilateral (4), pentagon (5), hexagon (6), etc.
- In general, an n-gon has n sides, n vertices, and n interior angles

**Classification**:
- **Convex polygon**: All interior angles are less than 180°
- **Concave polygon**: At least one interior angle is greater than 180°

### 3.4.2 Number of Diagonals

**Question**: From one vertex of an n-gon, how many diagonals can be drawn?

**Analysis**:
- From one vertex, you cannot connect to itself (subtract 1)
- You cannot connect to the two adjacent vertices (subtract 2)
- So you can connect to $(n-3)$ vertices

**Formula**: From one vertex of an n-gon, $(n-3)$ diagonals can be drawn.

**Question**: How many diagonals does an n-gon have in total?

**Analysis**:
- Each vertex can draw $(n-3)$ diagonals
- n vertices give $n(n-3)$ diagonals
- But each diagonal is counted twice (once from each endpoint)
- So divide by 2

**Formula**: An n-gon has $\frac{n(n-3)}{2}$ diagonals.

**Verification**:
- Quadrilateral: $\frac{4(4-3)}{2} = 2$ diagonals ✓
- Pentagon: $\frac{5(5-3)}{2} = 5$ diagonals ✓
- Hexagon: $\frac{6(6-3)}{2} = 9$ diagonals ✓

### 3.4.3 Examples on Diagonals

**Example 19**: Find the number of diagonals in a decagon (10-gon).

**Solution**:
$n = 10$
Number of diagonals = $\frac{10(10-3)}{2} = \frac{10 \times 7}{2} = 35$

**Example 20**: A polygon has 14 diagonals. Find the number of sides.

**Solution**:
Let the polygon have $n$ sides

$\frac{n(n-3)}{2} = 14$
$n(n-3) = 28$
$n^2 - 3n - 28 = 0$
$(n-7)(n+4) = 0$

$n = 7$ or $n = -4$ (rejected)

**Answer**: This is a **heptagon** (7-gon).

---

### 3.4.4 Interior Angle Sum of Polygons

**Observation of Patterns**:
- Triangle interior angle sum = $180° = (3-2) \times 180°$
- Quadrilateral interior angle sum = $360° = (4-2) \times 180°$
- Pentagon interior angle sum = $540° = (5-2) \times 180°$
- Hexagon interior angle sum = $720° = (6-2) \times 180°$

**Pattern Discovered**: Interior angle sum of an n-gon = $(n-2) \times 180°$

### 3.4.5 Interior Angle Sum Theorem

> **Theorem**: The sum of the interior angles of an n-gon equals $(n-2) \cdot 180°$.

### 3.4.6 Proof of the Theorem

**Method 1: Triangle Decomposition**

```
Proof: From one vertex of an n-gon, (n-3) diagonals can be drawn
      These diagonals divide the n-gon into (n-2) triangles
      Each triangle has an interior angle sum of 180°
      Therefore the n-gon has interior angle sum (n-2)·180°
```

**Proof Strategy**:
1. Draw diagonals from one vertex
2. Divide the n-gon into triangles
3. The sum of all triangle angles equals the polygon's interior angle sum
4. Number of triangles = n - 2

**Method 2: Interior Point Method**

```
Proof: Choose any point O inside the n-gon
      Connect O to each vertex
      This divides the n-gon into n triangles
      The sum of angles in these n triangles is n·180°
      Subtract the full angle at O (360°)
      Interior angle sum = n·180° - 360° = (n-2)·180°
```

### 3.4.7 Examples on Interior Angle Sum

**Example 21**:
(1) Find the interior angle sum of a heptagon (7-gon)
(2) A polygon has an interior angle sum of 1440°. Find the number of sides.

**Solution**:
(1) Heptagon interior angle sum = $(7 - 2) \times 180°$
                                = $5 \times 180°$
                                = $900°$

(2) Let the polygon have $n$ sides
    $(n - 2) \times 180° = 1440°$
    $n - 2 = 1440° \div 180° = 8$
    $n = 8 + 2 = 10$

    **Answer**: This is a **decagon** (10-gon).

**Example 22**: Each interior angle of a polygon equals 140°. Find the number of sides.

**Solution**:
Let the polygon have $n$ sides

Each interior angle = Interior angle sum ÷ Number of sides
$140° = \frac{(n-2) \cdot 180°}{n}$

$140n = 180(n-2)$
$140n = 180n - 360$
$40n = 360$
$n = 9$

**Answer**: This is a **nonagon** (9-gon).

**Verification**: Nonagon interior angle sum = $(9-2) \times 180° = 1260°$
Each interior angle = $1260° \div 9 = 140°$ ✓

---

### 3.4.8 Exterior Angle Sum of Polygons

**Observation of Patterns**:
- Triangle exterior angle sum = 360°
- Quadrilateral exterior angle sum = 360°
- Pentagon exterior angle sum = 360°
- Hexagon exterior angle sum = 360°

**Pattern Discovered**: The exterior angle sum of any polygon is always 360°, regardless of the number of sides!

### 3.4.9 Exterior Angle Sum Theorem

> **Theorem**: The sum of the exterior angles of an n-gon equals 360°.

**Note**: The exterior angle sum of an n-gon is independent of n; it is always 360°.

### 3.4.10 Proof of the Theorem

```
Proof: An n-gon has n interior angles and n exterior angles
      Each interior angle + adjacent exterior angle = 180°
      Sum of n interior angles + Sum of n exterior angles = n·180°

      Since sum of n interior angles = (n-2)·180°
      Therefore sum of n exterior angles = n·180° - (n-2)·180°
                                        = n·180° - n·180° + 360°
                                        = 360°
```

### 3.4.11 Applications of Exterior Angle Sum

**Application Scenarios**:
1. Given each exterior angle, find the number of sides
2. Given the number of sides and some exterior angles, find the others
3. Calculations involving regular polygons

**Problem-Solving Strategy**:
- Exterior angle sum is always 360°, independent of the number of sides
- Number of sides = 360° ÷ each exterior angle (for regular polygons)

### 3.4.12 Examples on Exterior Angle Sum

**Example 23**: Each exterior angle of a polygon equals 60°. Find the number of sides.

**Solution**:
The exterior angle sum of any polygon is 360°
Number of sides = $360° \div 60° = 6$

**Answer**: This is a **hexagon**.

**Example 24**: Each exterior angle of a polygon equals 40°. Find the number of sides and the interior angle sum.

**Solution**:
(1) Finding the number of sides:
Number of sides = $360° \div 40° = 9$

(2) Finding the interior angle sum:
Interior angle sum = $(9-2) \times 180° = 7 \times 180° = 1260°$

**Answer**: This is a nonagon with interior angle sum 1260°.

**Example 25**: In a regular n-gon, one exterior angle is $\frac{1}{3}$ of one interior angle. Find n.

**Solution**:
Let each exterior angle be $x°$, then each interior angle is $3x°$

Since exterior angle + interior angle = 180°
$x + 3x = 180°$
$4x = 180°$
$x = 45°$

Each exterior angle = 45°
Number of sides $n = 360° \div 45° = 8$

**Answer**: This is an **octagon**.

---

## 3.5 Congruent Triangles

### 3.5.1 Definition of Congruence

> **Definition**: Two triangles are congruent if they have the same shape and size. All corresponding sides and angles are equal.

**Notation**: $\triangle ABC \cong \triangle DEF$

**Key Points**:
1. Corresponding vertices are written in the same order
2. Congruent triangles can be superimposed exactly
3. Congruence is denoted by the symbol $\cong$

### 3.5.2 Properties of Congruent Triangles

If $\triangle ABC \cong \triangle DEF$, then:
- **Corresponding sides are equal**: $AB = DE$, $BC = EF$, $CA = FD$
- **Corresponding angles are equal**: $\angle A = \angle D$, $\angle B = \angle E$, $\angle C = \angle F$

### 3.5.3 Criteria for Triangle Congruence

| Criterion | Name | Condition |
|-----------|------|-----------|
| **SSS** | Side-Side-Side | Three sides of one triangle equal three sides of another |
| **SAS** | Side-Angle-Side | Two sides and the included angle are equal |
| **ASA** | Angle-Side-Angle | Two angles and the included side are equal |
| **AAS** | Angle-Angle-Side | Two angles and a non-included side are equal |
| **HL** | Hypotenuse-Leg | (Right triangles only) Hypotenuse and one leg are equal |

**Warning**: SSA (Side-Side-Angle) is NOT a valid criterion! (Ambiguous case)

### 3.5.4 SSS Criterion

> **Theorem**: If three sides of one triangle are equal to three sides of another triangle, then the two triangles are congruent.

**Example**: In triangles $ABC$ and $DEF$, if $AB = DE = 5$, $BC = EF = 6$, $CA = FD = 7$, then $\triangle ABC \cong \triangle DEF$ (SSS).

### 3.5.5 SAS Criterion

> **Theorem**: If two sides and the included angle of one triangle are equal to two sides and the included angle of another triangle, then the two triangles are congruent.

**Key Point**: The angle must be the INCLUDED angle (between the two sides).

**Example**: In triangles $ABC$ and $DEF$, if $AB = DE$, $\angle B = \angle E$, $BC = EF$, then $\triangle ABC \cong \triangle DEF$ (SAS).

### 3.5.6 ASA Criterion

> **Theorem**: If two angles and the included side of one triangle are equal to two angles and the included side of another triangle, then the two triangles are congruent.

**Example**: In triangles $ABC$ and $DEF$, if $\angle A = \angle D$, $AB = DE$, $\angle B = \angle E$, then $\triangle ABC \cong \triangle DEF$ (ASA).

### 3.5.7 AAS Criterion

> **Theorem**: If two angles and a non-included side of one triangle are equal to two angles and the corresponding non-included side of another triangle, then the two triangles are congruent.

**Note**: AAS can be derived from ASA using the angle sum property.

### 3.5.8 HL Criterion (Right Triangles Only)

> **Theorem**: If the hypotenuse and one leg of a right triangle are equal to the hypotenuse and one leg of another right triangle, then the two triangles are congruent.

**Key Point**: This criterion only applies to right triangles!

### 3.5.9 Proving Triangles Congruent - Examples

**Example 26**: In $\triangle ABC$, $D$ is the midpoint of $BC$, and $AD \perp BC$. Prove that $AB = AC$.

**Proof**:
```
In △ABD and △ACD:
  BD = CD (D is midpoint of BC)
  ∠ADB = ∠ADC = 90° (AD ⊥ BC)
  AD = AD (common side)

Therefore △ABD ≅ △ACD (SAS)

Therefore AB = AC (corresponding sides of congruent triangles)
```

**Example 27**: In $\triangle ABC$, $\angle B = \angle C$. Prove that $AB = AC$.

**Proof**:
```
Draw altitude AD from A to BC

In △ABD and △ACD:
  ∠ADB = ∠ADC = 90° (AD is altitude)
  ∠B = ∠C (given)
  AD = AD (common side)

Therefore △ABD ≅ △ACD (AAS)

Therefore AB = AC (corresponding sides of congruent triangles)
```

**This proves**: Equal angles are opposite to equal sides

### 3.5.10 Common Proof Strategies

1. **Find common sides**: Look for shared sides between triangles
2. **Find vertical angles**: Angles formed by intersecting lines
3. **Use given conditions**: Midpoints, perpendiculars, angle bisectors
4. **Add auxiliary lines**: Draw altitudes, medians, or parallel lines

---

## 3.6 Similar Triangles

### 3.6.1 Definition of Similarity

> **Definition**: Two triangles are similar if they have the same shape but not necessarily the same size. All corresponding angles are equal, and corresponding sides are proportional.

**Notation**: $\triangle ABC \sim \triangle DEF$

**Key Points**:
1. Similar triangles have equal corresponding angles
2. Corresponding sides are in the same ratio (called the scale factor or ratio of similarity)
3. Congruent triangles are a special case of similar triangles (ratio = 1)

### 3.6.2 Properties of Similar Triangles

If $\triangle ABC \sim \triangle DEF$ with ratio $k$, then:
- **Corresponding angles are equal**: $\angle A = \angle D$, $\angle B = \angle E$, $\angle C = \angle F$
- **Corresponding sides are proportional**: $\frac{AB}{DE} = \frac{BC}{EF} = \frac{CA}{FD} = k$
- **Ratio of perimeters** = $k$
- **Ratio of areas** = $k^2$

### 3.6.3 Criteria for Triangle Similarity

| Criterion | Name | Condition |
|-----------|------|-----------|
| **AA** | Angle-Angle | Two angles of one triangle equal two angles of another |
| **SAS** | Side-Angle-Side | Two sides are proportional and the included angles are equal |
| **SSS** | Side-Side-Side | All three pairs of sides are proportional |

### 3.6.4 AA Similarity Criterion

> **Theorem**: If two angles of one triangle are equal to two angles of another triangle, then the triangles are similar.

**Note**: Since the angle sum is 180°, if two angles are equal, the third must also be equal.

**Example**: In $\triangle ABC$ and $\triangle DEF$, if $\angle A = \angle D = 50°$ and $\angle B = \angle E = 60°$, then $\triangle ABC \sim \triangle DEF$ (AA).

### 3.6.5 SAS Similarity Criterion

> **Theorem**: If two sides of one triangle are proportional to two sides of another triangle, and the included angles are equal, then the triangles are similar.

**Example**: In $\triangle ABC$ and $\triangle DEF$, if $\frac{AB}{DE} = \frac{AC}{DF} = 2$ and $\angle A = \angle D$, then $\triangle ABC \sim \triangle DEF$ (SAS).

### 3.6.6 SSS Similarity Criterion

> **Theorem**: If three sides of one triangle are proportional to three sides of another triangle, then the triangles are similar.

**Example**: If $\frac{AB}{DE} = \frac{BC}{EF} = \frac{CA}{FD} = 3$, then $\triangle ABC \sim \triangle DEF$ (SSS).

### 3.6.7 Applications of Similar Triangles

**Example 28**: In $\triangle ABC$, $DE \parallel BC$, $AD = 2$, $DB = 4$, $DE = 3$. Find $BC$.

**Solution**:
Since $DE \parallel BC$, by AA similarity:
$\triangle ADE \sim \triangle ABC$

$\frac{AD}{AB} = \frac{DE}{BC}$

$\frac{2}{2+4} = \frac{3}{BC}$

$\frac{2}{6} = \frac{3}{BC}$

$BC = \frac{3 \times 6}{2} = 9$

**Example 29**: In $\triangle ABC$, $\angle ACB = 90°$, $CD \perp AB$ at $D$. Prove that $CD^2 = AD \cdot DB$.

**Proof**:
```
In △ACD and △CBD:
  ∠ADC = ∠CDB = 90°
  ∠A = 90° - ∠ACD = ∠DCB (complementary angles)

Therefore △ACD ∼ △CBD (AA)

Therefore AD/CD = CD/DB

Therefore CD² = AD · DB
```

---

## 3.7 The Pythagorean Theorem

### 3.7.1 Statement of the Theorem

> **Pythagorean Theorem**: In a right triangle, the square of the hypotenuse equals the sum of the squares of the two legs.

**Mathematical Expression**:
In right triangle $\triangle ABC$ with $\angle C = 90°$:
$$a^2 + b^2 = c^2$$

where $c$ is the hypotenuse and $a$, $b$ are the legs.

### 3.7.2 Proof of the Pythagorean Theorem

**Method 1: Area Method**

```
Construct a square with side (a + b)
Place four congruent right triangles inside

Area of large square = (a + b)²
Area of four triangles = 4 × (½ab) = 2ab
Area of inner square = c²

(a + b)² = 2ab + c²
a² + 2ab + b² = 2ab + c²
a² + b² = c²
```

**Method 2: Similar Triangles**

```
In right △ABC with ∠C = 90°, draw altitude CD to hypotenuse AB

△ACD ∼ △ABC (AA): AD/AC = AC/AB, so AC² = AD · AB
△CBD ∼ △ABC (AA): BD/BC = BC/AB, so BC² = BD · AB

Adding: AC² + BC² = AD · AB + BD · AB = (AD + BD) · AB = AB · AB = AB²

Therefore: b² + a² = c²
```

### 3.7.3 Common Pythagorean Triples

| Triple | Verification |
|--------|--------------|
| 3, 4, 5 | $3^2 + 4^2 = 9 + 16 = 25 = 5^2$ |
| 5, 12, 13 | $5^2 + 12^2 = 25 + 144 = 169 = 13^2$ |
| 8, 15, 17 | $8^2 + 15^2 = 64 + 225 = 289 = 17^2$ |
| 7, 24, 25 | $7^2 + 24^2 = 49 + 576 = 625 = 25^2$ |

**Note**: Multiples of Pythagorean triples are also Pythagorean triples (e.g., 6, 8, 10).

### 3.7.4 Converse of the Pythagorean Theorem

> **Converse**: If the sides of a triangle satisfy $a^2 + b^2 = c^2$, then the triangle is a right triangle with the right angle opposite to side $c$.

**Application**: Determining if a triangle is a right triangle.

**Example 30**: Is a triangle with sides 5, 12, 13 a right triangle?

**Solution**:
$5^2 + 12^2 = 25 + 144 = 169 = 13^2$ ✓

Yes, it is a right triangle with the right angle opposite to the side of length 13.

### 3.7.5 Generalization: Determining Triangle Type

For a triangle with sides $a \leq b \leq c$:
- If $a^2 + b^2 = c^2$: Right triangle
- If $a^2 + b^2 > c^2$: Acute triangle
- If $a^2 + b^2 < c^2$: Obtuse triangle

**Example 31**: Classify the triangle with sides 4, 5, 7.

**Solution**:
$4^2 + 5^2 = 16 + 25 = 41$
$7^2 = 49$

Since $41 < 49$, we have $a^2 + b^2 < c^2$

**Answer**: This is an obtuse triangle.

---

## 3.8 Angle Bisector Theorem

### 3.8.1 Statement of the Theorem

> **Angle Bisector Theorem**: The angle bisector of a triangle divides the opposite side in the ratio of the adjacent sides.

**Mathematical Expression**:
In $\triangle ABC$, if $AD$ bisects $\angle A$ and meets $BC$ at $D$, then:
$$\frac{BD}{DC} = \frac{AB}{AC}$$

### 3.8.2 Proof

```
Draw CE ∥ AD, meeting the extension of BA at E

Since AD ∥ CE:
  ∠BAD = ∠AEC (corresponding angles)
  ∠DAC = ∠ACE (alternate interior angles)

Since AD bisects ∠A:
  ∠BAD = ∠DAC

Therefore ∠AEC = ∠ACE
Therefore AC = AE (isosceles triangle)

In △BCE, AD ∥ CE:
  BD/DC = BA/AE = BA/AC = AB/AC
```

### 3.8.3 Applications

**Example 32**: In $\triangle ABC$, $AB = 6$, $AC = 4$, and $AD$ bisects $\angle A$. If $BC = 5$, find $BD$ and $DC$.

**Solution**:
By the angle bisector theorem:
$\frac{BD}{DC} = \frac{AB}{AC} = \frac{6}{4} = \frac{3}{2}$

Let $BD = 3k$ and $DC = 2k$

$BD + DC = BC$
$3k + 2k = 5$
$5k = 5$
$k = 1$

**Answer**: $BD = 3$, $DC = 2$

### 3.8.4 Angle Bisector Length Formula (Competition Level)

> **Formula**: The length of the angle bisector from vertex $A$ to side $BC$ is:
> $$t_a = \frac{2bc \cos\frac{A}{2}}{b+c}$$

Or equivalently:
$$t_a^2 = bc\left[1 - \left(\frac{a}{b+c}\right)^2\right]$$

---

# Part IV: Competition Level Problems

## 4.1 Classic Competition Problems

### Problem 1 (Angle Calculation with Multiple Elements)

> In $\triangle ABC$, $AD$ is an altitude, $AE$ is an angle bisector, $\angle B = 50°$, and $\angle C = 70°$.
> (1) Find $\angle BAC$
> (2) Find $\angle DAE$

**Solution**:

(1) By the interior angle sum theorem:
    $\angle BAC = 180° - \angle B - \angle C$
         $= 180° - 50° - 70°$
         $= 60°$

(2) Since $AD$ is an altitude, $\angle ADB = 90°$

In right triangle $ABD$:

$\angle BAD = 90° - \angle B = 90° - 50° = 40°$

Since $AE$ is an angle bisector:

$\angle BAE = \frac{1}{2}\angle BAC = \frac{1}{2} \times 60° = 30°$

$\angle DAE = \angle BAD - \angle BAE = 40° - 30° = 10°$

**Problem-Solving Strategy**:
1. First use interior angle sum to find $\angle BAC$
2. Use the altitude property to find $\angle BAD$
3. Use the angle bisector property to find $\angle BAE$
4. Finally use subtraction to find $\angle DAE$

---

### Problem 2 (Altitude and Angle Bisector Combined)

> In $\triangle ABC$, $AD \perp BC$ at $D$, $AE$ bisects $\angle BAC$, $\angle B = 70°$, and $\angle C = 34°$.
> (1) Find $\angle BAE$
> (2) Find $\angle DAE$

**Solution**:

(1) $\angle BAC = 180° - \angle B - \angle C = 180° - 70° - 34° = 76°$
    $\angle BAE = \frac{1}{2}\angle BAC = \frac{1}{2} \times 76° = 38°$

(2) Since $AD \perp BC$, $\angle ADB = 90°$
    In right triangle $ABD$:
    $\angle BAD = 90° - \angle B = 90° - 70° = 20°$

    $\angle DAE = \angle BAE - \angle BAD = 38° - 20° = 18°$

---

### Problem 3 (Exterior Angle with Angle Bisector)

> In $\triangle ABC$, exterior angle $\angle ACD = 120°$, $\angle B = 50°$, and $AE$ bisects $\angle CAD$. Find $\angle BAE$.

**Solution**:

Step 1: Find $\angle A$ using exterior angle property
$\angle ACD = \angle A + \angle B$
$120° = \angle A + 50°$
$\angle A = 70°$

Step 2: Find $\angle CAD$
$\angle CAD = 180° - \angle ACD = 180° - 120° = 60°$

Wait, let me reconsider. $\angle CAD$ is the exterior angle at vertex A, not at C.

Actually, $\angle CAD$ is the angle at A in the exterior region. Since $AE$ bisects $\angle CAD$:

$\angle CAD = 180° - \angle BAC = 180° - 70° = 110°$

$\angle CAE = \frac{1}{2}\angle CAD = \frac{1}{2} \times 110° = 55°$

$\angle BAE = \angle BAC + \angle CAE = 70° + 55° = 125°$

Or alternatively: $\angle BAE = 180° - \angle CAE = 180° - 55° = 125°$

**Answer**: $\angle BAE = 125°$

---

### Problem 4 (Triangle Inequality Application)

> The three sides of a triangle are $a$, $b$, and $c$ where $a < b < c$. If $a = 3$ and $c = 7$, find all possible integer values of $b$.

**Solution**:

By the triangle inequality:
- $a + b > c$: $3 + b > 7$, so $b > 4$
- $a + c > b$: $3 + 7 > b$, so $b < 10$
- $b + c > a$: Always satisfied since $b > 4 > 3 = a$

Also, we need $a < b < c$: $3 < b < 7$

Combining: $4 < b < 7$

**Answer**: $b \in \{5, 6\}$

---

### Problem 5 (Polygon Diagonal Problem)

> A polygon has 54 diagonals. Find the number of sides and the interior angle sum.

**Solution**:

Let the polygon have $n$ sides.

$\frac{n(n-3)}{2} = 54$
$n(n-3) = 108$
$n^2 - 3n - 108 = 0$
$(n-12)(n+9) = 0$

$n = 12$ or $n = -9$ (rejected)

Interior angle sum = $(12-2) \times 180° = 10 \times 180° = 1800°$

**Answer**: 12 sides, interior angle sum = 1800°

---

### Problem 6 (Regular Polygon Problem)

> The ratio of an interior angle to an exterior angle of a regular polygon is 5:1. Find the number of sides.

**Solution**:

Let the exterior angle be $x°$, then the interior angle is $5x°$.

Since interior angle + exterior angle = 180°:
$5x + x = 180°$
$6x = 180°$
$x = 30°$

Number of sides = $360° \div 30° = 12$

**Answer**: This is a regular 12-gon (dodecagon).

---

### Problem 7 (Isosceles Triangle Classification)

> One angle of an isosceles triangle is 80°. Find all possible values for the other two angles.

**Solution**:

**Case 1**: The vertex angle is 80°
Base angles = $(180° - 80°) \div 2 = 50°$
The three angles are: 80°, 50°, 50°

**Case 2**: A base angle is 80°
The other base angle is also 80°
Vertex angle = $180° - 80° - 80° = 20°$
The three angles are: 20°, 80°, 80°

**Answer**: Either (80°, 50°, 50°) or (20°, 80°, 80°)

---

### Problem 8 (Competition Classic - Angle Relationships)

> In $\triangle ABC$, $\angle A : \angle B : \angle C = 1 : 2 : 3$.
> (1) Classify this triangle by angles
> (2) If the perimeter is 30 cm and the shortest side is 5 cm, find all side lengths

**Solution**:

(1) Let $\angle A = k$, $\angle B = 2k$, $\angle C = 3k$

$k + 2k + 3k = 180°$
$6k = 180°$
$k = 30°$

So $\angle A = 30°$, $\angle B = 60°$, $\angle C = 90°$

**Answer**: This is a **right triangle**.

(2) In a 30-60-90 triangle, the sides are in ratio $1 : \sqrt{3} : 2$

The shortest side (opposite to 30°) is 5 cm.
Let the sides be $5$, $5\sqrt{3}$, and $10$.

Check: $5 + 5\sqrt{3} + 10 = 15 + 5\sqrt{3} \approx 23.66$ cm

This doesn't equal 30 cm, so we need to recalculate.

Let the sides be $a$, $a\sqrt{3}$, $2a$ where $a$ is the shortest side.
$a + a\sqrt{3} + 2a = 30$
$a(3 + \sqrt{3}) = 30$
$a = \frac{30}{3 + \sqrt{3}} = \frac{30(3 - \sqrt{3})}{(3+\sqrt{3})(3-\sqrt{3})} = \frac{30(3-\sqrt{3})}{6} = 5(3-\sqrt{3})$

But we're told the shortest side is 5 cm, so $a = 5$.

The sides are: $5$ cm, $5\sqrt{3}$ cm, $10$ cm

Perimeter = $5 + 5\sqrt{3} + 10 = 15 + 5\sqrt{3} \approx 23.66$ cm

Since this doesn't match 30 cm, there may be an inconsistency in the problem. If we use the perimeter of 30 cm:

$a(3 + \sqrt{3}) = 30$
$a \approx 6.34$ cm

**Answer**: The sides are approximately 6.34 cm, 10.98 cm, and 12.68 cm.

---

## 4.2 National Junior High Math League Level Problems

### Problem 9 (Congruence Proof)

> In $\triangle ABC$, $AB = AC$, $D$ is on $AB$, $E$ is on $AC$, and $BD = CE$. Prove that $\triangle BEC \cong \triangle CDB$.

**Solution**:
```
In △BEC and △CDB:
  BC = CB (common side)
  ∠ABC = ∠ACB (base angles of isosceles triangle)
  BD = CE (given)

Since AB = AC and BD = CE:
  AD = AE
  Therefore AB - AD = AC - AE
  So DB = EC... wait, that's given.

Let me reconsider:
  BE = AB - AE = AB - (AC - CE) = AB - AC + CE = CE (since AB = AC)
  CD = AC - AD = AC - (AB - BD) = AC - AB + BD = BD (since AB = AC)

Actually:
  BE = AB - AE and CD = AC - AD
  Since AB = AC, BD = CE, we need to show BE = CD

  AE = AC - CE and AD = AB - BD
  Since AB = AC and BD = CE: AD = AE
  Therefore BE = AB - AE = AC - AD = CD

In △BEC and △CDB:
  BC = CB (common side)
  ∠EBC = ∠DCB (base angles of isosceles triangle)
  BE = CD (proved above)

Therefore △BEC ≅ △CDB (SAS)
```

---

### Problem 10 (Similar Triangles Application)

> In $\triangle ABC$, $\angle ACB = 90°$, $CD \perp AB$ at $D$. If $AD = 4$ and $BD = 9$, find $CD$.

**Solution**:

In right triangle $ABC$ with altitude $CD$ to hypotenuse:

By the geometric mean relationship:
$CD^2 = AD \cdot BD$

$CD^2 = 4 \times 9 = 36$

$CD = 6$

**Proof of the relationship**:
$\triangle ACD \sim \triangle CBD$ (AA similarity)
Therefore $\frac{AD}{CD} = \frac{CD}{BD}$
So $CD^2 = AD \cdot BD$

---

### Problem 11 (Pythagorean Theorem)

> In $\triangle ABC$, $\angle C = 90°$, $AC = 3$, $BC = 4$. Point $D$ is on $AB$ such that $CD \perp AB$. Find $CD$.

**Solution**:

**Method 1: Area Method**
$AB = \sqrt{AC^2 + BC^2} = \sqrt{9 + 16} = 5$

Area of $\triangle ABC = \frac{1}{2} \times AC \times BC = \frac{1}{2} \times 3 \times 4 = 6$

Also, Area = $\frac{1}{2} \times AB \times CD = \frac{1}{2} \times 5 \times CD$

Therefore: $6 = \frac{5 \times CD}{2}$

$CD = \frac{12}{5} = 2.4$

**Method 2: Similar Triangles**
$\triangle ACD \sim \triangle ABC$ (AA)
$\frac{CD}{BC} = \frac{AC}{AB}$
$\frac{CD}{4} = \frac{3}{5}$
$CD = \frac{12}{5} = 2.4$

---

### Problem 12 (Competition Classic)

> In $\triangle ABC$, $AB = 13$, $BC = 14$, $CA = 15$. Find the area of $\triangle ABC$.

**Solution**:

Using Heron's formula:
$s = \frac{13 + 14 + 15}{2} = 21$

$S = \sqrt{s(s-a)(s-b)(s-c)}$
$S = \sqrt{21 \times (21-14) \times (21-15) \times (21-13)}$
$S = \sqrt{21 \times 7 \times 6 \times 8}$
$S = \sqrt{7056}$
$S = 84$

**Answer**: The area is 84 square units.

---

### Problem 13 (Angle Bisector Theorem)

> In $\triangle ABC$, $AB = 10$, $AC = 8$, $BC = 6$. The angle bisector from $A$ meets $BC$ at $D$. Find $BD$ and $DC$.

**Solution**:

By the angle bisector theorem:
$\frac{BD}{DC} = \frac{AB}{AC} = \frac{10}{8} = \frac{5}{4}$

Let $BD = 5k$, $DC = 4k$

$BD + DC = BC$
$5k + 4k = 6$
$9k = 6$
$k = \frac{2}{3}$

$BD = 5 \times \frac{2}{3} = \frac{10}{3}$
$DC = 4 \times \frac{2}{3} = \frac{8}{3}$

**Answer**: $BD = \frac{10}{3}$, $DC = \frac{8}{3}$

---

### Problem 14 (Integer Triangle Counting)

> How many triangles with integer sides have perimeter 15?

**Solution**:

Let sides be $a \leq b \leq c$ with $a + b + c = 15$.

By triangle inequality: $a + b > c$
Since $a + b + c = 15$: $a + b = 15 - c$
So $15 - c > c$, meaning $c < 7.5$, thus $c \leq 7$

Also $c \geq b \geq a \geq 1$ and $a + b = 15 - c$

**Case $c = 7$**: $a + b = 8$, $b \leq 7$
- $(1,7,7), (2,6,7), (3,5,7), (4,4,7)$ → 4 triangles

**Case $c = 6$**: $a + b = 9$, $b \leq 6$
- $(3,6,6), (4,5,6)$ → 2 triangles

**Case $c = 5$**: $a + b = 10$, $b \leq 5$
- $(5,5,5)$ → 1 triangle

**Answer**: 7 triangles

---

### Problem 15 (Midsegment Application)

> In $\triangle ABC$, $D$, $E$, $F$ are midpoints of $BC$, $CA$, $AB$ respectively. If the perimeter of $\triangle DEF$ is 12, find the perimeter of $\triangle ABC$.

**Solution**:

By the midsegment theorem:
- $DE = \frac{1}{2}AB$
- $EF = \frac{1}{2}BC$
- $FD = \frac{1}{2}CA$

Perimeter of $\triangle DEF = DE + EF + FD = \frac{1}{2}(AB + BC + CA) = \frac{1}{2} \times$ Perimeter of $\triangle ABC$

$12 = \frac{1}{2} \times$ Perimeter of $\triangle ABC$

Perimeter of $\triangle ABC = 24$

---

### Problem 16 (Competition Problem)

> In $\triangle ABC$, $\angle BAC = 90°$, $AD \perp BC$ at $D$. Prove that $\frac{1}{AD^2} = \frac{1}{AB^2} + \frac{1}{AC^2}$.

**Proof**:

In right triangle $ABC$:
$BC^2 = AB^2 + AC^2$ (Pythagorean theorem)

Area of $\triangle ABC = \frac{1}{2} \times AB \times AC = \frac{1}{2} \times BC \times AD$

Therefore: $AB \times AC = BC \times AD$

$AD = \frac{AB \times AC}{BC}$

$AD^2 = \frac{AB^2 \times AC^2}{BC^2} = \frac{AB^2 \times AC^2}{AB^2 + AC^2}$

$\frac{1}{AD^2} = \frac{AB^2 + AC^2}{AB^2 \times AC^2} = \frac{1}{AC^2} + \frac{1}{AB^2}$

**Q.E.D.**

---

# Part V: Practice Problems by Level

## 5.1 Basic Level (Foundation)

**Problem 1**: The two sides of a triangle have lengths 3 and 5. Find the range of the third side $x$.

**Problem 2**: In $\triangle ABC$, $\angle A = 40°$ and $\angle B = 60°$. Find $\angle C$.

**Problem 3**: In right triangle $\triangle ABC$, $\angle C = 90°$ and $\angle A = 35°$. Find $\angle B$.

**Problem 4**: Find the interior angle sum of a hexagon.

**Problem 5**: The exterior angle sum of any polygon equals ______ degrees.

**Problem 6**: From one vertex of an n-gon, ______ diagonals can be drawn.

**Problem 7**: In an isosceles triangle, the two base angles are ______.

**Problem 8**: The three ______ of a triangle meet at a point called the centroid.

**Problem 9**: Can line segments of lengths 1cm, 2cm, and 3cm form a triangle?

**Problem 10**: Can line segments of lengths 3cm, 4cm, and 5cm form a triangle?

---

## 5.2 Intermediate Level

**Problem 11**: Which of the following sets of line segments can form a triangle?
A. 1cm, 2cm, 3cm
B. 2cm, 3cm, 5cm
C. 3cm, 4cm, 5cm
D. 2cm, 3cm, 6cm

**Problem 12**: In $\triangle ABC$, if $\angle A = 50°$ and $\angle B = 60°$, then $\angle C = $?
A. 50°
B. 60°
C. 70°
D. 80°

**Problem 13**: A triangle has interior angles in the ratio 1:2:3. What type of triangle is it?
A. Acute triangle
B. Right triangle
C. Obtuse triangle
D. Isosceles triangle

**Problem 14**: Which of the following shapes has stability?
A. Square
B. Parallelogram
C. Triangle
D. Trapezoid

**Problem 15**: If one angle of an isosceles triangle is 70°, the other two angles are:
A. 70°, 40°
B. 55°, 55°
C. 70°, 70°
D. Either 55°, 55° or 70°, 40°

---

## 5.3 Advanced Level

**Problem 16**: The two sides of a triangle have lengths 4 and 7. Find the range of the third side $x$.

**Problem 17**: In $\triangle ABC$, $\angle A = 70°$ and $\angle B = 50°$. Find $\angle C$.

**Problem 18**: In right triangle $\triangle ABC$, $\angle C = 90°$ and $\angle A = 55°$. Find $\angle B$.

**Problem 19**: Find the interior angle sum and number of diagonals of a decagon.

**Problem 20**: Each exterior angle of a polygon equals 30°. Find the number of sides.

**Problem 21**: In $\triangle ABC$, $\angle A - \angle B = 20°$ and $\angle C = 80°$. Find $\angle A$ and $\angle B$.

**Problem 22**: In $\triangle ABC$, $AB = AC$ and $\angle A = 80°$. Find $\angle B$ and $\angle C$.

**Problem 23**: A polygon has interior angle sum 1260°. Find the number of sides and diagonals.

**Problem 24**: Each interior angle of a polygon equals 150°. Find the number of sides.

**Problem 25**: In $\triangle ABC$, $AD$ is an altitude, $AE$ is an angle bisector, $\angle B = 60°$, and $\angle C = 50°$. Find $\angle DAE$.

---

## 5.4 Competition Level

**Problem 26**: In $\triangle ABC$, exterior angle $\angle ACD = 120°$, $\angle B = 50°$, and $AE$ bisects $\angle CAD$. Find $\angle BAE$.

**Problem 27**: In $\triangle ABC$, $\angle A = 60°$, $\angle B = 70°$, $AD$ is an altitude, and $AE$ is an angle bisector. Find $\angle DAE$.

**Problem 28**: The ratio of an interior angle to an exterior angle of a regular n-gon is 3:1. Find n.

**Problem 29**: In $\triangle ABC$, $AB = AC$, and $D$ is a point on $BC$. If $BD = 5$ and $CD = 3$, find the range of $AD$.

**Problem 30**: In $\triangle ABC$, $\angle B = 50°$, $\angle C = 70°$, $AD \perp BC$ at $D$, and $AE$ bisects $\angle BAC$.
(1) Find $\angle BAE$
(2) Find $\angle DAE$
(3) If $AD = 6$ and $AB = 10$, find $BD$

**Problem 31** (Congruence): In $\triangle ABC$ and $\triangle DEF$, $AB = DE$, $\angle A = \angle D$, $AC = DF$. Are the triangles congruent? State the criterion.

**Problem 32** (Pythagorean): In right triangle $\triangle ABC$, $\angle C = 90°$, $AC = 5$, $BC = 12$. Find $AB$.

**Problem 33** (Pythagorean Converse): Is a triangle with sides 6, 8, 10 a right triangle? If so, which angle is the right angle?

**Problem 34** (Similar Triangles): In $\triangle ABC$, $DE \parallel BC$, $AD = 3$, $DB = 6$, $DE = 4$. Find $BC$.

**Problem 35** (Angle Bisector): In $\triangle ABC$, $AB = 8$, $AC = 6$, and $AD$ bisects $\angle A$. If $BC = 7$, find $BD$.

**Problem 36** (Area): Find the area of a triangle with sides 5, 5, 6.

**Problem 37** (Midsegment): In $\triangle ABC$, $D$ and $E$ are midpoints of $AB$ and $AC$. If $BC = 10$, find $DE$.

**Problem 38** (Competition): In $\triangle ABC$, $\angle ACB = 90°$, $CD \perp AB$ at $D$. If $AD = 3$, $BD = 12$, find $CD$ and $AC$.

**Problem 39** (Competition): How many triangles with integer sides have perimeter 12?

**Problem 40** (Competition): In $\triangle ABC$, $AB = AC = 10$, $BC = 12$. Find the length of the altitude from $A$ to $BC$.

---

# Part VI: Answer Key

## Basic Level Answers

**1.** $2 < x < 8$

**2.** $\angle C = 80°$

**3.** $\angle B = 55°$

**4.** $720°$

**5.** $360°$

**6.** $(n-3)$

**7.** equal

**8.** medians

**9.** No (1 + 2 = 3, not greater than 3)

**10.** Yes (3 + 4 = 7 > 5)

## Intermediate Level Answers

**11.** C

**12.** C (70°)

**13.** B (Right triangle)

**14.** C (Triangle)

**15.** D

## Advanced Level Answers

**16.** $3 < x < 11$

**17.** $\angle C = 60°$

**18.** $\angle B = 35°$

**19.** Interior angle sum = $1440°$; Diagonals = $35$

**20.** 12 sides

**21.** $\angle A = 60°$, $\angle B = 40°$

**22.** $\angle B = \angle C = 50°$

**23.** 9 sides; 27 diagonals

**24.** 12 sides

**25.** $\angle DAE = 5°$

## Competition Level Answers

**26.** $\angle BAE = 125°$

**27.** $\angle DAE = 5°$

**28.** $n = 8$

**29.** $1 < AD < 8$

**30.** (1) $\angle BAE = 30°$
      (2) $\angle DAE = 10°$
      (3) $BD = 8$

**31.** Yes, $\triangle ABC \cong \triangle DEF$ (SAS)

**32.** $AB = \sqrt{5^2 + 12^2} = \sqrt{169} = 13$

**33.** Yes, $6^2 + 8^2 = 36 + 64 = 100 = 10^2$. The right angle is opposite to the side of length 10.

**34.** $BC = 12$ (by similar triangles, $\frac{AD}{AB} = \frac{DE}{BC}$, so $\frac{3}{9} = \frac{4}{BC}$)

**35.** $BD = 4$ (by angle bisector theorem, $\frac{BD}{DC} = \frac{8}{6} = \frac{4}{3}$, so $BD = \frac{4}{7} \times 7 = 4$)

**36.** Area = $12$ (using Heron's formula with $s = 8$: $\sqrt{8 \times 3 \times 3 \times 2} = \sqrt{144} = 12$)

**37.** $DE = 5$ (midsegment theorem: $DE = \frac{1}{2}BC$)

**38.** $CD = 6$, $AC = 3\sqrt{5}$ (using $CD^2 = AD \cdot BD = 36$, and $AC^2 = AD \cdot AB = 3 \times 15 = 45$)

**39.** 3 triangles: (2,5,5), (3,4,5), (4,4,4)

**40.** Altitude = $8$ (using area method or Pythagorean theorem with half-base = 6)

---

# Part VII: Summary and Key Formulas

## Key Concepts

1. **Triangle**: A figure formed by three line segments connecting three non-collinear points
2. **Triangle inequality**: The sum of any two sides is greater than the third side
3. **Interior angle sum**: The sum of the three interior angles equals 180°
4. **Exterior angle**: Equals the sum of the two non-adjacent interior angles

## Special Line Segments

| Segment | Definition | Point of Concurrency |
|---------|------------|---------------------|
| Altitude | Perpendicular from vertex to opposite side | Orthocenter |
| Median | Segment from vertex to midpoint of opposite side | Centroid (divides median 2:1) |
| Angle Bisector | Segment that bisects an interior angle | Incenter |

## Triangle Classification

| By Angles | By Sides |
|-----------|----------|
| Acute (all angles < 90°) | Scalene (no equal sides) |
| Right (one angle = 90°) | Isosceles (two equal sides) |
| Obtuse (one angle > 90°) | Equilateral (all sides equal) |

## Key Formulas

**Triangle Inequality**:
$$a + b > c, \quad a + c > b, \quad b + c > a$$

**Interior Angle Sum of Triangle**:
$$\angle A + \angle B + \angle C = 180°$$

**Exterior Angle Property**:
$$\text{Exterior angle} = \text{Sum of two non-adjacent interior angles}$$

**Polygon Formulas** (for n-gon):
- Diagonals from one vertex: $n - 3$
- Total diagonals: $\frac{n(n-3)}{2}$
- Interior angle sum: $(n-2) \cdot 180°$
- Exterior angle sum: $360°$ (constant)

**Regular Polygon**:
- Each interior angle: $\frac{(n-2) \cdot 180°}{n}$
- Each exterior angle: $\frac{360°}{n}$

**Pythagorean Theorem** (Right Triangle):
$$a^2 + b^2 = c^2$$
where $c$ is the hypotenuse.

**Triangle Area**:
$$S = \frac{1}{2} \times \text{base} \times \text{height}$$

**Heron's Formula**:
$$S = \sqrt{s(s-a)(s-b)(s-c)}, \quad s = \frac{a+b+c}{2}$$

**Midsegment Theorem**:
$$DE \parallel BC, \quad DE = \frac{1}{2}BC$$

**Angle Bisector Theorem**:
$$\frac{BD}{DC} = \frac{AB}{AC}$$

**Congruence Criteria**: SSS, SAS, ASA, AAS, HL (right triangles)

**Similarity Criteria**: AA, SAS, SSS

**Right Triangle Altitude Theorem**:
$$CD^2 = AD \cdot BD$$

## Competition Tips

1. Look for patterns before computing
2. Use case analysis for isosceles triangle problems
3. Convert exterior angle problems to interior angle problems
4. Use the equation method for angle relationships
5. Verify solutions by checking all conditions
6. Draw accurate diagrams with clear labels
7. Look for congruent or similar triangles in complex figures
8. Use area methods for length calculations
9. Apply Pythagorean theorem for right triangle problems
10. Use angle bisector theorem for ratio problems

---

## Study Path

```
Beginner → Triangle basics → Triangle inequality →
Special segments → Interior angle sum → Exterior angles →
Polygon formulas → Congruent triangles → Similar triangles →
Pythagorean theorem → Angle bisector theorem → Competition techniques
```

---

# Part VIII: Essential Problem-Solving Techniques

## 8.1 Case Analysis (Classification Discussion)

### 8.1.1 Why Case Analysis is Needed

Some problems have different answers depending on conditions. Competition problems often test whether students can identify all cases.

**Key Principle**: When a problem involves:
- Isosceles triangles (vertex angle vs. base angle)
- Unknown position of a point
- Parameters with unknown signs

You must consider ALL possible cases.

### 8.1.2 Example: Isosceles Triangle Angles

**Problem**: One angle of an isosceles triangle is 50°. Find all possible values for the other two angles.

**Analysis**: We don't know if 50° is the vertex angle or a base angle.

**Case 1**: 50° is the vertex angle
- Base angles = $(180° - 50°) \div 2 = 65°$
- Answer: 65°, 65°

**Case 2**: 50° is a base angle
- Other base angle = 50°
- Vertex angle = $180° - 50° - 50° = 80°$
- Answer: 50°, 80°

**Final Answer**: Either (65°, 65°) or (50°, 80°)

---

## 8.2 The Equation Method

### 8.2.1 Core Idea

When angles or sides have specific relationships, set up equations to solve.

### 8.2.2 Example

**Problem**: In $\triangle ABC$, $\angle A : \angle B : \angle C = 2 : 3 : 4$. Find all angles.

**Solution**:
Let $\angle A = 2k$, $\angle B = 3k$, $\angle C = 4k$

$2k + 3k + 4k = 180°$
$9k = 180°$
$k = 20°$

**Answer**: $\angle A = 40°$, $\angle B = 60°$, $\angle C = 80°$

---

## 8.3 The Transformation Method

### 8.3.1 Core Idea

Convert complex problems into simpler ones:
- Exterior angle → Interior angles
- Polygon → Triangles

### 8.3.2 Example

**Problem**: Find the interior angle sum of a pentagon.

**Solution**:
Transform to triangles: Draw diagonals from one vertex to divide the pentagon into 3 triangles.

Interior angle sum = $3 \times 180° = 540°$

Or use formula: $(5-2) \times 180° = 540°$

---

## 8.4 Using Auxiliary Lines

### 8.4.1 When to Use

- When angles need to be "transferred" to a common vertex
- When proving angle relationships using parallel lines

### 8.4.2 Common Auxiliary Lines

1. **Parallel line through a vertex**: Used to prove interior angle sum
2. **Altitude from a vertex**: Creates right angles for calculation
3. **Angle bisector**: Divides angles into equal parts

---

## 8.5 The Bounding Method

### 8.5.1 Core Idea

Use inequalities to restrict possible values, especially for integer solutions.

### 8.5.2 Example

**Problem**: Find all integer values of $x$ if the sides of a triangle are 3, 5, and $x$.

**Solution**:
By triangle inequality:
- $3 + 5 > x \Rightarrow x < 8$
- $3 + x > 5 \Rightarrow x > 2$
- $5 + x > 3$ (always true for positive $x$)

So $2 < x < 8$

**Integer values**: $x \in \{3, 4, 5, 6, 7\}$

---

# Part IX: Historical Competition Problems

## 9.1 Entry-Level Competition Problems

### Problem 9.1.1 (Regional Competition)

**Problem**: If $x + y = 10$ and $x^2 - y^2 = 40$, find $x - y$.

**Key Insight**: $x^2 - y^2 = (x+y)(x-y)$

**Solution**:
$$x^2 - y^2 = (x+y)(x-y)$$
$$40 = 10 \cdot (x-y)$$
$$x - y = 4$$

**Lesson**: Always look for factorization before expanding.

---

### Problem 9.1.2 (Triangle Angle Problem)

**Problem**: In $\triangle ABC$, $\angle A = 40°$, $\angle B = 70°$. Find $\angle C$.

**Solution**:
$\angle C = 180° - \angle A - \angle B = 180° - 40° - 70° = 70°$

**Answer**: $\angle C = 70°$

---

## 9.2 Intermediate Competition Problems

### Problem 9.2.1 (Polygon Problem)

**Problem**: A polygon has interior angle sum equal to 5 times its exterior angle sum. Find the number of sides.

**Solution**:
Exterior angle sum = 360°
Interior angle sum = $5 \times 360° = 1800°$

$(n-2) \times 180° = 1800°$
$n - 2 = 10$
$n = 12$

**Answer**: 12 sides (dodecagon)

---

### Problem 9.2.2 (Altitude and Angle Bisector)

**Problem**: In $\triangle ABC$, $AD$ is an altitude, $AE$ is an angle bisector, $\angle B = 60°$, $\angle C = 40°$. Find $\angle DAE$.

**Solution**:
$\angle BAC = 180° - 60° - 40° = 80°$
$\angle BAE = \frac{1}{2} \times 80° = 40°$
$\angle BAD = 90° - 60° = 30°$
$\angle DAE = \angle BAE - \angle BAD = 40° - 30° = 10°$

**Answer**: $\angle DAE = 10°$

---

## 9.3 Advanced Competition Problems

### Problem 9.3.1 (Triangle Inequality Application)

**Problem**: How many triangles with integer sides have perimeter 12?

**Solution**:
Let sides be $a \leq b \leq c$ with $a + b + c = 12$.

By triangle inequality: $a + b > c$
Since $a + b + c = 12$: $a + b = 12 - c$
So $12 - c > c$, meaning $c < 6$

Also $c \geq b \geq a$ and $a \geq 1$.

Enumerate:
- $c = 5$: $a + b = 7$, with $b \leq 5$. Pairs: (2,5), (3,4) → triangles (2,5,5), (3,4,5)
- $c = 4$: $a + b = 8$, with $b \leq 4$. Pairs: (4,4) → triangle (4,4,4)
- $c = 3$: $a + b = 9$, impossible since $b \leq 3$ means $a + b \leq 6$

**Answer**: 3 triangles: (2,5,5), (3,4,5), (4,4,4)

---

# Part X: Essential Formulas Reference

## 10.1 Triangle Formulas

| Formula | Expression |
|---------|------------|
| Triangle inequality | $a + b > c$, $a + c > b$, $b + c > a$ |
| Third side range | $|a - b| < c < a + b$ |
| Interior angle sum | $\angle A + \angle B + \angle C = 180°$ |
| Exterior angle | $= $ sum of two non-adjacent interior angles |
| Right triangle | Two acute angles are complementary |

## 10.2 Polygon Formulas

| Formula | Expression |
|---------|------------|
| Diagonals from one vertex | $n - 3$ |
| Total diagonals | $\frac{n(n-3)}{2}$ |
| Interior angle sum | $(n-2) \cdot 180°$ |
| Exterior angle sum | $360°$ (constant) |

## 10.3 Regular Polygon Formulas

| Formula | Expression |
|---------|------------|
| Each interior angle | $\frac{(n-2) \cdot 180°}{n}$ |
| Each exterior angle | $\frac{360°}{n}$ |
| Number of sides | $n = \frac{360°}{\text{exterior angle}}$ |

## 10.4 Special Line Segments

| Segment | Definition | Concurrency Point |
|---------|------------|-------------------|
| Altitude | Perpendicular to opposite side | Orthocenter |
| Median | To midpoint of opposite side | Centroid (2:1 ratio) |
| Angle Bisector | Bisects interior angle | Incenter |

## 10.5 Isosceles Triangle Properties

| Property | Description |
|----------|-------------|
| Equal sides | Two sides (legs) are equal |
| Equal angles | Two base angles are equal |
| Three-in-one | Altitude, median, and angle bisector to base coincide |

## 10.6 Pythagorean Theorem

| Formula | Expression |
|---------|------------|
| Pythagorean theorem | $a^2 + b^2 = c^2$ (right triangle) |
| Converse | If $a^2 + b^2 = c^2$, then right triangle |
| Acute triangle test | $a^2 + b^2 > c^2$ |
| Obtuse triangle test | $a^2 + b^2 < c^2$ |

**Common Pythagorean Triples**: (3,4,5), (5,12,13), (8,15,17), (7,24,25)

## 10.7 Triangle Congruence Criteria

| Criterion | Condition |
|-----------|-----------|
| SSS | Three sides equal |
| SAS | Two sides and included angle equal |
| ASA | Two angles and included side equal |
| AAS | Two angles and non-included side equal |
| HL | Hypotenuse and leg equal (right triangles only) |

## 10.8 Triangle Similarity Criteria

| Criterion | Condition |
|-----------|-----------|
| AA | Two angles equal |
| SAS | Two sides proportional and included angle equal |
| SSS | Three sides proportional |

## 10.9 Area Formulas

| Formula | Expression |
|---------|------------|
| Basic area | $S = \frac{1}{2} \times \text{base} \times \text{height}$ |
| Heron's formula | $S = \sqrt{s(s-a)(s-b)(s-c)}$, $s = \frac{a+b+c}{2}$ |

## 10.10 Other Important Formulas

| Formula | Expression |
|---------|------------|
| Midsegment theorem | $DE = \frac{1}{2}BC$, $DE \parallel BC$ |
| Angle bisector theorem | $\frac{BD}{DC} = \frac{AB}{AC}$ |
| Right triangle altitude | $CD^2 = AD \cdot BD$ |
| Leg projection | $AC^2 = AD \cdot AB$, $BC^2 = BD \cdot AB$ |

---

**End of Chapter 11**

*This document covers triangles and polygons from beginner level through National Junior High Math League competition level.*
