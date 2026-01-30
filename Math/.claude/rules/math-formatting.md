# Math Document Formatting Guidelines

## Typora Settings

To enable inline math rendering in Typora:

1. Open Preferences (File > Preferences or `Ctrl + ,`)
2. Navigate to **Markdown** section
3. Enable **Inline Math** option
4. Restart Typora

## GitHub Math Rendering

### Problem

When `$$` and formula content are on the same line, GitHub does not render properly:

```markdown
<!-- Does NOT work on GitHub -->
$$\begin{cases} x + y = 5 \\ x - y = 1 \end{cases}$$
```

### Solution

Use block-level format with `$$` on separate lines:

```markdown
<!-- Works on both Typora and GitHub -->
$$
\begin{cases}
x + y = 5 \\
x - y = 1
\end{cases}
$$
```

### Key Rules

1. `$$` must be on its own line
2. Each equation in `\begin{cases}` should be on its own line
3. Use `\\` for line breaks within the formula

## Compatibility Summary

| Format | Typora | GitHub |
|--------|--------|--------|
| `$x$` inline math | Requires setting | Works |
| `$$...$$` same line | Works | NOT work |
| `$$` separate lines | Works | Works |

## Indentation and Code Blocks

### Problem

In Markdown, lines starting with 4 spaces or 1 tab are treated as code blocks (preformatted text). This causes `$...$` LaTeX formulas to display as raw text with dollar signs visible.

**Example of problematic formatting:**

```markdown
(2) Since $AD$ is an altitude, $\angle ADB = 90°$
    In right triangle $ABD$:
    $\angle BAD = 90° - \angle B = 90° - 50° = 40°$
```

The indented lines will render as code, showing `$\angle BAD = ...$` literally instead of the math formula.

### Solution

Remove leading indentation from math content, or use blank lines between paragraphs:

```markdown
(2) Since $AD$ is an altitude, $\angle ADB = 90°$

In right triangle $ABD$:

$\angle BAD = 90° - \angle B = 90° - 50° = 40°$
```

### Key Rules

1. Avoid 4+ spaces at the start of lines containing `$...$` formulas
2. Use blank lines to separate paragraphs instead of indentation
3. If indentation is needed for visual structure, use 2-3 spaces maximum

## Unicode Symbols in Code Blocks

### Advantage

Unicode math symbols inside fenced code blocks (``` `) display correctly on all platforms (GitHub, Typora, etc.) without any rendering issues.

### Common Unicode Symbols

| Symbol | Meaning |
|--------|---------|
| △ | Triangle |
| ∠ | Angle |
| ° | Degree |
| ≅ | Congruent |
| ∥ | Parallel |
| ⊥ | Perpendicular |
| ∴ | Therefore |
| ∵ | Because |
| ½ | One half |

### Example

```
Given: In △ABC, D is the midpoint of AB, E is the midpoint of AC
Prove: DE ∥ BC and DE = ½BC

Proof: In △ADE and △CFE:
       AE = CE (E is midpoint of AC)
       ∠AED = ∠CEF (vertical angles)
       DE = FE (construction)

       Therefore △ADE ≅ △CFE (SAS)
```

This displays correctly everywhere because Unicode symbols are plain text characters.
