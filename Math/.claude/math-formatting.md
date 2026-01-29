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
