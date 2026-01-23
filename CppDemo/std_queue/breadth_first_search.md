# 广度优先搜索（BFS）与 std::queue

## 目录

1. [概述](#概述)
2. [为什么使用 queue](#为什么使用-queue)
3. [基本 BFS 模板](#基本-bfs-模板)
4. [图的 BFS 遍历](#图的-bfs-遍历)
5. [树的层序遍历](#树的层序遍历)
6. [应用场景](#应用场景)
7. [典型例题](#典型例题)
8. [注意事项](#注意事项)

---

## 概述

**广度优先搜索**（**Breadth-First Search**，简称 **BFS**）是一种用于遍历或搜索树/图数据结构的算法。该算法从起始节点开始，首先访问所有相邻节点，然后再依次访问这些相邻节点的邻居。

> **BFS** = **Breadth-First Search**（广度优先搜索）

### 核心思想

- **逐层访问**：按照距离起点的层数逐层访问节点
- **先入先出**：使用队列保证先发现的节点先访问
- **最短路径**：在无权图中能找到最短路径

### 与 DFS 的对比

**DFS**（**Depth-First Search**，深度优先搜索）是另一种常用的图/树遍历算法，它沿着树的深度遍历树的节点，尽可能深地搜索树的分支。

> **DFS** = **Depth-First Search**（深度优先搜索）

| 特性 | BFS | DFS |
|------|-----|-----|
| 数据结构 | Queue（队列） | Stack（栈）或递归 |
| 遍历顺序 | 层级遍历 | 深度优先 |
| 时间复杂度 | O(V + E) | O(V + E) |
| 空间复杂度 | O(V) 最坏情况 | O(h) 或 O(V) 最坏情况 |
| 内存使用 | 较大（存储一层节点） | 较小（存储一条路径） |
| 适用场景 | 最短路径、层级遍历 | 路径查找、连通性 |
| 实现方式 | 迭代 | 递归或迭代 |

**复杂度说明**：
- **V**：顶点数（Vertices）
- **E**：边数（Edges）
- **h**：树的高度（Height）
- **O(V + E)**：访问所有顶点和边
- **空间复杂度**：BFS 最坏 O(V)（当所有节点都在同一层），DFS 最坏 O(V)（线性链），通常 O(h)

---

## 为什么使用 queue

### Queue 的特性完美匹配 BFS 需求

```
访问顺序：A → B → C → D → E → F → G

        A           (第 0 层)
      / | \
     B  C  D        (第 1 层)
    /|     |\
   E F     G H      (第 2 层)

Queue 操作序列：
push(A)           队列: [A]
pop() → A         队列: []
push(B), push(C), push(D)  队列: [B, C, D]
pop() → B         队列: [C, D]
push(E), push(F)          队列: [C, D, E, F]
pop() → C         队列: [D, E, F]
pop() → D         队列: [E, F]
push(G), push(H)          队列: [E, F, G, H]
...
```

### 为什么不用其他容器？

| 容器 | 是否适合 | 原因 |
|------|---------|------|
| `std::queue` | ✅ **完美** | 先进先出，符合 BFS 层级顺序 |
| `std::stack` | ❌ 不适合 | 后进先出，实现的是 DFS |
| `std::vector` | ❌ 效率低 | 从头部删除是 O(n) |
| `std::deque` | ⚠️ 可用 | 功能满足，但 queue 语义更清晰 |
| `std::priority_queue` | ❌ 错误 | 优先队列，实现的是 Dijkstra 等 |

---

## 基本 BFS 模板

### 标准模板

```cpp
#include <queue>
#include <vector>

void bfs(int start) {
    std::queue<int> q;
    std::vector<bool> visited(n, false);

    // 1. 将起点加入队列
    q.push(start);
    visited[start] = true;

    // 2. 队列不为空时继续处理
    while (!q.empty()) {
        // 3. 取出队首元素
        int current = q.front();
        q.pop();

        // 4. 处理当前节点
        process(current);

        // 5. 遍历所有未访问的邻居
        for (int neighbor : getNeighbors(current)) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}
```

### 带距离的 BFS 模板

```cpp
#include <queue>
#include <vector>

void bfs_with_distance(int start) {
    std::queue<int> q;
    std::vector<int> distance(n, -1);  // -1 表示未访问

    q.push(start);
    distance[start] = 0;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        for (int neighbor : getNeighbors(current)) {
            if (distance[neighbor] == -1) {
                distance[neighbor] = distance[current] + 1;
                q.push(neighbor);
            }
        }
    }

    // distance[i] 表示从 start 到节点 i 的最短距离
}
```

### 带路径记录的 BFS 模板

```cpp
#include <queue>
#include <vector>

void bfs_with_path(int start, int target) {
    std::queue<int> q;
    std::vector<bool> visited(n, false);
    std::vector<int> parent(n, -1);

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        if (current == target) break;  // 找到目标

        for (int neighbor : getNeighbors(current)) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                parent[neighbor] = current;  // 记录父节点
                q.push(neighbor);
            }
        }
    }

    // 回溯路径
    if (visited[target]) {
        std::vector<int> path;
        for (int v = target; v != -1; v = parent[v]) {
            path.push_back(v);
        }
        std::reverse(path.begin(), path.end());
        // path 即为从 start 到 target 的路径
    }
}
```

---

## 图的 BFS 遍历

### 邻接表表示

```cpp
#include <iostream>
#include <queue>
#include <vector>

class Graph {
    int V;                          // 顶点数
    std::vector<std::vector<int>> adj;  // 邻接表

public:
    Graph(int v) : V(v), adj(v) {}

    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v);  // 无向图
    }

    void BFS(int start) {
        std::vector<bool> visited(V, false);
        std::queue<int> q;

        visited[start] = true;
        q.push(start);

        std::cout << "BFS 从节点 " << start << " 开始: ";

        while (!q.empty()) {
            int current = q.front();
            q.pop();

            std::cout << current << " ";

            for (int neighbor : adj[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
        std::cout << std::endl;
    }
};

int main() {
    Graph g(6);

    // 构建图
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);

    g.BFS(0);  // 输出: BFS 从节点 0 开始: 0 1 2 3 4 5

    return 0;
}
```

### 邻接矩阵表示

```cpp
#include <iostream>
#include <queue>
#include <vector>

class GraphMatrix {
    int V;
    std::vector<std::vector<int>> matrix;

public:
    GraphMatrix(int v) : V(v), matrix(v, std::vector<int>(v, 0)) {}

    void addEdge(int v, int w) {
        matrix[v][w] = 1;
        matrix[w][v] = 1;
    }

    void BFS(int start) {
        std::vector<bool> visited(V, false);
        std::queue<int> q;

        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int current = q.front();
            q.pop();

            std::cout << current << " ";

            for (int i = 0; i < V; ++i) {
                if (matrix[current][i] == 1 && !visited[i]) {
                    visited[i] = true;
                    q.push(i);
                }
            }
        }
    }
};
```

---

## 树的层序遍历

### 二叉树层序遍历

```cpp
#include <iostream>
#include <queue>

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

void levelOrder(TreeNode* root) {
    if (!root) return;

    std::queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();

        std::cout << node->val << " ";

        if (node->left) q.push(node->left);
        if (node->right) q.push(node->right);
    }
}
```

### 分层输出（每层一行）

```cpp
#include <iostream>
#include <queue>

void levelOrder_line_by_line(TreeNode* root) {
    if (!root) return;

    std::queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int levelSize = q.size();  // 当前层的节点数

        // 处理当前层的所有节点
        for (int i = 0; i < levelSize; ++i) {
            TreeNode* node = q.front();
            q.pop();

            std::cout << node->val << " ";

            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }

        std::cout << std::endl;  // 每层结束后换行
    }
}

// 输出示例：
//     1
//    / \
//   2   3
//  / \
// 4   5
//
// 输出：
// 1
// 2 3
// 4 5
```

### 之字形遍历（锯齿形层序遍历）

```cpp
#include <iostream>
#include <queue>
#include <vector>
#include <algorithm>

void zigzagLevelOrder(TreeNode* root) {
    if (!root) return;

    std::queue<TreeNode*> q;
    q.push(root);
    bool leftToRight = true;

    while (!q.empty()) {
        int levelSize = q.size();
        std::vector<int> level;

        for (int i = 0; i < levelSize; ++i) {
            TreeNode* node = q.front();
            q.pop();

            level.push_back(node->val);

            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }

        // 根据方向决定是否反转
        if (!leftToRight) {
            std::reverse(level.begin(), level.end());
        }

        for (int val : level) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        leftToRight = !leftToRight;  // 切换方向
    }
}
```

---

## 应用场景

### 1. 最短路径问题

**无权图的最短路径**

```cpp
#include <queue>
#include <vector>

int shortestPath(std::vector<std::vector<int>>& graph, int start, int end) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::queue<int> q;
    std::vector<int> distance(n, 0);

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        if (current == end) {
            return distance[current];
        }

        for (int neighbor : graph[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                distance[neighbor] = distance[current] + 1;
                q.push(neighbor);
            }
        }
    }

    return -1;  // 无法到达
}
```

### 2. 连通性问题

**判断图是否连通**

```cpp
#include <queue>
#include <vector>

bool isConnected(std::vector<std::vector<int>>& graph) {
    int n = graph.size();
    if (n == 0) return true;

    std::vector<bool> visited(n, false);
    std::queue<int> q;

    q.push(0);
    visited[0] = true;
    int count = 1;

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        for (int neighbor : graph[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                count++;
                q.push(neighbor);
            }
        }
    }

    return count == n;  // 所有节点都被访问
}
```

### 3. 拓扑排序（Kahn 算法）

```cpp
#include <queue>
#include <vector>

std::vector<int> topologicalSort(int numCourses,
                                  std::vector<std::pair<int, int>>& prerequisites) {
    // 构建图和入度数组
    std::vector<std::vector<int>> graph(numCourses);
    std::vector<int> inDegree(numCourses, 0);

    for (auto& prereq : prerequisites) {
        int course = prereq.first;
        int prerequisite = prereq.second;
        graph[prerequisite].push_back(course);
        inDegree[course]++;
    }

    // 将入度为 0 的节点加入队列
    std::queue<int> q;
    for (int i = 0; i < numCourses; ++i) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    std::vector<int> result;
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        result.push_back(current);

        for (int neighbor : graph[current]) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    // 如果所有课程都能学完，返回结果；否则返回空
    if (result.size() == numCourses) {
        return result;
    }
    return {};
}
```

### 4. 网格 BFS

**岛屿数量**

```cpp
#include <queue>
#include <vector>

int numIslands(std::vector<std::vector<char>>& grid) {
    if (grid.empty()) return 0;

    int m = grid.size();
    int n = grid[0].size();
    int count = 0;

    std::queue<std::pair<int, int>> q;

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == '1') {
                count++;
                grid[i][j] = '0';
                q.push({i, j});

                // BFS 标记整个岛屿
                while (!q.empty()) {
                    auto [x, y] = q.front();
                    q.pop();

                    // 四个方向
                    int dx[] = {0, 0, 1, -1};
                    int dy[] = {1, -1, 0, 0};

                    for (int k = 0; k < 4; ++k) {
                        int nx = x + dx[k];
                        int ny = y + dy[k];

                        if (nx >= 0 && nx < m && ny >= 0 && ny < n &&
                            grid[nx][ny] == '1') {
                            grid[nx][ny] = '0';
                            q.push({nx, ny});
                        }
                    }
                }
            }
        }
    }

    return count;
}
```

---

## 典型例题

### 例题 1：二进制矩阵中的最短路径

```cpp
#include <queue>
#include <vector>

int shortestPathBinaryMatrix(std::vector<std::vector<int>>& grid) {
    int n = grid.size();
    if (grid[0][0] == 1 || grid[n-1][n-1] == 1) return -1;

    std::queue<std::pair<int, int>> q;
    q.push({0, 0});
    grid[0][0] = 1;  // 标记访问
    int pathLength = 1;

    // 8 个方向
    int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

    while (!q.empty()) {
        int size = q.size();

        for (int i = 0; i < size; ++i) {
            auto [x, y] = q.front();
            q.pop();

            if (x == n - 1 && y == n - 1) {
                return pathLength;
            }

            for (int k = 0; k < 8; ++k) {
                int nx = x + dx[k];
                int ny = y + dy[k];

                if (nx >= 0 && nx < n && ny >= 0 && ny < n &&
                    grid[nx][ny] == 0) {
                    grid[nx][ny] = 1;
                    q.push({nx, ny});
                }
            }
        }
        pathLength++;
    }

    return -1;
}
```

### 例题 2：单词阶梯

```cpp
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

int ladderLength(std::string beginWord, std::string endWord,
                 std::vector<std::string>& wordList) {
    std::unordered_set<std::string> wordSet(wordList.begin(), wordList.end());

    if (wordSet.find(endWord) == wordSet.end()) {
        return 0;  // endWord 不在列表中
    }

    std::queue<std::string> q;
    q.push(beginWord);

    int level = 1;

    while (!q.empty()) {
        int size = q.size();

        for (int i = 0; i < size; ++i) {
            std::string word = q.front();
            q.pop();

            if (word == endWord) {
                return level;
            }

            // 尝试改变每个字符
            for (int j = 0; j < word.length(); ++j) {
                char original = word[j];

                for (char c = 'a'; c <= 'z'; ++c) {
                    word[j] = c;

                    if (wordSet.find(word) != wordSet.end()) {
                        wordSet.erase(word);  // 标记访问
                        q.push(word);
                    }
                }

                word[j] = original;  // 恢复
            }
        }
        level++;
    }

    return 0;
}
```

### 例题 3：课程表（课程顺序）

```cpp
#include <queue>
#include <vector>

std::vector<int> findOrder(int numCourses,
                            std::vector<std::vector<int>>& prerequisites) {
    std::vector<std::vector<int>> graph(numCourses);
    std::vector<int> inDegree(numCourses, 0);

    for (auto& prereq : prerequisites) {
        int course = prereq[0];
        int prerequisite = prereq[1];
        graph[prerequisite].push_back(course);
        inDegree[course]++;
    }

    std::queue<int> q;
    for (int i = 0; i < numCourses; ++i) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }

    std::vector<int> result;
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        result.push_back(current);

        for (int neighbor : graph[current]) {
            inDegree[neighbor]--;
            if (inDegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    if (result.size() == numCourses) {
        return result;
    }
    return {};
}
```

---

## 注意事项

### 1. 避免重复访问

```cpp
// ❌ 错误：可能重复入队
if (!visited[neighbor]) {
    q.push(neighbor);
}
visited[neighbor] = true;

// ✅ 正确：入队时立即标记
if (!visited[neighbor]) {
    visited[neighbor] = true;  // 先标记
    q.push(neighbor);
}
```

### 2. 队列中不要存储引用

```cpp
// ❌ 危险：存储指针或引用可能失效
std::queue<TreeNode*> q;
q.push(root);
// 如果树结构被修改，指针可能失效

// ✅ 安全：存储值或索引
std::queue<int> q;
q.push(root->val);
```

### 3. BFS 空间复杂度

- 最坏情况：队列存储一层所有节点
- 二叉树最底层：O(n/2) ≈ O(n)
- 图的 BFS：O(min(V, E))

### 4. 双向 BFS 优化

对于大规模图，可以使用双向 BFS：

```cpp
#include <queue>
#include <unordered_set>
#include <string>

int bidirectionalBFS(std::string start, std::string end,
                     std::unordered_set<std::string>& wordSet) {
    std::unordered_set<std::string> beginSet, endSet, visited;
    beginSet.insert(start);
    endSet.insert(end);

    int len = 1;

    while (!beginSet.empty() && !endSet.empty()) {
        // 总是扩展较小的集合
        if (beginSet.size() > endSet.size()) {
            std::swap(beginSet, endSet);
        }

        std::unordered_set<std::string> temp;

        for (std::string word : beginSet) {
            if (endSet.find(word) != endSet.end()) {
                return len;
            }

            visited.insert(word);

            // 生成所有可能的下一个单词
            for (int i = 0; i < word.length(); ++i) {
                char c = word[i];
                for (char ch = 'a'; ch <= 'z'; ++ch) {
                    word[i] = ch;
                    if (wordSet.find(word) != wordSet.end() &&
                        visited.find(word) == visited.end()) {
                        temp.insert(word);
                    }
                }
                word[i] = c;
            }
        }

        beginSet = temp;
        len++;
    }

    return 0;
}
```

### 5. 记录访问时间

```cpp
struct Node {
    int id;
    int distance;  // 记录距离
    int timestamp; // 记录访问时间
};

void bfs_with_timestamp(int start) {
    std::queue<Node> q;
    std::vector<bool> visited(n, false);
    int timestamp = 0;

    q.push({start, 0, timestamp++});
    visited[start] = true;

    while (!q.empty()) {
        Node current = q.front();
        q.pop();

        for (int neighbor : adj[current.id]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push({neighbor, current.distance + 1, timestamp++});
            }
        }
    }
}
```

---

## 总结

### BFS 何时使用

✅ **使用 BFS 的情况**：
- 寻找最短路径（无权图）
- 需要按层级遍历
- 需要逐步扩展搜索
- 拓扑排序
- 连通性检查

❌ **不使用 BFS 的情况**：
- 需要找到所有可能的解（用 DFS）
- 内存受限（BFS 空间较大）
- 图很深但分支较少（DFS 更合适）

### Queue 在 BFS 中的优势

1. **天然的层级顺序**：先入先出保证层级遍历
2. **O(1) 操作**：push 和 pop 都是常数时间
3. **内存效率**：不需要递归栈空间
4. **易于实现**：代码简洁直观

### 性能对比

| 操作 | std::queue | 时间复杂度 |
|------|-----------|-----------|
| push() | ✅ | O(1) |
| pop() | ✅ | O(1) |
| front() | ✅ | O(1) |
| empty() | ✅ | O(1) |
| size() | ✅ | O(1) |

---

## 参考资料

- [LeetCode - BFS Tag](https://leetcode.com/tag/breadth-first-search/)
- [GeeksforGeeks - BFS Algorithm](https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/)
- [C++ Reference - std::queue](https://en.cppreference.com/w/cpp/container/queue)
