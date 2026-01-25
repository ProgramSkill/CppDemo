/*
 * 算法：动态规划空间优化技巧
 * 类型：递推优化
 * 时间复杂度：不变
 * 空间复杂度：从 O(n²) 优化到 O(n) 甚至 O(1)
 *
 * 核心思想：
 *   观察DP状态转移的依赖关系，只保留必要的状态
 *
 * 常见优化方式：
 *   1. 滚动数组：状态只依赖前几行
 *   2. 一维数组：状态只依赖前一个状态
 *   3. 状态压缩：用位运算压缩状态
 *   4. 矩阵快速幂：O(n) → O(log n)
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <bitset>
using namespace std;

// ==========================================
// 优化1：滚动数组（斐波那契数列）
// ==========================================

// 未优化：O(n) 空间
long long fibonacci_no_opt(int n) {
    if (n <= 1) return n;
    vector<long long> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
    }

    return dp[n];
}

// 优化后：O(1) 空间
long long fibonacci_opt(int n) {
    if (n <= 1) return n;

    long long prev2 = 0, prev1 = 1;
    for (int i = 2; i <= n; i++) {
        long long curr = prev1 + prev2;
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

// ==========================================
// 优化2：一维数组（0/1 背包）
// ==========================================

// 未优化：O(n×W) 空间
int knapsack_2D(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= W; j++) {
            dp[i][j] = dp[i-1][j];
            if (j >= weights[i-1]) {
                dp[i][j] = max(dp[i][j], dp[i-1][j - weights[i-1]] + values[i-1]);
            }
        }
    }

    return dp[n][W];
}

// 优化后：O(W) 空间
int knapsack_1D(vector<int>& weights, vector<int>& values, int W) {
    vector<int> dp(W + 1, 0);

    for (size_t i = 0; i < weights.size(); i++) {
        for (int j = W; j >= weights[i]; j--) {
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

// ==========================================
// 优化3：滚动数组（最长公共子序列）
// ==========================================

// 未优化：O(m×n) 空间
int lcs_2D(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[m][n];
}

// 优化后：O(min(m,n)) 空间
int lcs_1D(string text1, string text2) {
    int m = text1.length(), n = text2.length();

    // 确保 n 较小
    if (m < n) {
        swap(text1, text2);
        swap(m, n);
    }

    vector<int> prev(n + 1, 0);
    vector<int> curr(n + 1, 0);

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                curr[j] = prev[j-1] + 1;
            } else {
                curr[j] = max(prev[j], curr[j-1]);
            }
        }
        swap(prev, curr);
    }

    return prev[n];
}

// ==========================================
// 优化4：状态压缩（旅行商问题TSP）
// ==========================================

// 传统DP：O(n² × 2^n) 空间
int tsp_traditional(vector<vector<int>>& graph) {
    int n = graph.size();
    int FULL_MASK = (1 << n) - 1;

    // dp[mask][i] = 访问过mask中的城市，最后在城市i的最短路径
    vector<vector<int>> dp(1 << n, vector<int>(n, INT_MAX / 2));

    // 从城市0出发
    dp[1][0] = 0;

    for (int mask = 1; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            if (!(mask & (1 << i))) continue;

            int prev_mask = mask ^ (1 << i);
            for (int j = 0; j < n; j++) {
                if (prev_mask & (1 << j)) {
                    dp[mask][i] = min(dp[mask][i], dp[prev_mask][j] + graph[j][i]);
                }
            }
        }
    }

    // 回到城市0
    int result = INT_MAX;
    for (int i = 1; i < n; i++) {
        result = min(result, dp[FULL_MASK][i] + graph[i][0]);
    }

    return result;
}

// ==========================================
// 优化5：前缀和优化（区间查询）
// ==========================================

class PrefixSum {
private:
    vector<int> prefix;

public:
    PrefixSum(vector<int>& arr) {
        prefix.resize(arr.size() + 1, 0);
        for (size_t i = 0; i < arr.size(); i++) {
            prefix[i + 1] = prefix[i] + arr[i];
        }
    }

    // O(1) 查询区间和
    int query(int l, int r) {
        return prefix[r + 1] - prefix[l];
    }
};

// ==========================================
// 优化6：奇偶滚动（棋盘DP）
// ==========================================

// 棋盘从左上到右下的最大路径和
int maxPathSum_oddEven(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();

    // 只需要两行：奇数行和偶数行
    vector<vector<int>> dp(2, vector<int>(n + 2, 0));

    for (int i = 1; i <= m; i++) {
        int curr = i % 2;
        int prev = 1 - curr;

        for (int j = 1; j <= n; j++) {
            dp[curr][j] = grid[i-1][j-1] + max(dp[prev][j], dp[curr][j-1]);
        }
    }

    return dp[m % 2][n];
}

// ==========================================
// 优化7：差分数组（区间修改）
// ==========================================

class DifferenceArray {
private:
    vector<int> diff;

public:
    DifferenceArray(int n) : diff(n + 1, 0) {}

    // O(1) 区间修改
    void add(int l, int r, int val) {
        diff[l] += val;
        diff[r + 1] -= val;
    }

    // O(n) 构建最终数组
    vector<int> build() {
        vector<int> result(diff.size() - 1);
        result[0] = diff[0];
        for (size_t i = 1; i < result.size(); i++) {
            result[i] = result[i-1] + diff[i];
        }
        return result;
    }
};

// ==========================================
// 优化8：斜率优化 / 决策单调性
// ==========================================

// 斜率优化示例：将 O(n²) 优化到 O(n)
// dp[i] = min(dp[j] + cost(j+1, i))
// 当 cost 满足四边形不等式时，决策点单调

// ==========================================
// 打印对比
// ==========================================

void printComparison(const string& name, int space_before, int space_after) {
    cout << name << ":" << endl;
    cout << "  优化前空间：O(" << space_before << ")" << endl;
    cout << "  优化后空间：O(" << space_after << ")" << endl;
    cout << "  优化比例：" << (double)space_before / space_after << "x" << endl;
}

int main() {
    cout << "========== 动态规划空间优化技巧 ==========" << endl;

    // 斐波那契数列对比
    cout << "\n【1. 滚动数组：斐波那契数列】" << endl;
    int n = 100;
    cout << "F(" << n << ") = " << fibonacci_opt(n) << endl;
    printComparison("斐波那契", n, 2);

    // 背包问题对比
    cout << "\n【2. 一维数组：0/1 背包】" << endl;
    vector<int> weights = {2, 3, 4, 5};
    vector<int> values = {3, 4, 5, 6};
    int W = 10;

    int result_2d = knapsack_2D(weights, values, W);
    int result_1d = knapsack_1D(weights, values, W);

    cout << "二维DP：" << result_2d << endl;
    cout << "一维DP：" << result_1d << endl;
    printComparison("0/1 背包", weights.size() * W, W);

    // LCS 对比
    cout << "\n【3. 滚动数组：最长公共子序列】" << endl;
    string s1 = "abcde", s2 = "ace";

    int lcs_2d_result = lcs_2D(s1, s2);
    int lcs_1d_result = lcs_1D(s1, s2);

    cout << "LCS长度：" << lcs_2d_result << " / " << lcs_1d_result << endl;
    printComparison("LCS", s1.length() * s2.length(), min(s1.length(), s2.length()));

    // 前缀和示例
    cout << "\n【4. 前缀和：区间查询】" << endl;
    vector<int> arr = {1, 3, 5, 7, 9, 11};
    PrefixSum ps(arr);

    cout << "数组：[1, 3, 5, 7, 9, 11]" << endl;
    cout << "区间[1, 3]的和：" << ps.query(1, 3) << " (3+5+7=15)" << endl;
    cout << "区间[2, 5]的和：" << ps.query(2, 5) << " (5+7+9+11=32)" << endl;

    // 差分数组示例
    cout << "\n【5. 差分数组：区间修改】" << endl;
    DifferenceArray da(10);

    // 在区间[2, 5]加3
    da.add(2, 5, 3);
    // 在区间[4, 7]加2
    da.add(4, 7, 2);

    vector<int> result = da.build();
    cout << "操作后数组：";
    for (int val : result) cout << val << " ";
    cout << endl;

    // 奇偶滚动示例
    cout << "\n【6. 奇偶滚动：棋盘DP】" << endl;
    vector<vector<int>> grid = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int max_sum = maxPathSum_oddEven(grid);
    cout << "棋盘最大路径和：" << max_sum << endl;
    cout << "路径：1→4→7→8→9 = 29" << endl;

    // 优化技巧总结
    cout << "\n【优化技巧总结】" << endl;
    cout << "┌─────────────┬────────────┬─────────────┐" << endl;
    cout << "│   优化类型    │  适用场景   │  空间优化    │" << endl;
    cout << "├─────────────┼────────────┼─────────────┤" << endl;
    cout << "│  滚动数组    │ 依赖前几行  │ O(n²)→O(n)  │" << endl;
    cout << "│  一维数组    │ 依赖前一行  │ O(n×W)→O(W) │" << endl;
    cout << "│  状态压缩    │ 状态数量少  │ O(2^n)具体值 │" << endl;
    cout << "│  前缀和      │ 区间查询    │ O(n)查询    │" << endl;
    cout << "│  差分数组    │ 区间修改    │ O(1)修改    │" << endl;
    cout << "│  奇偶滚动    │ 棋盘问题    │ O(mn)→O(n)  │" << endl;
    cout << "│  斜率优化    │ 特定DP转移  │ O(n²)→O(n)  │" << endl;
    cout << "└─────────────┴────────────┴─────────────┘" << endl;

    // 优化原则
    cout << "\n【优化原则】" << endl;
    cout << "1. 分析状态依赖关系" << endl;
    cout << "   - dp[i] 依赖哪些之前的状态？" << endl;
    cout << "   - 是否可以丢弃更早的状态？" << endl;
    cout << endl;
    cout << "2. 选择合适的优化方式" << endl;
    cout << "   - 只依赖前一个：滚动变量" << endl;
    cout << "   - 只依赖前几个：滚动数组" << endl;
    cout << "   - 只依赖前一行：一维数组" << endl;
    cout << endl;
    cout << "3. 注意更新顺序" << endl;
    cout << "   - 从前往后 vs 从后往前" << endl;
    cout << "   - 0/1背包：从后往前（避免重复）" << endl;
    cout << "   - 完全背包：从前往后（允许重复）" << endl;
    cout << endl;
    cout << "4. 确保正确性" << endl;
    cout << "   - 小数据测试" << endl;
    cout << "   - 对比优化前后结果" << endl;
    cout << "   - 边界条件检查" << endl;

    // 性能对比
    cout << "\n【性能对比：n=1000】" << endl;
    int large_n = 1000;

    cout << "斐波那契数列：" << endl;
    clock_t start, end;

    start = clock();
    fibonacci_no_opt(large_n);
    end = clock();
    cout << "  未优化：用时 " << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    start = clock();
    fibonacci_opt(large_n);
    end = clock();
    cout << "  优化后：用时 " << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    return 0;
}
