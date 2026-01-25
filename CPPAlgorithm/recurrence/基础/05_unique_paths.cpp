/*
 * 算法：网格路径 (Unique Paths)
 * 类型：递推 - 二维递推
 * 时间复杂度：O(m×n)
 * 空间复杂度：O(m×n) 或 O(n) 优化
 *
 * 题目：从 m×n 网格的左上角走到右下角，每次只能向右或向下移动
 *      求有多少种不同的路径？
 *
 * 递推关系：
 *   dp[i][j] = dp[i-1][j] + dp[i][j-1]
 *   dp[0][j] = dp[i][0] = 1  (第一行和第一列只有一种路径)
 *
 * 思路：
 *   到达位置 (i, j) 只能从：
 *   - 上方 (i-1, j) 向下移动一步
 *   - 左方 (i, j-1) 向右移动一步
 *   因此路径数 = 从上方来的路径 + 从左方来的路径
 */

#include <iostream>
#include <vector>
using namespace std;

// 方法1：基本递推（二维DP）
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 1));

    // dp[i][j] = 从 (0,0) 到 (i,j) 的路径数
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            // 递推：从上方来 + 从左方来
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
        }
    }

    return dp[m-1][n-1];
}

// 方法2：空间优化（一维数组）
int uniquePaths_optimized(int m, int n) {
    vector<int> dp(n, 1);  // 初始化为1（第一行）

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            // dp[j] 保存的是上方值（上一行）
            // dp[j-1] 是左方值（当前行已计算）
            dp[j] += dp[j-1];
        }
    }

    return dp[n-1];
}

// 方法3：组合数学公式
// 从 (0,0) 到 (m-1, n-1) 需要移动 m-1 次向下，n-1 次向右
// 总共 m+n-2 步，选择其中的 m-1 步向下（或 n-1 步向右）
// C(m+n-2, m-1) = C(m+n-2, n-1)

long long combination(long long n, long long k) {
    if (k > n - k) k = n - k;  // 利用对称性

    long long result = 1;
    for (long long i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }

    return result;
}

int uniquePaths_combination(int m, int n) {
    return combination(m + n - 2, m - 1);
}

// 进阶：带障碍的网格路径
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    int m = obstacleGrid.size();
    int n = obstacleGrid[0].size();

    vector<vector<long long>> dp(m, vector<long long>(n, 0));

    // 初始化起点
    dp[0][0] = (obstacleGrid[0][0] == 0) ? 1 : 0;

    // 初始化第一行
    for (int j = 1; j < n; j++) {
        if (obstacleGrid[0][j] == 0) {
            dp[0][j] = dp[0][j-1];
        }
    }

    // 初始化第一列
    for (int i = 1; i < m; i++) {
        if (obstacleGrid[i][0] == 0) {
            dp[i][0] = dp[i-1][0];
        }
    }

    // 递推
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[i][j] == 0) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
    }

    return dp[m-1][n-1];
}

// 进阶：可以走1到k步
int uniquePaths_kSteps(int m, int n, int k) {
    vector<vector<int>> dp(m, vector<int>(n, 0));
    dp[0][0] = 1;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int step = 1; step <= k; step++) {
                // 向下移动 step 步
                if (i - step >= 0) {
                    dp[i][j] += dp[i - step][j];
                }
                // 向右移动 step 步
                if (j - step >= 0) {
                    dp[i][j] += dp[i][j - step];
                }
            }
        }
    }

    return dp[m-1][n-1];
}

// 打印DP表（用于调试和理解）
void printDPTable(const vector<vector<int>>& dp) {
    for (size_t i = 0; i < dp.size(); i++) {
        for (size_t j = 0; j < dp[i].size(); j++) {
            cout << dp[i][j] << "\t";
        }
        cout << endl;
    }
}

int main() {
    cout << "========== 网格路径问题 ==========" << endl;

    // 基本测试
    cout << "\n【基本测试：不同大小网格的路径数】" << endl;
    cout << "网格大小(m×n)\t路径数" << endl;
    cout << "----------------------------" << endl;

    vector<pair<int, int>> test_cases = {{3, 7}, {3, 2}, {7, 3}, {3, 3}, {5, 5}};

    for (auto& grid : test_cases) {
        int paths = uniquePaths(grid.first, grid.second);
        cout << grid.first << "×" << grid.second << "\t\t" << paths << endl;
    }

    // 方法对比
    cout << "\n【方法对比：3×7 网格】" << endl;
    int m = 3, n = 7;
    cout << "二维递推：\t" << uniquePaths(m, n) << endl;
    cout << "空间优化：\t" << uniquePaths_optimized(m, n) << endl;
    cout << "组合公式：\t" << uniquePaths_combination(m, n) << endl;

    // DP表演示（3×3网格）
    cout << "\n【DP表演示：3×3 网格】" << endl;
    vector<vector<int>> dp(3, vector<int>(3, 1));

    cout << "初始化（第一行和第一列都是1）：" << endl;
    printDPTable(dp);

    cout << "\n递推过程：" << endl;
    for (int i = 1; i < 3; i++) {
        for (int j = 1; j < 3; j++) {
            dp[i][j] = dp[i-1][j] + dp[i][j-1];
            cout << "\n计算 dp[" << i << "][" << j << "]:" << endl;
            cout << "  = dp[" << i-1 << "][" << j << "] + dp[" << i << "][" << j-1 << "]" << endl;
            cout << "  = " << dp[i-1][j] << " + " << dp[i][j-1] << " = " << dp[i][j] << endl;
            printDPTable(dp);
        }
    }

    // 可视化路径（2×2网格）
    cout << "\n【路径可视化：2×2 网格的 2 条路径】" << endl;
    cout << "路径1: → → ↓ ↓" << endl;
    cout << "路径2: ↓ ↓ → →" << endl;

    // 组合公式解释
    cout << "\n【组合公式解释】" << endl;
    cout << "对于 3×7 网格：" << endl;
    cout << "  需要移动：2次向下 + 6次向右 = 8步" << endl;
    cout << "  从8步中选择2步向下：C(8, 2) = " << combination(8, 2) << endl;
    cout << "  或者从8步中选择6步向右：C(8, 6) = " << combination(8, 6) << endl;

    // 带障碍的网格
    cout << "\n【进阶：带障碍的网格】" << endl;
    vector<vector<int>> obstacleGrid = {
        {0, 0, 0},
        {0, 1, 0},  // 1 表示障碍
        {0, 0, 0}
    };

    cout << "障碍网格（1表示障碍）：" << endl;
    printDPTable(obstacleGrid);

    int paths_with_obstacle = uniquePathsWithObstacles(obstacleGrid);
    cout << "路径数（避开障碍）：" << paths_with_obstacle << endl;

    // 可以走多步的情况
    cout << "\n【进阶：每次可以走 1-2 步】" << endl;
    cout << "3×3 网格，每次可以走1-2步：" << endl;
    cout << "  路径数 = " << uniquePaths_kSteps(3, 3, 2) << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "方法\t\t\t时间复杂度\t空间复杂度" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "二维递推\t\tO(m×n)\t\tO(m×n)" << endl;
    cout << "空间优化\t\tO(m×n)\t\tO(n)" << endl;
    cout << "组合公式\t\tO(min(m,n))\tO(1)" << endl;

    // 实际应用
    cout << "\n【实际应用】" << endl;
    cout << "1. 机器人路径规划" << endl;
    cout << "2. 城市街道导航（只允许向东和向北）" << endl;
    cout << "3. 排列组合问题" << endl;
    cout << "4. 动态规划基础训练" << endl;

    return 0;
}
