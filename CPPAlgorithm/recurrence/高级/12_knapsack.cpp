/*
 * 算法：背包问题 (Knapsack Problem)
 * 类型：递推 - 动态规划
 * 时间复杂度：O(n × W)
 * 空间复杂度：O(W) 优化后
 *
 * 题目：给定 n 个物品，每个物品有重量和价值
 *      背包容量为 W，求能装入的最大价值
 *
 * 递推关系（0/1 背包）：
 *   dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
 *   dp[j] 表示容量为 j 时的最大价值
 *
 * 变种：
 *   - 0/1 背包：每个物品最多选一次
 *   - 完全背包：每个物品可以选无限次
 *   - 多重背包：每个物品有数量限制
 */

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 方法1：0/1 背包问题（二维DP）
int knapsack_01_2D(vector<int>& weights, vector<int>& values, int W) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= W; j++) {
            // 不选第 i 个物品
            dp[i][j] = dp[i-1][j];

            // 选第 i 个物品（如果能装下）
            if (j >= weights[i-1]) {
                dp[i][j] = max(dp[i][j], dp[i-1][j - weights[i-1]] + values[i-1]);
            }
        }
    }

    return dp[n][W];
}

// 方法2：0/1 背包问题（一维优化）
int knapsack_01(vector<int>& weights, vector<int>& values, int W) {
    vector<int> dp(W + 1, 0);

    for (size_t i = 0; i < weights.size(); i++) {
        // 从后往前更新（避免重复使用）
        for (int j = W; j >= weights[i]; j--) {
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

// 方法3：完全背包问题（每个物品可以选无限次）
int knapsack_complete(vector<int>& weights, vector<int>& values, int W) {
    vector<int> dp(W + 1, 0);

    for (size_t i = 0; i < weights.size(); i++) {
        // 从前往后更新（允许重复使用）
        for (int j = weights[i]; j <= W; j++) {
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }

    return dp[W];
}

// 方法4：多重背包问题（每个物品有限次）
int knapsack_bounded(vector<int>& weights, vector<int>& values, vector<int>& counts, int W) {
    vector<int> dp(W + 1, 0);

    for (size_t i = 0; i < weights.size(); i++) {
        // 二进制优化
        int k = 1;
        while (k < counts[i]) {
            int w = k * weights[i];
            int v = k * values[i];

            for (int j = W; j >= w; j--) {
                dp[j] = max(dp[j], dp[j - w] + v);
            }

            counts[i] -= k;
            k *= 2;
        }

        // 剩余部分
        for (int j = W; j >= weights[i] * counts[i]; j--) {
            dp[j] = max(dp[j], dp[j - weights[i] * counts[i]] + values[i] * counts[i]);
        }
    }

    return dp[W];
}

// 方法5：返回选择的物品
vector<int> knapsack_01_withItems(vector<int>& weights, vector<int>& values, int W) {
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

    // 回溯找出选择的物品
    vector<int> items;
    int j = W;
    for (int i = n; i > 0; i--) {
        if (dp[i][j] != dp[i-1][j]) {
            items.push_back(i - 1);
            j -= weights[i - 1];
        }
    }

    reverse(items.begin(), items.end());
    return items;
}

// 打印DP表
void printDPTable(const vector<vector<int>>& dp) {
    for (size_t i = 0; i < dp.size(); i++) {
        for (size_t j = 0; j < dp[i].size(); j++) {
            cout << dp[i][j] << "\t";
        }
        cout << endl;
    }
}

int main() {
    cout << "========== 背包问题 ==========" << endl;

    // 0/1 背包测试
    cout << "\n【0/1 背包问题】" << endl;
    vector<int> weights = {2, 3, 4, 5};
    vector<int> values = {3, 4, 5, 6};
    int W = 8;

    cout << "物品信息：" << endl;
    cout << "序号\t重量\t价值" << endl;
    cout << "----------------------------" << endl;
    for (size_t i = 0; i < weights.size(); i++) {
        cout << i << "\t" << weights[i] << "\t" << values[i] << endl;
    }
    cout << "\n背包容量：" << W << endl;

    int maxValue = knapsack_01(weights, values, W);
    cout << "\n最大价值：" << maxValue << endl;

    // 获取选择的物品
    vector<int> items = knapsack_01_withItems(weights, values, W);
    cout << "选择的物品：";
    int totalWeight = 0, totalValue = 0;
    for (int idx : items) {
        cout << idx << " ";
        totalWeight += weights[idx];
        totalValue += values[idx];
    }
    cout << endl;
    cout << "总重量：" << totalWeight << "，总价值：" << totalValue << endl;

    // DP表演示
    cout << "\n【DP表填充过程】" << endl;
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j <= W; j++) {
            dp[i][j] = dp[i-1][j];
            if (j >= weights[i-1]) {
                dp[i][j] = max(dp[i][j], dp[i-1][j - weights[i-1]] + values[i-1]);
            }
        }
        cout << "考虑物品 " << (i-1) << " 后：" << endl;
        printDPTable(dp);
        cout << endl;
    }

    // 完全背包测试
    cout << "\n【完全背包问题】" << endl;
    vector<int> weights2 = {1, 3, 4};
    vector<int> values2 = {15, 20, 30};
    int W2 = 4;

    cout << "物品信息（可以无限选择）：" << endl;
    for (size_t i = 0; i < weights2.size(); i++) {
        cout << "物品" << i << "：重量=" << weights2[i] << "，价值=" << values2[i] << endl;
    }
    cout << "背包容量：" << W2 << endl;

    int maxValue2 = knapsack_complete(weights2, values2, W2);
    cout << "最大价值：" << maxValue2 << endl;

    // 多重背包测试
    cout << "\n【多重背包问题】" << endl;
    vector<int> weights3 = {2, 3, 4};
    vector<int> values3 = {4, 5, 6};
    vector<int> counts = {2, 1, 3};  // 每个物品的数量
    int W3 = 10;

    cout << "物品信息（有限数量）：" << endl;
    for (size_t i = 0; i < weights3.size(); i++) {
        cout << "物品" << i << "：重量=" << weights3[i] << "，价值=" << values3[i]
             << "，数量=" << counts[i] << endl;
    }
    cout << "背包容量：" << W3 << endl;

    int maxValue3 = knapsack_bounded(weights3, values3, counts, W3);
    cout << "最大价值：" << maxValue3 << endl;

    // 递推关系解释
    cout << "\n【递推关系】" << endl;
    cout << "dp[j] = 容量为 j 时的最大价值" << endl;
    cout << "\n0/1 背包：" << endl;
    cout << "  dp[j] = max(dp[j], dp[j - weight[i]] + value[i])" << endl;
    cout << "  从后往前更新（避免重复使用）" << endl;
    cout << "\n完全背包：" << endl;
    cout << "  dp[j] = max(dp[j], dp[j - weight[i]] + value[i])" << endl;
    cout << "  从前往后更新（允许重复使用）" << endl;

    // 空间优化
    cout << "\n【空间优化】" << endl;
    cout << "二维DP：" << endl;
    cout << "  dp[i][j] = 考虑前 i 个物品，容量为 j 的最大价值" << endl;
    cout << "  空间：O(n × W)" << endl;
    cout << "\n一维优化：" << endl;
    cout << "  dp[j] = 容量为 j 的最大价值" << endl;
    cout << "  空间：O(W)" << endl;
    cout << "\n关键：更新顺序不同" << endl;
    cout << "  0/1 背包：从后往前（j 从 W 到 weight[i]）" << endl;
    cout << "  完全背包：从前往后（j 从 weight[i] 到 W）" << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "类型\t\t时间复杂度\t空间复杂度" << endl;
    cout << "------------------------------------------------" << endl;
    cout << "0/1 背包（二维）\tO(n×W)\t\tO(n×W)" << endl;
    cout << "0/1 背包（优化）\tO(n×W)\t\tO(W)" << endl;
    cout << "完全背包\t\tO(n×W)\t\tO(W)" << endl;
    cout << "多重背包\t\tO(n×W×log(count))\tO(W)" << endl;

    // 实际应用
    cout << "\n【实际应用】" << endl;
    cout << "1. 资源分配问题" << endl;
    cout << "2. 投资组合选择" << endl;
    cout << "3. 货箱装载优化" << endl;
    cout << "4. 项目选择（预算限制）" << endl;
    cout << "5. 切割问题（最大价值）" << endl;

    // 变体问题
    cout << "\n【变体问题】" << endl;
    cout << "1. 恰好装满背包" << endl;
    cout << "   初始化：dp[0] = 0，其他为 -∞" << endl;
    cout << "2. 求方案数" << endl;
    cout << "   dp[j] += dp[j - weight[i]]" << endl;
    cout << "3. 求具体方案" << endl;
    cout << "   使用二维DP，回溯找出物品" << endl;
    cout << "4. 二维费用背包" << endl;
    cout << "   扩展到二维：dp[v][w]" << endl;

    return 0;
}
