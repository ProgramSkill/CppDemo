/*
 * 算法：零钱兑换 (Coin Change)
 * 类型：递推 - 一维DP（完全背包问题）
 * 时间复杂度：O(n × amount)
 * 空间复杂度：O(amount)
 *
 * 题目：给定不同面额的硬币和一个总金额，计算可以凑成总金额的最少硬币数
 *      如果无法凑出，返回 -1
 *      每种硬币可以使用无限次
 *
 * 递推关系：
 *   dp[i] = min(dp[i], dp[i - coin] + 1) for all coin in coins
 *   dp[0] = 0
 *   dp[i] = amount + 1 (初始值，表示不可达)
 *
 * 思路：
 *   dp[i] 表示凑成金额 i 所需的最少硬币数
 *   对于每个金额 i，尝试用每种硬币
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;

// 方法1：基本递推
int coinChange(vector<int>& coins, int amount) {
    // dp[i] = 凑成金额 i 所需的最少硬币数
    vector<int> dp(amount + 1, amount + 1);  // 初始化为最大值
    dp[0] = 0;  // 凑成金额 0 需要 0 个硬币

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = min(dp[i], dp[i - coin] + 1);
            }
        }
    }

    return dp[amount] > amount ? -1 : dp[amount];
}

// 方法2：返回硬币组合
vector<int> coinChange_withCoins(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, amount + 1);
    vector<int> prev_coin(amount + 1, -1);  // 记录使用的硬币
    dp[0] = 0;

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i && dp[i - coin] + 1 < dp[i]) {
                dp[i] = dp[i - coin] + 1;
                prev_coin[i] = coin;
            }
        }
    }

    if (dp[amount] > amount) return {};  // 无法凑出

    // 回溯构建硬币组合
    vector<int> result;
    int curr = amount;
    while (curr > 0) {
        result.push_back(prev_coin[curr]);
        curr -= prev_coin[curr];
    }

    return result;
}

// 方法3：计算组合总数（不同的组合顺序视为同一种）
int coinChange_numberOfCombinations(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;

    for (int coin : coins) {
        for (int i = coin; i <= amount; i++) {
            dp[i] += dp[i - coin];
        }
    }

    return dp[amount];
}

// 方法4：计算排列总数（不同的顺序视为不同排列）
int coinChange_numberOfPermutations(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] += dp[i - coin];
            }
        }
    }

    return dp[amount];
}

// 方法5：完全背包版（每个硬币只能用一次）
int coinChange_limited(vector<int>& coins, int amount) {
    int n = coins.size();
    vector<vector<int>> dp(n + 1, vector<int>(amount + 1, amount + 1));

    for (int i = 0; i <= n; i++) {
        dp[i][0] = 0;
    }

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= amount; j++) {
            if (coins[i-1] <= j) {
                dp[i][j] = min(dp[i-1][j], dp[i][j - coins[i-1]] + 1);
            } else {
                dp[i][j] = dp[i-1][j];
            }
        }
    }

    return dp[n][amount] > amount ? -1 : dp[n][amount];
}

// 打印DP表
void printDPTable(const vector<int>& dp) {
    cout << "金额: ";
    for (size_t i = 0; i < dp.size(); i++) {
        cout << i << " ";
    }
    cout << endl;

    cout << "dp:   ";
    for (int val : dp) {
        if (val > 1000) cout << "∞ ";
        else cout << val << " ";
    }
    cout << endl;
}

int main() {
    cout << "========== 零钱兑换问题 ==========" << endl;

    // 基本测试
    cout << "\n【基本测试】" << endl;
    vector<pair<vector<int>, int>> test_cases = {
        {{1, 2, 5}, 11},      // 输出：3 (5+5+1)
        {{2}, 3},             // 输出：-1 (无法凑出)
        {{1}, 0},             // 输出：0
        {{1, 2, 5}, 100},     // 输出：20 (20×5)
        {{2, 5, 10, 1}, 27},  // 输出：4
    };

    cout << "硬币\t\t\t金额\t最少硬币数" << endl;
    cout << "-----------------------------------------------" << endl;

    for (auto& test : test_cases) {
        cout << "[";
        for (size_t i = 0; i < test.first.size(); i++) {
            cout << test.first[i];
            if (i < test.first.size() - 1) cout << ", ";
        }
        cout << "]\t\t" << test.second << "\t"
             << coinChange(test.first, test.second) << endl;
    }

    // 详细示例
    cout << "\n【详细示例：coins = [1, 2, 5], amount = 11】" << endl;
    vector<int> coins = {1, 2, 5};
    int amount = 11;

    cout << "硬币面额：[1, 2, 5]" << endl;
    cout << "目标金额：" << amount << endl;
    cout << "\n递推过程：" << endl;

    vector<int> dp(amount + 1, amount + 1);
    dp[0] = 0;

    cout << "初始：dp[0] = 0" << endl;
    printDPTable(dp);

    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                if (dp[i - coin] + 1 < dp[i]) {
                    dp[i] = dp[i - coin] + 1;
                }
            }
        }
        cout << "\n计算 dp[" << i << "]：" << endl;
        cout << "  尝试硬币1：dp[" << i << "] = dp[" << i-1 << "] + 1 = " << dp[i] << endl;
        printDPTable(dp);
    }

    cout << "\n最少硬币数：" << dp[amount] << endl;

    // 获取硬币组合
    cout << "\n【获取硬币组合】" << endl;
    vector<int> combination = coinChange_withCoins(coins, amount);
    cout << "硬币组合：";
    for (int coin : combination) {
        cout << coin << " ";
    }
    cout << endl;
    cout << "总和：" << amount << endl;

    // 组合数 vs 排列数
    cout << "\n【组合数 vs 排列数】" << endl;
    vector<int> coins2 = {1, 2, 5};
    int amount2 = 5;

    int combinations = coinChange_numberOfCombinations(coins2, amount2);
    int permutations = coinChange_numberOfPermutations(coins2, amount2);

    cout << "金额 " << amount2 << "，硬币 [1, 2, 5]：" << endl;
    cout << "组合数：" << combinations << " (不同顺序视为同一种)" << endl;
    cout << "排列数：" << permutations << " (不同顺序视为不同)" << endl;
    cout << "\n组合：" << endl;
    cout << "  [1, 1, 1, 1, 1]" << endl;
    cout << "  [1, 1, 1, 2]" << endl;
    cout << "  [1, 2, 2]" << endl;
    cout << "  [1, 1, 3] × (不存在3)" << endl;
    cout << "  [5]" << endl;

    // 有限硬币数量
    cout << "\n【限制硬币使用次数】" << endl;
    vector<int> coins3 = {1, 2, 5};
    int amount3 = 11;

    int unlimited = coinChange(coins3, amount3);
    int limited = coinChange_limited(coins3, amount3);

    cout << "无限使用：最少硬币数 = " << unlimited << endl;
    cout << "每种一次：最少硬币数 = " << limited << endl;

    // 递推关系解释
    cout << "\n【递推关系】" << endl;
    cout << "dp[i] = 凑成金额 i 所需的最少硬币数" << endl;
    cout << "\n状态转移：" << endl;
    cout << "  dp[i] = min(dp[i], dp[i - coin] + 1) for all coin in coins" << endl;
    cout << "\n含义：" << endl;
    cout << "  对于金额 i，尝试用每种硬币 coin" << endl;
    cout << "  如果使用 coin，则需要先凑出 i-coin" << endl;
    cout << "  总硬币数 = dp[i-coin] + 1" << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "时间复杂度：O(n × amount)" << endl;
    cout << "  - n 是硬币种类数" << endl;
    cout << "  - amount 是目标金额" << endl;
    cout << "空间复杂度：O(amount)" << endl;

    // 图解
    cout << "\n【状态转移图解】" << endl;
    cout << "对于金额 i，尝试所有硬币：" << endl;
    cout << endl;
    cout << "    ┌─────────┐" << endl;
    cout << "    │  金额 i  │" << endl;
    cout << "    └────┬────┘" << endl;
    cout << "         │" << endl;
    cout << "    ┌────┴────┬─────────┬─────────┐" << endl;
    cout << "    ▼        ▼         ▼         ▼" << endl;
    cout << "  用硬币1   用硬币2   用硬币5   ..." << endl;
    cout << "  dp[i-1]+1  dp[i-2]+1  dp[i-5]+1" << endl;
    cout << "    └────────┴─────────┴─────────┘" << endl;
    cout << "                    │" << endl;
    cout << "                    ▼" << endl;
    cout << "              取最小值 = dp[i]" << endl;

    // 实际应用
    cout << "\n【实际应用】" << endl;
    cout << "1. 购物支付：最少硬币/纸币数量" << endl;
    cout << "2. 资源分配：最优资源组合" << endl;
    cout << "3. 投资组合：最小成本达成目标" << endl;
    cout << "4. 数据压缩：最优编码方案" << endl;

    // 变体问题
    cout << "\n【变体问题】" << endl;
    cout << "1. 完全背包问题" << endl;
    cout << "   每个物品有重量和价值，求最大价值" << endl;
    cout << "2. 最少邮票数" << endl;
    cout << "   给定邮票面额，贴足邮资的最少邮票" << endl;
    cout << "3. 找零方案数" << endl;
    cout << "   计算有多少种方式可以凑出金额" << endl;

    return 0;
}
