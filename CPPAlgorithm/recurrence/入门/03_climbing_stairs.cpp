/*
 * 算法：爬楼梯问题 (Climbing Stairs)
 * 类型：递推 - 二阶线性递推
 * 时间复杂度：O(n)
 * 空间复杂度：O(1)
 *
 * 题目：每次可以爬 1 或 2 个台阶，爬到 n 阶有多少种方法？
 *
 * 递推关系：
 *   f(n) = f(n-1) + f(n-2)
 *   f(1) = 1, f(2) = 2
 *
 * 思路推导：
 *   要到达第 n 阶，可以从：
 *   - 第 n-1 阶爬 1 阶上来（有 f(n-1) 种方法）
 *   - 第 n-2 阶爬 2 阶上来（有 f(n-2) 种方法）
 *   因此：f(n) = f(n-1) + f(n-2)
 */

#include <iostream>
#include <vector>
using namespace std;

// 基本递推解法
long long climbStairs(int n) {
    if (n <= 2) return n;

    long long prev2 = 1;  // f(1) = 1
    long long prev1 = 2;  // f(2) = 2
    long long curr;

    for (int i = 3; i <= n; i++) {
        curr = prev1 + prev2;  // f(i) = f(i-1) + f(i-2)
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

// 递归解法（对比，低效）
long long climbStairs_recursive(int n) {
    if (n <= 2) return n;
    return climbStairs_recursive(n-1) + climbStairs_recursive(n-2);
}

// 进阶：每次可以爬 1、2 或 3 个台阶
long long climbStairs3(int n) {
    if (n == 1) return 1;
    if (n == 2) return 2;
    if (n == 3) return 4;  // 1+1+1, 1+2, 2+1, 3

    long long prev3 = 1, prev2 = 2, prev1 = 4;
    long long curr;

    for (int i = 4; i <= n; i++) {
        curr = prev1 + prev2 + prev3;  // 三阶递推
        prev3 = prev2;
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

// 通用解法：每次可以爬 1 到 k 个台阶
long long climbStairs_general(int n, int k) {
    if (n <= 1) return 1;

    vector<long long> dp(n + 1, 0);
    dp[0] = 1;
    dp[1] = 1;

    for (int i = 2; i <= n; i++) {
        for (int j = 1; j <= k && j <= i; j++) {
            dp[i] += dp[i - j];
        }
    }

    return dp[n];
}

// 带路径输出的递推
void climbStairs_withPaths(int n, vector<vector<int>>& paths) {
    if (n == 1) {
        paths = {{1}};
        return;
    }
    if (n == 2) {
        paths = {{1, 1}, {2}};
        return;
    }

    vector<vector<vector<int>>> dp(n + 1);
    dp[1] = {{1}};
    dp[2] = {{1, 1}, {2}};

    for (int i = 3; i <= n; i++) {
        // 从 i-1 爬 1 阶
        for (auto& path : dp[i-1]) {
            vector<int> newPath = path;
            newPath.push_back(1);
            dp[i].push_back(newPath);
        }
        // 从 i-2 爬 2 阶
        for (auto& path : dp[i-2]) {
            vector<int> newPath = path;
            newPath.push_back(2);
            dp[i].push_back(newPath);
        }
    }

    paths = dp[n];
}

int main() {
    cout << "========== 爬楼梯问题 ==========" << endl;

    // 基本测试
    cout << "\n【基本测试：爬到 n 阶的方法数】" << endl;
    cout << "n\t方法数" << endl;
    cout << "----------------" << endl;
    for (int n = 1; n <= 10; n++) {
        cout << n << "\t" << climbStairs(n) << endl;
    }

    // 方法对比
    cout << "\n【递推 vs 递归对比】" << endl;
    cout << "n\t递推结果\t递归结果\t是否一致" << endl;
    cout << "------------------------------------------------" << endl;
    for (int n : {5, 10, 15, 20}) {
        long long iter_result = climbStairs(n);
        long long rec_result = climbStairs_recursive(n);
        cout << n << "\t" << iter_result << "\t\t" << rec_result << "\t\t";
        cout << (iter_result == rec_result ? "✓" : "✗") << endl;
    }

    // 递推过程演示
    cout << "\n【递推过程演示：爬到 5 阶】" << endl;
    int n = 5;
    cout << "初始条件：" << endl;
    cout << "  f(1) = 1 (方法: [1])" << endl;
    cout << "  f(2) = 2 (方法: [1,1], [2])" << endl;
    cout << "\n递推过程：" << endl;

    vector<long long> dp(n + 1);
    dp[1] = 1;
    dp[2] = 2;

    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];
        cout << "  f(" << i << ") = f(" << i-1 << ") + f(" << i-2 << ") = "
             << dp[i-1] << " + " << dp[i-2] << " = " << dp[i] << endl;
    }

    // 路径输出演示（n=4）
    cout << "\n【路径输出演示：爬到 4 阶的所有方法】" << endl;
    vector<vector<int>> paths;
    climbStairs_withPaths(4, paths);

    cout << "共有 " << paths.size() << " 种方法：" << endl;
    for (size_t i = 0; i < paths.size(); i++) {
        cout << "  方法" << (i+1) << ": ";
        for (size_t j = 0; j < paths[i].size(); j++) {
            cout << paths[i][j];
            if (j < paths[i].size() - 1) cout << " → ";
        }
        cout << endl;
    }

    // 进阶：每次爬 1-3 阶
    cout << "\n【进阶：每次可以爬 1、2 或 3 个台阶】" << endl;
    cout << "n\t方法数(1-3阶)" << endl;
    cout << "------------------------" << endl;
    for (int i = 1; i <= 10; i++) {
        cout << i << "\t" << climbStairs3(i) << endl;
    }

    // 通用解法演示
    cout << "\n【通用解法：每次可以爬 1 到 k 个台阶】" << endl;
    for (int k : {2, 3, 4}) {
        cout << "k = " << k << "（每次最多爬" << k << "阶）：" << endl;
        cout << "  n=5 时方法数 = " << climbStairs_general(5, k) << endl;
    }

    // 与斐波那契的关系
    cout << "\n【与斐波那契数列的关系】" << endl;
    cout << "爬楼梯问题的递推公式与斐波那契数列相同！" << endl;
    cout << "只是初始条件不同：" << endl;
    cout << "  斐波那契：F(0)=0, F(1)=1, F(2)=1, F(3)=2, ..." << endl;
    cout << "  爬楼梯：f(1)=1, f(2)=2, f(3)=3, f(4)=5, ..." << endl;
    cout << "\n对比：" << endl;
    cout << "n\t爬楼梯\t斐波那契" << endl;
    cout << "-----------------------" << endl;
    for (int i = 1; i <= 10; i++) {
        cout << i << "\t" << climbStairs(i) << "\t" << climbStairs(i) - climbStairs(i-1) << endl;
    }

    // 变种问题：带障碍
    cout << "\n【变种问题思路：带障碍的爬楼梯】" << endl;
    cout << "如果有某些台阶坏了不能踩，如何处理？" << endl;
    cout << "递推关系修改为：" << endl;
    cout << "  f(n) = f(n-1) + f(n-2) （如果 n、n-1、n-2 都不是障碍）" << endl;
    cout << "  f(n) = f(n-1) （如果 n-2 是障碍）" << endl;
    cout << "  f(n) = f(n-2) （如果 n-1 是障碍）" << endl;
    cout << "  f(n) = 0 （如果 n 是障碍）" << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "时间复杂度：O(n)" << endl;
    cout << "空间复杂度：O(1) - 只需保存前两个状态" << endl;
    cout << "递归时间复杂度：O(2^n) - 大量重复计算" << endl;

    return 0;
}
