/*
 * 算法：最长公共子序列 (Longest Common Subsequence - LCS)
 * 类型：递推 - 二维DP
 * 时间复杂度：O(m×n)
 * 空间复杂度：O(m×n) 或 O(min(m,n))
 *
 * 题目：给定两个字符串，求它们的最长公共子序列长度
 *      子序列不需要连续
 *
 * 递推关系：
 *   如果 text1[i-1] == text2[j-1]:
 *     dp[i][j] = dp[i-1][j-1] + 1
 *   否则:
 *     dp[i][j] = max(dp[i-1][j], dp[i][j-1])
 *
 * 思路：
 *   dp[i][j] 表示 text1[0...i-1] 和 text2[0...j-1] 的 LCS 长度
 *   如果当前字符匹配，LCS 长度加 1
 *   否则，取去掉 text1 当前字符或去掉 text2 当前字符的最大值
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

// 方法1：基本递推
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                // 当前字符匹配
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                // 当前字符不匹配，取最大值
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[m][n];
}

// 方法2：空间优化（使用两行）
int longestCommonSubsequence_optimized(string text1, string text2) {
    int m = text1.length(), n = text2.length();

    // 确保 n 较小，节省空间
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

// 方法3：返回LCS字符串
string longestCommonSubsequence_string(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    // 填充DP表
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    // 回溯构建LCS字符串
    string lcs = "";
    int i = m, j = n;

    while (i > 0 && j > 0) {
        if (text1[i-1] == text2[j-1]) {
            // 当前字符在LCS中
            lcs = text1[i-1] + lcs;
            i--;
            j--;
        } else if (dp[i-1][j] > dp[i][j-1]) {
            // 向上移动
            i--;
        } else {
            // 向左移动
            j--;
        }
    }

    return lcs;
}

// 方法4：打印所有LCS（可能有多个）
vector<string> findAllLCS(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    // 填充DP表
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    // 回溯收集所有LCS
    vector<string> result;
    function<void(int, int, string)> backtrack = [&](int i, int j, string current) {
        if (i == 0 || j == 0) {
            if (!current.empty()) {
                result.push_back(current);
            }
            return;
        }

        if (text1[i-1] == text2[j-1]) {
            backtrack(i-1, j-1, text1[i-1] + current);
        } else {
            if (dp[i-1][j] == dp[i][j]) {
                backtrack(i-1, j, current);
            }
            if (dp[i][j-1] == dp[i][j]) {
                backtrack(i, j-1, current);
            }
        }
    };

    backtrack(m, n, "");
    return result;
}

// 打印DP表
void printDPTable(const vector<vector<int>>& dp, const string& text1, const string& text2) {
    int m = text1.length(), n = text2.length();

    cout << "    ";
    for (char c : text2) cout << c << " ";
    cout << endl;

    for (int i = 0; i <= m; i++) {
        if (i > 0) cout << text1[i-1] << " ";
        else cout << "  ";
        for (int j = 0; j <= n; j++) {
            cout << dp[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    cout << "========== 最长公共子序列 (LCS) ==========" << endl;

    // 基本测试
    cout << "\n【基本测试】" << endl;
    vector<pair<string, string>> test_cases = {
        {"abcde", "ace"},         // LCS: "ace" 长度3
        {"abc", "abc"},           // LCS: "abc" 长度3
        {"abc", "def"},           // LCS: "" 长度0
        {"AGGTAB", "GXTXAYB"},    // LCS: "GTAB" 长度4
        {"abcdgh", "aedfhr"},     // LCS: "adh" 长度3
    };

    cout << "字符串1\t\t字符串2\t\tLCS长度" << endl;
    cout << "------------------------------------------------" << endl;

    for (auto& test : test_cases) {
        cout << test.first << "\t\t" << test.second << "\t\t"
             << longestCommonSubsequence(test.first, test.second) << endl;
    }

    // 详细示例
    cout << "\n【详细示例：abcde 和 ace】" << endl;
    string text1 = "abcde", text2 = "ace";
    int m = text1.length(), n = text2.length();

    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    cout << "DP表填充过程：" << endl;
    cout << "text1 = \"" << text1 << "\", text2 = \"" << text2 << "\"" << endl;
    cout << "\n初始DP表（全是0）：" << endl;
    printDPTable(dp, text1, text2);

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
                cout << "\ndp[" << i << "][" << j << "]: text1[" << i-1 << "]='"
                     << text1[i-1] << "' == text2[" << j-1 << "]='"
                     << text2[j-1] << "'，dp[" << i << "][" << j << "] = dp["
                     << i-1 << "][" << j-1 << "] + 1 = " << dp[i][j] << endl;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
                cout << "\ndp[" << i << "][" << j << "]: text1[" << i-1 << "]='"
                     << text1[i-1] << "' != text2[" << j-1 << "]='"
                     << text2[j-1] << "'，dp[" << i << "][" << j << "] = max(dp["
                     << i-1 << "][" << j << "], dp[" << i << "][" << j-1
                     << "]) = max(" << dp[i-1][j] << ", " << dp[i][j-1]
                     << ") = " << dp[i][j] << endl;
            }
        }
    }

    cout << "\n最终DP表：" << endl;
    printDPTable(dp, text1, text2);

    cout << "\nLCS长度：" << dp[m][n] << endl;

    // 获取LCS字符串
    cout << "\n【获取LCS字符串】" << endl;
    string lcs = longestCommonSubsequence_string(text1, text2);
    cout << "LCS: \"" << lcs << "\"" << endl;
    cout << "长度：" << lcs.length() << endl;

    // 多个LCS示例
    cout << "\n【多个LCS示例】" << endl;
    string s1 = "abcab", s2 = "acb";
    vector<string> all_lcs = findAllLCS(s1, s2);

    cout << "字符串1: \"" << s1 << "\"" << endl;
    cout << "字符串2: \"" << s2 << "\"" << endl;
    cout << "所有LCS：" << endl;
    for (size_t i = 0; i < all_lcs.size(); i++) {
        cout << "  " << (i+1) << ". \"" << all_lcs[i] << "\"" << endl;
    }

    // 方法对比
    cout << "\n【方法对比】" << endl;
    string long1 = "abcdefghijklmnopqrstuvwxyz";
    string long2 = "acegikmnpqsuvy";

    clock_t start, end;

    start = clock();
    int len1 = longestCommonSubsequence(long1, long2);
    end = clock();
    cout << "基本方法：长度 = " << len1 << "，用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    start = clock();
    int len2 = longestCommonSubsequence_optimized(long1, long2);
    end = clock();
    cout << "空间优化：长度 = " << len2 << "，用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    // 递推关系解释
    cout << "\n【递推关系】" << endl;
    cout << "dp[i][j] = text1[0...i-1] 和 text2[0...j-1] 的LCS长度" << endl;
    cout << "\n状态转移：" << endl;
    cout << "  如果 text1[i-1] == text2[j-1]:" << endl;
    cout << "    dp[i][j] = dp[i-1][j-1] + 1  (当前字符匹配，LCS长度+1)" << endl;
    cout << "  否则:" << endl;
    cout << "    dp[i][j] = max(dp[i-1][j], dp[i][j-1])  (取最大值)" << endl;
    cout << "\n边界条件：" << endl;
    cout << "  dp[0][j] = 0  (text1为空)" << endl;
    cout << "  dp[i][0] = 0  (text2为空)" << endl;

    // 图解
    cout << "\n【状态转移图解】" << endl;
    cout << "          text1[i-1] == text2[j-1]?" << endl;
    cout << "                    │" << endl;
    cout << "         ┌──────────┴──────────┐" << endl;
    cout << "         ▼                     ▼" << endl;
    cout << "        Yes                    No" << endl;
    cout << "  dp[i][j] = dp[i-1][j-1] + 1   dp[i][j] = max(dp[i-1][j], dp[i][j-1])" << endl;
    cout << "  (对角线 + 1)              (上或左，取最大)" << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "时间复杂度：O(m × n) - 需要填充整个DP表" << endl;
    cout << "空间复杂度：" << endl;
    cout << "  - 基本方法：O(m × n)" << endl;
    cout << "  - 空间优化：O(min(m, n))" << endl;

    // 实际应用
    cout << "\n【实际应用】" << endl;
    cout << "1. 版本控制系统：Git diff 的核心算法" << endl;
    cout << "2. 生物信息学：DNA序列比对" << endl;
    cout << "3. 拼写检查：找到最接近的正确拼写" << endl;
    cout << "4. 数据压缩：找到重复模式" << endl;

    // 变体问题
    cout << "\n【变体问题】" << endl;
    cout << "1. 最长公共子串（要求连续）" << endl;
    cout << "   修改：不匹配时dp[i][j] = 0" << endl;
    cout << "2. 最短公共超序列" << endl;
    cout << "   包含两个字符串的最短序列" << endl;
    cout << "3. 编辑距离" << endl;
    cout << "   允许插入、删除、替换操作" << endl;

    return 0;
}
