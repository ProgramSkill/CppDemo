/*
 * 算法：编辑距离 (Edit Distance / Levenshtein Distance)
 * 类型：递推 - 二维DP
 * 时间复杂度：O(m×n)
 * 空间复杂度：O(m×n) 或 O(min(m,n))
 *
 * 题目：给定两个单词，计算将 word1 转换为 word2 所需的最少操作数
 *      允许的操作：插入一个字符、删除一个字符、替换一个字符
 *
 * 递推关系：
 *   如果 word1[i-1] == word2[j-1]:
 *     dp[i][j] = dp[i-1][j-1]  (无需操作)
 *   否则:
 *     dp[i][j] = min(
 *       dp[i-1][j] + 1,      (删除 word1[i-1])
 *       dp[i][j-1] + 1,      (插入 word2[j-1])
 *       dp[i-1][j-1] + 1     (替换 word1[i-1] 为 word2[j-1])
 *     )
 *
 * 思路：
 *   dp[i][j] 表示将 word1[0...i-1] 转换为 word2[0...j-1] 的最少操作数
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

// 方法1：基本递推
int minDistance(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    // 边界条件：从空字符串转换
    for (int i = 0; i <= m; i++) {
        dp[i][0] = i;  // 删除 i 个字符
    }
    for (int j = 0; j <= n; j++) {
        dp[0][j] = j;  // 插入 j 个字符
    }

    // 填充DP表
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                // 当前字符相同，无需操作
                dp[i][j] = dp[i-1][j-1];
            } else {
                // 取三种操作的最小值
                dp[i][j] = min({
                    dp[i-1][j] + 1,      // 删除
                    dp[i][j-1] + 1,      // 插入
                    dp[i-1][j-1] + 1     // 替换
                });
            }
        }
    }

    return dp[m][n];
}

// 方法2：空间优化（使用两行）
int minDistance_optimized(string word1, string word2) {
    int m = word1.length(), n = word2.length();

    // 确保 n 较小
    if (m < n) {
        swap(word1, word2);
        swap(m, n);
    }

    vector<int> prev(n + 1), curr(n + 1);

    // 初始化
    for (int j = 0; j <= n; j++) {
        prev[j] = j;
    }

    for (int i = 1; i <= m; i++) {
        curr[0] = i;
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                curr[j] = prev[j-1];
            } else {
                curr[j] = min({
                    prev[j] + 1,
                    curr[j-1] + 1,
                    prev[j-1] + 1
                });
            }
        }
        swap(prev, curr);
    }

    return prev[n];
}

// 方法3：返回操作序列
vector<string> minDistance_withOperations(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = min({dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1});
            }
        }
    }

    // 回溯操作序列
    vector<string> operations;
    int i = m, j = n;

    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && word1[i-1] == word2[j-1]) {
            // 字符相同，无需操作
            i--;
            j--;
        } else if (i > 0 && j > 0 && dp[i][j] == dp[i-1][j-1] + 1) {
            // 替换
            operations.push_back("替换 '" + string(1, word1[i-1]) + "' 为 '" + string(1, word2[j-1]) + "'");
            i--;
            j--;
        } else if (j > 0 && dp[i][j] == dp[i][j-1] + 1) {
            // 插入
            operations.push_back("插入 '" + string(1, word2[j-1]) + "'");
            j--;
        } else if (i > 0 && dp[i][j] == dp[i-1][j] + 1) {
            // 删除
            operations.push_back("删除 '" + string(1, word1[i-1]) + "'");
            i--;
        }
    }

    reverse(operations.begin(), operations.end());
    return operations;
}

// 方法4：只允许插入和删除
int minDistance_insertDelete(string word1, string word2) {
    int m = word1.length(), n = word2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                // 只允许插入和删除
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1);
            }
        }
    }

    return dp[m][n];
}

// 打印DP表
void printDPTable(const vector<vector<int>>& dp, const string& word1, const string& word2) {
    cout << "    Ø ";
    for (char c : word2) cout << c << " ";
    cout << endl;

    for (int i = 0; i <= (int)word1.length(); i++) {
        if (i > 0) cout << word1[i-1] << " ";
        else cout << "Ø ";
        for (int j = 0; j <= (int)word2.length(); j++) {
            cout << dp[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    cout << "========== 编辑距离 (Edit Distance) ==========" << endl;

    // 基本测试
    cout << "\n【基本测试】" << endl;
    vector<pair<string, string>> test_cases = {
        {"horse", "ros"},      // 输出：3
        {"intention", "execution"},  // 输出：5
        {"abc", "abc"},        // 输出：0
        {"", "abc"},           // 输出：3
        {"abc", ""},           // 输出：3
        {"kitten", "sitting"}  // 输出：3
    };

    cout << "单词1\t\t单词2\t\t编辑距离" << endl;
    cout << "-----------------------------------------------" << endl;

    for (auto& test : test_cases) {
        cout << test.first << "\t\t" << test.second << "\t\t"
             << minDistance(test.first, test.second) << endl;
    }

    // 详细示例
    cout << "\n【详细示例：horse → ros】" << endl;
    string word1 = "horse", word2 = "ros";
    int m = word1.length(), n = word2.length();

    cout << "将 \"" << word1 << "\" 转换为 \"" << word2 << "\"" << endl;
    cout << "\n允许的操作：" << endl;
    cout << "  1. 插入一个字符" << endl;
    cout << "  2. 删除一个字符" << endl;
    cout << "  3. 替换一个字符" << endl;

    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    // 初始化边界
    cout << "\n初始化边界条件：" << endl;
    for (int i = 0; i <= m; i++) {
        dp[i][0] = i;
        if (i > 0) cout << "dp[" << i << "][0] = " << i << " (删除 " << i << " 个字符)" << endl;
    }
    for (int j = 0; j <= n; j++) {
        dp[0][j] = j;
        if (j > 0) cout << "dp[0][" << j << "] = " << j << " (插入 " << j << " 个字符)" << endl;
    }

    // 填充DP表
    cout << "\n填充DP表：" << endl;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
                cout << "dp[" << i << "][" << j << "]: '" << word1[i-1] << "' == '"
                     << word2[j-1] << "', dp[" << i << "][" << j << "] = dp["
                     << i-1 << "][" << j-1 << "] = " << dp[i][j] << endl;
            } else {
                int del = dp[i-1][j] + 1;
                int ins = dp[i][j-1] + 1;
                int rep = dp[i-1][j-1] + 1;
                dp[i][j] = min({del, ins, rep});

                cout << "dp[" << i << "][" << j << "]: '" << word1[i-1] << "' != '"
                     << word2[j-1] << "'" << endl;
                cout << "  删除: dp[" << i-1 << "][" << j << "] + 1 = " << del << endl;
                cout << "  插入: dp[" << i << "][" << j-1 << "] + 1 = " << ins << endl;
                cout << "  替换: dp[" << i-1 << "][" << j-1 << "] + 1 = " << rep << endl;
                cout << "  → dp[" << i << "][" << j << "] = " << dp[i][j] << endl;
            }
        }
    }

    cout << "\nDP表：" << endl;
    printDPTable(dp, word1, word2);

    cout << "\n编辑距离：" << dp[m][n] << endl;

    // 获取操作序列
    cout << "\n【操作序列】" << endl;
    vector<string> operations = minDistance_withOperations(word1, word2);
    cout << "将 \"" << word1 << "\" 转换为 \"" << word2 << "\" 的步骤：" << endl;
    for (size_t i = 0; i < operations.size(); i++) {
        cout << "  " << (i+1) << ". " << operations[i] << endl;
    }

    // 方法对比
    cout << "\n【方法对比】" << endl;
    string w1 = "intention", w2 = "execution";

    clock_t start, end;

    start = clock();
    int dist1 = minDistance(w1, w2);
    end = clock();
    cout << "基本方法：编辑距离 = " << dist1 << "，用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    start = clock();
    int dist2 = minDistance_optimized(w1, w2);
    end = clock();
    cout << "空间优化：编辑距离 = " << dist2 << "，用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    // 只允许插入删除
    cout << "\n【只允许插入和删除】" << endl;
    string word3 = "abc", word4 = "def";
    int dist_full = minDistance(word3, word4);
    int dist_no_replace = minDistance_insertDelete(word3, word4);

    cout << "完整版（允许替换）：编辑距离 = " << dist_full << endl;
    cout << "限制版（只允许插入删除）：编辑距离 = " << dist_no_replace << endl;

    // 递推关系解释
    cout << "\n【递推关系】" << endl;
    cout << "dp[i][j] = 将 word1[0...i-1] 转换为 word2[0...j-1] 的最少操作数" << endl;
    cout << "\n状态转移：" << endl;
    cout << "  如果 word1[i-1] == word2[j-1]:" << endl;
    cout << "    dp[i][j] = dp[i-1][j-1]  (字符相同，无需操作)" << endl;
    cout << "  否则:" << endl;
    cout << "    dp[i][j] = min(" << endl;
    cout << "      dp[i-1][j] + 1,      (删除 word1[i-1])" << endl;
    cout << "      dp[i][j-1] + 1,      (插入 word2[j-1])" << endl;
    cout << "      dp[i-1][j-1] + 1     (替换 word1[i-1])" << endl;
    cout << "    )" << endl;

    // 图解
    cout << "\n【操作示意】" << endl;
    cout << "    word1:  h  o  r  s  e" << endl;
    cout << "    word2:  r  o  s" << endl;
    cout << endl;
    cout << "    步骤1: 删除 'h'  →  orse" << endl;
    cout << "    步骤2: 删除 's'  →  ore" << endl;
    cout << "    步骤3: 替换 'e' 为 's'  →  ros" << endl;
    cout << endl;
    cout << "    总操作数：3" << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "时间复杂度：O(m × n)" << endl;
    cout << "  - m = word1.length()" << endl;
    cout << "  - n = word2.length()" << endl;
    cout << "空间复杂度：" << endl;
    cout << "  - 基本方法：O(m × n)" << endl;
    cout << "  - 空间优化：O(min(m, n))" << endl;

    // 实际应用
    cout << "\n【实际应用】" << endl;
    cout << "1. 拼写检查：寻找最接近的正确拼写" << endl;
    cout << "2. DNA序列分析：计算序列相似度" << endl;
    cout << "3. 版本控制系统：比较文件差异" << endl;
    cout << "4. 机器翻译：评估翻译质量" << endl;
    cout << "5. 语音识别：校正识别错误" << endl;

    // 变体问题
    cout << "\n【变体问题】" << endl;
    cout << "1. 最长公共子序列（LCS）" << endl;
    cout << "   不允许替换，只允许插入删除" << endl;
    cout << "2. Damerau-Levenshtein 距离" << endl;
    cout << "   额外允许交换相邻两个字符" << endl;
    cout << "3. 加权编辑距离" << endl;
    cout << "   不同操作有不同的代价" << endl;

    return 0;
}
