/*
 * 算法：最长递增子序列 (Longest Increasing Subsequence - LIS)
 * 类型：递推 - 一维DP
 * 时间复杂度：O(n²) 或 O(n log n)
 * 空间复杂度：O(n)
 *
 * 题目：给定一个无序整数数组，找到最长严格递增子序列的长度
 *      子序列不要求连续
 *
 * 递推关系（O(n²)解法）：
 *   dp[i] = max(dp[j] + 1) for all j < i and nums[j] < nums[i]
 *   dp[i] = 1 (如果没有这样的j)
 *
 * 思路：
 *   dp[i] 表示以 nums[i] 结尾的最长递增子序列长度
 *   对于每个 i，检查前面所有 j < i 且 nums[j] < nums[i] 的位置
 *   取最大的 dp[j] + 1
 */

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 方法1：基本递推（O(n²)）
int lengthOfLIS_basic(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;

    vector<int> dp(n, 1);  // 每个位置至少长度为1
    int maxLen = 1;

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
        maxLen = max(maxLen, dp[i]);
    }

    return maxLen;
}

// 方法2：返回LIS的具体序列
vector<int> LIS_sequence(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return {};

    vector<int> dp(n, 1);
    vector<int> parent(n, -1);  // 记录前驱
    int maxLen = 1, endIndex = 0;

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
        if (dp[i] > maxLen) {
            maxLen = dp[i];
            endIndex = i;
        }
    }

    // 回溯构建序列
    vector<int> lis;
    while (endIndex != -1) {
        lis.push_back(nums[endIndex]);
        endIndex = parent[endIndex];
    }
    reverse(lis.begin(), lis.end());

    return lis;
}

// 方法3：二分查找优化（O(n log n)）
int lengthOfLIS_binarySearch(vector<int>& nums) {
    vector<int> tails;  // tails[i] = 长度为i+1的LIS的最小末尾元素

    for (int num : nums) {
        // 二分查找第一个 >= num 的位置
        auto it = lower_bound(tails.begin(), tails.end(), num);

        if (it == tails.end()) {
            tails.push_back(num);  // 扩展
        } else {
            *it = num;  // 更新
        }
    }

    return tails.size();
}

// 方法4：二分查找 + 重建序列
vector<int> LIS_binarySearch_reconstruct(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return {};

    vector<int> tails;
    vector<int> tail_indices;  // tails中每个元素在原数组中的索引
    vector<int> prev_indices(n, -1);  // 前驱索引

    for (int i = 0; i < n; i++) {
        auto it = lower_bound(tails.begin(), tails.end(), nums[i]);
        int idx = it - tails.begin();

        if (it == tails.end()) {
            tails.push_back(nums[i]);
            tail_indices.push_back(i);
        } else {
            tails[idx] = nums[i];
            tail_indices[idx] = i;
        }

        if (idx > 0) {
            prev_indices[i] = tail_indices[idx - 1];
        }
    }

    // 重建序列
    vector<int> lis;
    int curr = tail_indices.back();
    while (curr != -1) {
        lis.push_back(nums[curr]);
        curr = prev_indices[curr];
    }
    reverse(lis.begin(), lis.end());

    return lis;
}

// 打印递推过程
void printLISProcess(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);

    cout << "数组：";
    for (int num : nums) cout << num << " ";
    cout << endl;

    cout << "\n递推过程：" << endl;
    cout << "i\tnums[i]\tdp[i]\t说明" << endl;
    cout << "------------------------------------------------" << endl;

    for (int i = 0; i < n; i++) {
        if (i == 0) {
            cout << i << "\t" << nums[i] << "\t" << dp[i] << "\t初始值" << endl;
        } else {
            cout << i << "\t" << nums[i] << "\t";
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = max(dp[i], dp[j] + 1);
                }
            }
            cout << dp[i] << "\t";
            cout << "max(dp[j]+1) for j<" << i << " and nums[j]<" << nums[i] << endl;
        }
    }
}

int main() {
    cout << "========== 最长递增子序列 (LIS) ==========" << endl;

    // 基本测试
    cout << "\n【基本测试】" << endl;
    vector<vector<int>> test_cases = {
        {10, 9, 2, 5, 3, 7, 101, 18},  // LIS: [2,3,7,101] 长度4
        {0, 1, 0, 3, 2, 3},            // LIS: [0,1,2,3] 长度4
        {7, 7, 7, 7, 7, 7, 7},         // LIS: [7] 长度1
        {1, 3, 6, 7, 9, 4, 10, 5, 6}   // LIS: [1,3,6,7,9,10] 长度6
    };

    cout << "数组\t\t\t\tLIS长度" << endl;
    cout << "------------------------------------------------" << endl;

    for (auto& nums : test_cases) {
        cout << "[";
        for (size_t i = 0; i < nums.size(); i++) {
            cout << nums[i];
            if (i < nums.size() - 1) cout << ", ";
        }
        cout << "]\t\t" << lengthOfLIS_basic(nums) << endl;
    }

    // 详细示例
    cout << "\n【详细示例：[10, 9, 2, 5, 3, 7, 101, 18]】" << endl;
    vector<int> nums = {10, 9, 2, 5, 3, 7, 101, 18};
    printLISProcess(nums);

    // 获取LIS序列
    cout << "\n【获取LIS序列】" << endl;
    vector<int> lis = LIS_sequence(nums);
    cout << "一个LIS序列：[";
    for (size_t i = 0; i < lis.size(); i++) {
        cout << lis[i];
        if (i < lis.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    cout << "长度：" << lis.size() << endl;

    // 方法对比
    cout << "\n【方法对比：O(n²) vs O(n log n)】" << endl;
    vector<int> large_test;
    for (int i = 0; i < 1000; i++) {
        large_test.push_back(rand() % 1000);
    }

    clock_t start, end;

    start = clock();
    int len1 = lengthOfLIS_basic(large_test);
    end = clock();
    cout << "O(n²) 方法：长度 = " << len1 << "，用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    start = clock();
    int len2 = lengthOfLIS_binarySearch(large_test);
    end = clock();
    cout << "O(n log n) 方法：长度 = " << len2 << "，用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    // 二分查找方法演示
    cout << "\n【二分查找方法演示】" << endl;
    vector<int> demo = {3, 1, 5, 2, 6, 4, 9};
    vector<int> lis_bs = LIS_binarySearch_reconstruct(demo);

    cout << "原数组：[3, 1, 5, 2, 6, 4, 9]" << endl;
    cout << "LIS序列：[";
    for (size_t i = 0; i < lis_bs.size(); i++) {
        cout << lis_bs[i];
        if (i < lis_bs.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    cout << "长度：" << lis_bs.size() << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "方法\t\t\t\t时间复杂度\t空间复杂度" << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << "基本递推\t\t\tO(n²)\t\tO(n)" << endl;
    cout << "二分查找优化\t\t\tO(n log n)\tO(n)" << endl;

    // 递推关系解释
    cout << "\n【递推关系】" << endl;
    cout << "dp[i] = 以 nums[i] 结尾的最长递增子序列长度" << endl;
    cout << "\n状态转移：" << endl;
    cout << "  dp[i] = max(dp[j] + 1) for all j < i and nums[j] < nums[i]" << endl;
    cout << "  dp[i] = 1 (如果没有满足条件的j)" << endl;
    cout << "\n最终答案：max(dp[i]) for all i" << endl;

    // 实际应用
    cout << "\n【实际应用】" << endl;
    cout << "1. 股票投资：最长的上涨周期" << endl;
    cout << "2. 版本控制：最长的不中断修改链" << endl;
    cout << "3. 排序算法：衡量逆序对" << endl;
    cout << "4. 生物信息学：DNA序列分析" << endl;

    // 变体问题
    cout << "\n【变体问题】" << endl;
    cout << "1. 最长非递减子序列（允许相等）" << endl;
    cout << "   修改：nums[j] <= nums[i]" << endl;
    cout << "2. 最长递减子序列" << endl;
    cout << "   方法：翻转数组或修改比较条件" << endl;
    cout << "3. 最长公共子序列（LCS）" << endl;
    cout << "   需要二维递推" << endl;

    return 0;
}
