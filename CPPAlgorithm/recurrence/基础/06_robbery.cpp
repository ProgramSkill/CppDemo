/*
 * 算法：打家劫舍 (House Robber)
 * 类型：递推 - 一维DP
 * 时间复杂度：O(n)
 * 空间复杂度：O(1)
 *
 * 题目：一排房子，每个房子有一定金额，不能偷相邻的房子
 *      求能偷到的最大金额
 *
 * 递推关系：
 *   dp[i] = max(dp[i-1], dp[i-2] + nums[i])
 *   dp[0] = nums[0]
 *   dp[1] = max(nums[0], nums[1])
 *
 * 思路：
 *   对于第 i 个房子，有两种选择：
 *   1. 不偷第 i 个房子：最大金额 = dp[i-1]
 *   2. 偷第 i 个房子：不能偷第 i-1 个，最大金额 = dp[i-2] + nums[i]
 *   取两者的最大值
 */

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 方法1：基本递推（O(n)空间）
int rob_basic(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    if (n == 2) return max(nums[0], nums[1]);

    vector<int> dp(n);
    dp[0] = nums[0];
    dp[1] = max(nums[0], nums[1]);

    for (int i = 2; i < n; i++) {
        // 递推：不偷第i个 vs 偷第i个
        dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
    }

    return dp[n-1];
}

// 方法2：空间优化（O(1)空间，推荐）
int rob_optimized(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    if (n == 2) return max(nums[0], nums[1]);

    int prev2 = nums[0];           // dp[i-2]
    int prev1 = max(nums[0], nums[1]);  // dp[i-1]

    for (int i = 2; i < n; i++) {
        int curr = max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

// 方法3：返回偷窃方案（不只是金额）
vector<int> rob_with_path(vector<int>& nums) {
    int n = nums.size();
    vector<int> result;

    if (n == 0) return result;
    if (n == 1) {
        result.push_back(0);
        return result;
    }

    vector<int> dp(n);
    vector<bool> stolen(n, false);

    dp[0] = nums[0];
    stolen[0] = true;
    dp[1] = max(nums[0], nums[1]);
    stolen[1] = (nums[1] > nums[0]);

    for (int i = 2; i < n; i++) {
        if (dp[i-2] + nums[i] > dp[i-1]) {
            dp[i] = dp[i-2] + nums[i];
            stolen[i] = true;
        } else {
            dp[i] = dp[i-1];
            stolen[i] = false;
        }
    }

    // 回溯找出偷窃的房子
    int i = n - 1;
    while (i >= 0) {
        if (stolen[i]) {
            result.push_back(i);
            i -= 2;
        } else {
            i--;
        }
    }

    reverse(result.begin(), result.end());
    return result;
}

// 进阶：房子排成一圈（首尾相连）
int rob_circle(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];
    if (n == 2) return max(nums[0], nums[1]);

    // 情况1：偷第一个，不偷最后一个（考虑 [0, n-2]）
    // 情况2：不偷第一个，偷最后一个（考虑 [1, n-1]）
    // 取两种情况的最大值

    // 辅助函数：计算区间 [start, end] 的最大金额
    auto rob_range = [](vector<int>& nums, int start, int end) {
        int prev2 = nums[start];
        int prev1 = max(nums[start], nums[start + 1]);

        for (int i = start + 2; i <= end; i++) {
            int curr = max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    };

    return max(rob_range(nums, 0, n - 2), rob_range(nums, 1, n - 1));
}

// 进阶：二叉树房子（不能偷相邻节点）
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

pair<int, int> rob_tree_helper(TreeNode* root) {
    // 返回 {不偷root, 偷root}
    if (!root) return {0, 0};

    auto left = rob_tree_helper(root->left);
    auto right = rob_tree_helper(root->right);

    // 不偷当前节点：可以偷或不偷子节点，取最大值
    int not_rob = max(left.first, left.second) + max(right.first, right.second);

    // 偷当前节点：不能偷子节点
    int rob = root->val + left.first + right.first;

    return {not_rob, rob};
}

int rob_tree(TreeNode* root) {
    auto result = rob_tree_helper(root);
    return max(result.first, result.second);
}

int main() {
    cout << "========== 打家劫舍问题 ==========" << endl;

    // 基本测试
    cout << "\n【基本测试】" << endl;
    vector<vector<int>> test_cases = {
        {1, 2, 3, 1},           // 输出：4 (偷房子0和2)
        {2, 7, 9, 3, 1},        // 输出：12 (偷房子0、2、4)
        {2, 1, 1, 2},           // 输出：4 (偷房子0和3)
        {1, 2, 3},              // 输出：4 (偷房子0和2)
        {1},                    // 输出：1
        {1, 2}                  // 输出：2
    };

    cout << "房子金额\t\t最大金额" << endl;
    cout << "----------------------------------------" << endl;

    for (auto& nums : test_cases) {
        cout << "[";
        for (size_t i = 0; i < nums.size(); i++) {
            cout << nums[i];
            if (i < nums.size() - 1) cout << ", ";
        }
        cout << "]\t\t" << rob_optimized(nums) << endl;
    }

    // 递推过程演示
    cout << "\n【递推过程演示：[2, 7, 9, 3, 1]】" << endl;
    vector<int> demo = {2, 7, 9, 3, 1};
    int n = demo.size();

    cout << "初始条件：" << endl;
    cout << "  dp[0] = " << demo[0] << " (偷第0个房子)" << endl;
    cout << "  dp[1] = max(" << demo[0] << ", " << demo[1] << ") = "
         << max(demo[0], demo[1]) << endl;

    int prev2 = demo[0];
    int prev1 = max(demo[0], demo[1]);

    for (int i = 2; i < n; i++) {
        int curr = max(prev1, prev2 + demo[i]);
        cout << "\ndp[" << i << "] = max(dp[" << i-1 << "], dp[" << i-2 << "] + nums[" << i << "])" << endl;
        cout << "       = max(" << prev1 << ", " << prev2 + demo[i] << ")" << endl;
        cout << "       = " << curr;
        if (curr == prev1) {
            cout << " (不偷第" << i << "个房子)" << endl;
        } else {
            cout << " (偷第" << i << "个房子)" << endl;
        }
        prev2 = prev1;
        prev1 = curr;
    }

    cout << "\n最终答案：" << prev1 << endl;

    // 偷窃方案
    cout << "\n【偷窃方案】" << endl;
    vector<int> nums = {2, 7, 9, 3, 1};
    vector<int> path = rob_with_path(nums);

    cout << "房子：[2, 7, 9, 3, 1]" << endl;
    cout << "偷窃：";
    if (path.empty()) {
        cout << "不偷任何房子" << endl;
    } else {
        for (size_t i = 0; i < path.size(); i++) {
            cout << "房子" << path[i] << "(金额" << nums[path[i]] << ")";
            if (i < path.size() - 1) cout << " → ";
        }
        cout << endl;
        cout << "总金额：" << rob_optimized(nums) << endl;
    }

    // 环形房子
    cout << "\n【进阶：房子排成一圈】" << endl;
    vector<int> circle = {2, 3, 2};
    cout << "房子：[2, 3, 2] (首尾相连，不能同时偷第0个和第2个)" << endl;
    cout << "最大金额：" << rob_circle(circle) << endl;

    vector<int> circle2 = {1, 2, 3, 1};
    cout << "房子：[1, 2, 3, 1]" << endl;
    cout << "最大金额：" << rob_circle(circle2) << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "时间复杂度：O(n) - 遍历一次数组" << endl;
    cout << "空间复杂度：O(1) - 只需保存前两个状态" << endl;
    cout << "（对比：递归解法需要 O(2^n) 时间）" << endl;

    // 状态转移图示
    cout << "\n【状态转移示意】" << endl;
    cout << "对于第 i 个房子：" << endl;
    cout << "          ┌─────────────────┐" << endl;
    cout << "          │   第 i 个房子   │" << endl;
    cout << "          └────────┬────────┘" << endl;
    cout << "                   │" << endl;
    cout << "        ┌──────────┴──────────┐" << endl;
    cout << "        ▼                     ▼" << endl;
    cout << "   不偷第 i 个          偷第 i 个" << endl;
    cout << "   dp[i-1]              dp[i-2] + nums[i]" << endl;
    cout << "   (可以偷i-1)          (不能偷i-1)" << endl;

    // 实际应用场景
    cout << "\n【实际应用】" << endl;
    cout << "1. 安全系统规划" << endl;
    cout << "2. 资源调度（相邻任务冲突）" << endl;
    cout << "3. 投资组合选择（限制性条件）" << endl;
    cout << "4. 区间调度问题" << endl;

    return 0;
}
