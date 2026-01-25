/*
 * 算法：杨辉三角 (Pascal's Triangle)
 * 类型：递推 - 二维递推
 * 时间复杂度：O(n²)
 * 空间复杂度：O(n²)
 *
 * 题目：生成前 n 行的杨辉三角
 *
 * 递推关系：
 *   C(n, k) = C(n-1, k-1) + C(n-1, k)
 *   C(n, 0) = C(n, n) = 1
 *
 * 数学含义：
 *   C(n, k) 表示从 n 个物品中选 k 个的组合数
 *   第 n 行第 k 个数 = 上一行第 k-1 个数 + 上一行第 k 个数
 */

#include <iostream>
#include <vector>
using namespace std;

// 方法1：基本递推生成杨辉三角
vector<vector<int>> generatePascalTriangle(int numRows) {
    vector<vector<int>> result(numRows);

    for (int n = 0; n < numRows; n++) {
        result[n].resize(n + 1);  // 第 n 行有 n+1 个元素

        // 边界条件：首尾都是1
        result[n][0] = result[n][n] = 1;

        // 递推计算中间元素
        for (int k = 1; k < n; k++) {
            // 递推关系：C(n, k) = C(n-1, k-1) + C(n-1, k)
            result[n][k] = result[n-1][k-1] + result[n-1][k];
        }
    }

    return result;
}

// 方法2：空间优化版（只用一维数组）
vector<vector<int>> generatePascalTriangle_optimized(int numRows) {
    vector<vector<int>> result;
    vector<int> row;

    for (int i = 0; i < numRows; i++) {
        // 从后往前更新，避免覆盖
        row.push_back(1);  // 每行最后一个元素是1

        for (int j = row.size() - 2; j > 0; j--) {
            row[j] = row[j-1] + row[j];  // 递推
        }

        result.push_back(row);
    }

    return result;
}

// 方法3：获取杨辉三角第 n 行（只计算一行）
vector<int> getPascalRow(int rowIndex) {
    vector<int> row(rowIndex + 1, 1);

    for (int i = 1; i < rowIndex; i++) {
        // 从后往前计算
        for (int j = i; j > 0; j--) {
            row[j] += row[j-1];
        }
    }

    return row;
}

// 方法4：使用组合公式直接计算（可能溢出）
long long combination(int n, int k) {
    if (k > n - k) k = n - k;  // 利用 C(n, k) = C(n, n-k)

    long long result = 1;
    for (int i = 0; i < k; i++) {
        result = result * (n - i) / (i + 1);
    }

    return result;
}

vector<int> getPascalRow_formula(int rowIndex) {
    vector<int> row(rowIndex + 1);
    for (int k = 0; k <= rowIndex; k++) {
        row[k] = combination(rowIndex, k);
    }
    return row;
}

// 打印杨辉三角（美观格式）
void printPascalTriangle(const vector<vector<int>>& triangle) {
    int numRows = triangle.size();
    int maxWidth = to_string(triangle[numRows-1][numRows/2]).length();

    for (int n = 0; n < numRows; n++) {
        // 打印前导空格（居中）
        int spaces = (numRows - n - 1) * (maxWidth + 1) / 2;
        cout << string(spaces, ' ');

        // 打印数字
        for (int k = 0; k <= n; k++) {
            cout << triangle[n][k];
            if (k < n) cout << string(maxWidth + 1 - to_string(triangle[n][k]).length(), ' ');
        }
        cout << endl;
    }
}

// 打印杨辉三角（简单格式）
void printPascalTriangle_simple(const vector<vector<int>>& triangle) {
    for (size_t n = 0; n < triangle.size(); n++) {
        // 打印前导空格
        cout << string(triangle.size() - n - 1, ' ');

        for (int num : triangle[n]) {
            cout << num << " ";
        }
        cout << endl;
    }
}

int main() {
    cout << "========== 杨辉三角 ==========" << endl;

    // 生成前10行
    int numRows = 10;
    cout << "\n【杨辉三角前" << numRows << "行】" << endl;
    vector<vector<int>> triangle = generatePascalTriangle(numRows);
    printPascalTriangle_simple(triangle);

    // 生成前5行（美观格式）
    cout << "\n【杨辉三角前5行（美观格式）】" << endl;
    vector<vector<int>> triangle5 = generatePascalTriangle(5);
    printPascalTriangle(triangle5);

    // 递推过程演示
    cout << "\n【递推过程演示：生成第5行】" << endl;
    cout << "第0行:           1" << endl;
    cout << "第1行:         1   1" << endl;
    cout << "第2行:       1   2   1     (1+1=2)" << endl;
    cout << "第3行:     1   3   3   1     (1+2=3, 2+1=3)" << endl;
    cout << "第4行:   1   4   6   4   1   (1+3=4, 3+3=6, 3+1=4)" << endl;

    // 获取指定行
    cout << "\n【获取第10行】" << endl;
    vector<int> row10 = getPascalRow(9);  // 第0行是row[0]
    cout << "第10行: ";
    for (int num : row10) {
        cout << num << " ";
    }
    cout << endl;

    // 验证组合数
    cout << "\n【验证组合数公式 C(n, k)】" << endl;
    cout << "C(5, 2) = " << combination(5, 2) << " (从5个中选2个)" << endl;
    cout << "C(10, 3) = " << combination(10, 3) << " (从10个中选3个)" << endl;
    cout << "C(10, 7) = " << combination(10, 7) << " (从10个中选7个，应等于C(10,3))" << endl;

    // 对比不同方法
    cout << "\n【方法对比：获取第10行】" << endl;
    vector<int> row_method1 = getPascalRow(9);
    vector<int> row_method2 = getPascalRow_formula(9);

    cout << "递推方法:  ";
    for (int num : row_method1) cout << num << " ";
    cout << endl;

    cout << "公式方法:  ";
    for (int num : row_method2) cout << num << " ";
    cout << endl;

    bool same = (row_method1 == row_method2);
    cout << "两种方法结果" << (same ? "相同 ✓" : "不同 ✗") << endl;

    // 杨辉三角的性质
    cout << "\n【杨辉三角的性质】" << endl;
    cout << "1. 每行数字之和：2^n" << endl;
    cout << "   例如第5行和 = ";
    int sum = 0;
    for (int num : triangle[4]) sum += num;
    cout << sum << " = 2^4 = " << (1 << 4) << endl;

    cout << "\n2. 对称性：C(n, k) = C(n, n-k)" << endl;
    cout << "   例如第7行: ";
    for (int num : getPascalRow(6)) cout << num << " ";
    cout << endl;

    cout << "\n3. 斐波那契数列：对角线之和" << endl;
    cout << "   ";
    int fib_sum = 0;
    for (int i = 0; i < 6; i++) {
        fib_sum += triangle[i][i > 0 ? i/2 : 0];  // 简化演示
        cout << fib_sum << " ";
    }
    cout << endl;

    // 应用：组合计数
    cout << "\n【应用：组合计数】" << endl;
    cout << "从5个球中选3个有多少种方法？" << endl;
    cout << "C(5, 3) = " << combination(5, 3) << endl;

    cout << "\n从10个人中选4个人组成委员会有多少种方法？" << endl;
    cout << "C(10, 4) = " << combination(10, 4) << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "生成前 n 行：" << endl;
    cout << "  时间复杂度：O(n²) - 需要计算 n(n+1)/2 个数" << endl;
    cout << "  空间复杂度：O(n²) - 存储整个三角形" << endl;
    cout << "只获取第 k 行：" << endl;
    cout << "  时间复杂度：O(k²)" << endl;
    cout << "  空间复杂度：O(k) - 只存储一行" << endl;

    return 0;
}
