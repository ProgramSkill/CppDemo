/*
 * 算法：斐波那契数列 (Fibonacci Sequence)
 * 类型：递推 - 二阶线性递推
 * 时间复杂度：O(n)
 * 空间复杂度：O(1)
 *
 * 题目：计算斐波那契数列第 n 项
 * F(n) = F(n-1) + F(n-2)，F(0) = 0, F(1) = 1
 *
 * 递推关系：
 *   F(n) = F(n-1) + F(n-2)
 *   F(0) = 0, F(1) = 1
 */

#include <iostream>
#include <vector>
using namespace std;

// 方法1：基本递推（推荐，空间最优）
long long fibonacci_basic(int n) {
    if (n <= 1) return n;

    long long prev2 = 0;  // F(0)
    long long prev1 = 1;  // F(1)
    long long curr;

    for (int i = 2; i <= n; i++) {
        curr = prev1 + prev2;  // F(i) = F(i-1) + F(i-2)
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

// 方法2：数组递推（空间 O(n)，便于理解）
long long fibonacci_array(int n) {
    if (n <= 1) return n;

    vector<long long> dp(n + 1);
    dp[0] = 0;  // 初始条件
    dp[1] = 1;  // 初始条件

    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i-1] + dp[i-2];  // 递推公式
    }

    return dp[n];
}

// 方法3：递归解法（对比，低效 O(2^n)）
long long fibonacci_recursive(int n) {
    if (n <= 1) return n;
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2);
}

// 方法4：记忆化递归（优化到 O(n)）
vector<long long> memo;
long long fibonacci_memo(int n) {
    if (n <= 1) return n;
    if (memo[n] != -1) return memo[n];  // 已计算过，直接返回
    return memo[n] = fibonacci_memo(n-1) + fibonacci_memo(n-2);
}

// 方法5：矩阵快速幂（O(log n)，高级）
struct Matrix {
    long long mat[2][2];
    Matrix() {
        mat[0][0] = mat[0][1] = mat[1][0] = mat[1][1] = 0;
    }
};

Matrix matrixMultiply(Matrix a, Matrix b) {
    Matrix c;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                c.mat[i][j] += a.mat[i][k] * b.mat[k][j];
            }
        }
    }
    return c;
}

Matrix matrixPow(Matrix a, int n) {
    Matrix result;
    result.mat[0][0] = result.mat[1][1] = 1;  // 单位矩阵

    while (n > 0) {
        if (n % 2 == 1) result = matrixMultiply(result, a);
        a = matrixMultiply(a, a);
        n /= 2;
    }
    return result;
}

long long fibonacci_matrix(int n) {
    if (n <= 1) return n;

    Matrix base;
    base.mat[0][0] = base.mat[0][1] = base.mat[1][0] = 1;
    base.mat[1][1] = 0;

    Matrix result = matrixPow(base, n - 1);
    return result.mat[0][0];
}

int main() {
    cout << "========== 斐波那契数列 ==========" << endl;

    // 输出前20项
    cout << "\n【斐波那契数列前20项】" << endl;
    cout << "n: ";
    for (int i = 0; i < 20; i++) {
        cout << i << " ";
    }
    cout << "\nF: ";
    for (int i = 0; i < 20; i++) {
        cout << fibonacci_basic(i) << " ";
    }
    cout << endl;

    // 各种方法对比
    cout << "\n【各种方法结果对比】" << endl;
    int test_cases[] = {10, 20, 30, 40, 50};
    cout << "n\t基本递推\t数组递推\t记忆化\t矩阵快速幂" << endl;
    cout << "----------------------------------------------------------------" << endl;

    for (int n : test_cases) {
        cout << n << "\t" << fibonacci_basic(n) << "\t\t"
             << fibonacci_array(n) << "\t\t";

        // 记忆化
        memo.assign(n + 1, -1);
        cout << fibonacci_memo(n) << "\t";

        // 矩阵快速幂
        cout << fibonacci_matrix(n) << endl;
    }

    // 递推过程演示
    cout << "\n【递推过程演示：计算 F(10)】" << endl;
    int target = 10;
    long long prev2 = 0, prev1 = 1;

    cout << "F(0) = " << prev2 << endl;
    cout << "F(1) = " << prev1 << endl;

    for (int i = 2; i <= target; i++) {
        long long curr = prev1 + prev2;
        cout << "F(" << i << ") = " << curr;
        cout << " (F(" << i-1 << ") + F(" << i-2 << ") = " << prev1 << " + " << prev2 << ")" << endl;
        prev2 = prev1;
        prev1 = curr;
    }

    // 复杂度对比
    cout << "\n【复杂度对比】" << endl;
    cout << "方法\t\t\t时间复杂度\t空间复杂度" << endl;
    cout << "---------------------------------------------------" << endl;
    cout << "递归\t\t\tO(2^n)\t\tO(n)" << endl;
    cout << "记忆化递归\t\tO(n)\t\tO(n)" << endl;
    cout << "基本递推\t\tO(n)\t\tO(1)" << endl;
    cout << "数组递推\t\tO(n)\t\tO(n)" << endl;
    cout << "矩阵快速幂\t\tO(log n)\tO(1)" << endl;

    // 性能测试
    cout << "\n【性能测试：计算 F(40)】" << endl;

    // 递归（不测试，太慢）
    cout << "递归方法：跳过（O(2^n) 太慢）" << endl;

    // 基本递推
    clock_t start = clock();
    long long result = fibonacci_basic(40);
    clock_t end = clock();
    cout << "基本递推：F(40) = " << result << "，用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    // 矩阵快速幂
    start = clock();
    result = fibonacci_matrix(40);
    end = clock();
    cout << "矩阵快速幂：F(40) = " << result << "，用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    // 黄金分割比例验证
    cout << "\n【黄金分割比例验证】" << endl;
    cout << "当 n 足够大时，F(n+1)/F(n) → φ ≈ 1.618..." << endl;
    for (int n : {10, 15, 20, 25, 30}) {
        double ratio = (double)fibonacci_basic(n+1) / fibonacci_basic(n);
        cout << "F(" << n+1 << ")/F(" << n << ") = " << ratio << endl;
    }

    return 0;
}
