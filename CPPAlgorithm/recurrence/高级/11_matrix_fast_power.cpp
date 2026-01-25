/*
 * 算法：矩阵快速幂 (Matrix Fast Exponentiation)
 * 类型：递推优化
 * 时间复杂度：O(log n)（每次计算）
 * 空间复杂度：O(d²)（d 是矩阵维度）
 *
 * 题目：使用矩阵快速幂优化线性递推计算
 *      将 O(n) 的递推优化到 O(log n)
 *
 * 核心思想：
 *   对于线性递推，可以构造转移矩阵，使用快速幂计算
 *   例如斐波那契：F(n) = F(n-1) + F(n-2)
 *   可以表示为：[F(n+1), F(n)]^T = [[1,1],[1,0]] × [F(n), F(n-1)]^T
 */

#include <iostream>
#include <vector>
using namespace std;

const int MOD = 1e9 + 7;  // 模数，防止溢出

// 2×2 矩阵结构
struct Matrix2x2 {
    long long mat[2][2];

    Matrix2x2() {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                mat[i][j] = 0;
            }
        }
    }

    Matrix2x2(long long a, long long b, long long c, long long d) {
        mat[0][0] = a; mat[0][1] = b;
        mat[1][0] = c; mat[1][1] = d;
    }
};

// 矩阵乘法
Matrix2x2 matrixMultiply(const Matrix2x2& a, const Matrix2x2& b) {
    Matrix2x2 result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                result.mat[i][j] = (result.mat[i][j] + a.mat[i][k] * b.mat[k][j]) % MOD;
            }
        }
    }
    return result;
}

// 矩阵快速幂
Matrix2x2 matrixPow(Matrix2x2 base, long long n) {
    Matrix2x2 result(1, 0, 0, 1);  // 单位矩阵

    while (n > 0) {
        if (n % 2 == 1) {
            result = matrixMultiply(result, base);
        }
        base = matrixMultiply(base, base);
        n /= 2;
    }

    return result;
}

// 方法1：斐波那契数列（矩阵快速幂）
long long fibonacci_matrix(long long n) {
    if (n <= 1) return n;

    // 转移矩阵：[[1, 1], [1, 0]]
    Matrix2x2 base(1, 1, 1, 0);
    Matrix2x2 result = matrixPow(base, n - 1);

    return result.mat[0][0];  // F(n)
}

// 通用矩阵结构
struct Matrix {
    vector<vector<long long>> mat;

    Matrix(int n, int m) : mat(n, vector<long long>(m, 0)) {}

    static Matrix identity(int n) {
        Matrix result(n, n);
        for (int i = 0; i < n; i++) {
            result.mat[i][i] = 1;
        }
        return result;
    }
};

Matrix operator*(const Matrix& a, const Matrix& b) {
    int n = a.mat.size();
    int m = b.mat[0].size();
    int k = b.mat.size();

    Matrix result(n, m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int p = 0; p < k; p++) {
                result.mat[i][j] = (result.mat[i][j] + a.mat[i][p] * b.mat[p][j]) % MOD;
            }
        }
    }

    return result;
}

Matrix matrixPow(Matrix base, long long n) {
    int size = base.mat.size();
    Matrix result = Matrix::identity(size);

    while (n > 0) {
        if (n % 2 == 1) {
            result = result * base;
        }
        base = base * base;
        n /= 2;
    }

    return result;
}

// 方法2：Tribonacci（三阶斐波那契）
// T(n) = T(n-1) + T(n-2) + T(n-3)
long long tribonacci_matrix(long long n) {
    if (n == 0) return 0;
    if (n <= 2) return 1;

    // 转移矩阵（3×3）：
    // [T(n)]   [1 1 1] [T(n-1)]
    // [T(n-1)] = [1 0 0] × [T(n-2)]
    // [T(n-2)]   [0 1 0] [T(n-3)]

    Matrix base(3, 3);
    base.mat[0][0] = base.mat[0][1] = base.mat[0][2] = 1;
    base.mat[1][0] = 1;
    base.mat[2][1] = 1;

    Matrix result = matrixPow(base, n - 2);

    // T(n) = result[0][0] * T(2) + result[0][1] * T(1) + result[0][2] * T(0)
    // T(2) = T(1) = 1, T(0) = 0
    return (result.mat[0][0] + result.mat[0][1]) % MOD;
}

// 方法3：快速幂（普通整数）
long long fastPower(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;

    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp /= 2;
    }

    return result;
}

// 对比：普通递推
long long fibonacci_normal(long long n) {
    if (n <= 1) return n;

    long long prev2 = 0, prev1 = 1, curr;
    for (long long i = 2; i <= n; i++) {
        curr = (prev1 + prev2) % MOD;
        prev2 = prev1;
        prev1 = curr;
    }

    return prev1;
}

int main() {
    cout << "========== 矩阵快速幂 ==========" << endl;

    // 斐波那契数列测试
    cout << "\n【斐波那契数列测试】" << endl;
    cout << "n\t普通递推\t矩阵快速幂\t是否一致" << endl;
    cout << "------------------------------------------------" << endl;

    for (long long n : {10, 20, 30, 50, 100}) {
        long long normal = fibonacci_normal(n);
        long long matrix = fibonacci_matrix(n);
        cout << n << "\t" << normal << "\t\t" << matrix << "\t\t"
             << (normal == matrix ? "✓" : "✗") << endl;
    }

    // 大数测试
    cout << "\n【大数测试：F(10^18)】" << endl;
    long long large_n = 1000000000000000000LL;

    clock_t start, end;

    start = clock();
    long long result_large = fibonacci_matrix(large_n);
    end = clock();

    cout << "F(" << large_n << ") mod " << MOD << " = " << result_large << endl;
    cout << "矩阵快速幂用时："
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    // Tribonacci 测试
    cout << "\n【Tribonacci 测试】" << endl;
    cout << "n\tT(n)" << endl;
    cout << "----------------" << endl;
    for (int n : {0, 1, 2, 3, 4, 5, 10}) {
        cout << n << "\t" << tribonacci_matrix(n) << endl;
    }

    // 快速幂测试
    cout << "\n【快速幂测试】" << endl;
    cout << "2^100 = " << fastPower(2, 100, LLONG_MAX) << endl;
    cout << "3^50 mod 1000 = " << fastPower(3, 50, 1000) << endl;

    // 性能对比
    cout << "\n【性能对比：F(10^7)】" << endl;
    long long perf_n = 10000000;

    start = clock();
    long long result_normal = fibonacci_normal(perf_n);
    end = clock();
    cout << "普通递推：用时 "
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    start = clock();
    long long result_matrix = fibonacci_matrix(perf_n);
    end = clock();
    cout << "矩阵快速幂：用时 "
         << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms" << endl;

    cout << "结果一致：" << (result_normal == result_matrix ? "是 ✓" : "否 ✗") << endl;

    // 矩阵快速幂原理
    cout << "\n【矩阵快速幂原理】" << endl;
    cout << "对于斐波那契数列：" << endl;
    cout << "  F(n) = F(n-1) + F(n-2)" << endl;
    cout << endl;
    cout << "可以表示为矩阵形式：" << endl;
    cout << "  [F(n+1)]   [1 1] [F(n)  ]" << endl;
    cout << "  [F(n)  ] = [1 0]×[F(n-1)]" << endl;
    cout << endl;
    cout << "递推 n-1 次：" << endl;
    cout << "  [F(n+1)]   [1 1]^(n-1) [F(1)]" << endl;
    cout << "  [F(n)  ] = [1 0]      ×[F(0)]" << endl;
    cout << endl;
    cout << "因此只需要计算转移矩阵的 (n-1) 次幂" << endl;

    // 快速幂原理
    cout << "\n【快速幂原理】" << endl;
    cout << "计算 a^n：" << endl;
    cout << "  例如：n = 13 (二进制：1101)" << endl;
    cout << "  a^13 = a^8 × a^4 × a^1" << endl;
    cout << endl;
    cout << "算法：" << endl;
    cout << "  1. 初始化 result = 1" << endl;
    cout << "  2. 当 n > 0：" << endl;
    cout << "     - 如果 n 是奇数：result = result × base" << endl;
    cout << "     - base = base × base" << endl;
    cout << "     - n = n / 2" << endl;
    cout << endl;
    cout << "时间复杂度：O(log n)" << endl;

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "方法\t\t\t时间复杂度" << endl;
    cout << "------------------------------------------------" << endl;
    cout << "普通递推\t\tO(n)" << endl;
    cout << "递归（无优化）\tO(2^n)" << endl;
    cout << "矩阵快速幂\t\tO(d³ × log n)，d是矩阵维度" << endl;
    cout << endl;
    cout << "对于斐波那契（d=2）：" << endl;
    cout << "  时间复杂度：O(8 × log n) = O(log n)" << endl;
    cout << "  空间复杂度：O(2²) = O(1)" << endl;

    // 适用场景
    cout << "\n【适用场景】" << endl;
    cout << "1. 计算 Fibonacci(n) 当 n 非常大时（如 n > 10^6）" << endl;
    cout << "2. 线性递推数列的快速计算" << endl;
    cout << "3. 图论中计算路径数（n步后的路径数）" << endl;
    cout << "4. 密码学中的大数模幂运算" << endl;
    cout << "5. 计数问题的快速求解" << endl;

    // 通用线性递推
    cout << "\n【通用线性递推】" << endl;
    cout << "对于递推关系：" << endl;
    cout << "  f(n) = a₁×f(n-1) + a₂×f(n-2) + ... + aₖ×f(n-k)" << endl;
    cout << endl;
    cout << "可以构造 k×k 转移矩阵：" << endl;
    cout << "  [a₁  a₂  ...  aₖ]" << endl;
    cout << "  [ 1   0  ...   0 ]" << endl;
    cout << "  [ 0   1  ...   0 ]" << endl;
    cout << "  [ ... ... ... ...]" << endl;
    cout << "  [ 0   0  ...   1 ]" << endl;
    cout << endl;
    cout << "然后用快速幂计算" << endl;

    return 0;
}
