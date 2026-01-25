/*
 * 算法：阶乘计算 (Factorial)
 * 类型：递推 - 线性递推
 * 时间复杂度：O(n)
 * 空间复杂度：O(1)
 *
 * 题目：计算 n! = n × (n-1) × ... × 1
 *
 * 递推关系：
 *   f(n) = f(n-1) × n
 *   f(0) = 1
 *
 * 递推思想：
 *   从 0! 开始，逐步计算到 n!
 */

#include <iostream>
using namespace std;

// 递推解法：自底向上
long long factorial_iterative(int n) {
    long long result = 1;  // 初始条件：0! = 1

    for (int i = 1; i <= n; i++) {
        result *= i;  // 递推：f(i) = f(i-1) × i
    }

    return result;
}

// 递归解法（对比）：自顶向下
long long factorial_recursive(int n) {
    if (n <= 1) return 1;  // 基本情况
    return n * factorial_recursive(n - 1);
}

// 带溢出检查的递推解法
long long factorial_safe(int n) {
    const long long LLONG_MAX_VAL = 9223372036854775807LL;
    long long result = 1;

    for (int i = 1; i <= n; i++) {
        // 检查乘法是否会溢出
        if (result > LLONG_MAX_VAL / i) {
            cout << "警告：阶乘计算溢出！i = " << i << endl;
            return -1;
        }
        result *= i;
    }

    return result;
}

int main() {
    cout << "========== 阶乘计算 ==========" << endl;

    // 测试用例
    int test_cases[] = {0, 1, 5, 10, 15, 20};

    cout << "\n【递推 vs 递归对比】" << endl;
    cout << "n\t递推结果\t递归结果\t是否一致" << endl;
    cout << "------------------------------------------------" << endl;

    for (int n : test_cases) {
        long long iter_result = factorial_iterative(n);
        long long rec_result = factorial_recursive(n);

        cout << n << "\t" << iter_result << "\t\t" << rec_result << "\t\t";
        if (iter_result == rec_result) {
            cout << "✓" << endl;
        } else {
            cout << "✗" << endl;
        }
    }

    // 演示递推过程
    cout << "\n【递推过程演示：计算 5!】" << endl;
    int target = 5;
    long long result = 1;
    cout << "初始：result = " << result << " (0!)" << endl;

    for (int i = 1; i <= target; i++) {
        result *= i;
        cout << "步骤 " << i << ": result = " << result;
        if (i > 1) {
            cout << " (" << i << " × " << (result / i) << ")";
        }
        cout << " (" << i << "!)" << endl;
    }

    // 溢出测试
    cout << "\n【溢出测试】" << endl;
    cout << "输入一个较大的数测试是否会溢出：" << endl;
    int input;
    cin >> input;

    long long safe_result = factorial_safe(input);
    if (safe_result != -1) {
        cout << input << "! = " << safe_result << endl;
    }

    // 复杂度分析
    cout << "\n【复杂度分析】" << endl;
    cout << "时间复杂度：O(n)" << endl;
    cout << "空间复杂度：O(1) - 只需要一个变量" << endl;
    cout << "递归空间复杂度：O(n) - 调用栈深度" << endl;

    cout << "\n【递推优势】" << endl;
    cout << "1. 不会栈溢出（递归在 n 很大时会栈溢出）" << endl;
    cout << "2. 空间效率更高（O(1) vs O(n)）" << endl;
    cout << "3. 无函数调用开销，速度更快" << endl;
    cout << "4. 逻辑更直观，易于理解" << endl;

    return 0;
}
