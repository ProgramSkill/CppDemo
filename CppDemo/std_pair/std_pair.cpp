#include <iostream>
#include <utility>
#include <string>
#include <map>
#include <vector>

// 函数返回两个值：成功标志和结果
std::pair<bool, double> safeDivide(double a, double b) {
    if (b == 0) {
        return { false, 0.0 };
    }
    return { true, a / b };
}

// 函数返回最小值和最大值
std::pair<int, int> findMinMax(const std::vector<int>& nums) {
    if (nums.empty()) return { 0, 0 };

    int minVal = nums[0];
    int maxVal = nums[0];

    for (int num : nums) {
        if (num < minVal) minVal = num;
        if (num > maxVal) maxVal = num;
    }

    return { minVal, maxVal };
}

int main() {
    // 1. 基本创建和访问
    std::cout << "=== 基本用法 ===" << std::endl;
    std::pair<int, std::string> p1(1, "hello");
    std::cout << "p1: " << p1.first << ", " << p1.second << std::endl;

    // 使用 make_pair
    auto p2 = std::make_pair(42, 3.14);
    std::cout << "p2: " << p2.first << ", " << p2.second << std::endl;

    // 2. 函数返回多个值
    std::cout << "\n=== 函数返回值 ===" << std::endl;
    auto result1 = safeDivide(10.0, 2.0);
    if (result1.first) {
        std::cout << "10 / 2 = " << result1.second << std::endl;
    }

    auto result2 = safeDivide(10.0, 0.0);
    if (!result2.first) {
        std::cout << "除以0失败" << std::endl;
    }

    // C++17 结构化绑定
    auto [success, value] = safeDivide(20.0, 4.0);
    std::cout << "20 / 4 = " << value << std::endl;

    // 3. 查找最小最大值
    std::cout << "\n=== 查找最小最大值 ===" << std::endl;
    std::vector<int> numbers = { 5, 2, 9, 1, 7, 3 };
    auto [minNum, maxNum] = findMinMax(numbers);
    std::cout << "最小值: " << minNum << ", 最大值: " << maxNum << std::endl;

    // 4. 与 map 配合使用
    std::cout << "\n=== 与 map 使用 ===" << std::endl;
    std::map<std::string, int> ages;

    // 插入元素
    ages.insert(std::make_pair("Alice", 30));
    ages.insert({ "Bob", 25 });

    // 遍历 map (每个元素是 pair)
    for (const auto& person : ages) {
        std::cout << person.first << " 的年龄是 " << person.second << std::endl;
    }

    // C++17 结构化绑定
    for (const auto& [name, age] : ages) {
        std::cout << name << ": " << age << " 岁" << std::endl;
    }

    // 5. pair 的比较
    std::cout << "\n=== pair 比较 ===" << std::endl;
    std::pair<int, int> a(1, 2);
    std::pair<int, int> b(1, 3);
    std::pair<int, int> c(2, 1);

    std::cout << "a < b: " << (a < b) << std::endl;  // true (比较 second)
    std::cout << "a < c: " << (a < c) << std::endl;  // true (比较 first)
    std::cout << "a == b: " << (a == b) << std::endl; // false

    // 6. 存储不同类型
    std::cout << "\n=== 不同类型组合 ===" << std::endl;
    std::pair<std::string, std::vector<int>> studentScores("张三", { 85, 90, 92 });
    std::cout << studentScores.first << " 的成绩: ";
    for (int score : studentScores.second) {
        std::cout << score << " ";
    }
    std::cout << std::endl;

    return 0;
}