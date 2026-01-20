#include <iostream>
#include <memory>

void use_shared_ptr_by_value(std::shared_ptr<int> sp) {
    std::cout << "按值传递内部引用计数: " << sp.use_count() << std::endl;
}

void use_shared_ptr_by_reference(std::shared_ptr<int>& sp) {
    std::cout << "按引用传递内部引用计数: " << sp.use_count() << std::endl;
}

void use_raw_pointer(int* p) {
    std::cout << "使用原始指针，值为: " << *p << std::endl;
}

int main() {
    auto sp = std::make_shared<int>(5);
    std::cout << "初始引用计数: " << sp.use_count() << std::endl;  // 输出: 1

    use_shared_ptr_by_value(sp);        // 函数内输出: 2，返回后恢复为 1
    std::cout << "调用后引用计数: " << sp.use_count() << std::endl;  // 输出: 1

    use_shared_ptr_by_reference(sp);    // 输出: 1
    std::cout << "调用后引用计数: " << sp.use_count() << std::endl;  // 输出: 1

    use_raw_pointer(sp.get());          // 输出: 使用原始指针，值为: 5
    std::cout << "调用后引用计数: " << sp.use_count() << std::endl;  // 输出: 1

    return 0;
}
