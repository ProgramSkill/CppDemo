//IntParameterPackExpansion
#include <iostream>

template<int... Values>
class DataSystem {
public:
    void sum_all() {
        //一元右折叠（Unary Right Fold）10+(20+(30+(40+50)))
        int total = (Values + ...);
        std::cout << "Sum: " << total << "\n";
    }

    void print_all() {
        //一元右折叠（Unary Right Fold）
        //((std::cout << 10 << " "), ((std::cout << 20 << " "), ((std::cout << 30 << " "), ((std::cout << 40 << " "), (std::cout << 50 << " ")))))
        ((std::cout << Values << " "), ...);
        std::cout << "\n";
    }
};

int main() {
    DataSystem<10, 20, 30, 40, 50> sys;
    sys.sum_all();    // 输出: Sum: 150
    sys.print_all();  // 输出: 10 20 30 40 50

    return 0;
}