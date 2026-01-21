#include <iostream>

template<int Priority, std::size_t BufferSize>
class DataSystem {
private:
    int priority_level;
    int buffer_capacity;

public:
    void display() {
        std::cout << "Priority: " << Priority << "\n";
        std::cout << "BufferSize: " << BufferSize << "\n";
    }

    void advanced_info() {
        std::cout << "\nAdvanced Configuration:\n";
        std::cout << "  Priority Level: " << Priority << "\n";
        std::cout << "  Buffer Size: " << BufferSize << " bytes\n";
        std::cout << "  Is High Priority: " << (Priority > 5 ? "Yes" : "No") << "\n";
        std::cout << "  Large Buffer: " << (BufferSize > 512 ? "Yes" : "No") << "\n";
    }

    int get_priority() {
        return Priority;
    }

    std::size_t get_buffer_size() {
        return BufferSize;
    }
};

template<int Version, int Major, int Minor>
class VersionInfo {
public:
    void show() {
        std::cout << "Version: " << Version << "." << Major << "." << Minor << "\n";
    }

    bool is_stable() {
        return Major > 0;
    }

    void detailed_info() {
        std::cout << "\nDetailed Version Info:\n";
        std::cout << "  Full Version: " << Version << "." << Major << "." << Minor << "\n";
        std::cout << "  Major Version: " << Major << "\n";
        std::cout << "  Minor Version: " << Minor << "\n";
        std::cout << "  Status: " << (is_stable() ? "Stable" : "Development") << "\n";
    }
};

int main() {
    std::cout << "=== ① 直接在代码中使用参数 ===\n\n";

    // 实例化不同优先级和缓冲区大小的系统
    std::cout << "--- System 1 ---\n";
    DataSystem<5, 256> sys1;
    sys1.display();
    sys1.advanced_info();

    std::cout << "\n--- System 2 ---\n";
    DataSystem<10, 512> sys2;
    sys2.display();
    sys2.advanced_info();

    std::cout << "\n--- System 3 ---\n";
    DataSystem<2, 1024> sys3;
    sys3.display();
    sys3.advanced_info();

    // 版本信息
    std::cout << "\n\n--- Version Control ---\n";
    VersionInfo<1, 2, 3> v1;
    v1.show();
    v1.detailed_info();

    std::cout << "\n";

    VersionInfo<2, 0, 1> v2;
    v2.show();
    v2.detailed_info();

    std::cout << "\n💡 参数值编译期确定，可直接用于条件和计算！\n";

    return 0;
}