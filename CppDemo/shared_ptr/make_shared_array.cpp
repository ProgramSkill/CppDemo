#include <iostream>
#include <memory>
using namespace std;

int main() {
    // C++20: 创建数组的 shared_ptr
    auto arr_ptr = make_shared<int[]>(5);

    // 初始化数组
    for (int i = 0; i < 5; ++i) {
        arr_ptr[i] = i * 10;
    }

    // 输出数组
    for (int i = 0; i < 5; ++i) {
        cout << arr_ptr[i] << " ";
    }
    cout << endl;

    return 0;
}
