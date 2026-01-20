#include <iostream>
#include <memory>
using namespace std;

int main() {
    // 创建指向 int 的 shared_ptr
    auto ptr1 = make_shared<int>(10);

    // 等同于（但效率更低）:
    // shared_ptr<int> ptr1(new int(10));

    cout << "值: " << *ptr1 << endl;
    cout << "引用计数: " << ptr1.use_count() << endl;

    return 0;
}
