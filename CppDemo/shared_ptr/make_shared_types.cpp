#include <iostream>
#include <memory>
#include <utility>
using namespace std;

int main() {
    // 创建 int 类型
    auto int_ptr = make_shared<int>(100);

    // 创建 pair 类型
    auto pair_ptr = make_shared<pair<int, int>>(30, 40);

    // 创建 string 类型
    auto str_ptr = make_shared<string>("Hello C++");

    cout << "int值: " << *int_ptr << endl;
    cout << "pair值: " << pair_ptr->first << ", "
        << pair_ptr->second << endl;
    cout << "string值: " << *str_ptr << endl;

    return 0;
}
