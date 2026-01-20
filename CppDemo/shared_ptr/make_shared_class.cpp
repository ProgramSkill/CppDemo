#include <iostream>
#include <memory>
#include <string>
using namespace std;

class Person {
public:
    string name;
    int age;

    Person(string n, int a) : name(n), age(a) {
        cout << "构造函数被调用: " << name << endl;
    }

    ~Person() {
        cout << "析构函数被调用: " << name << endl;
    }

    void display() {
        cout << "姓名: " << name << ", 年龄: " << age << endl;
    }
};

int main() {
    {
        // 创建 Person 对象的 shared_ptr
        auto person1 = make_shared<Person>("张三", 25);
        person1->display();

        {
            // 创建第二个指向同一对象的 shared_ptr
            auto person2 = person1;
            cout << "引用计数: " << person1.use_count() << endl;
        }
        // person2 离开作用域，引用计数减1

        cout << "引用计数: " << person1.use_count() << endl;
    }


    return 0;
}
// person1 离开作用域，对象被自动销毁
