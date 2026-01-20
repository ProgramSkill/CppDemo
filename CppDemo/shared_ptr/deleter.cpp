#include <iostream>
#include <memory>

struct Sample {
    operator int() const { return 42; }
    operator std::string() const { return "222"; }
    Sample() { std::cout << "Sample\n"; }
    ~Sample() { std::cout << "~Sample\n"; }
};

void deleter(Sample* x) {
    std::cout << "自定义删除器\n";
    delete[] x;
}

int main() {
    {
        std::shared_ptr<Sample[]> p3(new Sample[3], deleter);
        std::cout << "Shared pointer created\n";
        std::cout <<static_cast<std::string>(p3[0]) << "\n";
        std::cout << p3.use_count() << "\n";
    }
  

    return 0;
}
