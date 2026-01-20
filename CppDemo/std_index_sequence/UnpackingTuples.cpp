#include <tuple>
#include <utility>
#include <iostream>

void print(int a, double b, const char* c) {
    std::cout << a << ", " << b << ", " << c << "\n";
}

// Helper to unpack tuple
template<typename Func, typename Tuple, std::size_t... Is>
void call_impl(Func f, Tuple t, std::index_sequence<Is...>) {
    f(std::get<Is>(t)...);
    //这不是折叠表达式（fold expression），而是参数包展开（parameter pack expansion）。
    //折叠表达式（C++17引入）：
    //(... + args)  // ((a1 + a2) + a3) + ...
    //(args + ...)  // a1 + (a2 + (a3 + ...))
    //(init + ... + args)  // ((init + a1) + a2) + ...
    //(args + ... + init)  // a1 + (a2 + (a3 + ... + init))
    //折叠表达式需要操作符（如 +, *, , 等）配合使用。
}

template<typename Func, typename Tuple>
void call(Func f, Tuple t) {
    call_impl(f, t, std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

int main() {
    auto args = std::make_tuple(42, 3.14, "hello");
    call(print, args);  // Output: 42, 3.14, hello
}