//TupleUnpackingWithIndexSequence
#include <iostream>
#include <tuple>
#include <utility>

template<typename Func, typename Tuple, std::size_t... Is>
void call_with_tuple_impl(Func f, Tuple t, std::index_sequence<Is...>) {
    f(std::get<Is>(t)...);  // Is... 展开为 0, 1, 2, ...
}

template<typename Func, typename Tuple>
void call_with_tuple(Func f, Tuple t) {
    call_with_tuple_impl(
        f, t,
        std::make_index_sequence<std::tuple_size<Tuple>::value>{}
        // std::tuple_size<...>::value = 3
        // std::make_index_sequence<3> → std::index_sequence<0, 1, 2>{}
    );
}

void process(int a, double b, const char* c) {
    std::cout << "Received: " << a << ", " << b << ", " << c << "\n";
}

int main() {
    auto args = std::make_tuple(123, 2.71, "world");
    call_with_tuple(process, args);// Output: Received: 123, 2.71, world
}
/*
std::tuple_size<Tuple>::value 获取元组大小
std::make_index_sequence 生成编译时索引序列
std::get<Is>(t)... 展开为多个 std::get 调用

template<
    typename Func,        // ← 类型模板参数（需要 typename）Type Template Parameter
    typename Tuple,       // ← 类型模板参数（需要 typename）Type Template Parameter
    std::size_t... Is     // ← 非类型模板参数（不需要 typename）Non-type Template Parameter
>

template<std::size_t... Is>  // ← 这行的作用：使 Is... 在整个函数模板中都可用。
                             // 1. 声明存在一个参数包 Is
                             // 2. 指定它是非类型参数
                             // 3. 指定它的类型是 std::size_t
// ❌ 错误示例：缺少声明
template<typename Func, typename Tuple>
void call_with_tuple_impl(Func f, Tuple t, std::index_sequence<Is...>) {
    // 编译器看到 Is... 但不知道它是什么
    // 无法进行参数推导
    // 错误：未定义的标识符 'Is'
}*/
