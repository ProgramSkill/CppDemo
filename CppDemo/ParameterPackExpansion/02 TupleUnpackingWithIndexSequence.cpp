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
    );
}

void process(int a, double b, const char* c) {
    std::cout << "Received: " << a << ", " << b << ", " << c << "\n";
}

int main() {
    auto args = std::make_tuple(123, 2.71, "world");
    call_with_tuple(process, args);
    // Output: Received: 123, 2.71, world
}