// Array Size usage (Non-Type Template Parameters in Classes)
#include <iostream>
#include <array>

template<std::size_t BufferSize>
class DataSystem {
private:
    std::array<char, BufferSize> buffer;  // Fixed-size character array
    std::size_t data_count = 0;           // Number of bytes written (tracks write position)

public:
    // Default constructor to ensure buffer is initialized
    DataSystem() : buffer{} {}

    void show_size() {
        std::cout << "Buffer size: " << buffer.size() << " bytes\n";
    }

    void write(const char* data, std::size_t len) {
        if (data_count + len <= BufferSize) {
            for (std::size_t i = 0; i < len; i++) {
                buffer[data_count++] = data[i];
            }
            std::cout << "SUCCESS: Written " << len << " bytes\n";
        }
        else {
            std::cout << "ERROR: Buffer overflow! Cannot write " << len << " bytes\n";
        }
    }

    void display_buffer() {
        std::cout << "Buffer content (" << data_count << "/" << BufferSize << "): ";
        for (std::size_t i = 0; i < data_count; i++) {
            std::cout << buffer[i];
        }
        std::cout << "\n";
    }

    std::size_t available_space() {
        return BufferSize - data_count;
    }

    void clear() {
        data_count = 0;
        std::cout << "Buffer cleared\n";
    }
};

// Fixed-size stack array container
template<typename T, std::size_t Capacity>
class FixedArray {
private:
    std::array<T, Capacity> data;  // Template parameter T for element type, Capacity for array capacity
    std::size_t size = 0;           // Actual number of elements stored (dynamic)

public:
    // Default constructor  
    FixedArray() : data{} {}

    void push(const T& value) {
        if (size < Capacity) {
            data[size++] = value;
        }
        else {
            std::cout << "ERROR: Array is full!\n";
        }
    }   

    T* begin() { return data.data(); }
    T* end() { return data.data() + size; }

    void print() {
        std::cout << "Array[" << size << "/" << Capacity << "]: ";
        for (std::size_t i = 0; i < size; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";
    }

    std::size_t get_size() const { return size; }
    std::size_t get_capacity() const { return Capacity; }
};

int main() {
    std::cout << "=== Declaring Array Size (Most Common Usage) ===\n\n";

    // Different buffer sizes
    std::cout << "--- Buffer 256 bytes ---\n";
    DataSystem<256> buf256;
    buf256.show_size();
    buf256.write("Hello", 5);
    buf256.display_buffer();
    std::cout << "Available space: " << buf256.available_space() << "\n";

    std::cout << "\n--- Buffer 4096 bytes ---\n";
    DataSystem<4096> buf4096;
    buf4096.show_size();
    buf4096.write("World", 5);
    buf4096.display_buffer();
    std::cout << "Available space: " << buf4096.available_space() << "\n";

    std::cout << "\n--- Buffer 10 bytes (Small buffer test) ---\n";
    DataSystem<10> small_buf;
    small_buf.show_size();
    small_buf.write("Hello", 5);
    small_buf.display_buffer();
    small_buf.write("World", 5);
    small_buf.write("!", 1);  // This will overflow

    // Fixed-size arrays
    std::cout << "\n--- Fixed Integer Array (Capacity 10) ---\n";
    FixedArray<int, 10> int_array;
    int_array.push(10);
    int_array.push(20);
    int_array.push(30);
    int_array.push(40);
    int_array.push(50);
    int_array.print();
    std::cout << "Capacity: " << int_array.get_capacity()
        << ", Size: " << int_array.get_size() << "\n";

    std::cout << "\n--- Fixed Double Array (Capacity 5) ---\n";
    FixedArray<double, 5> double_array;
    double_array.push(3.14);
    double_array.push(2.71);
    double_array.push(1.41);
    double_array.print();
    std::cout << "Capacity: " << double_array.get_capacity()
        << ", Size: " << double_array.get_size() << "\n";

    std::cout << "\nArray size is determined at compile time, stack allocated, zero overhead!\n";

    return 0;
}