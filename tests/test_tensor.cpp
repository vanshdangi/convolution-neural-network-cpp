#include "math/tensor.h"
#include <cassert>
#include <iostream>

void tensor_test() {
    std::cout << "Running tensor tests...\n";

    // Create tensor
    Tensor t(2, 3, 4, 5);

    // Shape test
    assert(t.batch == 2);
    assert(t.rows == 3);
    assert(t.cols == 4);
    assert(t.depth == 5);

    // Write values
    t(0, 0, 0, 0) = 42.0f;
    t(1, 2, 3, 4) = 99.0f;

    // Read values
    assert(t(0, 0, 0, 0) == 42.0f);
    assert(t(1, 2, 3, 4) == 99.0f);

    // Copy test
    Tensor copy = t;
    assert(copy(0, 0, 0, 0) == 42.0f);
    assert(copy(1, 2, 3, 4) == 99.0f);

    // Resize test
    t.resize(1, 2, 2, 3);

    assert(t.batch == 1);
    assert(t.rows == 2);
    assert(t.cols == 2);
    assert(t.depth == 3);

    std::cout << "Tensor tests passed.\n";
}

int main() {
    tensor_test();
    return 0;
}