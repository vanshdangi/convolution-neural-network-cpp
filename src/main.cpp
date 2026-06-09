#include "math/tensor.h"

#include <iostream>

int main() {
    
    Tensor tensor(5,5,4, 1.0f);
    tensor.resize(5,5,5);
    tensor.print();

    return 0;
}
