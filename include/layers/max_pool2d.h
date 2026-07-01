#pragma once
#include "math/tensor.h"

class MaxPool2D {
public:
    Tensor input;

    std::vector<int> max_indices;

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);
};
