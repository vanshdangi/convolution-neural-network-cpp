#pragma once
#include "math/tensor.h"

class ReLU {
public:
    const Tensor* input = nullptr;

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);
};
