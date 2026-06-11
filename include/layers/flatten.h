#pragma once
#include "math/tensor.h"

class Flatten
{
public:
    const Tensor* input = nullptr;

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);
};