#pragma once
#include "math/tensor.h"

class GAP{
public:
    Tensor input;

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);
};

