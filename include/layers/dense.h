#pragma once
#include "math/tensor.h"


class Dense {
public:
    Tensor W, b;
    Tensor dW, db;
    Tensor input;

    Dense(int in, int out);
    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);
};
