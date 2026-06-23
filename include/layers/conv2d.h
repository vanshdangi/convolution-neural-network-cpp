#pragma once
#include "math/tensor.h"
#include <iostream>

class Conv2d {
public:
    const Tensor* input = nullptr;
    int kernel_size;
    int padding;
    std::vector<Tensor> filters;
    Tensor bias;

    std::vector<Tensor> dW;
    Tensor db;

    // in_channels = depth on input
    // out_channels = number of filters
    // kernel_size = size of filter
    Conv2d(int in_channels, int out_channels, int kernel_size, int padding = 0);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);
};
