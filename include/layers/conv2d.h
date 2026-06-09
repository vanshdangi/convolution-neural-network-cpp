#pragma once
#include "math/tensor.h"
#include <iostream>

class Conv2d
{
public:
    Tensor input;
    int kernel_size;
    std::vector<Tensor> filters;
    Tensor bias;

    // in_channels = depth on input
    // out_channels = number of filters
    // kernel_size = size of filter
    Conv2d(int in_channels, int out_channels, int kernel_size);

    Tensor forward(const Tensor& x);
};