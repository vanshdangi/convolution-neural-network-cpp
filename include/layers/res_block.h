#pragma once
#include "math/tensor.h"
#include "layers/conv2d.h"
#include "layers/batch_norm.h"
#include "layers/relu.h"

class ResidualBlock{
public:
    Conv2d conv1;
    BatchNorm b1;
    ReLU r1;
    Conv2d conv2;
    BatchNorm b2;
    ReLU r2;

    Conv2d shortcut;
    BatchNorm shortcut_bn;
    bool use_projection;

    ResidualBlock(int in_channels, int out_channels);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);

    void train();
    void eval();
};
