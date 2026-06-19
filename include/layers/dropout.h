#pragma once
#include "math/tensor.h"
#include <random>

class Dropout {
public:
    float probability;
    bool training = true;

    Dropout(float prob = 0.5f);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);

private:
    const Tensor* input = nullptr;
    Tensor mask;
    std::mt19937 rng;
};
