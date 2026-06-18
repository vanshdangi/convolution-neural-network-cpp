#pragma once
#include "math/tensor.h"
#include <iostream>

class BatchNorm {
public:
    const Tensor* input = nullptr;
    Tensor x_hat;
    std::vector<float> mean;
    std::vector<float> variance;
    std::vector<float> inv_std;

    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> dgamma;
    std::vector<float> dbeta;

    std::vector<float> running_mean;
    std::vector<float> running_variance;
    float momentum = 0.9f;
    bool training = true;

    BatchNorm(int channels);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);
};