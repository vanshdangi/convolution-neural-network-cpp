#pragma once
#include "math/tensor.h"

class SoftmaxCrossEntropyLoss {
public:
    float smoothing = 0.1f;
    Tensor probs;
    std::vector<int> labels;

    float forward(const Tensor& logits, const std::vector<int>& labels);
    Tensor backward();
};