#pragma once
#include "math/tensor.h"

class SoftmaxCrossEntropyLoss {
public:
    float smoothing;
    Tensor probs;
    std::vector<int> labels;

    SoftmaxCrossEntropyLoss(float smoothing = 0.1f);

    float forward(const Tensor& logits, const std::vector<int>& labels);
    Tensor backward();
};