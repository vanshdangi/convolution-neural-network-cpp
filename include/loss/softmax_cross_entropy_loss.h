#pragma once
#include "math/tensor.h"

class SoftmaxCrossEntropyLoss {
public:
    Tensor probs;
    int label;

    float forward(const Tensor& logits, int label);
    Tensor backward();
};