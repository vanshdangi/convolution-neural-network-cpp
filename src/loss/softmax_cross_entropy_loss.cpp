#include "loss/softmax_cross_entropy_loss.h"
#include <cmath>
#include <iostream>

float SoftmaxCrossEntropyLoss::forward(const Tensor& logits, int lbl) {
    label = lbl;
    probs = Tensor(logits.rows, 1, 1);

    // Stable softmax
    float max_val = logits(0, 0, 0);
    for (int i = 1; i < logits.rows; ++i)
        if (logits(i, 0, 0) > max_val)
            max_val = logits(i, 0, 0);

    float sum = 0.0f;
    for (int i = 0; i < logits.rows; ++i) {
        probs(i, 0, 0) = std::exp(logits(i, 0, 0) - max_val);
        sum += probs(i, 0, 0);
    }

    for (int i = 0; i < logits.rows; ++i) {
        probs(i, 0, 0) /= sum;
    }

    return -std::log(probs(label, 0, 0) + 1e-9f);
}

Tensor SoftmaxCrossEntropyLoss::backward() {
    Tensor grad(probs.rows, 1, 1);

    for (int i = 0; i < probs.rows; i++){
        grad(i, 0, 0) = probs(i, 0, 0);
    }
    grad(label, 0, 0) -= 1.0f;

    return grad;
}
