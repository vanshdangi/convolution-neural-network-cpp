#include "loss/softmax_cross_entropy_loss.h"
#include <cmath>
#include <iostream>

float SoftmaxCrossEntropyLoss::forward(const Tensor& logits, const std::vector<int>& labels)
{
    this->labels = labels;

    probs = Tensor(logits.batch, logits.rows, 1, 1);

    float total_loss = 0.0f;

    for (int b = 0; b < logits.batch; b++) {
        // Find max for this sample
        float max_val = logits(b, 0, 0, 0);
        for (int i = 1; i < logits.rows; i++) {
            if (logits(b, i, 0, 0) > max_val)
                max_val = logits(b, i, 0, 0);
        }

        // Exponentiate + sum
        float sum = 0.0f;
        for (int i = 0; i < logits.rows; i++) {
            probs(b, i, 0, 0) =
                std::exp(logits(b, i, 0, 0) - max_val);
            sum += probs(b, i, 0, 0);
        }

        // Normalize
        for (int i = 0; i < logits.rows; i++) {
            probs(b, i, 0, 0) /= sum;
        }

        // Loss for this sample
        total_loss += -std::log(
            probs(b, labels[b], 0, 0) + 1e-9f);
    }

    return total_loss / logits.batch;
}

Tensor SoftmaxCrossEntropyLoss::backward() {
    Tensor grad(probs.batch, probs.rows, 1, 1);

    for (int b = 0; b < probs.batch; b++) {
        for (int i = 0; i < probs.rows; i++){
            grad(b, i, 0, 0) = probs(b, i, 0, 0);
        }
        grad(b, labels[b], 0, 0) -= 1.0f;
    }
    grad /= probs.batch;

    return grad;
}
