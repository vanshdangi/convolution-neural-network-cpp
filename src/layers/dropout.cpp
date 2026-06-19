#include "layers/dropout.h"
#include <omp.h>
#include <stdexcept>

Dropout::Dropout(float prob)
    : probability(prob), training(true), rng(std::random_device{}())
{
    if(probability < 0.0f || probability >= 1.0f)
        throw std::invalid_argument("Dropout probability must be in [0, 1).");
}

Tensor Dropout::forward(const Tensor& x) {
    input = &x;
    mask = Tensor(x.batch, x.rows, x.cols, x.depth);
    Tensor out(x.batch, x.rows, x.cols, x.depth);

    if(training) {
        #pragma omp parallel
        {
            std::mt19937 local_rng(rng());
            std::bernoulli_distribution dist(1.0f - probability);

            #pragma omp for schedule(static)
            for (int i = 0; i < x.size(); i++) {
                bool keep = dist(local_rng);
                mask.data[i] = keep ? 1.0f : 0.0f;
                out.data[i] = x.data[i] * mask.data[i];
            }
        }

        if (probability > 0.0f) {
            float scale = 1.0f / (1.0f - probability);
            for (int i = 0; i < out.size(); i++) {
                out.data[i] *= scale;
            }
        }
    } else {
        out = x;
        mask.fill(1.0f);
    }

    return out;
}

Tensor Dropout::backward(const Tensor& grad_out) {
    assert(input != nullptr);
    Tensor grad_input(input->batch, input->rows, input->cols, input->depth);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < grad_out.size(); i++) {
        grad_input.data[i] = grad_out.data[i] * mask.data[i];
    }

    if (probability > 0.0f) {
        float scale = 1.0f / (1.0f - probability);
        for (int i = 0; i < grad_input.size(); i++) {
            grad_input.data[i] *= scale;
        }
    }

    return grad_input;
}
