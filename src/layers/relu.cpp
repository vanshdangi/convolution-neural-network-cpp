#include "layers/relu.h"
#include <omp.h>

Tensor ReLU::forward(const Tensor& x) {
    input = &x;  // cache for backprop later

    Tensor out(x.batch, x.rows, x.cols, x.depth);

    // Parallelize over batch
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < x.batch; b++) {
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < x.cols; j++) {
                for (int k = 0; k < x.depth; k++){
                    out(b, i, j, k) = std::max(0.0f, x(b, i, j, k));
                }
            }
        }
    }

    return out;
}

Tensor ReLU::backward(const Tensor& grad_out) {
    Tensor grad_input(input->batch, input->rows, input->cols, input->depth);

    // Parallelize over batch
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < input->batch; b++) {
        for (int i = 0; i < input->rows; i++) {
            for (int j = 0; j < input->cols; j++) {
                for (int k = 0; k < input->depth; k++){
                    if ((*input)(b, i, j, k) > 0.0f)
                        grad_input(b, i, j, k) = grad_out(b, i, j, k);
                    else
                        grad_input(b, i, j, k) = 0.0f;
                }
            }
        }
    }

    return grad_input;
}
