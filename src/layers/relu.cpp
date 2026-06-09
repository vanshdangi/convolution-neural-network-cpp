#include "layers/relu.h"

Tensor ReLU::forward(const Tensor& x) {
    input = x;  // cache for backprop later

    Tensor out(x.rows, x.cols, x.depth);

    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
            for (int k = 0; k < x.depth; k++){
                out(i, j, k) = std::max(0.0f, x(i, j, k));
            }
        }
    }

    return out;
}

Tensor ReLU::backward(const Tensor& grad_out) {
    Tensor grad_input(input.rows, input.cols, input.depth);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            for (int k = 0; k < input.depth; k++){
                if (input(i, j, k) > 0.0f)
                    grad_input(i, j, k) = grad_out(i, j, k);
                else
                    grad_input(i, j, k) = 0.0f;
            }
        }
    }

    return grad_input;
}
