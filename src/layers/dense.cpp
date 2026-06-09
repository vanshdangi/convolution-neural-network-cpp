#include "layers/dense.h"
#include <random>

Dense::Dense(int in, int out)
    : W(out, in, 1), b(out, 1,1)
{
    std::mt19937 gen(std::random_device{}());
    float stddev = std::sqrt(2.0f / in);
    std::normal_distribution<float> dist(0.0f, stddev);


    for (int i = 0; i < W.rows; ++i) {
        for (int j = 0; j < W.cols; ++j) {
            W(i, j, 0) = dist(gen);
        }
    }

    for (int i = 0; i < b.rows; ++i) {
        b(i, 0, 0) = 0.0f;
    }
}

Tensor Dense::forward(const Tensor& x) {
    // Cache input for backprop later
    input = x;

    // Output: (out, 1)
    Tensor out(W.rows, 1, 1);

    for (int i = 0; i < W.rows; ++i) {
        float sum = 0.0f;

        for (int j = 0; j < W.cols; ++j) {
            sum += W(i, j, 0) * x(j, 0, 0);
        }

        out(i, 0, 0) = sum + b(i, 0, 0);
    }

    // y = W*x + b
    return out;
}