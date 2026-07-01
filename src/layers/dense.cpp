#include "layers/dense.h"
#include <random>
#include <omp.h>
#include <vector>

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
    Tensor out(x.batch, W.rows, 1, 1);

    // Parallelize over batch
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < x.batch; n++) {
        for (int i = 0; i < W.rows; ++i) {
            float sum = 0.0f;

            for (int j = 0; j < W.cols; ++j) {
                sum += W(i, j, 0) * x(n, j, 0, 0);
            }

            out(n, i, 0, 0) = sum + b(i, 0, 0);
        }
    }

    // y = W*x + b
    return out;
}

Tensor Dense::backward(const Tensor& grad_out) {
    // grad_out shape: (out × 1)
    // input shape:    (in × 1)
    // W shape:        (out × in)

    int out = W.rows;
    int in  = W.cols;

    // Initialize gradients
    dW = Tensor(out, in, 1);
    db = Tensor(out, 1, 1);

    int num_threads = omp_get_max_threads();
    std::vector<Tensor> dW_private(num_threads);
    std::vector<Tensor> db_private(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        dW_private[t] = Tensor(out, in, 1);
        db_private[t] = Tensor(out, 1, 1);
    }

    // Accumulate per-thread
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < grad_out.batch; n++) {
        int tid = omp_get_thread_num();
        for (int i = 0; i < out; ++i) {
            for (int j = 0; j < in; ++j) {
                dW_private[tid](i, j, 0) += grad_out(n, i, 0, 0) * (input)(n, j, 0, 0);
            }
        }
        for (int i = 0; i < out; ++i) {
            db_private[tid](i, 0, 0) += grad_out(n, i, 0, 0);
        }
    }

    // Reduce
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < out; ++i) {
            for (int j = 0; j < in; ++j) {
                dW(i, j, 0) += dW_private[t](i, j, 0);
            }
            db(i, 0, 0) += db_private[t](i, 0, 0);
        }
    }

    // Compute grad_input = W^T * grad_out (parallelize over batch)
    Tensor grad_input(grad_out.batch, in, 1, 1);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < grad_out.batch; n++) {
        for (int j = 0; j < in; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < out; ++i) {
                sum += W(i, j, 0) * grad_out(n, i, 0, 0);
            }
            grad_input(n, j, 0, 0) = sum;
        }
    }

    return grad_input;
}