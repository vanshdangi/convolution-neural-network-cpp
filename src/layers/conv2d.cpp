#include "layers/conv2d.h"
#include <random>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size)
    : kernel_size(kernel_size){

    filters.reserve(out_channels);

    std::mt19937 gen(std::random_device{}());
    float stddev =
    std::sqrt(
            2.0f /
            (kernel_size *
            kernel_size *
            in_channels));
            std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < out_channels; i++){
        Tensor filter(kernel_size, kernel_size, in_channels);

        for (int r = 0; r < kernel_size; r++){
            for (int c = 0; c < kernel_size; c++){
                for (int d = 0; d < in_channels; d++){
                    filter(r,c,d) = dist(gen);
                }
            }
        }
        filters.push_back(filter);
    }
    bias = Tensor(out_channels, 1, 1);
    for(int i = 0; i < out_channels; i++){
        bias(i, 0, 0) = 0.0f;
    }
}

Tensor Conv2d::forward(const Tensor& x){
    input = &x;

    int H_out = x.rows - kernel_size + 1;
    int W_out = x.cols - kernel_size + 1;
    int out_channels = filters.size();

    Tensor output(x.batch, H_out, W_out, out_channels);
    
    for(int b = 0; b < x.batch; b++){
        for(int f = 0; f < out_channels; f++){
            const Tensor& filter = filters[f];

            for(int i = 0; i < H_out; i++){
                for(int j = 0; j < W_out; j++){
                    float sum = 0.0f;

                    for(int ki = 0; ki < kernel_size; ki++){
                        for(int kj = 0; kj < kernel_size; kj++){
                            for(int d = 0; d < x.depth; d++){
                                sum += x(b, i+ki, j+kj, d)*filter(ki,kj,d);
                            }
                        }
                    }
                    sum += bias(f, 0, 0);

                    output(b, i, j, f) = sum;
                }
            }
        }
    }

    return output;
}

Tensor Conv2d::backward(const Tensor& grad_out){

    dW.clear();
    dW.resize(filters.size());
    for (int f = 0; f < filters.size(); f++) {
        dW[f] = Tensor(kernel_size, kernel_size, input->depth);
    }
    db = Tensor(filters.size(), 1, 1);


    // Compute dW = grad_out * input^T
    for(int b = 0; b < grad_out.batch; b++){
        for(int f = 0; f < filters.size(); f++){
            for (int i = 0; i < grad_out.rows; ++i) {
                for (int j = 0; j < grad_out.cols; ++j) {
                    float grad = grad_out(b, i, j, f);

                    for(int ki = 0; ki < kernel_size; ki++){
                        for(int kj = 0; kj < kernel_size; kj++){
                            for(int d = 0; d < input->depth; d++){
                                dW[f](ki, kj, d) += (*input)(b, i+ki, j+kj, d)*grad;
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute grad_input = W^T * grad_out
    Tensor grad_input(input->batch, input->rows, input->cols, input->depth);

    for(int b = 0; b < grad_out.batch; b++){
        for(int f = 0; f < filters.size(); f++){

            for (int i = 0; i < grad_out.rows; ++i) {
                for (int j = 0; j < grad_out.cols; ++j) {
                    float grad = grad_out(b, i, j, f);

                    for(int ki = 0; ki < kernel_size; ki++){
                        for(int kj = 0; kj < kernel_size; kj++){
                            for(int d = 0; d < input->depth; d++){
                                grad_input(b, i+ki, j+kj, d) += filters[f](ki, kj, d) * grad;
                            }
                        }
                    }
                }
            }
        }
    }

    // Compute db = grad_out
    for(int b = 0; b < grad_out.batch; b++){
        for(int k = 0; k < grad_out.depth; k++){
            float sum = 0.0f;
            for(int i = 0; i < grad_out.rows; i++){
                for(int j = 0; j < grad_out.cols; j++){
                    sum += grad_out(b, i, j, k);
                }
            }
            db(k, 0, 0) += sum;
        }
    }

    return grad_input;
}