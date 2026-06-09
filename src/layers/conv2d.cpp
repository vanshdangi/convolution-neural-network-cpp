#include "layers/conv2d.h"
#include <random>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size)
    : kernel_size(kernel_size){

    filters.reserve(out_channels);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f);

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
    input = x;

    int H_out = x.rows - kernel_size + 1;
    int W_out = x.cols - kernel_size + 1;
    int out_channels = filters.size();

    Tensor output(H_out, W_out, out_channels);
    
    for(int f = 0; f < out_channels; f++){
        const Tensor& filter = filters[f];

        for(int i = 0; i < H_out; i++){
            for(int j = 0; j < W_out; j++){
                float sum = 0.0f;

                for(int ki = 0; ki < kernel_size; ki++){
                    for(int kj = 0; kj < kernel_size; kj++){
                        for(int d = 0; d < x.depth; d++){
                            sum += x(i+ki, j+kj, d)*filter(ki,kj,d);
                        }
                    }
                }
                sum += bias(f, 0, 0);

                output(i,j,f) = sum;
            }
        }
    }

    return output;
}