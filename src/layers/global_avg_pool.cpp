#include "layers/global_avg_pool.h"

Tensor GAP::forward(const Tensor& x){
    input = x;

    Tensor out(x.batch, x.depth, 1, 1);
    int map_size = x.rows*x.cols;

    for(int b = 0; b < x.batch; b++){
        for(int d = 0; d < x.depth; d++){
            float sum = 0.0f;
            for(int r = 0; r < x.rows; r++){
                for(int c = 0; c < x.cols; c++){
                    sum+= x(b, r, c, d);
                }
            }
            out(b, d, 0, 0) = sum/map_size;
        }
    }

    return out;
}

Tensor GAP::backward(const Tensor& grad_out){
    Tensor grad_input(input.batch, input.rows, input.cols, input.depth);

    int map_size = input.rows * input.cols;

    for(int b = 0; b < input.batch; b++){
        for(int d = 0; d < input.depth; d++){
            float g = grad_out(b, d, 0, 0) / map_size;

            for(int r = 0; r < input.rows; r++){
                for(int c = 0; c < input.cols; c++){
                    grad_input(b, r, c, d) = g;
                }
            }
        }
    }

    return grad_input;
}
