#include "layers/max_pool2d.h"
#include <omp.h>

// 2d max pool of size 2x2 with stride 2
Tensor MaxPool2D::forward(const Tensor& x){
    input = x;
    max_indices.resize(x.batch * x.rows / 2 * x.cols / 2 * x.depth);

    int out_rows = x.rows/2;
    int out_cols = x.cols/2;

    Tensor output(x.batch, out_rows, out_cols, x.depth);

    // Parallelize over batch
    #pragma omp parallel for schedule(static)
    for(int b = 0; b < x.batch; b++){
        for(int d = 0; d < x.depth; d++){
            for(int i = 0; i < out_rows; i++){
                for(int j = 0; j < out_cols; j++){
                    int start_row = i*2;
                    int start_col = j*2;

                    float max_val = x(b, start_row, start_col, d);
                    int max_idx = b*x.rows*x.cols*x.depth + start_row*x.cols*x.depth + start_col*x.depth + d;

                    //Find max inside 2x2 pool
                    for(int r = 0; r < 2; r++){
                        for(int c = 0; c < 2; c++){
                            int row = start_row + r;
                            int col = start_col + c;

                            if(x(b, row,col,d) > max_val){
                                max_val = x(b, row,col,d);
                                max_idx = b*x.rows*x.cols*x.depth + row*x.cols*x.depth + col*x.depth + d;
                            }
                        }
                    }
                    output(b, i,j,d) = max_val;
                    max_indices[b * out_rows * out_cols * x.depth + d * out_rows * out_cols + i * out_cols + j] = max_idx;
                }
            }
        }
    }
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_out){

    Tensor grad(input.batch, input.rows, input.cols, input.depth);

    // Parallelize over batch
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < grad_out.batch; b++) {
        for (int d = 0; d < grad_out.depth; d++) {
            for (int i = 0; i < grad_out.rows; i++) {
                for (int j = 0; j < grad_out.cols; j++) {
                    int idx = b * grad_out.rows * grad_out.cols * grad_out.depth + d * grad_out.rows * grad_out.cols + i * grad_out.cols + j;
                    grad.data[max_indices[idx]] += grad_out(b, i, j, d);
                }
            }
        }
    }

    return grad;
}