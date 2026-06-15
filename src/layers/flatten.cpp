#include "layers/flatten.h"
#include <omp.h>

Tensor Flatten::forward(const Tensor& x){
    
    input = &x;
    Tensor out(x.batch, x.rows*x.cols*x.depth, 1, 1);
    
    // Parallelize over batch
    #pragma omp parallel for schedule(static)
    for(int b = 0; b < x.batch; b++){
        int index = 0;
        for(int i = 0; i < x.rows; i++){
            for(int j = 0; j < x.cols; j++){
                for(int k = 0; k < x.depth; k++){
                    out(b, index++,0,0) = x(b, i, j, k);
                }
            }
        }
    }
    return out;
}

Tensor Flatten::backward(const Tensor& grad_out){

    Tensor grad(input->batch, input->rows, input->cols, input->depth);
    
    // Parallelize over batch
    #pragma omp parallel for schedule(static)
    for(int b = 0; b < input->batch; b++){
        int index = 0;
        for(int i = 0; i < input->rows; i++){
            for(int j = 0; j < input->cols; j++){
                for(int k = 0; k < input->depth; k++){
                    grad(b, i, j, k) = grad_out(b, index, 0, 0);
                    index++;
                }
            }
        }
    }

    return grad;
}