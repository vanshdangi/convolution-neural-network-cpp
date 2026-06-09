#include "layers/flatten.h"

Tensor Flatten::forward(const Tensor& x){
    
    input = x;
    Tensor out(x.rows*x.cols*x.depth, 1, 1);
    int index = 0;

    for(int i = 0; i < x.rows; i++){
        for(int j = 0; j < x.cols; j++){
            for(int k = 0; k < x.depth; k++){
                out(index++,0,0) = x(i,j,k);
            }
        }
    }
    return out;
}

Tensor Flatten::backward(const Tensor& grad_out){

    Tensor grad(input.rows, input.cols, input.depth);
    int index = 0;

    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){
            for(int k = 0; k < input.depth; k++){
                grad(i, j, k) = grad_out(index, 0, 0);
                index++;
            }
        }
    }

    return grad;
}