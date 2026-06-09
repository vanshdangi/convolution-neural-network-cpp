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