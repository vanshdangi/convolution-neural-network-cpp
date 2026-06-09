#include "layers/max_pool2d.h"

// 2d max pool of size 2x2 with stride 2
Tensor MaxPool2D::forward(const Tensor& x){
    input = x;
    max_indices.clear();

    int out_rows = x.rows/2;
    int out_cols = x.cols/2;

    Tensor output(out_rows, out_cols, x.depth);

    for(int d = 0; d < x.depth; d++){
        for(int i = 0; i < out_rows; i++){
            for(int j = 0; j < out_cols; j++){
                int start_row = i*2;
                int start_col = j*2;

                float max_val = x(start_row, start_col, d);
                float max_idx = start_row * x.cols + start_col;

                //Find max inside 2x2 pool
                for(int r = 0; r < 2; r++){
                    for(int c = 0; c < 2; c++){
                        int row = start_row + r;
                        int col = start_col + c;

                        if(x(row,col,d) > max_val){
                            max_val = x(row,col,d);
                            max_idx = row*x.cols + col;
                        }
                    }
                }
                output(i,j,d) = max_val;
                max_indices.push_back(max_idx);
            }
        }
    }
    return output;
}