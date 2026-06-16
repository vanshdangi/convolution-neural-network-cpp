#include "layers/batch_norm.h"

BatchNorm::BatchNorm(int channels){
    gamma = std::vector<float>(channels, 1.0f);
    beta  = std::vector<float>(channels, 0.0f);
}

Tensor BatchNorm::forward(const Tensor& x){
    input = &x;

    assert(x.depth == gamma.size());
    assert(x.depth == beta.size());
    mean.resize(x.depth);
    variance.resize(x.depth);
    int size = x.batch*x.rows*x.cols;

    // Mean
    for(int d = 0; d < x.depth; d++){
        float sum = 0.0f;
        for(int b = 0; b < x.batch; b++){
            for(int r = 0; r < x.rows; r++){
                for(int c = 0; c < x.cols; c++){
                    sum += x(b, r, c, d);
                }
            }
        }
        mean[d] = sum/size;
    }

    // Variance
    for(int d = 0; d < x.depth; d++){
        float sum = 0.0f;
        for(int b = 0; b < x.batch; b++){
            for(int r = 0; r < x.rows; r++){
                for(int c = 0; c < x.cols; c++){
                    sum += (x(b, r, c, d)-mean[d])*(x(b, r, c, d)-mean[d]);
                }
            }
        }
        variance[d] = sum/size;
    }

    // X Hat
    x_hat = Tensor(x.batch, x.rows, x.cols, x.depth);
    const float eps = 1e-5f;
    for(int d = 0; d < x.depth; d++){
        inv_std[d] = 1.0/std::sqrt(variance[d] + eps);
        for(int b = 0; b < x.batch; b++){
            for(int r = 0; r < x.rows; r++){
                for(int c = 0; c < x.cols; c++){
                    x_hat(b, r, c, d) = (x(b, r, c, d) - mean[d])*inv_std[d];
                }
            }
        }
    }

    Tensor out(x.batch, x.rows, x.cols, x.depth);

    for(int d = 0; d < x.depth; d++){
        for(int b = 0; b < x.batch; b++){
            for(int r = 0; r < x.rows; r++){
                for(int c = 0; c < x.cols; c++){
                    out(b, r, c, d) = x_hat(b, r, c, d)*gamma[d] + beta[d];
                }
            }
        }
    }

    return out;
}

Tensor BatchNorm::backward(const Tensor& grad_out){

}