#include "layers/batch_norm.h"
#include <omp.h>
#include <cmath>

BatchNorm::BatchNorm(int channels){
    gamma = std::vector<float>(channels, 1.0f);
    beta  = std::vector<float>(channels, 0.0f);
    running_mean = std::vector<float>(channels, 0.0f);
    running_variance = std::vector<float>(channels, 1.0f);
}

Tensor BatchNorm::forward(const Tensor& x){
    
    assert(x.depth == gamma.size());
    assert(x.depth == beta.size());
    mean.resize(x.depth);
    variance.resize(x.depth);
    inv_std.resize(x.depth);
    int size = x.batch*x.rows*x.cols;
    
    if(training) {
        input = x;
        // Mean
        #pragma omp parallel for schedule(static)
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
        #pragma omp parallel for schedule(static)
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
        
        // Running Mean and Variance
        #pragma omp parallel for schedule(static)
        for(int d = 0; d < x.depth; d++){
            running_mean[d] =
                momentum * running_mean[d] +
                (1.0f - momentum) * mean[d];

            running_variance[d] =
                momentum * running_variance[d] +
                (1.0f - momentum) * variance[d];
        }
    }
    else{
        for(int d = 0; d < x.depth; d++){
            mean[d] = running_mean[d];
            variance[d] = running_variance[d];
        }
    }

    // X Hat
    x_hat = Tensor(x.batch, x.rows, x.cols, x.depth);
    const float eps = 1e-5f;
    #pragma omp parallel for schedule(static)
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


    // Output
    Tensor out(x.batch, x.rows, x.cols, x.depth);

    #pragma omp parallel for schedule(static)
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

    assert(grad_out.batch == input.batch);
    assert(grad_out.rows  == input.rows);
    assert(grad_out.cols  == input.cols);
    assert(grad_out.depth == input.depth);

    dbeta.assign(input.depth, 0.0f);
    dgamma.assign(input.depth, 0.0f);

    // dBeta and dGamma
    #pragma omp parallel for schedule(static)
    for(int d = 0; d < grad_out.depth; d++){
        float beta_sum = 0.0f;
        float gamma_sum = 0.0f;

        for(int b = 0; b < grad_out.batch; b++){
            for(int r = 0; r < grad_out.rows; r++){
                for(int c = 0; c < grad_out.cols; c++){
                    float g = grad_out(b,r,c,d);
                    beta_sum += g;
                    gamma_sum += g * x_hat(b,r,c,d);
                }
            }
        }
        dbeta[d] = beta_sum;
        dgamma[d] = gamma_sum;
    }

    int size = input.batch*input.rows*input.cols;

    // dX
    Tensor grad_input(input.batch, input.rows, input.cols, input.depth);
    #pragma omp parallel for schedule(static)
    for(int d = 0; d < grad_out.depth; d++){
        for(int b = 0; b < grad_out.batch; b++){
            for(int r = 0; r < grad_out.rows; r++){
                for(int c = 0; c < grad_out.cols; c++){
                    grad_input(b, r, c, d) = ((gamma[d]*inv_std[d])/size)*(size*grad_out(b, r, c, d) - dbeta[d] - (x_hat(b, r, c, d)*dgamma[d]));
                }
            }
        }
    }

    return grad_input;
}