#pragma once
#include "network/network.h"

class SGD {
public:
    float lr;
    float initial_lr;
    float min_lr;
    float weight_decay;
    float momentum;

    bool velocity_initialized = false;

    std::vector<Tensor> v_conv1_filters;
    Tensor v_conv1_bias;
    std::vector<float> v_bn1_gamma;
    std::vector<float> v_bn1_beta;

    std::vector<Tensor> v_conv2_filters;
    Tensor v_conv2_bias;
    std::vector<float> v_bn2_gamma;
    std::vector<float> v_bn2_beta;

    std::vector<Tensor> v_conv3_filters;
    Tensor v_conv3_bias;
    std::vector<float> v_bn3_gamma;
    std::vector<float> v_bn3_beta;

    Tensor v_d1_W;
    Tensor v_d1_b;
    Tensor v_d2_W;
    Tensor v_d2_b;

    SGD(float learning_rate, float wd, float momentum_coeff = 0.0f);

    void step(Network& net);

private:
    void init_velocity(Network& net);
};
