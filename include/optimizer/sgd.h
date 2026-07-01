#pragma once
#include "network/network.h"

struct ResidualVelocity {
    // conv1
    std::vector<Tensor> v_conv1_filters;
    Tensor v_conv1_bias;
    std::vector<float> v_bn1_gamma;
    std::vector<float> v_bn1_beta;

    // conv2
    std::vector<Tensor> v_conv2_filters;
    Tensor v_conv2_bias;
    std::vector<float> v_bn2_gamma;
    std::vector<float> v_bn2_beta;

    // shortcut (projection)
    std::vector<Tensor> v_shortcut_filters;
    Tensor v_shortcut_bias;
    std::vector<float> v_shortcut_bn_gamma;
    std::vector<float> v_shortcut_bn_beta;
};

class SGD {
public:
    float lr;
    float initial_lr;
    float min_lr;
    float weight_decay;
    float momentum;

    bool velocity_initialized = false;

    // Stem
    std::vector<Tensor> v_conv1_filters;
    Tensor v_conv1_bias;
    std::vector<float> v_bn1_gamma;
    std::vector<float> v_bn1_beta;

    // Residual stages
    ResidualVelocity v_res1;

    ResidualVelocity v_res2;

    ResidualVelocity v_res3;

    // Dense layers
    Tensor v_d1_W;
    Tensor v_d1_b;

    Tensor v_d2_W;
    Tensor v_d2_b;

    SGD(float learning_rate, float wd, float momentum_coeff = 0.0f);

    void step(Network& net);

private:
    void init_velocity(Network& net);

    void init_residual_velocity(
        ResidualBlock& block,
        ResidualVelocity& v
    );

    void update_conv(
        Conv2d& conv,
        std::vector<Tensor>& v_filters,
        Tensor& v_bias
    );

    void update_bn(
        BatchNorm& bn,
        std::vector<float>& v_gamma,
        std::vector<float>& v_beta
    );

    void update_resblock(
        ResidualBlock& block,
        ResidualVelocity& v
    );
};