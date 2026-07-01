#include "optimizer/sgd.h"

SGD::SGD(float learning_rate, float wd, float momentum_coeff)
    : lr(learning_rate),
      initial_lr(lr),
      min_lr(1e-5f),
      weight_decay(wd),
      momentum(momentum_coeff),
      velocity_initialized(false) {}

void SGD::update_conv(Conv2d& conv,
                      std::vector<Tensor>& v_filters,
                      Tensor& v_bias) {

    for (int f = 0; f < conv.filters.size(); f++) {
        for (int i = 0; i < conv.filters[f].rows; i++) {
            for (int j = 0; j < conv.filters[f].cols; j++) {
                for (int k = 0; k < conv.filters[f].depth; k++) {

                    float grad =
                        conv.dW[f](i, j, k) +
                        weight_decay * conv.filters[f](i, j, k);

                    float& v = v_filters[f](i, j, k);

                    v = momentum * v - lr * grad;
                    conv.filters[f](i, j, k) += v;
                }
            }
        }
    }

    for (int i = 0; i < conv.bias.rows; i++) {
        float grad = conv.db(i, 0, 0);
        float& v = v_bias(i, 0, 0);

        v = momentum * v - lr * grad;
        conv.bias(i, 0, 0) += v;
    }
}

void SGD::update_bn(BatchNorm& bn,
                    std::vector<float>& v_gamma,
                    std::vector<float>& v_beta) {

    for (int d = 0; d < bn.gamma.size(); d++) {
        float& vg = v_gamma[d];
        float& vb = v_beta[d];

        vg = momentum * vg - lr * bn.dgamma[d];
        vb = momentum * vb - lr * bn.dbeta[d];

        bn.gamma[d] += vg;
        bn.beta[d] += vb;
    }
}

void SGD::update_resblock(
    ResidualBlock& block,
    ResidualVelocity& v) {

    update_conv(block.conv1, v.v_conv1_filters, v.v_conv1_bias);
    update_bn(block.b1, v.v_bn1_gamma, v.v_bn1_beta);

    update_conv(block.conv2, v.v_conv2_filters, v.v_conv2_bias);
    update_bn(block.b2, v.v_bn2_gamma, v.v_bn2_beta);

    if (block.use_projection) {
        update_conv(block.shortcut,
                    v.v_shortcut_filters,
                    v.v_shortcut_bias);

        update_bn(block.shortcut_bn,
                  v.v_shortcut_bn_gamma,
                  v.v_shortcut_bn_beta);
    }
}

void SGD::init_velocity(Network& net) {
    if (velocity_initialized)
        return;

    // stem
    for (const auto& f : net.conv1.filters)
        v_conv1_filters.emplace_back(f.rows, f.cols, f.depth);

    v_conv1_bias =
        Tensor(net.conv1.bias.rows,
               net.conv1.bias.cols,
               net.conv1.bias.depth);

    v_bn1_gamma.assign(net.bn1.gamma.size(), 0.0f);
    v_bn1_beta.assign(net.bn1.beta.size(), 0.0f);

    // residual blocks
    init_residual_velocity(net.res1, v_res1);

    init_residual_velocity(net.res2, v_res2);

    init_residual_velocity(net.res3, v_res3);

    // dense
    v_d1_W = Tensor(net.d1.W.rows, net.d1.W.cols, net.d1.W.depth);
    v_d1_b = Tensor(net.d1.b.rows, net.d1.b.cols, net.d1.b.depth);

    v_d2_W = Tensor(net.d2.W.rows, net.d2.W.cols, net.d2.W.depth);
    v_d2_b = Tensor(net.d2.b.rows, net.d2.b.cols, net.d2.b.depth);

    velocity_initialized = true;
}

void SGD::init_residual_velocity(
    ResidualBlock& block,
    ResidualVelocity& v) {

    for (const auto& f : block.conv1.filters)
        v.v_conv1_filters.emplace_back(f.rows, f.cols, f.depth);

    v.v_conv1_bias =
        Tensor(block.conv1.bias.rows,
               block.conv1.bias.cols,
               block.conv1.bias.depth);

    v.v_bn1_gamma.assign(block.b1.gamma.size(), 0.0f);
    v.v_bn1_beta.assign(block.b1.beta.size(), 0.0f);

    for (const auto& f : block.conv2.filters)
        v.v_conv2_filters.emplace_back(f.rows, f.cols, f.depth);

    v.v_conv2_bias =
        Tensor(block.conv2.bias.rows,
               block.conv2.bias.cols,
               block.conv2.bias.depth);

    v.v_bn2_gamma.assign(block.b2.gamma.size(), 0.0f);
    v.v_bn2_beta.assign(block.b2.beta.size(), 0.0f);

    if (block.use_projection) {
        for (const auto& f : block.shortcut.filters)
            v.v_shortcut_filters.emplace_back(f.rows, f.cols, f.depth);

        v.v_shortcut_bias =
            Tensor(block.shortcut.bias.rows,
                   block.shortcut.bias.cols,
                   block.shortcut.bias.depth);

        v.v_shortcut_bn_gamma.assign(
            block.shortcut_bn.gamma.size(), 0.0f);

        v.v_shortcut_bn_beta.assign(
            block.shortcut_bn.beta.size(), 0.0f);
    }
}

void SGD::step(Network& net) {
    init_velocity(net);

    update_conv(net.conv1, v_conv1_filters, v_conv1_bias);
    update_bn(net.bn1, v_bn1_gamma, v_bn1_beta);

    update_resblock(net.res1, v_res1);

    update_resblock(net.res2, v_res2);

    update_resblock(net.res3, v_res3);

    // d1
    for (int i = 0; i < net.d1.W.rows; i++) {
        for (int j = 0; j < net.d1.W.cols; j++) {
            float grad =
                net.d1.dW(i, j, 0) +
                weight_decay * net.d1.W(i, j, 0);

            float& v = v_d1_W(i, j, 0);

            v = momentum * v - lr * grad;
            net.d1.W(i, j, 0) += v;
        }
    }

    for (int i = 0; i < net.d1.b.rows; i++) {
        float& v = v_d1_b(i, 0, 0);

        v = momentum * v - lr * net.d1.db(i, 0, 0);
        net.d1.b(i, 0, 0) += v;
    }

    // d2
    for (int i = 0; i < net.d2.W.rows; i++) {
        for (int j = 0; j < net.d2.W.cols; j++) {
            float grad =
                net.d2.dW(i, j, 0) +
                weight_decay * net.d2.W(i, j, 0);

            float& v = v_d2_W(i, j, 0);

            v = momentum * v - lr * grad;
            net.d2.W(i, j, 0) += v;
        }
    }

    for (int i = 0; i < net.d2.b.rows; i++) {
        float& v = v_d2_b(i, 0, 0);

        v = momentum * v - lr * net.d2.db(i, 0, 0);
        net.d2.b(i, 0, 0) += v;
    }
}