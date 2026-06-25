#include "optimizer/sgd.h"

SGD::SGD(float learning_rate, float wd, float momentum_coeff)
    : lr(learning_rate), initial_lr(lr), min_lr(1e-5f),
    weight_decay(wd), momentum(momentum_coeff), velocity_initialized(false) {}

void SGD::init_velocity(Network& net) {
    if (velocity_initialized)
        return;

    v_conv1_filters.clear();
    for (const auto& filter : net.conv1.filters)
        v_conv1_filters.emplace_back(filter.rows, filter.cols, filter.depth);
    v_conv1_bias = Tensor(net.conv1.bias.rows, net.conv1.bias.cols, net.conv1.bias.depth);
    v_bn1_gamma.assign(net.bn1.gamma.size(), 0.0f);
    v_bn1_beta.assign(net.bn1.beta.size(), 0.0f);

    v_conv2_filters.clear();
    for (const auto& filter : net.conv2.filters)
        v_conv2_filters.emplace_back(filter.rows, filter.cols, filter.depth);
    v_conv2_bias = Tensor(net.conv2.bias.rows, net.conv2.bias.cols, net.conv2.bias.depth);
    v_bn2_gamma.assign(net.bn2.gamma.size(), 0.0f);
    v_bn2_beta.assign(net.bn2.beta.size(), 0.0f);

    v_conv3_filters.clear();
    for (const auto& filter : net.conv3.filters)
        v_conv3_filters.emplace_back(filter.rows, filter.cols, filter.depth);
    v_conv3_bias = Tensor(net.conv3.bias.rows, net.conv3.bias.cols, net.conv3.bias.depth);
    v_bn3_gamma.assign(net.bn3.gamma.size(), 0.0f);
    v_bn3_beta.assign(net.bn3.beta.size(), 0.0f);

    v_d1_W = Tensor(net.d1.W.rows, net.d1.W.cols, net.d1.W.depth);
    v_d1_b = Tensor(net.d1.b.rows, net.d1.b.cols, net.d1.b.depth);
    v_d2_W = Tensor(net.d2.W.rows, net.d2.W.cols, net.d2.W.depth);
    v_d2_b = Tensor(net.d2.b.rows, net.d2.b.cols, net.d2.b.depth);

    velocity_initialized = true;
}

void SGD::step(Network& net){
    init_velocity(net);

    // Conv1
    for (int f = 0; f < net.conv1.filters.size(); f++){
        for (int i = 0; i < net.conv1.filters[f].rows; i++){
            for (int j = 0; j < net.conv1.filters[f].cols; j++){
                for (int k = 0; k < net.conv1.filters[f].depth; k++){
                    float grad = net.conv1.dW[f](i, j, k) + (weight_decay * net.conv1.filters[f](i, j, k));
                    float& v = v_conv1_filters[f](i, j, k);
                    v = momentum * v - lr * grad;
                    net.conv1.filters[f](i, j, k) += v;
                }
            }
        }
    }
    for (int i = 0; i < net.conv1.bias.rows; i++){
        float grad = net.conv1.db(i, 0, 0);
        float& v = v_conv1_bias(i, 0, 0);
        v = momentum * v - lr * grad;
        net.conv1.bias(i, 0, 0) += v;
    }

    // BN1
    for (int d = 0; d < net.bn1.gamma.size(); d++){
        float grad_gamma = net.bn1.dgamma[d];
        float& v_gamma = v_bn1_gamma[d];
        v_gamma = momentum * v_gamma - lr * grad_gamma;
        net.bn1.gamma[d] += v_gamma;

        float grad_beta = net.bn1.dbeta[d];
        float& v_beta = v_bn1_beta[d];
        v_beta = momentum * v_beta - lr * grad_beta;
        net.bn1.beta[d] += v_beta;
    }

    // Conv2
    for (int f = 0; f < net.conv2.filters.size(); f++){
        for (int i = 0; i < net.conv2.filters[f].rows; i++){
            for (int j = 0; j < net.conv2.filters[f].cols; j++){
                for (int k = 0; k < net.conv2.filters[f].depth; k++){
                    float grad = net.conv2.dW[f](i, j, k) + (weight_decay * net.conv2.filters[f](i, j, k));
                    float& v = v_conv2_filters[f](i, j, k);
                    v = momentum * v - lr * grad;
                    net.conv2.filters[f](i, j, k) += v;
                }
            }
        }
    }
    for (int i = 0; i < net.conv2.bias.rows; i++){
        float grad = net.conv2.db(i, 0, 0);
        float& v = v_conv2_bias(i, 0, 0);
        v = momentum * v - lr * grad;
        net.conv2.bias(i, 0, 0) += v;
    }

    // BN2
    for (int d = 0; d < net.bn2.gamma.size(); d++){
        float grad_gamma = net.bn2.dgamma[d];
        float& v_gamma = v_bn2_gamma[d];
        v_gamma = momentum * v_gamma - lr * grad_gamma;
        net.bn2.gamma[d] += v_gamma;

        float grad_beta = net.bn2.dbeta[d];
        float& v_beta = v_bn2_beta[d];
        v_beta = momentum * v_beta - lr * grad_beta;
        net.bn2.beta[d] += v_beta;
    }

    // Conv3
    for (int f = 0; f < net.conv3.filters.size(); f++){
        for (int i = 0; i < net.conv3.filters[f].rows; i++){
            for (int j = 0; j < net.conv3.filters[f].cols; j++){
                for (int k = 0; k < net.conv3.filters[f].depth; k++){
                    float grad = net.conv3.dW[f](i, j, k) + (weight_decay * net.conv3.filters[f](i, j, k));
                    float& v = v_conv3_filters[f](i, j, k);
                    v = momentum * v - lr * grad;
                    net.conv3.filters[f](i, j, k) += v;
                }
            }
        }
    }
    for (int i = 0; i < net.conv3.bias.rows; i++){
        float grad = net.conv3.db(i, 0, 0);
        float& v = v_conv3_bias(i, 0, 0);
        v = momentum * v - lr * grad;
        net.conv3.bias(i, 0, 0) += v;
    }

    // BN3
    for (int d = 0; d < net.bn3.gamma.size(); d++){
        float grad_gamma = net.bn3.dgamma[d];
        float& v_gamma = v_bn3_gamma[d];
        v_gamma = momentum * v_gamma - lr * grad_gamma;
        net.bn3.gamma[d] += v_gamma;

        float grad_beta = net.bn3.dbeta[d];
        float& v_beta = v_bn3_beta[d];
        v_beta = momentum * v_beta - lr * grad_beta;
        net.bn3.beta[d] += v_beta;
    }

    // d1
    for (int i = 0; i < net.d1.W.rows; i++) {
        for (int j = 0; j < net.d1.W.cols; j++) {
            float grad = net.d1.dW(i, j, 0) + (weight_decay * net.d1.W(i, j, 0));
            float& v = v_d1_W(i, j, 0);
            v = momentum * v - lr * grad;
            net.d1.W(i, j, 0) += v;
        }
    }
    for (int i = 0; i < net.d1.b.rows; i++){
        float grad = net.d1.db(i, 0, 0);
        float& v = v_d1_b(i, 0, 0);
        v = momentum * v - lr * grad;
        net.d1.b(i, 0, 0) += v;
    }


    // d2
    for (int i = 0; i < net.d2.W.rows; i++) {
        for (int j = 0; j < net.d2.W.cols; j++) {
            float grad = net.d2.dW(i, j, 0) + (weight_decay * net.d2.W(i, j, 0));
            float& v = v_d2_W(i, j, 0);
            v = momentum * v - lr * grad;
            net.d2.W(i, j, 0) += v;
        }
    }
    for (int i = 0; i < net.d2.b.rows; i++){
        float grad = net.d2.db(i, 0, 0);
        float& v = v_d2_b(i, 0, 0);
        v = momentum * v - lr * grad;
        net.d2.b(i, 0, 0) += v;
    }
}