#include "optimizer/sgd.h"

SGD::SGD(float learning_rate, float wd)
    :lr(learning_rate), weight_decay(wd){}

void SGD::step(Network& net){

    
    // Conv1
    for (int f = 0; f < net.conv1.filters.size(); f++){
        for (int i = 0; i < net.conv1.filters[f].rows; i++){
            for (int j = 0; j < net.conv1.filters[f].cols; j++){
                for (int k = 0; k < net.conv1.filters[f].depth; k++){
                    net.conv1.filters[f](i, j, k) -= lr * (net.conv1.dW[f](i, j, k) + (weight_decay*net.conv1.filters[f](i, j, k)));
                }
            }
        }
    }
    for (int i = 0; i < net.conv1.bias.rows; i++){
        net.conv1.bias(i, 0, 0) -= lr * net.conv1.db(i, 0, 0);
    }

    // BN1
    for (int d = 0; d < net.bn1.gamma.size(); d++){
        net.bn1.gamma[d] -= lr * net.bn1.dgamma[d];
        net.bn1.beta[d] -= lr * net.bn1.dbeta[d];
    }

    // Conv2
    for (int f = 0; f < net.conv2.filters.size(); f++){
        for (int i = 0; i < net.conv2.filters[f].rows; i++){
            for (int j = 0; j < net.conv2.filters[f].cols; j++){
                for (int k = 0; k < net.conv2.filters[f].depth; k++){
                    net.conv2.filters[f](i, j, k) -= lr * (net.conv2.dW[f](i, j, k) + (weight_decay*net.conv2.filters[f](i, j, k)));
                }
            }
        }
    }
    for (int i = 0; i < net.conv2.bias.rows; i++){
        net.conv2.bias(i, 0, 0) -= lr * net.conv2.db(i, 0, 0);
    }

    // BN1
    for (int d = 0; d < net.bn2.gamma.size(); d++){
        net.bn2.gamma[d] -= lr * net.bn2.dgamma[d];
        net.bn2.beta[d] -= lr * net.bn2.dbeta[d];
    }

    // d1
    for (int i = 0; i < net.d1.W.rows; i++) {
        for (int j = 0; j < net.d1.W.cols; j++) {
            net.d1.W(i, j, 0) -= lr * (net.d1.dW(i, j, 0) + (weight_decay*net.d1.W(i, j, 0)));
        }
    }
    for (int i = 0; i < net.d1.b.rows; i++){
        net.d1.b(i, 0, 0) -= lr * net.d1.db(i, 0, 0);
    }


    // d2
    for (int i = 0; i < net.d2.W.rows; i++) {
        for (int j = 0; j < net.d2.W.cols; j++) {
            net.d2.W(i, j, 0) -= lr * (net.d2.dW(i, j, 0) + (weight_decay*net.d2.W(i, j, 0)));
        }
    }
    for (int i = 0; i < net.d2.b.rows; i++){
        net.d2.b(i, 0, 0) -= lr * net.d2.db(i, 0, 0);
    }
}