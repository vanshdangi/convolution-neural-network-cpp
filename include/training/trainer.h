#pragma once
#include "network/network.h"
#include "loss/softmax_cross_entropy_loss.h"
#include "optimizer/SGD.h"
#include "training/cifar_10.h"

class Trainer {
public:
    Trainer(Network& net,
            SoftmaxCrossEntropyLoss& loss_fn,
            SGD& optimizer);

    void train(const std::vector<Sample>& train_data,const std::vector<Sample>& test_data, int epochs);
    void evaluate(const std::vector<Sample>& test_data);

private:
    Network& net;
    SoftmaxCrossEntropyLoss& loss_fn;
    SGD& optimizer;
};
