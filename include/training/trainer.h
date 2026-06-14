#pragma once
#include "network/network.h"
#include "loss/softmax_cross_entropy_loss.h"
#include "optimizer/SGD.h"
#include "training/cifar_10.h"
#include "math/tensor.h"
#include <random>

class Trainer {
public:
    Trainer(Network& net,
            SoftmaxCrossEntropyLoss& loss_fn,
            SGD& optimizer);

    void train(const std::vector<Sample>& train_data,const std::vector<Sample>& test_data, int epochs);
    std::pair<float, float> evaluate(const std::vector<Sample>& test_data);
    
    // Augmentation
    Tensor augment(const Tensor& image, std::mt19937& rng);
    Tensor horizontal_flip(const Tensor& img);
    Tensor pad_image(const Tensor& img, int pad);
    Tensor random_crop(const Tensor& img, int crop_size, std::mt19937& rng);

private:
    Network& net;
    SoftmaxCrossEntropyLoss& loss_fn;
    SGD& optimizer;
};
