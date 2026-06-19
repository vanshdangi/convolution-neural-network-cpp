#pragma once
#include "math/tensor.h"
#include <vector>
//Layers
#include "layers/conv2d.h"
#include "layers/relu.h"
#include "layers/max_pool2d.h"
#include "layers/flatten.h"
#include "layers/dense.h"
#include "layers/relu.h"
#include "layers/dropout.h"
#include "layers/batch_norm.h"

class Network {
public:
    Conv2d conv1;
    BatchNorm bn1;
    ReLU r1;
    MaxPool2D maxPool1;
    
    Conv2d conv2;
    BatchNorm bn2;
    ReLU r2;
    MaxPool2D maxPool2;

    Conv2d conv3;
    BatchNorm bn3;
    ReLU r3;
    MaxPool2D maxPool3;

    Flatten flatten;
    Dense d1;
    ReLU r4;
    Dropout dropout;
    Dense d2;

    std::vector<Tensor> activations;

    Network();
    Tensor forward(const Tensor& x);
    void backward(const Tensor& grad_logits);

    void train();
    void eval();

    void save(const std::string& filename) const;
    void load(const std::string& filename);
};