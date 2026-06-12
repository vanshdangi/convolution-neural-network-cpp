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

class Network {
public:
    Conv2d conv1;
    ReLU r1;
    MaxPool2D maxPool1;
    Conv2d conv2;
    ReLU r2;
    MaxPool2D maxPool2;
    Flatten flatten;
    Dense d1;
    ReLU r3;
    Dense d2;

    std::vector<Tensor> activations;

    Network();
    Tensor forward(const Tensor& x);
    void backward(const Tensor& grad_logits);

    void save(const std::string& filename) const;
    void load(const std::string& filename);
};