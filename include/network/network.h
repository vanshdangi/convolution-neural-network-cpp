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
#include "layers/res_block.h"
#include "layers/global_avg_pool.h"

class Network {
public:
    Conv2d conv1;
    BatchNorm bn1;
    ReLU r1;
    ReLU r2;

    Dense d1;
    Dense d2;

    Dropout dropout;

    ResidualBlock res1;
    MaxPool2D maxPool1;

    ResidualBlock res2;
    MaxPool2D maxPool2;

    ResidualBlock res3;
    MaxPool2D maxPool3;

    GAP gap;

    std::vector<Tensor> activations;

    Network();
    Tensor forward(const Tensor& x);
    void backward(const Tensor& grad_logits);

    void train();
    void eval();

    void save(const std::string& filename) const;
    void load(const std::string& filename);
};