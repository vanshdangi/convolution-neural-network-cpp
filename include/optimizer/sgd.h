#pragma once
#include "network/network.h"

class SGD {
public:
    float lr;
    float weight_decay;

    SGD(float learning_rate, float wd);

    void step(Network& net);
};
