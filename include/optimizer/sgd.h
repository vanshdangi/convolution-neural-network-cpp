#pragma once
#include "network/network.h"

class SGD {
public:
    float lr;

    SGD(float learning_rate);

    void step(Network& net);
};
