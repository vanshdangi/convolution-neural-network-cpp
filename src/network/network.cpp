#include "network/network.h"

Network::Network()
    : conv1(3, 16, 3),      // RGB input -> 16 feature maps
    conv2(16, 32, 3),     // 16 channels -> 32 feature maps
    d1(1152, 128),
    d2(128, 10)
{
    activations.reserve(10);
}

Tensor Network::forward(const Tensor& x) {
    activations.clear();

    Tensor out = conv1.forward(x);
    activations.push_back(out);

    out = r1.forward(activations.back());
    activations.push_back(out);

    out = maxPool1.forward(activations.back());
    activations.push_back(out);

    out = conv2.forward(activations.back());
    activations.push_back(out);

    out = r2.forward(activations.back());
    activations.push_back(out);

    out = maxPool2.forward(activations.back());
    activations.push_back(out);

    out = flatten.forward(activations.back());
    activations.push_back(out);

    out = d1.forward(activations.back());
    activations.push_back(out);

    out = r3.forward(activations.back());
    activations.push_back(out);

    out = d2.forward(activations.back());

    return out; // logits (10 × 1)
}

void Network::backward(const Tensor& grad_logits){
    // grad_logits: (10 × 1)
    
    Tensor grad = grad_logits;

    grad = d2.backward(grad);
    grad = r3.backward(grad);
    grad = d1.backward(grad);
    grad = flatten.backward(grad);
    grad = maxPool2.backward(grad);
    grad = r2.backward(grad);
    grad = conv2.backward(grad);
    grad = maxPool1.backward(grad);
    grad = r1.backward(grad);
    grad = conv1.backward(grad);
}

/*
32x32x3
-> Conv(16,3)
30x30x16
-> Pool
15x15x16
-> Conv(32,3)
13x13x32
-> Pool
6x6x32
-> Flatten
1152
-> Dense(128)
128
-> Dense(10)
10
*/