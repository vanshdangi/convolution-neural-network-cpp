#include "network/network.h"

Network::Network()
    : conv1(3, 16, 3),      // RGB input -> 16 feature maps
    conv2(16, 32, 3),     // 16 channels -> 32 feature maps
    d1(1152, 128),
    d2(128, 10)
{
}

Tensor Network::forward(const Tensor& x) {
    Tensor out = conv1.forward(x);
    out = r1.forward(out);
    out = maxPool1.forward(out);
    out = conv2.forward(out);
    out = r2.forward(out);
    out = maxPool2.forward(out);
    out = flatten.forward(out);
    out = d1.forward(out);
    out = r3.forward(out);
    out = d2.forward(out);


    return out; // logits (10 × 1)
}

void Network::backward(const Tensor& grad_logits){
    // grad_logits: (10 × 1)
    
    Tensor grad = grad_logits;
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