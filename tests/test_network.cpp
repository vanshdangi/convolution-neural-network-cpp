#include "network/network.h"
#include "loss/softmax_cross_entropy_loss.h"
#include <cassert>
#include <iostream>

void network_test() {
    std::cout << "Running network test...\n";

    Network net;
    net.train();

    // CIFAR-10 style input: batch=4, 32x32x3
    Tensor x(4, 32, 32, 3);
    x.randomize();

    std::vector<int> labels = {1, 3, 5, 7};

    // Forward
    Tensor logits = net.forward(x);

    // Output shape check
    assert(logits.batch == 4);
    assert(logits.rows == 10);
    assert(logits.cols == 1);
    assert(logits.depth == 1);

    // Loss + backward
    SoftmaxCrossEntropyLoss loss_fn;
    float loss = loss_fn.forward(logits, labels);

    assert(loss > 0.0f);

    Tensor grad = loss_fn.backward();
    net.backward(grad);

    std::cout << "Network test passed.\n";
}

int main() {
    network_test();
    return 0;
}