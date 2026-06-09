#include <iostream>
#include "math/tensor.h"
#include "network/network.h"
#include "loss./softmax_cross_entropy_loss.h"

int main() {

    Tensor x(32, 32, 3);
    x.randomize();

    Network net;
    SoftmaxCrossEntropyLoss loss_fn;

    Tensor logits = net.forward(x);

    for (int i = 0; i < logits.rows; i++)
    {
        std::cout << logits(i,0,0) << '\n';
    }

    float loss = loss_fn.forward(logits, 3);

    std::cout << "Loss :: " << loss << '\n';



    return 0;
}
