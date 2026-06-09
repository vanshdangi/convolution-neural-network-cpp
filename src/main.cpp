#include <iostream>
#include "math/tensor.h"
#include "network/network.h"
#include "loss./softmax_cross_entropy_loss.h"
#include "training/trainer.h"
#include "training/cifar_10.h"

int main() {

    CIFAR10 dataset("data/");

    std::cout << "Img-1 :: " << dataset.train[0].label << '\n';


    Network net;
    SoftmaxCrossEntropyLoss loss_fn;

    Tensor logits = net.forward(dataset.train[0].image);

    for (int i = 0; i < logits.rows; i++)
    {
        std::cout << logits(i,0,0) << '\n';
    }

    float loss = loss_fn.forward(logits, 3);

    std::cout << "Loss :: " << loss << '\n';



    return 0;
}
