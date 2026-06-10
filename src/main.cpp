#include <iostream>
#include "math/tensor.h"
#include "network/network.h"
#include "loss./softmax_cross_entropy_loss.h"
#include "training/trainer.h"
#include "training/cifar_10.h"
#include "optimizer/sgd.h"

int main() {

    CIFAR10 dataset("data/");

    std::cout << "Img-1 :: " << dataset.train[0].label << '\n';


    Network net;
    SGD optimizer(0.01f);
    SoftmaxCrossEntropyLoss loss_fn;

    for(int iter = 0; iter < 1000; iter++)
    {
        Tensor logits =
            net.forward(dataset.train[0].image);

        float loss =
            loss_fn.forward(
                logits,
                dataset.train[0].label);

        Tensor grad =
            loss_fn.backward();

        net.backward(grad);

        optimizer.step(net);

        if(iter % 50 == 0)
        {
            std::cout
                << iter
                << " "
                << loss
                << '\n';
        }
    }
    Tensor logits =
            net.forward(dataset.train[0].image);
    int pred = 0;

    for(int i = 1; i < logits.rows; i++)
    {
        if(logits(i,0,0) > logits(pred,0,0))
            pred = i;
    }

    std::cout
        << "Prediction: "
        << pred
        << '\n';
    return 0;
}
