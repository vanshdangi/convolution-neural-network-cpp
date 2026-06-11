#include <iostream>
#include "math/tensor.h"
#include "network/network.h"
#include "loss./softmax_cross_entropy_loss.h"
#include "training/trainer.h"
#include "training/cifar_10.h"
#include "optimizer/sgd.h"
#include "omp.h"

int main() {
    std::cout
    << "Threads: "
    << omp_get_max_threads()
    << '\n';

    try
    {
        CIFAR10 dataset("data/");

        Network net;
        SGD optimizer(0.01f);
        SoftmaxCrossEntropyLoss loss_fn;

        Trainer trainer(net, loss_fn, optimizer);
        trainer.train(dataset.train,dataset.test, 25);
        trainer.evaluate(dataset.test);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
