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

        std::vector<Sample> tiny_train(
            dataset.train.begin(),
            dataset.train.begin() + 32
        );

        Network net;
        SGD optimizer(0.02f, 0.0005f);
        SoftmaxCrossEntropyLoss loss_fn;

        Trainer trainer(net, loss_fn, optimizer);

        //trainer.train(tiny_train, tiny_train, 30, 8);
        //trainer.evaluate(tiny_train, 8);

        trainer.train(dataset.train,dataset.test, 100, 32);
        //trainer.evaluate(dataset.test, 32);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return 0;
}
