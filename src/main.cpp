#include <iostream>
#include "math/tensor.h"
#include "network/network.h"
#include "loss/softmax_cross_entropy_loss.h"
#include "training/trainer.h"
#include "training/cifar_10.h"
#include "optimizer/sgd.h"
#include "omp.h"

int main() {
    try{
        CIFAR10 dataset("data/");

        Network net;
        SGD optimizer(0.02f, 0.0005f, 0.9f);
        SoftmaxCrossEntropyLoss loss_fn(0.1f);

        Trainer trainer(net, loss_fn, optimizer);
        trainer.train(dataset.train,dataset.test, 150, 32);
        //trainer.evaluate(dataset.test, 32);
    }
    catch(const std::exception& e){
        std::cerr << e.what() << '\n';
    }
    return 0;
}
