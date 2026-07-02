#include "network/network.h"
#include "loss/softmax_cross_entropy_loss.h"
#include "optimizer/sgd.h"
#include "training/cifar_10.h"
#include "math/tensor.h"
#include "training/trainer.h"
#include "math/utils.h"
#include <chrono>



void overfit_test() {
    try{
        std::cout << "Running overfit test...\n";

        CIFAR10 dataset("data/");
        std::vector<Sample> small_train(
            dataset.train.begin(),
            dataset.train.begin() + 32
        );
        
        Network net;
        SGD optimizer(0.02f, 0.0f, 0.9f);
        SoftmaxCrossEntropyLoss loss_fn(0.0f);
        
        Trainer trainer(net, loss_fn, optimizer);
        
        trainer.train(small_train, small_train, 300, 32, false);

        auto [loss, acc] = trainer.evaluate(small_train, 32);

        std::cout << "Final Overfit Accuracy: " << acc * 100.0f << "%\n";

        if (acc < 0.95f) {
            throw std::runtime_error("Overfit test failed.");
        }
    }
    catch(const std::exception& e){
        std::cerr << e.what() << '\n';
    }
}

int main() {

    overfit_test();
    return 0;
}