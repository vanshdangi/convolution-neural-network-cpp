#include <iostream>
#include "training/trainer.h"
#include "math/utils.h"
#include <algorithm>
#include <random>
#include <chrono>

Trainer::Trainer(Network& n,
                SoftmaxCrossEntropyLoss& lf,
                SGD& opt)
    : net(n), loss_fn(lf), optimizer(opt) {}

void Trainer::train(const std::vector<Sample>& train_data, const std::vector<Sample>& test_data, int epochs) {

    std::cout << "Training Begins...\n";

    std::mt19937 rng(std::random_device{}());
    std::vector<Sample> shuffled_training_data = train_data;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        std::shuffle(shuffled_training_data.begin(), shuffled_training_data.end(), rng);

        float total_loss = 0.0f;
        int correct = 0;

        if(epoch == 5) {
            optimizer.lr *= 0.1f;
        }

        for (size_t i = 0; i < shuffled_training_data.size(); ++i) {
            const Tensor& image = shuffled_training_data[i].image;
            int label = shuffled_training_data[i].label;

            // Forward
            Tensor logits = net.forward(image);
            float loss = loss_fn.forward(logits, label);
            total_loss += loss;

            // Accuracy
            int pred = argmax(logits);
            if (pred == label)
                correct++;

            // Backward
            Tensor grad = loss_fn.backward();
            net.backward(grad);

            // Update
            optimizer.step(net);
        }

        float avg_loss = total_loss / shuffled_training_data.size();
        float accuracy = (float)correct / shuffled_training_data.size();

        auto end = std::chrono::high_resolution_clock::now();

        std::cout
            << "Epoch Time: "
            << std::chrono::duration<double>(end - start).count()
            << " sec\n";

        std::cout << "Epoch " << epoch + 1
                << " | LR: " << optimizer.lr
                << " | Loss: " << avg_loss
                << " | Accuracy: " << accuracy * 100.0f << "%\n";
        
        evaluate(test_data);
    }
}

void Trainer::evaluate(const std::vector<Sample>& test_data) {

    int correct = 0;
    float total_loss = 0.0f;

    for (size_t i = 0; i < test_data.size(); ++i) {
        const Tensor& image = test_data[i].image;
        int label = test_data[i].label;

        Tensor logits = net.forward(image);
        float loss = loss_fn.forward(logits, label);
        total_loss += loss;

        int pred = argmax(logits);
        if (pred == label)
            correct++;
    }

    float accuracy =
        (float)correct / test_data.size();
    float avg_loss =
        total_loss / test_data.size();

    std::cout << "Test Loss: " << avg_loss << "\n";
    std::cout << "Test Accuracy: "
              << accuracy * 100.0f << "%\n";
}