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

        if(epoch == 5 || epoch == 15) {
            optimizer.lr *= 0.1f;
        }

        for (size_t i = 0; i < shuffled_training_data.size(); ++i) {
            Tensor image = shuffled_training_data[i].image;
            int label = shuffled_training_data[i].label;

            std::uniform_int_distribution<int> flip_dist(0,1);
            if(flip_dist(rng)){
                image = horizontal_flip(image);
            }
            image = pad_image(image, 4);
            image = random_crop(image, 32);


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
        float best_accuracy = 0.0f;

        // Save best model
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            net.save("best_model.bin");
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "Epoch " << epoch + 1
                << " | LR: " << optimizer.lr
                << " | Loss: " << avg_loss
                << " | Accuracy: " << accuracy * 100.0f << "%\n";
                
        evaluate(test_data);

        std::cout
            << "Epoch Time: "
            << std::chrono::duration<double>(end - start).count()
            << " sec\n";
        std::cout << "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||" << "\n";
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

// Augmentation
Tensor Trainer::horizontal_flip(const Tensor& img){
    Tensor out(img.rows, img.cols, img.depth);

    for(int r = 0; r < img.rows; r++){
        for(int c = 0; c < img.cols; c++){
            for(int d = 0; d < img.depth; d++){
                out(r,c,d) = img(r, img.cols - 1 - c, d);
            }
        }
    }
    return out;
}

Tensor Trainer::pad_image(const Tensor& img, int pad) {
    Tensor padded(
        img.rows + 2 * pad,
        img.cols + 2 * pad,
        img.depth
    );

    for(int r = 0; r < img.rows; r++) {
        for(int c = 0; c < img.cols; c++) {
            for(int d = 0; d < img.depth; d++) {
                padded(r + pad, c + pad, d) = img(r, c, d);
            }
        }
    }

    return padded;
}

Tensor Trainer::random_crop(const Tensor& img, int crop_size) {
    int max_r = img.rows - crop_size;
    int max_c = img.cols - crop_size;

    int start_r = rand() % (max_r + 1);
    int start_c = rand() % (max_c + 1);

    Tensor cropped(crop_size, crop_size, img.depth);

    for(int r = 0; r < crop_size; r++) {
        for(int c = 0; c < crop_size; c++) {
            for(int d = 0; d < img.depth; d++) {
                cropped(r, c, d) =
                    img(start_r + r, start_c + c, d);
            }
        }
    }

    return cropped;
}