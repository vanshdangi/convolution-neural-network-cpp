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

void Trainer::train(const std::vector<Sample>& train_data, const std::vector<Sample>& test_data, int epochs){
    std::cout << "Training Begins...\n";

    std::mt19937 rng(std::random_device{}());
    std::vector<Sample> shuffled_data = train_data;

    float best_test_loss = std::numeric_limits<float>::max();
    float best_test_accuracy = 0.0f;
    int patience_counter = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();

        // Shuffle dataset every epoch
        std::shuffle(shuffled_data.begin(), shuffled_data.end(), rng);

        float total_loss = 0.0f;
        int correct = 0;

        for (const auto& sample : shuffled_data) {
            Tensor image = augment(sample.image, rng);
            int label = sample.label;

            // Forward
            Tensor logits = net.forward(image);
            float loss = loss_fn.forward(logits, label);
            total_loss += loss;

            // Accuracy
            if (argmax(logits) == label)
                correct++;

            // Backward
            Tensor grad = loss_fn.backward();
            net.backward(grad);

            // Update
            optimizer.step(net);
        }

        float train_loss = total_loss / shuffled_data.size();
        float train_accuracy =
            static_cast<float>(correct) / shuffled_data.size();

        // Evaluate on test set
        auto [test_loss, test_accuracy] = evaluate(test_data);

        // Save best model
        if (test_accuracy > best_test_accuracy){
            best_test_accuracy = test_accuracy;
            net.save("best_model.bin");
        }

        if (test_loss < best_test_loss){
            best_test_loss = test_loss;
            patience_counter = 0;
        } else{
            patience_counter++;
        }

        
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Epoch " << epoch + 1
        << " | LR: " << optimizer.lr
        << " | Train Loss: " << train_loss
        << " | Train Accuracy: "
        << train_accuracy * 100.0f << "%\n";
        
        std::cout << "Epoch Time: "
        << std::chrono::duration<double>(end - start).count()
        << " sec\n";
        
        // Auto LR scheduler
        if (patience_counter >= 3 && optimizer.lr > 1e-5f) {
            optimizer.lr *= 0.1f;
            patience_counter = 0;
            std::cout << "LR scheduled for next epoch: " << optimizer.lr << "\n";
        }
        if (patience_counter >= 8 && optimizer.lr <= 1e-5f)
            break;
        std::cout << "--------------------------------------------------\n";
    }
}

std::pair<float, float> Trainer::evaluate(const std::vector<Sample>& test_data) {

    int correct = 0;
    float total_loss = 0.0f;

    for (const auto& sample : test_data) {
        const Tensor& image = sample.image;
        int label = sample.label;

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

    return {avg_loss, accuracy};
}

Tensor Trainer::augment(const Tensor& image, std::mt19937& rng){
    Tensor out = image;

    std::uniform_int_distribution<int> flip_dist(0, 1);

    if (flip_dist(rng))
        out = horizontal_flip(out);

    out = pad_image(out, 4);
    out = random_crop(out, 32, rng);

    return out;
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

Tensor Trainer::random_crop(const Tensor& img, int crop_size, std::mt19937& rng) {
    int max_r = img.rows - crop_size;
    int max_c = img.cols - crop_size;

    std::uniform_int_distribution<int> row_dist(0, max_r);
    std::uniform_int_distribution<int> col_dist(0, max_c);

    int start_r = row_dist(rng);
    int start_c = col_dist(rng);

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