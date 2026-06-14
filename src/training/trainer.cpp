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

void Trainer::train(const std::vector<Sample>& train_data, const std::vector<Sample>& test_data, int epochs, int batch_size){
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

        for (size_t i = 0; i < shuffled_data.size(); i += batch_size) {
            int current_batch =
                std::min(batch_size,
                        (int)(shuffled_data.size() - i));

            Tensor batch(current_batch, 32, 32, 3);
            std::vector<int> labels(current_batch);

            for (int n = 0; n < current_batch; n++) {
                Tensor image = augment(shuffled_data[i + n].image, rng);
                labels[n] = shuffled_data[i + n].label;

                for (int r = 0; r < image.rows; r++) {
                    for (int c = 0; c < image.cols; c++) {
                        for (int d = 0; d < image.depth; d++) {
                            batch(n, r, c, d) = image(r, c, d);
                        }
                    }
                }
            }

            Tensor logits = net.forward(batch);

            float loss = loss_fn.forward(logits, labels);
            total_loss += loss * current_batch;

            for (int n = 0; n < current_batch; n++) {
                if (argmax(logits, n) == labels[n])
                    correct++;
            }

            Tensor grad = loss_fn.backward();
            net.backward(grad);

            optimizer.step(net);
        }

        float train_loss = total_loss / shuffled_data.size();
        float train_accuracy =
            static_cast<float>(correct) / shuffled_data.size();

        // Evaluate on test set
        auto [test_loss, test_accuracy] = evaluate(test_data, batch_size);

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

std::pair<float, float> Trainer::evaluate(const std::vector<Sample>& test_data, int batch_size)
{
    int correct = 0;
    float total_loss = 0.0f;

    for (size_t i = 0; i < test_data.size(); i += batch_size) {
        int current_batch =
            std::min(batch_size, (int)(test_data.size() - i));

        Tensor batch(current_batch, 32, 32, 3);
        std::vector<int> labels(current_batch);

        // Pack batch
        for (int n = 0; n < current_batch; n++) {
            const Tensor& image = test_data[i + n].image;
            labels[n] = test_data[i + n].label;

            for (int r = 0; r < image.rows; r++) {
                for (int c = 0; c < image.cols; c++) {
                    for (int d = 0; d < image.depth; d++) {
                        batch(n, r, c, d) = image(r, c, d);
                    }
                }
            }
        }

        // Forward
        Tensor logits = net.forward(batch);

        // Loss (already averaged over batch)
        float loss = loss_fn.forward(logits, labels);

        // Scale back to total
        total_loss += loss * current_batch;

        // Accuracy
        for (int n = 0; n < current_batch; n++) {
            int pred = argmax(logits, n);
            if (pred == labels[n])
                correct++;
        }
    }

    float avg_loss = total_loss / test_data.size();
    float accuracy =
        static_cast<float>(correct) / test_data.size();

    std::cout << "Test Loss: " << avg_loss << "\n";
    std::cout << "Test Accuracy: "
              << accuracy * 100.0f << "%\n";

    return {avg_loss, accuracy};
}

Tensor Trainer::augment(const Tensor& image, std::mt19937& rng){
    assert(image.batch == 1);
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
    assert(img.batch == 1);
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
    assert(img.batch == 1);
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
    assert(img.batch == 1);
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