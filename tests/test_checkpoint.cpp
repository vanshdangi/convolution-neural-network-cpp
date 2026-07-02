#include "network/network.h"
#include "training/cifar_10.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

void checkpoint_test() {
    try {
        std::cout << "Running checkpoint consistency test...\n";

        CIFAR10 dataset("data/");

        Tensor input = dataset.test[0].image;

        // Original model
        Network net1;
        Tensor out1 = net1.forward(input);

        // Save checkpoint
        net1.save("checkpoint_test.bin");

        // New model
        Network net2;
        net2.load("checkpoint_test.bin");

        Tensor out2 = net2.forward(input);

        // Compare outputs
        float max_diff = 0.0f;

        for (size_t i = 0; i < out1.data.size(); i++) {
            float diff = std::abs(out1.data[i] - out2.data[i]);
            if (diff > max_diff)
                max_diff = diff;
        }

        std::cout << "Max difference: " << max_diff << "\n";

        if (max_diff > 1e-6f) {
            throw std::runtime_error("Checkpoint consistency test failed.");
        }

        std::cout << "Checkpoint consistency test passed.\n";
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
}

int main() {
    checkpoint_test();
    return 0;
}