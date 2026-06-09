#include "training/cifar_10.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

CIFAR10::CIFAR10(const std::string& dataset_path)
{
    train.reserve(50000);
    test.reserve(10000);

    load_batch(
        dataset_path + "/data_batch_1.bin",
        train
    );

    load_batch(
        dataset_path + "/data_batch_2.bin",
        train
    );

    load_batch(
        dataset_path + "/data_batch_3.bin",
        train
    );

    load_batch(
        dataset_path + "/data_batch_4.bin",
        train
    );

    load_batch(
        dataset_path + "/data_batch_5.bin",
        train
    );

    load_batch(
        dataset_path + "/test_batch.bin",
        test
    );

    std::cout
        << "Loaded CIFAR-10\n"
        << "Train samples: "
        << train.size()
        << "\nTest samples: "
        << test.size()
        << '\n';
}

void CIFAR10::load_batch(
    const std::string& filename,
    std::vector<Sample>& destination)
{
    std::ifstream file(
        filename,
        std::ios::binary
    );

    if (!file)
    {
        throw std::runtime_error(
            "Failed to open: " + filename
        );
    }

    while (true)
    {
        unsigned char label;

        file.read(
            reinterpret_cast<char*>(&label),
            1
        );

        if (!file)
            break;

        unsigned char buffer[3072];

        file.read(
            reinterpret_cast<char*>(buffer),
            3072
        );

        if (!file)
            break;

        Tensor image(32, 32, 3);

        for (int pixel = 0; pixel < 1024; pixel++)
        {
            int row = pixel / 32;
            int col = pixel % 32;

            image(row, col, 0) =
                static_cast<float>(
                    buffer[pixel]
                ) / 255.0f;

            image(row, col, 1) =
                static_cast<float>(
                    buffer[1024 + pixel]
                ) / 255.0f;

            image(row, col, 2) =
                static_cast<float>(
                    buffer[2048 + pixel]
                ) / 255.0f;
        }

        destination.push_back({
            image,
            static_cast<int>(label)
        });
    }
}