#pragma once

#include "math/tensor.h"
#include <string>
#include <vector>

struct Sample
{
    Tensor image;
    int label;
};

class CIFAR10
{
public:
    std::vector<Sample> train;
    std::vector<Sample> test;

    explicit CIFAR10(const std::string& dataset_path);

private:
    void load_batch(
        const std::string& filename,
        std::vector<Sample>& destination
    );
};