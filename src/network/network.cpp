#include "network/network.h"
#include <fstream>
#include <stdexcept>

Network::Network()
    : conv1(3, 32, 3),      // RGB input -> 32 feature maps
    bn1(32),
    conv2(32, 64, 3),     // 32 channels -> 64 feature maps
    bn2(64),
    d1(2304, 128),
    d2(128, 10)
{
    activations.reserve(12);
}

Tensor Network::forward(const Tensor& x) {
    activations.clear();

    activations.push_back(conv1.forward(x));
    activations.push_back(bn1.forward(activations.back()));
    activations.push_back(r1.forward(activations.back()));
    activations.push_back(maxPool1.forward(activations.back()));

    activations.push_back(conv2.forward(activations.back()));
    activations.push_back(bn2.forward(activations.back()));
    activations.push_back(r2.forward(activations.back()));
    activations.push_back(maxPool2.forward(activations.back()));

    activations.push_back(flatten.forward(activations.back()));
    activations.push_back(d1.forward(activations.back()));
    activations.push_back(r3.forward(activations.back()));

    return d2.forward(activations.back());
}

void Network::backward(const Tensor& grad_logits){
    Tensor grad = grad_logits;

    grad = d2.backward(grad);
    grad = r3.backward(grad);
    grad = d1.backward(grad);
    grad = flatten.backward(grad);

    grad = maxPool2.backward(grad);
    grad = r2.backward(grad);
    grad = bn2.backward(grad);
    grad = conv2.backward(grad);

    grad = maxPool1.backward(grad);
    grad = r1.backward(grad);
    grad = bn1.backward(grad);
    grad = conv1.backward(grad);
}

void Network::train() {
    bn1.training = true;
    bn2.training = true;
}

void Network::eval() {
    bn1.training = false;
    bn2.training = false;
}

/*
32x32x3
-> Conv(16,3)
30x30x32
-> Pool
15x15x32
-> Conv(32,3)
13x13x64
-> Pool
6x6x64
-> Flatten
2304
-> Dense(128)
128
-> Dense(10)
10
*/

void Network::save(const std::string& filename) const
{
    std::ofstream out(filename, std::ios::binary);

    if(!out)
        throw std::runtime_error("Failed to open checkpoint for saving.");

    // Conv1 filters
    for(const auto& filter : conv1.filters)
    {
        out.write(
            reinterpret_cast<const char*>(filter.data.data()),
            filter.data.size() * sizeof(float));
    }

    // Conv1 bias
    out.write(
        reinterpret_cast<const char*>(conv1.bias.data.data()),
        conv1.bias.data.size() * sizeof(float));

    // Conv2 filters
    for(const auto& filter : conv2.filters)
    {
        out.write(
            reinterpret_cast<const char*>(filter.data.data()),
            filter.data.size() * sizeof(float));
    }

    // Conv2 bias
    out.write(
        reinterpret_cast<const char*>(conv2.bias.data.data()),
        conv2.bias.data.size() * sizeof(float));

    // Dense1 weights
    out.write(
        reinterpret_cast<const char*>(d1.W.data.data()),
        d1.W.data.size() * sizeof(float));

    // Dense1 bias
    out.write(
        reinterpret_cast<const char*>(d1.b.data.data()),
        d1.b.data.size() * sizeof(float));

    // Dense2 weights
    out.write(
        reinterpret_cast<const char*>(d2.W.data.data()),
        d2.W.data.size() * sizeof(float));

    // Dense2 bias
    out.write(
        reinterpret_cast<const char*>(d2.b.data.data()),
        d2.b.data.size() * sizeof(float));

    if(!out)
        throw std::runtime_error("Failed while writing checkpoint.");
}

void Network::load(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);

    if(!in)
        throw std::runtime_error("Failed to open checkpoint for loading.");

    // Conv1 filters
    for(auto& filter : conv1.filters)
    {
        in.read(
            reinterpret_cast<char*>(filter.data.data()),
            filter.data.size() * sizeof(float));
    }

    // Conv1 bias
    in.read(
        reinterpret_cast<char*>(conv1.bias.data.data()),
        conv1.bias.data.size() * sizeof(float));

    // Conv2 filters
    for(auto& filter : conv2.filters)
    {
        in.read(
            reinterpret_cast<char*>(filter.data.data()),
            filter.data.size() * sizeof(float));
    }

    // Conv2 bias
    in.read(
        reinterpret_cast<char*>(conv2.bias.data.data()),
        conv2.bias.data.size() * sizeof(float));

    // Dense1 weights
    in.read(
        reinterpret_cast<char*>(d1.W.data.data()),
        d1.W.data.size() * sizeof(float));

    // Dense1 bias
    in.read(
        reinterpret_cast<char*>(d1.b.data.data()),
        d1.b.data.size() * sizeof(float));

    // Dense2 weights
    in.read(
        reinterpret_cast<char*>(d2.W.data.data()),
        d2.W.data.size() * sizeof(float));

    // Dense2 bias
    in.read(
        reinterpret_cast<char*>(d2.b.data.data()),
        d2.b.data.size() * sizeof(float));

    if(!in)
        throw std::runtime_error("Checkpoint is corrupted or incomplete.");
}