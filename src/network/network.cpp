#include "network/network.h"
#include <fstream>
#include <stdexcept>

Network::Network()
    : conv1(3, 32, 3, 1),
      bn1(32),

      res1(32, 32),
      res2(32, 64),
      res3(64, 128),

      d1(128, 256),
      dropout(0.0f),
      d2(256, 10)
{}

Tensor Network::forward(const Tensor& x) {

    Tensor out = conv1.forward(x);
    out = bn1.forward(out);
    out = r1.forward(out);

    // Stage 1
    out = res1.forward(out);
    out = maxPool1.forward(out);

    // Stage 2
    out = res2.forward(out);
    out = maxPool2.forward(out);

    // Stage 3
    out = res3.forward(out);
    out = maxPool3.forward(out);

    // Global Average Pool
    out = gap.forward(out);

    // Classifier
    out = d1.forward(out);
    out = r2.forward(out);
    out = dropout.forward(out);

    return d2.forward(out);
}

void Network::backward(const Tensor& grad_logits) {
    Tensor grad = grad_logits;

    grad = d2.backward(grad);
    grad = dropout.backward(grad);
    grad = r2.backward(grad);
    grad = d1.backward(grad);

    grad = gap.backward(grad);

    // Stage 3
    grad = maxPool3.backward(grad);
    grad = res3.backward(grad);

    // Stage 2
    grad = maxPool2.backward(grad);
    grad = res2.backward(grad);

    // Stage 1
    grad = maxPool1.backward(grad);
    grad = res1.backward(grad);

    grad = r1.backward(grad);
    grad = bn1.backward(grad);
    grad = conv1.backward(grad);
}

void Network::train() {
    bn1.training = true;

    res1.train();
    res2.train();
    res3.train();

    dropout.training = true;
}

void Network::eval() {
    bn1.training = false;

    res1.eval();
    res2.eval();
    res3.eval();

    dropout.training = false;
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

void Network::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);

    if (!out)
        throw std::runtime_error("Failed to open checkpoint for saving.");

    auto save_conv = [&](const Conv2d& conv) {
        for (const auto& filter : conv.filters)
            out.write(reinterpret_cast<const char*>(filter.data.data()),
                filter.data.size() * sizeof(float));

        out.write(reinterpret_cast<const char*>(conv.bias.data.data()),
            conv.bias.data.size() * sizeof(float));
            
    };

    auto save_bn = [&](const BatchNorm& bn) {
        out.write(reinterpret_cast<const char*>(bn.gamma.data()),
            bn.gamma.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(bn.beta.data()),
            bn.beta.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(bn.running_mean.data()),
            bn.running_mean.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(bn.running_variance.data()),
            bn.running_variance.size() * sizeof(float));
    };

    auto save_dense = [&](const Dense& dense) {
        out.write(reinterpret_cast<const char*>(dense.W.data.data()),
            dense.W.data.size() * sizeof(float));
        out.write(reinterpret_cast<const char*>(dense.b.data.data()),
            dense.b.data.size() * sizeof(float));
    };

    auto save_resblock = [&](const ResidualBlock& block) {
        save_conv(block.conv1);
        save_bn(block.b1);

        save_conv(block.conv2);
        save_bn(block.b2);

        if (block.use_projection) {
            save_conv(block.shortcut);
            save_bn(block.shortcut_bn);
        }
    };

    // Stem
    save_conv(conv1);
    save_bn(bn1);

    // Stage 1
    save_resblock(res1);

    // Stage 2
    save_resblock(res2);

    // Stage 3
    save_resblock(res3);

    // Classifier
    save_dense(d1);
    save_dense(d2);

    if (!out)
        throw std::runtime_error("Failed while writing checkpoint.");
}

void Network::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);

    if (!in)
        throw std::runtime_error("Failed to open checkpoint for loading.");

    auto load_conv = [&](Conv2d& conv) {
        for (auto& filter : conv.filters)
            in.read(reinterpret_cast<char*>(filter.data.data()),
                filter.data.size() * sizeof(float));

        in.read(reinterpret_cast<char*>(conv.bias.data.data()),
            conv.bias.data.size() * sizeof(float));
    };

    auto load_bn = [&](BatchNorm& bn) {
        in.read(reinterpret_cast<char*>(bn.gamma.data()),
            bn.gamma.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(bn.beta.data()),
            bn.beta.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(bn.running_mean.data()),
            bn.running_mean.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(bn.running_variance.data()),
            bn.running_variance.size() * sizeof(float));
    };

    auto load_dense = [&](Dense& dense) {
        in.read(reinterpret_cast<char*>(dense.W.data.data()),
            dense.W.data.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(dense.b.data.data()),
            dense.b.data.size() * sizeof(float));
    };

    auto load_resblock = [&](ResidualBlock& block) {
        load_conv(block.conv1);
        load_bn(block.b1);

        load_conv(block.conv2);
        load_bn(block.b2);

        if (block.use_projection) {
            load_conv(block.shortcut);
            load_bn(block.shortcut_bn);
        }
    };

    // Stem
    load_conv(conv1);
    load_bn(bn1);

    // Stage 1
    load_resblock(res1);

    // Stage 2
    load_resblock(res2);

    // Stage 3
    load_resblock(res3);

    // Classifier
    load_dense(d1);
    load_dense(d2);

    if (!in)
        throw std::runtime_error("Checkpoint corrupted or incomplete.");
}