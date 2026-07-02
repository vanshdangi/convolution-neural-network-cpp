#include "layers/conv2d.h"
#include "layers/relu.h"
#include "layers/max_pool2d.h"
#include "layers/dense.h"
#include "layers/batch_norm.h"
#include "layers/global_avg_pool.h"
#include <cassert>
#include <iostream>

void test_conv2d() {
    Conv2d conv(3, 16, 3, 1);

    Tensor x(4, 32, 32, 3);
    x.randomize();

    Tensor out = conv.forward(x);

    assert(out.batch == 4);
    assert(out.rows == 32);
    assert(out.cols == 32);
    assert(out.depth == 16);

    Tensor grad(out.batch, out.rows, out.cols, out.depth);
    grad.fill(1.0f);

    Tensor dx = conv.backward(grad);

    assert(dx.batch == 4);
    assert(dx.rows == 32);
    assert(dx.cols == 32);
    assert(dx.depth == 3);

    std::cout << "Conv2D test passed\n";
}

void test_relu() {
    ReLU relu;

    Tensor x(1, 2, 2, 1);
    x(0,0,0,0) = -1;
    x(0,0,1,0) = 2;
    x(0,1,0,0) = -3;
    x(0,1,1,0) = 4;

    Tensor out = relu.forward(x);

    assert(out(0,0,0,0) == 0);
    assert(out(0,0,1,0) == 2);
    assert(out(0,1,0,0) == 0);
    assert(out(0,1,1,0) == 4);

    std::cout << "ReLU test passed\n";
}

void test_maxpool() {
    MaxPool2D pool;

    Tensor x(4, 32, 32, 16);
    x.randomize();

    Tensor out = pool.forward(x);

    assert(out.rows == 16);
    assert(out.cols == 16);
    assert(out.depth == 16);

    std::cout << "MaxPool test passed\n";
}

void test_dense() {
    Dense dense(128, 10);

    Tensor x(4, 128, 1, 1);
    x.randomize();

    Tensor out = dense.forward(x);

    assert(out.batch == 4);
    assert(out.rows == 10);

    Tensor grad(4, 10, 1, 1);
    grad.fill(1.0f);

    Tensor dx = dense.backward(grad);

    assert(dx.rows == 128);

    std::cout << "Dense test passed\n";
}

void test_batchnorm() {
    BatchNorm bn(32);

    Tensor x(4, 32, 32, 32);
    x.randomize();

    Tensor out = bn.forward(x);

    assert(out.shape() == x.shape());

    std::cout << "BatchNorm test passed\n";
}

void test_gap() {
    GAP gap;

    Tensor x(4, 8, 8, 128);
    x.randomize();

    Tensor out = gap.forward(x);

    assert(out.batch == 4);
    assert(out.rows == 128);
    assert(out.cols == 1);
    assert(out.depth == 1);

    std::cout << "GAP test passed\n";
}

int main() {
    test_conv2d();
    test_relu();
    test_maxpool();
    test_dense();
    test_batchnorm();
    test_gap();

    std::cout << "All layer tests passed.\n";
    return 0;
}