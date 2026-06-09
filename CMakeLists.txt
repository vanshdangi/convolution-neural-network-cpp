cmake_minimum_required(VERSION 3.16)

project(convolution_neural_network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(convolution_neural_network
    src/main.cpp
    src/math/tensor.cpp

    #layers
    src/layers/conv2d.cpp
    src/layers/relu.cpp
    src/layers/max_pool2d.cpp
    src/layers/flatten.cpp
    src/layers/dense.cpp

    #Loss
    src/loss/softmax_cross_entropy_loss.cpp

    #Network
    src/network/network.cpp

    #Trainin
    src/training/trainer.cpp
)

target_include_directories(convolution_neural_network PRIVATE
    include
)