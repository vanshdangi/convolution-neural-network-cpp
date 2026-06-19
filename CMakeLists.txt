cmake_minimum_required(VERSION 3.16)

project(convolution_neural_network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_executable(convolution_neural_network
    src/main.cpp
    src/math/tensor.cpp

    #layers
    src/layers/conv2d.cpp
    src/layers/relu.cpp
    src/layers/max_pool2d.cpp
    src/layers/flatten.cpp
    src/layers/dense.cpp
    src/layers/batch_norm.cpp
    src/layers/dropout.cpp

    #Loss
    src/loss/softmax_cross_entropy_loss.cpp

    #Network
    src/network/network.cpp

    #Training
    src/training/trainer.cpp
    src/training/cifar_10.cpp

    #Optimizer
    src/optimizer/sgd.cpp
)

target_include_directories(convolution_neural_network PRIVATE
    include
)

target_link_libraries(convolution_neural_network PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(convolution_neural_network PRIVATE /O2 /arch:AVX2)
endif()