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
)

target_include_directories(convolution_neural_network PRIVATE
    include
)