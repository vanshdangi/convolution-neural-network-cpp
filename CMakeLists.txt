cmake_minimum_required(VERSION 3.16)

project(convolution_neural_network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

# Shared source files
set(CNN_SOURCES
    src/math/tensor.cpp

    # Layers
    src/layers/conv2d.cpp
    src/layers/relu.cpp
    src/layers/max_pool2d.cpp
    src/layers/flatten.cpp
    src/layers/dense.cpp
    src/layers/batch_norm.cpp
    src/layers/dropout.cpp
    src/layers/global_avg_pool.cpp
    src/layers/res_block.cpp

    # Loss
    src/loss/softmax_cross_entropy_loss.cpp

    # Network
    src/network/network.cpp

    # Training
    src/training/trainer.cpp
    src/training/cifar_10.cpp

    # Optimizer
    src/optimizer/sgd.cpp
)

# Main executable
add_executable(convolution_neural_network
    src/main.cpp
    ${CNN_SOURCES}
)

target_include_directories(convolution_neural_network PRIVATE include)
target_link_libraries(convolution_neural_network PRIVATE OpenMP::OpenMP_CXX)

# Overfit test executable
add_executable(test_overfit
    tests/test_overfit.cpp
    ${CNN_SOURCES}
)

target_include_directories(test_overfit PRIVATE include)
target_link_libraries(test_overfit PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(convolution_neural_network PRIVATE /O2 /arch:AVX2)
    target_compile_options(test_overfit PRIVATE /O2 /arch:AVX2)
endif()

# Checkpoint Consistency Test

add_executable(test_checkpoint
    tests/test_checkpoint.cpp
    ${CNN_SOURCES}
)

target_include_directories(test_checkpoint PRIVATE include)
target_link_libraries(test_checkpoint PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(test_checkpoint PRIVATE /O2 /arch:AVX2)
endif()

# Tensor Test

add_executable(test_tensor
    tests/test_tensor.cpp
    src/math/tensor.cpp
)

target_include_directories(test_tensor PRIVATE include)

# Network Test

add_executable(test_network
    tests/test_network.cpp
    ${CNN_SOURCES}
)

target_include_directories(test_network PRIVATE include)
target_link_libraries(test_network PRIVATE OpenMP::OpenMP_CXX)

# Layers Tests

add_executable(test_layers
    tests/test_layers.cpp
    ${CNN_SOURCES}
)

target_include_directories(test_layers PRIVATE include)
target_link_libraries(test_layers PRIVATE OpenMP::OpenMP_CXX)