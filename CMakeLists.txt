cmake_minimum_required(VERSION 3.16)

project(deep_learning_framework_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

add_executable(deep_learning_framework_cpp
    ${SOURCES}
)

target_include_directories(deep_learning_framework_cpp PRIVATE
    include
)