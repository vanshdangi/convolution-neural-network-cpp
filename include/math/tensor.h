#pragma once
#include <vector>
#include <cassert>

class Tensor
{
public:
    int batch;
    int rows;
    int cols;
    int depth;
    std::vector<float> data;

    // Constructors
    Tensor();
    Tensor(int r, int c, int d);
    Tensor(int r, int c, int d, float initValue);
    Tensor(int b, int r, int c, int d);
    Tensor(int b, int r, int c, int d, float initValue);

    // Element access
    float& operator()(int r, int c, int d);
    const float& operator()(int r, int c, int d) const;
    float& operator()(int b, int r, int c, int d);
    const float& operator()(int b, int r, int c, int d) const;

    // Utilities
    void fill(float value);
    void randomize(float min = -0.1f, float max = 0.1f);
    void resize(int r, int c, int d);
    void resize(int b, int r, int c, int d);
    Tensor pad(int pad) const;

    // Helpers
    std::tuple<int,int,int,int> Tensor::shape() const;
    int size() const;
    float max_value() const;
    float min_value() const;

    int getBatch() const;
    int getRows() const;
    int getCols() const;
    int getDepth() const;

    float sum() const;
    float mean() const;

    // Comparison
    bool operator==(const Tensor& other) const;
    bool operator!=(const Tensor& other) const;

    // Scalar Operations
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // Tensor-Tensor Operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;

    // In-Place Operations
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(float scalar);
    Tensor& operator/=(float scalar);

    // Visualize tensor
    void print() const;
};
