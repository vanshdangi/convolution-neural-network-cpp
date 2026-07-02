#include "math/tensor.h"
#include <algorithm>
#include <random>

// Constructors
Tensor::Tensor()
    : batch(0), rows(0), cols(0), depth(0) {}

Tensor::Tensor(int r, int c, int d)
    : batch(1), rows(r), cols(c), depth(d), data(r * c * d, 0.0f) {}

Tensor::Tensor(int r, int c, int d, float initValue)
    : batch(1), rows(r), cols(c), depth(d), data(r * c * d, initValue) {}

Tensor::Tensor(int b, int r, int c, int d)
    : batch(b), rows(r), cols(c), depth(d), data(b * r * c * d, 0.0f) {}

Tensor::Tensor(int b, int r, int c, int d, float initValue)
    : batch(b), rows(r), cols(c), depth(d), data(b * r * c * d, initValue) {}

// Element access
float& Tensor::operator()(int r, int c, int d){
    assert(batch == 1);
    return operator()(0, r, c, d);
}

const float& Tensor::operator()(int r, int c, int d) const{
    assert(batch == 1);
    return operator()(0, r, c, d);
}

float& Tensor::operator()(int b, int r, int c, int d){
    assert(b >= 0 && b < batch);
    assert(r >= 0 && r < rows);
    assert(c >= 0 && c < cols);
    assert(d >= 0 && d < depth);
    return data[((b * rows + r) * cols + c) * depth + d];
}

const float& Tensor::operator()(int b, int r, int c, int d) const{
    assert(b >= 0 && b < batch);
    assert(r >= 0 && r < rows);
    assert(c >= 0 && c < cols);
    assert(d >= 0 && d < depth);
    return data[((b * rows + r) * cols + c) * depth + d];
}

// Utilities
void Tensor::fill(float value){
    std::fill(data.begin(), data.end(), value);
}

void Tensor::randomize(float min, float max)
{
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(min, max);

    for(auto& v : data){
        v = dist(gen);
    }
}

void Tensor::resize(int r, int c, int d){
    batch = 1;
    rows = r;
    cols = c;
    depth = d;
    data.resize(r * c * d);
}

void Tensor::resize(int b, int r, int c, int d){
    batch = b;
    rows = r;
    cols = c;
    depth = d;
    data.resize(b * r * c * d);
}

Tensor Tensor::pad(int pad) const
{
    Tensor result(batch, rows + 2 * pad, cols + 2 * pad, depth);

    for (int b = 0; b < batch; b++)
    {
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                for (int d = 0; d < depth; d++)
                {
                    result(b, r + pad, c + pad, d) = (*this)(b, r, c, d);
                }
            }
        }
    }

    return result;
}

// Helpers
std::tuple<int,int,int,int> Tensor::shape() const {
    return {batch, rows, cols, depth};
}

int Tensor::size() const{
    return batch * rows * cols * depth;
}

float Tensor::max_value() const{
    assert(!data.empty());
    float max_v = data[0];
    for(float v : data){
        if(v > max_v){
            max_v = v;
        }
    }
    return max_v;
}

float Tensor::min_value() const{
    assert(!data.empty());
    float min_v = data[0];
    for(float v: data){
        if(v < min_v){
            min_v = v;
        }
    }
    return min_v;
}

int Tensor::getBatch() const{
    return batch;
}

int Tensor::getRows() const{
    return rows;
}

int Tensor::getCols() const{
    return cols;
}

int Tensor::getDepth() const{
    return depth;
}

float Tensor::sum() const{
    assert(!data.empty());
    float sum = 0;
    for(float v: data){
        sum += v;
    }
    return sum;
}

float Tensor::mean() const{
    return data.empty() ? 0.0f : sum() / size();
}

// Comparison
bool Tensor::operator==(const Tensor& other) const
{
    return batch == other.batch &&
            rows == other.rows &&
            cols == other.cols &&
            depth == other.depth &&
            data == other.data;
}

bool Tensor::operator!=(const Tensor& other) const
{
    return !(*this == other);
}

// Scalar Operations
Tensor Tensor::operator+(float scalar) const{
    Tensor result(batch, rows, cols, depth);

    for(int i = 0; i < size(); i++){
        result.data[i] = data[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const{
    Tensor result(batch, rows, cols, depth);
    
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i] - scalar;
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const{
    Tensor result(batch, rows, cols, depth);
    
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const{
    assert(scalar != 0);
    Tensor result(batch, rows, cols, depth);
    
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i] / scalar;
    }
    return result;
}

// Tensor-Tensor Operations
Tensor Tensor::operator+(const Tensor& other) const{
    assert(batch == other.batch);
    assert(rows == other.rows);
    assert(cols == other.cols);
    assert(depth == other.depth);

    Tensor result(batch, rows, cols, depth);
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const{
    assert(batch == other.batch);
    assert(rows == other.rows);
    assert(cols == other.cols);
    assert(depth == other.depth);

    Tensor result(batch, rows, cols, depth);
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// In-Place Operations
Tensor& Tensor::operator+=(const Tensor& other){
    assert(batch == other.batch);
    assert(rows == other.rows);
    assert(cols == other.cols);
    assert(depth == other.depth);

    for(int i = 0; i < size(); i++){
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){
    assert(batch == other.batch);
    assert(rows == other.rows);
    assert(cols == other.cols);
    assert(depth == other.depth);

    for(int i = 0; i < size(); i++){
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(float scalar){
    for(int i = 0; i < size(); i++){
        data[i] *= scalar;
    }
    return *this;
}

Tensor& Tensor::operator/=(float scalar){
    for(int i = 0; i < size(); i++){
        data[i] /= scalar;
    }
    return *this;
}
