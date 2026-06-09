#include "math/tensor.h"
#include <random>
#include <iostream>
#include <iomanip>

// Constructors
Tensor::Tensor()
    : rows(0), cols(0), depth(0){};

Tensor::Tensor(int r, int c, int d)
    : rows(r), cols(c), depth(d), data(r*c*d, 0.0f){};

Tensor::Tensor(int r, int c, int d, float initValue)
    : rows(r), cols(c), depth(d), data(r*c*d, initValue){};

// Element access
float& Tensor::operator()(int r, int c, int d){
    assert(r >= 0 && r < rows);
    assert(c >= 0 && c < cols);
    assert(d >= 0 && d < depth);
    return data[r*cols*depth + c*depth + d];
}

const float& Tensor::operator()(int r, int c, int d) const{
    assert(r >= 0 && r < rows);
    assert(c >= 0 && c < cols);
    assert(d >= 0 && d < depth);
    return data[r*cols*depth + c*depth + d];
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
    rows = r;
    cols = c; 
    depth = d;
    data.resize(r*c*d);
}

Tensor Tensor::pad(int pad)
{
    Tensor result(
        rows + 2 * pad,
        cols + 2 * pad,
        depth
    );

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            for (int d = 0; d < depth; d++)
            {
                result(r + pad, c + pad, d) =
                    (*this)(r, c, d);
            }
        }
    }

    return result;
}

// Helpers
int Tensor::size() const{
    return rows*cols*depth;
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
    return rows == other.rows &&
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
    Tensor result(rows, cols, depth);

    for(int i = 0; i < size(); i++){
        result.data[i] = data[i]+scalar;
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const{
    Tensor result(rows, cols, depth);
    
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i]-scalar;
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const{
    Tensor result(rows, cols, depth);
    
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i]*scalar;
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const{
    assert(scalar != 0);
    Tensor result(rows, cols, depth);
    
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i]/scalar;
    }
    return result;
}

// Tensor-Tensor Operations
Tensor Tensor::operator+(const Tensor& other) const{
    assert(rows == other.rows);
    assert(cols == other.cols);
    assert(depth == other.depth);

    Tensor result(rows, cols, depth);
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const{
    assert(rows == other.rows);
    assert(cols == other.cols);
    assert(depth == other.depth);

    Tensor result(rows, cols, depth);
    for(int i = 0; i < size(); i++){
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// In-Place Operations
Tensor& Tensor::operator+=(const Tensor& other){
    assert(rows == other.rows);
    assert(cols == other.cols);
    assert(depth == other.depth);

    for(int i = 0; i < size(); i++){
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other){
    assert(rows == other.rows);
    assert(cols == other.cols);
    assert(depth == other.depth);

    for(int i = 0; i < size(); i++){
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(float scalar){
    Tensor result(rows, cols, depth);
    for(int i = 0; i < size(); i++){
        data[i] *= scalar;
    }
    return *this;
}

Tensor& Tensor::operator/=(float scalar){
    Tensor result(rows, cols, depth);
    for(int i = 0; i < size(); i++){
        data[i] /= scalar;
    }
    return *this;
}


void Tensor::print() const
{
    for(int d = 0; d < depth; d++)
    {
        std::cout << "Depth " << d << ":\n";

        for(int r = 0; r < rows; r++)
        {
            for(int c = 0; c < cols; c++)
            {
                std::cout << std::setw(8) << (*this)(r, c, d) << " ";
            }

            std::cout << '\n';
        }

        std::cout << '\n';
    }
}