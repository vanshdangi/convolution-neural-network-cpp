#pragma once
#include "math/tensor.h"

inline int argmax(const Tensor& x) {
    int idx = 0;
    float max_val = x(0, 0, 0);

    for (int i = 1; i < x.rows; ++i) {
        if (x(i, 0, 0) > max_val) {
            max_val = x(i, 0, 0);
            idx = i;
        }
    }
    return idx;
}
