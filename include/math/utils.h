#pragma once
#include "math/tensor.h"

inline int argmax(const Tensor& x, int batch_idx) {
    int idx = 0;
    float max_val = x(batch_idx, 0, 0, 0);

    for (int i = 1; i < x.rows; ++i) {
        if (x(batch_idx, i, 0, 0) > max_val) {
            max_val = x(batch_idx, i, 0, 0);
            idx = i;
        }
    }

    return idx;
}
