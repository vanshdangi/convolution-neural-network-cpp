# Residual Convolutional Neural Network in C++

A **Convolutional Neural Network (CNN) / Residual Neural Network (ResNet)** implemented entirely from scratch in **C++**, featuring a custom tensor class, manual forward and backward propagation, a custom optimization pipeline, and training on the **CIFAR-10** dataset.

This project was built to understand deep learning internals without relying on external machine learning libraries.

---

## Project Structure

```text id="u5yk5u"
include/
├── layers/
├── math/
├── loss/
├── network/
├── optimizer/
├── training/

src/
├── layers/
├── math/
├── loss/
├── network/
├── optimizer/
├── training/
├── main.cpp

tests/
data/
models/
```

---

## Features

* Conv2D implemented from scratch
* Batch Normalization (forward + backward)
* Residual Blocks (ResNet-style)
* Global Average Pooling
* SGD with Momentum
* Weight Decay regularization
* Cosine Annealing learning rate scheduling
* Label Smoothing
* Data augmentation:

  * Random Horizontal Flip
  * Cutout
* Checkpoint save/load system
* Unit and integration tests

---

## Implementation Details

* Custom Tensor class with batch-first memory layout
* Manual gradient propagation for every layer
* He initialization for convolutional and dense layers
* BatchNorm running statistics for inference
* Residual skip connections with projection shortcuts
* Binary checkpoint serialization

---

## Build Instructions

```bash id="yejndm"
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

---

## Training Configuration & Results

### Dataset

**CIFAR-10**

* 50,000 training images
* 10,000 test images
* Image size: **32×32 RGB**
* 10 classes

### Model Architecture

Custom CNN/ResNet hybrid implemented entirely from scratch:

```text id="i5m4pq"
Input (32×32×3)
↓
Conv2D (3 → 32, kernel=3, padding=1)
↓
BatchNorm + ReLU

Stage 1:
ResidualBlock (32 → 32)
MaxPool

Stage 2:
ResidualBlock (32 → 64)
MaxPool

Stage 3:
ResidualBlock (64 → 128)
MaxPool

Global Average Pooling
↓
Dense (128 → 256)
ReLU
Dense (256 → 10)
```

### Optimization

* **Optimizer:** SGD
* **Initial Learning Rate:** 0.02
* **Minimum Learning Rate:** 1e-5
* **Momentum:** 0.9
* **Weight Decay:** 5e-4
* **Scheduler:** Cosine Annealing
* **Batch Size:** 32
* **Epochs:** 150

### Regularization & Augmentation

* **Label Smoothing:** Enabled
* **Cutout Augmentation:** 6×6
* **Random Horizontal Flip:** Enabled
* **Dropout:** Tested and removed (reduced performance)

---

## Final Results

| Metric               | Value       |
| -------------------- | ----------- |
| Best Test Accuracy   | **92.93%**  |
| Best Test Loss       | **0.71532** |
| Final Train Accuracy | **98.53%**  |
| Final Train Loss     | **0.59758** |
| Best Epoch           | **146**     |

---

## Training Performance

* **Training Time per Epoch:** ~42–43 minutes
* **Total Training Time:** ~105 hours
* **Hardware:** Intel Core i7-12700F (20 threads)
* **Acceleration:** CPU-only (no CUDA)

---

## Accuracy Progression

| Epoch | Test Accuracy |
| ----- | ------------- |
| 1     | 58.54%        |
| 10    | 80.41%        |
| 20    | 84.52%        |
| 50    | 86.10%        |
| 100   | 89.79%        |
| 125   | 92.04%        |
| 146   | **92.93%**    |

---

## Experiment Log

| Experiment                                           | Result         |
| ---------------------------------------------------- | -------------- |
| Plain CNN baseline (2 Conv2D layers)                 | **81.7%**      |
| + BatchNorm + 3rd Conv2D + Momentum SGD              | **86.77%**     |
| + Dropout                                            | Underperformed |
| + Cutout augmentation + Conv padding                 | **88.7%**      |
| + Cosine Annealing scheduler                         | **89.7%**      |
| + Global Average Pooling (non-residual architecture) | Underperformed |
| + Residual Blocks (ResNet-style)                     | **92.93%**     |

---

## Tests

### Overfit Sanity Test

To validate forward/backward propagation and optimizer correctness, the model was trained on a **32-image CIFAR-10 subset** with augmentation disabled.

**Results:**

* **100% training accuracy** reached at **epoch 17**
* **100% test accuracy** reached at **epoch 32** (same subset)

This confirms correct gradient flow and parameter updates.

---

### Checkpoint Consistency Test

Validation process:

1. Forward pass on an input sample
2. Save model weights
3. Load weights into a new model instance
4. Forward pass on the same input
5. Compare outputs

**Result:**

* **Maximum output difference:** `0.0`

This confirms checkpoint serialization is lossless.

---

### Tensor Validation Test

Validated:

* Shape initialization
* Indexing correctness
* Copy semantics
* Resize functionality

**Result:** Passed

---

### Network Integration Test

Validated:

* Full forward pass
* Output shape correctness
* Loss computation
* Full backward propagation

**Result:** Passed

---

### Layer Unit Tests

Validated:

* Conv2D forward/backward
* ReLU correctness
* MaxPool downsampling
* Dense forward/backward
* BatchNorm shape preservation
* Global Average Pooling correctness

**Result:** Passed

---

## What I Learned

* Implementing backpropagation manually across deep CNN architectures
* Debugging exploding and vanishing gradients
* Effects of regularization techniques like dropout and label smoothing
* Tradeoffs between Flatten and Global Average Pooling
* Benefits of residual connections in deeper networks
* Learning rate scheduling effects on convergence

---

## License

Licensed under the terms in the `LICENSE` file.
