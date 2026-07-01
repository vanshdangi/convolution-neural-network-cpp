# convolution-neural-network-cpp

A Convolutional Neural Network (CNN) implemented from scratch in C++ featuring a custom tensor class, manual forward and backward propagation, and training on the CIFAR-10 dataset.

# Training Configuration & Results

## Configuration

### Dataset

**CIFAR-10**

* 50,000 training images
* 10,000 test images
* Image size: **32×32 RGB**
* 10 classes

---

### Model Architecture

Custom CNN/ResNet hybrid implemented **entirely from scratch in C++**

```text
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

---

### Optimization

- **Optimizer:** SGD
- **Initial Learning Rate:** 0.02
- **Minimum Learning Rate:** 1e-5
- **Momentum:** 0.9
- **Weight Decay:** 5e-4
- **Scheduler:** Cosine Annealing
- **Batch Size:** 32
- **Epochs:** 150

---

### Regularization & Augmentation

- **Label Smoothing:** Enabled
- **Cutout Augmentation:** 6×6
- **Random Horizontal Flip:** Enabled
- **Dropout:** Disabled (tested, reduced performance)

---

## Final Results

- **Best Test Accuracy:** **92.93%** (Epoch 146)
- **Best Test Loss:** **0.71532**
- **Final Train Accuracy:** **98.53%**
- **Final Train Loss:** **0.59758**

---

## Training Performance

- **Training Time per Epoch:** ~42–43 minutes
- **Total Training Time:** ~105 hours
- **Hardware:** Intel Core i7-12700F (20 threads)
- **Acceleration:** CPU-only (no CUDA)

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

## Experiments

* **Plain CNN baseline (2 Conv2D layers)** → **81.7%**
* Added **BatchNorm + 3rd Conv2D + Momentum SGD** → **86.77%**
* Added **Dropout** → underperformed and removed
* Added **Cutout augmentation + Conv padding** → **88.7%**
* Added **Cosine Annealing scheduler** → **89.7%**
* Added **Global Average Pooling (non-residual architecture)** → underperformed
* Added **Residual Blocks (ResNet-style)** → **92.93%** (best)
