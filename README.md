<div align="center">

# 🧠 Brain Tumor Detection Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A Convolutional Neural Network (CNN) system for automated brain tumor detection from MRI images.**

*Final Year B.Sc. Project — Department of Computer Science, University of Benin (2022)*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset Structure](#-dataset-structure)
- [Getting Started](#-getting-started)
- [Training the Model](#-training-the-model)
- [Running Inference](#-running-inference)
- [Repository Structure](#-repository-structure)
- [Project Documentation](#-project-documentation)
- [Author](#-author)

---

## 🔍 Overview

Brain tumor detection is a critical task in medical diagnostics, often requiring expert radiologists to manually review MRI scans — a process that is both time-consuming and prone to human error. This project presents a **deep learning-based solution** that automates this detection process.

Using a custom **Convolutional Neural Network (CNN)** trained on labeled MRI images, the system classifies scans into two categories:

| Class | Label | Description |
|-------|-------|-------------|
| Healthy | `0` | No tumor detected in the MRI scan |
| Tumor | `1` | Tumor presence detected in the MRI scan |

> The trained model is saved as `BrainTumorModel.h5` and is ready for inference out of the box.

---

## ⚙️ How It Works

```
MRI Image Input
      │
      ▼
 Pre-processing
 (Resize to 64×64, Normalize pixel values)
      │
      ▼
 CNN Feature Extraction
 (Conv2D → ReLU → MaxPooling  ×3 blocks)
      │
      ▼
 Fully Connected Classifier
 (Flatten → Dense(64) → Dropout(0.5) → Dense(1))
      │
      ▼
 Binary Prediction
 (0 = No Tumor  |  1 = Tumor)
```

---

## 🏗️ Model Architecture

The model follows a sequential CNN design with three convolutional blocks for progressively deeper feature extraction, followed by a dense classification head.

| # | Layer | Type | Details |
|---|-------|------|---------|
| 1 | Conv Block 1 | `Conv2D` | 32 filters, 3×3 kernel, ReLU activation |
| 2 | | `MaxPooling2D` | 2×2 pool size |
| 3 | Conv Block 2 | `Conv2D` | 32 filters, 3×3 kernel, He Uniform init, ReLU |
| 4 | | `MaxPooling2D` | 2×2 pool size |
| 5 | Conv Block 3 | `Conv2D` | 64 filters, 3×3 kernel, He Uniform init, ReLU |
| 6 | | `MaxPooling2D` | 2×2 pool size |
| 7 | Classifier | `Flatten` | — |
| 8 | | `Dense` | 64 units, ReLU activation |
| 9 | | `Dropout` | Rate = 0.5 (regularization against overfitting) |
| 10 | Output | `Dense` | 1 unit, Softmax activation |

**Training Configuration:**

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Metric | Accuracy |
| Epochs | 10 |
| Batch Size | 16 |
| Train/Test Split | 80% / 20% |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **TensorFlow / Keras** | 2.x | Deep learning framework & model training |
| **OpenCV** (`cv2`) | — | Image reading and pixel-level processing |
| **Pillow** (PIL) | — | Image resizing and format conversion |
| **NumPy** | — | Numerical operations and array handling |
| **scikit-learn** | — | Train/test split and evaluation utilities |

---

## 📂 Dataset Structure

The training script expects the dataset to be organized in the following directory structure:

```
datasets/
├── yes/          # MRI scans WITH tumor present  (label = 1)
│   ├── y1.jpg
│   ├── y2.jpg
│   └── ...
└── no/           # MRI scans WITHOUT tumor        (label = 0)
    ├── no1.jpg
    ├── no2.jpg
    └── ...
```

> ⚠️ All images are automatically resized to **64×64 pixels** during pre-processing. Only `.jpg` files are currently supported by the training pipeline.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Neural-network.git
cd Neural-network
```

### 2. Install Dependencies

```bash
pip install tensorflow keras numpy opencv-python pillow scikit-learn
```

> **Recommended:** Use a virtual environment to keep dependencies isolated.
> ```bash
> python -m venv venv
> source venv/bin/activate      # On Linux/macOS
> venv\Scripts\activate         # On Windows
> ```

---

## 🎓 Training the Model

1. Ensure your dataset is organized as described in [Dataset Structure](#-dataset-structure).
2. Run the training script:

```bash
python mainTrain.py
```

The script will:
- Load and pre-process all `.jpg` images from `datasets/yes/` and `datasets/no/`
- Normalize pixel values and split data 80/20 into training and test sets
- Build and compile the CNN model
- Train for **10 epochs** with a **batch size of 16**
- Save the trained model to `BrainTumorcategoricalModel.h5`

You can adjust training parameters directly in `mainTrain.py`:

```python
model.fit(x_train, y_train,
    batch_size=16,   # ← adjust batch size
    epochs=10,       # ← adjust number of epochs
    ...
)
```

---

## 🔬 Running Inference

A pre-trained model (`BrainTumorModel.h5`) is included in this repository and ready to use.

1. Open `MainTest.py` and update the image path to point to your MRI scan:

```python
# Update this line with the path to your image
image = cv2.imread('path/to/your/mri_scan.jpg')
```

2. Run the inference script:

```bash
python MainTest.py
```

**Interpreting the output:**

| Output | Meaning |
|--------|---------|
| `[0]` | ✅ No Tumor Detected — scan appears healthy |
| `[1]` | ⚠️ Tumor Detected — further clinical review advised |

---

## 📁 Repository Structure

```
Neural-network-main/
│
├── 📄 README.md                                        ← You are here
├── 🤖 BrainTumorModel.h5                              ← Pre-trained CNN model (ready to use)
├── 🧪 MainTest.py                                      ← Run inference on a single MRI image
├── 🏋️ mainTrain.py                                    ← Train the CNN model from scratch
└── 📚 BRAIN TUMOR DETECTION USING DEEP LEARNING.pdf   ← Full academic project report
```

---

## 📝 Project Documentation

The complete academic report for this project is included in this repository:

📄 **[BRAIN TUMOR DETECTION USING DEEP LEARNING.pdf](./BRAIN%20TUMOR%20DETECTION%20USING%20DEEP%20LEARNING.docx.pdf)**

The report covers:
- Background & motivation
- Literature review of existing approaches
- System design & methodology
- Model training results & performance analysis
- Conclusions and future work recommendations

---

## 👤 Author

<table>
  <tr>
    <td><strong>Name</strong></td>
    <td>Isah Okhai Jeffery</td>
  </tr>
  <tr>
    <td><strong>Institution</strong></td>
    <td>University of Benin</td>
  </tr>
  <tr>
    <td><strong>Department</strong></td>
    <td>Computer Science</td>
  </tr>
  <tr>
    <td><strong>Year</strong></td>
    <td>Class of 2022</td>
  </tr>
</table>

---

<div align="center">

*Built with ❤️ as a Final Year Project in Computer Science*

</div>
