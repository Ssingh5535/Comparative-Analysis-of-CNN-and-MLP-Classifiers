## Comparative Analysis of CNN and MLP Classifiers  
**Kuzushiji-MNIST Case Study**  
Stephen Singh  
University of Florida — Stephensingh953@gmail.com

### Table of Contents
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Environment & Dependencies](#environment--dependencies)  
- [Model Architectures](#model-architectures)  
  - [CNN](#cnn)  
  - [MLP](#mlp)  
- [Training & Hyperparameters](#training--hyperparameters)  
- [Results](#results)  
  - [Accuracy & Loss](#accuracy--loss)  
  - [Confusion Matrices](#confusion-matrices)  
  - [Hyperparameter Tuning](#hyperparameter-tuning)  
  - [Misclassifications](#misclassifications)  
- [Usage](#usage)  
- [Directory Structure](#directory-structure)  
- [License](#license)  

---

## Project Overview  
This repo compares a Convolutional Neural Network (CNN) against a Multi-Layer Perceptron (MLP) on the Kuzushiji-MNIST dataset (70 k 28×28 grayscale Japanese characters). We evaluate which architecture better captures spatial features and generalizes to unseen glyphs.

## Dataset  
- **Kuzushiji-MNIST** (60 k train / 10 k test)  
- 10 classes of cursive Japanese characters  
- Download via `tensorflow.keras.datasets.kuzushiji_mnist` or from [Kuzushiji-MNIST GitHub](https://github.com/rois-codh/kmnist)

## Environment & Dependencies  
- Python 3.8+  
- TensorFlow 2.x or Keras  
- NumPy, Matplotlib, scikit-learn (for PCA/t-SNE)  
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Model Architectures

### CNN  
| Layer       | Output Shape     | # Params  |
|-------------|------------------|-----------|
| Conv2D(32)  | (26,26,32)       | 320       |
| MaxPool2D   | (13,13,32)       | 0         |
| Conv2D(64)  | (11,11,64)       | 18,496    |
| MaxPool2D   | (5,5,64)         | 0         |
| Flatten     | (1600,)          | 0         |
| Dense(128)  | (128,)           | 204,928   |
| Dense(10)   | (10,)            | 1,290     |

### MLP  
| Layer      | Output Shape | # Params |
|------------|--------------|----------|
| Flatten    | (784,)       | 0        |
| Dense(128) | (128,)       | 100,480  |
| Dense(64)  | (64,)        | 8,256    |
| Dense(10)  | (10,)        | 650      |

## Training & Hyperparameters  
- Optimizer: Adam  
- Loss: Sparse Categorical Crossentropy  
- Batch size: 64  
- Epochs: up to 100 with EarlyStopping (patience=5)  
- Learning rate: 0.001  

## Results

### Accuracy & Loss  
- **MLP** test accuracy: **89.58%**  
- **CNN** test accuracy: **94.73%**  

### Confusion Matrices  
- **MLP Confusion Matrix** shows some confusions around similar glyphs.  
- **CNN Confusion Matrix** is tighter, reflecting better class separation.  

### Hyperparameter Tuning  
| LR    | Hidden Layers | Epochs | Test Acc. |
|-------|---------------|--------|-----------|
| 0.001 | (64, 32)      | 10     | 86%       |
| 0.01  | (128,64,32)   | 20     | 85%       |
| 0.1   | (128,128,64)  | 30     | 10%       |

### Misclassifications  
Common confusions include “2”→“3” and “5”→“6”.

## Usage  
1. Clone this repo  
2. Prepare a Python virtual env and install dependencies  
3. Run  
   ```bash
   python train_cnn.py   # trains and saves CNN
   python train_mlp.py   # trains and saves MLP
   ```  
4. Evaluate  
   ```bash
   python evaluate.py --model cnn
   python evaluate.py --model mlp
   ```  
5. Visualize results via the provided notebooks.

## Directory Structure  
```
├── data/
├── notebooks/
│   ├── Project 1 CNN.ipynb  
│   └── Project 1 MLP.ipynb  
├── models/
├── train_cnn.py  
├── train_mlp.py  
├── evaluate.py  
└── README.md
```

## License  
Released under the MIT License.
