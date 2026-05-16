# Fashion Item Classification using Fashion-MNIST

**Badr University in Assiut — School of Artificial Intelligence & Data Management**  
**Neural Networks Course Project**

---

## Problem Description

Fashion-MNIST is a dataset of Zalando's article images. The goal of this project is to build a **Multilayer Perceptron (MLP)** neural network capable of classifying 28×28 grayscale fashion images into 10 clothing categories.

This project was designed to satisfy all academic requirements for the Neural Networks course, covering data preprocessing, model building, training with regularization, evaluation, experimentation, and visualization.

---

## Dataset Information

| Property | Details |
|----------|---------|
| Source | `torchvision.datasets.FashionMNIST` (auto-downloaded) |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28 × 28 pixels (grayscale) |
| MLP input size | 784 (flattened) |
| Num classes | 10 |

### Class Labels

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Model Architecture

The MLP is a configurable fully connected network:

```
Input (784)
    ↓
Linear(784 → H1) → BatchNorm → Activation → Dropout(0.3)
    ↓
Linear(H1 → H2) → BatchNorm → Activation → Dropout(0.3)
    ↓
Linear(H2 → 10)   ← Output logits (Softmax applied by CrossEntropyLoss)
```

**Design choices:**
- **Batch Normalization:** Stabilizes and accelerates training by normalizing intermediate activations
- **Dropout (p=0.3):** Prevents overfitting by randomly deactivating neurons during training
- **Early Stopping:** Restores best weights when validation loss stops improving (patience=5)
- **LR Scheduler:** `ReduceLROnPlateau` halves the learning rate if val loss stagnates for 3 epochs

---

## Experiments

### Experiment 1: ReLU Activation

| Parameter | Value |
|-----------|-------|
| Activation | ReLU |
| Hidden layers | [128, 64] |
| Learning rate | 0.001 |
| Optimizer | Adam |

### Experiment 2: Tanh Activation

| Parameter | Value |
|-----------|-------|
| Activation | Tanh |
| Hidden layers | [256, 128] |
| Learning rate | 0.0005 |
| Optimizer | Adam |

---

## Results Comparison Table

| Metric | Experiment 1 (ReLU) | Experiment 2 (Tanh) |
|--------|---------------------|---------------------|
| Activation | ReLU | Tanh |
| Hidden layers | [128, 64] | [256, 128] |
| Learning rate | 0.001 | 0.0005 |
| Test Accuracy | *run notebook* | *run notebook* |
| Test Loss | *run notebook* | *run notebook* |
| Training time | *run notebook* | *run notebook* |

> Run the notebook to populate results. They are also saved to `results/experiment_comparison.csv`.

---

## Regularization Techniques

| Technique | Purpose | Effect |
|-----------|---------|--------|
| **Batch Normalization** | Normalizes layer activations | Faster convergence, more stable training |
| **Dropout (0.3)** | Randomly deactivates neurons | Reduces overfitting, improves generalization |
| **Early Stopping** | Stops when val loss stagnates | Prevents over-training, restores best model |
| **LR Scheduler** | Reduces LR when plateau | Fine-tuned convergence in later epochs |

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Upload `fashion_mnist_classification.ipynb` to Google Colab
2. Runtime → Run All
3. Results are saved to the `results/` folder automatically

### Option 2: Local Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook fashion_mnist_classification.ipynb
```

### Option 3: Python Script

```bash
pip install -r requirements.txt
python fashion_mnist_train.py
```

---

## Generated Output Files

All results are saved to `results/`:

| File | Description |
|------|-------------|
| `sample_images.png` | Grid of sample Fashion-MNIST images |
| `class_distribution.png` | Bar chart of class frequencies |
| `training_curves_exp1.png` | Loss & accuracy curves — Experiment 1 |
| `training_curves_exp2.png` | Loss & accuracy curves — Experiment 2 |
| `confusion_matrix_exp1.png` | Confusion matrix — Experiment 1 |
| `confusion_matrix_exp2.png` | Confusion matrix — Experiment 2 |
| `sample_predictions_exp1.png` | 15 sample predictions — Experiment 1 |
| `sample_predictions_exp2.png` | 15 sample predictions — Experiment 2 |
| `experiment_comparison_plot.png` | Side-by-side validation metric comparison |
| `experiment_comparison.csv` | Tabular summary of both experiments |
| `model_exp1.pth` | Saved weights — Experiment 1 |
| `model_exp2.pth` | Saved weights — Experiment 2 |

---

## Project Structure

```
fashion_mnist_project/
├── fashion_mnist_classification.ipynb   ← Main Jupyter Notebook
├── fashion_mnist_train.py               ← Standalone Python script
├── README.md                            ← This file
├── requirements.txt                     ← Dependencies
├── results/                             ← Auto-generated outputs
│   ├── sample_images.png
│   ├── class_distribution.png
│   ├── training_curves_exp1.png
│   ├── training_curves_exp2.png
│   ├── confusion_matrix_exp1.png
│   ├── confusion_matrix_exp2.png
│   ├── sample_predictions_exp1.png
│   ├── sample_predictions_exp2.png
│   ├── experiment_comparison_plot.png
│   ├── experiment_comparison.csv
│   ├── model_exp1.pth
│   └── model_exp2.pth
└── data/                                ← Auto-downloaded by torchvision
    └── FashionMNIST/
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `torch` | Neural network construction and training |
| `torchvision` | Dataset loading and image transforms |
| `numpy` | Numerical computation |
| `pandas` | Tabular data (comparison table, CSV export) |
| `matplotlib` | Plotting training curves and visualizations |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Classification report and confusion matrix |

---

## Future Improvements

1. **Convolutional Neural Network (CNN):** Replace the MLP with a CNN to exploit spatial structure in images, likely achieving 92%+ accuracy
2. **Data Augmentation:** Apply random flips, crops, and rotations to reduce overfitting
3. **Learning Rate Warmup:** Gradually increase LR at the start of training for more stable convergence
4. **Ensemble Methods:** Average predictions from multiple models to improve robustness
5. **Hyperparameter Search:** Use Optuna or Ray Tune for automated hyperparameter optimization
6. **Transfer Learning:** Fine-tune a pretrained ResNet or EfficientNet on Fashion-MNIST

---

*Badr University in Assiut — School of Artificial Intelligence & Data Management*
