# CIFAR-10 Convolutional Neural Network

A from-scratch CNN classifier trained on the CIFAR-10 benchmark dataset, covering the full ML pipeline: data loading, preprocessing, model design, training with callbacks, evaluation, and artifact export.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Model Architecture](#model-architecture)
4. [Data Augmentation](#data-augmentation)
5. [Training](#training)
6. [Evaluation & Outputs](#evaluation--outputs)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [How to Run](#how-to-run)
9. [Requirements](#requirements)

---

## Dataset

**CIFAR-10** consists of 60,000 colour images (32 × 32 pixels, RGB) split across 10 mutually exclusive classes:

| Label | Class      |
|-------|------------|
| 0     | airplane   |
| 1     | automobile |
| 2     | bird       |
| 3     | cat        |
| 4     | deer       |
| 5     | dog        |
| 6     | frog       |
| 7     | horse      |
| 8     | ship       |
| 9     | truck      |

- **Training set**: 50,000 images
- **Test set**: 10,000 images

**Preprocessing applied:**
- Pixel values normalised from `[0, 255]` integers to `[0.0, 1.0]` floats — required for stable gradient flow.
- Labels one-hot encoded into 10-element vectors (e.g. class `3` → `[0,0,0,1,0,0,0,0,0,0]`).

---

## Project Structure

```
convolutional-neural-network/
├── 06/
│   └── cifar10_cnn.py        # Full training pipeline
├── .github/
│   └── workflows/
│       └── ci.yml            # Two-step CI/CD (lint → smoke test)
├── requirements.txt
├── task-description.md
│
│   ── generated outputs ──
├── cifar10_model.keras        # Saved model
├── model_plot.png             # Architecture diagram
├── history_plot.png           # Accuracy/loss curves
├── confusion_matrix.png       # Per-class confusion heatmap
└── misclassified_*.png        # Sample misclassified images
```

---

## Model Architecture

The network is a **three-block CNN** with a fully-connected classifier head. Because the total layer count exceeds 10, `kernel_initializer='he_uniform'` is used throughout — this keeps ReLU activations well-scaled at initialisation and prevents the vanishing gradient problem that default random initialisation causes in deeper networks.

```
Input (32, 32, 3)
    │
    ├── [Augmentation]
    │       RandomFlip (horizontal)
    │       RandomTranslation (±10 % height & width)
    │
    ├── [Block 1 — low-level features: edges, colours]
    │       Conv2D  32 filters, 3×3, same, ReLU  → BatchNorm
    │       Conv2D  32 filters, 3×3, same, ReLU  → BatchNorm
    │       MaxPooling2D 2×2    (32×32 → 16×16)
    │       Dropout 0.2
    │
    ├── [Block 2 — mid-level features: textures, shapes]
    │       Conv2D  64 filters, 3×3, same, ReLU  → BatchNorm
    │       Conv2D  64 filters, 3×3, same, ReLU  → BatchNorm
    │       MaxPooling2D 2×2    (16×16 → 8×8)
    │       Dropout 0.3
    │
    ├── [Block 3 — high-level features: object parts]
    │       Conv2D 128 filters, 3×3, same, ReLU  → BatchNorm
    │       Conv2D 128 filters, 3×3, same, ReLU  → BatchNorm
    │       MaxPooling2D 2×2    (8×8 → 4×4)
    │       Dropout 0.4
    │
    ├── [Classifier head]
    │       Flatten              (4×4×128 = 2,048 units)
    │       Dense 512, ReLU      → BatchNorm
    │       Dropout 0.5
    │       Dense 10, Softmax    (one logit per class)
    │
Output: class probabilities (10,)
```

**Why BatchNormalization after every Conv2D?**  
BN re-centres and re-scales the layer's output at every mini-batch. This speeds up convergence (allows higher learning rates), reduces sensitivity to initialisation, and acts as a mild regulariser by adding noise during training.

**Why increasing Dropout rates (0.2 → 0.3 → 0.4 → 0.5)?**  
Overfitting is most severe at the deeper, more abstract layers where the network has had many transformations to "memorise" the training set. Scaling dropout up with depth applies stronger regularisation exactly where it is needed most.

**Optimizer:** Adam with initial learning rate `1e-3` and default momentum parameters (`β₁ = 0.9`, `β₂ = 0.999`).  
**Loss:** Categorical cross-entropy (standard for multi-class one-hot targets).

---

## Data Augmentation

Two preprocessing layers are embedded **inside** the model graph:

| Layer | Effect |
|-------|--------|
| `RandomFlip('horizontal')` | Randomly mirrors each image left–right with 50 % probability |
| `RandomTranslation(0.1, 0.1)` | Randomly shifts the image up to ±10 % in both axes |

Because these are Keras layers, they are **automatically disabled** during `model.predict()` and `model.evaluate()` — the test set is always evaluated on un-augmented images. No external data-augmentation library or generator is needed.

**Why does augmentation help?**  
The training set has 50,000 images. Augmentation effectively multiplies the variety of examples the model sees without collecting more data, forcing it to learn features that are invariant to horizontal flips (cars look like cars from either direction) and small translations (the object's location in the frame should not change the prediction).

---

## Training

### Hyperparameters

| Parameter    | Value |
|--------------|-------|
| Epochs       | 40 (default) |
| Batch size   | 64 |
| Optimizer    | Adam lr=1e-3 |
| Loss         | Categorical cross-entropy |

### Callback — ReduceLROnPlateau

```
monitor  : val_loss
factor   : 0.5      (new_lr = lr × 0.5)
patience : 4 epochs
min_lr   : 1e-6
```

When validation loss has not improved for 4 consecutive epochs, the learning rate is halved. This allows the optimiser to make large updates early in training (fast convergence) and then take smaller, more precise steps later (fine-tuning). Without this, the model typically stalls 3–5 epochs before the end of training.

### Training loop summary

```
load_and_preprocess_data()   # normalise + one-hot encode
build_model()                # construct the Keras functional graph
visualize_architecture()     # model.summary() + model_plot.png
build_callbacks()            # ReduceLROnPlateau
model.fit(...)               # 40 epochs, val on test set, callbacks active
evaluate_model()             # final loss + accuracy printed
save_model()                 # cifar10_model.keras
plot_training_history()      # history_plot.png
plot_confusion_matrix()      # confusion_matrix.png
show_misclassified()         # misclassified_1-5.png
```

---

## Evaluation & Outputs

After training completes, the following files are written to the project root:

| File | Description |
|------|-------------|
| `cifar10_model.keras` | Full saved model (architecture + weights + optimiser state). Reload with `keras.models.load_model()`. |
| `model_plot.png` | Visual diagram of every layer, its type, output shape, and parameter count. Generated by `plot_model` (graphviz) or a matplotlib table fallback if graphviz is not installed. |
| `history_plot.png` | Four curves plotted together: training accuracy, validation accuracy, training loss, validation loss — one point per epoch. Useful for diagnosing overfitting (gap between train and val curves) or underfitting (both curves still descending at epoch 40). |
| `confusion_matrix.png` | 10×10 heatmap. Cell (i, j) counts how many images of true class i were predicted as class j. Diagonal = correct predictions. Off-diagonal cells reveal which class pairs the model confuses most (e.g. cats vs. dogs). |
| `misclassified_1-5.png` | Five images where the model was wrong, annotated with the true label and the model's predicted label. Good for quick qualitative inspection. |

**Expected test accuracy:** ~83–86 % after 40 epochs with this architecture.

---

## CI/CD Pipeline

The GitHub Actions workflow at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request to `main`. It is two sequential jobs — the second only runs if the first passes.

### Job 1 — Lint

```yaml
flake8 06/cifar10_cnn.py --max-line-length=100
```

Checks code style and catches obvious errors (undefined names, unused imports, syntax mistakes) before wasting compute on training. The 100-character line limit is a deliberate relaxation of flake8's default 79 to accommodate readable layer definitions.

### Job 2 — Smoke Test (`needs: lint`)

Runs the full pipeline end-to-end on a tiny slice of data:

```bash
python 06/cifar10_cnn.py --ci
```

The `--ci` flag activates smoke-test mode:
- **Training set**: 500 images (instead of 50,000)
- **Test set**: 100 images (instead of 10,000)
- **Epochs**: 1 (instead of 40)
- **Callbacks**: disabled

This verifies that every function in the pipeline (load → build → train → evaluate → save → all four plots) runs without crashing, in seconds rather than minutes. It does **not** verify accuracy — just that the code is executable end-to-end.

The graphviz system binary is installed in CI via:
```yaml
- name: Install system dependencies (graphviz for plot_model)
  run: sudo apt-get install -y graphviz
```

All generated artifacts (`*.keras`, `*.png`) are uploaded via `actions/upload-artifact` so they can be inspected in the GitHub Actions UI after each run.

---

## How to Run

### Full training run (local)

```bash
cd convolutional-neural-network
pip install -r requirements.txt
python 06/cifar10_cnn.py
```

This trains for 40 epochs and writes all output files to the project root. Expect ~30–40 minutes on CPU.

### Custom epochs / batch size

```bash
python 06/cifar10_cnn.py --epochs 60 --batch-size 128
```

### CI smoke test (fast, ~30 seconds)

```bash
python 06/cifar10_cnn.py --ci
```

---

## Requirements

```
tensorflow
keras
numpy
matplotlib
pandas
scikit-learn
flake8
pydot
graphviz   # Python bindings — system binary also needed for plot_model
```

Install everything with:

```bash
pip install -r requirements.txt
```

> **Note on GPU:** TensorFlow ≥ 2.11 does not support GPU acceleration on native Windows. To use a GPU, run inside WSL2 or use the `tensorflow-directml` plugin. The model trains correctly on CPU; 40 epochs takes approximately 30–40 minutes.
