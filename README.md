# Hand Tremor Detection using Temporal Deep Learning
## A Comparative Study of Frame-Based (MLP) and Temporal (LSTM) Neural Networks

---

## Overview

This project investigates whether temporal deep learning is necessary for automated hand tremor detection from wrist-worn IMU sensor data. Two models are compared:

| Model | Type | Input | Accuracy |
|-------|------|-------|----------|
| MLP | Frame-based (static) | Single sensor reading | 99.57% |
| LSTM | Temporal sequence | 50-frame window | 95.09% |

**Key finding:** The MLP outperforms on this dataset due to frame-level labeling, but the LSTM is the more appropriate architecture for real-world deployment where continuous unlabeled sensor streams must be classified.

---

## Repository Structure

```
cmpe401-tremor/
├── tremor_lstm.ipynb          # LSTM temporal sequence model
├── tremor_mlp.ipynb           # MLP frame-based model + comparison
├── README.md
```

> **Note:** Training outputs may not be stored in notebook cells due to long-running Colab sessions. All results are documented in this README and the technical report.

---

## Dataset

**MPU9250 Hand Tremor Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/aaryapandya/hand-tremor-dataset-collected-using-mpu9250-sensor)

| Property | Value |
|----------|-------|
| Total samples | 27,995 |
| Features used | 6 (aX, aY, aZ, gX, gY, gZ) |
| Excluded | Magnetometer (mX, mY, mZ) — constant -1 |
| Stable samples | 12,749 (45.5%) |
| Tremor samples | 15,246 (54.5%) |
| Label type | Frame-level (per sensor reading) |

> **Note on dataset choice:** MediaPipe landmark extraction was initially proposed. An established sensor-based dataset was used instead as it provides a more direct and noise-free measurement of tremor oscillations than visual landmark detection, and avoids issues with lighting, occlusion, and camera angle.

---

## Preprocessing

- **Features:** aX, aY, aZ, gX, gY, gZ (magnetometer excluded — constant values)
- **Normalization:** StandardScaler (zero mean, unit variance), fitted on training set only
- **Split:** 64% train / 16% val / 20% test (stratified)
- **LSTM windowing:** 50-timestep sliding windows, step size 25 (50% overlap) → 1,118 windows
- **MLP:** Individual samples (no windowing) → 17,916 training samples

---

## Model Architectures

### LSTM (Temporal Sequence Classifier)
```
Input: (50 timesteps, 6 features)
→ LSTM(64 units, return_sequences=True) → Dropout(0.3)
→ LSTM(32 units) → Dropout(0.3)
→ Dense(32, ReLU) → Dropout(0.3)
→ Dense(1, Sigmoid)
Total parameters: 31,681
```

### MLP (Frame-Based Classifier)
```
Input: (6 features)
→ Dense(128, ReLU) → Dropout(0.3)
→ Dense(64, ReLU) → Dropout(0.3)
→ Dense(32, ReLU) → Dropout(0.3)
→ Dense(1, Sigmoid)
Total parameters: ~12,000
```

### Training Configuration (both models)
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Max epochs: 50
- Early stopping: patience=10 on val_accuracy
- ReduceLROnPlateau: factor=0.5, patience=5

---

## Results

### LSTM — Test Set Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **95.09%** |
| Precision (Tremor) | 95.97% |
| Recall (Tremor) | 95.20% |
| F1-Score (Tremor) | 95.58% |
| Precision (Stable) | 94.00% |
| Recall (Stable) | 94.95% |
| Epochs (early stop) | 30 / 50 |
| Parameters | 31,681 |

### MLP — Test Set Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **99.57%** |
| Precision (Tremor) | 99.74% |
| Recall (Tremor) | 99.48% |
| F1-Score (Tremor) | 99.61% |
| Precision (Stable) | 99.36% |
| Recall (Stable) | 99.70% |
| Epochs (early stop) | 25 / 50 |
| Parameters | ~12,000 |

### Head-to-Head Comparison

| Model | Accuracy | Precision | Recall | F1 | Params |
|-------|----------|-----------|--------|----|--------|
| MLP (Frame-Based) | 99.57% | 99.74% | 99.48% | 99.61% | ~12K |
| LSTM (Temporal) | 95.09% | 95.97% | 95.20% | 95.58% | 31.7K |

---

## Key Findings

**Why MLP outperforms on this dataset:**
The dataset assigns labels at the frame level — each individual sensor reading is labeled based on the recording session (stable or tremor) in which it was collected. This means individual sensor readings are already discriminative between classes, making temporal context partially redundant. The MLP exploits this and converges to near-perfect accuracy within 25 epochs.

**Why LSTM is better for real deployment:**
In a real wearable system, only a continuous unlabeled sensor stream is available. A single accelerometer reading is ambiguous — high acceleration could be intentional movement or tremor oscillation. The LSTM's 50-frame window (~0.5 seconds) captures the sustained rhythmic oscillation pattern characteristic of tremor, which is the clinically relevant signal. In real deployment, the LSTM's temporal approach is essential and more generalizable.

---

## How to Run

1. Download `Dataset.csv` from the Kaggle link above
2. Open `tremor_lstm.ipynb` in Google Colab
3. Upload `Dataset.csv` when prompted
4. Runtime → Change runtime type → GPU
5. Run All cells (~5 minutes on T4, ~2 minutes on A100)

Repeat with `tremor_mlp.ipynb` for the MLP comparison.

---

## References

- MPU9250 Hand Tremor Dataset: https://www.kaggle.com/datasets/aaryapandya/hand-tremor-dataset-collected-using-mpu9250-sensor
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.
- Ordóñez & Roggen (2016) — Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition. Sensors, 16(1), 115.
- Elble & Koller (1990) — Tremor. Johns Hopkins University Press.
