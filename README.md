# Sleep Stage Classification: ML vs Deep Learning

Comparing KNN, SVM, Random Forest, and LSTM on the same dataset and task to identify where traditional ML outperforms deep learning and vice versa.

---

## 1. Background and Motivation

Sleep stage classification assigns one of five labels (Wake, N1, N2, N3, REM) to each 30-second EEG epoch. Accurate classification across all five stages, especially REM and N3, is essential for meaningful sleep quality assessment.

This repository documents a controlled comparison experiment. Three classical ML models (KNN, SVM, Random Forest) were first compared under identical conditions, with RF selected as the best-performing baseline. LSTM was then introduced to test whether modeling the temporal structure of EEG signals improves classification. The ML comparison revealed consistent weakness in REM (RF recall 0.31) and N3, LSTM was introduced to test whether sequential modeling of EEG improves classification for these stages.

The core question is not which model wins overall, but which model is stronger for which sleep stage and why.

---

## 2. Dataset

**Sleep-EDF Cassette** (PhysioNet, open access)

- Subjects: 5
- Channel: EEG Fpz-Cz (single channel, 100 Hz)
- Epochs: 668 total (30 seconds per epoch)
- Labels: Wake / N1 / N2 / N3 / REM (AASM criteria)

| Stage | Samples |
|-------|---------|
| Wake  | 80      |
| N1    | 152     |
| N2    | 225     |
| N3    | 148     |
| REM   | 63      |

All models use identical data splits (random_state=42, 80/20 train-test, stratified).

---

## 3. Models and Experiments

### ML Models (KNN, SVM, Random Forest)

All three models share the same feature set: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz), theta/beta ratio, and delta/theta ratio extracted via FFT from each 30-second epoch. Input shape is (668, 6).

Class imbalance was handled with class_weight='balanced'. Undersampling was tested but discarded as data loss outweighed the balancing benefit. All models were tuned with grid search and 5-fold stratified cross-validation.

| Stage | KNN | SVM | RF |
|-------|-----|-----|----|
| Wake  | 61.5% | 69.2% | 76.9% |
| N1    | 40.0% | 40.0% | 26.7% |
| N2    | 33.3% | 50.0% | 50.0% |
| N3    | 66.7% | 77.8% | 66.7% |
| REM   | 57.1% | 85.7% | 78.6% |

RF achieved the highest overall accuracy (67.91%) and was selected as the ML baseline. N1 was consistently the weakest stage across all three models, and REM recall remained below 0.85 even for the best ML model (SVM), indicating a structural ceiling for FFT-based ML on this task.

### LSTM

LSTM receives the same 6 FFT features but as a sequential input of shape (668, 30, 6), splitting each 30-second epoch into 30 one-second timesteps. Architecture: hidden_size=64, num_layers=2, dropout=0.3, batch_size=32, Adam optimizer. Class imbalance handled with CrossEntropyLoss inverse-frequency weights.

Improvements were applied sequentially to isolate the effect of each.

| Step | Configuration | Accuracy |
|------|--------------|----------|
| Baseline | LSTM, epoch=50, lr=0.001 fixed | 64.18% |
| Step 1 | + LR Scheduler (StepLR, step=20, gamma=0.5) | 70.90% |
| Step 2 | + Early Stopping (patience=15) | 70.15% |
| **Step 3** | **+ Early Stopping (patience=20) + torch.save** | **71.64%** |

All results are based on random_state=42.

---

## 4. Results

### Overall Accuracy

| Model | Accuracy |
|-------|----------|
| KNN   | 62.69%   |
| SVM   | 64.93%   |
| RF    | 67.91%   |
| **LSTM** | **71.64%** |

### RF Classification Report

| Stage | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Wake  | 0.56 | 0.63 | 0.59 |
| N1    | 0.74 | 0.69 | 0.71 |
| **N2** | **0.81** | **0.83** | **0.82** |
| N3    | 0.65 | 0.85 | 0.73 |
| REM   | 0.50 | 0.31 | 0.38 |
| Macro avg | 0.65 | 0.66 | 0.65 |

### LSTM Classification Report

| Stage | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Wake  | 0.64 | 0.56 | 0.60 |
| N1    | 0.71 | 0.73 | 0.72 |
| **N2** | **0.76** | **0.71** | **0.74** |
| **N3** | **0.81** | **0.87** | **0.84** |
| REM   | 0.47 | 0.54 | 0.50 |
| Macro avg | 0.68 | 0.68 | 0.68 |

### Stage-level Analysis

**REM** (RF F1 0.38 vs LSTM F1 0.50, +0.12): RF recall of 0.31 means 69% of actual REM epochs were missed. LSTM captured REM patterns that FFT summary features missed.

**N3** (RF F1 0.73 vs LSTM F1 0.84, +0.11): Delta wave activity in N3 is sustained over time, which LSTM can track across timesteps. LSTM recall reached 0.87.

**N2** (RF F1 0.82 vs LSTM F1 0.74, -0.08): N2 has the most samples (225) and stable spectral patterns. RF benefits from this distribution and classifies N2 reliably. LSTM shows more confusion with adjacent stages. N2 does not require temporal modeling to classify reliably, so LSTM's sequential structure adds complexity without benefit here.

**N1 and Wake**: Comparable performance across both models. N1 remained the weakest stage for all models in this experiment.

Overall accuracy favors LSTM (71.64% vs 67.91%). On a per-stage basis, LSTM is stronger for REM and N3, and RF is stronger for N2.

Neither model is strictly better than the other. Overall accuracy alone does not capture which stages each model handles well. The appropriate choice depends on the target use case: if REM and N3 detection matters most (sleep quality assessment, clinical screening), LSTM is the better fit. If stable high-sample stages like N2 are the priority, RF is a more practical choice when the target stages have stable spectral patterns and sufficient samples.

---

## 5. Limitations and Next Step

The main structural limitation is that LSTM received pre-computed FFT features rather than raw waveforms. LSTM is designed to learn temporal patterns directly from sequential input. Feeding summarized features reduces its advantage over RF since both models ultimately receive the same 6 numbers per segment. Combined with a dataset of only 668 epochs, deep learning could not fully demonstrate its potential here.

A follow-up experiment applies EEGNet (a depthwise separable CNN designed for EEG) to a larger dataset (DEAP, 32 subjects) with raw signal input to test whether raw signal input and more data improve performance.

---

## 6. Setup

```bash
pip install mne scikit-learn torch numpy scipy matplotlib
```

Data: Download Sleep-EDF Cassette from PhysioNet (open access, no registration required).
https://physionet.org/content/sleep-edfx/1.0.0/

---

## Repository Structure

```
sleep-stage-ml-vs-dl/
├── sleep_ml.ipynb      # KNN / SVM / RF comparison
├── sleep_lstm.ipynb    # LSTM training and evaluation
└── README.md
```

---

## Citation

Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation. 101(23), 2000.

Kemp B, et al. Analysis of a sleep-dependent neuronal feedback loop. IEEE-BME 47(9):1185-1194, 2000.
