# 🌸 Project 2 — Data Classification Using AI

---

## 📋 Table of Contents

1. [Project Idea](#project-idea)
2. [What Is Required](#what-is-required)
3. [What Is Implemented](#what-is-implemented)
4. [Core Concepts](#core-concepts)
5. [The Full Pipeline](#the-full-pipeline)
6. [Project Structure](#project-structure)
7. [Setup & Run](#setup--run)
8. [How to Test](#how-to-test)
9. [Output Files](#output-files)
10. [Experiments to Try](#experiments-to-try)
11. [Real-World Applications](#real-world-applications)

---

## Project Idea

Classification is one of the most fundamental tasks in machine learning: given a set of measurements about something, determine which category it belongs to. This project implements a complete, production-style **supervised learning classification pipeline** from raw data all the way to a fully evaluated and visualised model.

The dataset used is the **Iris dataset** — 150 flower samples described by 4 measurements (sepal length, sepal width, petal length, petal width), each belonging to one of 3 species: Setosa, Versicolor, or Virginica. The algorithm used is **K-Nearest Neighbours (KNN)** — one of the most intuitive yet powerful classifiers in machine learning.

The key shift from rule-based logic: instead of writing rules manually, you **provide labelled historical data and let the algorithm derive the decision boundaries itself**.

---

## What Is Required

| Requirement | Description |
|------------|-------------|
| **Load a dataset** | Read and understand a real structured dataset |
| **Explore the data** | Summarise shape, class distribution, and statistics |
| **Preprocess** | Apply feature scaling to remove bias |
| **Split data** | Divide into training and testing sets correctly |
| **Train a model** | Apply a classification algorithm |
| **Evaluate** | Measure performance beyond just accuracy |
| **Visualise** | Produce charts that explain the model's behaviour |

---

## What Is Implemented

### Complete 8-Step Pipeline

**Step 1 — Exploratory Data Analysis (EDA)**
Loads the Iris dataset, prints shape, class distribution, and full descriptive statistics (mean, std, min, quartiles, max) for all 4 features.

**Step 2 — Feature Scaling (StandardScaler)**
Applies `StandardScaler` to transform all features to mean=0, std=1. Fit only on training data to prevent data leakage.

**Step 3 — Train/Test Split**
80% training / 20% test, with `shuffle=True` to remove ordering bias and `random_state=42` for reproducibility.

**Step 4 — Model Training (KNN, K=5)**
Trains `KNeighborsClassifier` on the scaled training set using scikit-learn's standard fit/predict API.

**Step 5 & 6 — Prediction + Full Evaluation**
Produces accuracy score, weighted F1 score, full classification report (precision, recall, F1 per class), and confusion matrix.

**Step 7 — Hyperparameter Tuning (Elbow Method)**
Tests K values from 1 to 30, plots error rate vs K, identifies the optimal K at the "elbow" of the curve. Retrains if optimal K differs from the default.

**Step 8 — Visualisations**
Three output charts: elbow curve, confusion matrix heatmap, and 2D decision boundary using petal features.

### Why F1 Score, Not Just Accuracy

```
Accuracy alone is the "Accuracy Mirage":
  → On imbalanced data, a model predicting only the majority class
    can score 99% accuracy while being completely useless.

F1 Score = Harmonic mean of Precision and Recall
  → Catches models that are gaming accuracy via class imbalance
  → The professional standard for classification evaluation
```

### Why Feature Scaling Matters for KNN

```
Without scaling:
  Feature A: 0–1000  dominates distance calculations
  Feature B: 0–1     almost ignored

After StandardScaler:
  Both features: mean=0, std=1  →  equal contribution to distance
```

---

## Core Concepts

### K-Nearest Neighbours

The Proximity Principle: *similar things exist in close proximity.*

```
To classify a new data point:
  1. Calculate distance to every training point
  2. Find the K nearest neighbours
  3. Majority vote → assign that class

K=1  →  Overfitting  (memorises noise)
K=100 → Underfitting (too generic)
Optimal K → minimum validation error (the Elbow)
```

### The Confusion Matrix

```
                  Predicted Positive    Predicted Negative
Actual Positive      TP  ✅                 FN  ⚠️ (missed)
Actual Negative      FP  🔔 (false alarm)   TN  ✅
```

Reading the confusion matrix tells you *how* a model fails — not just *whether* it fails.

### Precision vs Recall Trade-off

| Metric | Measures | Key Use Case |
|--------|----------|-------------|
| **Precision** | Of all predicted positives, how many are actually positive? | Spam filter — avoid false alarms |
| **Recall** | Of all actual positives, how many were caught? | Medical diagnosis — avoid missed detections |
| **F1** | Harmonic mean of both | General-purpose professional metric |

---

## The Full Pipeline

```
Raw Dataset (150 × 4)
      ↓
 [Step 1] EDA — shape, distribution, statistics
      ↓
 [Step 2] StandardScaler — mean=0, std=1 per feature
      ↓
 [Step 3] Train/Test Split — 80% / 20%, shuffled
      ↓
 [Step 4] KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
      ↓
 [Step 5] model.predict(X_test)
      ↓
 [Step 6] Accuracy, F1, Classification Report, Confusion Matrix
      ↓
 [Step 7] Elbow Method — test K=1 to 30, find optimal K
      ↓
 [Step 8] Save: elbow_curve.png, confusion_matrix.png, decision_boundary.png
```

---

## Project Structure

```
Project_2_Data_Classification/
│
├── classifier.py           ← Full ML pipeline — run this
├── test_classifier.py      ← Unit test suite (5 tests)
├── elbow_curve.png         ← Generated: optimal K visualisation
├── confusion_matrix.png    ← Generated: TP/TN/FP/FN heatmap
├── decision_boundary.png   ← Generated: 2D class regions
└── README.md               ← This file
```

---

## Setup & Run

```bash
pip install numpy pandas scikit-learn matplotlib
python classifier.py
```

**Results achieved:**
```
Accuracy  : 100.00%
F1 Score  : 1.0000
Optimal K : 2  (found via Elbow Method)
```

---

## How to Test

```bash
python test_classifier.py
```

Tests cover: data loading shape/names, StandardScaler mean/std validation, 80/20 split size, full pipeline accuracy ≥ 90%, and multiple K values producing valid scores.

```
Result: 5/5 tests passed
```

---

## Output Files

| File | What It Shows |
|------|--------------|
| `elbow_curve.png` | Error rate vs K — reveals where adding more neighbours stops helping |
| `confusion_matrix.png` | Per-class TP/TN/FP/FN breakdown — the diagnostic tool |
| `decision_boundary.png` | 2D view of learned class regions using petal length & petal width |

---

## Experiments to Try

1. **Remove scaling** — Does accuracy drop without StandardScaler? By how much?
2. **Try different K values** — What happens at K=1? K=50? K=149?
3. **Swap the algorithm** — Replace KNN with `DecisionTreeClassifier` or `RandomForestClassifier`
4. **Different dataset** — Try `load_wine()` or `load_breast_cancer()` from sklearn
5. **Cross-validation** — Use `cross_val_score` for a more robust accuracy estimate
6. **Feature selection** — Which 2 of the 4 features are most discriminative on their own?

---

## Real-World Applications

| Domain | Classification Problem |
|--------|----------------------|
| **Healthcare** | Tumour benign vs malignant, disease diagnosis |
| **Finance** | Loan default risk, transaction fraud detection |
| **Email** | Spam vs legitimate messages |
| **Manufacturing** | Defective vs passing quality control |
| **Agriculture** | Plant species and disease identification |
| **Autonomous Vehicles** | Road sign and obstacle recognition |

---

> Supervised learning classification is the engine behind most AI-powered decision systems in production today. Mastering the full pipeline — from raw data to evaluated model — is the bedrock skill of every professional ML engineer.
