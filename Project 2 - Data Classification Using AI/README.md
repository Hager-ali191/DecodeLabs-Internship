# 🌸 Project 2 — Data Classification Using AI

> **DecodeLabs Industrial Training Kit · Batch 2026**  
> **Track:** Predictive Phase — Supervised Learning

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Core Concepts](#core-concepts)
4. [The Pipeline — 8 Steps](#the-pipeline--8-steps)
5. [Project Structure](#project-structure)
6. [Setup & Installation](#setup--installation)
7. [How to Run](#how-to-run)
8. [How to Test](#how-to-test)
9. [Output Files](#output-files)
10. [Understanding the Results](#understanding-the-results)
11. [Experiments to Try](#experiments-to-try)
12. [Real-World Applications](#real-world-applications)
13. [What's Next (Project 3)](#whats-next-project-3)

---

## Overview

Project 2 is the **predictive phase** of the DecodeLabs AI internship. This is where you cross the boundary from deterministic rules into **supervised machine learning** — you stop writing the rules and start letting the data derive the logic.

You will build a complete **classification pipeline** using the legendary **Iris dataset** and the **K-Nearest Neighbours (KNN)** algorithm. The full journey from raw data to evaluated model is implemented and explained in detail.

> **"We do not write the rules. We provide history, and the machine derives the logic."**

---

## Learning Objectives

By completing this project, you will:

| Skill | Concept |
|-------|---------|
| **EDA** | Load, inspect, and summarise a real dataset |
| **Preprocessing** | Apply StandardScaler to remove feature bias |
| **Data Splitting** | Correctly partition data into train / test sets |
| **ML Algorithm** | Implement and understand K-Nearest Neighbours |
| **Evaluation** | Interpret Accuracy, Precision, Recall, F1, Confusion Matrix |
| **Tuning** | Find optimal K using the Elbow Method |
| **Visualisation** | Plot decision boundaries and confusion matrices |

---

## Core Concepts

### 1. The Logic Skeleton Shift

```
OLD WAY: Heuristic (if-elif)        NEW WAY: Supervised Learning
────────────────────────────        ────────────────────────────
Human writes every rule             Machine derives rules from data
Fragile at scale                    Generalises to new data
No learning                         Improves with more examples
```

### 2. The Iris Benchmark Dataset

The Iris dataset is the "Hello World" of machine learning:

```
Samples    : 150  (50 per class — perfectly balanced)
Classes    : 3    → Setosa | Versicolor | Virginica
Features   : 4    → Sepal Length, Sepal Width,
                     Petal Length, Petal Width  (all in cm)
```

Each flower is described by 4 numerical measurements that form its **feature vector**. The model learns which regions of this 4D space correspond to which species.

### 3. The Gatekeeper Rule — Feature Scaling

Without scaling, KNN is **biased**:

```
Raw Data (Biased)              Standard Scaled (Balanced)
──────────────────             ──────────────────────────
Feature A: 0–1000              Feature A: mean=0, std=1
Feature B: 0–1                 Feature B: mean=0, std=1

Feature A dominates!           All features contribute equally ✅
```

`StandardScaler` formula: `z = (x - mean) / std`

> ⚠️ **Critical Rule:** Fit the scaler **only on training data**. Never let test data influence the scaler — that's data leakage!

### 4. The K-Nearest Neighbours Algorithm

**The Proximity Principle:** *Similar things exist in close proximity.*

```
When classifying a new point:
  1. Calculate distance to ALL training points
  2. Find the K nearest neighbours
  3. Majority vote → assign that class

K=5 means: "Ask your 5 closest neighbours what class you are."
```

### 5. Choosing K — The Elbow Method

```
K too small (K=1)   → Overfitting  → memorises noise → unstable
K too large (K=100) → Underfitting → too generic      → inaccurate
THE ELBOW           → minimum error rate              → optimal K
```

### 6. The Accuracy Mirage

> **"99% accuracy is a lie in imbalanced data. We must look deeper."**

| Metric | What it measures | When it matters |
|--------|------------------|-----------------|
| **Accuracy** | % of all correct predictions | Balanced datasets only |
| **Precision** | Of predicted positives, how many are real? | Spam filters |
| **Recall** | Of actual positives, how many were caught? | Medical diagnosis |
| **F1 Score** | Harmonic mean of Precision & Recall | Professional standard |

### 7. The Confusion Matrix — Diagnostic Tool

```
              Predicted Positive    Predicted Negative
Actual Positive     TP (✅)               FN (⚠️ missed)
Actual Negative     FP (🔔 false alarm)   TN (✅)
```

---

## The Pipeline — 8 Steps

```
Step 1 → LOAD & EXPLORE
         EDA: shape, class distribution, descriptive stats

Step 2 → SCALE (StandardScaler)
         Remove feature bias → mean=0, std=1

Step 3 → SPLIT (80% train / 20% test)
         Shuffle=True → removes order bias
         Random state=42 → reproducible

Step 4 → TRAIN (KNeighborsClassifier, K=5)
         model.fit(X_train, y_train) ← "memorise the map"

Step 5 → PREDICT
         predictions = model.predict(X_test)

Step 6 → EVALUATE
         Accuracy, F1 Score, Classification Report, Confusion Matrix

Step 7 → TUNE (Elbow Method, K=1→30)
         Find optimal K → retrain if different from 5

Step 8 → VISUALISE
         Elbow curve, Confusion Matrix heatmap, Decision Boundary
```

---

## Project Structure

```
Project_2_Data_Classification/
│
├── classifier.py           ← Full ML pipeline (run this!)
├── test_classifier.py      ← Unit test suite
├── elbow_curve.png         ← Generated: optimal K visualisation
├── confusion_matrix.png    ← Generated: diagnostic heatmap
├── decision_boundary.png   ← Generated: 2D class regions
└── README.md               ← This file
```

---

## Setup & Installation

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Or use the global requirements file
pip install -r ../requirements.txt
```

**Python version:** 3.9+

---

## How to Run

```bash
cd Project_2_Data_Classification
python classifier.py
```

**Sample output (abbreviated):**

```
🌸 🌸 🌸 PROJECT 2 — DATA CLASSIFICATION USING AI 🌸 🌸 🌸

=================================================================
  📊  STEP 1 — DATASET EXPLORATION
=================================================================

  Dataset : Iris Benchmark
  Samples : 150
  Features: 4  → ['sepal length (cm)', ...]
  Classes : 3  → ['setosa', 'versicolor', 'virginica']

=================================================================
  ⚖️   STEP 2 — FEATURE SCALING (StandardScaler)
=================================================================
  Before scaling — mean: [5.84  3.05  3.76  1.20]
  After  scaling — mean: [0.000 0.000 0.000 0.000]

=================================================================
  🤖  STEP 4 — MODEL TRAINING (KNN, K=5)
=================================================================
  ✅ Model fitted successfully.

=================================================================
  📈  STEPS 5 & 6 — PREDICTIONS & EVALUATION
=================================================================
  Accuracy  : 100.00%
  F1 Score  : 1.0000

=================================================================
  🏆  PIPELINE COMPLETE
=================================================================
  Optimal K      : 7
  Test Accuracy  : 100.00%
  F1 Score       : 1.0000
```

---

## How to Test

```bash
python test_classifier.py
```

**Expected output:**

```
====================================================
  Running Project 2 Test Suite
====================================================
✅  load_and_explore() — shape and names validated
✅  preprocess() — StandardScaler validated (mean≈0, std≈1)
✅  split_data() — 80/20 split validated
✅  Full pipeline — Accuracy=100.00%, F1=1.0000
✅  Multiple K values (1,3,5,7,11) — all valid accuracy scores
====================================================
  Result: 5/5 tests passed
====================================================
```

---

## Output Files

After running, three PNG files are generated:

| File | Description |
|------|-------------|
| `elbow_curve.png` | Error rate vs K value — the elbow shows optimal K |
| `confusion_matrix.png` | Heatmap of TP/TN/FP/FN per class |
| `decision_boundary.png` | 2D visualisation of learned class regions (petal features) |

---

## Understanding the Results

### Why does KNN achieve ~97–100% on Iris?

The Iris dataset is designed to be learnable. Setosa is **linearly separable** from the other two — the petal features alone can perfectly distinguish it. Versicolor and Virginica overlap slightly, which is where KNN may make occasional errors.

### Why do we use F1 instead of accuracy?

In real-world classification (cancer detection, fraud detection), the dataset is often **imbalanced**. A model that always predicts "not cancer" gets 99% accuracy on a dataset where only 1% of cases are positive — but it's completely useless. F1 catches this.

---

## Experiments to Try

1. **Change K** — What happens at K=1? K=50? K=149?
2. **Remove scaling** — Does accuracy drop without StandardScaler?
3. **Try a different algorithm** — Replace KNN with `DecisionTreeClassifier` or `SVC`
4. **Feature importance** — Which 2 features are most discriminative?
5. **Cross-validation** — Use `cross_val_score` for a more robust accuracy estimate
6. **Different dataset** — Try `load_wine()` or `load_breast_cancer()` from sklearn

---

## Real-World Applications

| Domain | Classification Problem |
|--------|----------------------|
| **Healthcare** | Tumour malignancy (benign vs malignant) |
| **Finance** | Loan default risk, fraud detection |
| **Email** | Spam vs legitimate messages |
| **Manufacturing** | Defective vs non-defective parts |
| **Agriculture** | Plant disease identification |
| **Autonomous Vehicles** | Road sign recognition |

---

## What's Next (Project 3)

Project 2 classifies **what data IS** (passive labelling).  
Project 3 will predict **what a user WANTS** (active personalisation).

```
Project 2:  Features ──[KNN]──► Class Label
            "What species is this flower?"

Project 3:  User Profile ──[Cosine Similarity]──► Ranked Items
            "What job roles match this person's skills?"
```

The shift: from **classifying existing data** to **proactively recommending relevant content**.

---

> **"Mastering the pipeline from raw data to evaluated model is the bedrock of every professional AI engineer's skill set."**  
> — DecodeLabs Architecture Briefing
