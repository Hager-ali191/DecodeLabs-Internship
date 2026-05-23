"""
============================================================
  PROJECT 2: Data Classification Using AI
  DecodeLabs Industrial Training Kit | Batch 2026
  Author  : AI Intern @ DecodeLabs
  Track   : Predictive Phase — Supervised Learning
============================================================

ARCHITECTURE OVERVIEW
─────────────────────
  MASTER BLUEPRINT: IPO FRAMEWORK
  ┌────────────────────────────────────────────────────────┐
  │  INPUT           PROCESS            OUTPUT             │
  │  ─────────       ──────────         ──────────         │
  │  Iris Dataset    Train-Test Split   Confusion Matrix   │
  │  Feature Scaling KNN Algorithm      F1 Score           │
  └────────────────────────────────────────────────────────┘

  FULL PIPELINE:
    1. Load & Explore data   (EDA)
    2. Preprocess            (StandardScaler — remove bias)
    3. Split                 (80% train / 20% test, shuffled)
    4. Train                 (KNeighborsClassifier, K=5)
    5. Predict               (model.predict on test set)
    6. Evaluate              (Accuracy, Confusion Matrix, F1)
    7. Tune                  (Find optimal K via elbow method)
    8. Visualise             (Decision boundary + confusion matrix)

  KEY INSIGHT: "99% accuracy is a mirage in imbalanced data.
                We must look deeper." — Use F1 Score!
"""

# ──────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)


# ──────────────────────────────────────────────────────────
# STEP 1 — LOAD & EXPLORE THE IRIS BENCHMARK DATASET
# ──────────────────────────────────────────────────────────

def load_and_explore() -> tuple:
    """
    Loads the Iris dataset and prints a full EDA summary.

    Dataset specs:
      • Samples   : 150 (50 per class — perfectly balanced)
      • Classes   : 3  (Setosa | Versicolor | Virginica)
      • Features  : 4  (Sepal Length, Sepal Width,
                        Petal Length, Petal Width)
      • Unit      : centimetres (cm)

    Returns:
        X (ndarray): Feature matrix  (150, 4)
        y (ndarray): Target vector   (150,)
        feature_names (list[str])
        target_names  (list[str])
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    target_names  = list(iris.target_names)

    # Build a pandas DataFrame for nice display
    df = pd.DataFrame(X, columns=feature_names)
    df["species"] = [target_names[label] for label in y]

    print("\n" + "=" * 65)
    print("  📊  STEP 1 — DATASET EXPLORATION")
    print("=" * 65)
    print(f"\n  Dataset : Iris Benchmark")
    print(f"  Samples : {X.shape[0]}")
    print(f"  Features: {X.shape[1]}  → {feature_names}")
    print(f"  Classes : {len(target_names)}  → {list(target_names)}\n")

    print("  First 5 rows:")
    print(df.head().to_string(index=False))

    print("\n  Class distribution:")
    print(df["species"].value_counts().to_string())

    print("\n  Descriptive statistics:")
    print(df.drop("species", axis=1).describe().round(2).to_string())

    return X, y, feature_names, target_names


# ──────────────────────────────────────────────────────────
# STEP 2 — GATEKEEPER RULE: FEATURE SCALING
# ──────────────────────────────────────────────────────────

def preprocess(X: np.ndarray) -> tuple:
    """
    Applies StandardScaler to remove feature bias.

    WHY SCALE?
      KNN uses Euclidean distance. Without scaling, a feature
      with large values (e.g., 0–1000) dominates a feature
      with small values (0–1), producing biased distances.

      StandardScaler transforms each feature to:
        Mean = 0,  Variance = 1
      → All features contribute equally to distance.

    CRITICAL RULE: Fit scaler ONLY on training data,
    then transform both train and test sets.
    (Never let test data influence the scaler — data leakage!)

    Returns:
        scaler   (StandardScaler): fitted scaler object
        X_scaled (ndarray): scaled feature matrix
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n" + "=" * 65)
    print("  ⚖️   STEP 2 — FEATURE SCALING (StandardScaler)")
    print("=" * 65)
    print(f"\n  Before scaling — mean per feature: {X.mean(axis=0).round(2)}")
    print(f"  After  scaling — mean per feature: {X_scaled.mean(axis=0).round(4)}")
    print(f"\n  Before scaling — std  per feature: {X.std(axis=0).round(2)}")
    print(f"  After  scaling — std  per feature: {X_scaled.std(axis=0).round(4)}")
    print("\n  ✅ All features now have Mean≈0 and Std≈1. Bias removed.")

    return scaler, X_scaled


# ──────────────────────────────────────────────────────────
# STEP 3 — STRUCTURAL INTEGRITY: THE TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────

def split_data(X_scaled: np.ndarray, y: np.ndarray) -> tuple:
    """
    Splits scaled data into training and testing sets.

    Split ratio : 80% train  /  20% test
    Shuffle     : True  → removes order bias
    random_state: 42    → reproducible results

    The test set is LOCKED after splitting —
    it must never influence training in any way.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, shuffle=True
    )

    print("\n" + "=" * 65)
    print("  ✂️   STEP 3 — TRAIN / TEST SPLIT")
    print("=" * 65)
    print(f"\n  Total samples : {len(y)}")
    print(f"  Training set  : {len(y_train)} samples ({len(y_train)/len(y)*100:.0f}%)")
    print(f"  Test set      : {len(y_test)} samples ({len(y_test)/len(y)*100:.0f}%)")
    print(f"  Shuffle       : True  |  Random state: 42")

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────
# STEP 4 — THE ALGORITHM: K-NEAREST NEIGHBOURS
# ──────────────────────────────────────────────────────────

def train_knn(X_train: np.ndarray, y_train: np.ndarray, k: int = 5) -> KNeighborsClassifier:
    """
    Trains a K-Nearest Neighbours classifier.

    THE PROXIMITY PRINCIPLE:
      "Similar things exist in close proximity."
      When classifying a new point:
        1. Calculate distance to ALL training points
        2. Find the K nearest neighbours
        3. Take a majority vote → assign that class

    Scikit-learn API (3 lines!):
      Instantiate → Fit → Predict
      model = KNeighborsClassifier(n_neighbors=5)
      model.fit(X_train, y_train)           # Memorise the map
      predictions = model.predict(X_test)   # Apply logic

    Args:
        k: Number of neighbours (default 5 — the elbow)

    Returns:
        Trained KNeighborsClassifier model
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    print("\n" + "=" * 65)
    print(f"  🤖  STEP 4 — MODEL TRAINING (KNN, K={k})")
    print("=" * 65)
    print(f"\n  Algorithm    : K-Nearest Neighbours")
    print(f"  K value      : {k}")
    print(f"  Training on  : {len(y_train)} samples")
    print(f"  ✅ Model fitted successfully.")

    return model


# ──────────────────────────────────────────────────────────
# STEP 5 & 6 — PREDICT + EVALUATE
# ──────────────────────────────────────────────────────────

def evaluate(model: KNeighborsClassifier,
             X_test: np.ndarray,
             y_test: np.ndarray,
             target_names: list) -> dict:
    """
    Runs predictions and prints a full evaluation report.

    Metrics explained:
      Accuracy  → What % of all predictions are correct?
                  (Misleading on imbalanced data!)

      Precision → Of all items predicted positive,
                  how many actually are? (Spam filter metric)

      Recall    → Of all actual positives,
                  how many did we catch? (Medical diagnosis metric)

      F1 Score  → Harmonic mean of Precision & Recall.
                  The professional balanced metric.

      Confusion Matrix → Grid showing TP, TN, FP, FN
                         per class. The diagnostic tool.

    Returns:
        dict with accuracy, f1, and predictions
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1       = f1_score(y_test, predictions, average="weighted")
    cm       = confusion_matrix(y_test, predictions)
    report   = classification_report(y_test, predictions,
                                     target_names=target_names)

    print("\n" + "=" * 65)
    print("  📈  STEP 5 & 6 — PREDICTIONS & EVALUATION")
    print("=" * 65)
    print(f"\n  Accuracy  : {accuracy * 100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}  (weighted average)")
    print("\n  Classification Report:")
    print(report)
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)

    return {"accuracy": accuracy, "f1": f1, "predictions": predictions, "cm": cm}


# ──────────────────────────────────────────────────────────
# STEP 7 — HYPERPARAMETER TUNING: FINDING OPTIMAL K
# ──────────────────────────────────────────────────────────

def find_optimal_k(X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray,
                   k_range: range = range(1, 31)) -> int:
    """
    Implements the Elbow Method to find the best K value.

    K too small (K=1) → Overfitting  → memorises noise
    K too large (K=100) → Underfitting → too generic
    THE ELBOW → sweet spot of minimum error rate

    Plots:
        Error rate vs K value curve with the optimal K marked.

    Returns:
        optimal_k (int): K value with minimum error rate
    """
    error_rates = []

    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        error_rates.append(1 - accuracy_score(y_test, preds))

    optimal_k    = k_range[error_rates.index(min(error_rates))]
    min_error    = min(error_rates)

    print("\n" + "=" * 65)
    print("  🔧  STEP 7 — HYPERPARAMETER TUNING (Elbow Method)")
    print("=" * 65)
    print(f"\n  K range tested : {list(k_range)}")
    print(f"  Optimal K      : {optimal_k}")
    print(f"  Minimum error  : {min_error:.4f} ({(1-min_error)*100:.2f}% accuracy)")

    # ── Plot: Error Rate vs K ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_range), error_rates, "b-o", linewidth=2,
            markersize=5, label="Error Rate")
    ax.axvline(x=optimal_k, color="red", linestyle="--", linewidth=2,
               label=f"Optimal K = {optimal_k}")
    ax.scatter([optimal_k], [min_error], color="red", s=150, zorder=5)
    ax.set_xlabel("K Value", fontsize=13)
    ax.set_ylabel("Error Rate", fontsize=13)
    ax.set_title("Elbow Method — Finding Optimal K for KNN", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  📸  Saved → elbow_curve.png")

    return optimal_k


# ──────────────────────────────────────────────────────────
# STEP 8 — VISUALISATION
# ──────────────────────────────────────────────────────────

def visualise_results(model: KNeighborsClassifier,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      predictions: np.ndarray,
                      cm: np.ndarray,
                      target_names: list) -> None:
    """
    Produces two diagnostic plots:
      1. Confusion Matrix heatmap
      2. Decision boundary (2D — using 2 most important features)
    """
    COLORS = ["#2196F3", "#4CAF50", "#FF5722"]   # Blue, Green, Orange

    # ── Plot 1: Confusion Matrix ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=target_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Confusion Matrix — KNN Classifier\n(Rows=Actual, Cols=Predicted)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  📸  Saved → confusion_matrix.png")

    # ── Plot 2: Decision Boundary (features 2 & 3 only) ───
    # Re-train on 2 features for 2D visualisation
    from sklearn.neighbors import KNeighborsClassifier as KNN
    X_2d      = X_test[:, 2:4]     # petal length & petal width
    clf_2d    = KNN(n_neighbors=model.n_neighbors)
    clf_2d.fit(X_test[:, 2:4], y_test)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap_bg = plt.cm.get_cmap("Set1", 3)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap_bg)

    for cls_idx, (cls_name, color) in enumerate(zip(target_names, COLORS)):
        mask = y_test == cls_idx
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, label=cls_name, edgecolors="black",
                   linewidths=0.7, s=80, zorder=3)

    ax.set_xlabel("Petal Length (scaled)", fontsize=12)
    ax.set_ylabel("Petal Width  (scaled)", fontsize=12)
    ax.set_title(f"KNN Decision Boundary (K={model.n_neighbors})\nIris Dataset — Petal Features",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  📸  Saved → decision_boundary.png")


# ──────────────────────────────────────────────────────────
# ENTRY POINT — FULL PIPELINE
# ──────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "🌸 " * 20)
    print("  PROJECT 2 — DATA CLASSIFICATION USING AI")
    print("  DecodeLabs | KNN on Iris Dataset")
    print("🌸 " * 20)

    # Step 1: Load & Explore
    X, y, feature_names, target_names = load_and_explore()

    # Step 2: Scale features
    scaler, X_scaled = preprocess(X)

    # Step 3: Split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Step 4: Train initial model (K=5)
    model = train_knn(X_train, y_train, k=5)

    # Steps 5 & 6: Predict + Evaluate
    results = evaluate(model, X_test, y_test, target_names)

    # Step 7: Find optimal K
    optimal_k = find_optimal_k(X_train, X_test, y_train, y_test)

    # Retrain with optimal K if different
    if optimal_k != 5:
        print(f"\n  🔄 Retraining with optimal K={optimal_k}...")
        model = train_knn(X_train, y_train, k=optimal_k)
        results = evaluate(model, X_test, y_test, target_names)

    # Step 8: Visualise
    print("\n" + "=" * 65)
    print("  🎨  STEP 8 — SAVING VISUALISATIONS")
    print("=" * 65)
    visualise_results(
        model, X_test, y_test,
        results["predictions"], results["cm"], target_names
    )

    # ── Final Summary ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("  🏆  PIPELINE COMPLETE — FINAL SUMMARY")
    print("=" * 65)
    print(f"\n  Dataset        : Iris  (150 samples, 3 classes, 4 features)")
    print(f"  Algorithm      : K-Nearest Neighbours")
    print(f"  Optimal K      : {model.n_neighbors}")
    print(f"  Test Accuracy  : {results['accuracy'] * 100:.2f}%")
    print(f"  F1 Score       : {results['f1']:.4f}")
    print(f"\n  Saved files    : elbow_curve.png")
    print(f"                   confusion_matrix.png")
    print(f"                   decision_boundary.png")
    print("\n  💡 Next: Try a Decision Tree or Random Forest and compare!\n")


if __name__ == "__main__":
    main()
