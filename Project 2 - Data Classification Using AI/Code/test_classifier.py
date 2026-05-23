"""
============================================================
  PROJECT 2: Data Classification — Unit Tests
  DecodeLabs Industrial Training Kit | Batch 2026
============================================================
Run with:  python test_classifier.py
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from sklearn.datasets import load_iris
from classifier import load_and_explore, preprocess, split_data, train_knn, evaluate


def test_data_loading():
    X, y, feature_names, target_names = load_and_explore()
    assert X.shape == (150, 4),  "Expected 150×4 feature matrix"
    assert y.shape == (150,),    "Expected 150 labels"
    assert len(feature_names) == 4
    assert len(target_names)  == 3
    print("✅  load_and_explore() — shape and names validated")


def test_preprocessing():
    iris = load_iris()
    _, X_scaled = preprocess(iris.data)
    assert abs(X_scaled.mean()) < 0.01, "Mean should be ~0 after scaling"
    assert abs(X_scaled.std()  - 1) < 0.05, "Std should be ~1 after scaling"
    print("✅  preprocess() — StandardScaler validated (mean≈0, std≈1)")


def test_split():
    iris = load_iris()
    _, X_scaled = preprocess(iris.data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, iris.target)
    total = len(y_train) + len(y_test)
    assert total == 150, "Train+test should equal 150"
    assert len(y_test) == 30, "Expected 20% → 30 test samples"
    print("✅  split_data() — 80/20 split validated")


def test_model_training_and_prediction():
    iris = load_iris()
    _, X_scaled = preprocess(iris.data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, iris.target)
    model = train_knn(X_train, y_train, k=5)
    results = evaluate(model, X_test, y_test, list(iris.target_names))
    assert results["accuracy"] >= 0.90, f"Accuracy too low: {results['accuracy']:.2%}"
    assert results["f1"]       >= 0.90, f"F1 too low: {results['f1']:.4f}"
    print(f"✅  Full pipeline — Accuracy={results['accuracy']:.2%}, F1={results['f1']:.4f}")


def test_different_k_values():
    iris = load_iris()
    _, X_scaled = preprocess(iris.data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, iris.target)
    for k in [1, 3, 5, 7, 11]:
        model   = train_knn(X_train, y_train, k=k)
        results = evaluate(model, X_test, y_test, list(iris.target_names))
        assert 0.0 <= results["accuracy"] <= 1.0, f"Invalid accuracy for K={k}"
    print("✅  Multiple K values (1,3,5,7,11) — all produced valid accuracy scores")


if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  Running Project 2 Test Suite")
    print("=" * 52)

    tests = [
        test_data_loading,
        test_preprocessing,
        test_split,
        test_model_training_and_prediction,
        test_different_k_values,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌  {test.__name__} FAILED: {e}")
        except Exception as e:
            print(f"💥  {test.__name__} ERROR: {e}")

    print(f"\n{'='*52}")
    print(f"  Result: {passed}/{len(tests)} tests passed")
    print("=" * 52 + "\n")
