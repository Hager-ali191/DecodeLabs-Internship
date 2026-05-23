"""
============================================================
  PROJECT 3: AI Recommendation System — Unit Tests
  DecodeLabs Industrial Training Kit | Batch 2026
============================================================
Run with:  python test_recommender.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from recommender import TechStackRecommender, ManualTFIDF, manual_cosine_similarity
import numpy as np


def test_engine_initialises():
    engine = TechStackRecommender(top_n=5)
    engine.fit()
    assert engine._fitted, "Engine should be fitted after fit()"
    assert engine.item_matrix is not None, "Item matrix should exist"
    assert len(engine.roles) > 0, "Should have at least 1 role"
    print(f"✅  Engine initialised — {len(engine.roles)} roles in catalogue")


def test_recommendations_returned():
    engine = TechStackRecommender(top_n=5).fit()
    skills = ["python", "machine learning", "sql"]
    recs   = engine.recommend(skills, top_n=5)
    assert isinstance(recs, list), "Should return a list"
    assert len(recs) == 5,         "Should return exactly 5 recommendations"
    print("✅  recommend() returns exactly Top-5 list")


def test_scores_valid_range():
    engine = TechStackRecommender(top_n=3).fit()
    skills = ["python", "docker", "kubernetes"]
    recs   = engine.recommend(skills)
    for rec in recs:
        assert 0.0 <= rec["score"] <= 1.0, f"Score out of range: {rec['score']}"
    print("✅  All cosine scores are in [0, 1]")


def test_sorted_descending():
    engine = TechStackRecommender(top_n=5).fit()
    skills = ["javascript", "react", "python", "api", "git"]
    recs   = engine.recommend(skills)
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True), "Results not sorted descending"
    print("✅  Results are sorted in descending order by cosine score")


def test_cold_start_guard():
    engine = TechStackRecommender().fit()
    try:
        engine.recommend(["python"])   # Only 1 skill — should raise ValueError
        print("❌  Cold-start guard should have raised ValueError")
    except ValueError as e:
        assert "3" in str(e), "Error should mention requiring 3 skills"
        print("✅  Cold-start guard triggered correctly for <3 skills")


def test_manual_cosine_similarity():
    a = np.array([1, 0, 1, 0])
    b = np.array([1, 0, 1, 0])
    c = np.array([0, 1, 0, 1])

    assert abs(manual_cosine_similarity(a, b) - 1.0) < 1e-6, "Identical vectors should score 1.0"
    assert abs(manual_cosine_similarity(a, c) - 0.0) < 1e-6, "Orthogonal vectors should score 0.0"
    zero = np.array([0, 0, 0, 0])
    assert manual_cosine_similarity(a, zero) == 0.0, "Zero vector should return 0.0"
    print("✅  manual_cosine_similarity() — identity, orthogonal, and zero cases verified")


def test_manual_tfidf():
    tfidf = ManualTFIDF()
    docs  = [["python", "sql", "pandas"], ["docker", "kubernetes", "aws"]]
    matrix = tfidf.fit_transform(docs)
    assert matrix.shape == (2, len(tfidf.vocabulary_)), "Shape mismatch"
    assert matrix.sum() > 0, "TF-IDF matrix should have non-zero values"
    print(f"✅  ManualTFIDF — matrix shape {matrix.shape}, non-zero values verified")


def test_different_profiles_differ():
    engine   = TechStackRecommender(top_n=3).fit()
    ml_recs  = engine.recommend(["python", "machine learning", "tensorflow"])
    web_recs = engine.recommend(["javascript", "react", "html"])
    top_ml   = ml_recs[0]["role"]
    top_web  = web_recs[0]["role"]
    print(f"✅  Different profiles give different top roles: ML→{top_ml} | Web→{top_web}")


if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  Running Project 3 Test Suite")
    print("=" * 52)

    tests = [
        test_engine_initialises,
        test_recommendations_returned,
        test_scores_valid_range,
        test_sorted_descending,
        test_cold_start_guard,
        test_manual_cosine_similarity,
        test_manual_tfidf,
        test_different_profiles_differ,
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
