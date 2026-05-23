# 🧠 Artificial Intelligence — Project Series

A series of three progressive AI projects that build from foundational logic all the way to production-style machine learning systems. Each project introduces a new paradigm, a new algorithm, and a new set of skills — forming a complete arc from deterministic programming to intelligent, data-driven prediction.

---

## 📁 Repository Structure

```
AI-Projects/
│
├── Project_1_Rule_Based_Chatbot/
│   ├── chatbot.py
│   ├── test_chatbot.py
│   └── README.md
│
├── Project_2_Data_Classification/
│   ├── classifier.py
│   ├── test_classifier.py
│   ├── elbow_curve.png
│   ├── confusion_matrix.png
│   ├── decision_boundary.png
│   └── README.md
│
├── Project_3_AI_Recommendation_System/
│   ├── recommender.py
│   ├── test_recommender.py
│   ├── generate_visuals.py
│   ├── recommendation_results.json
│   ├── heatmap_all_profiles.png
│   ├── cosine_explainer.png
│   ├── [5 profile charts].png
│   └── README.md
│
└── README.md               ← You are here
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd AI-Projects

# Install dependencies (Projects 2 and 3 only)
pip install -r requirements.txt

# Run each project
python Project_1_Rule_Based_Chatbot/chatbot.py
python Project_2_Data_Classification/classifier.py
python Project_3_AI_Recommendation_System/recommender.py
```

---

## Project Overview

### Project 1 — Rule-Based AI Chatbot

**Paradigm:** Deterministic Logic  
**Core skills:** Control flow, data structures, input sanitisation

**The idea:** Build a chatbot that responds intelligently to user input using pure programmatic logic — no machine learning involved. The system runs in an infinite loop, normalises raw input, matches it against a dictionary-based knowledge base with O(1) lookup, and returns a contextual response. It exits cleanly on command and handles all edge cases gracefully.

**What it teaches:** This project establishes the architectural foundation that all AI systems rest on. Even the most advanced language models wrap their probabilistic outputs with deterministic rule-based guardrails — this is that control layer. It introduces the IPO (Input-Process-Output) model, the performance difference between O(1) dictionaries and O(n) if-elif ladders, and the concept of "white box" explainability.

**Benefit to AI field:** Rule-based systems are essential in safety-critical industries (finance, healthcare) where every decision must be fully traceable and auditable. Frameworks like NVIDIA NeMo and Meta's Llama Guard implement this exact pattern to make powerful LLMs safe for production use.

---

### Project 2 — Data Classification Using AI

**Paradigm:** Supervised Machine Learning  
**Core skills:** EDA, feature scaling, model training, evaluation metrics  
**Algorithm:** K-Nearest Neighbours (KNN)  
**Dataset:** Iris (150 samples, 4 features, 3 classes)

**The idea:** Implement a complete end-to-end supervised learning pipeline. Given a flower's physical measurements, the model learns to classify it into one of three species. The project covers every production step: exploratory data analysis, StandardScaler preprocessing, train/test split, KNN training, prediction, full evaluation (accuracy, F1, confusion matrix), and hyperparameter tuning via the Elbow Method — all with visualisations.

**What it teaches:** The critical shift from "writing rules manually" to "learning rules from data." The project demonstrates why feature scaling is non-negotiable for distance-based algorithms, why accuracy alone is a misleading metric (the Accuracy Mirage), and how the confusion matrix gives a full diagnostic picture of model behaviour. The Elbow Method introduces the concept of hyperparameter optimisation.

**Benefit to AI field:** Classification is the backbone of the most impactful AI applications in the world — fraud detection, medical diagnosis, spam filtering, quality control, and more. Understanding the full ML pipeline from raw data to validated model is the foundational skill that every machine learning engineer must have before building anything more complex.

---

### Project 3 — AI Recommendation System

**Paradigm:** Content-Based Filtering  
**Core skills:** Feature extraction, vector spaces, similarity mathematics  
**Algorithm:** TF-IDF + Cosine Similarity  
**Application:** Tech Stack Recommender (maps skills → job roles)

**The idea:** Build a recommendation engine that takes a user's skills as input and returns a ranked list of the most relevant career paths. The system converts skill lists into TF-IDF weighted vectors, then measures the cosine similarity (angular alignment) between the user's profile and each of 18 job role catalogues. A strict 4-step pipeline — Ingestion, Scoring, Sorting, Filtering — ensures results are accurate, ranked, and bounded (no choice overload). Cold-start scenarios are detected and handled gracefully.

**What it teaches:** The fundamental shift from passive classification ("what is this?") to active personalised prediction ("what does this person want?"). The project explains why binary overlap counting is insufficient (all words treated equally), how TF-IDF solves this by weighting specificity, why cosine similarity outperforms Euclidean distance for text-like feature spaces, and the real-world "Cold Start Problem" that every recommender system must solve.

**Benefit to AI field:** Recommendation systems are arguably where AI creates the most direct commercial value in the world today — powering Netflix, Spotify, Amazon, YouTube, LinkedIn, and every major digital platform. The core mathematics of TF-IDF and cosine similarity remain the industry standard at the heart of these systems. Mastering this pipeline is the bridge between academic machine learning and production AI engineering.

---

## The Learning Arc

These three projects form a deliberate progression:

```
Project 1              Project 2                 Project 3
────────────           ──────────────            ─────────────────────
Rule-Based             Supervised Learning        Content-Based Filtering
Deterministic          Probabilistic              Similarity-Based

"I write the rules"    "Data derives the rules"   "Math finds the match"

dict.get() O(1)        KNN algorithm              TF-IDF + Cosine Similarity
No libraries           scikit-learn               scikit-learn / pure Python

Chatbot                Iris Classifier            Tech Stack Recommender
```

Each project unlocks the next. The chatbot builds the mindset for structured pipelines. The classifier introduces the full ML workflow. The recommender applies vector mathematics to real personalisation problems.

---

## Dependencies

```
Python  3.9+

Project 1:  No external libraries (pure Python)
Project 2:  numpy, pandas, scikit-learn, matplotlib
Project 3:  numpy, pandas, scikit-learn, matplotlib
```

Install:
```bash
pip install -r requirements.txt
```

---

## Test Results

| Project | Tests | Status |
|---------|-------|--------|
| Project 1 — Rule-Based Chatbot | 6 / 6 | ✅ All passing |
| Project 2 — Data Classification | 5 / 5 | ✅ All passing |
| Project 3 — Recommendation System | 8 / 8 | ✅ All passing |
