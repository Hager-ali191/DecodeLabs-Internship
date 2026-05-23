# 🎯 Project 3 — AI Recommendation Logic (Tech Stack Recommender)

> **DecodeLabs Industrial Training Kit · Batch 2026**  
> **Track:** Personalisation Phase — Pattern Alignment

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Core Concepts](#core-concepts)
4. [The Architecture](#the-architecture)
5. [The 4-Step Pipeline](#the-4-step-pipeline)
6. [Project Structure](#project-structure)
7. [Setup & Installation](#setup--installation)
8. [How to Run](#how-to-run)
9. [How to Test](#how-to-test)
10. [Output Files](#output-files)
11. [Understanding the Results](#understanding-the-results)
12. [Experiments to Try](#experiments-to-try)
13. [Real-World Applications](#real-world-applications)
14. [Graduation to Commercial-Grade Logic](#graduation-to-commercial-grade-logic)

---

## Overview

Project 3 is the **personalisation phase** and the capstone of the DecodeLabs AI internship. This is where you become a **Digital Matchmaker** — building a system that connects users to the exact content they need, even before they can fully articulate it themselves.

You will build a **Tech Stack Recommender** — a content-based filtering engine that:
- Accepts a user's known skills/technologies as input
- Transforms them into TF-IDF weighted feature vectors
- Computes cosine similarity against a catalogue of 18 job roles
- Returns a ranked Top-N list of the most relevant career paths

This is the same core logic powering Netflix, Spotify, Amazon, and every major tech company's recommendation engine.

> **"Recommendation engines serve as digital matchmakers, connecting users to their specific needs before those needs are explicitly articulated."**

---

## Learning Objectives

By completing this project, you will:

| Skill | Concept |
|-------|---------|
| **Paradigm Shift** | Move from passive classification to active prediction |
| **Feature Extraction** | Build TF-IDF weighted vectors from text/tag data |
| **Similarity Math** | Implement and understand cosine similarity |
| **Pipeline Design** | Architect a 4-step recommendation ranking system |
| **Cold-Start Handling** | Detect and bypass zero-vector user profiles |
| **Vector Spaces** | Understand orientation vs magnitude in high dimensions |
| **Production Thinking** | Export results, log recommendations, handle edge cases |

---

## Core Concepts

### 1. The Paradigm Shift

```
Project 2:  Passive Classification
            "What class does this data point belong to?"
            You label existing data.

Project 3:  Active Prediction
            "What does this user want next?"
            You generate personalised content proactively.
```

Modern digital environments generate immense noise. Recommendation engines **cure "Choice Overload"** by surfacing the right item at the right time.

---

### 2. Two Recommendation Methodologies

| Method | How it Works | Limitation |
|--------|-------------|------------|
| **Collaborative Filtering** | "Users like you also liked..." — driven by community behaviour | Requires massive historical interaction data |
| **Content-Based Filtering** ✅ | Maps user preferences to item attributes directly — driven by item features | Cold-start for new users |

**Project 3 uses Content-Based Filtering** — it works immediately without any historical data, making it perfect for our use case.

---

### 3. Bridging the Language Barrier — Vector Mapping

Machines understand numbers, not words like `"Python"` or `"Docker"`.

To make similarity math work, we must convert qualitative skills into **numerical vectors** within a **shared vocabulary space**:

```
User skills: ["Python", "Cloud", "Automation"]
                ↓ Vector Mapping ↓
User vector: [1, 0, 1, 0, 0, 1, 0, ...]   ← TF-IDF weighted

Job Role A:  [1, 1, 1, 0, 0, 1, 0, ...]
Job Role B:  [0, 0, 0, 1, 1, 0, 1, ...]

Cosine(User, Role A) = 0.89  ← Strong match!
Cosine(User, Role B) = 0.02  ← Weak match
```

> ⚠️ **Critical:** Item features and user features **must use the exact same vocabulary**. Naming discrepancies (`"Web Design"` vs `"Frontend Development"`) will cause the similarity math to fail silently.

---

### 4. The Limitation of Binary Vectors

Simple binary (1/0) matching treats every word equally:

```
"software" == "neural networks"  ← both score 1, but very different specificity!
```

This is the flaw that TF-IDF solves.

---

### 5. TF-IDF Weighting — The Upgrade

**Term Frequency–Inverse Document Frequency (TF-IDF)** is the statistical revolution:

```
TF(t, d)  = Count of term t in document d
            ─────────────────────────────
            Total terms in document d

IDF(t)    = log(  Total Documents  )
                ( ──────────────── )
                ( Docs with term t )

TF-IDF = TF × IDF
```

**Effect:**
- Common terms (appear in all roles) → **LOW weight** (penalised)
- Specific terms (appear in few roles) → **HIGH weight** (rewarded)

```
Example:
"python"          → appears in 14/18 roles → LOW weight
"quantum computing" → appears in 1/18 roles → HIGH weight
```

The logarithm in IDF creates a dampening effect — preventing extreme values while preserving the penalty gradient.

---

### 6. Cosine Similarity — The Industry Standard

**Why NOT Euclidean distance?**

If two items share identical tags, but one has a much larger feature set (e.g., a massive job description vs a short one), their Euclidean distance will be high — even though they're thematically identical. Euclidean is **magnitude-sensitive**.

**Cosine similarity** measures the **angle** between two vectors:

```
         A · B
cos(θ) = ──────
         ‖A‖‖B‖

Score  1 → Vectors perfectly aligned (identical orientation) ✅
Score  0 → Vectors orthogonal (no common characteristics)
Score -1 → Vectors opposite (TF-IDF is non-negative, so this won't occur)
```

Cosine is **invariant to magnitude** — it cares about the *direction* of preferences, not the size of the profile. This is exactly what we want.

---

### 7. The Cold-Start Problem

Even the best pipeline fails without initial data:

| Cold Start Type | Problem | Solution |
|-----------------|---------|----------|
| **User Cold Start** | New user → zero vector → cosine = 0 for all items | Onboarding surveys, trending fallback, metadata inference |
| **Item Cold Start** | New item → no interaction history | Content-based filtering is inherently immune (uses metadata!) |

Our engine implements:
- **Guard:** Raises an error if fewer than 3 skills are provided
- **Fallback:** Returns trending popular roles when no vocabulary overlap exists

---

## The Architecture

```
TechStackRecommender
│
├── JOB_ROLES catalogue (18 roles, each with skills list)
│
├── fit()
│     Converts skill lists → TF-IDF item matrix
│     Cached once at startup for efficiency
│
├── _build_user_vector(skills)
│     Projects user skills into the SAME vector space
│
└── recommend(user_skills, top_n=5)
      Step 1: Ingestion  → validate + vectorise
      Step 2: Scoring    → cosine similarity vs all roles
      Step 3: Sorting    → descending order
      Step 4: Filtering  → Top-N slice
```

---

## The 4-Step Pipeline

### Step 1 — Ingestion
Capture the user state. Accepts minimum **3 skills** to ensure sufficient data density for accurate matching.

### Step 2 — Scoring
Loop through every item in the catalogue, calculate its cosine similarity score against the user vector, and store the resulting value.

```python
score = cosine_similarity(user_vector, item_vector)
# Score 0.95 → 95% thematic alignment
```

### Step 3 — Sorting
Organise the scored dataset in **descending order** — the most relevant roles rise to the top.

### Step 4 — Filtering (Top-N)
Prevent choice overload. Truncate the output to the Top-N list (default: Top 5), displaying only the highest-scoring matches.

---

## Project Structure

```
Project_3_AI_Recommendation_System/
│
├── recommender.py                 ← Full recommendation engine (run this!)
├── test_recommender.py            ← Unit test suite
├── demo_recommendations.png       ← Generated: bar chart of top matches
├── interactive_recommendations.png ← Generated: from interactive session
├── recommendation_results.json   ← Generated: exported results
└── README.md                      ← This file
```

---

## Setup & Installation

```bash
pip install numpy pandas scikit-learn matplotlib

# Or use the global requirements file
pip install -r ../requirements.txt
```

> **Note:** The engine also includes a **pure Python fallback** (`ManualTFIDF`) that works without scikit-learn, for maximum educational transparency.

**Python version:** 3.9+

---

## How to Run

```bash
cd Project_3_AI_Recommendation_System
python recommender.py
```

The program runs two modes:
1. **Demo Mode** — automatically tests 5 predefined skill profiles
2. **Interactive Mode** — accepts your own skills as input

**Sample interactive session:**

```
==================================================================
  🎮  INTERACTIVE MODE — Tech Stack Recommender
==================================================================

  Enter your skills/technologies (comma-separated).
  Example: python, machine learning, sql, docker

  Your skills: python, machine learning, sql, pandas, tensorflow

  🎯  TOP RECOMMENDATIONS FOR YOUR SKILL PROFILE
==================================================================

  Your skills : python, machine learning, sql, pandas, tensorflow

  🥇  Rank 1: Data Scientist
       Cosine Similarity : 0.8742  (87.4% match)
       Matching Skills   : python, machine learning, sql, pandas

  🥈  Rank 2: Machine Learning Engineer
       Cosine Similarity : 0.8159  (81.6% match)
       Matching Skills   : python, machine learning, tensorflow

  🥉  Rank 3: AI Research Scientist
       Cosine Similarity : 0.7203  (72.0% match)
       Matching Skills   : python, deep learning, tensorflow

  🏅  Rank 4: Data Engineer
       Cosine Similarity : 0.5841  (58.4% match)
       Matching Skills   : python, sql

  🏅  Rank 5: MLOps Engineer
       Cosine Similarity : 0.5320  (53.2% match)
       Matching Skills   : python, machine learning
```

---

## How to Test

```bash
python test_recommender.py
```

**Expected output:**

```
====================================================
  Running Project 3 Test Suite
====================================================
✅  Engine initialised — 18 roles in catalogue
✅  recommend() returns exactly Top-5 list
✅  All cosine scores are in [0, 1]
✅  Results are sorted in descending order by cosine score
✅  Cold-start guard triggered correctly for <3 skills
✅  manual_cosine_similarity() — identity, orthogonal, zero cases verified
✅  ManualTFIDF — matrix shape (2, N), non-zero values verified
✅  Different profiles give different top roles: ML→Data Scientist | Web→Frontend Developer
====================================================
  Result: 8/8 tests passed
====================================================
```

---

## Output Files

| File | Description |
|------|-------------|
| `demo_recommendations.png` | Horizontal bar chart of cosine similarity scores for demo profile |
| `interactive_recommendations.png` | Chart from your interactive session |
| `recommendation_results.json` | JSON export of top recommendations for downstream use |

---

## Understanding the Results

### What does a cosine score of 0.85 mean?

The user's skill vector and the job role's feature vector have an angular alignment of approximately 32° — they point in nearly the same direction in the high-dimensional feature space. Their interests are highly aligned.

### Why might two roles with different skills have the same score?

TF-IDF weighting can equalise roles that share highly specific (rare) terms even if they share fewer total terms. This is intentional — specificity matters more than volume.

---

## Experiments to Try

1. **Add more roles** — Add "Blockchain Developer" or "AR/VR Engineer" to `JOB_ROLES`
2. **Tune Top-N** — Change `top_n` to see Top-3 vs Top-10
3. **Try TF-IDF variants** — Experiment with `max_features`, `sublinear_tf=True`
4. **Add rating weights** — Let users rate their skills (beginner/expert) and weight the vector
5. **Multi-domain recommender** — Recommend courses, books, or projects instead of job roles
6. **Collaborative filter comparison** — What would change if you used a user-user similarity approach?

---

## Real-World Applications

| Company | What Their Recommender Does |
|---------|---------------------------|
| **Netflix** | Recommends shows based on watch history and genre preferences |
| **Spotify** | Suggests songs via audio feature vectors and listening patterns |
| **Amazon** | Surfaces products via purchase history and item attribute matching |
| **LinkedIn** | Recommends job listings based on profile skills and experience |
| **YouTube** | Recommends videos based on watch time, topics, and engagement |
| **Coursera** | Suggests courses based on enrolled topics and career goals |

The fundamental principles — **feature extraction** and **cosine similarity** — are the bedrock of all of these.

---

## Graduation to Commercial-Grade Logic

Mastering this project transitions you from classifying what data *is* to **predicting what a user wants**.

```
Project 1:   Key ──logic──► Value           (deterministic rules)
Project 2:   Features ──learned──► Label    (supervised classification)
Project 3:   Profile ──similarity──► Items  (personalised prediction)
             ↑
             This is where AI creates direct commercial value.
```

Whether applied to streaming movies, retail products, or tech stacks, the principles of **feature extraction** and **cosine similarity** remain the bedrock of modern AI. You now have the keys to build engines that navigate the noise of the internet and output flawless, relevant matches.

> **"You are ready to build the Digital Matchmaker."**  
> — DecodeLabs Architecture Briefing
