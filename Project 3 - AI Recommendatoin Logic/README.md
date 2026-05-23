# 🎯 Project 3 — AI Recommendation System (Tech Stack Recommender)

---

## 📋 Table of Contents

1. [Project Idea](#project-idea)
2. [What Is Required](#what-is-required)
3. [What Is Implemented](#what-is-implemented)
4. [Core Concepts](#core-concepts)
5. [The 4-Step Pipeline](#the-4-step-pipeline)
6. [Project Structure](#project-structure)
7. [Setup & Run](#setup--run)
8. [How to Test](#how-to-test)
9. [Output Files](#output-files)
10. [Experiments to Try](#experiments-to-try)
11. [Real-World Applications](#real-world-applications)

---

## Project Idea

Every time Netflix suggests a show, Spotify generates a playlist, or LinkedIn recommends a job — a recommendation engine is running. These systems solve a fundamental problem: given a near-infinite catalogue of items, how do you surface exactly what *this specific user* wants, right now?

This project builds a **content-based filtering recommendation engine** applied to career guidance: a **Tech Stack Recommender** that takes a user's skills as input and returns a ranked list of the most relevant job roles. The system uses **TF-IDF feature extraction** and **cosine similarity** — the same mathematical foundation used by production recommendation systems at scale.

The key shift from classification: instead of asking *"what category does this data belong to?"*, you ask *"what items best match this user's preferences?"* — moving from passive labelling to active, personalised prediction.

---

## What Is Required

| Requirement | Description |
|------------|-------------|
| **User input** | Accept at least 3 user preferences (skills or interests) |
| **Item catalogue** | A dataset of items with associated attributes/tags |
| **Matching logic** | Compare user profile to items using similarity |
| **Ranked output** | Return a Top-N list of the best-matching items |
| **Cold-start handling** | Gracefully handle users with no or insufficient data |

---

## What Is Implemented

### The Engine — `TechStackRecommender`

A full content-based filtering engine with:
- **18 job roles** as the item catalogue, each described by a list of relevant skills and tools
- **TF-IDF vectorisation** to convert skill lists into weighted numerical vectors
- **Cosine similarity** to measure the angular alignment between user profile and each job role
- **4-step ranking pipeline**: Ingestion → Scoring → Sorting → Filtering
- **Cold-start guard**: rejects profiles with fewer than 3 skills; falls back to trending roles on zero-vector
- **Interactive mode**: live terminal session for real-time recommendations
- **JSON export**: saves results to file for downstream use
- **Pure Python fallback** (`ManualTFIDF`): works even without scikit-learn, showing the math from scratch

### TF-IDF Weighting

Simple binary (1/0) matching treats every word equally — `"python"` and `"quantum computing"` get the same weight even though one appears in 14 roles and the other in only 1. TF-IDF fixes this:

```
TF(t, d)  = occurrences of term t in document d / total terms in d
IDF(t)    = log( total documents / documents containing t )
TF-IDF    = TF × IDF

Effect:
  "python"            → appears in 14/18 roles → LOW weight  (too common)
  "quantum computing" → appears in  1/18 roles → HIGH weight (very specific)
```

### Cosine Similarity

Euclidean distance is sensitive to vector magnitude — a job description with 50 skills will always be "far" from one with 5, even if they share the same themes. Cosine similarity solves this by measuring **angle**, not distance:

```
         A · B
cos(θ) = ──────       Range: 0 (no match) → 1 (perfect match)
         ‖A‖ · ‖B‖

Score 1.0  →  vectors point in identical directions (perfect alignment)
Score 0.5  →  moderate thematic overlap
Score 0.0  →  nothing in common
```

### Cold-Start Handling

```
User Cold Start  →  profile is a zero vector → cosine = 0 for everything
Solution         →  require min 3 skills (onboarding) + trending fallback

Item Cold Start  →  new item with no interaction history
Solution         →  content-based filtering is inherently immune
                    (new items are recommended immediately via their metadata)
```

---

## Core Concepts

### Content-Based vs Collaborative Filtering

| Method | How It Works | Needs | Used When |
|--------|-------------|-------|-----------|
| **Collaborative** | "Users like you also liked..." | Large historical interaction data | Netflix, Amazon at scale |
| **Content-Based** ✅ | Maps user profile directly to item attributes | Only item metadata | New platforms, no interaction history needed |

This project uses content-based filtering — it works on day one with no historical data.

### Vector Space Representation

Machines understand numbers, not words. To compare a user's skills to a job role's requirements, both must be encoded into the **same numerical vector space**:

```
Vocabulary (shared):  ["aws", "docker", "python", "sql", ...]
                           ↑ position 0    ↑ pos 1   ...

User:     ["python", "sql"]     →  [0, 0, 0.6, 0.8, ...]
Role A:   ["python", "docker"]  →  [0, 0.7, 0.5, 0,  ...]
Role B:   ["aws", "docker"]     →  [0.9, 0.7, 0, 0,  ...]

cosine(User, Role A) = 0.74  ← good match
cosine(User, Role B) = 0.01  ← poor match
```

---

## The 4-Step Pipeline

```
Step 1 — INGESTION
  Accept user skills (minimum 3)
  Validate input → detect cold-start
  Build user TF-IDF vector in the shared vocabulary space

Step 2 — SCORING
  Loop through all 18 job roles
  Compute cosine similarity: user_vector vs each role_vector
  Store (role, score, matching_skills) tuples

Step 3 — SORTING
  Sort all scored roles in descending order
  Most relevant roles rise to the top

Step 4 — FILTERING
  Slice Top-N results (default: 5)
  Prevents choice overload
  Returns clean ranked list to the user
```

---

## Project Structure

```
Project_3_AI_Recommendation_System/
│
├── recommender.py                  ← Full engine — run this
├── test_recommender.py             ← Unit test suite (8 tests)
├── generate_visuals.py             ← Generates all charts
├── recommendation_results.json     ← Generated: exported results
│
├── python_ml_data.png              ← Generated: ML/Data profile chart
├── cloud_devops.png                ← Generated: DevOps profile chart
├── nlp_research.png                ← Generated: NLP profile chart
├── full_stack_web.png              ← Generated: Web Dev profile chart
├── computer_vision.png             ← Generated: CV profile chart
├── heatmap_all_profiles.png        ← Generated: all profiles × all roles
├── cosine_explainer.png            ← Generated: similarity math diagram
│
└── README.md                       ← This file
```

---

## Setup & Run

```bash
pip install numpy pandas scikit-learn matplotlib
python recommender.py
```

The program runs two modes automatically:
1. **Demo mode** — tests 5 predefined skill profiles, saves charts and JSON
2. **Interactive mode** — enter your own skills, get real-time recommendations

**Sample output:**
```
Your skills: python, machine learning, sql, pandas, tensorflow

  🥇  Rank 1: Data Scientist          0.7335  (73.3% match)
  🥈  Rank 2: Machine Learning Eng    0.4151  (41.5% match)
  🥉  Rank 3: Data Analyst            0.3981  (39.8% match)
  🏅  Rank 4: AI Research Scientist   0.3702  (37.0% match)
  🏅  Rank 5: MLOps Engineer          0.2891  (28.9% match)
```

To regenerate all charts:
```bash
python generate_visuals.py
```

---

## How to Test

```bash
python test_recommender.py
```

Tests cover: engine initialisation, Top-N count, score range [0,1], descending sort order, cold-start guard (< 3 skills), cosine similarity edge cases (identical/orthogonal/zero vectors), TF-IDF matrix shape, and different profiles producing different top results.

```
Result: 8/8 tests passed
```

---

## Output Files

| File | What It Shows |
|------|--------------|
| `python_ml_data.png` | Top 5 roles for Python/ML/Data profile |
| `cloud_devops.png` | Top 5 roles for Cloud/DevOps profile |
| `nlp_research.png` | Top 5 roles for NLP/Research profile |
| `full_stack_web.png` | Top 5 roles for Full Stack Web profile |
| `computer_vision.png` | Top 5 roles for Computer Vision profile |
| `heatmap_all_profiles.png` | All 5 profiles × 18 roles — full similarity matrix |
| `cosine_explainer.png` | Visual explanation of the cosine similarity geometry |
| `recommendation_results.json` | Top matches for first demo profile, JSON format |

---

## Experiments to Try

1. **Add more job roles** — Add "Blockchain Developer", "AR/VR Engineer" to `JOB_ROLES`
2. **Weighted skills** — Let users rate expertise level (1–5) and multiply the vector weights
3. **Change Top-N** — Try returning Top-3 vs Top-10, observe the difference in usefulness
4. **Different domain** — Replace job roles with courses, books, or movies and their tags
5. **Evaluate quality** — Ask domain experts to rate the recommendations and compute NDCG
6. **Compare methods** — Implement a simple collaborative filter and compare outputs

---

## Real-World Applications

| Platform | What the Recommender Does |
|----------|--------------------------|
| **Netflix / Spotify** | Matches user taste profile to content feature vectors |
| **LinkedIn** | Surfaces job listings aligned with profile skills |
| **Amazon** | Recommends products based on item attribute overlap |
| **Coursera / edX** | Suggests courses matching stated learning goals |
| **GitHub** | Recommends repositories based on language and topic tags |
| **Stack Overflow Jobs** | Matches developer skills to role requirements |

The mathematical foundation — **TF-IDF feature extraction + cosine similarity** — is identical across all of these at their core, regardless of scale.

---

> Recommendation systems are where AI creates direct commercial value. They cure "choice overload" by connecting users to what they need before they can fully articulate it themselves. Mastering this pipeline is the bridge between academic ML and production AI engineering.
