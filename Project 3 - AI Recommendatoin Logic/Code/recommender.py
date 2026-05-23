"""
============================================================
  PROJECT 3: AI Recommendation Logic — Tech Stack Recommender
  DecodeLabs Industrial Training Kit | Batch 2026
  Author  : AI Intern @ DecodeLabs
  Track   : Personalisation Phase — Pattern Alignment
============================================================

ARCHITECTURE OVERVIEW
─────────────────────
  SYSTEM: Content-Based Filtering (Digital Matchmaker)

  INPUT-PROCESS-OUTPUT MODEL:
  ┌─────────────────────────────────────────────────────────┐
  │  INPUT               PROCESS              OUTPUT        │
  │  ─────────           ────────             ──────────    │
  │  User State          Similarity Logic     Top-N List   │
  │  (skills/goals)      (TF-IDF + Cosine)   (ranked)     │
  └─────────────────────────────────────────────────────────┘

  4-STEP RANKING PIPELINE:
    1. INGESTION  → Capture the user state (min 3 skills)
    2. SCORING    → Cosine similarity for every job role
    3. SORTING    → Rank results in descending order
    4. FILTERING  → Return Top-N to prevent choice overload

  WHY CONTENT-BASED (NOT COLLABORATIVE)?
    ✅ No historical interaction data required
    ✅ Cold-start resistant for new items
    ✅ Fully explainable (white-box)
    ✅ Works immediately on first use

  SIMILARITY METRICS:
    ❌ Euclidean Distance  → sensitive to vector magnitude
    ✅ Cosine Similarity   → measures ANGLE (orientation)
                            = invariant to magnitude
                            Score 1 = perfect match
                            Score 0 = no overlap

  FEATURE EXTRACTION:
    Binary (1/0) → O(1) lookup but loses nuance
    TF-IDF       → Weights specific/rare terms HIGHER
                   Penalises generic/common terms
                   Professional-grade standard
"""

# ──────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────

import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ──────────────────────────────────────────────────────────
# DATASET: JOB ROLES KNOWLEDGE BASE
# Each role has an associated list of relevant skills/tools.
# This IS the "item catalogue" of our recommendation engine.
# ──────────────────────────────────────────────────────────

JOB_ROLES: dict[str, list[str]] = {
    "Data Scientist": [
        "python", "sql", "machine learning", "statistics",
        "pandas", "numpy", "scikit-learn", "tensorflow",
        "data analysis", "data visualization", "jupyter",
        "deep learning", "feature engineering", "r",
    ],
    "Machine Learning Engineer": [
        "python", "machine learning", "deep learning", "tensorflow",
        "pytorch", "scikit-learn", "mlops", "docker", "kubernetes",
        "model deployment", "api", "feature engineering", "numpy",
        "data pipelines",
    ],
    "Data Analyst": [
        "sql", "excel", "data visualization", "python", "pandas",
        "tableau", "power bi", "statistics", "reporting",
        "data analysis", "business intelligence", "google sheets",
    ],
    "AI Research Scientist": [
        "python", "deep learning", "pytorch", "tensorflow",
        "mathematics", "linear algebra", "statistics",
        "natural language processing", "computer vision",
        "reinforcement learning", "research", "publication",
    ],
    "Backend Developer": [
        "python", "java", "sql", "api", "rest", "django",
        "fastapi", "databases", "docker", "git", "linux",
        "microservices", "postgresql", "mongodb",
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "ci/cd", "linux",
        "terraform", "ansible", "bash", "git", "jenkins",
        "monitoring", "cloud computing", "azure", "gcp",
    ],
    "Cloud Architect": [
        "aws", "azure", "gcp", "cloud computing", "terraform",
        "kubernetes", "docker", "networking", "security",
        "microservices", "serverless", "databases", "architecture",
    ],
    "Frontend Developer": [
        "javascript", "react", "html", "css", "typescript",
        "nextjs", "vue", "ui/ux", "figma", "responsive design",
        "git", "web development", "api",
    ],
    "Full Stack Developer": [
        "javascript", "python", "react", "nodejs", "sql",
        "html", "css", "api", "docker", "git",
        "databases", "rest", "web development",
    ],
    "Cybersecurity Engineer": [
        "networking", "linux", "python", "security",
        "ethical hacking", "penetration testing", "firewalls",
        "encryption", "siem", "risk assessment", "bash",
    ],
    "NLP Engineer": [
        "python", "natural language processing", "transformers",
        "bert", "gpt", "pytorch", "tensorflow", "spacy",
        "text classification", "sentiment analysis", "nlp",
        "machine learning", "huggingface",
    ],
    "Computer Vision Engineer": [
        "python", "computer vision", "opencv", "deep learning",
        "pytorch", "tensorflow", "convolutional neural networks",
        "image processing", "object detection", "yolo",
    ],
    "Data Engineer": [
        "python", "sql", "spark", "hadoop", "kafka",
        "airflow", "data pipelines", "etl", "aws", "docker",
        "databases", "cloud computing", "postgresql",
    ],
    "MLOps Engineer": [
        "python", "docker", "kubernetes", "mlops", "aws",
        "model deployment", "ci/cd", "airflow", "mlflow",
        "monitoring", "machine learning", "data pipelines",
    ],
    "Business Intelligence Analyst": [
        "sql", "tableau", "power bi", "data visualization",
        "statistics", "excel", "business intelligence",
        "reporting", "data analysis", "google sheets",
    ],
    "Robotics Engineer": [
        "python", "c++", "ros", "robotics", "embedded systems",
        "computer vision", "control systems", "mathematics",
        "sensor fusion", "automation",
    ],
    "Quantum Computing Researcher": [
        "python", "quantum computing", "qiskit", "mathematics",
        "linear algebra", "physics", "algorithms", "research",
        "optimization", "statistics",
    ],
    "AI Ethics Researcher": [
        "research", "machine learning", "statistics", "python",
        "fairness", "bias detection", "nlp", "policy",
        "philosophy", "social sciences",
    ],
}


# ──────────────────────────────────────────────────────────
# PURE PYTHON FALLBACK — Manual TF-IDF + Cosine Similarity
# (Used if scikit-learn is not installed)
# ──────────────────────────────────────────────────────────

class ManualTFIDF:
    """
    A from-scratch TF-IDF implementation for educational clarity.

    Formula:
      TF(t, d)  = count(t in d) / total_terms(d)
      IDF(t)    = log(N / df(t))          [N = total docs]
      TF-IDF    = TF × IDF

    The log in IDF acts as a dampening function —
    ensuring the penalty for high-frequency words
    scales logarithmically rather than linearly.
    """

    def __init__(self):
        self.vocabulary_: dict[str, int] = {}
        self.idf_: dict[str, float] = {}
        self._docs: list[list[str]] = []

    def fit(self, documents: list[list[str]]) -> "ManualTFIDF":
        self._docs = documents
        N = len(documents)

        # Build vocabulary
        all_terms = set(term for doc in documents for term in doc)
        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted(all_terms))}

        # Compute IDF
        for term in self.vocabulary_:
            df = sum(1 for doc in documents if term in doc)
            self.idf_[term] = math.log((N + 1) / (df + 1)) + 1   # smoothed
        return self

    def transform(self, documents: list[list[str]]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(self.vocabulary_)))
        for d_idx, doc in enumerate(documents):
            term_count = len(doc)
            if term_count == 0:
                continue
            freq = {}
            for term in doc:
                freq[term] = freq.get(term, 0) + 1
            for term, count in freq.items():
                if term in self.vocabulary_:
                    tf  = count / term_count
                    idf = self.idf_.get(term, 0)
                    col = self.vocabulary_[term]
                    matrix[d_idx, col] = tf * idf
        return matrix

    def fit_transform(self, documents: list[list[str]]) -> np.ndarray:
        return self.fit(documents).transform(documents)


def manual_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity: dot(A,B) / (||A|| × ||B||)"""
    dot    = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ──────────────────────────────────────────────────────────
# CORE ENGINE
# ──────────────────────────────────────────────────────────

class TechStackRecommender:
    """
    Content-Based Filtering Recommendation Engine.

    Workflow:
      1. Build TF-IDF matrix for all job roles (item catalogue)
      2. Accept user skill input → build user profile vector
      3. Compute cosine similarity: user_vector vs every item
      4. Sort by score → return Top-N

    Cold-Start handling:
      • User Cold Start  → Prompt user to provide at least 3 skills
      • Item Cold Start  → Content-based inherently handles this
                           (new items recommended immediately via metadata)
    """

    def __init__(self, top_n: int = 5):
        self.top_n     = top_n
        self.roles     = list(JOB_ROLES.keys())
        self.role_docs = list(JOB_ROLES.values())  # list of skill lists
        self.vectorizer  = None
        self.item_matrix = None
        self._fitted     = False

    def _build_corpus(self) -> None:
        """
        Converts skill lists into text 'documents' and fits
        the TF-IDF vectorizer on the entire item catalogue.

        This is done ONCE at startup and cached.
        """
        # Convert skill lists → single strings for sklearn TF-IDF
        corpus = [" ".join(skills) for skills in self.role_docs]

        if SKLEARN_AVAILABLE:
            self.vectorizer  = TfidfVectorizer(lowercase=True, stop_words=None)
            self.item_matrix = self.vectorizer.fit_transform(corpus).toarray()
        else:
            self.vectorizer  = ManualTFIDF()
            self.item_matrix = self.vectorizer.fit_transform(self.role_docs)

        self._fitted = True

    def fit(self) -> "TechStackRecommender":
        """Initialises the engine by building the TF-IDF item catalogue."""
        print("\n  🔧 Building TF-IDF catalogue for all job roles...")
        self._build_corpus()
        print(f"  ✅ Catalogue ready: {len(self.roles)} roles | "
              f"Vocabulary size: {len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 'N/A'}")
        return self

    def _build_user_vector(self, user_skills: list[str]) -> np.ndarray:
        """
        Projects user skills into the SAME TF-IDF vector space
        as the item catalogue.

        CRITICAL: Item features and user features MUST map to the
        exact same vocabulary. Naming discrepancies break the math.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommending.")

        if SKLEARN_AVAILABLE:
            user_doc = " ".join(s.lower().strip() for s in user_skills)
            user_vec = self.vectorizer.transform([user_doc]).toarray()[0]
        else:
            user_doc = [s.lower().strip() for s in user_skills]
            user_vec = self.vectorizer.transform([user_doc])[0]

        return user_vec

    def recommend(self, user_skills: list[str],
                  top_n: Optional[int] = None) -> list[dict]:
        """
        4-Step Ranking Pipeline:

        Step 1 — INGESTION  : Validate & vectorise user skills
        Step 2 — SCORING    : Cosine similarity for each role
        Step 3 — SORTING    : Descending order by score
        Step 4 — FILTERING  : Slice Top-N list

        Args:
            user_skills: List of skills the user knows / is interested in
            top_n      : Override default top_n value

        Returns:
            List of dicts: [{role, score, matching_skills, description}]
        """
        if top_n is None:
            top_n = self.top_n

        # ── Cold-Start Guard ──────────────────────────────
        if len(user_skills) < 3:
            raise ValueError(
                "⚠️  Cold Start Detected! Provide at least 3 skills "
                "to ensure sufficient data density for accurate matching."
            )

        # ── Step 1: Ingestion — build user vector ─────────
        user_vec = self._build_user_vector(user_skills)

        # ── Cold-Start check: zero vector ─────────────────
        if np.linalg.norm(user_vec) == 0:
            print("  ⚠️  Warning: None of your skills matched the vocabulary.")
            print("       Falling back to trending popular roles...")
            return self._trending_fallback(top_n)

        # ── Step 2: Scoring ───────────────────────────────
        scored = []
        clean_skills = [s.lower().strip() for s in user_skills]

        for idx, (role, role_skills) in enumerate(zip(self.roles, self.role_docs)):
            # Cosine similarity between user vector and item vector
            score = manual_cosine_similarity(user_vec, self.item_matrix[idx])

            # Identify which of the user's skills explicitly match this role
            matching = [s for s in clean_skills if s in role_skills]

            scored.append({
                "role"           : role,
                "score"          : round(score, 4),
                "match_percent"  : round(score * 100, 1),
                "matching_skills": matching,
                "total_role_skills": len(role_skills),
            })

        # ── Step 3: Sorting ───────────────────────────────
        scored.sort(key=lambda x: x["score"], reverse=True)

        # ── Step 4: Filtering (Top-N) ─────────────────────
        return scored[:top_n]

    def _trending_fallback(self, top_n: int) -> list[dict]:
        """
        Cold-Start bypass: Return globally popular roles when
        no vocabulary overlap exists (e.g., brand-new user).
        """
        trending = [
            "Data Scientist", "Full Stack Developer", "ML Engineer",
            "DevOps Engineer", "Backend Developer",
        ]
        return [{"role": r, "score": 0.0, "match_percent": 0.0,
                 "matching_skills": [], "total_role_skills": 0,
                 "note": "Trending fallback"} for r in trending[:top_n]]


# ──────────────────────────────────────────────────────────
# DISPLAY UTILITIES
# ──────────────────────────────────────────────────────────

def print_recommendations(recommendations: list[dict],
                           user_skills: list[str]) -> None:
    """Pretty-prints the Top-N recommendation list."""
    print("\n" + "=" * 65)
    print("  🎯  TOP RECOMMENDATIONS FOR YOUR SKILL PROFILE")
    print("=" * 65)
    print(f"\n  Your skills : {', '.join(user_skills)}\n")

    medals = ["🥇", "🥈", "🥉"] + ["🏅"] * 20

    for i, rec in enumerate(recommendations):
        medal    = medals[i] if i < len(medals) else f"#{i+1}"
        score    = rec["score"]
        pct      = rec["match_percent"]
        role     = rec["role"]
        matching = rec["matching_skills"]

        print(f"  {medal}  Rank {i+1}: {role}")
        print(f"       Cosine Similarity : {score:.4f}  ({pct:.1f}% match)")
        print(f"       Matching Skills   : {', '.join(matching) if matching else 'N/A (via TF-IDF weighting)'}")
        print()


def visualise_recommendations(recommendations: list[dict],
                               save_path: str = "recommendations.png") -> None:
    """
    Produces a horizontal bar chart of cosine similarity scores.
    Higher score = better match with user's profile.
    """
    roles  = [r["role"] for r in recommendations]
    scores = [r["score"] for r in recommendations]

    # Colour gradient: best match = darkest blue
    colours = [cm.Blues(0.4 + 0.6 * (1 - i / max(len(roles) - 1, 1)))
               for i in range(len(roles))]

    fig, ax = plt.subplots(figsize=(10, max(5, len(roles) * 0.8)))
    bars = ax.barh(roles[::-1], scores[::-1], color=colours[::-1],
                   edgecolor="navy", linewidth=0.7, height=0.6)

    # Add score labels on bars
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", ha="left",
                fontsize=10, fontweight="bold", color="#1a237e")

    ax.set_xlim(0, min(1.0, max(scores) + 0.15))
    ax.set_xlabel("Cosine Similarity Score  (1.0 = perfect match)", fontsize=12)
    ax.set_title("AI Tech Stack Recommender — Top Role Matches\n"
                 "Content-Based Filtering via TF-IDF + Cosine Similarity",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📸  Saved → {save_path}")


def export_results(recommendations: list[dict],
                   user_skills: list[str],
                   filepath: str = "recommendation_results.json") -> None:
    """Exports results to JSON for downstream use."""
    output = {
        "user_skills"    : user_skills,
        "recommendations": recommendations,
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  💾  Results exported → {filepath}")


# ──────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ──────────────────────────────────────────────────────────

def interactive_mode(engine: TechStackRecommender) -> None:
    """
    Runs an interactive terminal session where the user
    inputs their skills and receives real-time recommendations.

    Minimum 3 skills required (Cold-Start prevention).
    """
    print("\n" + "=" * 65)
    print("  🎮  INTERACTIVE MODE — Tech Stack Recommender")
    print("=" * 65)
    print("\n  Enter your skills/technologies (comma-separated).")
    print("  Example: python, machine learning, sql, docker")
    print("  Type 'exit' to quit.\n")

    while True:
        try:
            raw = input("  Your skills: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended. Goodbye!")
            break

        if raw.lower() in {"exit", "quit", "q"}:
            print("  Goodbye! Keep building your tech skills! 🚀")
            break

        skills = [s.strip() for s in raw.split(",") if s.strip()]

        if len(skills) < 3:
            print(f"  ⚠️  Please enter at least 3 skills (you entered {len(skills)}).\n")
            continue

        try:
            recs = engine.recommend(skills, top_n=5)
            print_recommendations(recs, skills)
            visualise_recommendations(recs, "interactive_recommendations.png")
        except ValueError as e:
            print(f"  ⚠️  {e}\n")


# ──────────────────────────────────────────────────────────
# DEMO — PREDEFINED SCENARIOS
# ──────────────────────────────────────────────────────────

DEMO_PROFILES: dict[str, list[str]] = {
    "Python + ML + Data": [
        "python", "machine learning", "sql", "pandas",
        "statistics", "data analysis", "scikit-learn",
    ],
    "Cloud + DevOps": [
        "docker", "kubernetes", "aws", "ci/cd", "terraform",
        "linux", "bash",
    ],
    "NLP + Research": [
        "python", "natural language processing", "pytorch",
        "transformers", "research", "statistics",
    ],
    "Full Stack Web": [
        "javascript", "react", "python", "sql", "api",
        "docker", "git",
    ],
    "Computer Vision": [
        "python", "computer vision", "opencv", "deep learning",
        "pytorch", "convolutional neural networks",
    ],
}


def run_demo(engine: TechStackRecommender) -> None:
    """Runs all predefined demo profiles."""
    print("\n" + "🎯 " * 21)
    print("  DEMO — Running 5 Predefined Skill Profiles")
    print("🎯 " * 21)

    for profile_name, skills in DEMO_PROFILES.items():
        print(f"\n{'─'*65}")
        print(f"  Profile: {profile_name}")
        print(f"{'─'*65}")
        recs = engine.recommend(skills, top_n=3)
        print_recommendations(recs, skills)

    # Full visualisation for first profile
    first_profile_skills = list(DEMO_PROFILES.values())[0]
    recs = engine.recommend(first_profile_skills, top_n=8)
    visualise_recommendations(recs, "demo_recommendations.png")
    export_results(recs, first_profile_skills)


# ──────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "🎯 " * 21)
    print("  PROJECT 3 — AI RECOMMENDATION LOGIC")
    print("  Tech Stack Recommender | DecodeLabs")
    print("🎯 " * 21)

    # Initialise and fit the engine
    engine = TechStackRecommender(top_n=5)
    engine.fit()

    # Run demo scenarios
    run_demo(engine)

    # Launch interactive mode
    interactive_mode(engine)


if __name__ == "__main__":
    main()
