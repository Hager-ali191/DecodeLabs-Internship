"""
Generates all visualisation assets for Project 3 submission.
Run once after recommender.py to produce polished charts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from recommender import TechStackRecommender, DEMO_PROFILES

engine = TechStackRecommender(top_n=5).fit()

# ── 1. Individual bar charts for every demo profile ────────
PALETTE = {
    "Python + ML + Data" : "#1565C0",
    "Cloud + DevOps"     : "#2E7D32",
    "NLP + Research"     : "#6A1B9A",
    "Full Stack Web"     : "#E65100",
    "Computer Vision"    : "#AD1457",
}

for profile_name, skills in DEMO_PROFILES.items():
    recs   = engine.recommend(skills, top_n=5)
    roles  = [r["role"]  for r in recs]
    scores = [r["score"] for r in recs]
    color  = PALETTE.get(profile_name, "#37474F")

    shades = [plt.matplotlib.colors.to_rgba(color, alpha=0.4 + 0.6*(i/(max(len(roles)-1,1))))
              for i in range(len(roles)-1, -1, -1)]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(roles[::-1], scores[::-1], color=shades,
                   edgecolor="white", linewidth=1.2, height=0.55)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.008,
                bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", ha="left",
                fontsize=10.5, fontweight="bold",
                color=color)

    ax.set_xlim(0, min(1.0, max(scores) + 0.22))
    ax.set_xlabel("Cosine Similarity Score  (1.0 = perfect match)", fontsize=11)
    ax.set_title(f"Profile: {profile_name}\nTop 5 Recommended Career Paths",
                 fontsize=13, fontweight="bold", color="#212121", pad=14)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.axvline(0.5, color="#BDBDBD", linestyle="--", linewidth=1, label="0.5 threshold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    plt.tight_layout()

    fname = profile_name.lower().replace(" + ","_").replace(" ","_") + ".png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📸 Saved → {fname}")


# ── 2. All-profiles heatmap comparison ─────────────────────
all_roles  = list(engine.roles)
score_mat  = np.zeros((len(DEMO_PROFILES), len(all_roles)))

for pi, (pname, skills) in enumerate(DEMO_PROFILES.items()):
    recs_full = engine.recommend(skills, top_n=len(all_roles))
    role_scores = {r["role"]: r["score"] for r in recs_full}
    for ri, role in enumerate(all_roles):
        score_mat[pi, ri] = role_scores.get(role, 0.0)

fig, ax = plt.subplots(figsize=(18, 5))
im = ax.imshow(score_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(all_roles)))
ax.set_xticklabels(all_roles, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(DEMO_PROFILES)))
ax.set_yticklabels(list(DEMO_PROFILES.keys()), fontsize=10)

for pi in range(len(DEMO_PROFILES)):
    for ri in range(len(all_roles)):
        val = score_mat[pi, ri]
        if val > 0.05:
            ax.text(ri, pi, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color="black" if val < 0.6 else "white",
                    fontweight="bold")

plt.colorbar(im, ax=ax, label="Cosine Similarity Score", shrink=0.8)
ax.set_title("Skill Profile × Job Role Similarity Heatmap\nAll 5 Demo Profiles vs 18 Career Paths",
             fontsize=13, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("heatmap_all_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  📸 Saved → heatmap_all_profiles.png")


# ── 3. Cosine geometry explainer ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#FAFAFA")

# Left: 2D vector illustration
ax = axes[0]
ax.set_facecolor("#F5F5F5")
origin = [0, 0]
user_vec  = np.array([0.82, 0.57])
match_vec = np.array([0.75, 0.66])
weak_vec  = np.array([0.20, 0.98])

for vec, color, label, lw in [
    (user_vec,  "#1565C0", "User Profile",       2.5),
    (match_vec, "#2E7D32", "Strong Match (0.98)",2.0),
    (weak_vec,  "#C62828", "Weak Match (0.61)",  2.0),
]:
    ax.annotate("", xy=vec, xytext=origin,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    ax.text(vec[0]+0.02, vec[1]+0.02, label, color=color,
            fontsize=9, fontweight="bold")

theta_strong = np.linspace(
    np.arctan2(user_vec[1], user_vec[0]),
    np.arctan2(match_vec[1], match_vec[0]), 30)
theta_weak = np.linspace(
    np.arctan2(user_vec[1], user_vec[0]),
    np.arctan2(weak_vec[1], weak_vec[0]), 30)

ax.plot(0.25*np.cos(theta_strong), 0.25*np.sin(theta_strong), color="#2E7D32", lw=1.5, ls="--")
ax.plot(0.35*np.cos(theta_weak),   0.35*np.sin(theta_weak),   color="#C62828", lw=1.5, ls="--")
ax.text(0.22, 0.10, "small θ", color="#2E7D32", fontsize=8)
ax.text(0.28, 0.35, "large θ", color="#C62828", fontsize=8)

ax.set_xlim(-0.05, 1.1); ax.set_ylim(-0.05, 1.15)
ax.set_xlabel("Feature Dimension 1", fontsize=10)
ax.set_ylabel("Feature Dimension 2", fontsize=10)
ax.set_title("Cosine Similarity:\nAngle Between Vectors", fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.3)

# Right: score scale
ax2 = axes[1]
ax2.set_facecolor("#F5F5F5")
labels = ["0.0\n(No match)", "0.25\n(Weak)", "0.50\n(Moderate)", "0.75\n(Strong)", "1.0\n(Perfect)"]
xvals  = [0.0, 0.25, 0.50, 0.75, 1.0]
colors = ["#C62828","#EF6C00","#F9A825","#558B2F","#1B5E20"]

gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax2.imshow(gradient, extent=[0, 1, -0.1, 0.1], aspect="auto",
           cmap="RdYlGn", alpha=0.85)

for xv, lb, co in zip(xvals, labels, colors):
    ax2.axvline(xv, color=co, lw=2, ymin=0.2, ymax=0.8)
    ax2.text(xv, 0.22, lb, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=co)

# Plot actual demo scores
demo_colors = list(PALETTE.values())
for pi, (pname, skills) in enumerate(list(DEMO_PROFILES.items())[:3]):
    recs  = engine.recommend(skills, top_n=1)
    score = recs[0]["score"]
    ax2.scatter(score, -0.06 + pi*0.03, color=demo_colors[pi],
                s=80, zorder=5, label=f"{pname.split('+')[0].strip()} ({score:.2f})")

ax2.set_xlim(-0.05, 1.1)
ax2.set_ylim(-0.2, 0.45)
ax2.axis("off")
ax2.set_title("Score Interpretation Scale\n(Top match of 3 demo profiles shown)",
              fontsize=11, fontweight="bold")
ax2.legend(loc="lower right", fontsize=8, framealpha=0.9)

plt.suptitle("Understanding Cosine Similarity in Recommendation Systems",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("cosine_explainer.png", dpi=150, bbox_inches="tight")
plt.close()
print("  📸 Saved → cosine_explainer.png")

print("\n  ✅ All visualisation assets generated.")
