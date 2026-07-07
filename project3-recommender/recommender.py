"""
Project 3: AI Recommendation Logic - Tech Stack Recommender
DecodeLabs Industrial Training Kit

Content-based filtering: map a user's skills to the closest job roles
using TF-IDF vectorization + Cosine Similarity (angle, not distance).
"""

from pathlib import Path

import pandas as pd
from scipy.sparse._matrix import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- INPUT ----------
# Load CSV relative to this script so the module is cwd-agnostic
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "raw_skills.csv")
print(f"Loaded {len(df)} job roles.\n")


def recommend(user_skills: list, top_n: int = 3) -> pd.DataFrame:
    """
    user_skills: list of skill strings, e.g. ["Python", "Cloud Computing", "Automation"]
    Returns the top_n closest job roles ranked by cosine similarity.
    """
    user_profile = " ".join(user_skills)

    # ---------- PROCESS ----------
    # Shared vocabulary space: fit TF-IDF on job-role skills + the user profile together
    corpus = df["skills"].tolist() + [user_profile]
    vectorizer = TfidfVectorizer()
    tfidf_matrix: spmatrix = vectorizer.fit_transform(corpus)

    user_vector = tfidf_matrix[-1]         # type: ignore # last row = user profile
    role_vectors = tfidf_matrix[:-1]       # type: ignore # everything else = job roles

    scores = cosine_similarity(user_vector, role_vectors).flatten()

    # ---------- OUTPUT ----------
    result = df.copy()
    result["match_score"] = scores
    result = result.sort_values("match_score", ascending=False).head(top_n)
    result["match_score"] = (result["match_score"] * 100).round(1).astype(str) + "%"
    return result[["role", "skills", "match_score"]]


if __name__ == "__main__":
    # Minimum 3 user inputs, per spec
    test_skills = ["Python", "Cloud Computing", "Automation"]
    print(f"User skills: {test_skills}\n")

    top_matches = recommend(test_skills, top_n=3)
    print("--- Top 3 Recommended Career Paths ---")
    print(top_matches.to_string(index=False))
