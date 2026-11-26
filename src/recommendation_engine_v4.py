import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# 1. Project paths
# -----------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "unified_movies_enriched.csv"
)

TFIDF_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "tfidf_matrix_v2.pkl"
)

SEMANTIC_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "semantic_embeddings_v3.pkl"
)


# -----------------------------------
# 2. Load dataset and models
# -----------------------------------

print("Loading movie dataset...")

movies = pd.read_csv(DATA_PATH).fillna("")
movies.reset_index(drop=True, inplace=True)

print("Loading TF-IDF V2 matrix...")

with open(TFIDF_PATH, "rb") as file:
    tfidf_matrix = pickle.load(file)

print("Loading semantic V3 embeddings...")

with open(SEMANTIC_PATH, "rb") as file:
    semantic_embeddings = pickle.load(file)


# -----------------------------------
# 3. Validate model alignment
# -----------------------------------

if not (
    len(movies)
    == tfidf_matrix.shape[0]
    == semantic_embeddings.shape[0]
):
    raise ValueError("Dataset and model dimensions do not match!")

print("\nAll models loaded successfully!")
print("Movies:", len(movies))
print("TF-IDF matrix:", tfidf_matrix.shape)
print("Semantic embeddings:", semantic_embeddings.shape)


# -----------------------------------
# 4. Hybrid recommendation function
# -----------------------------------

def recommend_movies(movie_name, top_n=10):

    matches = movies[
        movies["title"].str.strip().str.lower()
        == movie_name.strip().lower()
    ]

    if matches.empty:
        print(f"\nMovie '{movie_name}' not found in dataset.")
        return

    if len(matches) > 1:
        print("\nMultiple movies found. Using the first matching record.")

    movie_index = matches.index[0]

    # Calculate V2 similarity
    tfidf_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Calculate V3 similarity
    semantic_scores = cosine_similarity(
        semantic_embeddings[movie_index].reshape(1, -1),
        semantic_embeddings
    ).flatten()

    # Test different hybrid weight combinations
    experiments = {
        "70% V2 + 30% V3": (0.70, 0.30),
        "50% V2 + 50% V3": (0.50, 0.50),
        "30% V2 + 70% V3": (0.30, 0.70)
    }

    # Exclude the selected movie
    valid_indices = np.array([
        i for i in range(len(movies))
        if i != movie_index
    ])

    for experiment_name, (v2_weight, v3_weight) in experiments.items():

        hybrid_scores = (
            v2_weight * tfidf_scores
            + v3_weight * semantic_scores
        )

        top_indices = valid_indices[
            np.argsort(hybrid_scores[valid_indices])[::-1][:top_n]
        ]

        recommendations = movies.iloc[top_indices].copy()

        recommendations["similarity_score"] = [
            round(float(hybrid_scores[i]), 4)
            for i in top_indices
        ]

        print("\n" + "=" * 75)
        print(experiment_name)
        print("=" * 75)

        print(
            recommendations[
                ["title", "industry", "genres", "similarity_score"]
            ].to_string(index=False)
        )


# -----------------------------------
# 5. Run the recommendation system
# -----------------------------------

if __name__ == "__main__":

    print("\n===== 4Cinephiles V4 Hybrid Recommendation System =====")

    movie_name = input("\nEnter a movie name: ")

    recommend_movies(movie_name)
