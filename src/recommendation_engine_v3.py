
import os
import pickle

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "unified_movies_enriched.csv"
)

EMBEDDING_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "semantic_embeddings_v3.pkl"
)

print("Loading movie dataset...")

movies = pd.read_csv(DATA_PATH).fillna("")
movies.reset_index(drop=True, inplace=True)

print("Loading semantic embeddings...")

with open(EMBEDDING_PATH, "rb") as file:
    embeddings = pickle.load(file)

# Verify dataset and embedding alignment
if len(movies) != len(embeddings):
    raise ValueError(
        f"Mismatch: {len(movies)} movies but {len(embeddings)} embeddings."
    )

print("Movies loaded:", len(movies))
print("Embedding shape:", embeddings.shape)


def recommend_movies(movie_name, top_n=10):

    # Find movie by exact title
    matches = movies[
        movies["title"].str.strip().str.lower()
        == movie_name.strip().lower()
    ]

    if matches.empty:
        print(f"\nMovie '{movie_name}' not found in dataset.")
        return

    if len(matches) > 1:
        print("\nMultiple movies found with this title:")
        print(matches[["title", "industry", "release_date"]].to_string(index=False))
        print("\nUsing the first matching record.")

    movie_index = matches.index[0]

    # Calculate semantic similarity
    similarity_scores = cosine_similarity(
        embeddings[movie_index].reshape(1, -1),
        embeddings
    ).flatten()

    # Sort similarity scores
    similar_indices = similarity_scores.argsort()[::-1]

    # Exclude the selected movie
    similar_indices = [
        index for index in similar_indices
        if index != movie_index
    ]

    # Get top recommendations
    recommendations = movies.iloc[similar_indices[:top_n]].copy()

    recommendations["similarity_score"] = [
        round(float(similarity_scores[index]), 4)
        for index in similar_indices[:top_n]
    ]

    print("\nRecommended Movies:\n")

    print(
        recommendations[
            ["title", "industry", "genres", "similarity_score"]
        ].to_string(index=False)
    )


if __name__ == "__main__":

    print("\n===== 4Cinephiles V3 Semantic Recommendation System =====")

    movie_name = input("\nEnter a movie name: ")

    recommend_movies(movie_name)
