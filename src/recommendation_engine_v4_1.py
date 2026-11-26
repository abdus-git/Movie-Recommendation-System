import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------
# 1. Project paths
# -----------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data", "processed", "unified_movies_enriched.csv"
)

TFIDF_PATH = os.path.join(
    PROJECT_ROOT, "model", "tfidf_matrix_v2.pkl"
)

SEMANTIC_PATH = os.path.join(
    PROJECT_ROOT, "model", "semantic_embeddings_v3.pkl"
)


# -----------------------------------
# 2. Load dataset and models
# -----------------------------------

print("Loading movie dataset...")

movies = pd.read_csv(DATA_PATH).fillna("")
movies.reset_index(drop=True, inplace=True)

with open(TFIDF_PATH, "rb") as file:
    tfidf_matrix = pickle.load(file)

with open(SEMANTIC_PATH, "rb") as file:
    semantic_embeddings = pickle.load(file)


# -----------------------------------
# 3. Validate alignment
# -----------------------------------

if not (
    len(movies)
    == tfidf_matrix.shape[0]
    == semantic_embeddings.shape[0]
):
    raise ValueError("Dataset and model dimensions do not match!")

required_columns = [
    "title", "overview", "genres", "keywords",
    "cast", "director", "industry"
]

for column in required_columns:
    if column not in movies.columns:
        raise ValueError(f"Missing required column: {column}")

print("\nV4.1 models loaded successfully!")
print("Total movies:", len(movies))


# -----------------------------------
# 4. Helper functions
# -----------------------------------

def normalize_title(title):
    return re.sub(
        r"\s+", " ", str(title).strip().lower()
    )


def get_genres(value):
    return {
        genre.strip().lower()
        for genre in str(value).split()
        if genre.strip()
    }


def metadata_completeness(movie):
    fields = [
        "overview", "genres", "keywords",
        "cast", "director"
    ]

    available = sum(
        bool(str(movie[field]).strip())
        for field in fields
    )

    return available / len(fields)


def calculate_genre_score(source_genres, candidate_genres):
    if not source_genres or not candidate_genres:
        return 0.5  # Neutral when genre data is unavailable

    intersection = len(source_genres & candidate_genres)
    union = len(source_genres | candidate_genres)

    return intersection / union if union else 0.5


def directly_references_title(source_title, candidate_overview):
    source_title = str(source_title).strip().lower()
    candidate_overview = str(candidate_overview).lower()

    if len(source_title) < 4:
        return False

    pattern = r"(?<!\w)" + re.escape(source_title) + r"(?!\w)"

    return bool(re.search(pattern, candidate_overview))


# -----------------------------------
# 5. V4.1 recommendation engine
# -----------------------------------

def recommend_movies(movie_name, top_n=10):

    matches = movies[
        movies["title"].apply(normalize_title)
        == normalize_title(movie_name)
    ]

    if matches.empty:
        print(f"\nMovie '{movie_name}' not found.")
        return

    if len(matches) > 1:
        print("\nMultiple matching movies found:")

        columns = [
            col for col in [
                "title", "industry", "release_date", "movie_id"
            ] if col in matches.columns
        ]

        print(matches[columns].to_string(index=False))
        print("\nUsing the first matching record for this experiment.")

    movie_index = matches.index[0]
    source_movie = movies.iloc[movie_index]

    source_genres = get_genres(source_movie["genres"])

    # Calculate similarities
    tfidf_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    semantic_scores = cosine_similarity(
        semantic_embeddings[movie_index].reshape(1, -1),
        semantic_embeddings
    ).flatten()

    experiments = {
        "70% V2 + 30% V3": (0.70, 0.30),
        "50% V2 + 50% V3": (0.50, 0.50),
        "30% V2 + 70% V3": (0.30, 0.70)
    }

    valid_indices = [
        i for i in range(len(movies))
        if i != movie_index
    ]

    # -----------------------------------
    # Run experiments
    # -----------------------------------

    for experiment_name, (v2_weight, v3_weight) in experiments.items():

        hybrid_scores = (
            v2_weight * tfidf_scores
            + v3_weight * semantic_scores
        )

        adjusted_scores = hybrid_scores.copy()

        for i in valid_indices:

            candidate = movies.iloc[i]

            candidate_genres = get_genres(
                candidate["genres"]
            )

            genre_score = calculate_genre_score(
                source_genres,
                candidate_genres
            )

            # Modest genre adjustment
            adjusted_scores[i] = (
                0.85 * hybrid_scores[i]
                + 0.15 * genre_score
            )

            # Penalize direct title references.
            # Keep the movie in the dataset.
            if directly_references_title(
                source_movie["title"],
                candidate["overview"]
            ):
                adjusted_scores[i] *= 0.85

        # Rank candidates
        ranked_indices = sorted(
            valid_indices,
            key=lambda i: adjusted_scores[i],
            reverse=True
        )

        # Remove duplicate titles from displayed results
        selected_indices = []
        seen_titles = set()

        for i in ranked_indices:

            title_key = normalize_title(
                movies.iloc[i]["title"]
            )

            if not title_key or title_key in seen_titles:
                continue

            seen_titles.add(title_key)
            selected_indices.append(i)

            if len(selected_indices) == top_n:
                break

        recommendations = movies.iloc[
            selected_indices
        ].copy()

        recommendations["similarity_score"] = [
            round(float(adjusted_scores[i]), 4)
            for i in selected_indices
        ]

        recommendations["metadata_completeness"] = [
            round(metadata_completeness(movies.iloc[i]) * 100, 1)
            for i in selected_indices
        ]

        recommendations["title_reference"] = [
            directly_references_title(
                source_movie["title"],
                movies.iloc[i]["overview"]
            )
            for i in selected_indices
        ]

        print("\n" + "=" * 85)
        print(experiment_name)
        print("=" * 85)

        print(
            recommendations[
                [
                    "title",
                    "industry",
                    "genres",
                    "similarity_score",
                    "metadata_completeness",
                    "title_reference"
                ]
            ].to_string(index=False)
        )


# -----------------------------------
# 6. Run the system
# -----------------------------------

if __name__ == "__main__":

    print("\n===== 4Cinephiles V4.1 Recommendation System =====")

    movie_name = input("\nEnter a movie name: ")

    recommend_movies(movie_name)
