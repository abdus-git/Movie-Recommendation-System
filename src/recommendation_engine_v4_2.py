import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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


# ==========================================
# 2. CONFIGURATION
# ==========================================

# Starting hybrid weights.
# These are experimental settings, not proven optimal weights.

V2_WEIGHT = 0.50
V3_WEIGHT = 0.50

# Genre adjustment strength.
GENRE_ADJUSTMENT = 0.20

TOP_N = 10

METADATA_FIELDS = [
    "overview",
    "genres",
    "keywords",
    "cast",
    "director"
]


# ==========================================
# 3. LOAD DATASET AND MODELS
# ==========================================

print("\nLoading 4Cinephiles V4.2...")

movies = pd.read_csv(DATA_PATH).fillna("")
movies.reset_index(drop=True, inplace=True)

with open(TFIDF_PATH, "rb") as file:
    tfidf_matrix = pickle.load(file)

with open(SEMANTIC_PATH, "rb") as file:
    semantic_embeddings = pickle.load(file)


# ==========================================
# 4. VALIDATE DATA
# ==========================================

required_columns = [
    "title",
    "industry",
    "genres",
    "overview",
    "keywords",
    "cast",
    "director"
]

for column in required_columns:
    if column not in movies.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

if not (
    len(movies)
    == tfidf_matrix.shape[0]
    == semantic_embeddings.shape[0]
):
    raise ValueError(
        "Dataset and model dimensions do not match!"
    )

print("Models loaded successfully!")
print("Total movies:", len(movies))
print("TF-IDF shape:", tfidf_matrix.shape)
print("Semantic shape:", semantic_embeddings.shape)


# ==========================================
# 5. HELPER FUNCTIONS
# ==========================================

def normalize_title(title):

    return re.sub(
        r"\s+",
        " ",
        str(title).strip().lower()
    )


def get_genres(value):

    return {
        genre.strip().lower()
        for genre in re.split(r"[,|;/\s]+", str(value))
        if genre.strip()
    }


def get_metadata_completeness(movie):

    available = sum(
        bool(str(movie[field]).strip())
        for field in METADATA_FIELDS
    )

    percentage = (
        available / len(METADATA_FIELDS)
    ) * 100

    return round(percentage, 1)


def get_genre_similarity(source_genres, candidate_genres):

    if not source_genres or not candidate_genres:
        return None

    intersection = source_genres & candidate_genres
    union = source_genres | candidate_genres

    return len(intersection) / len(union)


def get_shared_genres(source_genres, candidate_genres):

    return sorted(source_genres & candidate_genres)


def directly_references_title(source_title, candidate_overview):

    source_title = str(source_title).strip().lower()
    candidate_overview = str(candidate_overview).lower()

    if len(source_title) < 4:
        return False

    pattern = (
        r"(?<!\w)"
        + re.escape(source_title)
        + r"(?!\w)"
    )

    return bool(
        re.search(pattern, candidate_overview)
    )


def select_source_movie(movie_name):

    matches = movies[
        movies["title"].apply(normalize_title)
        == normalize_title(movie_name)
    ]

    if matches.empty:
        print(
            f"\nMovie '{movie_name}' not found in dataset."
        )
        return None

    if len(matches) == 1:
        return matches.index[0]

    print("\nMultiple movies found. Select one:\n")

    for number, (index, movie) in enumerate(
        matches.iterrows(), start=1
    ):

        release_date = movie.get("release_date", "")
        movie_id = movie.get("movie_id", "")

        print(
            f"{number}. {movie['title']} | "
            f"{movie['industry']} | "
            f"{release_date} | "
            f"{movie_id}"
        )

    while True:

        choice = input(
            "\nEnter movie number (default 1): "
        ).strip()

        if choice == "":
            return matches.index[0]

        if choice.isdigit():

            choice = int(choice)

            if 1 <= choice <= len(matches):
                return matches.index[choice - 1]

        print("Invalid selection. Try again.")


# ==========================================
# 6. RECOMMENDATION ENGINE V4.2
# ==========================================

def recommend_movies(movie_name, top_n=TOP_N):

    movie_index = select_source_movie(movie_name)

    if movie_index is None:
        return

    source_movie = movies.iloc[movie_index]

    source_title = source_movie["title"]

    source_genres = get_genres(
        source_movie["genres"]
    )

    print("\n" + "=" * 75)
    print("SELECTED MOVIE")
    print("=" * 75)

    print("Title:", source_title)
    print("Industry:", source_movie["industry"])
    print("Genres:", source_movie["genres"])

    # --------------------------------------
    # A. TF-IDF V2 similarity
    # --------------------------------------

    tfidf_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # --------------------------------------
    # B. Semantic V3 similarity
    # --------------------------------------

    semantic_scores = cosine_similarity(
        semantic_embeddings[movie_index].reshape(1, -1),
        semantic_embeddings
    ).flatten()

    # --------------------------------------
    # C. Hybrid similarity
    # --------------------------------------

    hybrid_scores = (
        V2_WEIGHT * tfidf_scores
        + V3_WEIGHT * semantic_scores
    )

    # --------------------------------------
    # D. Exclude selected title
    # --------------------------------------

    source_title_key = normalize_title(source_title)

    valid_indices = [
        i for i in range(len(movies))
        if normalize_title(movies.iloc[i]["title"])
        != source_title_key
    ]

    # --------------------------------------
    # E. Genre-aware score adjustment
    # --------------------------------------

    final_scores = hybrid_scores.copy()

    for i in valid_indices:

        candidate = movies.iloc[i]

        candidate_genres = get_genres(
            candidate["genres"]
        )

        genre_score = get_genre_similarity(
            source_genres,
            candidate_genres
        )

        # If genre metadata is missing,
        # do not apply a genre adjustment.

        if genre_score is not None:

            genre_factor = (
                1 - GENRE_ADJUSTMENT
                + GENRE_ADJUSTMENT * genre_score
            )

            final_scores[i] = (
                hybrid_scores[i] * genre_factor
            )

    # --------------------------------------
    # F. Rank recommendations
    # --------------------------------------

    ranked_indices = sorted(
        valid_indices,
        key=lambda i: final_scores[i],
        reverse=True
    )

    # --------------------------------------
    # G. Remove duplicate titles
    # --------------------------------------

    selected_indices = []
    seen_titles = set()

    for i in ranked_indices:

        candidate = movies.iloc[i]

        title_key = normalize_title(
            candidate["title"]
        )

        if not title_key or title_key in seen_titles:
            continue

        seen_titles.add(title_key)
        selected_indices.append(i)

        if len(selected_indices) >= top_n:
            break

    # --------------------------------------
    # H. Prepare recommendation output
    # --------------------------------------

    results = []

    for i in selected_indices:

        candidate = movies.iloc[i]

        candidate_genres = get_genres(
            candidate["genres"]
        )

        shared_genres = get_shared_genres(
            source_genres,
            candidate_genres
        )

        genre_score = get_genre_similarity(
            source_genres,
            candidate_genres
        )

        if shared_genres:

            reason = (
                "Shared genres: "
                + ", ".join(shared_genres)
            )

        else:

            reason = "Content and semantic similarity"

        reference_flag = directly_references_title(
            source_title,
            candidate["overview"]
        )

        results.append({
            "title": candidate["title"],
            "industry": candidate["industry"],
            "genres": candidate["genres"],
            "similarity_score": round(
                float(final_scores[i]), 4
            ),
            "genre_similarity": (
                round(float(genre_score), 3)
                if genre_score is not None
                else "N/A"
            ),
            "metadata_completeness": (
                get_metadata_completeness(candidate)
            ),
            "title_reference": reference_flag,
            "recommendation_reason": reason
        })

    recommendations = pd.DataFrame(results)

    # --------------------------------------
    # I. Display recommendations
    # --------------------------------------

    print("\n" + "=" * 95)
    print("4CINEPHILES V4.2 RECOMMENDATIONS")
    print("=" * 95)

    print(
        recommendations.to_string(index=False)
    )

    print("\n" + "=" * 95)
    print("RECOMMENDATION SUMMARY")
    print("=" * 95)

    print("Selected movie:", source_title)
    print("Recommendations generated:", len(recommendations))
    print("V2 weight:", V2_WEIGHT)
    print("V3 weight:", V3_WEIGHT)

    print(
        "\nNote: Similarity scores are ranking signals, "
        "not probabilities of user preference."
    )


# ==========================================
# 7. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("4CINEPHILES - V4.2 HYBRID RECOMMENDATION SYSTEM")
    print("=" * 65)

    movie_name = input(
        "\nEnter a movie name: "
    )

    recommend_movies(movie_name)
