
import pandas as pd
import pickle

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# =====================================
# 1. PROJECT PATHS
# =====================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT / "data" / "processed" / "movies_nlp_ready.csv"
)

MATRIX_FILE = ROOT / "model" / "tfidf_matrix.pkl"

# =====================================
# 2. LOAD DATA AND TF-IDF MATRIX
# =====================================

print("Loading movie dataset...")

movies = pd.read_csv(DATA_FILE)

print("Loading TF-IDF matrix...")

with open(MATRIX_FILE, "rb") as file:
    tfidf_matrix = pickle.load(file)

# =====================================
# 3. RECOMMENDATION FUNCTION
# =====================================

def recommend_movies(movie_title, top_n=10):

    # Find movie by title (case-insensitive)
    matches = movies[
        movies["title"].str.casefold()
        == movie_title.strip().casefold()
    ]

    if matches.empty:
        print(f"\nMovie '{movie_title}' not found!")
        return None

    # Use the first matching movie for now
    movie_index = matches.index[0]

    # Calculate similarity against all movies
    similarity_scores = cosine_similarity(
        tfidf_matrix[movie_index],
        tfidf_matrix
    ).flatten()

    # Sort by similarity score
    similar_movies = sorted(
        enumerate(similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the selected movie itself
    similar_movies = [
        item for item in similar_movies
        if item[0] != movie_index
    ]

    # Select top recommendations
    top_movies = similar_movies[:top_n]

    results = []

    for index, score in top_movies:

        results.append({
            "title": movies.iloc[index]["title"],
            "industry": movies.iloc[index]["industry"],
            "genres": movies.iloc[index]["genres"],
            "similarity_score": round(float(score), 4)
        })

    return pd.DataFrame(results)


# =====================================
# 4. TEST RECOMMENDATION ENGINE
# =====================================

if __name__ == "__main__":

    print("\n========== 4CINEPHILES RECOMMENDATION ENGINE ==========")

    movie_name = input("\nEnter a movie name: ")

    recommendations = recommend_movies(movie_name)

    if recommendations is not None:

        print("\nRecommended Movies:\n")

        print(
            recommendations.to_string(index=False)
        )
