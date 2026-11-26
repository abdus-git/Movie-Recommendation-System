
import pandas as pd
import pickle

from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================
# 1. PROJECT PATHS
# =====================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "unified_movies_enriched.csv"
)

MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_FILE = MODEL_DIR / "tfidf_vectorizer_v2.pkl"
MATRIX_FILE = MODEL_DIR / "tfidf_matrix_v2.pkl"

# =====================================
# 2. LOAD DATASET
# =====================================

print("Loading enriched movie dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# =====================================
# 3. SELECT MOVIE FEATURES
# =====================================

features = [
    "overview",
    "keywords",
    "genres",
    "cast",
    "director",
    "title"
]

for column in features:
    df[column] = df[column].fillna("").astype(str)

# =====================================
# 4. DEFINE FEATURE WEIGHTS
# =====================================

weights = {
    "overview": 3,
    "keywords": 3,
    "genres": 2,
    "cast": 1,
    "director": 1,
    "title": 1
}

# =====================================
# 5. CREATE SEPARATE TF-IDF MATRICES
# =====================================

print("\nCreating separate TF-IDF matrices...")

vectorizers = {}
weighted_matrices = []

for feature in features:

    print(f"Processing: {feature}")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2)
    )

    feature_matrix = vectorizer.fit_transform(df[feature])

    weighted_matrix = feature_matrix * weights[feature]

    vectorizers[feature] = vectorizer

    weighted_matrices.append(weighted_matrix)

    print("  Matrix shape:", feature_matrix.shape)
    print("  Weight:", weights[feature])

# =====================================
# 6. COMBINE ALL MATRICES
# =====================================

print("\nCombining weighted feature matrices...")

tfidf_matrix = hstack(weighted_matrices).tocsr()

# =====================================
# 7. SAVE MODEL AND MATRIX
# =====================================

print("\nSaving V2 model...")

with open(VECTOR_FILE, "wb") as file:
    pickle.dump(vectorizers, file)

with open(MATRIX_FILE, "wb") as file:
    pickle.dump(tfidf_matrix, file)

# =====================================
# 8. FINAL REPORT
# =====================================

print("\n========== TF-IDF V2 COMPLETED ==========")

print("Number of movies:", tfidf_matrix.shape[0])
print("Total features:", tfidf_matrix.shape[1])
print("Matrix shape:", tfidf_matrix.shape)

print("\nFeature weights:")

for feature, weight in weights.items():
    print(f"{feature}: {weight}")

print("\nSaved files:")
print(VECTOR_FILE)
print(MATRIX_FILE)
