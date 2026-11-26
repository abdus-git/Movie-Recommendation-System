
import pandas as pd
import pickle

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================
# 1. PROJECT PATHS
# =====================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "movies_nlp_ready.csv"
)

MODEL_DIR = ROOT / "model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"
MATRIX_FILE = MODEL_DIR / "tfidf_matrix.pkl"

# =====================================
# 2. LOAD NLP DATASET
# =====================================

print("Loading NLP-ready dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

# =====================================
# 3. INITIALIZE TF-IDF
# =====================================

print("\nInitializing TF-IDF Vectorizer...")

tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

# =====================================
# 4. CONVERT TEXT INTO VECTORS
# =====================================

print("\nCreating TF-IDF matrix...")

tfidf_matrix = tfidf.fit_transform(
    df["tags"].fillna("")
)

# =====================================
# 5. SAVE MODEL AND MATRIX
# =====================================

print("\nSaving TF-IDF model...")

with open(VECTORIZER_FILE, "wb") as file:
    pickle.dump(tfidf, file)

with open(MATRIX_FILE, "wb") as file:
    pickle.dump(tfidf_matrix, file)

# =====================================
# 6. DISPLAY RESULTS
# =====================================

print("\n========== TF-IDF COMPLETED ==========")

print("Number of movies:", tfidf_matrix.shape[0])

print("Number of features:", tfidf_matrix.shape[1])

print("Matrix shape:", tfidf_matrix.shape)

print("Matrix type:", type(tfidf_matrix))

print("\nSaved files:")
print(VECTORIZER_FILE)
print(MATRIX_FILE)
