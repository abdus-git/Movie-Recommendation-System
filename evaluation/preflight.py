"""
Preflight checks before the evaluation. Read-only.

1. Do the saved model files really correspond to unified_movies_enriched.csv?
   (Row counts match, but that alone does not prove the rows line up.)
   We rebuild 20 random rows from the CSV and compare them with the stored
   TF-IDF V2 and MiniLM vectors.
2. Data-quality facts that affect recommendations.

Run from the project root:
    python evaluation/preflight.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "unified_movies_enriched.csv"
TFIDF = ROOT / "model" / "tfidf_matrix_v2.pkl"
VECTORIZERS = ROOT / "model" / "tfidf_vectorizer_v2.pkl"
SEMANTIC = ROOT / "model" / "semantic_embeddings_v3.pkl"

SAMPLE_SIZE = 20
TFIDF_TOLERANCE = 1e-6      # float rounding only
SEMANTIC_MIN_COSINE = 0.999  # float32 / batch-composition rounding only

# Same order and weights as tfidf_model_v2.py
FEATURES = ["overview", "keywords", "genres", "cast", "director", "title"]
WEIGHTS = {"overview": 3, "keywords": 3, "genres": 2,
           "cast": 1, "director": 1, "title": 1}

raw = pd.read_csv(DATA)
filled = raw.fillna("")

with open(TFIDF, "rb") as f:
    tfidf = pickle.load(f)
with open(SEMANTIC, "rb") as f:
    emb = pickle.load(f)

print("=" * 70, "\n1. MODEL / DATASET CONTENT CHECK\n" + "=" * 70)
print("CSV rows:", len(raw), "| TF-IDF rows:", tfidf.shape[0],
      "| embedding rows:", emb.shape[0])

if not (len(raw) == tfidf.shape[0] == emb.shape[0]):
    raise SystemExit("FAIL: row counts differ.")

rng = np.random.default_rng(0)
rows = np.sort(rng.choice(len(raw), SAMPLE_SIZE, replace=False))
print("Sample rows:", rows.tolist())

# ---- TF-IDF V2: rebuild the sampled rows with the saved vectorizers ----
tfidf_ok = False
try:
    with open(VECTORIZERS, "rb") as f:
        vectorizers = pickle.load(f)

    sample = raw.iloc[rows].copy()
    blocks = []
    for feature in FEATURES:
        text = sample[feature].fillna("").astype(str)
        blocks.append(vectorizers[feature].transform(text) * WEIGHTS[feature])

    rebuilt = hstack(blocks).tocsr()
    difference = abs(rebuilt - tfidf[rows]).max()
    tfidf_ok = difference < TFIDF_TOLERANCE
    print(f"\nTF-IDF V2 max absolute difference: {difference:.2e} "
          f"-> {'PASS' if tfidf_ok else 'FAIL'}")
except Exception as error:
    print("\nTF-IDF V2 check could not run:", repr(error))

# ---- Semantic V3: re-encode the sampled rows exactly as semantic_model_v3.py ----
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []
for _, m in filled.iloc[rows].iterrows():
    texts.append(
        f"Movie: {m['title']}\n"
        f"    Overview: {m['overview']}\n"
        f"    Genres: {m['genres']}\n"
        f"    Keywords: {m['keywords']}\n"
        f"    Cast: {m['cast']}\n"
        f"    Director: {m['director']}"
    )

new = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
cosines = (new * emb[rows]).sum(axis=1)
semantic_ok = cosines.min() > SEMANTIC_MIN_COSINE
print(f"Semantic V3 minimum cosine to stored vector: {cosines.min():.6f} "
      f"-> {'PASS' if semantic_ok else 'FAIL'}")

print("\nPREFLIGHT RESULT:",
      "PASS - models match this CSV" if (tfidf_ok and semantic_ok)
      else "CHECK FAILED - do not trust model/CSV alignment yet")

# ---------------------------------------------------------------
# 2. Data quality that affects recommendations
# ---------------------------------------------------------------
print("\n" + "=" * 70, "\n2. DATA QUALITY\n" + "=" * 70)

votes = pd.to_numeric(raw["vote_count"], errors="coerce")

print("Release status by industry:")
print(pd.crosstab(raw["industry"], raw["release_status"]))

print("\nShare of movies with 0 votes, by industry:")
print((votes == 0).groupby(raw["industry"]).agg(["sum", "mean"]).round(3))

print("\nMovies with no release date, by industry:")
print((filled["release_date"] == "").groupby(raw["industry"]).sum())
