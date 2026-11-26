
import os
import pickle

import pandas as pd
from sentence_transformers import SentenceTransformer

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "unified_movies_enriched.csv"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
EMBEDDING_PATH = os.path.join(MODEL_DIR, "semantic_embeddings_v3.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading movie dataset...")

movies = pd.read_csv(DATA_PATH).fillna("")

print("Dataset shape:", movies.shape)

# Combine meaningful movie information
print("\nPreparing movie descriptions...")

text_columns = [
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "director"
]

for column in text_columns:
    if column not in movies.columns:
        raise ValueError(f"Missing required column: {column}")

movie_descriptions = []

for _, movie in movies.iterrows():
    description = f"""
    Movie: {movie['title']}
    Overview: {movie['overview']}
    Genres: {movie['genres']}
    Keywords: {movie['keywords']}
    Cast: {movie['cast']}
    Director: {movie['director']}
    """

    movie_descriptions.append(description.strip())

print("Total descriptions:", len(movie_descriptions))

# Load Sentence Transformer
print("\nLoading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("\nGenerating semantic embeddings...")

embeddings = model.encode(
    movie_descriptions,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\nEmbedding generation completed!")

print("Embedding shape:", embeddings.shape)

# Save embeddings
print("\nSaving embeddings...")

with open(EMBEDDING_PATH, "wb") as file:
    pickle.dump(embeddings, file)

print("Saved at:", EMBEDDING_PATH)

print("\nSEMANTIC MODEL V3 COMPLETED")
