
import re
import pandas as pd
from pathlib import Path
from nltk.stem.porter import PorterStemmer

# =====================================
# 1. PROJECT PATHS
# =====================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "unified_movies_enriched.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "movies_nlp_ready.csv"
)

# =====================================
# 2. LOAD DATASET
# =====================================

print("Loading enriched movie dataset...")

df = pd.read_csv(INPUT_FILE)

print("Original dataset shape:", df.shape)

# =====================================
# 3. HANDLE MISSING VALUES
# =====================================

text_columns = [
    "title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "director"
]

for column in text_columns:
    df[column] = df[column].fillna("").astype(str)

# =====================================
# 4. TEXT CLEANING
# =====================================

stemmer = PorterStemmer()

def clean_text(text):

    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================
# 5. COMBINE MOVIE FEATURES
# =====================================

def create_movie_tags(row):

    title = clean_text(row["title"])
    overview = clean_text(row["overview"])
    genres = clean_text(row["genres"])
    keywords = clean_text(row["keywords"])
    cast = clean_text(row["cast"])
    director = clean_text(row["director"])

    combined_text = (
        title + " " +
        overview + " " +
        genres + " " +
        keywords + " " +
        cast + " " +
        director
    )

    return combined_text


print("Combining movie information...")

df["tags"] = df.apply(create_movie_tags, axis=1)

# =====================================
# 6. APPLY STEMMING
# =====================================

def stem_text(text):

    words = text.split()

    stemmed_words = [
        stemmer.stem(word) for word in words
    ]

    return " ".join(stemmed_words)


print("Applying Porter Stemming...")

df["tags"] = df["tags"].apply(stem_text)

# =====================================
# 7. REMOVE EMPTY TAG RECORDS
# =====================================

df = df[df["tags"].str.strip() != ""].copy()

# =====================================
# 8. SAVE NLP READY DATASET
# =====================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== NLP PREPROCESSING COMPLETED ==========")

print("Final dataset shape:", df.shape)

print("Saved file:", OUTPUT_FILE)

print("\nSample movie tags:")

print(
    df[["title", "tags"]]
    .head(5)
    .to_string(index=False)
)
