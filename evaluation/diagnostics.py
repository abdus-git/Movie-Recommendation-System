"""
Read-only diagnostics for 4Cinephiles evaluation preparation.
Changes no project file except writing evaluation/seed_candidates.csv.

Run from the project root:
    python evaluation/diagnostics.py
"""

import pickle
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "unified_movies_enriched.csv"
TFIDF = ROOT / "model" / "tfidf_matrix_v2.pkl"
SEMANTIC = ROOT / "model" / "semantic_embeddings_v3.pkl"
OUT = Path(__file__).resolve().parent / "seed_candidates.csv"

df = pd.read_csv(DATA).fillna("")
df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

# ---------------------------------------------------------------
# 1. Alignment and identifier checks
# ---------------------------------------------------------------
print("=" * 70, "\n1. ALIGNMENT AND IDS\n" + "=" * 70)

with open(TFIDF, "rb") as f:
    tfidf = pickle.load(f)
with open(SEMANTIC, "rb") as f:
    emb = pickle.load(f)

print("Rows in CSV:", len(df))
print("TF-IDF V2 shape:", tfidf.shape)
print("Semantic shape:", emb.shape)
print("tmdb_id unique:", df["tmdb_id"].is_unique)
print("movie_id unique:", df["movie_id"].is_unique)
print("Titles that occur more than once:", df["title"].duplicated().sum())
print("\noriginal_language by industry (top 6):")
print(
    df.groupby("industry")["original_language"]
    .value_counts().groupby(level=0).head(6)
)

# ---------------------------------------------------------------
# 2. MiniLM truncation check (same text order as semantic_model_v3.py)
# ---------------------------------------------------------------
print("\n" + "=" * 70, "\n2. MiniLM 256-TOKEN TRUNCATION\n" + "=" * 70)

model = SentenceTransformer("all-MiniLM-L6-v2")
limit = model.max_seq_length
tok = model.tokenizer

fields = [
    ("Movie", "title"), ("Overview", "overview"), ("Genres", "genres"),
    ("Keywords", "keywords"), ("Cast", "cast"), ("Director", "director"),
]

prefixes = {name: [] for name, _ in fields}
for _, row in df.iterrows():
    text = ""
    for label, col in fields:
        text += f"{label}: {row[col]}\n"
        prefixes[label].append(text.strip())

full_len = [len(x) for x in tok(prefixes["Director"], truncation=False)["input_ids"]]
df["total_tokens"] = full_len

print("Token limit:", limit)
print("Total tokens per description:")
print(df.groupby("industry")["total_tokens"].describe()[["mean", "50%", "max"]].round(1))
print("\nShare of descriptions longer than the limit:")
print((df["total_tokens"] > limit).groupby(df["industry"]).mean().round(3))

print("\nShare of rows where each field is fully inside the limit:")
intact = {}
for label, _ in fields:
    lengths = [len(x) for x in tok(prefixes[label], truncation=False)["input_ids"]]
    intact[label] = pd.Series(lengths) <= limit
print(pd.DataFrame(intact).groupby(df["industry"].values).mean().round(3))

# ---------------------------------------------------------------
# 3. Seed candidates (you choose the final seeds)
# ---------------------------------------------------------------
print("\n" + "=" * 70, "\n3. SEED CANDIDATES\n" + "=" * 70)

full = df[
    (df["overview"] != "") & (df["genres"] != "")
    & (df["cast"] != "") & (df["director"] != "")
]

groups = []
for industry in ["Hollywood", "Bollywood"]:
    part = (
        full[full["industry"] == industry]
        .sort_values("vote_count", ascending=False).head(20).copy()
    )
    part["group"] = f"popular_{industry.lower()}"
    groups.append(part)

sparse = df[
    (df["industry"] == "Bollywood") & (df["overview"] != "")
    & ((df["genres"] == "") | (df["keywords"] == ""))
].sample(n=6, random_state=42).copy()
sparse["group"] = "sparse_bollywood"
groups.append(sparse)

dup_titles = df[df["title"].duplicated(keep=False)]
dup = dup_titles.sort_values("vote_count", ascending=False).head(6).copy()
dup["group"] = "duplicate_title"
groups.append(dup)

cands = pd.concat(groups)[
    ["group", "tmdb_id", "title", "industry", "year", "genres", "vote_count"]
]
cands.to_csv(OUT, index=False)
print(cands.to_string(index=False))
print("\nSaved:", OUT)