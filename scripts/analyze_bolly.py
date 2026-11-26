
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BOLLYWOOD_FILE = (
    PROJECT_ROOT / "data" / "raw" / "bollywood_discover.csv"
)

HOLLYWOOD_FILE = (
    PROJECT_ROOT / "data" / "tmdb_5000_movies.csv"
)

df = pd.read_csv(BOLLYWOOD_FILE)
hollywood = pd.read_csv(HOLLYWOOD_FILE)

# Convert release dates
df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

# --------------------------------
# 1. Basic information
# --------------------------------

print("========== BOLLYWOOD DATASET AUDIT ==========")

print("\nDataset shape:", df.shape)

print("\nDuplicate TMDB IDs:")
print(df["id"].duplicated().sum())

print("\nMissing values:")
print(df[[
    "title",
    "original_title",
    "overview",
    "release_date",
    "poster_path"
]].isnull().sum())

# --------------------------------
# 2. Release year analysis
# --------------------------------

df["release_year"] = df["release_date"].dt.year

print("\nMovies by release year:")
print(df["release_year"].value_counts(
    dropna=False
).sort_index().to_string())

# --------------------------------
# 3. Upcoming movies
# --------------------------------

today = pd.Timestamp.today().normalize()

upcoming = df[
    df["release_date"].notna()
    & (df["release_date"] > today)
]

print("\nUpcoming movies:", len(upcoming))

print(
    upcoming[["title", "release_date"]]
    .sort_values("release_date")
    .head(20)
    .to_string(index=False)
)

# --------------------------------
# 4. Missing overview
# --------------------------------

missing_overview = df[
    df["overview"].isna()
    | (df["overview"].astype(str).str.strip() == "")
]

print("\nMovies with missing overview:", len(missing_overview))

# --------------------------------
# 5. Language and country check
# --------------------------------

print("\nOriginal language distribution:")
print(df["original_language"].value_counts(dropna=False))

# --------------------------------
# 6. Compare TMDB IDs with Hollywood
# --------------------------------

bollywood_ids = set(df["id"].dropna().astype(int))
hollywood_ids = set(hollywood["id"].dropna().astype(int))

overlap = bollywood_ids.intersection(hollywood_ids)

print("\nHollywood-Bollywood ID overlap:", len(overlap))

if overlap:
    print("\nOverlapping movie records:")
    print(
        df[df["id"].isin(overlap)][
            ["id", "title", "original_title"]
        ].to_string(index=False)
    )

# --------------------------------
# 7. Final summary
# --------------------------------

print("\n========== AUDIT SUMMARY ==========")
print("Total unique Bollywood candidates:", df["id"].nunique())
print("Missing release dates:", df["release_date"].isna().sum())
print("Missing overviews:", len(missing_overview))
print("Upcoming movies:", len(upcoming))
print("Hollywood ID overlaps:", len(overlap))

print("\nAudit completed.")
