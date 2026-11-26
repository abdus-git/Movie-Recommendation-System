
from pathlib import Path
import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

FILE = (
    ROOT / "data" / "processed"
    / "unified_movies_enriched.csv"
)

df = pd.read_csv(FILE)

bollywood = df[df["industry"] == "Bollywood"]
hollywood = df[df["industry"] == "Hollywood"]


# ==========================================
# 2. BASIC DATASET INFORMATION
# ==========================================

print("========== ENRICHED DATASET VALIDATION ==========")

print("\nDataset shape:", df.shape)

print("\nIndustry distribution:")
print(df["industry"].value_counts())

print("\nDuplicate TMDB IDs:")
print(df["tmdb_id"].duplicated().sum())


# ==========================================
# 3. CHECK MISSING INFORMATION
# ==========================================

columns = [
    "overview",
    "genres",
    "keywords",
    "cast",
    "director",
    "poster_path",
    "backdrop_path"
]

print("\nMissing information in Bollywood:")

for column in columns:

    missing = (
        bollywood[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    percentage = (
        missing / len(bollywood) * 100
        if len(bollywood) else 0
    )

    print(
        f"{column}: {missing} "
        f"({percentage:.2f}%)"
    )


# ==========================================
# 4. RELEASE STATUS
# ==========================================

print("\nRelease status distribution:")

print(bollywood["release_status"].value_counts())


# ==========================================
# 5. CHECK BOLLYWOOD SAMPLE
# ==========================================

print("\nBollywood sample records:")

print(
    bollywood[
        [
            "title",
            "genres",
            "cast",
            "director",
            "keywords"
        ]
    ].head(10).to_string(index=False)
)


# ==========================================
# 6. CHECK HOLLYWOOD SAMPLE
# ==========================================

print("\nHollywood sample records:")

print(
    hollywood[
        [
            "title",
            "genres",
            "cast",
            "director"
        ]
    ].head(5).to_string(index=False)
)


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

print("\n========== VALIDATION SUMMARY ==========")

print("Total movies:", len(df))
print("Bollywood candidates:", len(bollywood))
print("Hollywood records:", len(hollywood))
print("Duplicate IDs:", df["tmdb_id"].duplicated().sum())

print("\nValidation completed.")
