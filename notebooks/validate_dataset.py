
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE_PATH = PROJECT_ROOT / "data" / "processed" / "unified_movies.csv"

df = pd.read_csv(FILE_PATH)

print("========== DATASET VALIDATION ==========")

print("\n1. Dataset shape:")
print(df.shape)

print("\n2. Industry distribution:")
print(df["industry"].value_counts())

print("\n3. Duplicate TMDB IDs:")
print(df["tmdb_id"].duplicated().sum())

print("\n4. Missing values:")
print(df.isnull().sum())

print("\n5. Missing overview:")
print((df["overview"].fillna("").str.strip() == "").sum())

print("\n6. Missing genres:")
print((df["genres"].fillna("").str.strip() == "").sum())

print("\n7. Missing cast:")
print((df["cast"].fillna("").str.strip() == "").sum())

print("\n8. Sample Bollywood movies:")
print(
    df[df["industry"] == "Bollywood"][
        ["title", "release_date", "genres", "vote_average"]
    ].head(5).to_string(index=False)
)

print("\n9. Sample Hollywood movies:")
print(
    df[df["industry"] == "Hollywood"][
        ["title", "release_date", "genres", "vote_average"]
    ].head(5).to_string(index=False)
)

print("\n========== VALIDATION COMPLETED ==========")
