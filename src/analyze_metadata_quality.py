import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "unified_movies_enriched.csv"
)

movies = pd.read_csv(DATA_PATH).fillna("")

metadata_columns = [
    "overview",
    "genres",
    "keywords",
    "cast",
    "director"
]

print("TOTAL MOVIES:", len(movies))

print("\nMISSING METADATA REPORT")
print("-" * 50)

for column in metadata_columns:
    missing = movies[column].astype(str).str.strip().eq("").sum()
    percentage = (missing / len(movies)) * 100

    print(
        f"{column:12}: {missing:5} missing "
        f"({percentage:.2f}%)"
    )

# Count missing fields per movie
movies["missing_fields"] = (
    movies[metadata_columns]
    .astype(str)
    .apply(lambda row: row.str.strip().eq("").sum(), axis=1)
)

print("\nMETADATA COMPLETENESS DISTRIBUTION")
print("-" * 50)

print(
    movies["missing_fields"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nMOVIES WITH 3 OR MORE MISSING FIELDS")
print("-" * 50)

poor_metadata = movies[
    movies["missing_fields"] >= 3
]

print(
    poor_metadata[
        ["title", "industry", "missing_fields"]
    ].head(20).to_string(index=False)
)

print("\nMETADATA ANALYSIS COMPLETED")
