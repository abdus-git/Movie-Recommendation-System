import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILE_PATH = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "bollywood_discover.csv"
)

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Total Movies:", len(df))
print("Total Columns:", len(df.columns))

print("\n========== DUPLICATE CHECK ==========")

print("Duplicate Movie IDs:", df["id"].duplicated().sum())

print("\n========== MISSING VALUES ==========")

print(
    df[
        ["title", "original_title", "overview", "release_date"]
    ].isnull().sum()
)

print("\n========== LANGUAGE DISTRIBUTION ==========")

print(df["original_language"].value_counts())

print("\n========== RELEASE YEAR DISTRIBUTION ==========")

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

print(df["release_date"].dt.year.value_counts().sort_index())

print("\n========== SAMPLE MOVIES ==========")

print(
    df[
        ["id", "title", "original_language", "release_date", "vote_average"]
    ].head(10).to_string(index=False)
)

print("\n========== MISSING DATA RECORDS ==========")

print(
    df[
        df["overview"].isnull() |
        df["release_date"].isnull()
    ][
        ["id", "title", "overview", "release_date"]
    ].to_string(index=False)
)


print("\n========== UPCOMING MOVIES ==========")

today = pd.Timestamp.today().normalize()

upcoming = df[
    df["release_date"].notnull() &
    (df["release_date"] > today)
]

print("Upcoming movies:", len(upcoming))

print(
    upcoming[
        ["id", "title", "release_date"]
    ].to_string(index=False)
)


print("\n========== 2026 MOVIES ==========")

movies_2026 = df[
    df["release_date"].dt.year == 2026
]

print("Total 2026 movies:", len(movies_2026))

print(
    movies_2026[
        ["id", "title", "release_date", "vote_average"]
    ].to_string(index=False)
)