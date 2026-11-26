
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

titles_to_check = ["3 Idiots", "4 Idiots"]

for title in titles_to_check:

    result = movies[
        movies["title"].str.strip().str.lower() == title.lower()
    ]

    print("\n" + "=" * 70)
    print("MOVIE:", title)
    print("=" * 70)

    if result.empty:
        print("Movie not found.")
        continue

    columns = [
        "title",
        "industry",
        "release_date",
        "overview",
        "genres",
        "keywords",
        "cast",
        "director"
    ]

    print(result[columns].to_string(index=False))

