
from pathlib import Path
import pandas as pd
import json

# --------------------------------
# 1. Project paths
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

HOLLYWOOD_MOVIES = DATA_DIR / "tmdb_5000_movies.csv"
HOLLYWOOD_CREDITS = DATA_DIR / "tmdb_5000_credits.csv"
BOLLYWOOD_FILE = RAW_DIR / "bollywood_discover.csv"

OUTPUT_FILE = PROCESSED_DIR / "unified_movies.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------
# 2. Helper functions
# --------------------------------

def parse_json(value):
    """Convert JSON stored in CSV cells into Python data."""
    if pd.isna(value) or value == "":
        return []

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def extract_names(value, limit=None):
    """Extract names from TMDB JSON columns."""
    items = parse_json(value)
    names = [item.get("name", "") for item in items if item.get("name")]

    if limit:
        names = names[:limit]

    return " ".join(names)


def extract_director(value):
    """Extract director from the crew column."""
    crew = parse_json(value)

    for person in crew:
        if person.get("job") == "Director":
            return person.get("name", "")

    return ""


# TMDB genre ID mapping for Bollywood records
GENRE_MAP = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western"
}


def convert_genre_ids(value):
    ids = parse_json(value)

    if not isinstance(ids, list):
        return ""

    return " ".join(
        GENRE_MAP.get(int(genre_id), "")
        for genre_id in ids
        if str(genre_id).isdigit()
        and GENRE_MAP.get(int(genre_id))
    )


# --------------------------------
# 3. Load Hollywood datasets
# --------------------------------

print("Loading Hollywood datasets...")

movies = pd.read_csv(HOLLYWOOD_MOVIES)
credits = pd.read_csv(HOLLYWOOD_CREDITS)

# Remove duplicate IDs before merging
movies = movies.drop_duplicates(subset=["id"])
credits = credits.drop_duplicates(subset=["movie_id"])

# Merge using TMDB movie ID
hollywood = movies.merge(
    credits[["movie_id", "cast", "crew"]],
    left_on="id",
    right_on="movie_id",
    how="left"
)

print("Hollywood records:", len(hollywood))


# --------------------------------
# 4. Prepare Hollywood dataset
# --------------------------------

hollywood_data = pd.DataFrame()

hollywood_data["movie_id"] = hollywood["id"].apply(
    lambda x: f"tmdb_{x}"
)

hollywood_data["tmdb_id"] = hollywood["id"]
hollywood_data["title"] = hollywood["title"]
hollywood_data["original_title"] = hollywood["original_title"]
hollywood_data["industry"] = "Hollywood"
hollywood_data["original_language"] = hollywood["original_language"]
hollywood_data["release_date"] = hollywood["release_date"]
hollywood_data["overview"] = hollywood["overview"].fillna("")

hollywood_data["genres"] = hollywood["genres"].apply(extract_names)
hollywood_data["keywords"] = hollywood["keywords"].apply(extract_names)
hollywood_data["cast"] = hollywood["cast"].apply(
    lambda x: extract_names(x, limit=3)
)
hollywood_data["director"] = hollywood["crew"].apply(extract_director)

hollywood_data["vote_average"] = hollywood["vote_average"]
hollywood_data["vote_count"] = hollywood["vote_count"]
hollywood_data["popularity"] = hollywood["popularity"]
hollywood_data["poster_path"] = None
hollywood_data["backdrop_path"] = None


# --------------------------------
# 5. Load Bollywood dataset
# --------------------------------

print("\nLoading Bollywood dataset...")

bollywood = pd.read_csv(BOLLYWOOD_FILE)

print("Bollywood records:", len(bollywood))


# --------------------------------
# 6. Prepare Bollywood dataset
# --------------------------------

bollywood_data = pd.DataFrame()

bollywood_data["movie_id"] = bollywood["id"].apply(
    lambda x: f"tmdb_{x}"
)

bollywood_data["tmdb_id"] = bollywood["id"]
bollywood_data["title"] = bollywood["title"]
bollywood_data["original_title"] = bollywood["original_title"]
bollywood_data["industry"] = "Bollywood"
bollywood_data["original_language"] = bollywood["original_language"]
bollywood_data["release_date"] = bollywood["release_date"]
bollywood_data["overview"] = bollywood["overview"].fillna("")

bollywood_data["genres"] = bollywood["genre_ids"].apply(
    convert_genre_ids
)

# These fields will be enriched later using TMDB details
bollywood_data["keywords"] = ""
bollywood_data["cast"] = ""
bollywood_data["director"] = ""

bollywood_data["vote_average"] = bollywood["vote_average"]
bollywood_data["vote_count"] = bollywood["vote_count"]
bollywood_data["popularity"] = bollywood["popularity"]
bollywood_data["poster_path"] = bollywood["poster_path"]
bollywood_data["backdrop_path"] = bollywood["backdrop_path"]


# --------------------------------
# 7. Combine both datasets
# --------------------------------

print("\nCombining datasets...")

unified = pd.concat(
    [hollywood_data, bollywood_data],
    ignore_index=True
)

# Remove duplicate TMDB movie IDs
unified = unified.drop_duplicates(
    subset=["tmdb_id"],
    keep="first"
)

# Clean important text columns
text_columns = [
    "title",
    "original_title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "director"
]

for column in text_columns:
    unified[column] = unified[column].fillna("").astype(str).str.strip()

# Save final dataset
unified.to_csv(OUTPUT_FILE, index=False)

# --------------------------------
# 8. Final report
# --------------------------------

print("\n========== MERGE COMPLETED ==========")
print("Hollywood records:", len(hollywood_data))
print("Bollywood records:", len(bollywood_data))
print("Unified records:", len(unified))
print("Duplicate TMDB IDs:", unified["tmdb_id"].duplicated().sum())

print("\nIndustry distribution:")
print(unified["industry"].value_counts())

print("\nMissing values:")
print(unified.isnull().sum())

print("\nSaved file:")
print(OUTPUT_FILE)
