
from pathlib import Path
import pandas as pd
import json

# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

HOLLYWOOD_MOVIES = DATA / "tmdb_5000_movies.csv"
HOLLYWOOD_CREDITS = DATA / "tmdb_5000_credits.csv"

BOLLYWOOD_FILE = DATA / "raw" / "bollywood_discover.csv"

OUTPUT_DIR = DATA / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "unified_movies_v2.csv"


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_json(value):
    if pd.isna(value) or value == "":
        return []

    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return []


def extract_names(value, limit=None):
    items = parse_json(value)

    names = [
        item.get("name", "")
        for item in items
        if item.get("name")
    ]

    if limit:
        names = names[:limit]

    return " ".join(names)


def extract_director(value):
    crew = parse_json(value)

    for person in crew:
        if person.get("job") == "Director":
            return person.get("name", "")

    return ""


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


def convert_genres(value):
    genre_ids = parse_json(value)

    return " ".join(
        GENRE_MAP.get(int(genre_id), "")
        for genre_id in genre_ids
        if str(genre_id).isdigit()
        and GENRE_MAP.get(int(genre_id))
    )


# ==========================================
# 3. LOAD HOLLYWOOD DATA
# ==========================================

print("Loading Hollywood datasets...")

movies = pd.read_csv(HOLLYWOOD_MOVIES)
credits = pd.read_csv(HOLLYWOOD_CREDITS)

movies = movies.drop_duplicates(subset=["id"])
credits = credits.drop_duplicates(subset=["movie_id"])

hollywood = movies.merge(
    credits[["movie_id", "cast", "crew"]],
    left_on="id",
    right_on="movie_id",
    how="left"
)

h = pd.DataFrame()

h["tmdb_id"] = hollywood["id"]
h["title"] = hollywood["title"]
h["original_title"] = hollywood["original_title"]
h["industry"] = "Hollywood"
h["original_language"] = hollywood["original_language"]
h["release_date"] = hollywood["release_date"]
h["overview"] = hollywood["overview"].fillna("")

h["genres"] = hollywood["genres"].apply(extract_names)
h["keywords"] = hollywood["keywords"].apply(extract_names)

h["cast"] = hollywood["cast"].apply(
    lambda x: extract_names(x, limit=3)
)

h["director"] = hollywood["crew"].apply(extract_director)

h["vote_average"] = hollywood["vote_average"]
h["vote_count"] = hollywood["vote_count"]
h["popularity"] = hollywood["popularity"]

# Original TMDB 5000 dataset does not contain poster paths
h["poster_path"] = ""
h["backdrop_path"] = ""


# ==========================================
# 4. LOAD BOLLYWOOD DATA
# ==========================================

print("Loading Bollywood dataset...")

bollywood = pd.read_csv(BOLLYWOOD_FILE)

b = pd.DataFrame()

b["tmdb_id"] = bollywood["id"]
b["title"] = bollywood["title"]
b["original_title"] = bollywood["original_title"]
b["industry"] = "Bollywood"
b["original_language"] = bollywood["original_language"]
b["release_date"] = bollywood["release_date"]
b["overview"] = bollywood["overview"].fillna("")

b["genres"] = bollywood["genre_ids"].apply(convert_genres)

# These details will be fetched later
b["keywords"] = ""
b["cast"] = ""
b["director"] = ""

b["vote_average"] = bollywood["vote_average"]
b["vote_count"] = bollywood["vote_count"]
b["popularity"] = bollywood["popularity"]

b["poster_path"] = bollywood["poster_path"].fillna("")
b["backdrop_path"] = bollywood["backdrop_path"].fillna("")


# ==========================================
# 5. COMBINE WITH BOLLYWOOD PRIORITY
# ==========================================

print("\nCombining datasets...")

# Bollywood comes first so its classification wins
unified = pd.concat(
    [b, h],
    ignore_index=True
)

unified = unified.drop_duplicates(
    subset=["tmdb_id"],
    keep="first"
)

unified["movie_id"] = unified["tmdb_id"].apply(
    lambda x: f"tmdb_{int(x)}"
)


# ==========================================
# 6. CLEAN DATA
# ==========================================

text_columns = [
    "title",
    "original_title",
    "overview",
    "genres",
    "keywords",
    "cast",
    "director",
    "poster_path",
    "backdrop_path"
]

for column in text_columns:
    unified[column] = (
        unified[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

unified["release_date"] = pd.to_datetime(
    unified["release_date"],
    errors="coerce"
)

today = pd.Timestamp.today().normalize()

unified["release_status"] = "Unknown"

unified.loc[
    unified["release_date"].notna()
    & (unified["release_date"] <= today),
    "release_status"
] = "Released"

unified.loc[
    unified["release_date"].notna()
    & (unified["release_date"] > today),
    "release_status"
] = "Upcoming"


# ==========================================
# 7. SAVE DATASET
# ==========================================

unified.to_csv(OUTPUT_FILE, index=False)


# ==========================================
# 8. FINAL REPORT
# ==========================================

print("\n========== DATASET BUILD COMPLETED ==========")

print("Original Hollywood records:", len(h))
print("Original Bollywood records:", len(b))
print("Unified unique records:", len(unified))

print(
    "Duplicate TMDB IDs:",
    unified["tmdb_id"].duplicated().sum()
)

print("\nIndustry distribution:")
print(unified["industry"].value_counts())

print("\nRelease status:")
print(unified["release_status"].value_counts())

print("\nMissing values:")
print(unified.isnull().sum())

print("\nSaved file:")
print(OUTPUT_FILE)
