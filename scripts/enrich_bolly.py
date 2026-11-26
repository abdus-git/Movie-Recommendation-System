
from pathlib import Path
import os
import time
import requests
import pandas as pd

from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ==========================================
# 1. PATHS AND CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("TMDB_API_KEY", "").strip()
ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN", "").strip()

INPUT_FILE = ROOT / "data" / "processed" / "unified_movies_v2.csv"

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "unified_movies_enriched.csv"
)

if not API_KEY and not ACCESS_TOKEN:
    raise ValueError("TMDB credentials missing in .env")

CHECKPOINT_INTERVAL = 20


# ==========================================
# 2. REQUEST SESSION WITH RETRIES
# ==========================================

session = requests.Session()

retry_strategy = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("https://", adapter)


# ==========================================
# 3. FETCH MOVIE DETAILS
# ==========================================

def fetch_movie_details(tmdb_id):

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"

    params = {
        "append_to_response": "credits,keywords",
        "language": "en-US"
    }

    headers = {}

    if ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
    else:
        params["api_key"] = API_KEY

    response = session.get(
        url,
        params=params,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ==========================================
# 4. EXTRACT INFORMATION
# ==========================================

def extract_movie_data(data):

    genres = data.get("genres", [])

    credits = data.get("credits", {})

    cast = credits.get("cast", [])
    crew = credits.get("crew", [])

    keyword_data = data.get("keywords", {})

    keywords = keyword_data.get(
        "keywords",
        keyword_data.get("results", [])
    )

    genre_names = [
        item["name"] for item in genres if item.get("name")
    ]

    keyword_names = [
        item["name"] for item in keywords if item.get("name")
    ]

    cast_names = [
        person["name"]
        for person in cast[:3]
        if person.get("name")
    ]

    directors = [
        person["name"]
        for person in crew
        if person.get("job") == "Director" and person.get("name")
    ]

    return {
        "genres": " ".join(genre_names),
        "keywords": " ".join(keyword_names),
        "cast": " ".join(cast_names),
        "director": directors[0] if directors else "",
        "poster_path": data.get("poster_path") or "",
        "backdrop_path": data.get("backdrop_path") or ""
    }


# ==========================================
# 5. LOAD DATASET OR RESUME CHECKPOINT
# ==========================================

print("Loading dataset...")

if OUTPUT_FILE.exists():

    print("Existing enriched file found. Resuming from it.")

    df = pd.read_csv(OUTPUT_FILE)

else:

    print("Starting from unified dataset.")

    df = pd.read_csv(INPUT_FILE)


bollywood_indices = df.index[
    df["industry"] == "Bollywood"
].tolist()

total = len(bollywood_indices)

print("Bollywood movies:", total)


# ==========================================
# 6. ENRICH MISSING MOVIE DETAILS
# ==========================================

success_count = 0
failed_count = 0
skipped_count = 0

for position, index in enumerate(bollywood_indices, start=1):

    movie = df.loc[index]

    tmdb_id = int(movie["tmdb_id"])

    fields_to_check = [
        "genres",
        "keywords",
        "cast",
        "director",
        "poster_path",
        "backdrop_path"
    ]

    # Skip movies that are already fully enriched
    if all(
        pd.notna(movie.get(field))
        and str(movie.get(field)).strip() != ""
        for field in fields_to_check
    ):
        skipped_count += 1
        continue

    try:

        data = fetch_movie_details(tmdb_id)

        extracted = extract_movie_data(data)

        for column, value in extracted.items():

            if value:
                df.at[index, column] = value

        success_count += 1

        print(
            f"[{position}/{total}] "
            f"Updated: {movie['title']}"
        )

    except Exception as error:

        failed_count += 1

        print(
            f"[{position}/{total}] "
            f"Failed: {movie['title']} | {error}"
        )

    # Save progress regularly
    if position % CHECKPOINT_INTERVAL == 0:

        df.to_csv(OUTPUT_FILE, index=False)

        print("\nCheckpoint saved!\n")

    time.sleep(0.3)


# ==========================================
# 7. SAVE FINAL DATASET
# ==========================================

df.to_csv(OUTPUT_FILE, index=False)


# ==========================================
# 8. FINAL REPORT
# ==========================================

print("\n========== ENRICHMENT COMPLETED ==========")

print("Total Bollywood movies:", total)
print("Successfully updated:", success_count)
print("Skipped (already complete):", skipped_count)
print("Failed requests:", failed_count)

print("\nRemaining missing information:")

for column in [
    "genres",
    "keywords",
    "cast",
    "director",
    "poster_path"
]:

    missing = (
        df.loc[df["industry"] == "Bollywood", column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"{column}: {missing}")

print("\nSaved enriched dataset:")
print(OUTPUT_FILE)
