
from pathlib import Path
import os
import time
import requests
import pandas as pd

from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------------------
# 1. Configuration
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("TMDB_API_KEY", "").strip()
ACCESS_TOKEN = os.getenv("TMDB_ACCESS_TOKEN", "").strip()

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "bollywood_discover.csv"
)

MAX_PAGES = 50

if not API_KEY and not ACCESS_TOKEN:
    raise ValueError("TMDB API credentials missing!")

# --------------------------------
# 2. Configure retry session
# --------------------------------

session = requests.Session()

retry = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)

# --------------------------------
# 3. Fetch movie pages
# --------------------------------

url = "https://api.themoviedb.org/3/discover/movie"

all_movies = []
failed_pages = []
total_pages_available = None

headers = {}

if ACCESS_TOKEN:
    headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"

for page in range(1, MAX_PAGES + 1):

    params = {
        "with_original_language": "hi",
        "with_origin_country": "IN",
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "language": "en-US",
        "page": page
    }

    if not ACCESS_TOKEN:
        params["api_key"] = API_KEY

    try:

        response = session.get(
            url,
            params=params,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        if total_pages_available is None:
            total_pages_available = data.get("total_pages", 0)

            print("TMDB available pages:", total_pages_available)
            print("Our page limit:", MAX_PAGES)

        results = data.get("results", [])

        all_movies.extend(results)

        print(
            f"Page {page}: {len(results)} movies fetched | "
            f"Total collected: {len(all_movies)}"
        )

        if not results:
            break

        if page >= total_pages_available:
            print("Reached the last available page.")
            break

    except requests.RequestException as error:

        failed_pages.append(page)

        print(f"Page {page} failed: {error}")

    time.sleep(0.3)

# --------------------------------
# 4. Create dataset
# --------------------------------

if not all_movies:
    raise RuntimeError(
        "No movies fetched. Check your internet connection and try again."
    )

df = pd.DataFrame(all_movies)

before = len(df)

df = df.drop_duplicates(subset=["id"], keep="first")

after = len(df)

# --------------------------------
# 5. Save dataset
# --------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== COLLECTION COMPLETED ==========")

print("Records fetched:", before)
print("Unique movie records:", after)
print("Duplicates removed:", before - after)
print("Failed pages:", failed_pages)
print("Saved file:", OUTPUT_FILE)
