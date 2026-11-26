"""
One-time fetch of extra TMDB metadata for every movie in the unified dataset.

Adds what the dataset lacks: runtime, poster/backdrop paths (Hollywood rows
have none) and the franchise "collection" a movie belongs to.

Writes a NEW file, data/processed/movie_extras.csv (keyed by tmdb_id).
It never modifies unified_movies_enriched.csv, so row order and model
alignment are unaffected. Safe to stop and re-run: finished IDs are skipped.

Run from the project root:
    python scripts/fetch_movie_extras.py --ids 155 27205 20453   (small test)
    python scripts/fetch_movie_extras.py                         (full run)
"""

import argparse
import csv
import os
import time
from pathlib import Path

import pandas as pd
import requests
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

INPUT_FILE = ROOT / "data" / "processed" / "unified_movies_enriched.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "movie_extras.csv"

COLUMNS = [
    "tmdb_id",
    "runtime",
    "poster_path",
    "backdrop_path",
    "collection_id",
    "collection_name",
]

FLUSH_EVERY = 25

if not API_KEY and not ACCESS_TOKEN:
    raise ValueError("TMDB credentials missing in .env")


# ==========================================
# 2. COMMAND-LINE OPTIONS
# ==========================================

parser = argparse.ArgumentParser(description="Fetch TMDB movie extras.")
parser.add_argument("--ids", type=int, nargs="+",
                    help="fetch only these tmdb_ids (for testing)")
parser.add_argument("--limit", type=int,
                    help="fetch at most this many pending movies")
parser.add_argument("--delay", type=float, default=0.1,
                    help="seconds to wait between requests (default 0.1)")
args = parser.parse_args()


# ==========================================
# 3. REQUEST SESSION WITH RETRIES
#    (same pattern as enrich_bolly.py)
# ==========================================

session = requests.Session()

retry_strategy = Retry(
    total=5,
    connect=5,
    read=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)

session.mount("https://", HTTPAdapter(max_retries=retry_strategy))


# ==========================================
# 4. FETCH AND EXTRACT
# ==========================================

def fetch_details(tmdb_id):
    """Return TMDB's movie JSON, or None if TMDB no longer has the ID."""

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"

    params = {"language": "en-US"}
    headers = {}

    if ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
    else:
        params["api_key"] = API_KEY

    response = session.get(url, params=params, headers=headers, timeout=30)

    if response.status_code == 404:
        return None

    response.raise_for_status()

    return response.json()


def extract_extras(tmdb_id, data):
    """Build one output row. A missing movie gives a row of blanks."""

    row = {column: "" for column in COLUMNS}
    row["tmdb_id"] = tmdb_id

    if data is None:
        return row

    # TMDB returns null or 0 when the runtime is unknown. Store blank.
    row["runtime"] = data.get("runtime") or ""
    row["poster_path"] = data.get("poster_path") or ""
    row["backdrop_path"] = data.get("backdrop_path") or ""

    collection = data.get("belongs_to_collection") or {}
    row["collection_id"] = collection.get("id") or ""
    row["collection_name"] = collection.get("name") or ""

    return row


# ==========================================
# 5. WORK OUT WHAT IS STILL PENDING
# ==========================================

all_ids = (
    pd.read_csv(INPUT_FILE, usecols=["tmdb_id"])["tmdb_id"]
    .astype(int)
    .drop_duplicates()
    .tolist()
)

done = set()

if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
    done = set(
        pd.read_csv(OUTPUT_FILE, usecols=["tmdb_id"])["tmdb_id"].astype(int)
    )

pending = [i for i in all_ids if i not in done]

if args.ids:
    wanted = set(args.ids)
    pending = [i for i in pending if i in wanted]

if args.limit:
    pending = pending[: args.limit]

print("Movies in dataset:", len(all_ids))
print("Already fetched:", len(done))
print("To fetch now:", len(pending))


# ==========================================
# 6. FETCH, APPENDING TO THE FILE AS WE GO
# ==========================================

needs_header = (
    not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0
)

fetched = 0
not_found = []
failed = []

try:
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=COLUMNS)

        if needs_header:
            writer.writeheader()

        for position, tmdb_id in enumerate(pending, start=1):

            try:
                data = fetch_details(tmdb_id)

            except requests.RequestException as error:
                # Not written, so the next run will try this ID again.
                failed.append(tmdb_id)
                print(f"[{position}/{len(pending)}] Failed {tmdb_id}: {error}")
                time.sleep(args.delay)
                continue

            if data is None:
                not_found.append(tmdb_id)

            writer.writerow(extract_extras(tmdb_id, data))
            fetched += 1

            if position % FLUSH_EVERY == 0:
                file.flush()
                print(f"[{position}/{len(pending)}] saved")

            time.sleep(args.delay)

except KeyboardInterrupt:
    print("\nInterrupted. Progress is saved; run the script again to resume.")


# ==========================================
# 7. REPORT
# ==========================================

print("\n========== MOVIE EXTRAS FETCH ==========")
print("Fetched this run:", fetched)
print("Not found on TMDB (stored as blank rows):", len(not_found))
print("Failed (will retry next run):", len(failed))

if not_found:
    print("Not-found IDs:", not_found[:20])
if failed:
    print("Failed IDs:", failed[:20])

extras = pd.read_csv(OUTPUT_FILE)
unified = pd.read_csv(INPUT_FILE, usecols=["tmdb_id", "industry"])

report = unified.merge(extras, on="tmdb_id", how="left", indicator=True)

summary = report.groupby("industry").agg(
    movies=("tmdb_id", "size"),
    fetched=("_merge", lambda s: int((s == "both").sum())),
    with_runtime=("runtime", lambda s: int(s.notna().sum())),
    with_poster=("poster_path", lambda s: int(s.notna().sum())),
    in_collection=("collection_id", lambda s: int(s.notna().sum())),
)

print("\nCoverage by industry:")
print(summary.to_string())
print("\nSaved:", OUTPUT_FILE)
