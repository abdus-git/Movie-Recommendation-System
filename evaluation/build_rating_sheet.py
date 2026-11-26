"""
Builds the blinded rating sheet for the 4Cinephiles engine evaluation.

Reproduces four EXISTING engines from the saved pickles. It does not import,
change or replace any of them:

    V2       TF-IDF V2 cosine                       recommendation_engine_v2.py
    V3       MiniLM semantic cosine                 recommendation_engine_v3.py
    V4       50% V2 + 50% V3                        recommendation_engine_v4.py
    V4.2     50/50 blend + genre factor + title     recommendation_engine_v4_2.py
             exclusion + title de-duplication

Seeds are selected by tmdb_id (evaluation/seeds.csv), never by title.
The scoring logic below is copied from the engines; the only difference is
that per-movie genre sets and title keys are computed once instead of inside
the ranking loop, which does not change any result.

Run from the project root:
    python evaluation/build_rating_sheet.py
"""

import argparse
import pickle
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
EVAL = Path(__file__).resolve().parent

DATA = ROOT / "data" / "processed" / "unified_movies_enriched.csv"
EXTRAS = ROOT / "data" / "processed" / "movie_extras.csv"
TFIDF = ROOT / "model" / "tfidf_matrix_v2.pkl"
SEMANTIC = ROOT / "model" / "semantic_embeddings_v3.pkl"

SEEDS = EVAL / "seeds.csv"
SHEET = EVAL / "relevance_labels.csv"
LEGACY_SHEET = EVAL / "rating_sheet.csv"
PRIVATE = EVAL / "private"
RESULTS = EVAL / "model_recommendations.csv"
LEGACY_RESULTS = PRIVATE / "engine_results.csv"

TOP_N = 10
V2_WEIGHT = 0.50          # V4 and V4.2 blend, as in the engines
V3_WEIGHT = 0.50
GENRE_ADJUSTMENT = 0.20   # V4.2
SHUFFLE_SEED = 42
OVERVIEW_CHARS = 350

ENGINES = [
    "V2",
    "V3",
    "V4 (50:50)",
    "V4.1 (70:30)",
    "V4.1 (50:50)",
    "V4.1 (30:70)",
    "V4.2",
]

parser = argparse.ArgumentParser()
parser.add_argument("--fidelity", type=int, nargs="*",
                    default=[27205, 20453, 534780],
                    help="tmdb_ids whose top-10 lists are printed for "
                         "comparison with your command-line engines")
args = parser.parse_args()


# ==========================================
# 1. LOAD DATA AND MODELS (same as the engines)
# ==========================================

movies = pd.read_csv(DATA).fillna("")
movies.reset_index(drop=True, inplace=True)

with open(TFIDF, "rb") as f:
    tfidf = pickle.load(f)
with open(SEMANTIC, "rb") as f:
    emb = pickle.load(f)

if not (len(movies) == tfidf.shape[0] == emb.shape[0]):
    raise SystemExit("Dataset and model dimensions do not match!")

years = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
movies["year"] = years.map(lambda y: "" if pd.isna(y) else str(int(y)))


# ==========================================
# 2. HELPERS (copied from recommendation_engine_v4_2.py)
# ==========================================

def normalize_title(title):
    return re.sub(r"\s+", " ", str(title).strip().lower())


def get_genres(value):
    return {
        g.strip().lower()
        for g in re.split(r"[,|;/\s]+", str(value))
        if g.strip()
    }


def get_genre_similarity(source, candidate):
    if not source or not candidate:
        return None
    return len(source & candidate) / len(source | candidate)


title_keys = [normalize_title(t) for t in movies["title"]]
genre_sets = [get_genres(g) for g in movies["genres"]]


# ==========================================
# 3. THE FOUR ENGINES (each returns [(row_index, score), ...])
# ==========================================

def engine_v2(i, tfidf_scores):
    ranked = sorted(enumerate(tfidf_scores), key=lambda x: x[1], reverse=True)
    ranked = [x for x in ranked if x[0] != i]
    return [(int(j), float(s)) for j, s in ranked[:TOP_N]]


def engine_v3(i, semantic_scores):
    order = semantic_scores.argsort()[::-1]
    order = [j for j in order if j != i]
    return [(int(j), float(semantic_scores[j])) for j in order[:TOP_N]]


def engine_v4(i, hybrid):
    valid = np.array([j for j in range(len(movies)) if j != i])
    top = valid[np.argsort(hybrid[valid])[::-1][:TOP_N]]
    return [(int(j), float(hybrid[j])) for j in top]


def engine_v41(i, hybrid, v2_weight, v3_weight):
    source_key = title_keys[i]
    source_genres = genre_sets[i]

    valid = [j for j in range(len(movies)) if title_keys[j] != source_key]
    adjusted = hybrid.copy()

    for j in valid:
        genre_score = get_genre_similarity(source_genres, genre_sets[j])
        if genre_score is not None:
            adjusted[j] = 0.85 * hybrid[j] + 0.15 * genre_score
        if directly_references_title(movies.at[i, "title"], movies.at[j, "overview"]):
            adjusted[j] *= 0.85

    ranked = sorted(valid, key=lambda j: adjusted[j], reverse=True)
    selected, seen = [], set()
    for j in ranked:
        key = title_keys[j]
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(j)
        if len(selected) >= TOP_N:
            break

    return [(int(j), float(adjusted[j])) for j in selected]


def engine_v42(i, hybrid):
    source_key = title_keys[i]
    source_genres = genre_sets[i]

    valid = [j for j in range(len(movies)) if title_keys[j] != source_key]

    final = hybrid.copy()
    for j in valid:
        genre_score = get_genre_similarity(source_genres, genre_sets[j])
        if genre_score is not None:
            final[j] = hybrid[j] * (
                1 - GENRE_ADJUSTMENT + GENRE_ADJUSTMENT * genre_score
            )

    ranked = sorted(valid, key=lambda j: final[j], reverse=True)

    selected, seen = [], set()
    for j in ranked:
        key = title_keys[j]
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(j)
        if len(selected) >= TOP_N:
            break

    return [(int(j), float(final[j])) for j in selected]


# ==========================================
# 4. LOAD AND VALIDATE SEEDS
# ==========================================

seeds = pd.read_csv(SEEDS)
seed_rows = {}

print("Validating seeds...")
for _, seed in seeds.iterrows():
    hits = movies.index[movies["tmdb_id"] == seed["tmdb_id"]].tolist()
    if len(hits) != 1:
        raise SystemExit(f"Seed {seed['seed_id']}: tmdb_id "
                         f"{seed['tmdb_id']} found {len(hits)} times.")
    i = hits[0]
    if normalize_title(seed["title"]) != title_keys[i]:
        raise SystemExit(f"Seed {seed['seed_id']}: title mismatch "
                         f"('{seed['title']}' vs '{movies.at[i, 'title']}').")
    if seed["industry"] != movies.at[i, "industry"]:
        raise SystemExit(f"Seed {seed['seed_id']}: industry mismatch.")
    seed_rows[seed["seed_id"]] = i
    shared = sum(1 for k in title_keys if k == title_keys[i])
    if shared > 1:
        print(f"  note: {seed['seed_id']} '{seed['title']}' shares its title "
              f"with {shared - 1} other record(s).")

print(f"  {len(seeds)} seeds OK "
      f"({(seeds['industry'] == 'Hollywood').sum()} Hollywood, "
      f"{(seeds['industry'] == 'Bollywood').sum()} Bollywood)")


# ==========================================
# 5. OPTIONAL FRANCHISE INFORMATION
# ==========================================

collection = {}
if EXTRAS.exists():
    extras = pd.read_csv(EXTRAS)
    coverage = len(extras) / len(movies)
    collection = dict(zip(extras["tmdb_id"], extras["collection_id"]))
    print(f"\nmovie_extras.csv found: {len(extras)} of {len(movies)} movies "
          f"({coverage:.0%}).")
    if coverage < 0.99:
        print("  WARNING: extras are incomplete, so same_franchise will be "
              "missing for some movies. Finish the full fetch first, or "
              "fill that column by hand.")
else:
    print("\nmovie_extras.csv not found: same_franchise column left blank.")


def same_collection(seed_tmdb, cand_tmdb):
    a, b = collection.get(seed_tmdb), collection.get(cand_tmdb)
    return "Y" if pd.notna(a) and pd.notna(b) and a == b else ""


# ==========================================
# 6. RUN ALL ENGINES FOR ALL SEEDS
# ==========================================

records = []

for _, seed in seeds.iterrows():
    i = seed_rows[seed["seed_id"]]

    tfidf_scores = cosine_similarity(tfidf[i], tfidf).flatten()
    semantic_scores = cosine_similarity(
        emb[i].reshape(1, -1), emb
    ).flatten()
    hybrid = V2_WEIGHT * tfidf_scores + V3_WEIGHT * semantic_scores

    results = {
        "V2": engine_v2(i, tfidf_scores),
        "V3": engine_v3(i, semantic_scores),
        "V4 (50:50)": engine_v4(i, hybrid),
        "V4.1 (70:30)": engine_v41(i, hybrid, 0.70, 0.30),
        "V4.1 (50:50)": engine_v41(i, hybrid, 0.50, 0.50),
        "V4.1 (30:70)": engine_v41(i, hybrid, 0.30, 0.70),
        "V4.2": engine_v42(i, hybrid),
    }

    for engine, ranked in results.items():
        for rank, (j, score) in enumerate(ranked, start=1):
            records.append({
                "seed_id": seed["seed_id"],
                "seed_tmdb_id": int(seed["tmdb_id"]),
                "seed_title": seed["title"],
                "seed_industry": seed["industry"],
                "engine": engine,
                "rank": rank,
                "cand_index": j,
                "cand_tmdb_id": int(movies.at[j, "tmdb_id"]),
                "cand_title": movies.at[j, "title"],
                "cand_industry": movies.at[j, "industry"],
                "cand_year": movies.at[j, "year"],
                "score": round(score, 4),
            })

results_df = pd.DataFrame(records)


# ==========================================
# 7. BLINDED SHEET: ONE ROW PER UNIQUE CANDIDATE PER SEED
# ==========================================

rng = np.random.default_rng(SHUFFLE_SEED)
sheet_rows = []
candidate_ids = {}

for _, seed in seeds.iterrows():
    sid = seed["seed_id"]
    i = seed_rows[sid]

    pool = results_df.loc[results_df["seed_id"] == sid, "cand_index"].unique()
    pool = rng.permutation(pool)

    for n, j in enumerate(pool, start=1):
        cid = f"{sid}-{n:02d}"
        candidate_ids[(sid, int(j))] = cid

        sheet_rows.append({
            "seed_id": sid,
            "seed_title": seed["title"],
            "seed_industry": seed["industry"],
            "seed_year": movies.at[i, "year"],
            "seed_genres": movies.at[i, "genres"],
            "seed_overview": textwrap.shorten(
                str(movies.at[i, "overview"]), OVERVIEW_CHARS,
                placeholder="..."),
            "candidate_id": cid,
            "cand_title": movies.at[j, "title"],
            "cand_industry": movies.at[j, "industry"],
            "cand_year": movies.at[j, "year"],
            "cand_genres": movies.at[j, "genres"],
            "cand_overview": textwrap.shorten(
                str(movies.at[j, "overview"]), OVERVIEW_CHARS,
                placeholder="..."),
            "same_title_as_seed": "Y" if title_keys[j] == title_keys[i] else "",
            "same_franchise": same_collection(
                int(seed["tmdb_id"]), int(movies.at[j, "tmdb_id"])),
            "rating": "",
            "notes": "",
        })

sheet = pd.DataFrame(sheet_rows)

results_df["candidate_id"] = [
    candidate_ids[(s, j)]
    for s, j in zip(results_df["seed_id"], results_df["cand_index"])
]
results_df = results_df.drop(columns="cand_index")


# ==========================================
# 8. SAVE (utf-8-sig so Excel shows accents correctly)
# ==========================================

PRIVATE.mkdir(exist_ok=True)
sheet.to_csv(SHEET, index=False, encoding="utf-8-sig")
results_df.to_csv(RESULTS, index=False, encoding="utf-8-sig")

sizes = sheet.groupby("seed_id").size()

print("\n========== RATING SHEET BUILT ==========")
print("Rows to rate:", len(sheet))
print("Candidates per seed: min", sizes.min(),
      "| mean", round(sizes.mean(), 1), "| max", sizes.max())
print("Sheet (give this one to the rater):", SHEET)
print("Engine results (do NOT open until rating is finished):", RESULTS)


# ==========================================
# 9. FIDELITY PRINT-OUT
# ==========================================

print("\n" + "=" * 70)
print("FIDELITY CHECK: compare with your command-line engines")
print("=" * 70)

for tmdb_id in args.fidelity:
    block = results_df[results_df["seed_tmdb_id"] == tmdb_id]
    if block.empty:
        print(f"\n(tmdb_id {tmdb_id} is not a seed; skipped)")
        continue
    print(f"\n##### {block['seed_title'].iloc[0]} (tmdb_id {tmdb_id}) #####")
    for engine in ENGINES:
        part = block[block["engine"] == engine].sort_values("rank")
        print(f"\n--- {engine} ---")
        for _, r in part.iterrows():
            print(f"{r['rank']:>2}. {r['cand_title']}  [{r['cand_industry']}]"
                  f"  {r['score']}")
