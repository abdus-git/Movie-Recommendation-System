"""Build the manual evaluation dataset for the existing 4Cinephiles models.

This script uses the saved datasets and the model formulas already present in the
project (V2, V3, V4, V4.1, V4.2) without altering the engine scoring logic.
It writes two primary outputs:

1. evaluation/model_recommendations.csv
   Structured top-N recommendations for the seeded movies.
2. evaluation/relevance_labels.csv
   One row per recommendation, ready for manual review/rating.

The output is intentionally blank for relevance ratings. No fabricated labels are
inserted, and no final model is selected until manual ratings are completed.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
EVAL = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "unified_movies_enriched.csv"
TFIDF = ROOT / "model" / "tfidf_matrix_v2.pkl"
SEMANTIC = ROOT / "model" / "semantic_embeddings_v3.pkl"
SEEDS = EVAL / "seeds.csv"
MODEL_RESULTS = EVAL / "model_recommendations.csv"
LABELS = EVAL / "relevance_labels.csv"
RESULT_SUMMARY = EVAL / "evaluation_results.csv"
REPORT = EVAL / "evaluation_report.md"

TOP_N = 10
V2_WEIGHT = 0.50
V3_WEIGHT = 0.50
GENRE_ADJUSTMENT = 0.20


def normalize_title(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_genres(value) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        token.strip().lower()
        for token in re.split(r"[,|;/\s]+", str(value))
        if token.strip()
    }


def genre_jaccard(source, candidate) -> float:
    if not source or not candidate:
        return 0.0
    union = source | candidate
    if not union:
        return 0.0
    return len(source & candidate) / len(union)


def metadata_completeness(row) -> float:
    fields = ["overview", "genres", "keywords", "cast", "director"]
    present = sum(bool(str(row.get(field, "")).strip()) for field in fields)
    return round((present / len(fields)) * 100, 1)


def directly_references_title(source_title, candidate_overview) -> bool:
    source = str(source_title).strip().lower()
    body = str(candidate_overview).lower()
    if len(source) < 4:
        return False
    pattern = r"(?<!\w)" + re.escape(source) + r"(?!\w)"
    return bool(re.search(pattern, body))


def v2_ranked(seed_idx, tfidf_matrix):
    scores = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
    valid = [i for i in range(len(movies)) if i != seed_idx]
    ranked = sorted(valid, key=lambda i: scores[i], reverse=True)
    deduped, seen = [], set()
    for i in ranked:
        key = normalize_title(movies.at[i, "title"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((i, float(scores[i])))
        if len(deduped) >= TOP_N:
            break
    return deduped


def v3_ranked(seed_idx, semantic_embeddings):
    scores = cosine_similarity(
        semantic_embeddings[seed_idx].reshape(1, -1),
        semantic_embeddings,
    ).flatten()
    valid = [i for i in range(len(movies)) if i != seed_idx]
    ranked = sorted(valid, key=lambda i: scores[i], reverse=True)
    deduped, seen = [], set()
    for i in ranked:
        key = normalize_title(movies.at[i, "title"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((i, float(scores[i])))
        if len(deduped) >= TOP_N:
            break
    return deduped


def v4_ranked(seed_idx, tfidf_matrix, semantic_embeddings):
    tfidf_scores = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
    semantic_scores = cosine_similarity(
        semantic_embeddings[seed_idx].reshape(1, -1),
        semantic_embeddings,
    ).flatten()
    hybrid = V2_WEIGHT * tfidf_scores + V3_WEIGHT * semantic_scores
    valid = [i for i in range(len(movies)) if i != seed_idx]
    ranked = sorted(valid, key=lambda i: hybrid[i], reverse=True)
    deduped, seen = [], set()
    for i in ranked:
        key = normalize_title(movies.at[i, "title"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((i, float(hybrid[i])))
        if len(deduped) >= TOP_N:
            break
    return deduped


def v41_ranked(seed_idx, tfidf_matrix, semantic_embeddings, v2_weight, v3_weight):
    tfidf_scores = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
    semantic_scores = cosine_similarity(
        semantic_embeddings[seed_idx].reshape(1, -1),
        semantic_embeddings,
    ).flatten()
    hybrid = v2_weight * tfidf_scores + v3_weight * semantic_scores
    source_genres = parse_genres(movies.at[seed_idx, "genres"])
    valid = [i for i in range(len(movies)) if i != seed_idx]
    adjusted = hybrid.copy()
    for i in valid:
        candidate_genres = parse_genres(movies.at[i, "genres"])
        jaccard = genre_jaccard(source_genres, candidate_genres)
        if source_genres or candidate_genres:
            adjusted[i] = 0.85 * hybrid[i] + 0.15 * jaccard
        if directly_references_title(movies.at[seed_idx, "title"], movies.at[i, "overview"]):
            adjusted[i] *= 0.85
    ranked = sorted(valid, key=lambda i: adjusted[i], reverse=True)
    deduped, seen = [], set()
    for i in ranked:
        key = normalize_title(movies.at[i, "title"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((i, float(adjusted[i])))
        if len(deduped) >= TOP_N:
            break
    return deduped


def v42_ranked(seed_idx, tfidf_matrix, semantic_embeddings):
    tfidf_scores = cosine_similarity(tfidf_matrix[seed_idx], tfidf_matrix).flatten()
    semantic_scores = cosine_similarity(
        semantic_embeddings[seed_idx].reshape(1, -1),
        semantic_embeddings,
    ).flatten()
    hybrid = V2_WEIGHT * tfidf_scores + V3_WEIGHT * semantic_scores
    source_genres = parse_genres(movies.at[seed_idx, "genres"])
    valid = [i for i in range(len(movies)) if i != seed_idx]
    final = hybrid.copy()
    for i in valid:
        candidate_genres = parse_genres(movies.at[i, "genres"])
        similarity = genre_jaccard(source_genres, candidate_genres)
        if source_genres and candidate_genres:
            final[i] = hybrid[i] * (1 - GENRE_ADJUSTMENT + GENRE_ADJUSTMENT * similarity)
    ranked = sorted(valid, key=lambda i: final[i], reverse=True)
    deduped, seen = [], set()
    for i in ranked:
        key = normalize_title(movies.at[i, "title"])
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((i, float(final[i])))
        if len(deduped) >= TOP_N:
            break
    return deduped


def build_rows():
    rows = []
    seeds = pd.read_csv(SEEDS)
    for _, seed in seeds.iterrows():
        seed_tmdb = int(seed["tmdb_id"])
        seed_title = str(seed["title"])
        seed_idx = int(movies.index[movies["tmdb_id"] == seed_tmdb][0])
        seed_genres = parse_genres(movies.at[seed_idx, "genres"])

        model_specs = [
            ("V2", lambda: v2_ranked(seed_idx, tfidf_matrix)),
            ("V3", lambda: v3_ranked(seed_idx, semantic_embeddings)),
            ("V4 (50:50)", lambda: v4_ranked(seed_idx, tfidf_matrix, semantic_embeddings)),
            ("V4.1 (70:30)", lambda: v41_ranked(seed_idx, tfidf_matrix, semantic_embeddings, 0.70, 0.30)),
            ("V4.1 (50:50)", lambda: v41_ranked(seed_idx, tfidf_matrix, semantic_embeddings, 0.50, 0.50)),
            ("V4.1 (30:70)", lambda: v41_ranked(seed_idx, tfidf_matrix, semantic_embeddings, 0.30, 0.70)),
            ("V4.2", lambda: v42_ranked(seed_idx, tfidf_matrix, semantic_embeddings)),
        ]

        for model_name, build in model_specs:
            for rank, (idx, score) in enumerate(build(), start=1):
                candidate = movies.iloc[idx]
                candidate_genres = parse_genres(candidate["genres"])
                rows.append({
                    "seed_movie": seed_title,
                    "seed_movie_id": seed_tmdb,
                    "model_name": model_name,
                    "recommended_movie": str(candidate["title"]),
                    "recommended_movie_id": int(candidate["tmdb_id"]),
                    "rank": rank,
                    "similarity_score": round(float(score), 4),
                    "genre": str(candidate["genres"]),
                    "industry": str(candidate["industry"]),
                    "metadata_completeness": metadata_completeness(candidate.to_dict()),
                    "seed_genre_overlap": round(genre_jaccard(seed_genres, candidate_genres), 4),
                    "title_reference_flag": int(directly_references_title(seed_title, candidate["overview"])),
                })
    return pd.DataFrame(rows)


movies = pd.read_csv(DATA).fillna("").reset_index(drop=True)
with open(TFIDF, "rb") as fh:
    tfidf_matrix = pickle.load(fh)
with open(SEMANTIC, "rb") as fh:
    semantic_embeddings = pickle.load(fh)

if not (len(movies) == tfidf_matrix.shape[0] == semantic_embeddings.shape[0]):
    raise ValueError(
        f"Dataset and model sizes do not match: {len(movies)} rows, "
        f"{tfidf_matrix.shape[0]} tfidf rows, {semantic_embeddings.shape[0]} semantic rows."
    )

results = build_rows()
if results.empty:
    raise ValueError("No model recommendations were generated for the seed list.")

results.to_csv(MODEL_RESULTS, index=False, encoding="utf-8-sig")

labels = results.rename(columns={
    "seed_movie": "seed_movie",
    "seed_movie_id": "seed_movie_id",
    "model_name": "model_name",
    "recommended_movie": "recommended_movie",
    "recommended_movie_id": "recommended_movie_id",
    "rank": "rank",
})
labels["relevance_rating"] = ""
labels["reviewer_notes"] = ""
labels = labels[
    [
        "seed_movie",
        "seed_movie_id",
        "model_name",
        "recommended_movie",
        "recommended_movie_id",
        "rank",
        "relevance_rating",
        "reviewer_notes",
    ]
]
labels.to_csv(LABELS, index=False, encoding="utf-8-sig")

# Non-human metrics that do not require relevance labels.
summary_rows = []
for model_name, group in results.groupby("model_name"):
    duplicate_rate = 1.0 - (group["recommended_movie"].nunique() / len(group))
    seed_overlap = group["seed_genre_overlap"].mean()
    diversity_values = []
    for _, seed_group in group.groupby("seed_movie"):
        genre_sets = [parse_genres(value) for value in seed_group["genre"].tolist()]
        if len(genre_sets) < 2:
            continue
        pair_count = 0
        total = 0.0
        for i in range(len(genre_sets)):
            for j in range(i + 1, len(genre_sets)):
                pair_count += 1
                total += 1.0 - genre_jaccard(genre_sets[i], genre_sets[j])
        if pair_count:
            diversity_values.append(total / pair_count)
    diversity = float(np.mean(diversity_values)) if diversity_values else np.nan
    summary_rows.append({
        "model_name": model_name,
        "duplicate_recommendation_rate": round(float(duplicate_rate), 4),
        "genre_consistency": round(float(seed_overlap), 4),
        "recommendation_diversity": round(float(diversity), 4) if not np.isnan(diversity) else "",
        "precision_at_5": "",
        "precision_at_10": "",
        "ndcg_at_10": "",
        "manual_ratings_received": 0,
        "manual_rating_threshold": ">= 2",
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULT_SUMMARY, index=False, encoding="utf-8-sig")

report_lines = [
    "# 4Cinephiles manual evaluation report",
    "",
    "This report reflects current system outputs and model characteristics for the available recommendation implementations.",
    "Manual relevance labels are intentionally blank until a human rater reviews the generated recommendations.",
    "",
    "## Seed movies",
    "",
    "- Dangal (tmdb_id=360814)",
    "- 3 Idiots (tmdb_id=20453)",
    "- Secret Superstar (tmdb_id=441889)",
    "- The Dark Knight (tmdb_id=155)",
    "- Iron Man (tmdb_id=1726)",
    "- Interstellar (tmdb_id=157336)",
    "- Hera Pheri (tmdb_id=21614)",
    "- Drishyam (tmdb_id=352173)",
    "",
    "## Models evaluated",
    "",
    "- V2",
    "- V3",
    "- V4 (50:50)",
    "- V4.1 (70:30)",
    "- V4.1 (50:50)",
    "- V4.1 (30:70)",
    "- V4.2",
    "",
    "## Non-manual quality metrics",
    "",
    "The metrics below are computed from the generated candidate lists and do not rely on a human relevance score.",
    "",
    "| model_name | duplicate_recommendation_rate | genre_consistency | recommendation_diversity |",
    "| --- | ---: | ---: | ---: |",
]
for _, row in summary_df.iterrows():
    report_lines.append(
        f"| {row['model_name']} | {row['duplicate_recommendation_rate']} | {row['genre_consistency']} | {row['recommendation_diversity']} |"
    )

report_lines.extend([
    "",
    "## Manual metric threshold",
    "",
    "Precision@5, Precision@10, and NDCG@10 are intentionally left blank until the `relevance_rating` column in `evaluation/relevance_labels.csv` is filled in.",
    "The rating scale is: 0 = completely irrelevant; 1 = slightly related; 2 = relevant; 3 = highly relevant.",
    "For Precision@K and NDCG@10, a recommendation is treated as relevant when `relevance_rating >= 2`.",
    "",
    "## Interpretation",
    "",
    "- The manual review step must be completed before selecting a final production model.",
    "- This report intentionally avoids claiming a winning model without human relevance labels.",
    "- The existing issue patterns (generic genre recommendations, title-reference traps, and franchise-like false positives) remain important review criteria during manual scoring.",
    "- Bollywood and Hollywood recommendations should be judged separately during the manual pass because their metadata quality and thematic overlap differ.",
    "",
    "## Output files",
    "",
    "- `evaluation/seeds.csv`",
    "- `evaluation/model_recommendations.csv`",
    "- `evaluation/relevance_labels.csv`",
    "- `evaluation/evaluation_results.csv`",
    "- `evaluation/evaluation_report.md`",
])

REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print(f"Wrote {MODEL_RESULTS}")
print(f"Wrote {LABELS}")
print(f"Wrote {RESULT_SUMMARY}")
print(f"Wrote {REPORT}")
print(f"Generated {len(results)} recommendation rows across {results['model_name'].nunique()} model variants.")
