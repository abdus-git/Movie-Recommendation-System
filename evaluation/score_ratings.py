"""Compute evaluation metrics from the AI-assisted/manual relevance sheet.

Precision@K treats ratings >= 2 as relevant. NDCG uses graded ratings 0-3.
Metrics are averaged per seed. When some labels are missing, only complete
seed/model lists contribute; coverage is reported so partial results are clear.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

EVAL = Path(__file__).resolve().parent
RECOMMENDATIONS = EVAL / "model_recommendations.csv"
LABELS = EVAL / "relevance_labels.csv"
DATA = EVAL.parent / "data" / "processed" / "unified_movies_enriched.csv"
OUT = EVAL / "evaluation_results.csv"


def parse_genres(value) -> set[str]:
    if pd.isna(value):
        return set()
    text = str(value)
    text = re.sub(r"Science Fiction", "Science_Fiction", text, flags=re.IGNORECASE)
    text = re.sub(r"TV Movie", "TV_Movie", text, flags=re.IGNORECASE)
    return {
        token.strip().replace("_", " ").lower()
        for token in re.split(r"[,|;/\s]+", text)
        if token.strip()
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def precision_at_k(ratings: list[float], k: int) -> float:
    return sum(rating >= 2 for rating in ratings[:k]) / min(k, len(ratings))


def ndcg_at_10(ratings: list[float]) -> float:
    top = ratings[:10]

    def dcg(values: list[float]) -> float:
        return sum(
            (2 ** int(value) - 1) / math.log2(rank + 2)
            for rank, value in enumerate(values)
        )

    ideal = sorted(top, reverse=True)
    ideal_dcg = dcg(ideal)
    return 0.0 if ideal_dcg == 0 else dcg(top) / ideal_dcg


if not RECOMMENDATIONS.exists():
    raise FileNotFoundError(f"Expected recommendation file missing: {RECOMMENDATIONS}")
if not LABELS.exists():
    raise FileNotFoundError(f"Expected relevance labels file missing: {LABELS}")

recommendations = pd.read_csv(RECOMMENDATIONS, encoding="utf-8-sig")
labels = pd.read_csv(LABELS, encoding="utf-8-sig")
required_recommendation_columns = {
    "seed_movie", "seed_movie_id", "model_name", "recommended_movie",
    "recommended_movie_id", "rank", "genre",
}
required_label_columns = {
    "seed_movie", "seed_movie_id", "model_name", "recommended_movie",
    "recommended_movie_id", "rank", "relevance_rating",
}
missing_recs = required_recommendation_columns - set(recommendations.columns)
missing_labels = required_label_columns - set(labels.columns)
if missing_recs or missing_labels:
    raise ValueError(
        f"Missing required columns; recommendations={sorted(missing_recs)}, "
        f"labels={sorted(missing_labels)}"
    )

labels["relevance_rating"] = pd.to_numeric(labels["relevance_rating"], errors="coerce")
bad = labels["relevance_rating"].dropna()
bad = bad[~bad.isin([0, 1, 2, 3])]
if len(bad):
    raise ValueError(f"Ratings must be integers in [0, 1, 2, 3]. Found: {sorted(bad.unique())}")

key = ["seed_movie_id", "recommended_movie_id", "model_name"]
if labels.duplicated(key).any():
    raise ValueError("The rating sheet has duplicate seed/candidate/model rows.")

merged = recommendations.merge(
    labels[key + ["relevance_rating"]],
    on=key,
    how="left",
    validate="one_to_one",
    suffixes=("", "_label"),
)
if len(merged) != len(recommendations):
    raise ValueError("Recommendation rows did not map one-to-one to relevance labels.")

if not DATA.exists():
    raise FileNotFoundError(f"Expected movie dataset missing: {DATA}")
movie_genres = pd.read_csv(DATA, encoding="utf-8-sig", usecols=["tmdb_id", "genres"])
if movie_genres["tmdb_id"].duplicated().any():
    raise ValueError("Movie dataset has duplicate TMDB IDs; genre metrics cannot be keyed reliably.")
genres_by_id = movie_genres.set_index("tmdb_id")["genres"].to_dict()

models = sorted(merged["model_name"].unique())
complete_seeds_by_model = {}
for model_name in models:
    model_recs = merged[merged["model_name"] == model_name]
    complete_seeds = set()
    for seed_id, seed_group in model_recs.groupby("seed_movie_id", sort=True):
        seed_group = seed_group.sort_values("rank")
        if (
            len(seed_group) >= 10
            and seed_group["rank"].head(10).tolist() == list(range(1, 11))
            and seed_group["relevance_rating"].head(10).notna().all()
        ):
            complete_seeds.add(seed_id)
    complete_seeds_by_model[model_name] = complete_seeds

common_complete_seeds = (
    set.intersection(*(complete_seeds_by_model[model] for model in models))
    if models else set()
)

summary_rows = []
for model_name in models:
    model_recs = merged[merged["model_name"] == model_name]
    seed_metrics = []
    total_seed_lists = model_recs["seed_movie_id"].nunique()
    model_complete_seed_lists = len(complete_seeds_by_model[model_name])
    rated_rows = int(model_recs["relevance_rating"].notna().sum())

    for seed_id in sorted(common_complete_seeds):
        seed_group = model_recs[model_recs["seed_movie_id"] == seed_id].sort_values("rank")
        top_ratings = [float(value) for value in seed_group["relevance_rating"].head(10)]
        seed_metrics.append({
            "precision_at_5": precision_at_k(top_ratings, 5),
            "precision_at_10": precision_at_k(top_ratings, 10),
            "ndcg_at_10": ndcg_at_10(top_ratings),
        })

    def average_metric(name):
        if not seed_metrics:
            return ""
        return round(sum(item[name] for item in seed_metrics) / len(seed_metrics), 4)

    duplicate_count = 0
    genre_overlap_values = []
    diversity_values = []
    for _, seed_group in model_recs.groupby("seed_movie_id", sort=True):
        duplicate_count += len(seed_group) - seed_group["recommended_movie_id"].nunique()
        seed_id = seed_group["seed_movie_id"].iloc[0]
        if seed_id not in genres_by_id:
            raise ValueError(f"Seed movie ID {seed_id} is missing from the dataset.")
        source_genres = parse_genres(genres_by_id[seed_id])
        genre_sets = [parse_genres(value) for value in seed_group["genre"]]
        genre_overlap_values.extend(jaccard(source_genres, genres) for genres in genre_sets)
        pairwise = [
            1.0 - jaccard(genre_sets[i], genre_sets[j])
            for i in range(len(genre_sets))
            for j in range(i + 1, len(genre_sets))
        ]
        if pairwise:
            diversity_values.append(sum(pairwise) / len(pairwise))

    row_count = len(model_recs)
    summary_rows.append({
        "model_name": model_name,
        "duplicate_recommendation_rate": round(duplicate_count / row_count, 4) if row_count else "",
        "genre_consistency": round(sum(genre_overlap_values) / len(genre_overlap_values), 4)
        if genre_overlap_values else "",
        "recommendation_diversity": round(sum(diversity_values) / len(diversity_values), 4)
        if diversity_values else "",
        "precision_at_5": average_metric("precision_at_5"),
        "precision_at_10": average_metric("precision_at_10"),
        "ndcg_at_10": average_metric("ndcg_at_10"),
        "manual_ratings_received": rated_rows,
        "model_complete_seed_lists": model_complete_seed_lists,
        "common_comparison_seed_lists": len(common_complete_seeds),
        "total_seed_lists": total_seed_lists,
        "label_completion": "Complete" if rated_rows == len(model_recs) else "Incomplete",
        "manual_rating_threshold": ">= 2",
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT, index=False, encoding="utf-8-sig")

print(f"Saved {OUT}")
for row in summary.itertuples(index=False):
    print(
        f"{row.model_name}: {row.manual_ratings_received} rated rows; "
        f"{row.model_complete_seed_lists}/{row.total_seed_lists} individually complete seed lists; "
        f"{row.common_comparison_seed_lists} common comparison seed lists; "
        f"Precision@5={row.precision_at_5}; Precision@10={row.precision_at_10}; "
        f"NDCG@10={row.ndcg_at_10}"
    )
