"""
Compatibility wrapper.

Keeps the original contract, recommend(movie_title) -> (titles, movie_ids),
but is now backed by engine.RecommendationEngine (unified dataset, Bollywood +
Hollywood, hybrid model). The returned ids are TMDB ids, the same numbers the
legacy `movie_id` column held, so the old fetch_poster() still works with them.

The original file is preserved as recommender_legacy.py.
"""

from engine import get_engine


def recommend(movie, top_n=5):
    engine = get_engine()

    matches = engine.find_by_title(movie)
    if not matches:
        raise ValueError(f"Movie not found: {movie!r}")

    # Several films can share a title; use the most-voted one.
    result = engine.recommend(matches[0], top_n=top_n)

    titles = [item["title"] for item in result["recommendations"]]
    ids = [item["tmdb_id"] for item in result["recommendations"]]
    return titles, ids
