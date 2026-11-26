import re

import numpy as np
import pytest

from conftest import movie_row, write_project, SMALL_ROWS
from engine import (EngineConfig, RecommendationEngine, normalize_title,
                    parse_genres)


# ---------- genre parsing ----------

def test_multiword_genres_stay_whole():
    assert parse_genres("Action Science Fiction Thriller") == \
        ["Action", "Science Fiction", "Thriller"]
    assert parse_genres("TV Movie Drama") == ["TV Movie", "Drama"]
    assert parse_genres("") == []


# ---------- loading ----------

def test_construction_prints_nothing(small_paths, capsys):
    RecommendationEngine(**small_paths, today="2026-01-01")
    assert capsys.readouterr().out == ""


def test_misaligned_models_are_rejected(tmp_path):
    paths = write_project(tmp_path, SMALL_ROWS)
    rows = SMALL_ROWS[:-1]                       # CSV now has one row fewer
    import pandas as pd
    from conftest import COLUMNS
    pd.DataFrame(rows, columns=COLUMNS).to_csv(paths["data_path"], index=False)
    with pytest.raises(ValueError, match="do not line up"):
        RecommendationEngine(**paths)


def test_works_without_extras_file(tmp_path):
    paths = write_project(tmp_path, SMALL_ROWS, extras=None)
    engine = RecommendationEngine(**paths, today="2026-01-01")
    movie = engine.details(12)
    assert movie["runtime"] is None
    assert movie["poster_url"].endswith("/unified_poster.jpg")   # falls back
    assert engine.related_films(1) == []


# ---------- recommendations ----------

def test_seed_and_same_title_films_are_excluded(small_engine):
    ids = [r["tmdb_id"] for r in small_engine.recommend(5, top_n=20)["recommendations"]]
    assert 5 not in ids
    assert 6 not in ids      # the other "Twin" has the same normalized title


def test_exact_duplicate_records_are_collapsed(small_engine):
    ids = [r["tmdb_id"] for r in small_engine.recommend(1, top_n=20)["recommendations"]]
    assert len({10, 11} & set(ids)) == 1       # same title + year -> one entry


def test_unreleased_films_are_allowed_by_default(small_engine):
    ids = [r["tmdb_id"] for r in small_engine.recommend(1, top_n=20)["recommendations"]]
    assert 7 in ids


def test_unreleased_can_be_excluded(small_paths):
    engine = RecommendationEngine(**small_paths, today="2026-01-01",
                                  config=EngineConfig(exclude_unreleased=True))
    ids = [r["tmdb_id"] for r in engine.recommend(1, top_n=20)["recommendations"]]
    assert 7 not in ids


def test_industry_filter(small_engine):
    recs = small_engine.recommend(1, top_n=20, industry="Bollywood")["recommendations"]
    assert recs and all(r["industry"] == "Bollywood" for r in recs)


def test_genre_filter_is_any_of(small_engine):
    recs = small_engine.recommend(1, top_n=20, genres=["Comedy", "Romance"])["recommendations"]
    assert {r["tmdb_id"] for r in recs} == {6, 9}
    assert small_engine.recommend(1, genres=["Nonexistent"])["recommendations"] == []


def test_franchise_films_are_recommended_or_returned_as_related(small_engine):
    result = small_engine.recommend(1, top_n=20)
    recommendation_ids = [r["tmdb_id"] for r in result["recommendations"]]
    related_ids = [r["tmdb_id"] for r in result["related"]]
    assert 2 in recommendation_ids
    assert 2 not in related_ids


def test_unrecommended_franchise_films_remain_related(small_engine):
    result = small_engine.recommend(1, top_n=20, genres=["Comedy"])
    assert 2 not in [r["tmdb_id"] for r in result["recommendations"]]
    assert [r["tmdb_id"] for r in result["related"]] == [2]
    assert result["related"][0]["collection_name"] == "Alpha Collection"


def test_franchise_separation_can_be_switched_off(small_paths):
    engine = RecommendationEngine(**small_paths, today="2026-01-01",
                                  config=EngineConfig(separate_franchise=False))
    ids = [r["tmdb_id"] for r in engine.recommend(1, top_n=20)["recommendations"]]
    assert 2 in ids
    assert engine.recommend(1, top_n=20)["related"] == []


def test_result_fields_and_missing_values(small_engine):
    rec = {r["tmdb_id"]: r for r in
           small_engine.recommend(1, top_n=20)["recommendations"]}
    beta = rec[3]
    assert beta["runtime"] is None            # TMDB runtime 0 -> unknown
    assert beta["poster_url"] is None         # no poster anywhere
    assert small_engine.details(1)["poster_url"].endswith("/a.jpg")
    assert small_engine.details(1)["runtime"] == 120
    assert rec[8]["genres"] == []             # no genres -> empty list, no crash


def test_poster_url_handles_missing_poster_columns(small_engine):
    small_engine.movies.drop(columns=["poster_path_x", "poster_path"], inplace=True)
    assert small_engine.details(3)["poster_url"] is None


def test_reasons_are_truthful(small_engine):
    for rec in small_engine.recommend(12, top_n=20)["recommendations"]:
        seed_genres = {"Action", "Science Fiction"}
        listed = [r for r in rec["reasons"] if r.startswith("Shared genres")]
        shared = seed_genres & set(rec["genres"])
        if shared:
            assert listed == ["Shared genres: " + ", ".join(
                g for g in rec["genres"] if g in seed_genres)]
        else:
            assert listed == []
        has_director = any(r.startswith("Same director") for r in rec["reasons"])
        assert has_director == (rec["director"] == "D1")
        assert rec["reasons"][0].startswith("Similar")


def test_search_orders_exact_then_prefix_then_substring(small_engine):
    titles = [m["title"] for m in small_engine.search("alpha")]
    assert titles == ["Alpha", "Alpha 2"]
    assert small_engine.search("   ") == []
    assert [m["title"] for m in small_engine.search("lta")] == ["Delta"]


def test_find_by_title_prefers_most_voted(tmp_path):
    rows = [movie_row(1, "Same", "Hollywood", "2000-01-01", "Drama", votes=5),
            movie_row(2, "Same", "Bollywood", "2010-01-01", "Drama", votes=500)]
    engine = RecommendationEngine(**write_project(tmp_path, rows))
    assert engine.find_by_title("same") == [2, 1]


def test_unknown_id_raises_keyerror(small_engine):
    with pytest.raises(KeyError):
        small_engine.recommend(99999)


# ---------- numerical equivalence with the V4.2 formula ----------

def test_scores_match_v42_reference(tmp_path):
    sklearn_metrics = pytest.importorskip("sklearn.metrics.pairwise")
    rng = np.random.default_rng(3)
    vocab = ["Action", "Drama", "Comedy", "Crime", "Romance"]
    rows = []
    for n in range(60):
        genres = "" if n % 9 == 0 else " ".join(
            rng.choice(vocab, rng.integers(1, 4), replace=False))
        rows.append(movie_row(1000 + n, f"M{n}", "Hollywood", "2000-01-01", genres))
    paths = write_project(tmp_path, rows, zero_tfidf_rows=(4, 13))
    engine = RecommendationEngine(**paths, today="2026-01-01")

    import pickle
    tfidf = pickle.load(open(paths["tfidf_path"], "rb"))
    emb = pickle.load(open(paths["semantic_path"], "rb"))

    def tokens(value):
        return {g.lower() for g in re.split(r"[,|;/\s]+", str(value)) if g.strip()}

    for seed in (0, 9, 21, 33):
        t = sklearn_metrics.cosine_similarity(tfidf[seed], tfidf).ravel()
        s = sklearn_metrics.cosine_similarity(emb[seed].reshape(1, -1), emb).ravel()
        hybrid = 0.5 * t + 0.5 * s
        expected = hybrid.copy()
        src = tokens(rows[seed]["genres"])
        for j in range(60):
            cand = tokens(rows[j]["genres"])
            if src and cand:
                jac = len(src & cand) / len(src | cand)
                expected[j] = hybrid[j] * (1 - 0.2 + 0.2 * jac)
        final, tfidf_scores, semantic_scores = engine._scores(seed)
        np.testing.assert_allclose(tfidf_scores, t, atol=1e-6)
        np.testing.assert_allclose(semantic_scores, s, atol=1e-6)
        np.testing.assert_allclose(final, expected, atol=1e-6)


# ---------- compatibility wrapper ----------

def test_legacy_wrapper_contract(small_paths, monkeypatch):
    import engine as engine_module
    import recommender
    monkeypatch.setattr(engine_module, "DATA_PATH", small_paths["data_path"])
    monkeypatch.setattr(engine_module, "EXTRAS_PATH", small_paths["extras_path"])
    monkeypatch.setattr(engine_module, "TFIDF_PATH", small_paths["tfidf_path"])
    monkeypatch.setattr(engine_module, "SEMANTIC_PATH", small_paths["semantic_path"])
    engine_module.get_engine.cache_clear()
    try:
        titles, ids = recommender.recommend("Alpha", top_n=3)
        assert len(titles) == len(ids) == 3
        assert all(isinstance(i, int) for i in ids)
        with pytest.raises(ValueError):
            recommender.recommend("No such film")
    finally:
        engine_module.get_engine.cache_clear()
