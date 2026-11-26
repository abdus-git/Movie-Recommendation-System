"""Shared fixtures: tiny synthetic projects with the same file layout as the real one."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

COLUMNS = ["tmdb_id", "title", "original_title", "industry", "original_language",
           "release_date", "overview", "genres", "keywords", "cast", "director",
           "vote_average", "vote_count", "popularity", "poster_path",
           "backdrop_path", "movie_id", "release_status"]


def movie_row(tmdb_id, title, industry, date, genres, director="", votes=100,
              poster=""):
    return {
        "tmdb_id": tmdb_id, "title": title, "original_title": title,
        "industry": industry,
        "original_language": "hi" if industry == "Bollywood" else "en",
        "release_date": date, "overview": f"Overview of {title}",
        "genres": genres, "keywords": "k", "cast": "Actor One Actor Two",
        "director": director, "vote_average": 7.0, "vote_count": votes,
        "popularity": 1.0, "poster_path": poster, "backdrop_path": "",
        "movie_id": f"tmdb_{tmdb_id}", "release_status": "Released",
    }


def write_project(root: Path, rows, extras=None, seed=0, zero_tfidf_rows=()):
    (root / "data" / "processed").mkdir(parents=True)
    (root / "model").mkdir()

    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        root / "data" / "processed" / "unified_movies_enriched.csv", index=False)

    if extras is not None:
        pd.DataFrame(extras).to_csv(
            root / "data" / "processed" / "movie_extras.csv", index=False)

    rng = np.random.default_rng(seed)
    n = len(rows)
    dense = rng.random((n, 30)) * (rng.random((n, 30)) < 0.4)
    for z in zero_tfidf_rows:
        dense[z] = 0
    emb = rng.normal(size=(n, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    with open(root / "model" / "tfidf_matrix_v2.pkl", "wb") as f:
        pickle.dump(sparse.csr_matrix(dense), f)
    with open(root / "model" / "semantic_embeddings_v3.pkl", "wb") as f:
        pickle.dump(emb.astype("float32"), f)

    return {
        "data_path": root / "data" / "processed" / "unified_movies_enriched.csv",
        "extras_path": root / "data" / "processed" / "movie_extras.csv",
        "tfidf_path": root / "model" / "tfidf_matrix_v2.pkl",
        "semantic_path": root / "model" / "semantic_embeddings_v3.pkl",
    }


SMALL_ROWS = [
    movie_row(1, "Alpha", "Hollywood", "2010-01-01", "Action Science Fiction", "D1"),
    movie_row(2, "Alpha 2", "Hollywood", "2013-01-01", "Action Science Fiction", "D1"),
    movie_row(3, "Beta", "Bollywood", "2012-01-01", "Action Thriller", "D2"),
    movie_row(4, "Gamma", "Hollywood", "2015-01-01", "Science Fiction Adventure", "D3"),
    movie_row(5, "Twin", "Hollywood", "2005-01-01", "Drama", "D4"),
    movie_row(6, "Twin", "Bollywood", "2018-01-01", "Drama Romance", "D5"),
    movie_row(7, "Future", "Hollywood", "2099-01-01", "Action", "D6"),
    movie_row(8, "NoGenre", "Bollywood", "2001-01-01", "", ""),
    movie_row(9, "Delta", "Hollywood", "2008-01-01", "Comedy", "D7"),
    movie_row(10, "Dup", "Hollywood", "2004-01-01", "Action", "D8"),
    movie_row(11, "Dup", "Hollywood", "2004-01-01", "Action", "D9"),
    movie_row(12, "Epsilon", "Bollywood", "2016-01-01", "Action Science Fiction", "D1",
              poster="/unified_poster.jpg"),
]

SMALL_EXTRAS = [
    {"tmdb_id": 1, "runtime": 120, "poster_path": "/a.jpg", "backdrop_path": "/ab.jpg",
     "collection_id": 100, "collection_name": "Alpha Collection"},
    {"tmdb_id": 2, "runtime": 130, "poster_path": "/a2.jpg", "backdrop_path": "",
     "collection_id": 100, "collection_name": "Alpha Collection"},
    {"tmdb_id": 3, "runtime": 0, "poster_path": "", "backdrop_path": "",
     "collection_id": "", "collection_name": ""},
]


@pytest.fixture
def small_paths(tmp_path):
    return write_project(tmp_path, SMALL_ROWS, SMALL_EXTRAS,
                         zero_tfidf_rows=(7,))


@pytest.fixture
def small_engine(small_paths):
    from engine import RecommendationEngine
    return RecommendationEngine(**small_paths, today="2026-01-01")
