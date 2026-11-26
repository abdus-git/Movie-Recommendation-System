"""
4Cinephiles recommendation engine.

One importable module that replaces the legacy movie_list.pkl / similarity.pkl
pipeline. It reuses the EXISTING model files and dataset; nothing is retrained:

    data/processed/unified_movies_enriched.csv   movies (5,721 rows)
    data/processed/movie_extras.csv              runtime, posters, franchise (optional)
    model/tfidf_matrix_v2.pkl                    TF-IDF V2 matrix
    model/semantic_embeddings_v3.pkl             MiniLM embeddings

Scoring follows recommendation_engine_v4_2.py:
    hybrid = 0.5 * TF-IDF cosine + 0.5 * semantic cosine
    final  = hybrid * (1 - 0.2 + 0.2 * genre_jaccard)     (no factor if genres missing)

Recommendation behavior:
    * "Science Fiction" and "TV Movie" are single genres (V4.2 split them in two).
    * All films with the same normalized title as the seed are excluded.
    * Duplicate recommendation titles are removed; franchise entries not already
      recommended are returned separately as "related".
    * Cosine similarities are computed with numpy/scipy (no scikit-learn, no
      sentence-transformers needed at run time) and the genre factor is vectorised.

Nothing is printed or read at import time.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "processed" / "unified_movies_enriched.csv"
EXTRAS_PATH = ROOT / "data" / "processed" / "movie_extras.csv"
TFIDF_PATH = ROOT / "model" / "tfidf_matrix_v2.pkl"
SEMANTIC_PATH = ROOT / "model" / "semantic_embeddings_v3.pkl"

IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# TMDB genre names that contain a space. They are protected before splitting.
MULTIWORD_GENRES = {"Science Fiction": "Science_Fiction", "TV Movie": "TV_Movie"}

REQUIRED_COLUMNS = [
    "tmdb_id", "title", "industry", "release_date", "overview", "genres",
    "keywords", "cast", "director", "vote_average", "vote_count",
]


@dataclass(frozen=True)
class EngineConfig:
    """V4.2 scoring defaults: 50:50 blend and 0.20 genre adjustment."""

    tfidf_weight: float = 0.50
    semantic_weight: float = 0.50
    genre_adjustment: float = 0.20
    exclude_unreleased: bool = False
    min_votes: int = 0
    separate_franchise: bool = True


def normalize_title(title) -> str:
    return re.sub(r"\s+", " ", str(title).strip().lower())


def parse_genres(value) -> list[str]:
    """'Action Science Fiction' -> ['Action', 'Science Fiction']."""
    text = str(value)
    for name, token in MULTIWORD_GENRES.items():
        text = re.sub(re.escape(name), token, text, flags=re.IGNORECASE)

    genres: list[str] = []
    for token in re.split(r"[,|;/\s]+", text):
        token = token.strip().replace("_", " ")
        if token and token not in genres:
            genres.append(token)
    return genres


def _blank_to_none(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return None if text == "" or text.lower() in {"nan", "<na>", "nat"} else value


class RecommendationEngine:
    """Loads the dataset and models once, then answers recommendation queries."""

    def __init__(self, data_path=None, extras_path=None, tfidf_path=None,
                 semantic_path=None, config: EngineConfig | None = None,
                 today=None):

        self.config = config or EngineConfig()
        data_path = Path(data_path or DATA_PATH)
        extras_path = Path(extras_path or EXTRAS_PATH)
        tfidf_path = Path(tfidf_path or TFIDF_PATH)
        semantic_path = Path(semantic_path or SEMANTIC_PATH)

        movies = pd.read_csv(data_path).fillna("").reset_index(drop=True)

        missing = [c for c in REQUIRED_COLUMNS if c not in movies.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")

        with open(tfidf_path, "rb") as file:
            tfidf = sparse.csr_matrix(pickle.load(file))
        with open(semantic_path, "rb") as file:
            embeddings = np.asarray(pickle.load(file), dtype=np.float64)

        if not (len(movies) == tfidf.shape[0] == embeddings.shape[0]):
            raise ValueError(
                f"Dataset and models do not line up: {len(movies)} movies, "
                f"{tfidf.shape[0]} TF-IDF rows, {embeddings.shape[0]} embeddings."
            )

        self.movies = self._attach_extras(movies, extras_path)
        self._prepare_columns(today)
        self._prepare_models(tfidf, embeddings)
        self._prepare_genres()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_extras(movies: pd.DataFrame, extras_path: Path) -> pd.DataFrame:
        """Left-join runtime, posters and collection data. Row order is preserved."""
        columns = ["runtime", "poster_path_x", "backdrop_path_x",
                   "collection_id", "collection_name"]

        if extras_path.exists() and extras_path.stat().st_size > 0:
            extras = pd.read_csv(extras_path).drop_duplicates(
                "tmdb_id", keep="last")
            extras = extras.rename(columns={"poster_path": "poster_path_x",
                                            "backdrop_path": "backdrop_path_x"})
            movies = movies.merge(extras, on="tmdb_id", how="left")
        else:
            for column in columns:
                movies[column] = np.nan

        return movies.reset_index(drop=True)

    def _prepare_columns(self, today) -> None:
        m = self.movies
        today = pd.Timestamp(today) if today is not None \
            else pd.Timestamp.today().normalize()

        release = pd.to_datetime(m["release_date"], errors="coerce")
        m["year"] = release.map(lambda d: "" if pd.isna(d) else str(d.year))

        self._unreleased = (release > today).to_numpy()
        self._votes = pd.to_numeric(m["vote_count"], errors="coerce") \
            .fillna(0).to_numpy()
        self._industry = m["industry"].to_numpy()
        self._collection_id = pd.to_numeric(
            m["collection_id"], errors="coerce").to_numpy(dtype=float)
        self._title_key = [normalize_title(t) for t in m["title"]]
        self._id_to_index = {}
        for i, tmdb_id in enumerate(m["tmdb_id"]):
            try:
                self._id_to_index[int(tmdb_id)] = i
            except (KeyError, ValueError, TypeError):
                continue

    def _prepare_models(self, tfidf, embeddings) -> None:
        # Row-normalise the TF-IDF matrix so a dot product is a cosine.
        norms = np.sqrt(tfidf.multiply(tfidf).sum(axis=1)).A1
        inverse = np.divide(1.0, norms, out=np.zeros_like(norms),
                            where=norms > 0)
        self._tfidf = (sparse.diags(inverse) @ tfidf).tocsr()

        lengths = np.linalg.norm(embeddings, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        self._embeddings = embeddings / lengths

    def _prepare_genres(self) -> None:
        self.movies["genre_list"] = [parse_genres(g) for g in self.movies["genres"]]
        vocabulary = sorted({g for gl in self.movies["genre_list"] for g in gl})
        self._genre_index = {g: i for i, g in enumerate(vocabulary)}

        matrix = np.zeros((len(self.movies), len(vocabulary)), dtype=np.float64)
        for row, genre_list in enumerate(self.movies["genre_list"]):
            for genre in genre_list:
                matrix[row, self._genre_index[genre]] = 1.0

        self._genre_matrix = matrix
        self._genre_counts = matrix.sum(axis=1)

    # ------------------------------------------------------------------
    # Public lookups
    # ------------------------------------------------------------------

    @property
    def genres(self) -> list[str]:
        return list(self._genre_index)

    def stats(self) -> dict:
        counts = self.movies["industry"].value_counts().to_dict()
        return {"total": len(self.movies), **counts}

    def index_of(self, tmdb_id) -> int:
        try:
            return self._id_to_index[int(tmdb_id)]
        except (KeyError, ValueError, TypeError):
            raise KeyError(f"Unknown tmdb_id: {tmdb_id!r}") from None

    def label(self, tmdb_id) -> str:
        row = self.movies.iloc[self.index_of(tmdb_id)]
        year = f" ({row['year']})" if row["year"] else ""
        return f"{row['title']}{year} · {row['industry']}"

    def choices(self) -> list[tuple[str, int]]:
        """[(label, tmdb_id)] sorted by title, for a selection box."""
        m = self.movies
        years = m["year"].map(lambda y: f" ({y})" if y else "")
        labels = m["title"].astype(str) + years + " · " + m["industry"].astype(str)
        order = np.argsort(m["title"].astype(str).str.casefold().to_numpy(),
                           kind="stable")
        return [(labels.iat[i], int(m["tmdb_id"].iat[i])) for i in order]

    def find_by_title(self, title: str) -> list[int]:
        """tmdb_ids whose title matches exactly (case-insensitive), most voted first."""
        key = normalize_title(title)
        hits = [i for i, k in enumerate(self._title_key) if k == key]
        hits.sort(key=lambda i: -self._votes[i])
        return [int(self.movies["tmdb_id"].iat[i]) for i in hits]

    def search(self, query: str, industry: str | None = None,
               limit: int = 25) -> list[dict]:
        """Title search: exact matches first, then prefix, then substring."""
        key = normalize_title(query)
        if not key:
            return []

        scored = []
        for i, title_key in enumerate(self._title_key):
            if industry and self._industry[i] != industry:
                continue
            if title_key == key:
                tier = 0
            elif title_key.startswith(key):
                tier = 1
            elif key in title_key:
                tier = 2
            else:
                continue
            scored.append((tier, -self._votes[i], i))

        scored.sort()
        return [self._movie_dict(i) for _, _, i in scored[:limit]]

    def details(self, tmdb_id) -> dict:
        return self._movie_dict(self.index_of(tmdb_id))

    def related_films(self, tmdb_id) -> list[dict]:
        """Other films in the same TMDB collection, oldest first."""
        i = self.index_of(tmdb_id)
        return self._related(i)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _scores(self, i: int):
        """Return (final, tfidf, semantic) score arrays for seed row i."""
        cfg = self.config

        tfidf_scores = (self._tfidf @ self._tfidf[i].T).toarray().ravel()
        semantic_scores = self._embeddings @ self._embeddings[i]
        hybrid = (cfg.tfidf_weight * tfidf_scores
                  + cfg.semantic_weight * semantic_scores)

        factor = np.ones(len(hybrid))
        if self._genre_counts[i] > 0 and cfg.genre_adjustment:
            intersection = self._genre_matrix @ self._genre_matrix[i]
            union = self._genre_counts + self._genre_counts[i] - intersection
            has_genres = self._genre_counts > 0
            jaccard = np.divide(intersection, union,
                                out=np.zeros_like(intersection),
                                where=union > 0)
            adjusted = 1 - cfg.genre_adjustment + cfg.genre_adjustment * jaccard
            factor = np.where(has_genres, adjusted, 1.0)

        return hybrid * factor, tfidf_scores, semantic_scores

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def recommend(self, tmdb_id, top_n: int = 10, industry: str | None = None,
                  genres: list[str] | None = None) -> dict:
        """
        Recommend films using the evaluated V4.2 scoring approach.

        Returns:
            {
                "seed": dict,
                "recommendations": [dict],
                "related": [dict]
            }
        """
        cfg = self.config
        i = self.index_of(tmdb_id)

        final, tfidf_scores, semantic_scores = self._scores(i)

        allowed = np.asarray(
            [key != self._title_key[i] for key in self._title_key],
            dtype=bool
        )

        if cfg.exclude_unreleased:
            allowed &= ~self._unreleased

        if cfg.min_votes:
            allowed &= self._votes >= cfg.min_votes

        # Apply optional industry filter.
        pool = allowed.copy()

        if industry in ("Bollywood", "Hollywood"):
            pool &= self._industry == industry

        # Apply optional genre filter.
        if genres:
            columns = [
                self._genre_index[g]
                for g in genres
                if g in self._genre_index
            ]

            if columns:
                pool &= self._genre_matrix[:, columns].sum(axis=1) > 0
            else:
                pool[:] = False

        # Rank directly using the V4.2 final score.
        candidates = np.flatnonzero(pool)
        order = candidates[
            np.argsort(-final[candidates], kind="stable")
        ]

        # Remove duplicate movie titles.
        selected = []
        seen = set()

        for j in order:
            key = self._title_key[j]

            if key in seen:
                continue

            seen.add(key)
            selected.append(int(j))

            if len(selected) >= top_n:
                break

        # Prepare recommendation results.
        recommendations = []

        for j in selected:
            item = self._movie_dict(j)

            item["score"] = round(float(final[j]), 4)
            item["tfidf_score"] = round(float(tfidf_scores[j]), 4)
            item["semantic_score"] = round(float(semantic_scores[j]), 4)

            item["reasons"] = self._reasons(
                i, j, tfidf_scores, semantic_scores, pool)

            recommendations.append(item)

        # Show franchise movies separately only if they are not
        # already present in the recommendation list.
        recommended_ids = {
            item["tmdb_id"] for item in recommendations
        }

        related = [
            item for item in self._related(i)
            if item["tmdb_id"] not in recommended_ids
        ] if cfg.separate_franchise else []

        return {
            "seed": self._movie_dict(i),
            "recommendations": recommendations,
            "related": related,
        }

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------

    def _reasons(self, i, j, tfidf_scores, semantic_scores, pool) -> list[str]:
        """Plain-language reasons that come from the actual computation."""
        reasons = []

        # Which signal ranks this film higher among the candidates?
        scope = pool if pool.any() else np.ones(len(pool), dtype=bool)
        semantic_top = float((semantic_scores[scope] > semantic_scores[j]).mean())
        text_top = float((tfidf_scores[scope] > tfidf_scores[j]).mean())

        if semantic_top <= text_top:
            share = max(1, int(round(semantic_top * 100)))
            reasons.append(f"Similar plot and themes (top {share}% semantic match)")
        else:
            share = max(1, int(round(text_top * 100)))
            reasons.append(
                f"Similar keywords, cast and crew wording (top {share}% text match)")

        shared = [g for g in self.movies["genre_list"].iat[j]
                  if g in self.movies["genre_list"].iat[i]]
        if shared:
            reasons.append("Shared genres: " + ", ".join(shared))

        director = str(self.movies["director"].iat[i]).strip()
        if director and director == str(self.movies["director"].iat[j]).strip():
            reasons.append(f"Same director: {director}")

        return reasons

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _related(self, i: int) -> list[dict]:
        collection = self._collection_id[i]
        if np.isnan(collection):
            return []

        rows = np.flatnonzero(self._collection_id == collection)
        rows = [int(r) for r in rows if r != i
                and not (self.config.exclude_unreleased and self._unreleased[r])]
        rows.sort(key=lambda r: (self.movies["year"].iat[r] or "9999", r))
        return [self._movie_dict(r) for r in rows]

    def _poster_url(self, i: int):
        for column in ("poster_path_x", "poster_path"):
            if column not in self.movies.columns:
                continue

            path = _blank_to_none(self.movies[column].iat[i])

            if path:
                return IMAGE_BASE + str(path)

        return None

    def _movie_dict(self, i: int) -> dict:
        row = self.movies.iloc[i]

        votes = int(self._votes[i])
        rating = pd.to_numeric(row["vote_average"], errors="coerce")
        runtime = pd.to_numeric(row["runtime"], errors="coerce")
        collection_name = _blank_to_none(row["collection_name"])

        return {
            "tmdb_id": int(row["tmdb_id"]),
            "title": str(row["title"]),
            "industry": str(row["industry"]),
            "year": row["year"],
            "genres": list(row["genre_list"]),
            "rating": None if votes == 0 or pd.isna(rating) else float(rating),
            "vote_count": votes,
            "runtime": None if pd.isna(runtime) or runtime <= 0 else int(runtime),
            "overview": str(row["overview"]),
            "director": str(row["director"]),
            "cast": str(row["cast"]),
            "poster_url": self._poster_url(i),
            "collection_name": None if collection_name is None
            else str(collection_name),
        }


@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    """Shared engine instance for scripts and the compatibility wrapper."""
    return RecommendationEngine()
