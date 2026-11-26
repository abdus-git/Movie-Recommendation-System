# 4Cinephiles: project guide

## What it is
A Streamlit movie recommender for Bollywood and Hollywood films. Pick a film, get
similar films with posters, metadata, filters and a plain-language reason for each.

## Architecture
```
data/processed/unified_movies_enriched.csv  --+
data/processed/movie_extras.csv (optional)  --+--> engine.py --> app.py (Streamlit UI, ui.py)
model/tfidf_matrix_v2.pkl                   --+         |
model/semantic_embeddings_v3.pkl            --+         +--> recommender.py (legacy-style wrapper)
```
Models are NOT retrained or reloaded at run time; the engine reads the existing
pickles, so MiniLM, torch and scikit-learn are not needed to run the app.

## Data pipeline (existing, unchanged)
| Script | Job |
|---|---|
| `fetch_bolly.py` | Collect Hindi-language Indian films from TMDB discover |
| `build_unified_dataset.py` | Merge TMDB 5000 (Hollywood) and Bollywood, Bollywood wins on shared IDs |
| `enrich_bolly.py` | Add genres, keywords, cast, director, posters to Bollywood rows |
| `validate_enriched_dataset.py` | Dataset report |
| `scripts/fetch_movie_extras.py` | NEW: runtime, posters, franchise collection for all rows, once |
| `tfidf_model_v2.py`, `semantic_model_v3.py` | Build the two model files |

## How a recommendation is made
1. TF-IDF V2 cosine and MiniLM cosine between the selected film and every film.
2. `hybrid = 0.5 * tfidf + 0.5 * semantic`.
3. Genre adjustment: `final = hybrid * (0.8 + 0.2 * genre_jaccard)`; skipped when genres are missing.
4. Remove the film itself, unreleased films and exact duplicates; apply industry/genre filters.
5. Films in the same TMDB collection (sequels, prequels) are shown separately as "Related Films".
6. Each result carries reasons computed from the same scores (see "Explanations").

These defaults reproduce `recommendation_engine_v4_2.py`. Weights are in `EngineConfig`.

## Technical decisions (and why)
| Decision | Reason |
|---|---|
| One importable `engine.py` | The V1-V4.2 scripts load data and call `input()` at import, so they cannot be imported by the app. |
| Genre vocabulary keeps "Science Fiction" and "TV Movie" whole | V4.1/V4.2 split on whitespace, which counted them as two genres. |
| Exclude the seed by row, not title | Title exclusion also hid other films with the same name (remakes, other-industry films). |
| De-duplicate by (title, year) | Same title and year is the same film; same title and different year is not. |
| Franchise films shown separately | A blanket similarity penalty for title mentions was rejected; TMDB's collection link is explicit and reliable. |
| Hide unreleased films | The Bollywood set contains future-dated placeholder entries. |
| Posters and runtime from a one-time file | No TMDB request or API key at run time. |
| numpy/scipy cosine | Identical numbers to scikit-learn (tested), one less dependency, vectorised genre factor. |

## Explanations
Each recommendation lists: the stronger signal (semantic or text) with its percentile among
candidates, shared genres, and same director when true. Cast overlap is not reported because
the dataset stores cast as space-joined names with no delimiter, so names cannot be split reliably.
The score is a ranking signal, not a probability of liking the film.

## Evaluation
`evaluation/` contains a blinded, manual-rating evaluation of V2, V3, V4 (50:50) and V4.2:
`preflight.py` -> `build_rating_sheet.py` -> you rate `rating_sheet.csv` (0/1/2) -> `score_ratings.py`.
**Status: the engine defaults are NOT yet chosen from your ratings.** They are V4.2 plus technical fixes.
If the ratings favour another configuration, change `EngineConfig` (for example the weights) rather than the code.

## Tests
`python -m pytest tests -q` runs 25 tests on synthetic data: genre parsing, exclusions, filters, franchise
separation, missing data, numerical agreement with scikit-learn and the V4.2 formula, the compatibility
wrapper, and the whole Streamlit script run with a stand-in for Streamlit.

## Known limitations
- "Hollywood" means "TMDB 5000 set" and includes about 290 non-English films.
- 59% of Bollywood films have no keywords, which weakens TF-IDF for them.
- Only the first three cast members are stored.
- Zero-vote placeholder films that are not future-dated can still appear; raise `EngineConfig.min_votes` to hide them.
- The Streamlit interface was tested with a stand-in, not in a browser, and only on synthetic data.
