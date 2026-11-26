# 4Cinephiles

4Cinephiles is a Streamlit movie recommendation system for Bollywood and
Hollywood films. It combines TF-IDF and semantic embeddings, applies a
genre-aware adjustment, and shows matching franchises separately.

## Features

- Searchable movie selection
- Hybrid TF-IDF and MiniLM recommendation scoring
- Bollywood, Hollywood, and combined-industry filtering
- Optional genre filtering
- Duplicate-title prevention
- Franchise/collection suggestions
- Local movie posters, runtime, and metadata
- Missing-poster and missing-data fallbacks

## Technologies

- Python
- Streamlit
- Pandas and NumPy
- SciPy sparse matrices
- Scikit-learn
- Sentence Transformers for model preparation
- Requests and python-dotenv for data preparation

## Project Structure

```text
app.py                     Streamlit application entry point
engine.py                  Recommendation engine and public API
ui.py                      Streamlit UI components
recommender.py             Compatibility wrapper
assets/                    Banner and fallback poster
data/                      Source and processed datasets
model/                     TF-IDF and semantic model artifacts
src/                       Dataset and model-building scripts
scripts/                   TMDB collection and enrichment utilities
evaluation/                Relevance labels and model evaluation artifacts
tests/                     Engine and Streamlit smoke tests
notebooks/                 Data analysis notebooks and scripts
docs/                      Additional project documentation
```

## Installation

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Configuration

The running application does not require a TMDB API key because posters and
metadata are loaded from local processed files. TMDB credentials are only
needed when regenerating or enriching data.

Copy `.env.example` to `.env`, then set a key or an access token:

```text
TMDB_API_KEY=your_api_key
TMDB_ACCESS_TOKEN=your_read_access_token
```

Never commit `.env` or credentials. `.gitignore` excludes the local
environment file.

## Run the Application

From the project root:

```powershell
streamlit run app.py
```

## Recommendation Method

The current engine uses the existing V4.2 artifacts:

1. Normalize TF-IDF vectors and semantic embeddings.
2. Calculate 50% TF-IDF cosine similarity and 50% semantic cosine similarity.
3. Apply the genre adjustment:
   `final = hybrid * (0.8 + 0.2 * genre_jaccard)`.
4. Exclude the selected film and normalized duplicate titles.
5. Apply industry and genre filters.
6. Rank the remaining films and place franchise collection matches in a
   separate related-movies section.

The evaluated weights and genre adjustment are configuration defaults, not
runtime-tuned results.

## Evaluation

The `evaluation/` folder contains relevance labels, manual and AI-assisted
rating workflows, generated recommendation records, and
`evaluation/evaluation_report.md`. The report compares V2, V3, V4, V4.1, and
V4.2 variants. Results include a limited human-validation sample and remain
predominantly AI-assisted, so they do not establish real-world user
satisfaction.

## Testing

```powershell
python -m pytest tests -q
python -m compileall app.py engine.py recommender.py ui.py src scripts evaluation tests
```

## Limitations

- “Hollywood” means the TMDB 5000 source set and includes some non-English
  films.
- Missing Bollywood text metadata may reduce text-similarity quality.
- Evaluation coverage is small and includes AI-assisted judgments.
- The final model is not retrained at application startup.

## Author

Abdus Sami
