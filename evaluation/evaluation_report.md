# 4Cinephiles relevance evaluation

## Status and scope

All recommendation rows are now rated. Labels combine AI-assisted metadata judgments with five user-provided manual decisions. In addition, an independent human validation sample of 30 AI-labelled pairs was reviewed, and all 30 ratings were confirmed. The remaining AI-labelled pairs have not been independently human-verified.

- Recommendation rows: 560 (8 seeds × 7 model configurations × 10 ranks).
- Distinct `(seed_movie_id, recommended_movie_id)` pairs: 211.
- AI-rated unique pairs: 206.
- User-provided manual ratings: 5 unique pairs, mapped consistently to all 11 matching recommendation rows.
- Rated recommendation rows: 560/560.
- Complete seed/model lists: 56/56 (8 seeds × 7 model variants).
- Repeated model-configuration rows avoided by evaluating each pair once: 349.

The recommendation algorithms, scores, model files, and generated recommendation records were not modified. Relevance ratings were assigned without using recommendation similarity scores.

## Independent human validation (30-pair sample)

Thirty recommendation pairs were independently reviewed by a human using their exact `pair_id` and compared with the corresponding AI-assigned ratings. All 30 AI ratings were confirmed.

- Exact agreement: 30/30 pairs.
- Agreement percentage: 100%.
- Agreement by rating: rating 0 — 3/3 (100%); rating 1 — 10/10 (100%); rating 2 — 9/9 (100%); rating 3 — 8/8 (100%).

Confusion matrix (rows are human ratings; columns are AI ratings):

| Human \ AI | 0 | 1 | 2 | 3 |
|---:|---:|---:|---:|---:|
| 0 | 3 | 0 | 0 | 0 |
| 1 | 0 | 10 | 0 | 0 |
| 2 | 0 | 0 | 9 | 0 |
| 3 | 0 | 0 | 0 | 8 |

This 30-pair audit is a validation sample, not a replacement for the AI labels across the full dataset. Its scope is limited to the reviewed pairs; the sample's representativeness is not established, it reflects one human reviewer, and agreement on these pairs does not establish the correctness of the remaining AI labels.

## Method and rating scale

The AI judgments used the actual movie dataset's overview/storyline, genres, keywords, cast, and director fields where available. Plot/theme evidence is primary; genre is supporting evidence only. Five pairs with insufficient or potentially unreliable story information were left for the user to judge rather than being assigned fabricated AI ratings.

The user supplied these manual ratings:

| Seed | Candidate | Rating |
|---|---|---:|
| The Dark Knight | Amidst the Devil's Wings | 2 |
| 3 Idiots | Shararat | 3 |
| 3 Idiots | Jacob Cardoso | 0 |
| Dangal | Baghban | 1 |
| Secret Superstar | Jashnn: The Music Within | 1 |

- **0** — Completely irrelevant.
- **1** — Slightly related; only broad or weak similarities.
- **2** — Relevant; meaningful similarities in theme, subject, or storyline.
- **3** — Highly relevant; strong similarities in storyline, theme, or concept.

Precision@K counts ratings **>= 2** as relevant. NDCG@10 uses graded gains `2^rating - 1`. Precision and NDCG are averaged per seed across all eight seeds.

## Model comparison

| Model | Precision@5 | Precision@10 | NDCG@10 | Duplicate rate | Genre consistency | Diversity |
|---|---:|---:|---:|---:|---:|---:|
| V2 | 0.7000 | 0.6875 | 0.8788 | 0.0000 | 0.7735 | 0.2980 |
| V3 | 0.6250 | 0.5375 | **0.9716** | 0.0000 | 0.3987 | **0.6772** |
| V4 (50:50) | 0.6750 | **0.7125** | 0.8550 | 0.0000 | 0.6742 | 0.4608 |
| V4.1 (30:70) | 0.6500 | **0.7125** | 0.9112 | 0.0000 | 0.8165 | 0.2772 |
| V4.1 (50:50) | **0.7000** | **0.7125** | 0.8723 | 0.0000 | 0.8623 | 0.2066 |
| V4.1 (70:30) | **0.7000** | **0.7125** | 0.8793 | 0.0000 | **0.9098** | 0.1475 |
| V4.2 | **0.7000** | **0.7125** | 0.9214 | 0.0000 | 0.7354 | 0.3911 |

Duplicate rate is the share of repeated candidate IDs within each model's per-seed lists. Genre consistency is mean seed-to-candidate genre Jaccard overlap. Recommendation diversity is mean pairwise genre Jaccard distance within each seed's list. These list-level metrics use recommendation metadata, not relevance ratings.

The full eight-seed comparison is now covered, but the labels remain predominantly AI-generated and the sample is small. These metrics do not prove real-world user satisfaction. **No final model is selected.**

## Files and audit trail

- `evaluation/model_recommendations.csv` — unchanged recommendation records.
- `evaluation/ai_relevance_labels.csv` — unchanged record of the 206 AI ratings and five original manual-review flags.
- `evaluation/manual_review_queue.csv` — source metadata and previous AI rationale for the five flagged pairs.
- `evaluation/relevance_labels.csv` — all 560 workflow rows are rated; manually supplied decisions are noted in `reviewer_notes`.
- `evaluation/evaluation_results.csv` — full-coverage metrics for all seven model variants.

The AI-only file intentionally retains its original five “Manual review required” statuses; the user's ratings were entered in the workflow rating sheet and were not relabeled as AI judgments.

## Limitations and next step

AI judgments can be inconsistent or overstate thematic similarity. Movie metadata may be incomplete or inaccurate. The 30-pair validation is only a sample of the 206 AI-rated unique pairs; its representativeness is not established, it involves one human reviewer, and it does not validate the remaining AI judgments. The five original manual decisions resolve missing labels but are separate from this validation sample. For a model decision, independently audit a representative sample of AI labels and evaluate with actual user preferences before consolidating a model.
