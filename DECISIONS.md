# Decisions

Choices made while building this project, and why. Measured values live in `reports/results.md`.

## Data

**Interactions come from `azathoth42/myanimelist`, not the newer `svanoo/myanimelist-dataset`.**
The newer crawl looked ideal because it has synopses and a `last_interaction_date`, so I audited that column first. Almost all values equal the title's air date or a single placeholder date, which means they say when a show aired and not when a user acted. A split on it would look temporal without being temporal. `azathoth42` has per-user `my_last_updated` epochs with a natural daily cycle. The audit is rerun by `make data` and reported in `reports/results.md`.

**Synopses are joined from `svanoo` by anime ID.** The interaction source has no synopses. The ID match rate is reported. Titles with no usable synopsis keep a zero text vector and still get genre, studio, type and year features. Those unmatched titles account for a small share of interactions.

**`my_last_updated` is the last edit, not the watch date.** Each user-title pair has one row, so a title re-edited later appears at the later time and is absent from earlier history. This is the main remaining limit of the temporal split.

**User sample: 100,000 of the users left after cleaning, seeded.** This keeps EASE and ALS tuning practical on 16 GB of RAM. All cleaning filters run on the full population before sampling, so the filter log describes the whole dataset.

**Study window ends 2018-05-01.** The crawl tails off during May 2018, so the partial last month is removed.

**Splits.** Train is everything before 2017-11-01, validation is 2017-11-01 to 2018-02-01, and test is 2018-02-01 to 2018-05-01. Each window is three months and spans a change of anime season, so new titles appear in it.

**Bot-like filters.** Four rules: lists over 5,000 titles; at least half of a user's rows in the same second; 50 or more ratings with one distinct score; 100 or more completions in one day. The last rule removes the most data. Most of those accounts are probably people back-filling a list rather than bots, but their timestamps do not reflect viewing order, so they cannot support a temporal evaluation.

**Adult titles and music videos are excluded** from the catalog and from evaluation, because the demo is public and because music videos are not comparable viewing decisions.

**Crawl-time aggregates are not used** (global score, rank, members, favorites). Only allow-listed metadata columns reach the catalog, and a test enforces this. Synopsis text and studio credits come from later crawls, but they describe the title rather than its audience.

## Target and evaluation

- **Label definition is fixed for evaluation** (completed and score at least 7, or above the user's fit-window mean), so all models are compared on the same labels. The training target is tuned separately on validation, and label-threshold sensitivity is reported.
- **Unrated completions are not positives**, following the brief. They still count as seen, so they are never recommended back.
- **Plan-to-watch titles do not count as seen**, so they can be recommended.
- **Eligible catalog comes from metadata only**: titles with a known release date before the end of the evaluation window. Using titles that received interactions in the window would leak the answer.
- **"All users" requires some earlier history**, because no model can personalize without it. Users with no history are reported separately.
- **Cold-title results appear in two forms**: within the full-catalog ranking, and ranked only among cold titles. Full-catalog is the honest product measure; the restricted form isolates ordering quality.
- **The test window was evaluated only in the final pipeline run.** All development used validation. `scripts/train.py` removes test-window rows immediately after loading, and `scripts/evaluate.py` refuses to run against a different frozen configuration once the test set has been scored.

## Models

- **EASE was chosen over Item-KNN** because it is closed-form and has one main hyperparameter.
- **ALS uses `implicit`.** Fold-in is reimplemented in NumPy so the demo does not need that package, and a test checks it against `implicit`.
- **Popularity baselines count labelled positives** using the fixed label definition, regardless of the target the collaborative models train on.
- **Hybrid selection uses a constraint.** Cold titles are a small share of positives, so overall NDCG alone would not reward handling them. Among configurations within 1% of the best validation NDCG@10, the one ranking cold titles best is selected.
- **No two-tower model or LightGBM reranker.** The brief makes these conditional on a clear validation need. The hybrid led every other model on validation, narrowly ahead of EASE, so none was added. On the test set it was level with EASE, as reported in `reports/results.md`.

## Demo

- **Streamlit**, loading a small precomputed bundle. EASE is truncated to the strongest neighbors per title, and the effect of truncation on validation NDCG is reported.
- **Official covers are shown, at the owner's explicit request.** The original brief allowed images only if permitted. MyAnimeList covers are copyrighted and neither dataset license covers them, so this needs review before any public deployment. Covers are hotlinked and no image files are in the repository. Wide hero banners are hotlinked from AniList's public API on the same basis, added by `scripts/add_banners.py`. `ANIMEREC_SHOW_COVERS=0` switches to generated art.
- **Community scores appear on cards for display only.** They are loaded while exporting the demo bundle, stay out of the catalog, and never reach a model.
- **Same-franchise titles are hidden by default**, with a checkbox to allow sequels.
- **Disliked titles** get negative weight in fold-in and are excluded from results.
- **Cards open an info popup first**, built from dataset fields added by `scripts/add_details.py`. Details are display-only, and the link to MyAnimeList sits inside the popup.
- **Onboarding answers persist in the visitor's browser** through local storage and the page URL, with no server-side storage, so a refresh or return visit skips onboarding. **Retake preferences** clears them.
- **The demo model is the evaluated model**: fit on activity before 2018-02-01. It is not refit on later data, so what it serves matches what was tested.
