# 15-feature pooled Ridge

Predicts the daily `reputation_score` for the 28 Yammer holdout days
(2026-08-10 to 2026-09-06) from the sentiment of internal posts. The model is
a Ridge regression on 15 standardised daily sentiment features, fit on pooled
Workplace + Yammer days.

Every script in this folder reads only from `data/` and `labels/` (read-only)
plus whatever an earlier stage wrote to `processed_final/`.

## Environment

`code_final/environment.yml` pins the exact versions used.

```
conda env create -f code_final/environment.yml
```

Only stage 2 needs the GPU stack (torch >= 2.6, transformers). Once
`processed_final/post_sentiment_scores.parquet` exists, stages 1, 3 and 4 run
on CPU with pandas, numpy, scikit-learn, pyarrow and joblib.

## Run order

```
python code_final/01_build_post_tables.py     data/meta-workplace/*, data/yammer/*.csv
                                               -> processed_final/workplace_posts.json
                                                  processed_final/yammer_messages.json

python code_final/02_score_sentiment.py       the two post tables above
                                               -> processed_final/post_sentiment_scores.parquet
                                               (GPU, cuda:0; torch >= 2.6)

python code_final/03_build_pooled_dataset.py  post tables + sentiment scores + labels/*.csv
                                               -> processed_final/pooled_dataset.json

python code_final/04_train_and_evaluate.py    processed_final/pooled_dataset.json
                                               + data/submission_template.csv (read-only)
                                               -> processed_final/model_results.json
                                                  processed_final/model_report.txt
                                                  processed_final/pooled_ridge_internal_test.joblib
                                                  processed_final/pooled_ridge_final_holdout.joblib
                                                  processed_final/holdout_predictions.json
                                                  processed_final/submission_template.csv
```

## Data preparation (stages 1 and 2)

**Stage 1** builds one clean table per platform from the raw data. Workplace
posts are deduplicated by post id (the paginated feed arrays serve some posts
twice), a leading title that merely restates the next paragraph is stripped
from the message, and `group_name` is resolved through `groups.json`. Yammer
messages are taken as they are from `microsoft-graph-yammer-messages.csv`. A
post is dated by `created_time` / `created_at` only, never by `updated_time`
or `collected_at`.

| table | rows | date range |
|---|--:|---|
| `workplace_posts.json` | 23,350 | 2024-12-31 to 2026-05-31 |
| `yammer_messages.json` | 6,832 | 2026-05-31 to 2026-09-06 |

**Stage 2** scores every distinct text once with
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest),
a RoBERTa-base model pre-trained on ~124M English tweets and fine-tuned for
three-class sentiment on TweetEval (Loureiro et al., 2022, *TimeLMs*). It is
used off the shelf; nothing is fine-tuned on this project's data. For each
text it returns a softmax over three classes, `p_neg`, `p_neu` and `p_pos`,
which sum to 1. Texts are deduplicated by exact string before scoring (the
same short reply often appears on both platforms) and truncated at 512
tokens, which never binds here (the longest text is 93 words). Output is
`post_sentiment_scores.parquet` with 22,956 rows and columns
`text, p_neg, p_neu, p_pos`.

At the post level the two platforms look like this.

| | posts | argmax class neg / neu / pos | mean polarity | share polarity > 0 |
|---|--:|:--:|--:|--:|
| Workplace | 23,350 | 25% / 27% / 48% | +0.28 | 73% |
| Yammer | 6,832 | 18% / 20% / 61% | +0.45 | 79% |

Yammer reads noticeably more positive than Workplace on every summary. That
platform offset is one reason two thirds of the features are *changes*
rather than levels.

## Features (stage 3)

### One number per post

```
polarity = p_pos - p_neg          in [-1, +1]
```

A confidently positive post is near +1, a confidently negative one near -1,
a neutral or mixed one near 0. `p_neu` is not used directly; since
`p_neu = 1 - p_pos - p_neg`, a large neutral mass simply pulls polarity
toward 0. Nothing else about a post enters the features. No author, no
reactions, no length, no time of day, and no group metadata beyond the one
group filter used by `mean_top3`.

### Daily statistics

All posts dated the same calendar day, one platform at a time, form that
day's sample of polarities, summarised by five statistics.

| statistic | definition | what it captures |
|---|---|---|
| `mean` | average polarity over the day's posts | the day's overall mood |
| `mean_top3` | average polarity over posts in the three fixed high-volume all-company groups (Company News, Safety Matters, Wins & Shout-outs) | the mood on the main stage, ignoring the long tail of small groups and personal posts |
| `pct_pos` | share of posts with polarity > 0 (i.e. `p_pos > p_neg`) | how *widespread* positivity is, regardless of strength |
| `p90` | 90th percentile of polarity | the day's positive edge, a few loud posts the mean would wash out |
| `p10` | 10th percentile of polarity | the day's negative edge, likewise |

`pct_neg` (share with polarity < 0) is not computed. No scored text has
`p_pos == p_neg` exactly (0 of 22,956), so `pct_neg = 1 - pct_pos` holds on
every day and the column would be perfectly redundant.

### Three views

| view | definition | reads as |
|---|---|---|
| `level_*` | today's value | how positive is sentiment today |
| `shock7_*` | today's value minus its mean over the 7 preceding calendar days | how different is today from the past week |
| `shock14_*` | today's value minus its mean over the 14 preceding calendar days | how different is today from the past two weeks |

Why shocks. A reputation score is expected to react to something
*happening*, a departure from normal internal sentiment, more than to the
level of that sentiment, which is positive on most days. The level also
carries a platform-specific offset (see the table above), whereas a deviation
from a platform's own recent baseline is on a comparable scale on both
platforms. That matters because the model is fit mostly on Workplace and
applied to Yammer. Two windows let the model weigh a sharp one-week move
against a slower two-week drift. The day itself is never part of its own
baseline, and both lookbacks require a full, contiguous window of days with
posts, which is what fixes the TRAIN start dates below.

### The 15 features

Five statistics x three views, all kept.

```
level_mean,   level_mean_top3,   level_pct_pos,   level_p90,   level_p10
shock7_mean,  shock7_mean_top3,  shock7_pct_pos,  shock7_p90,  shock7_p10
shock14_mean, shock14_mean_top3, shock14_pct_pos, shock14_p90, shock14_p10
```

No feature selection beyond the exact redundancy above is applied. The
features are shrunk by Ridge rather than screened, because univariate
correlation filters are not reliable at the size of the Yammer TRAIN sample
(26 days).

## The three-way split (stage 3)

The single earliest calendar day of raw data on each platform (Workplace
2024-12-31, Yammer 2026-05-31) is a partial collection-window edge day. It is
identified from the data and dropped before anything else, so it never
enters a shock lookback. A day is usable only with a full, contiguous 14-day
lookback; there is no partial-baseline fallback. This is what sets the TRAIN
start dates.

| split | source | date range | days | label |
|---|---|---|--:|:--:|
| `TRAIN` | Workplace | 2025-01-15 to 2026-05-31 | 502 | yes |
| `TRAIN` | Yammer | 2026-06-15 to 2026-07-10 | 26 | yes |
| `internal_TEST` | Yammer | 2026-07-11 to 2026-08-09 | 30 | yes |
| `HOLDOUT` | Yammer | 2026-08-10 to 2026-09-06 | 28 | **no** |

`HOLDOUT` is `labels/holdout_dates.csv`. Yammer message data exists through
2026-09-06, so the features are fully computable; only `reputation_score` is
missing, which is the prediction target.

## Model (stage 4)

### Why Ridge

The 15 features are strongly collinear by construction. On TRAIN, 27 of the
105 feature pairs have |r| > 0.7 (`level_mean` vs `level_pct_pos` 0.95,
`shock7_p10` vs `shock14_p10` 0.94, `shock14_mean` vs `shock14_pct_pos`
0.94), and there are only about 530 training days. Plain least squares would
give large, unstable coefficients; Ridge's L2 penalty shrinks them and
spreads weight across correlated features instead of picking one
arbitrarily. Each feature is z-scored with the fitting set's own mean and SD,
so the standardised coefficients in `model_report.txt` are the change in
predicted score per one-SD move in that feature. Because of the collinearity
they should be read as a group rather than individually.

There is no platform indicator column and no lagged reputation score. The
penalty strength `alpha` is chosen by walk-forward cross-validation
(`TimeSeriesSplit(6)`, MAE, argmin over a log grid capped at n/10), once per
fit, and then held fixed. Both fits in this run selected `alpha = 30`.

### Two fits from one dataset

**A. Internal test.** Fit on `TRAIN` only (528 pooled days). Rolling-origin
evaluated within `TRAIN` (expanding window from row 200, giving 328
one-step predictions, with `alpha` fixed at the value selected on the full
TRAIN set), then scored once on the frozen `internal_TEST` (30 days), which
is the honest held-out number. Saved as `pooled_ridge_internal_test.joblib`.

**B. Holdout / submission.** Refit on `TRAIN + internal_TEST` (558 days,
every labelled day available), predicting the 28 `HOLDOUT` days. No metrics
are possible here (no ground truth); this is the submission output. Saved as
`pooled_ridge_final_holdout.joblib`, a different fitted model from A since it
saw 30 more days.

### Results

| | MAE | RMSE | r | R² vs constant |
|---|--:|--:|--:|--:|
| A, rolling origin, 328 predictions | 2.764 ± 0.143 | 3.783 | 0.515 | 0.263 |
| A, frozen `internal_TEST`, 30 days | 3.024 ± 0.468 | 3.937 | 0.701 | 0.395 |

The constant baseline for "R² vs constant" is the training mean. The ± is
the standard error of the MAE.

Holdout (model B, 558 labelled days, `alpha = 30`): 28 predictions, mean
66.243, sd 3.840. The labelled days used to fit it have mean 65.619 and sd
4.476. Full per-day predictions and both sets of standardised coefficients
are in `model_report.txt`; the three largest coefficients in model B are
`shock7_mean` (+1.70), `level_p90` (+1.68) and `shock14_mean` (+0.82).

## Submission file

`data/submission_template.csv` has 28 rows and a `date` column only,
matching `labels/holdout_dates.csv`. Stage 4 reads it, never writes to it,
and saves a filled copy with a `reputation_score` column (model B's
predictions, 4 decimals) to `processed_final/submission_template.csv`. An
assertion checks that the template's dates exactly match the dataset's
`HOLDOUT` rows before writing, so a mismatch fails loudly instead of silently
misaligning dates and predictions.

## Outputs

| file | contents |
|---|---|
| `processed_final/workplace_posts.json` | cleaned Workplace posts |
| `processed_final/yammer_messages.json` | Yammer messages |
| `processed_final/post_sentiment_scores.parquet` | one row per distinct text, `p_neg, p_neu, p_pos` |
| `processed_final/pooled_dataset.json` | `feature_names` plus one row per usable day (`date, source, split, reputation_score, 15 features`) |
| `processed_final/model_results.json` | metrics, coefficients and predictions for both fits |
| `processed_final/model_report.txt` | the same, human-readable |
| `processed_final/pooled_ridge_internal_test.joblib` | model A with its standardisation stats, `alpha` and feature list |
| `processed_final/pooled_ridge_final_holdout.joblib` | model B, likewise |
| `processed_final/holdout_predictions.json` | model B's 28 predictions with dates |
| `processed_final/submission_template.csv` | the filled submission file |

## Notes

- **Workplace dedup and title strip matter.** The feed arrays contain
  pagination-boundary duplicate posts, and many posts open with a title that
  restates the next sentence. Both are handled in `01_build_post_tables.py`.
- **Only `created_time` / `created_at` dates a post.**
- **Edge days are derived, not hardcoded.** The earliest raw day on each
  platform is dropped before any statistic is computed.
- **Alpha is selected once per fit and held fixed.** Re-selecting it at every
  step of the rolling-origin loop would be a different method from the one
  whose frozen-test number is reported.
- **Standardisation uses the fitting set's own statistics only.** Model A
  and model B therefore have different means and SDs stored alongside them.
