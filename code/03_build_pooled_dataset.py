"""Stage 3: per-day sentiment statistics, the 15 features, and the three-way
split, written as one row per usable day.

Per post, polarity = p_pos - p_neg. Per day, over that day's posts on one
platform, five statistics are computed:
    mean, mean_top3, pct_pos, p90, p10
mean_top3 restricts to the three highest-volume groups, which share names on
both platforms: Company News, Safety Matters, Wins & Shout-outs. pct_neg is
not computed because pct_pos + pct_neg is identically 1.

15 features = the 5 statistics x 3 views (level, shock7, shock14):
    shock7_X  = X[today] - mean(X over the 7 preceding calendar days)
    shock14_X = X[today] - mean(X over the 14 preceding calendar days)

The earliest calendar day of raw data on each platform is a partial
collection-window edge day and is dropped from the series before anything
else (derived from the data, not hardcoded), so it never enters a shock
lookback. A day is usable only with a full, contiguous 14-day lookback; there
is no partial-baseline fallback.

Split:
    TRAIN          Workplace  2025-01-15 -> 2026-05-31   (502 days)
                   Yammer     2026-06-15 -> 2026-07-10    (26 days)
    internal_TEST  Yammer     2026-07-11 -> 2026-08-09    (30 days, labelled)
    HOLDOUT        Yammer     2026-08-10 -> 2026-09-06    (28 days, no label;
                   the dates in labels/holdout_dates.csv)

Input : processed/workplace_posts.json, processed/yammer_messages.json
        processed/post_sentiment_scores.parquet
        labels/reputation_daily.csv, labels/holdout_dates.csv
Output: processed/pooled_dataset.json
        {feature_names, rows: [{date, source, split, reputation_score, <15 features>}]}
        reputation_score is null for HOLDOUT rows.

Usage:
    python code/03_build_pooled_dataset.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "processed"
WP_POSTS = OUT_DIR / "workplace_posts.json"
YM_MESSAGES = OUT_DIR / "yammer_messages.json"
SENTIMENT_PARQUET = OUT_DIR / "post_sentiment_scores.parquet"
LABELS_CSV = REPO_ROOT / "labels" / "reputation_daily.csv"
HOLDOUT_CSV = REPO_ROOT / "labels" / "holdout_dates.csv"
OUT_DATASET = OUT_DIR / "pooled_dataset.json"

TOP3_GROUPS = {"Company News", "Safety Matters", "Wins & Shout-outs"}

WP_TRAIN_START, WP_TRAIN_END = "2025-01-15", "2026-05-31"
YM_TRAIN_START, YM_TRAIN_END = "2026-06-15", "2026-07-10"
TEST_START, TEST_END = "2026-07-11", "2026-08-09"

# 15 features = 5 statistics x 3 views
_STATS = ("mean", "mean_top3", "pct_pos", "p90", "p10")
FEATURE_NAMES = (
    [f"level_{x}" for x in _STATS]
    + [f"shock7_{x}" for x in _STATS]
    + [f"shock14_{x}" for x in _STATS]
)
assert len(FEATURE_NAMES) == 15


def load_labels() -> dict[str, float]:
    with LABELS_CSV.open(encoding="utf-8") as fh:
        return {r["date"]: float(r["reputation_score"]) for r in csv.DictReader(fh)}


def load_holdout() -> set[str]:
    with HOLDOUT_CSV.open(encoding="utf-8") as fh:
        return {r["date"] for r in csv.DictReader(fh) if r["date"]}


def day_stats(rows: list[tuple[tuple[float, float], str | None]]) -> dict:
    pol = np.array([p_pos - p_neg for (p_neg, p_pos), _ in rows])
    top3_pol = np.array([p_pos - p_neg for (p_neg, p_pos), g in rows if g in TOP3_GROUPS])
    return {
        "mean": float(pol.mean()), "p10": float(np.percentile(pol, 10)),
        "p90": float(np.percentile(pol, 90)), "pct_pos": float((pol > 0).mean()),
        "mean_top3": float(top3_pol.mean()) if len(top3_pol) else None,
    }


def build_stats_series(day_rows: dict[str, list], scores: dict[str, tuple]) -> dict[str, dict]:
    out = {}
    for day, texts_groups in day_rows.items():
        rows = [(scores[t], g) for t, g in texts_groups if t in scores]
        if rows:
            out[day] = day_stats(rows)
    return out


def shock(stats_series: dict[str, dict], day: str, stat: str, window: int) -> float | None:
    if day not in stats_series or stats_series[day].get(stat) is None:
        return None
    d = date.fromisoformat(day)
    vals = []
    for k in range(1, window + 1):
        prior = (d - timedelta(days=k)).isoformat()
        if prior in stats_series and stats_series[prior].get(stat) is not None:
            vals.append(stats_series[prior][stat])
        else:
            break  # contiguous lookback only
    if len(vals) < window:  # full baseline required
        return None
    return float(stats_series[day][stat] - np.mean(vals))


def build_features_for_day(stats_series: dict[str, dict], day: str) -> dict | None:
    if day not in stats_series:
        return None
    s = stats_series[day]
    feat = {f"level_{x}": s[x] for x in _STATS}
    for x in _STATS:
        feat[f"shock7_{x}"] = shock(stats_series, day, x, 7)
    for x in _STATS:
        feat[f"shock14_{x}"] = shock(stats_series, day, x, 14)
    if any(feat[n] is None for n in FEATURE_NAMES):
        return None
    return feat


def main() -> int:
    wp_posts = json.load(WP_POSTS.open(encoding="utf-8"))
    ym_messages = json.load(YM_MESSAGES.open(encoding="utf-8"))
    scores_df = pd.read_parquet(SENTIMENT_PARQUET)
    scores = {row.text: (row.p_neg, row.p_pos) for row in scores_df.itertuples(index=False)}
    labels = load_labels()
    holdout = load_holdout()

    wp_day_rows: dict[str, list] = defaultdict(list)
    for p in wp_posts:
        wp_day_rows[p["created_time"][:10]].append((p["message_clean"], p["group_name"]))
    ym_day_rows: dict[str, list] = defaultdict(list)
    for m in ym_messages:
        ym_day_rows[m["created_at"][:10]].append((m["body"], m["group_name"]))

    wp_edge = min(wp_day_rows)
    ym_edge = min(ym_day_rows)
    print(f"dropping collection-window edge days: workplace={wp_edge} ({len(wp_day_rows[wp_edge])} posts), "
          f"yammer={ym_edge} ({len(ym_day_rows[ym_edge])} messages)")
    del wp_day_rows[wp_edge]
    del ym_day_rows[ym_edge]

    wp_stats = build_stats_series(wp_day_rows, scores)
    ym_stats = build_stats_series(ym_day_rows, scores)

    rows_out = []
    for day in sorted(wp_stats):
        if not (WP_TRAIN_START <= day <= WP_TRAIN_END) or day not in labels:
            continue
        feat = build_features_for_day(wp_stats, day)
        if feat is None:
            continue
        rows_out.append({"date": day, "source": "workplace", "split": "TRAIN",
                         "reputation_score": labels[day], **feat})

    n_ym_dropped_no_history = 0
    for day in sorted(ym_stats):
        if YM_TRAIN_START <= day <= YM_TRAIN_END:
            split, has_label = "TRAIN", True
        elif TEST_START <= day <= TEST_END:
            split, has_label = "internal_TEST", True
        elif day in holdout:
            split, has_label = "HOLDOUT", False
        else:
            continue
        if has_label and day not in labels:
            continue
        feat = build_features_for_day(ym_stats, day)
        if feat is None:
            n_ym_dropped_no_history += 1
            continue
        rows_out.append({"date": day, "source": "yammer", "split": split,
                         "reputation_score": labels[day] if has_label else None, **feat})

    rows_out.sort(key=lambda r: (r["date"], r["source"]))

    counts = defaultdict(int)
    for r in rows_out:
        counts[(r["split"], r["source"])] += 1
    print(f"TRAIN: workplace={counts[('TRAIN','workplace')]} yammer={counts[('TRAIN','yammer')]} "
          f"pooled={counts[('TRAIN','workplace')] + counts[('TRAIN','yammer')]}")
    print(f"internal_TEST: {counts[('internal_TEST','yammer')]}")
    print(f"HOLDOUT: {counts[('HOLDOUT','yammer')]}")
    if n_ym_dropped_no_history:
        print(f"NOTE: {n_ym_dropped_no_history} Yammer candidate day(s) dropped for lacking full 14-day history")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_DATASET.open("w", encoding="utf-8") as fh:
        json.dump({"feature_names": FEATURE_NAMES, "rows": rows_out}, fh, indent=2)
    print(f"\nwrote {len(rows_out)} rows -> {OUT_DATASET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
