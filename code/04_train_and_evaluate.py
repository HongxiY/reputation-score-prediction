"""Stage 4: fit the pooled Ridge twice from the one dataset.

    A. INTERNAL TEST  fit on TRAIN only, rolling-origin evaluate within TRAIN
       (expanding window, first prediction at row START), then score once on
       the frozen internal_TEST days. This is the held-out validation number.

    B. HOLDOUT        refit on TRAIN + internal_TEST (every labelled day) and
       predict the unlabelled HOLDOUT days. This is the submission output. It
       is a different fitted model from (A), trained on 30 more days.

Both fits: features standardised on the fit's own training statistics, no
platform indicator column, alpha chosen by walk-forward CV (TimeSeriesSplit,
argmin MAE over a log grid capped at n/10), selected once per fit and then
held fixed, including through the rolling-origin loop. The feature list is
read from pooled_dataset.json, not hardcoded.

Input : processed_final/pooled_dataset.json
        data/submission_template.csv           (read-only; the 28 holdout dates)
Output: processed_final/model_results.json           metrics, coefficients, predictions
        processed_final/model_report.txt
        processed_final/pooled_ridge_internal_test.joblib
        processed_final/pooled_ridge_final_holdout.joblib
        processed_final/holdout_predictions.json
        processed_final/submission_template.csv       the template plus a filled
                                                        reputation_score column

Usage:
    python code_final/04_train_and_evaluate.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "processed_final"
DATASET_JSON = OUT_DIR / "pooled_dataset.json"
SUBMISSION_TEMPLATE_SRC = REPO_ROOT / "data" / "submission_template.csv"

OUT_JSON = OUT_DIR / "model_results.json"
OUT_TXT = OUT_DIR / "model_report.txt"
OUT_MODEL_INTERNAL = OUT_DIR / "pooled_ridge_internal_test.joblib"
OUT_MODEL_HOLDOUT = OUT_DIR / "pooled_ridge_final_holdout.joblib"
OUT_HOLDOUT_PRED = OUT_DIR / "holdout_predictions.json"
OUT_SUBMISSION = OUT_DIR / "submission_template.csv"

ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
START = 200


def alpha_grid_for(n: int) -> list[float]:
    cap = max(n / 10.0, 0.1)
    return [a for a in ALPHA_GRID if a <= cap] or [cap]


def pick_alpha_argmin(X: np.ndarray, y: np.ndarray, n_splits: int) -> float:
    grid = alpha_grid_for(len(y))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    means = {a: float(np.mean([mean_absolute_error(y[va], Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[va]))
                               for tr, va in tscv.split(X)])) for a in grid}
    return min(means, key=means.get)


def standardize(X: np.ndarray):
    mean, std = X.mean(axis=0), X.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    return (X - mean) / std_safe, mean, std_safe


def metrics_block(y_true: np.ndarray, y_pred: np.ndarray, const_pred: np.ndarray) -> dict:
    err = np.abs(y_true - y_pred)
    ss_res, ss_const = ((y_true - y_pred) ** 2).sum(), ((y_true - const_pred) ** 2).sum()
    return {"n": int(len(y_true)), "mae": float(err.mean()),
            "mae_se": float(err.std(ddof=1) / np.sqrt(len(err))),
            "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
            "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0,
            "r2_vs_constant": float(1 - ss_res / ss_const) if ss_const > 0 else float("nan"),
            "pred_sd": float(np.std(y_pred)), "actual_sd": float(np.std(y_true))}


def fit_ridge(rows: list[dict], y: np.ndarray, feature_names: list[str], n_splits: int):
    X = np.array([[r[n] for n in feature_names] for r in rows], dtype=float)
    Xs, mean, std = standardize(X)
    alpha = pick_alpha_argmin(Xs, y, n_splits)
    model = Ridge(alpha=alpha).fit(Xs, y)
    return model, mean, std, alpha


def predict(model, mean, std, rows: list[dict], feature_names: list[str]) -> np.ndarray:
    X = np.array([[r[n] for n in feature_names] for r in rows], dtype=float)
    return model.predict((X - mean) / std)


def main() -> int:
    blob = json.load(DATASET_JSON.open(encoding="utf-8"))
    feature_names = blob["feature_names"]
    rows = blob["rows"]
    train_rows = [r for r in rows if r["split"] == "TRAIN"]
    test_rows = [r for r in rows if r["split"] == "internal_TEST"]
    holdout_rows = [r for r in rows if r["split"] == "HOLDOUT"]
    print(f"TRAIN: {len(train_rows)}  internal_TEST: {len(test_rows)}  HOLDOUT: {len(holdout_rows)}  "
          f"features: {len(feature_names)}")

    y_train = np.array([r["reputation_score"] for r in train_rows])
    y_test = np.array([r["reputation_score"] for r in test_rows])

    # =================== A. INTERNAL TEST =================== #
    Xs_full, _, _ = standardize(np.array([[r[n] for n in feature_names] for r in train_rows], dtype=float))
    alpha_a = pick_alpha_argmin(Xs_full, y_train, n_splits=6)
    print(f"\n[A: internal test] alpha (selected once, held fixed): {alpha_a}")

    preds, actuals = [], []
    for i in range(START, len(train_rows)):
        Xs, mean, std = standardize(np.array([[r[n] for n in feature_names] for r in train_rows[:i]], dtype=float))
        model = Ridge(alpha=alpha_a).fit(Xs, y_train[:i])
        x_next = predict(model, mean, std, [train_rows[i]], feature_names)
        preds.append(float(x_next[0]))
        actuals.append(float(y_train[i]))
    preds, actuals = np.array(preds), np.array(actuals)
    const_pred_ro = np.full(len(actuals), y_train[:START].mean())
    ro_metrics = metrics_block(actuals, preds, const_pred_ro)
    print(f"  rolling origin, n={ro_metrics['n']}: MAE={ro_metrics['mae']:.3f}+-{ro_metrics['mae_se']:.3f}  "
          f"RMSE={ro_metrics['rmse']:.3f}  r={ro_metrics['pearson_r']:+.3f}  "
          f"R2vC={ro_metrics['r2_vs_constant']:+.3f}  pred_sd={ro_metrics['pred_sd']:.2f} "
          f"(actual_sd={ro_metrics['actual_sd']:.2f})")

    model_a, mean_a, std_a, alpha_a2 = fit_ridge(train_rows, y_train, feature_names, n_splits=6)
    assert alpha_a2 == alpha_a  # same recipe, same data -> must reproduce
    pred_test = predict(model_a, mean_a, std_a, test_rows, feature_names)
    const_pred_test = np.full(len(y_test), y_train.mean())
    test_metrics = metrics_block(y_test, pred_test, const_pred_test)
    print(f"  frozen internal_TEST, n={test_metrics['n']}: MAE={test_metrics['mae']:.3f}+-"
          f"{test_metrics['mae_se']:.3f}  RMSE={test_metrics['rmse']:.3f}  r={test_metrics['pearson_r']:+.3f}  "
          f"R2vC={test_metrics['r2_vs_constant']:+.3f}  pred_sd={test_metrics['pred_sd']:.2f} "
          f"(actual_sd={test_metrics['actual_sd']:.2f})")

    coefs_a = sorted(zip(feature_names, model_a.coef_.tolist()), key=lambda kv: -abs(kv[1]))
    print("\n  standardised coefficients (internal-test model), sorted by |value|:")
    for name, c in coefs_a:
        print(f"    {name:18s} {c:+.4f}")

    joblib.dump({"model": model_a, "mean": mean_a, "std": std_a, "alpha": alpha_a,
                 "feature_names": feature_names}, OUT_MODEL_INTERNAL)

    # =================== B. HOLDOUT / SUBMISSION =================== #
    all_labelled_rows = train_rows + test_rows
    y_all = np.array([r["reputation_score"] for r in all_labelled_rows])
    print(f"\n[B: holdout] refitting on all {len(all_labelled_rows)} labelled days")

    model_b, mean_b, std_b, alpha_b = fit_ridge(all_labelled_rows, y_all, feature_names, n_splits=6)
    print(f"  alpha: {alpha_b}")
    coefs_b = sorted(zip(feature_names, model_b.coef_.tolist()), key=lambda kv: -abs(kv[1]))
    print("  standardised coefficients (final holdout model), sorted by |value|:")
    for name, c in coefs_b:
        print(f"    {name:18s} {c:+.4f}")

    pred_holdout = predict(model_b, mean_b, std_b, holdout_rows, feature_names)
    holdout_dates = [r["date"] for r in holdout_rows]
    print(f"\n  holdout predictions ({len(holdout_dates)} days):")
    for d, p in zip(holdout_dates, pred_holdout):
        print(f"    {d}  {p:.3f}")
    print(f"  pred mean={pred_holdout.mean():.3f}  pred sd={pred_holdout.std():.3f}  "
          f"(TRAIN+internal_TEST actual mean={y_all.mean():.3f}, sd={y_all.std():.3f})")

    joblib.dump({"model": model_b, "mean": mean_b, "std": std_b, "alpha": alpha_b,
                 "feature_names": feature_names}, OUT_MODEL_HOLDOUT)

    with OUT_HOLDOUT_PRED.open("w", encoding="utf-8") as fh:
        json.dump({"dates": holdout_dates, "reputation_score": pred_holdout.tolist(),
                   "alpha": alpha_b, "n_train": len(all_labelled_rows)}, fh, indent=2)

    # Fill the read-only template's dates with model B's predictions and save
    # the filled copy under processed_final/. The template itself is not written.
    with SUBMISSION_TEMPLATE_SRC.open(encoding="utf-8") as fh:
        template_dates = [r["date"] for r in csv.DictReader(fh)]
    pred_by_date = dict(zip(holdout_dates, pred_holdout.tolist()))
    assert template_dates == holdout_dates, "submission template dates don't match the built HOLDOUT rows"
    with OUT_SUBMISSION.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "reputation_score"])
        for d in template_dates:
            w.writerow([d, f"{pred_by_date[d]:.4f}"])

    # =================== save everything =================== #
    results = {
        "internal_test": {
            "alpha": alpha_a,
            "rolling_origin": {**ro_metrics, "predictions": preds.tolist(), "actuals": actuals.tolist()},
            "test": {**test_metrics, "predictions": pred_test.tolist()},
            "coefficients": [{"feature": n, "standardized_coef": c} for n, c in coefs_a],
        },
        "holdout": {
            "alpha": alpha_b, "n_train": len(all_labelled_rows),
            "coefficients": [{"feature": n, "standardized_coef": c} for n, c in coefs_b],
            "dates": holdout_dates, "predictions": pred_holdout.tolist(),
        },
    }
    with OUT_JSON.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    lines = ["=" * 80, f"A. INTERNAL TEST -- {len(feature_names)}-feature pooled Ridge", "=" * 80, "",
             f"TRAIN n={len(train_rows)}  alpha={alpha_a}",
             f"rolling origin, n={ro_metrics['n']}: MAE={ro_metrics['mae']:.3f} +/- {ro_metrics['mae_se']:.3f}  "
             f"RMSE={ro_metrics['rmse']:.3f}  r={ro_metrics['pearson_r']:.3f}  "
             f"R2vC={ro_metrics['r2_vs_constant']:.3f}  pred_sd={ro_metrics['pred_sd']:.3f} "
             f"(actual_sd={ro_metrics['actual_sd']:.3f})",
             f"frozen internal_TEST, n={test_metrics['n']}: MAE={test_metrics['mae']:.3f} +/- "
             f"{test_metrics['mae_se']:.3f}  RMSE={test_metrics['rmse']:.3f}  r={test_metrics['pearson_r']:.3f}  "
             f"R2vC={test_metrics['r2_vs_constant']:.3f}  pred_sd={test_metrics['pred_sd']:.3f} "
             f"(actual_sd={test_metrics['actual_sd']:.3f})", "",
             "standardised coefficients:"]
    for n, c in coefs_a:
        lines.append(f"  {n:18s} {c:+.4f}")
    lines += ["", "=" * 80, "B. HOLDOUT / SUBMISSION", "=" * 80, "",
              f"trained on TRAIN + internal_TEST, n={len(all_labelled_rows)}  alpha={alpha_b}",
              f"holdout predictions: mean={pred_holdout.mean():.3f}  sd={pred_holdout.std():.3f}", "",
              "standardised coefficients:"]
    for n, c in coefs_b:
        lines.append(f"  {n:18s} {c:+.4f}")
    lines.append("")
    lines.append("date        reputation_score")
    for d, p in zip(holdout_dates, pred_holdout):
        lines.append(f"{d}  {p:.4f}")
    lines += ["", "=" * 80]
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nwrote {OUT_MODEL_INTERNAL}")
    print(f"wrote {OUT_MODEL_HOLDOUT}")
    print(f"wrote {OUT_HOLDOUT_PRED}")
    print(f"wrote {OUT_SUBMISSION}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
