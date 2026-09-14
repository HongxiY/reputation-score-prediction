"""Stage 2: score every distinct post/message text for sentiment with
cardiffnlp/twitter-roberta-base-sentiment-latest -> p_neg, p_neu, p_pos
(raw output order [negative, neutral, positive], matching model.config.id2label).

Texts are deduplicated by exact string and pooled across both platforms.
Requires torch >= 2.6 (the checkpoint is a pre-safetensors .bin) and a single
visible GPU, pinned to cuda:0 before torch is imported.

Input : processed_final/workplace_posts.json, processed_final/yammer_messages.json
Output: processed_final/post_sentiment_scores.parquet   [text, p_neg, p_neu, p_pos]

Usage:
    python code_final/02_score_sentiment.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "processed_final"
WP_POSTS = OUT_DIR / "workplace_posts.json"
YM_MESSAGES = OUT_DIR / "yammer_messages.json"
OUT_PARQUET = OUT_DIR / "post_sentiment_scores.parquet"

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 128
MAX_LENGTH = 512
RAW_LABEL_ORDER = ["negative", "neutral", "positive"]


def collect_distinct_texts() -> list[str]:
    wp = json.load(WP_POSTS.open(encoding="utf-8"))
    ym = json.load(YM_MESSAGES.open(encoding="utf-8"))
    wp_texts = {p["message_clean"] for p in wp if p.get("message_clean")}
    ym_texts = {m["body"] for m in ym if m.get("body")}
    combined = sorted(wp_texts | ym_texts)
    print(f"Workplace distinct texts: {len(wp_texts)}  Yammer distinct texts: {len(ym_texts)}  "
          f"pooled (overlap removed): {len(combined)}")
    return combined


def score(texts: list[str]) -> pd.DataFrame:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    assert torch.cuda.device_count() == 1, "expected exactly one visible GPU (cuda:0 only)"
    tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).to("cuda:0").eval()
    assert model.config.num_labels == len(RAW_LABEL_ORDER)

    out = np.zeros((len(texts), len(RAW_LABEL_ORDER)), dtype="float32")
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i:i + BATCH_SIZE]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_LENGTH).to("cuda:0")
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out[i:i + len(chunk)] = probs
    return pd.DataFrame({"text": texts, "p_neg": out[:, 0], "p_neu": out[:, 1], "p_pos": out[:, 2]})


def main() -> int:
    texts = collect_distinct_texts()
    print(f"scoring {len(texts):,} texts with {SENTIMENT_MODEL} ...")
    df = score(texts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {len(df):,} rows -> {OUT_PARQUET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
