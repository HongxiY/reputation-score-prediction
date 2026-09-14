"""Stage 1: build two clean per-post tables from the raw read-only data
(data/meta-workplace/*, data/yammer/*.csv).

Workplace
    1. Dedupe by post id. The paginated group and personal feed arrays serve
       some posts twice; the copies are identical on every field kept here,
       so the first occurrence wins.
    2. Strip a leading title (markdown heading or bold sentence) from
       `message` when it merely restates the paragraph that follows, giving
       `message_clean`.
    3. Resolve `group_name` from the outer group-data dict key via
       groups.json. Personal posts (members_feed.json) have no group.

Yammer
    microsoft-graph-yammer-messages.csv already has one row per message with
    `body` and `group_name` columns; rows without body text are skipped.

A post is dated by `created_time` (Workplace) / `created_at` (Yammer) only.

Output: processed_final/workplace_posts.json   [{id, created_time, message_clean, group_name}]
        processed_final/yammer_messages.json   [{id, created_at, body, group_name}]

Usage:
    python code_final/01_build_post_tables.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MW_DIR = REPO_ROOT / "data" / "meta-workplace"
YM_MESSAGES_CSV = REPO_ROOT / "data" / "yammer" / "microsoft-graph-yammer-messages.csv"
OUT_DIR = REPO_ROOT / "processed_final"
OUT_WP = OUT_DIR / "workplace_posts.json"
OUT_YM = OUT_DIR / "yammer_messages.json"

# Title lead-in patterns: a markdown heading or a bold sentence. Searched
# anywhere in the message, since a greeting paragraph may precede the title.
HEADING_RE = re.compile(r"(?P<lead>\A|\n\n)#\s+(?P<title>[^\n]+)\n\n")
BOLD_RE = re.compile(r"(?P<lead>\A|\n\n)\*\*(?P<title>.+?)\*\*\.?\s*")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().rstrip(".").strip()).lower()


def strip_duplicate_title(message: str) -> str:
    for pattern in (HEADING_RE, BOLD_RE):
        m = pattern.search(message)
        if not m:
            continue
        title, rest = m.group("title"), message[m.end():]
        if rest.strip() and _norm(rest).startswith(_norm(title)):
            cleaned = message[:m.start()] + m.group("lead") + rest
            return cleaned.strip("\n")
    return message


def load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
def build_workplace_posts() -> list[dict]:
    groups_index = {g["id"]: g for g in load_json(MW_DIR / "groups.json")}

    raw_posts: list[tuple[str, dict, str | None]] = []  # (post_id, post, group_name)

    for fp in sorted((MW_DIR / "group-data").glob("data-part*.json")):
        for group_id, group_meta in load_json(fp).items():
            group_name = (groups_index.get(group_id) or {}).get("name")
            for post in group_meta.get("feed") or []:
                raw_posts.append((post.get("id"), post, group_name))

    members_feed = load_json(MW_DIR / "members_feed.json")
    for _user_id, blob in members_feed.items():
        for post in blob.get("feed") or []:
            raw_posts.append((post.get("id"), post, None))  # personal post: no group

    # dedupe by id, first occurrence wins
    seen: dict[str, tuple] = {}
    for post_id, post, group_name in raw_posts:
        seen.setdefault(post_id, (post, group_name))
    n_dropped = len(raw_posts) - len(seen)

    out = []
    n_no_message = 0
    for post_id, (post, group_name) in seen.items():
        message = post.get("message")
        if not message:
            n_no_message += 1
            continue
        out.append({
            "id": post_id,
            "created_time": post.get("created_time"),
            "message_clean": strip_duplicate_title(message),
            "group_name": group_name,
        })

    print(f"Workplace: {len(raw_posts)} raw feed entries, {n_dropped} pagination duplicates "
          f"dropped, {n_no_message} with no message text dropped -> {len(out)} posts")
    return out


def build_yammer_messages() -> list[dict]:
    out = []
    with YM_MESSAGES_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get("body"):
                continue
            out.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "body": row["body"],
                "group_name": row.get("group_name") or None,
            })
    print(f"Yammer: {len(out)} messages with body text")
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wp_posts = build_workplace_posts()
    with OUT_WP.open("w", encoding="utf-8") as fh:
        json.dump(wp_posts, fh, ensure_ascii=False)
    print(f"wrote {len(wp_posts):,} rows -> {OUT_WP}")

    ym_messages = build_yammer_messages()
    with OUT_YM.open("w", encoding="utf-8") as fh:
        json.dump(ym_messages, fh, ensure_ascii=False)
    print(f"wrote {len(ym_messages):,} rows -> {OUT_YM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
