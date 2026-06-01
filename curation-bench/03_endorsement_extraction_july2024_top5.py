#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "diskcache",
#   "groq",
#   "httpx",
#   "openai",
#   "pandas",
#   "python-dotenv",
#   "requests",
#   "tqdm",
# ]
# ///

"""Endorsement target extraction for the hardcoded top-5 cohort, July 2024 window.

Same path/extraction logic as 01_endorsement_extraction.py but:
  * users are HARDCODED (the 5 users from the Jun-Aug 2024 cohort),
  * window is the full month of July 2024,
  * LLM output schema includes a `url` field,
  * default model is Gemini Flash Lite,
  * orchestration delegates to ``lib.endorsement_extraction``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

CURATION_BENCH_DIR = Path(__file__).resolve().parent
if str(CURATION_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(CURATION_BENCH_DIR))

from lib import (  # noqa: E402
    created_at_str,
    has_present_parent,
    is_candidate_tweet,
    load_caches,
    load_url_resolution_cache,
    normalize_username,
    run_endorsement_pipeline,
)

WORKSPACE_DIR = CURATION_BENCH_DIR.parent
SCRATCHPADS_DIR = WORKSPACE_DIR / "scratchpads"

TWEET_ID_SUBSETS_PATH = SCRATCHPADS_DIR / "data" / "problem_resolution" / "tweet_id_subsets_aug2024.json"

TOP5_USERNAMES = [
    "goblinodds",
    "daniellefong",
    "exgenesis",
    "archived_videos",
    "danielbrottman",
]

DEFAULT_WINDOW_START = "2024-07-01 00:00:00"
DEFAULT_WINDOW_END = "2024-08-01 00:00:00"

DEFAULT_PATH_CONCURRENCY = 50
DEFAULT_BATCH_SIZE = 12
DEFAULT_CONCURRENCY = 50
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract endorsement / disendorsement targets from reply paths anchored at "
            f"tweets authored in {DEFAULT_WINDOW_START[:10]}..{DEFAULT_WINDOW_END[:10]} by "
            f"the hardcoded top-5 cohort: {TOP5_USERNAMES}."
        )
    )
    parser.add_argument("--start", default=DEFAULT_WINDOW_START, help="Inclusive window start.")
    parser.add_argument("--end", default=DEFAULT_WINDOW_END, help="Exclusive window end.")
    parser.add_argument("--path-concurrency", type=int, default=DEFAULT_PATH_CONCURRENCY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Stop after generating the rendered-paths JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CURATION_BENCH_DIR / "data",
        help="Directory for output JSON files.",
    )
    return parser.parse_args()


def collect_window_candidates(
    tweet_dict: Any,
    subset_ids: list[int],
    window_start: str,
    window_end: str,
    target_users: set[str],
) -> tuple[list[dict[str, Any]], int, int]:
    """Walk the precomputed subset and pull candidate tweets in window for target_users."""
    candidates: list[dict[str, Any]] = []
    excluded_missing_parent = 0
    in_window_total = 0

    for tweet_id in tqdm(subset_ids, desc="Scanning subset for window"):
        tweet = tweet_dict.get(tweet_id) or tweet_dict.get(str(tweet_id))
        if not tweet:
            continue

        created_at = created_at_str(tweet)
        if not (window_start <= created_at < window_end):
            continue

        in_window_total += 1

        username = normalize_username(tweet.get("username", ""))
        if username not in target_users:
            continue

        if not is_candidate_tweet(tweet):
            continue

        if not has_present_parent(tweet, tweet_dict):
            excluded_missing_parent += 1
            continue

        candidates.append(
            {
                "tweet_id": int(tweet_id),
                "username": username,
                "created_at": created_at,
                "reply_to_tweet_id": tweet.get("reply_to_tweet_id"),
                "quoted_tweet_id": tweet.get("quoted_tweet_id"),
                "full_text": tweet.get("full_text", ""),
            }
        )

    return candidates, in_window_total, excluded_missing_parent


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_url_resolution_cache()

    if args.path_concurrency <= 0:
        raise ValueError("--path-concurrency must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.start >= args.end:
        raise ValueError("--start must be earlier than --end")

    target_users = set(TOP5_USERNAMES)
    print(f"Target users ({len(target_users)}): {sorted(target_users)}")

    tweet_dict, conversation_trees = load_caches(auto_generate=False)
    print(f"Loaded {len(tweet_dict):,} tweets and {len(conversation_trees):,} conversation trees")

    if not TWEET_ID_SUBSETS_PATH.exists():
        raise FileNotFoundError(f"Missing subset file: {TWEET_ID_SUBSETS_PATH}")

    with TWEET_ID_SUBSETS_PATH.open("r", encoding="utf-8") as file:
        subset_data = json.load(file)
    subset_ids = [int(tweet_id) for tweet_id in subset_data["tweet_ids"]["six_month_to_sep_2024"]]

    candidates, in_window_total, excluded_missing_parent = collect_window_candidates(
        tweet_dict=tweet_dict,
        subset_ids=subset_ids,
        window_start=args.start,
        window_end=args.end,
        target_users=target_users,
    )
    user_counts = Counter(item["username"] for item in candidates)
    print(f"Tweets in window {args.start}..{args.end}: {in_window_total:,}")
    print(f"Candidate tweets (top-5 cohort, non-RT, parent present): {len(candidates):,}")
    print(f"Excluded replies with missing parent tweet: {excluded_missing_parent:,}")
    for username in sorted(target_users):
        print(f"  @{username:<22} {user_counts.get(username, 0):>5}")

    if not candidates:
        raise ValueError("No candidate tweets in window for the top-5 cohort.")

    window_slug = (
        args.start[:10].replace("-", "") + "_" + args.end[:10].replace("-", "") + "_top5"
    )
    cohort_meta: dict[str, Any] = {
        "cohort": "hardcoded_top5_jun_aug_2024",
        "users": sorted(target_users),
        "window": {"start_inclusive": args.start, "end_exclusive": args.end},
    }
    per_user_counts = {
        username: user_counts.get(username, 0) for username in sorted(target_users)
    }

    run_endorsement_pipeline(
        candidates=candidates,
        tweet_dict=tweet_dict,
        conversation_trees=conversation_trees,
        paths_path=args.output_dir / f"{window_slug}_paths.json",
        targets_path=args.output_dir / f"{window_slug}_endorsement_targets.json",
        paths_payload_extras=cohort_meta,
        targets_payload_extras=cohort_meta,
        paths_stats_extras={
            "excluded_missing_parent": excluded_missing_parent,
            "per_user_candidate_counts": per_user_counts,
        },
        targets_stats_extras={
            "per_user_candidate_counts": per_user_counts,
        },
        model=args.model,
        batch_size=args.batch_size,
        llm_concurrency=args.concurrency,
        path_concurrency=args.path_concurrency,
        paths_only=args.paths_only,
        window_start=args.start,
        window_end=args.end,
    )


if __name__ == "__main__":
    main()
