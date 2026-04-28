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

"""One-off sanity check: is `six_month_to_sep_2024` a complete enumeration of
every tweet in [2024-03-01, 2024-09-01) inside tweet_dict?

Procedure:
  1. Sample N random tweet IDs from tweet_dict.
  2. Pull each tweet, parse `created_at`.
  3. Compute:
       - % of sampled tweets that fall in the 6-month window.
       - % of sampled tweets in the window that are NOT in the subset
         (should be 0 if the subset is complete).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

CURATION_BENCH_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURATION_BENCH_DIR.parent
SCRATCHPADS_DIR = WORKSPACE_DIR / "scratchpads"

if str(SCRATCHPADS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCHPADS_DIR))

from lib.strand_caches import load_caches  # noqa: E402

TWEET_ID_SUBSETS_PATH = SCRATCHPADS_DIR / "data" / "problem_resolution" / "tweet_id_subsets_aug2024.json"

WINDOW_START = "2024-03-01 00:00:00"
WINDOW_END = "2024-09-01 00:00:00"
SUBSET_KEY = "six_month_to_sep_2024"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading subset from {TWEET_ID_SUBSETS_PATH.name}...")
    with TWEET_ID_SUBSETS_PATH.open("r", encoding="utf-8") as file:
        subset_data = json.load(file)
    subset_meta = subset_data["ranges"][SUBSET_KEY]
    subset_ids = {int(tid) for tid in subset_data["tweet_ids"][SUBSET_KEY]}
    print(
        f"Subset {SUBSET_KEY!r}: "
        f"{subset_meta['start_inclusive']} .. {subset_meta['end_exclusive']} "
        f"({len(subset_ids):,} tweet ids)"
    )

    print("Loading tweet_dict diskcache...")
    tweet_dict, _ = load_caches(auto_generate=False)
    total_tweets = len(tweet_dict)
    print(f"Loaded {total_tweets:,} tweets")

    print("Enumerating tweet_dict keys (one pass)...")
    all_keys = list(tweet_dict)
    print(f"Got {len(all_keys):,} keys")

    sample_size = min(args.sample_size, len(all_keys))
    random.seed(args.seed)
    sampled = random.sample(all_keys, sample_size)
    print(f"Sampling {sample_size:,} random tweet ids (seed={args.seed})")

    in_window = 0
    in_window_missing_from_subset = 0
    missing_examples: list[tuple[int, str]] = []
    skipped_no_created_at = 0
    skipped_unparseable_id = 0

    for raw_key in tqdm(sampled, desc="Checking sampled tweets"):
        try:
            tweet_id_int = int(raw_key)
        except (TypeError, ValueError):
            skipped_unparseable_id += 1
            continue
        tweet = tweet_dict.get(raw_key)
        if tweet is None:
            tweet = tweet_dict.get(tweet_id_int) or tweet_dict.get(str(tweet_id_int))
        if tweet is None:
            continue
        created_at = str(tweet.get("created_at", ""))[:19]
        if not created_at:
            skipped_no_created_at += 1
            continue
        if WINDOW_START <= created_at < WINDOW_END:
            in_window += 1
            if tweet_id_int not in subset_ids:
                in_window_missing_from_subset += 1
                if len(missing_examples) < 10:
                    missing_examples.append((tweet_id_int, created_at))

    pct_in_window = 100.0 * in_window / sample_size if sample_size else 0.0
    pct_missing = (
        100.0 * in_window_missing_from_subset / in_window if in_window else 0.0
    )

    print()
    print("=" * 60)
    print(f"Sample size:                              {sample_size:,}")
    print(f"Skipped (no created_at):                  {skipped_no_created_at:,}")
    print(f"Skipped (unparseable id):                 {skipped_unparseable_id:,}")
    print(f"In 6-month window:                        {in_window:,}  ({pct_in_window:.2f}%)")
    print(
        f"In window AND missing from subset:        {in_window_missing_from_subset:,}  "
        f"({pct_missing:.2f}% of in-window)"
    )
    print("=" * 60)

    if missing_examples:
        print("First missing examples (tweet_id, created_at):")
        for tweet_id, created_at in missing_examples:
            print(f"  {tweet_id}  {created_at}")
    else:
        print("No in-window tweets were missing from the subset ✓")


if __name__ == "__main__":
    main()
