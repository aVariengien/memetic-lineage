#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "diskcache",
#   "pandas",
#   "tqdm",
# ]
# ///

"""Build a monthly direct-reply graph for the top-N Jun-Aug 2024 cohort.

Default N is 100; pass --top-n-users to override. The default output
filename includes the chosen N (e.g. monthly_reply_graph_top100.json)
so different cohort sizes don't overwrite each other.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from diskcache import Cache
import pandas as pd
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
CURATION_BENCH_DIR = Path(__file__).resolve().parent
SCRATCHPADS_DIR = WORKSPACE_DIR / "scratchpads"

RAW_USER_DIRECTORY_PATH = SCRATCHPADS_DIR / "data" / "raw_copy_paste_user_directory.txt"
TWEET_ID_SUBSETS_PATH = SCRATCHPADS_DIR / "data" / "problem_resolution" / "tweet_id_subsets_aug2024.json"
DISKCACHE_ROOT = Path.home() / "data" / "scratchpads"
TWEET_DICT_DISKCACHE = DISKCACHE_ROOT / "tweet_dict.diskcache"
REPLY_TREES_DISKCACHE = DISKCACHE_ROOT / "reply_trees.diskcache"

THREE_MONTH_START = "2024-06-01 00:00:00"
THREE_MONTH_END = "2024-09-01 00:00:00"
ARCHIVE_UPLOAD_CUTOFF = pd.Timestamp("2025-09-01")

DEFAULT_SUBSET_KEY = "six_month_to_sep_2024"
DEFAULT_TOP_N_USERS = 100


def default_output_path(top_n_users: int) -> Path:
    return CURATION_BENCH_DIR / "data" / f"monthly_reply_graph_top{top_n_users}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each month in the selected subset time span, compute direct reply counts "
            "from the top active Jun-Aug 2024 users to the usernames they replied to."
        )
    )
    parser.add_argument("--subset-key", default=DEFAULT_SUBSET_KEY, help="Subset key inside tweet_id_subsets JSON.")
    parser.add_argument("--top-n-users", type=int, default=DEFAULT_TOP_N_USERS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSON output path. If omitted, defaults to "
            "data/monthly_reply_graph_top{TOP_N}.json so different N values "
            "don't overwrite each other."
        ),
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_username(username: str) -> str:
    return username.strip().lstrip("@").lower() if username else ""


def parse_eligible_usernames(path: Path, cutoff: pd.Timestamp) -> set[str]:
    eligible = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) < 8 or not parts[1].strip().startswith("@"):
                continue
            uploaded_at = pd.to_datetime(parts[7].strip(), format="%d/%m/%Y", errors="coerce")
            if pd.notna(uploaded_at) and pd.Timestamp(uploaded_at) < cutoff:
                eligible.add(normalize_username(parts[1].strip()))
    print(f"Eligible users (archive before {cutoff.date()}): {len(eligible):,}")
    return eligible


def created_at_str(tweet: dict) -> str:
    return str(tweet.get("created_at", ""))[:19]


def month_slug(created_at: str) -> str:
    return created_at[:7]


def iter_months(start_month: str, end_month: str) -> list[str]:
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def load_subset_ids(path: Path, subset_key: str) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing subset file: {path}")

    with path.open("r", encoding="utf-8") as file:
        subset_data = json.load(file)

    tweet_ids = subset_data.get("tweet_ids", {}).get(subset_key)
    if tweet_ids is None:
        available = sorted(subset_data.get("tweet_ids", {}).keys())
        raise KeyError(f"Subset key {subset_key!r} not found. Available keys: {available}")

    return [int(tweet_id) for tweet_id in tweet_ids]


def load_tweet_cache() -> Cache:
    if not TWEET_DICT_DISKCACHE.exists():
        raise FileNotFoundError(
            f"Tweet diskcache not found at {TWEET_DICT_DISKCACHE}. "
            "Generate the caches first if needed."
        )
    if not REPLY_TREES_DISKCACHE.exists():
        raise FileNotFoundError(
            f"Reply tree diskcache not found at {REPLY_TREES_DISKCACHE}. "
            "Generate the caches first if needed."
        )

    print("Opening diskcache stores...")
    tweet_dict = Cache(str(TWEET_DICT_DISKCACHE))
    reply_trees = Cache(str(REPLY_TREES_DISKCACHE))
    print(f"Loaded {len(tweet_dict):,} tweets and {len(reply_trees):,} reply trees")
    return tweet_dict


def build_top_usernames(tweet_dict: dict, subset_ids: list[int], top_n_users: int) -> list[str]:
    eligible_users = parse_eligible_usernames(RAW_USER_DIRECTORY_PATH, ARCHIVE_UPLOAD_CUTOFF)

    three_month_ids: list[int] = []
    missing_subset_tweets = 0
    for tweet_id in tqdm(subset_ids, desc="Filtering subset to Jun-Aug"):
        tweet = tweet_dict.get(tweet_id) or tweet_dict.get(str(tweet_id))
        if not tweet:
            missing_subset_tweets += 1
            tqdm.write(f"[error] missing tweet in cache for subset id {tweet_id}")
            continue

        created_at = created_at_str(tweet)
        if THREE_MONTH_START <= created_at < THREE_MONTH_END:
            three_month_ids.append(tweet_id)

    if missing_subset_tweets:
        tqdm.write(f"[error] missing subset tweets total: {missing_subset_tweets}")

    user_counts: Counter[str] = Counter()
    for tweet_id in tqdm(three_month_ids, desc="Counting Jun-Aug cohort activity"):
        tweet = tweet_dict.get(tweet_id) or tweet_dict.get(str(tweet_id))
        if not tweet:
            continue
        username = normalize_username(tweet.get("username", ""))
        if username in eligible_users:
            user_counts[username] += 1

    top_users = user_counts.most_common(top_n_users)
    print(f"Top {top_n_users} users from Jun-Aug 2024 cohort:")
    for index, (username, count) in enumerate(top_users, start=1):
        print(f"  {index:>2}. @{username:<22} {count:>5} tweets")
    return [username for username, _ in top_users]


def compute_month_range(tweet_dict: dict, subset_ids: list[int]) -> list[str]:
    months: list[str] = []
    missing_subset_tweets = 0
    for tweet_id in tqdm(subset_ids, desc="Scanning subset month range"):
        tweet = tweet_dict.get(tweet_id) or tweet_dict.get(str(tweet_id))
        if not tweet:
            missing_subset_tweets += 1
            continue

        created_at = created_at_str(tweet)
        if created_at:
            months.append(month_slug(created_at))

    if missing_subset_tweets:
        tqdm.write(f"[error] skipped {missing_subset_tweets} subset tweets while scanning month range")
    if not months:
        raise ValueError("No dated tweets found in subset.")

    return iter_months(min(months), max(months))


def build_monthly_reply_graph(tweet_dict: dict, subset_ids: list[int], top_usernames: set[str], months: list[str]) -> dict:
    graph: dict[str, dict[str, dict[str, int]]] = {month: {} for month in months}
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))

    missing_subset_tweets = 0
    missing_parent_tweets = 0
    parent_without_username = 0

    for tweet_id in tqdm(subset_ids, desc="Aggregating direct replies by month"):
        tweet = tweet_dict.get(tweet_id) or tweet_dict.get(str(tweet_id))
        if not tweet:
            missing_subset_tweets += 1
            tqdm.write(f"[error] missing tweet in cache for subset id {tweet_id}")
            continue

        source_user = normalize_username(tweet.get("username", ""))
        if source_user not in top_usernames:
            continue

        parent_id = tweet.get("reply_to_tweet_id")
        if parent_id is None:
            continue

        parent_tweet = tweet_dict.get(parent_id) or tweet_dict.get(str(parent_id))
        if not parent_tweet:
            missing_parent_tweets += 1
            continue

        target_user = normalize_username(parent_tweet.get("username", ""))
        if not target_user:
            parent_without_username += 1
            tqdm.write(f"[error] parent tweet {parent_id} for reply {tweet_id} has no username")
            continue

        created_at = created_at_str(tweet)
        if not created_at:
            tqdm.write(f"[error] reply tweet {tweet_id} has no created_at")
            continue

        counts[month_slug(created_at)][source_user][target_user] += 1

    if missing_subset_tweets:
        tqdm.write(f"[error] missing subset tweets total during aggregation: {missing_subset_tweets}")
    if missing_parent_tweets:
        tqdm.write(f"[error] missing parent tweets total: {missing_parent_tweets}")
    if parent_without_username:
        tqdm.write(f"[error] parent tweets missing username total: {parent_without_username}")

    for month in months:
        month_map = counts.get(month, {})
        graph[month] = {
            source_user: dict(sorted(targets.items()))
            for source_user, targets in sorted(month_map.items())
        }

    return graph


def main() -> None:
    args = parse_args()
    if args.top_n_users <= 0:
        raise ValueError("--top-n-users must be positive")

    output_path: Path = args.output or default_output_path(args.top_n_users)

    tweet_dict = load_tweet_cache()

    subset_ids = load_subset_ids(TWEET_ID_SUBSETS_PATH, args.subset_key)
    print(f"Loaded {len(subset_ids):,} tweet ids from subset {args.subset_key!r}")

    top_usernames_list = build_top_usernames(tweet_dict, subset_ids, args.top_n_users)
    top_usernames = set(top_usernames_list)
    months = compute_month_range(tweet_dict, subset_ids)
    print(f"Month range: {months[0]} -> {months[-1]}")

    graph = build_monthly_reply_graph(tweet_dict, subset_ids, top_usernames, months)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(graph, file, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Saved monthly reply graph to {output_path}")
    print(f"Generated at {now_utc()}")


if __name__ == "__main__":
    main()
