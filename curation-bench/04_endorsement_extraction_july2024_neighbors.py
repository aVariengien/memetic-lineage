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

"""Endorsement extraction for the top-5 cohort's 10 closest July 2024 neighbors.

For each focal user (hardcoded top-5):
  * pull their July 2024 outgoing reply counts from monthly_reply_graph_top100.json,
  * drop self-replies,
  * take the top 10 distinct reply targets,
  * filter against the eligible-users set (skip + log non-eligible),
  * union all eligible neighbors across the 5 focal users,
  * extract their July 2024 endorsement targets via the shared pipeline.

Output JSON contains:
  * an intermediate effective-neighbor-graph JSON with only the eligible
    neighbors actually used per focal user,
  * resolved neighbor lists per focal user (with eligibility flags + skip reasons),
  * the union of eligible neighbors actually processed,
  * the flat list of endorsement targets, each carrying which neighbor authored
    the anchor tweet so downstream tooling can re-group per focal user.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
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
    parse_eligible_usernames,
    run_endorsement_pipeline,
)

WORKSPACE_DIR = CURATION_BENCH_DIR.parent
SCRATCHPADS_DIR = WORKSPACE_DIR / "scratchpads"

MONTHLY_REPLY_GRAPH_PATH = CURATION_BENCH_DIR / "data" / "monthly_reply_graph_top100.json"
RAW_USER_DIRECTORY_PATH = SCRATCHPADS_DIR / "data" / "raw_copy_paste_user_directory.txt"
TWEET_ID_SUBSETS_PATH = SCRATCHPADS_DIR / "data" / "problem_resolution" / "tweet_id_subsets_aug2024.json"
TWEET_ID_SUBSET_KEY = "six_month_to_sep_2024"

TOP5_USERNAMES = [
    "goblinodds",
    "daniellefong",
    "exgenesis",
    "archived_videos",
    "danielbrottman",
]

JULY_MONTH_KEY = "2024-07"
DEFAULT_WINDOW_START = "2024-07-01 00:00:00"
DEFAULT_WINDOW_END = "2024-08-01 00:00:00"
DEFAULT_TOP_NEIGHBORS = 10
ARCHIVE_UPLOAD_CUTOFF = pd.Timestamp("2025-09-01")

DEFAULT_PATH_CONCURRENCY = 50
DEFAULT_BATCH_SIZE = 12
DEFAULT_CONCURRENCY = 6
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract July 2024 endorsement / disendorsement targets for the top-10 "
            "outgoing-reply neighbors of the hardcoded top-5 cohort."
        )
    )
    parser.add_argument("--start", default=DEFAULT_WINDOW_START, help="Inclusive window start.")
    parser.add_argument("--end", default=DEFAULT_WINDOW_END, help="Exclusive window end.")
    parser.add_argument("--top-neighbors", type=int, default=DEFAULT_TOP_NEIGHBORS)
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


def select_neighbors_for_focal(
    july_graph: dict[str, dict[str, int]],
    focal_username: str,
    top_n: int,
    eligible_users: set[str],
) -> dict[str, Any]:
    """Pick top-N outgoing-reply neighbors for one focal user (excludes self-reply).

    Returns a record of:
      * raw_top_n: ordered [{username, reply_count, eligible, skip_reason}]
      * eligible_neighbors: ordered list of usernames that survive eligibility filter
    """
    if focal_username not in july_graph:
        return {
            "focal_username": focal_username,
            "raw_top_n": [],
            "eligible_neighbors": [],
            "warning": f"focal user @{focal_username} has no outgoing entries in {JULY_MONTH_KEY}",
        }

    targets = july_graph[focal_username]
    filtered = [
        (target, count)
        for target, count in targets.items()
        if normalize_username(target) != focal_username
    ]
    filtered.sort(key=lambda item: (-item[1], item[0]))
    top = filtered[:top_n]

    raw_top_n: list[dict[str, Any]] = []
    eligible_neighbors: list[str] = []
    for target_username, reply_count in top:
        is_eligible = target_username in eligible_users
        skip_reason = None if is_eligible else "not in eligible users (no archive uploaded before 2025-09-01)"
        raw_top_n.append(
            {
                "username": target_username,
                "reply_count": int(reply_count),
                "eligible": is_eligible,
                "skip_reason": skip_reason,
            }
        )
        if is_eligible:
            eligible_neighbors.append(target_username)

    return {
        "focal_username": focal_username,
        "raw_top_n": raw_top_n,
        "eligible_neighbors": eligible_neighbors,
    }


def build_effective_neighbor_graph(neighbor_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return the filtered neighbor graph actually used downstream."""
    effective_graph: dict[str, list[dict[str, Any]]] = {}
    for record in neighbor_records:
        focal_username = record["focal_username"]
        effective_graph[focal_username] = [
            {
                "username": entry["username"],
                "reply_count": entry["reply_count"],
            }
            for entry in record["raw_top_n"]
            if entry["eligible"]
        ]
    return effective_graph


def collect_neighbor_window_candidates(
    subset_ids: list[int],
    tweet_dict: Any,
    window_start: str,
    window_end: str,
    target_users: set[str],
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Walk a pre-filtered subset of tweet_ids to gather candidate tweets for the neighbors.

    `subset_ids` must be a complete enumeration of every tweet in the archive
    whose `created_at` falls inside [window_start, window_end). The
    six_month_to_sep_2024 subset (Mar 2024 .. Sep 2024) satisfies this for any
    sub-window inside that range -- we verified completeness via random
    sampling against tweet_dict.
    """
    candidates: list[dict[str, Any]] = []
    excluded_missing_parent = 0
    in_window_total = 0
    matched_target = 0

    for tweet_id in tqdm(subset_ids, desc="Scanning subset for neighbor window"):
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
        matched_target += 1

        if not is_candidate_tweet(tweet):
            continue

        if not has_present_parent(tweet, tweet_dict):
            excluded_missing_parent += 1
            continue

        try:
            tweet_id_int = int(tweet_id)
        except (TypeError, ValueError):
            tqdm.write(f"[error] non-integer tweet_id encountered: {tweet_id!r}")
            continue

        candidates.append(
            {
                "tweet_id": tweet_id_int,
                "username": username,
                "created_at": created_at,
                "reply_to_tweet_id": tweet.get("reply_to_tweet_id"),
                "quoted_tweet_id": tweet.get("quoted_tweet_id"),
                "full_text": tweet.get("full_text", ""),
            }
        )

    return candidates, in_window_total, matched_target, excluded_missing_parent


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
    if args.top_neighbors <= 0:
        raise ValueError("--top-neighbors must be positive")
    if args.start >= args.end:
        raise ValueError("--start must be earlier than --end")

    if not MONTHLY_REPLY_GRAPH_PATH.exists():
        raise FileNotFoundError(f"Missing reply graph: {MONTHLY_REPLY_GRAPH_PATH}")

    with MONTHLY_REPLY_GRAPH_PATH.open("r", encoding="utf-8") as file:
        graph_payload = json.load(file)
    july_graph = graph_payload.get(JULY_MONTH_KEY)
    if not isinstance(july_graph, dict):
        raise KeyError(f"No '{JULY_MONTH_KEY}' month in {MONTHLY_REPLY_GRAPH_PATH}")

    if not TWEET_ID_SUBSETS_PATH.exists():
        raise FileNotFoundError(f"Missing subset file: {TWEET_ID_SUBSETS_PATH}")
    with TWEET_ID_SUBSETS_PATH.open("r", encoding="utf-8") as file:
        subset_payload = json.load(file)
    subset_range = subset_payload["ranges"][TWEET_ID_SUBSET_KEY]
    subset_start = subset_range["start_inclusive"]
    subset_end = subset_range["end_exclusive"]
    if args.start < subset_start or args.end > subset_end:
        raise ValueError(
            f"Window [{args.start}..{args.end}) is not contained in subset "
            f"{TWEET_ID_SUBSET_KEY!r} [{subset_start}..{subset_end})."
        )
    subset_ids = [int(tid) for tid in subset_payload["tweet_ids"][TWEET_ID_SUBSET_KEY]]
    print(
        f"Loaded subset {TWEET_ID_SUBSET_KEY!r}: {len(subset_ids):,} tweet ids "
        f"({subset_start} .. {subset_end})"
    )

    eligible_users = parse_eligible_usernames(RAW_USER_DIRECTORY_PATH, ARCHIVE_UPLOAD_CUTOFF)

    neighbor_records: list[dict[str, Any]] = []
    union_neighbors: set[str] = set()
    skipped_log: list[dict[str, Any]] = []
    for focal in TOP5_USERNAMES:
        record = select_neighbors_for_focal(
            july_graph=july_graph,
            focal_username=focal,
            top_n=args.top_neighbors,
            eligible_users=eligible_users,
        )
        neighbor_records.append(record)
        union_neighbors.update(record["eligible_neighbors"])
        for entry in record["raw_top_n"]:
            if not entry["eligible"]:
                skipped_log.append(
                    {
                        "focal_username": focal,
                        "neighbor_username": entry["username"],
                        "reply_count": entry["reply_count"],
                        "skip_reason": entry["skip_reason"],
                    }
                )

    print(f"Focal users: {len(TOP5_USERNAMES)}")
    for record in neighbor_records:
        focal = record["focal_username"]
        eligible = record["eligible_neighbors"]
        skipped_for_focal = [
            entry for entry in record["raw_top_n"] if not entry["eligible"]
        ]
        print(
            f"  @{focal:<22}  eligible_neighbors={len(eligible):>2}  "
            f"skipped_in_top10={len(skipped_for_focal):>2}"
        )
        for entry in skipped_for_focal:
            print(
                f"    [skip] @{entry['username']:<22} replies={entry['reply_count']:<4} "
                f"reason: {entry['skip_reason']}"
            )

    print(f"Union of eligible neighbors across focal users: {len(union_neighbors):,}")

    if not union_neighbors:
        raise ValueError("No eligible neighbors found across the focal users.")

    window_slug = (
        args.start[:10].replace("-", "") + "_" + args.end[:10].replace("-", "") + "_top5_neighbors"
    )
    effective_neighbor_graph = build_effective_neighbor_graph(neighbor_records)
    effective_neighbor_graph_path = args.output_dir / f"{window_slug}_effective_neighbor_graph.json"
    effective_neighbor_graph_payload = {
        "cohort": "top5_neighbors_jul2024",
        "month_key": JULY_MONTH_KEY,
        "window": {"start_inclusive": args.start, "end_exclusive": args.end},
        "top_neighbors_requested": args.top_neighbors,
        "selection_rule": (
            "Take each focal user's top-N outgoing-reply neighbors for the month, "
            "exclude self-replies, then drop usernames that are not in the eligible "
            "user set. This file records the filtered graph actually used downstream."
        ),
        "effective_neighbor_graph": effective_neighbor_graph,
    }
    with effective_neighbor_graph_path.open("w", encoding="utf-8") as file:
        json.dump(effective_neighbor_graph_payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(f"Wrote effective neighbor graph: {effective_neighbor_graph_path}")

    tweet_dict, conversation_trees = load_caches(auto_generate=False)
    print(f"Loaded {len(tweet_dict):,} tweets and {len(conversation_trees):,} conversation trees")

    candidates, in_window_total, matched_target, excluded_missing_parent = collect_neighbor_window_candidates(
        subset_ids=subset_ids,
        tweet_dict=tweet_dict,
        window_start=args.start,
        window_end=args.end,
        target_users=union_neighbors,
    )
    user_counts = Counter(item["username"] for item in candidates)
    print(f"Tweets in window {args.start}..{args.end}: {in_window_total:,}")
    print(f"Tweets matching neighbor cohort: {matched_target:,}")
    print(f"Candidate tweets (non-RT, parent present): {len(candidates):,}")
    print(f"Excluded replies with missing parent tweet: {excluded_missing_parent:,}")
    for username in sorted(union_neighbors):
        print(f"  @{username:<22} {user_counts.get(username, 0):>5}")

    if not candidates:
        raise ValueError("No candidate tweets in window for the eligible neighbors.")

    cohort_meta: dict[str, Any] = {
        "cohort": "top5_neighbors_jul2024",
        "focal_users": TOP5_USERNAMES,
        "window": {"start_inclusive": args.start, "end_exclusive": args.end},
        "neighbor_selection": {
            "month_key": JULY_MONTH_KEY,
            "top_neighbors": args.top_neighbors,
            "exclude_self_replies": True,
            "graph_source": str(MONTHLY_REPLY_GRAPH_PATH),
            "effective_neighbor_graph_path": str(effective_neighbor_graph_path),
        },
        "neighbors_per_focal": neighbor_records,
        "effective_neighbor_graph": effective_neighbor_graph,
        "skipped_neighbors_log": skipped_log,
        "union_eligible_neighbors": sorted(union_neighbors),
    }
    per_user_counts = {
        username: user_counts.get(username, 0) for username in sorted(union_neighbors)
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
    )


if __name__ == "__main__":
    main()
