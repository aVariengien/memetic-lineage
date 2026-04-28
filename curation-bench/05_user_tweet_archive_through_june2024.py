#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "diskcache",
#   "httpx",
#   "pandas",
#   "pyarrow",
#   "requests",
#   "tqdm",
# ]
# ///

"""Per-user tweet archive (through end of June 2024) for the hardcoded top-5 cohort.

For each user we walk the FULL tweet cache once to collect every tweet they authored
(originals, replies, retweets, quotes). Tweets are grouped into conversation trees:
  * the tree key is the tweet's conversation_id when present, else the tweet itself,
  * we keep trees that contain at least one user-authored tweet strictly before
    2024-07-01 00:00:00 UTC,
  * each tree is trimmed so we render the user's nodes, all of their ancestors, and
    their direct replies (one level of children),
  * any rendered tweet with created_at >= cutoff is redacted in place rather than
    removing the whole tree,
  * the tree is filed under the year/month of the user's FIRST pre-cutoff tweet in
    that tree.

Outputs per user @username:
  * data/user_archives/<username>/all_tweets_through_jun2024.md   (one big file)
  * data/user_archives/<username>/<YYYY>/<MM>.md                  (indexed by month)

`t.co` URLs in rendered text can optionally be resolved using the shared cache
(live HTTP for misses).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.dataset as ds
from tqdm import tqdm

CURATION_BENCH_DIR = Path(__file__).resolve().parent
if str(CURATION_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(CURATION_BENCH_DIR))

from lib import (  # noqa: E402
    TCO_URL_PATTERN,
    clean_rendered_tree,
    created_at_str,
    load_caches,
    load_url_resolution_cache,
    make_header_renderer,
    normalize_username,
    now_utc_iso,
    persist_url_resolution_cache,
    render_conversation_trees,
    resolve_urls_in_text,
    trim_tree_to_user,
    warm_url_cache,
)

TOP5_USERNAMES = [
    "goblinodds",
    "daniellefong",
    "exgenesis",
    "archived_videos",
    "danielbrottman",
]

DEFAULT_CUTOFF = "2024-07-01 00:00:00"
DEFAULT_OUTPUT_DIR = CURATION_BENCH_DIR / "data" / "user_archives"
DEFAULT_URL_CONCURRENCY = 500
DEFAULT_TWEETS_PARQUET = Path(
    os.environ.get("ENRICHED_TWEETS_PATH", str(Path.home() / "data" / "enriched_tweets.parquet"))
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive every tweet authored by the hardcoded top-5 cohort from the start "
            "of their history up to (exclusive) the cutoff date, grouped by conversation "
            "tree and rendered to Markdown."
        )
    )
    parser.add_argument(
        "--cutoff",
        default=DEFAULT_CUTOFF,
        help=(
            "Exclusive cutoff (UTC). Trees with no user tweet before cutoff are skipped; "
            "rendered tweets at or after cutoff are redacted in place."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for per-user archive folders.",
    )
    parser.add_argument(
        "--url-concurrency",
        type=int,
        default=DEFAULT_URL_CONCURRENCY,
        help="Concurrency for warming the t.co URL resolution cache when URL resolution is enabled.",
    )
    parser.add_argument(
        "--resolve-urls",
        action="store_true",
        help="Resolve t.co URLs in rendered output. Disabled by default to avoid extra network work.",
    )
    parser.add_argument(
        "--tweets-parquet",
        type=Path,
        default=DEFAULT_TWEETS_PARQUET,
        help=(
            "Parquet dataset used to prefilter authored tweets by username before rendering. "
            "Falls back to scanning tweet_dict if the path does not exist."
        ),
    )
    return parser.parse_args()


def collect_user_tweets_from_cache(
    tweet_dict: Any,
    target_users: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Single pass over tweet_dict to gather every tweet authored by each target user."""
    per_user: dict[str, list[dict[str, Any]]] = {user: [] for user in target_users}

    iterable = tweet_dict.iterkeys() if hasattr(tweet_dict, "iterkeys") else iter(tweet_dict)
    for tweet_id in tqdm(iterable, total=len(tweet_dict), desc="Scanning full tweet cache"):
        tweet = tweet_dict.get(tweet_id)
        if not tweet:
            continue
        username = normalize_username(tweet.get("username", ""))
        if username not in target_users:
            continue

        try:
            tweet_id_int = int(tweet_id)
        except (TypeError, ValueError):
            tqdm.write(f"[error] non-integer tweet_id encountered: {tweet_id!r}")
            continue

        per_user[username].append(
            {
                "tweet_id": tweet_id_int,
                "created_at": created_at_str(tweet),
                "conversation_id": tweet.get("conversation_id"),
                "tweet": tweet,
            }
        )

    for user, items in per_user.items():
        items.sort(key=lambda entry: (entry["created_at"], entry["tweet_id"]))
        print(f"  @{user:<22} authored tweets in cache: {len(items):>6}")

    return per_user


def collect_user_tweets_from_parquet(
    parquet_path: Path,
    target_users: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Filter authored tweets from parquet using only the columns we need."""
    per_user: dict[str, list[dict[str, Any]]] = {user: [] for user in target_users}

    print(f"Prefiltering authored tweets from parquet: {parquet_path}")
    dataset = ds.dataset(str(parquet_path), format="parquet")
    username_table = dataset.to_table(columns=["username"])
    unique_usernames = pc.unique(username_table["username"].combine_chunks()).to_pylist()
    matched_usernames = sorted(
        {
            str(username)
            for username in unique_usernames
            if username is not None and normalize_username(username) in target_users
        }
    )

    if not matched_usernames:
        print("[warn] No matching usernames found in parquet after case-insensitive normalization")
        return per_user

    print(f"Matched parquet username variants: {matched_usernames}")
    table = dataset.to_table(
        columns=["tweet_id", "username", "created_at", "conversation_id"],
        filter=ds.field("username").isin(matched_usernames),
    )

    print(f"Parquet prefilter matched {table.num_rows:,} rows")
    rows = table.to_pylist()
    for row in tqdm(rows, desc="Grouping parquet-filtered tweets"):
        username = normalize_username(row.get("username", ""))
        if username not in target_users:
            continue

        tweet_id = row.get("tweet_id")
        if tweet_id is None:
            tqdm.write(f"[error] parquet row missing tweet_id for @{username}: {row!r}")
            continue

        try:
            tweet_id_int = int(tweet_id)
        except (TypeError, ValueError):
            tqdm.write(f"[error] non-integer tweet_id encountered in parquet: {tweet_id!r}")
            continue

        created_at = str(row.get("created_at", ""))[:19]
        if not created_at:
            tqdm.write(f"[error] parquet row missing created_at for tweet {tweet_id_int}")
            continue

        conversation_id = row.get("conversation_id")
        if conversation_id is not None:
            try:
                conversation_id = int(conversation_id)
            except (TypeError, ValueError):
                tqdm.write(
                    f"[error] invalid conversation_id in parquet for tweet {tweet_id_int}: "
                    f"{conversation_id!r}"
                )
                conversation_id = None

        per_user[username].append(
            {
                "tweet_id": tweet_id_int,
                "created_at": created_at,
                "conversation_id": conversation_id,
            }
        )

    for user, items in per_user.items():
        items.sort(key=lambda entry: (entry["created_at"], entry["tweet_id"]))
        print(f"  @{user:<22} authored tweets in parquet: {len(items):>6}")

    return per_user


def collect_user_tweets(
    tweet_dict: Any,
    target_users: set[str],
    tweets_parquet: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Use parquet prefilter when available, otherwise scan the full tweet cache."""
    tweets_parquet = tweets_parquet.expanduser()
    if tweets_parquet.exists():
        return collect_user_tweets_from_parquet(tweets_parquet, target_users)

    print(
        f"[warn] tweets parquet not found at {tweets_parquet}; "
        "falling back to scanning the full tweet cache"
    )
    return collect_user_tweets_from_cache(tweet_dict, target_users)


def group_user_tweets_by_tree(
    user_tweets: list[dict[str, Any]],
    conversation_trees: Any,
) -> dict[int, dict[str, Any]]:
    """Group a user's tweets by their conversation/reply tree key.

    Returns: tree_key -> {tree_key, user_tweet_ids (sorted), first_created_at, last_created_at}.
    """
    groups: dict[int, dict[str, Any]] = {}
    for entry in user_tweets:
        tweet_id = entry["tweet_id"]
        conversation_id = entry.get("conversation_id")
        tree_key = tweet_id
        if conversation_id is not None:
            tree = conversation_trees.get(conversation_id) or conversation_trees.get(str(conversation_id))
            if tree:
                tree_key = int(conversation_id)
            else:
                fallback_tree = conversation_trees.get(tweet_id) or conversation_trees.get(str(tweet_id))
                if fallback_tree:
                    tree_key = int(tweet_id)
        else:
            tree = conversation_trees.get(tweet_id) or conversation_trees.get(str(tweet_id))
            if tree:
                tree_key = int(tweet_id)

        bucket = groups.get(tree_key)
        if bucket is None:
            groups[tree_key] = {
                "tree_key": tree_key,
                "user_tweet_ids": [tweet_id],
                "first_created_at": entry["created_at"],
                "last_created_at": entry["created_at"],
            }
        else:
            bucket["user_tweet_ids"].append(tweet_id)
            if entry["created_at"] < bucket["first_created_at"]:
                bucket["first_created_at"] = entry["created_at"]
            if entry["created_at"] > bucket["last_created_at"]:
                bucket["last_created_at"] = entry["created_at"]

    for bucket in groups.values():
        bucket["user_tweet_ids"].sort()
    return groups


def _collect_tree_node_ids(tree: dict[str, Any]) -> set[int]:
    node_ids: set[int] = set()
    root = tree.get("root")
    if root is not None:
        node_ids.add(int(root))

    for node_id, parent_id in (tree.get("parents", {}) or {}).items():
        node_ids.add(int(node_id))
        node_ids.add(int(parent_id))

    for node_id, child_ids in (tree.get("children", {}) or {}).items():
        node_ids.add(int(node_id))
        node_ids.update(int(child_id) for child_id in child_ids)

    return node_ids


class _TweetRenderView:
    """Overlay redacted tweet payloads on top of the shared tweet cache."""

    def __init__(self, base_tweets: Any, overrides: dict[int, dict[str, Any]]) -> None:
        self.base_tweets = base_tweets
        self.overrides = overrides

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self.overrides:
            return self.overrides[key]
        try:
            key_int = int(key)
        except (TypeError, ValueError):
            key_int = None
        if key_int is not None and key_int in self.overrides:
            return self.overrides[key_int]

        if hasattr(self.base_tweets, "get"):
            result = self.base_tweets.get(key)
            if result is not None:
                return result
            if key_int is not None:
                result = self.base_tweets.get(key_int)
                if result is not None:
                    return result
        return default


def _build_render_tweet_dict(
    tweet_dict: Any,
    tree: dict[str, Any],
    cutoff: str,
) -> tuple[_TweetRenderView, list[int]]:
    redacted_overrides: dict[int, dict[str, Any]] = {}
    redacted_tweet_ids: list[int] = []

    for node_id in sorted(_collect_tree_node_ids(tree)):
        tweet = tweet_dict.get(node_id) or tweet_dict.get(str(node_id))
        if not tweet:
            continue

        created_at = created_at_str(tweet)
        if created_at and created_at >= cutoff:
            redacted = dict(tweet)
            redacted["full_text"] = f"[redacted: created_at >= {cutoff} UTC]"
            redacted_overrides[node_id] = redacted
            redacted_tweet_ids.append(node_id)

    return _TweetRenderView(tweet_dict, redacted_overrides), redacted_tweet_ids


def render_tree_for_user(
    tree_key: int,
    user_tweet_ids: list[int],
    tweet_dict: Any,
    conversation_trees: Any,
    cutoff: str,
) -> tuple[str | None, list[int]]:
    """Render the trimmed tree (user nodes + ancestors + direct children) to Markdown."""
    base_tree = conversation_trees.get(tree_key) or conversation_trees.get(str(tree_key))
    if base_tree is None:
        # Solo tweet (no parent, no replies present in cache). Render the lone tweet.
        single_tweet_id = user_tweet_ids[0]
        synthetic_tree = {
            "root": single_tweet_id,
            "children": {},
            "parents": {},
        }
        render_tweets, redacted_tweet_ids = _build_render_tweet_dict(
            tweet_dict=tweet_dict,
            tree=synthetic_tree,
            cutoff=cutoff,
        )
        rendered = render_conversation_trees(
            {single_tweet_id: synthetic_tree},
            render_tweets,
            render_header=make_header_renderer(),
        )
        return clean_rendered_tree(rendered), redacted_tweet_ids

    trimmed = trim_tree_to_user(base_tree, set(user_tweet_ids))
    if trimmed.get("root") is None:
        return None, []

    render_tweets, redacted_tweet_ids = _build_render_tweet_dict(
        tweet_dict=tweet_dict,
        tree=trimmed,
        cutoff=cutoff,
    )

    rendered = render_conversation_trees(
        {tree_key: trimmed},
        render_tweets,
        render_header=make_header_renderer(),
    )
    return clean_rendered_tree(rendered), redacted_tweet_ids


def collect_tco_urls(text: str) -> list[str]:
    return TCO_URL_PATTERN.findall(text)


def write_markdown_files(
    username: str,
    cutoff: str,
    rendered_trees: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """Write the big-file and the year/month-indexed Markdown files for one user."""
    user_dir = output_dir / username
    user_dir.mkdir(parents=True, exist_ok=True)

    rendered_trees_sorted = sorted(
        rendered_trees,
        key=lambda entry: (entry["first_created_at"], entry["tree_key"]),
    )

    big_path = user_dir / "all_tweets_through_jun2024.md"
    big_lines: list[str] = [
        f"# @{username} — tweet archive (created_at < {cutoff} UTC)",
        "",
        f"Generated at {now_utc_iso()}",
        "",
        f"Trees included: {len(rendered_trees_sorted):,}",
        "",
        "Conversation trees are printed below in chronological order. Trees are "
        "trimmed to the user's nodes, all of their ancestors, and their direct replies.",
        "",
        "---",
        "",
    ]

    monthly_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in rendered_trees_sorted:
        first_at = entry["first_created_at"]
        year = first_at[:4]
        month = first_at[5:7]
        monthly_buckets[(year, month)].append(entry)

    for entry in rendered_trees_sorted:
        big_lines.extend(_format_tree_section(entry))

    big_path.write_text("\n".join(big_lines), encoding="utf-8")

    monthly_summary: list[dict[str, Any]] = []
    for (year, month), entries in sorted(monthly_buckets.items()):
        month_dir = user_dir / year
        month_dir.mkdir(parents=True, exist_ok=True)
        month_path = month_dir / f"{month}.md"

        month_lines: list[str] = [
            f"# @{username} — {year}-{month} (created_at < {cutoff} UTC)",
            "",
            f"Trees in this month: {len(entries):,}",
            "",
            "Conversation trees are printed below in chronological order.",
            "",
            "---",
            "",
        ]
        for entry in entries:
            month_lines.extend(_format_tree_section(entry))
        month_path.write_text("\n".join(month_lines), encoding="utf-8")
        monthly_summary.append(
            {
                "year": year,
                "month": month,
                "tree_count": len(entries),
                "path": str(month_path.relative_to(output_dir)),
            }
        )

    return {
        "big_file": str(big_path.relative_to(output_dir)),
        "monthly_files": monthly_summary,
        "tree_count_total": len(rendered_trees_sorted),
    }


def _format_tree_section(entry: dict[str, Any]) -> list[str]:
    return [
        "```text",
        entry["rendered"].rstrip(),
        "```",
        "",
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resolve_urls:
        load_url_resolution_cache()
    else:
        print("URL resolution: disabled")

    if args.resolve_urls and args.url_concurrency <= 0:
        raise ValueError("--url-concurrency must be positive")

    cutoff = args.cutoff
    print(f"Archive cutoff (exclusive): {cutoff}")
    print(f"Target users ({len(TOP5_USERNAMES)}): {TOP5_USERNAMES}")
    print(f"Tweets parquet: {args.tweets_parquet.expanduser()}")

    tweet_dict, conversation_trees = load_caches(auto_generate=False)
    print(f"Loaded {len(tweet_dict):,} tweets and {len(conversation_trees):,} reply trees")

    target_users = set(TOP5_USERNAMES)
    per_user_tweets = collect_user_tweets(tweet_dict, target_users, args.tweets_parquet)

    per_user_summary: dict[str, Any] = {}

    for username in TOP5_USERNAMES:
        all_tweets = per_user_tweets.get(username, [])
        if not all_tweets:
            print(f"[warn] @{username}: no authored tweets found in cache")
            per_user_summary[username] = {
                "user_tweet_count_total": 0,
                "tree_count_total": 0,
                "tree_count_dropped_post_cutoff": 0,
                "tree_count_excluded_without_pre_cutoff_user_tweet": 0,
                "files": None,
            }
            continue

        groups = group_user_tweets_by_tree(all_tweets, conversation_trees)
        created_at_by_tweet_id = {entry["tweet_id"]: entry["created_at"] for entry in all_tweets}
        kept_groups = []
        excluded_without_pre_cutoff_user_tweet = 0
        for bucket in groups.values():
            pre_cutoff_user_tweet_ids = [
                tweet_id
                for tweet_id in bucket["user_tweet_ids"]
                if created_at_by_tweet_id.get(tweet_id, "") < cutoff
            ]
            if not pre_cutoff_user_tweet_ids:
                excluded_without_pre_cutoff_user_tweet += 1
                continue

            redacted_user_tweet_ids = [
                tweet_id
                for tweet_id in bucket["user_tweet_ids"]
                if created_at_by_tweet_id.get(tweet_id, "") >= cutoff
            ]
            kept_groups.append(
                {
                    **bucket,
                    "all_user_tweet_ids": list(bucket["user_tweet_ids"]),
                    "user_tweet_ids": pre_cutoff_user_tweet_ids,
                    "first_created_at": min(
                        created_at_by_tweet_id[tweet_id] for tweet_id in pre_cutoff_user_tweet_ids
                    ),
                    "last_created_at": max(
                        created_at_by_tweet_id[tweet_id] for tweet_id in pre_cutoff_user_tweet_ids
                    ),
                    "redacted_user_tweet_ids": redacted_user_tweet_ids,
                }
            )

        kept_groups.sort(key=lambda b: (b["first_created_at"], b["tree_key"]))
        print(
            f"@{username}: {len(all_tweets):,} authored tweets across {len(groups):,} trees; "
            f"keeping {len(kept_groups):,} trees (excluded "
            f"{excluded_without_pre_cutoff_user_tweet:,} with no user tweet before {cutoff}; "
            "post-cutoff rendered tweets are redacted)"
        )

        if not kept_groups:
            per_user_summary[username] = {
                "user_tweet_count_total": len(all_tweets),
                "tree_count_total": len(groups),
                "tree_count_dropped_post_cutoff": excluded_without_pre_cutoff_user_tweet,
                "tree_count_excluded_without_pre_cutoff_user_tweet": (
                    excluded_without_pre_cutoff_user_tweet
                ),
                "files": None,
            }
            continue

        rendered_entries: list[dict[str, Any]] = []
        for bucket in tqdm(kept_groups, desc=f"@{username} render"):
            rendered, redacted_tweet_ids = render_tree_for_user(
                tree_key=bucket["tree_key"],
                user_tweet_ids=bucket["all_user_tweet_ids"],
                tweet_dict=tweet_dict,
                conversation_trees=conversation_trees,
                cutoff=cutoff,
            )
            if rendered is None:
                tqdm.write(
                    f"[error] @{username}: tree {bucket['tree_key']} could not be rendered "
                    f"(skipping). user_tweet_ids={bucket['all_user_tweet_ids'][:5]}..."
                )
                continue
            rendered_entries.append(
                {
                    **bucket,
                    "rendered": rendered,
                    "redacted_tweet_ids": redacted_tweet_ids,
                }
            )

        if args.resolve_urls:
            unresolved = sorted({
                url
                for entry in rendered_entries
                for url in collect_tco_urls(entry["rendered"])
            })
            print(f"@{username}: warming {len(unresolved):,} unique t.co URLs")
            if unresolved:
                warm_url_cache(unresolved, args.url_concurrency, desc=f"@{username} URLs")

            for entry in rendered_entries:
                entry["rendered"] = resolve_urls_in_text(entry["rendered"])

        files_summary = write_markdown_files(
            username=username,
            cutoff=cutoff,
            rendered_trees=rendered_entries,
            output_dir=args.output_dir,
        )
        per_user_summary[username] = {
            "user_tweet_count_total": len(all_tweets),
            "tree_count_total": len(groups),
            "tree_count_dropped_post_cutoff": excluded_without_pre_cutoff_user_tweet,
            "tree_count_excluded_without_pre_cutoff_user_tweet": (
                excluded_without_pre_cutoff_user_tweet
            ),
            "tree_count_kept": len(kept_groups),
            "tree_count_rendered": len(rendered_entries),
            "files": files_summary,
        }

    if args.resolve_urls:
        persist_url_resolution_cache(force=True)

    summary_path = args.output_dir / "_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": now_utc_iso(),
                "cutoff_exclusive": cutoff,
                "users": TOP5_USERNAMES,
                "per_user": per_user_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote summary {summary_path}")


if __name__ == "__main__":
    main()
