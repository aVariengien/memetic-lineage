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

"""Build one ground-truth-labels file per focal user for the top-5 cohort.

For each focal user X (top-5 cohort, July 2024):
  * Gather the items the labeling UI would show:
      - X's own targets (from ground_truth_labels/...top5_endorsement_targets.json,
        path_anchor_username == X)
      - Targets from X's social neighborhood (from neighbors/...endorsement_targets.json,
        path_anchor_username in effective_neighbor_graph[X])
  * For each item, look up the author of representative_tweet_id in tweet_dict
    (the source of truth of who actually authored the endorsing/disendorsing tweet).
  * Assign ground_truth_label:
      - if representative_tweet_author == X: use the LLM-extracted direction
        (endorsing / disendorsing)
      - else: neutral (X did not author the endorsement)
  * Dedup by (representative_tweet_id, target_entity, direction); focal-source wins
    on conflict.
  * Sort items by representative_tweet_id (matches the UI's pre-shuffle ordering).

Writes one file per focal user to:
  data/curation_bench_clean_data/labels/ground_truth_<focal_user>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CURATION_BENCH_DIR = Path(__file__).resolve().parent
if str(CURATION_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(CURATION_BENCH_DIR))

from lib import load_caches, normalize_username  # noqa: E402

DATA_DIR = CURATION_BENCH_DIR / "data"
CLEAN_DIR = DATA_DIR / "curation_bench_clean_data"

DEFAULT_FOCAL_TARGETS = CLEAN_DIR / "ground_truth_labels" / "20240701_20240801_top5_endorsement_targets.json"
DEFAULT_NEIGHBOR_TARGETS = CLEAN_DIR / "neighbors" / "20240701_20240801_top5_neighbors_endorsement_targets.json"
DEFAULT_NEIGHBOR_GRAPH = CLEAN_DIR / "20240701_20240801_top5_neighbors_effective_neighbor_graph.json"
DEFAULT_OUTPUT_DIR = CLEAN_DIR / "labels"

DEFAULT_FOCAL_USERS = [
    "goblinodds",
    "daniellefong",
    "exgenesis",
    "archived_videos",
    "danielbrottman",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal-targets", type=Path, default=DEFAULT_FOCAL_TARGETS)
    parser.add_argument("--neighbor-targets", type=Path, default=DEFAULT_NEIGHBOR_TARGETS)
    parser.add_argument("--neighbor-graph", type=Path, default=DEFAULT_NEIGHBOR_GRAPH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--focal-users",
        nargs="+",
        default=DEFAULT_FOCAL_USERS,
        help="Focal usernames to build ground-truth files for.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def lookup_author(tweet_dict: Any, tweet_id: Any) -> str | None:
    """Look up the author username for a tweet_id; tries both int and str keys."""
    if tweet_id is None:
        return None
    tweet = tweet_dict.get(tweet_id) or tweet_dict.get(str(tweet_id))
    if not tweet:
        try:
            tweet = tweet_dict.get(int(tweet_id))
        except (TypeError, ValueError):
            return None
    if not tweet:
        return None
    username = tweet.get("username")
    if not username:
        return None
    return normalize_username(username)


def build_per_focal(
    focal: str,
    focal_items: list[dict[str, Any]],
    neighbor_items: list[dict[str, Any]],
    neighbors_in_scope: set[str],
    rep_tweet_authors: dict[Any, str | None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (items, stats) for one focal user."""
    selected: list[tuple[dict[str, Any], str]] = []  # (item, source)

    for item in focal_items:
        if item.get("path_anchor_username") == focal:
            selected.append((item, "focal"))
    for item in neighbor_items:
        anchor = item.get("path_anchor_username")
        if anchor in neighbors_in_scope:
            selected.append((item, "neighbor"))

    # Dedup: (rep_tweet_id, target_entity, direction); focal-source wins on tie.
    # Iterate focal first so its items land in the dict before neighbor copies.
    selected.sort(key=lambda pair: 0 if pair[1] == "focal" else 1)
    deduped: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    skipped_dups = 0
    for item, source in selected:
        key = (
            item.get("representative_tweet_id"),
            item.get("target_entity"),
            item.get("direction"),
        )
        if key in deduped:
            skipped_dups += 1
            continue
        rep_id = item.get("representative_tweet_id")
        author = rep_tweet_authors.get(rep_id)
        if author == focal:
            ground_truth_label = item.get("direction")  # endorsing or disendorsing
        else:
            ground_truth_label = "neutral"
        deduped[key] = {
            "representative_tweet_id": rep_id,
            "representative_tweet_author": author,
            "direction": item.get("direction"),
            "target_entity": item.get("target_entity"),
            "longer_name": item.get("longer_name"),
            "context": item.get("context"),
            "url": item.get("url"),
            "path_anchor_tweet_id": item.get("path_anchor_tweet_id"),
            "path_anchor_username": item.get("path_anchor_username"),
            "source": source,
            "ground_truth_label": ground_truth_label,
        }

    items = sorted(
        deduped.values(),
        key=lambda it: (it.get("representative_tweet_id") or 0),
    )

    label_counts = Counter(it["ground_truth_label"] for it in items)
    source_counts = Counter(it["source"] for it in items)
    author_focal = sum(1 for it in items if it["representative_tweet_author"] == focal)
    author_neighbor = sum(
        1 for it in items if it["representative_tweet_author"] in neighbors_in_scope
    )
    author_other = sum(
        1
        for it in items
        if it["representative_tweet_author"] not in (neighbors_in_scope | {focal})
    )

    stats = {
        "total": len(items),
        "by_label": dict(label_counts),
        "by_source_file": dict(source_counts),
        "by_representative_author_bucket": {
            "focal_self": author_focal,
            "neighbor_in_scope": author_neighbor,
            "other_or_unknown": author_other,
        },
        "skipped_duplicate_keys": skipped_dups,
    }
    return items, stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading focal targets:    {args.focal_targets}")
    focal_data = load_json(args.focal_targets)
    print(f"Loading neighbor targets: {args.neighbor_targets}")
    neighbor_data = load_json(args.neighbor_targets)
    print(f"Loading neighbor graph:   {args.neighbor_graph}")
    graph_data = load_json(args.neighbor_graph)

    focal_items = focal_data.get("items", [])
    neighbor_items = neighbor_data.get("items", [])
    neighbor_graph = graph_data.get("effective_neighbor_graph", {})

    print(f"  focal items:    {len(focal_items):,}")
    print(f"  neighbor items: {len(neighbor_items):,}")

    # Collect all unique representative_tweet_ids we need to look up
    rep_ids_needed: set[Any] = set()
    for item in focal_items:
        rep_ids_needed.add(item.get("representative_tweet_id"))
    for item in neighbor_items:
        rep_ids_needed.add(item.get("representative_tweet_id"))
    rep_ids_needed.discard(None)
    print(f"\nNeed to look up authors for {len(rep_ids_needed):,} unique representative_tweet_ids.")

    print("Loading tweet_dict (this can take a moment)...")
    tweet_dict, _conv_trees = load_caches(auto_generate=False)
    print(f"  tweet_dict size: {len(tweet_dict):,}")

    rep_tweet_authors: dict[Any, str | None] = {}
    missing = 0
    for rid in rep_ids_needed:
        author = lookup_author(tweet_dict, rid)
        rep_tweet_authors[rid] = author
        if author is None:
            missing += 1
    print(
        f"  authors resolved: {len(rep_tweet_authors) - missing:,} / "
        f"{len(rep_tweet_authors):,}  (missing: {missing:,})"
    )

    window = focal_data.get("window") or neighbor_data.get("window")
    generated_at = datetime.now(timezone.utc).isoformat()
    common_meta = {
        "generated_at": generated_at,
        "cohort": "top5_jul2024",
        "window": window,
        "ground_truth_rule": (
            "ground_truth_label is the LLM-extracted `direction` (endorsing / "
            "disendorsing) when the focal user authored representative_tweet_id; "
            "otherwise neutral. Author is looked up from tweet_dict; if the tweet "
            "is missing from the cache, label defaults to neutral."
        ),
        "label_options": ["endorsing", "disendorsing", "neutral"],
        "source_files": {
            "focal_targets": str(args.focal_targets),
            "neighbor_targets": str(args.neighbor_targets),
            "neighbor_graph": str(args.neighbor_graph),
        },
        "models": {
            "focal_targets_model": focal_data.get("model"),
            "neighbor_targets_model": neighbor_data.get("model"),
        },
    }

    print()
    print("Per-focal-user output:")
    print(f"  {'focal':<20} {'total':>6} {'endorsing':>10} {'disendorsing':>13} {'neutral':>8}  -> file")
    for focal in args.focal_users:
        neighbors = {
            entry["username"]
            for entry in neighbor_graph.get(focal, [])
        }
        items, stats = build_per_focal(
            focal=focal,
            focal_items=focal_items,
            neighbor_items=neighbor_items,
            neighbors_in_scope=neighbors,
            rep_tweet_authors=rep_tweet_authors,
        )

        payload = {
            "focal_user": focal,
            **common_meta,
            "neighbors_in_scope": sorted(neighbors),
            "stats": stats,
            "items": items,
        }

        out_path = args.output_dir / f"ground_truth_{focal}.json"
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
        tmp.replace(out_path)

        by_label = stats["by_label"]
        print(
            f"  {focal:<20} {stats['total']:>6} "
            f"{by_label.get('endorsing', 0):>10} "
            f"{by_label.get('disendorsing', 0):>13} "
            f"{by_label.get('neutral', 0):>8}"
            f"  -> {out_path.relative_to(CURATION_BENCH_DIR)}"
        )
        bucket = stats["by_representative_author_bucket"]
        if bucket["other_or_unknown"]:
            print(
                f"    (representative-author breakdown: focal={bucket['focal_self']}, "
                f"in-scope-neighbor={bucket['neighbor_in_scope']}, "
                f"other/unknown={bucket['other_or_unknown']})"
            )

    print(f"\nDone. Files written to {args.output_dir}")


if __name__ == "__main__":
    main()
