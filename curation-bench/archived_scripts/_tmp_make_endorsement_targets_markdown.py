#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TARGETS_PATH = Path("curation-bench/data/20240604_20240607_sample353_endorsement_targets.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert endorsement target JSON into markdown with printed paths."
    )
    parser.add_argument(
        "targets_path",
        nargs="?",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
        help="Path to *_endorsement_targets.json",
    )
    parser.add_argument(
        "--paths",
        dest="paths_path",
        type=Path,
        default=None,
        help="Optional path to matching *_paths.json file",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=None,
        help="Optional markdown output path",
    )
    return parser.parse_args()


def infer_paths_path(targets_path: Path) -> Path:
    name = targets_path.name
    if not name.endswith("_endorsement_targets.json"):
        raise ValueError(
            f"Cannot infer matching paths file from {targets_path}. "
            "Expected a filename ending with _endorsement_targets.json"
        )
    return targets_path.with_name(name.replace("_endorsement_targets.json", "_paths.json"))


def infer_output_path(targets_path: Path) -> Path:
    return targets_path.with_suffix(".md")


def choose_best_path_item(path_items: list[dict[str, Any]], representative_tweet_id: int) -> dict[str, Any] | None:
    if not path_items:
        return None

    def sort_key(item: dict[str, Any]) -> tuple[int, int, str, int]:
        final_tweet_id = int(item["tweet_id"])
        path_length = len(item.get("path_tweet_ids", []))
        created_at = str(item.get("created_at", ""))
        direct_match_penalty = 0 if final_tweet_id == representative_tweet_id else 1
        return (direct_match_penalty, path_length, created_at, final_tweet_id)

    return min(path_items, key=sort_key)


def main() -> None:
    args = parse_args()
    targets_path = args.targets_path
    paths_path = args.paths_path or infer_paths_path(targets_path)
    output_path = args.output_path or infer_output_path(targets_path)

    with targets_path.open("r", encoding="utf-8") as file:
        targets_data = json.load(file)
    with paths_path.open("r", encoding="utf-8") as file:
        paths_data = json.load(file)

    path_items = list(paths_data["items"])
    path_items_by_member_id: dict[int, list[dict[str, Any]]] = {}
    for item in path_items:
        for tweet_id in item.get("path_tweet_ids", []):
            path_items_by_member_id.setdefault(int(tweet_id), []).append(item)

    target_items = list(targets_data["items"])

    lines: list[str] = []
    lines.append("# Endorsement Targets")
    lines.append("")
    lines.append(f"Targets source: `{targets_path.name}`")
    lines.append(f"Paths source: `{paths_path.name}`")
    lines.append("")
    lines.append(f"Model: `{targets_data['model']}`")
    lines.append(
        f"Window: `{targets_data['window']['start_inclusive']}` to `{targets_data['window']['end_exclusive']}`"
    )
    lines.append("")
    lines.append(f"Total extracted targets: `{len(target_items)}`")
    lines.append("")

    for index, target in enumerate(target_items, start=1):
        representative_tweet_id = int(target["representative_tweet_id"])
        matching_paths = path_items_by_member_id.get(representative_tweet_id, [])
        path_item = choose_best_path_item(matching_paths, representative_tweet_id)

        lines.append(f"## {index}. `{target['target_entity']}`")
        lines.append("")
        lines.append(f"- Representative tweet id: `{representative_tweet_id}`")
        lines.append(f"- Direction: `{target['direction']}`")
        lines.append(f"- Longer name: `{target['longer_name']}`")
        lines.append("")

        if path_item is None:
            lines.append("- Matching path: `not found in paths JSON`")
            lines.append("")
        else:
            lines.append(f"- Path username: `@{path_item['username']}`")
            lines.append(f"- Path final tweet id: `{path_item['tweet_id']}`")
            lines.append(f"- Path created at: `{path_item['created_at']}`")
            lines.append(f"- Path length: `{len(path_item.get('path_tweet_ids', []))}`")
            lines.append(f"- Full path ids: `{path_item.get('path_tweet_ids', [])}`")
            if len(matching_paths) > 1:
                lines.append(f"- Matching maximal paths containing representative tweet: `{len(matching_paths)}`")
            lines.append("")
            lines.append("```text")
            lines.append(str(path_item["path_text"]).rstrip())
            lines.append("```")
            lines.append("")

        lines.append("```json")
        lines.append(json.dumps(target, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(target_items)} endorsement targets to {output_path}")


if __name__ == "__main__":
    main()
