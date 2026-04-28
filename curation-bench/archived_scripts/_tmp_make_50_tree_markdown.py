#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

import json
from pathlib import Path


INPUT_PATH = Path("curation-bench/data/20240601_20240615_sample300_paths.json")
OUTPUT_PATH = Path("curation-bench/data/20240601_20240615_first50_trees.md")
N_TREES = 50


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    items = data["items"][:N_TREES]

    lines: list[str] = []
    lines.append("# First 50 Printed Trees")
    lines.append("")
    lines.append(f"Source: `{INPUT_PATH.name}`")
    lines.append("")
    lines.append(
        f"Window: `{data['window']['start_inclusive']}` to `{data['window']['end_exclusive']}`"
    )
    lines.append("")
    lines.append(f"Showing the first `{len(items)}` sampled items.")
    lines.append("")

    for index, item in enumerate(items, start=1):
        lines.append(f"## {index}. @{item['username']} `{item['tweet_id']}`")
        lines.append("")
        lines.append(f"- Created at: `{item['created_at']}`")
        lines.append(f"- Path length: `{len(item.get('path_tweet_ids', []))}`")
        lines.append("")
        lines.append("```text")
        lines.append(item["path_text"].rstrip())
        lines.append("```")
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(items)} trees to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
