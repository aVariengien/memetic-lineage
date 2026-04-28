#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

import json
from pathlib import Path


CLASSIFICATION_PATH = Path("curation-bench/data/20240604_20240607_sample562_stance_classification.json")
PATHS_PATH = Path("curation-bench/data/20240604_20240607_sample562_paths.json")
OUTPUT_PATH = Path("curation-bench/data/20240604_20240607_has_stance_true.md")


def main() -> None:
    with CLASSIFICATION_PATH.open("r", encoding="utf-8") as file:
        classification_data = json.load(file)
    with PATHS_PATH.open("r", encoding="utf-8") as file:
        paths_data = json.load(file)

    path_by_id = {
        int(item["tweet_id"]): item
        for item in paths_data["items"]
    }

    positive_results = [
        result
        for result in classification_data["results"]
        if result.get("has_stance") is True
    ]

    lines: list[str] = []
    lines.append("# Tweets With `has_stance = true`")
    lines.append("")
    lines.append(f"Classification source: `{CLASSIFICATION_PATH.name}`")
    lines.append(f"Paths source: `{PATHS_PATH.name}`")
    lines.append("")
    lines.append(f"Model: `{classification_data['model']}`")
    lines.append(
        f"Window: `{classification_data['window']['start_inclusive']}` to `{classification_data['window']['end_exclusive']}`"
    )
    lines.append("")
    lines.append(f"Total with `has_stance = true`: `{len(positive_results)}`")
    lines.append("")

    for index, result in enumerate(positive_results, start=1):
        tweet_id = int(result["tweet_id"])
        path_item = path_by_id.get(tweet_id)

        lines.append(f"## {index}. `{tweet_id}`")
        lines.append("")
        if path_item is not None:
            lines.append(f"- Username: `@{path_item['username']}`")
            lines.append(f"- Created at: `{path_item['created_at']}`")
            lines.append(f"- Path length: `{len(path_item.get('path_tweet_ids', []))}`")
            lines.append(f"- Full path ids: `{path_item.get('path_tweet_ids', [])}`")
            lines.append("")
            lines.append("```text")
            lines.append(path_item["path_text"].rstrip())
            lines.append("```")
            lines.append("")

        lines.append("```json")
        lines.append(json.dumps(result, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(positive_results)} stance-positive tweets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
