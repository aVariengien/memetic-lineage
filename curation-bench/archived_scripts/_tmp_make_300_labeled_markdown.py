#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

import json
from pathlib import Path


PATHS_JSON = Path("curation-bench/data/20240601_20240615_sample300_paths.json")
LABELS_JSON = Path("curation-bench/data/20240601_20240615_sample300_stance_classification.json")
OUTPUT_MD = Path("curation-bench/data/20240601_20240615_sample300_labeled_paths.md")
LABEL_ORDER = ["endorsing", "disendorsing", "no_stance"]


def label_key(result: dict) -> str:
    if result.get("has_stance"):
        return str(result.get("stance"))
    return "no_stance"


def main() -> None:
    with PATHS_JSON.open("r", encoding="utf-8") as file:
        paths_data = json.load(file)
    with LABELS_JSON.open("r", encoding="utf-8") as file:
        labels_data = json.load(file)

    path_items = paths_data["items"]
    labels_by_id = {
        int(result["tweet_id"]): result
        for result in labels_data["results"]
    }

    grouped: dict[str, list[dict]] = {label: [] for label in LABEL_ORDER}
    for item in path_items:
        tweet_id = int(item["tweet_id"])
        result = labels_by_id[tweet_id]
        grouped[label_key(result)].append({**item, "classification": result})

    lines: list[str] = []
    lines.append("# Sample 300 Labeled Paths")
    lines.append("")
    lines.append(f"Paths source: `{PATHS_JSON.name}`")
    lines.append(f"Labels source: `{LABELS_JSON.name}`")
    lines.append("")
    lines.append(
        f"Window: `{paths_data['window']['start_inclusive']}` to `{paths_data['window']['end_exclusive']}`"
    )
    lines.append(f"Model: `{labels_data['model']}`")
    lines.append("")

    for label in LABEL_ORDER:
        items = grouped[label]
        lines.append(f"## {label} ({len(items)})")
        lines.append("")

        for index, item in enumerate(items, start=1):
            result = item["classification"]
            lines.append(f"### {index}. @{item['username']} `{item['tweet_id']}`")
            lines.append("")
            lines.append(f"- Created at: `{item['created_at']}`")
            lines.append(f"- DeepSeek label: `{label}`")
            lines.append(f"- Confidence: `{result['confidence']}`")
            if result.get("target_entity"):
                lines.append(f"- Target entity: `{result['target_entity']}`")
            lines.append(f"- Reason: {result['reason']}")
            lines.append("")
            lines.append("```text")
            lines.append(item["path_text"].rstrip())
            lines.append("```")
            lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote labeled markdown to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
