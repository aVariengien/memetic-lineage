#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

import json
from pathlib import Path


PATHS_JSON = Path("curation-bench/data/20240601_20240615_sample300_paths.json")
LABELS_JSON = Path("curation-bench/data/20240601_20240615_sample300_stance_classification.json")
OUTPUT_MD = Path("curation-bench/data/20240601_20240615_sample300_reasoner_grouped.md")

STANCE_ORDER = ["endorsing", "disendorsing"]
VAGUENESS_ORDER = ["crisp", "vague"]
CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def sort_key(item: dict) -> tuple:
    result = item["classification"]
    strength = result.get("endorsement_strength")
    strength_rank = -(strength if isinstance(strength, int) else 0)
    confidence_rank = CONFIDENCE_RANK.get(result.get("confidence"), 9)
    return (strength_rank, confidence_rank, item["created_at"], item["tweet_id"])


def render_item(lines: list[str], index: int, item: dict, label: str) -> None:
    result = item["classification"]
    lines.append(f"### {index}. @{item['username']} `{item['tweet_id']}`")
    lines.append("")
    lines.append(f"- Created at: `{item['created_at']}`")
    lines.append(f"- Label: `{label}`")
    lines.append(f"- Confidence: `{result['confidence']}`")
    if result.get("target_entity"):
        lines.append(f"- Target entity: `{result['target_entity']}`")
    if result.get("target_entity_type"):
        lines.append(f"- Target entity type: `{result['target_entity_type']}`")
    if result.get("endorsement_vagueness"):
        lines.append(f"- Endorsement vagueness: `{result['endorsement_vagueness']}`")
    if result.get("endorsement_strength") is not None:
        lines.append(f"- Endorsement strength: `{result['endorsement_strength']}`")
    lines.append(f"- Reason: {result['reason']}")
    lines.append("")
    lines.append("```text")
    lines.append(item["path_text"].rstrip())
    lines.append("```")
    lines.append("")


def main() -> None:
    with PATHS_JSON.open("r", encoding="utf-8") as file:
        paths_data = json.load(file)
    with LABELS_JSON.open("r", encoding="utf-8") as file:
        labels_data = json.load(file)

    labels_by_id = {int(result["tweet_id"]): result for result in labels_data["results"]}
    joined_items = []
    for item in paths_data["items"]:
        tweet_id = int(item["tweet_id"])
        joined_items.append({**item, "classification": labels_by_id[tweet_id]})

    lines: list[str] = []
    lines.append("# Reasoner Grouped Endorsement Report")
    lines.append("")
    lines.append(f"Paths source: `{PATHS_JSON.name}`")
    lines.append(f"Labels source: `{LABELS_JSON.name}`")
    lines.append("")
    lines.append(
        f"Window: `{paths_data['window']['start_inclusive']}` to `{paths_data['window']['end_exclusive']}`"
    )
    lines.append(f"Model: `{labels_data['model']}`")
    lines.append("")

    for stance in STANCE_ORDER:
        stance_items = [item for item in joined_items if item["classification"].get("stance") == stance]
        lines.append(f"## {stance} ({len(stance_items)})")
        lines.append("")

        for vagueness in VAGUENESS_ORDER:
            bucket = [
                item
                for item in stance_items
                if item["classification"].get("endorsement_vagueness") == vagueness
            ]
            bucket.sort(key=sort_key)
            lines.append(f"### {vagueness} ({len(bucket)})")
            lines.append("")
            for index, item in enumerate(bucket, start=1):
                render_item(lines, index, item, stance)

    no_stance_items = [item for item in joined_items if not item["classification"].get("has_stance")]
    no_stance_items.sort(
        key=lambda item: (
            CONFIDENCE_RANK.get(item["classification"].get("confidence"), 9),
            item["created_at"],
            item["tweet_id"],
        )
    )
    lines.append(f"## no_stance ({len(no_stance_items)})")
    lines.append("")
    for index, item in enumerate(no_stance_items, start=1):
        render_item(lines, index, item, "no_stance")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote grouped markdown to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
