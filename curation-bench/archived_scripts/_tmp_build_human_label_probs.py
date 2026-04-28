#!/usr/bin/env python3
"""Derive soft endorsement labels (3-way probabilities) from discrete UI labels.

Reads predictions_top5_jul2024.json, drops wrong_target rows, and attaches
prob_endorse / prob_disendorse / prob_neutral using the user's mapping.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def _label_to_probs(label: str, confidence_pct: int) -> tuple[dict[str, float], bool]:
    """Return probs and whether clip+renormalize was applied."""
    x = confidence_pct / 100.0
    if label == "neutral":
        endorse = (1.0 - x) * 0.9
        disendorse = (1.0 - x) * 0.1
        neutral = x
    elif label == "endorse":
        endorse = x
        disendorse = 0.1 * x
        neutral = 1.0 - x - 0.1 * x
    elif label == "disendorse":
        endorse = 0.2 * x
        neutral = 1.0 - 0.2 * x - x
        disendorse = x
    else:
        raise ValueError(f"unexpected label for prob mapping: {label!r}")

    probs = {"endorse": endorse, "disendorse": disendorse, "neutral": neutral}
    clipped = min(probs.values()) < -1e-12
    if clipped:
        for k in probs:
            probs[k] = max(0.0, probs[k])
        total = sum(probs.values())
        if total <= 0.0:
            raise ValueError(f"degenerate distribution after clip: {label=} {confidence_pct=}")
        probs = {k: v / total for k, v in probs.items()}
    return probs, clipped


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "data" / "labels" / "predictions_top5_jul2024.json"
    dst = root / "data" / "labels" / "predictions_top5_jul2024_human_probs.json"

    with src.open(encoding="utf-8") as f:
        data = json.load(f)

    out = deepcopy(data)
    out.pop("label_options", None)
    out.pop("label_display", None)
    src_ver = data.get("version")
    out["human_probs_schema_version"] = 1
    out["source_predictions_file"] = str(src.name)
    out["source_predictions_version"] = src_ver
    out.pop("version", None)
    out["task"] = (data.get("task") or "") + "_human_probs"
    out["probability_mapping"] = {
        "neutral_confidence_X": {
            "endorse": "(1-X)*0.9",
            "disendorse": "(1-X)*0.1",
            "neutral": "X",
        },
        "endorse_confidence_X": {
            "endorse": "X",
            "disendorse": "0.1*X",
            "neutral": "1 - X - 0.1*X",
        },
        "disendorse_confidence_X": {
            "endorse": "0.2*X",
            "disendorse": "X",
            "neutral": "1 - 0.2*X - X",
        },
        "note": (
            "confidence_correct_pct is treated as X in [0,1]. If the closed-form "
            "neutral mass is negative, values are clipped to 0 and renormalized."
        ),
    }

    excluded = 0
    renormalized = 0

    users_out: dict = {}
    for uname, udata in data.get("users", {}).items():
        kept: list = []
        for rec in udata.get("labels", []):
            if rec.get("label") == "wrong_target":
                excluded += 1
                continue
            label = rec["label"]
            pct = int(rec["confidence_correct_pct"])
            probs, did_clip = _label_to_probs(label, pct)
            if did_clip:
                renormalized += 1
            new_rec = dict(rec)
            new_rec["prob_endorse"] = round(probs["endorse"], 10)
            new_rec["prob_disendorse"] = round(probs["disendorse"], 10)
            new_rec["prob_neutral"] = round(probs["neutral"], 10)
            kept.append(new_rec)
        users_out[uname] = {"label_count": len(kept), "labels": kept}

    out["users"] = users_out
    out["excluded_wrong_target_count"] = excluded
    out["renormalized_after_negative_neutral_count"] = renormalized
    for k in (
        "output_path",
        "confidence_prior_note",
        "confidence_options",
        "last_updated_at",
    ):
        out.pop(k, None)
    out["written_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with dst.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {dst}")
    print(f"Excluded wrong_target: {excluded}")
    print(f"Clip+renormalized (high confidence vs. formula): {renormalized}")


if __name__ == "__main__":
    main()
