#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "streamlit>=1.36",
#   "scikit-learn>=1.3",
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///
"""Streamlit UI to manually label endorsement-target predictions
for the top-5 focal users + their effective neighborhood (Jul 2024 cohort).

For each focal user the app collects:
  - the focal user's own model-extracted targets
    (items in the ground-truth file whose path_anchor_username == focal user)
  - the targets attributed to neighbors of that focal user
    (items in the neighbors file whose path_anchor_username is in the
     focal user's effective_neighbor_graph entry)

The two pools are deduplicated, sorted by representative_tweet_id, and
shuffled with a seed tied to the focal username so the order is
deterministic across reloads. In normal mode the UI hides everything that would leak ground truth or
source: no direction, no path_anchor_username, no indication of focal vs
neighbor, no tweet text, no tweet id, no link. **Review mode** (sidebar
checkbox; on by default) shows `clean_labels` ground truth plus **every** Cursor agent
prediction run folder under `data/labels/cursor_agent_predictions/`
(auto-discovered subfolders), as a compact emoji table plus reasoning — read-only — and
disables labeling controls.

**Review mode** also adds an **Aggregate metrics** tab with accuracy, Brier
skill vs marginal baseline, **weighted AUROC** (one-vs-rest average weighted
by GT class frequency; same family as the ROC figures), weighted and macro F1,
per-class F1, multiclass Brier,
marginal baseline Brier, one-vs-rest **ROC** curves (human + models),
a single **precision-at-20%-endorsing-prior vs recall** curve
(``R = (TPR · 20) / (TPR · 20 + FPR · 80)`` — endorsing vs rest only, human +
models, each curve scored on **only** the rows that predictor can score — i.e.
its own support — and the support shown in each legend entry), and confusion
matrices (vs `clean_labels` GT). Human probability metrics (Brier / skill / ROC /
prior-precision) use **only** soft probabilities (**prob_endorse** / **prob_disendorse** /
**prob_neutral**) from ``predictions_top5_jul2024_human_probs.json``; rows without
that triple are excluded from the human track (no one-hot fallback).

The labeler sees `target_entity`, `longer_name`, and `url` when the URL
is not an x.com link (x.com URLs are hidden in the UI). `context` is not
shown and is not written to the labels file.

Labels are one of:
  - Endorse                   -> "endorse"
  - Disendorse                -> "disendorse"
  - Neutral                   -> "neutral"
  - Wrong endorsement target  -> "wrong_target"

Each labeled row also stores subjective confidence that the chosen label
is correct (`confidence_correct_pct`: one of 20, 30, 40, 50, 60, 70, 80,
90, 95, 99). The UI reminds you that most items are neutral (~2/10 non-neutral
prior) when calibrating.

All labels for all 5 focal users are persisted into a single shared
JSON file at:
  curation-bench/data/labels/predictions_top5_jul2024.json

Saves happen asynchronously on a background daemon thread so each
button click returns immediately. Saved records omit model ground-truth
leaks (direction, focal vs neighbor source, path anchor username) and
omit `context` so the labels file can be inspected without spoiling the task.

Run:
  uv run --with 'streamlit>=1.36' streamlit run \
      curation-bench/06_label_endorsement_targets_ui.py
"""

from __future__ import annotations

import json
import queue
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Any, Callable
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

FOCAL_USERS: list[str] = [
    "goblinodds",
    "daniellefong",
    "exgenesis",
    "archived_videos",
    "danielbrottman",
]

LABEL_OPTIONS: list[tuple[str, str]] = [
    ("endorse", "Endorse"),
    ("disendorse", "Disendorse"),
    ("neutral", "Neutral"),
    ("wrong_target", "Wrong endorsement target"),
]
LABEL_VALUES = [v for v, _ in LABEL_OPTIONS]
LABEL_DISPLAY = dict(LABEL_OPTIONS)

# Subjective P(label is correct); integers persisted as `confidence_correct_pct`.
CONFIDENCE_OPTIONS: list[int] = [20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
CONFIDENCE_OPTION_SET: frozenset[int] = frozenset(CONFIDENCE_OPTIONS)
DEFAULT_CONFIDENCE_PCT: int = 50
CONFIDENCE_PRIOR_NOTE: str = (
    "Calibration hint: only about 2 in 10 items are non-neutral — use that "
    "base rate when picking how confident you are."
)

_SCRIPT_DATA = Path(__file__).parent / "data"
# Bundle root contains clean_labels/, neighbors/, paths_and_targets/, and the graph JSON.
_DATA_BUNDLE_CANDIDATES: tuple[Path, ...] = (
    _SCRIPT_DATA / "paths_and_targets",
    _SCRIPT_DATA / "curation_bench_clean_data",
)


def _default_data_bundle() -> Path:
    for p in _DATA_BUNDLE_CANDIDATES:
        if p.is_dir():
            return p
    return _DATA_BUNDLE_CANDIDATES[0]


DATA_DIR = _default_data_bundle()

_FOCAL_TARGETS_NAME = "20240701_20240801_top5_endorsement_targets.json"


def _default_focal_endorsement_targets_path(bundle: Path) -> Path:
    """Focal pool file lives under paths_and_targets/; legacy layout used ground_truth_labels/."""
    nested = bundle / "paths_and_targets" / _FOCAL_TARGETS_NAME
    legacy = bundle / "ground_truth_labels" / _FOCAL_TARGETS_NAME
    if nested.is_file():
        return nested
    if legacy.is_file():
        return legacy
    return nested


DEFAULT_GROUND_TRUTH = _default_focal_endorsement_targets_path(DATA_DIR)
DEFAULT_NEIGHBORS = (
    DATA_DIR
    / "neighbors"
    / "20240701_20240801_top5_neighbors_endorsement_targets.json"
)
DEFAULT_NEIGHBOR_GRAPH = (
    DATA_DIR / "20240701_20240801_top5_neighbors_effective_neighbor_graph.json"
)
DEFAULT_CLEAN_LABELS_DIR = DATA_DIR / "clean_labels"

# Cursor agent prediction runs (under data/labels/cursor_agent_predictions/<run>/).
CURSOR_PREDICTIONS_DIR = (
    Path(__file__).parent / "data" / "labels" / "cursor_agent_predictions"
)

# Argmax labels → one emoji per column in review mode (endorsing / neutral / disendorsing).
EMOJI_ENDORSING = "👍"
EMOJI_NEUTRAL = "😐"
EMOJI_DISENDORING = "👎"
EMOJI_HUMAN_WRONG_TARGET = "⚠️"
EMOJI_MISSING = "—"

# 3-way eval (must match clean_labels ground_truth_label strings).
GT_CLASSES: tuple[str, ...] = ("endorsing", "disendorsing", "neutral")

# Composite eval key. The same tweet can carry multiple endorsement targets, each with
# its own GT label and per-model probability — every source file
# (`clean_labels/ground_truth_*.json`, `cursor_agent_predictions/<run>/predictions_*.json`,
# `predictions_top5_jul2024_human_probs.json`) treats them as distinct rows. Keying by
# `representative_tweet_id` alone silently collapses them and risks pairing a row's GT
# with another target's prediction. We key all eval lookups by `(tid, target_entity)`.
EvalKey = tuple[int, str]


def _eval_key(rec: dict[str, Any]) -> EvalKey | None:
    """Composite key from any clean / pred / human record; ``None`` if either field is missing."""
    tid = rec.get("representative_tweet_id")
    tgt = rec.get("target_entity")
    if tid is None or tgt is None:
        return None
    return (int(tid), str(tgt))

# ROC stacked figure (3 OvR panels): each panel uses ``set_aspect("equal")`` so the axes box
# is square; ``figsize`` width = axes box (height-bound at ~H/3) + horizontal room for the
# right-side legend. ``dpi`` fixes native raster density so Streamlit scaling stays sharp.
ROC_OVR_FIGSIZE_W_IN: float = 6.5
ROC_OVR_FIGSIZE_H_IN: float = 8.4
ROC_OVR_FIGURE_DPI: float = 100.0
ROC_OVR_LINEWIDTH: float = 1.2
ROC_OVR_DIAG_LINEWIDTH: float = 0.7
ROC_OVR_TITLE_FONTSIZE: float = 9.5
ROC_OVR_AXIS_LABEL_FONTSIZE: float = 8.5
ROC_OVR_TICK_FONTSIZE: float = 7.5
ROC_OVR_LEGEND_FONTSIZE: float = 7.0
ROC_OVR_GRID_LINEWIDTH: float = 0.5
ROC_OVR_GRID_LINESTYLE: str = ":"
ROC_OVR_GRID_COLOR: str = "0.7"
ROC_OVR_GRID_ALPHA: float = 0.6


def list_cursor_prediction_run_folders(cursor_root: str) -> list[str]:
    """All direct subfolders under cursor_agent_predictions/ (excluding dot-prefixed names).

    Not cached so newly added runs appear without clearing Streamlit cache.
    """
    root = Path(cursor_root)
    if not root.is_dir():
        return []
    runs = [
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ]
    return sorted(runs, reverse=True)


# e.g. claude-…-thinking_20260427T161505 → drop the _YYYYMMDDTHHMMSS suffix for display.
_RUN_FOLDER_TS_SUFFIX = re.compile(r"_\d{8}T\d{6}$")


def _abbrev_run_folder(name: str, max_len: int = 40) -> str:
    """Short label for table headers / expanders: no long timestamp, optional ellipsis."""
    base = _RUN_FOLDER_TS_SUFFIX.sub("", name).strip()
    if not base:
        base = name
    if len(base) <= max_len:
        return base
    return base[: max_len - 1] + "…"


def _unique_abbrev_labels_for_runs(prediction_runs: list[str]) -> list[str]:
    """Display names for run folders: after timestamp strip, suffix -1, -2, … when bases collide."""
    abbrevs = [_abbrev_run_folder(r) for r in prediction_runs]
    dup_count: dict[str, int] = {}
    for a in abbrevs:
        dup_count[a] = dup_count.get(a, 0) + 1
    index_in_base: dict[str, int] = {}
    out: list[str] = []
    for a in abbrevs:
        if dup_count[a] == 1:
            out.append(a)
        else:
            index_in_base[a] = index_in_base.get(a, 0) + 1
            out.append(f"{a}-{index_in_base[a]}")
    return out


# Snapshot of `agent --model` registry as of 2026-05-11; used to split a run-folder name
# into (base model, prompt/calib variant suffix). Refresh from `agent --model` if a new
# model isn't recognised — unrecognised folders fall back to "model = whole name,
# variant = (none)" so the chart still renders, they just won't share a color with peers.
_KNOWN_AGENT_MODEL_NAMES: frozenset[str] = frozenset({
    "auto",
    "composer-2-fast", "composer-2",
    "gpt-5.3-codex-low", "gpt-5.3-codex-low-fast",
    "gpt-5.3-codex", "gpt-5.3-codex-fast",
    "gpt-5.3-codex-high", "gpt-5.3-codex-high-fast",
    "gpt-5.3-codex-xhigh", "gpt-5.3-codex-xhigh-fast",
    "gpt-5.2",
    "gpt-5.3-codex-spark-preview-low",
    "gpt-5.3-codex-spark-preview",
    "gpt-5.3-codex-spark-preview-high",
    "gpt-5.3-codex-spark-preview-xhigh",
    "gpt-5.2-codex-low", "gpt-5.2-codex-low-fast",
    "gpt-5.2-codex", "gpt-5.2-codex-fast",
    "gpt-5.2-codex-high", "gpt-5.2-codex-high-fast",
    "gpt-5.2-codex-xhigh", "gpt-5.2-codex-xhigh-fast",
    "gpt-5.1-codex-max-low", "gpt-5.1-codex-max-low-fast",
    "gpt-5.1-codex-max-medium", "gpt-5.1-codex-max-medium-fast",
    "gpt-5.1-codex-max-high", "gpt-5.1-codex-max-high-fast",
    "gpt-5.1-codex-max-xhigh", "gpt-5.1-codex-max-xhigh-fast",
    "gpt-5.5-high", "gpt-5.5-high-fast",
    "claude-opus-4-7-thinking-high",
    "gpt-5.4-high", "gpt-5.4-high-fast",
    "claude-4.6-opus-high-thinking", "claude-4.6-opus-high-thinking-fast",
    "gpt-5.5-none", "gpt-5.5-none-fast",
    "gpt-5.5-low", "gpt-5.5-low-fast",
    "gpt-5.5-medium", "gpt-5.5-medium-fast",
    "gpt-5.5-extra-high", "gpt-5.5-extra-high-fast",
    "claude-4.6-sonnet-medium", "claude-4.6-sonnet-medium-thinking",
    "claude-opus-4-7-low", "claude-opus-4-7-medium",
    "claude-opus-4-7-high", "claude-opus-4-7-xhigh", "claude-opus-4-7-max",
    "claude-opus-4-7-thinking-low", "claude-opus-4-7-thinking-medium",
    "claude-opus-4-7-thinking-xhigh", "claude-opus-4-7-thinking-max",
    "grok-4.3",
    "gpt-5.4-low", "gpt-5.4-medium", "gpt-5.4-medium-fast",
    "gpt-5.4-xhigh", "gpt-5.4-xhigh-fast",
    "claude-4.6-opus-high", "claude-4.6-opus-max",
    "claude-4.6-opus-max-thinking", "claude-4.6-opus-max-thinking-fast",
    "claude-4.5-opus-high", "claude-4.5-opus-high-thinking",
    "gpt-5.2-low", "gpt-5.2-low-fast", "gpt-5.2-fast",
    "gpt-5.2-high", "gpt-5.2-high-fast",
    "gpt-5.2-xhigh", "gpt-5.2-xhigh-fast",
    "gemini-3.1-pro",
    "gpt-5.4-mini-none", "gpt-5.4-mini-low", "gpt-5.4-mini-medium",
    "gpt-5.4-mini-high", "gpt-5.4-mini-xhigh",
    "gpt-5.4-nano-none", "gpt-5.4-nano-low", "gpt-5.4-nano-medium",
    "gpt-5.4-nano-high", "gpt-5.4-nano-xhigh",
    "claude-4.5-sonnet", "claude-4.5-sonnet-thinking",
    "gpt-5.1-low", "gpt-5.1", "gpt-5.1-high",
    "gemini-3-flash",
    "gpt-5.1-codex-mini-low", "gpt-5.1-codex-mini", "gpt-5.1-codex-mini-high",
    "claude-4-sonnet", "claude-4-sonnet-thinking",
    "gpt-5-mini",
    "kimi-k2.5",
})


def _split_run_into_model_and_variant(run: str) -> tuple[str, str]:
    """Return ``(model_family, variant_suffix)`` for a run folder name.

    Strips the trailing ``_YYYYMMDDTHHMMSS`` timestamp, then finds the longest
    hyphen-token prefix that exists in ``_KNOWN_AGENT_MODEL_NAMES``. Everything
    before that boundary is the model; everything after is the variant suffix
    (e.g. ``calib-prompt``, ``calib-v2``, or ``""`` for a base run).

    If no prefix matches the registry, returns ``(abbrev_name, "")`` so the
    caller still gets a usable family key — the run will just sit alone in its
    own color slot.
    """
    base = _RUN_FOLDER_TS_SUFFIX.sub("", run).strip()
    if not base:
        return run, ""
    tokens = base.split("-")
    for k in range(len(tokens), 0, -1):
        prefix = "-".join(tokens[:k])
        if prefix in _KNOWN_AGENT_MODEL_NAMES:
            return prefix, "-".join(tokens[k:])
    return base, ""


# Cycle of distinguishable line styles for variants within one model family.
# Beyond 8 variants the cycle repeats; same-color same-style collisions only happen
# at 9+ variants per model — log that off to the user via a Streamlit caption if it
# ever happens (see render functions). Tuples are matplotlib dash patterns.
_VARIANT_LINE_STYLES: tuple[Any, ...] = (
    "-",                       # solid
    "--",                      # dashed
    ":",                       # dotted
    "-.",                      # dashdot
    (0, (5, 1)),               # densely dashed
    (0, (1, 1)),               # densely dotted
    (0, (3, 1, 1, 1)),         # densely dashdotted
    (0, (3, 5, 1, 5)),         # dashdotted (loose)
)


def _assign_run_styles(
    prediction_runs: list[str],
) -> tuple[dict[str, tuple[Any, Any]], list[tuple[str, list[str]]]]:
    """Map each run to ``(color, linestyle)``; same model family ⇒ same color.

    Returns ``(style_map, families)`` where ``families`` is the ordered list of
    ``(model_family, [run_names sorted by variant])`` so callers can write a
    short legend hint. Color cycles ``tab10`` in family-first-appearance order;
    line style cycles ``_VARIANT_LINE_STYLES`` per variant within a family.
    """
    cmap = plt.get_cmap("tab10")
    runs_by_family: dict[str, list[str]] = {}
    family_order: list[str] = []
    for run in prediction_runs:
        family, _variant = _split_run_into_model_and_variant(run)
        if family not in runs_by_family:
            runs_by_family[family] = []
            family_order.append(family)
        runs_by_family[family].append(run)

    style_map: dict[str, tuple[Any, Any]] = {}
    families_ordered: list[tuple[str, list[str]]] = []
    for color_idx, family in enumerate(family_order):
        color = cmap(color_idx % 10)
        runs_in_family = sorted(
            runs_by_family[family],
            key=lambda r: _split_run_into_model_and_variant(r)[1],
        )
        for variant_idx, run in enumerate(runs_in_family):
            ls = _VARIANT_LINE_STYLES[variant_idx % len(_VARIANT_LINE_STYLES)]
            style_map[run] = (color, ls)
        families_ordered.append((family, runs_in_family))
    return style_map, families_ordered


def _truncate_display(s: str, max_len: int = 64) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _emoji_for_agent_prediction(pred: dict[str, Any] | None) -> str:
    """Argmax over p_endorsing / p_neutral / p_disendorsing."""
    if pred is None:
        return EMOJI_MISSING
    label = _agent_prediction_argmax_label(pred)
    return {
        "endorsing": EMOJI_ENDORSING,
        "neutral": EMOJI_NEUTRAL,
        "disendorsing": EMOJI_DISENDORING,
    }.get(label, EMOJI_NEUTRAL)


def _emoji_for_ground_truth_label(gt: str | None) -> str:
    if not gt:
        return EMOJI_MISSING
    gt_l = gt.strip().lower()
    if gt_l == "endorsing":
        return EMOJI_ENDORSING
    if gt_l == "disendorsing":
        return EMOJI_DISENDORING
    if gt_l == "neutral":
        return EMOJI_NEUTRAL
    return EMOJI_MISSING


def _emoji_for_human_label(label: str | None) -> str:
    """Human taxonomy: endorse/disendorse/neutral/wrong_target."""
    if not label:
        return EMOJI_MISSING
    low = label.strip().lower()
    return {
        "endorse": EMOJI_ENDORSING,
        "neutral": EMOJI_NEUTRAL,
        "disendorse": EMOJI_DISENDORING,
        "wrong_target": EMOJI_HUMAN_WRONG_TARGET,
    }.get(low, label)


def _review_comparison_md_row(
    *,
    clean_rec: dict[str, Any] | None,
    human_label: str | None,
    prediction_runs: list[str],
    run_display_labels: list[str],
    pred_by_run: dict[str, dict[EvalKey, dict[str, Any]]],
    eval_key: EvalKey,
) -> None:
    """Single-row GitHub-flavored markdown table: GT · Human · each model.

    Columns are omitted when the value would be absent (—): no GT row, no saved
    human label, or no model prediction for this ``(tid, target_entity)``/run.
    """
    hdrs: list[str] = []
    cells: list[str] = []

    gt_raw = (
        clean_rec.get("ground_truth_label") if clean_rec else None
    )
    gt_cell = _emoji_for_ground_truth_label(gt_raw)
    if gt_cell != EMOJI_MISSING:
        hdrs.append("GT")
        cells.append(gt_cell)

    hum_cell = _emoji_for_human_label(human_label)
    if hum_cell != EMOJI_MISSING:
        hdrs.append("Human")
        cells.append(hum_cell)

    for run, disp in zip(prediction_runs, run_display_labels, strict=True):
        pr = pred_by_run.get(run, {}).get(eval_key)
        if pr is None:
            continue
        hdrs.append(disp)
        cells.append(_emoji_for_agent_prediction(pr))

    if not hdrs:
        st.caption(
            "_No ground truth row, saved human label, or model predictions for this item._"
        )
        return

    header_line = "| " + " | ".join(hdrs) + " |"
    separator_line = "| " + " | ".join([":---:"] * len(hdrs)) + " |"
    data_line = "| " + " | ".join(cells) + " |"
    st.markdown(header_line + "\n" + separator_line + "\n" + data_line)


DEFAULT_OUTPUT = (
    Path(__file__).parent
    / "data"
    / "labels"
    / "predictions_top5_jul2024.json"
)
DEFAULT_HUMAN_PROBS_JSON = (
    Path(__file__).parent / "data" / "labels" / "predictions_top5_jul2024_human_probs.json"
)

# Omitted from persisted labels (ground-truth leaks + fields kept off disk).
_LABEL_RECORD_DROP_KEYS: frozenset[str] = frozenset(
    ("_actual_direction", "_actual_source", "_path_anchor_username", "context")
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _sanitize_label_record(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in _LABEL_RECORD_DROP_KEYS}


def _is_x_com_url(url: str) -> bool:
    """True if the URL targets x.com (scheme optional), including subdomains."""
    raw = (url or "").strip()
    if not raw:
        return False
    low = raw.lower()
    if low.startswith("x.com"):
        return True
    candidate = low if "://" in low else f"https://{low}"
    try:
        host = (urlparse(candidate).hostname or "").lower()
    except ValueError:
        return False
    return host == "x.com" or host.endswith(".x.com")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_clean_label_allowed_ids(clean_labels_dir: str) -> dict[str, set[int]]:
    """Return the representative_tweet_ids retained for each focal user's clean-label set."""
    base_dir = Path(clean_labels_dir)
    allowed_ids_by_user: dict[str, set[int]] = {user: set() for user in FOCAL_USERS}
    for user in FOCAL_USERS:
        path = base_dir / f"ground_truth_{user}.json"
        if not path.exists():
            continue
        data = load_json(path)
        allowed_ids_by_user[user] = {
            int(item["representative_tweet_id"])
            for item in (data.get("items") or [])
            if item.get("representative_tweet_id") is not None
        }
    return allowed_ids_by_user


@st.cache_data(show_spinner=False)
def load_clean_label_items_by_key(
    clean_labels_dir: str, focal_user: str
) -> dict[EvalKey, dict[str, Any]]:
    """Map (representative_tweet_id, target_entity) -> full row from ground_truth_{focal}.json.

    Multi-target tweets occupy multiple keys, one per target.
    """
    path = Path(clean_labels_dir) / f"ground_truth_{focal_user}.json"
    if not path.exists():
        return {}
    data = load_json(path)
    out: dict[EvalKey, dict[str, Any]] = {}
    for item in data.get("items") or []:
        key = _eval_key(item)
        if key is not None:
            out[key] = item
    return out


@st.cache_data(show_spinner=False)
def load_agent_predictions_by_key(
    predictions_json_path: str,
) -> dict[EvalKey, dict[str, Any]]:
    """Map (representative_tweet_id, target_entity) -> prediction dict (reasoning, p_*).

    Agent runs predict per (tid, target), so multi-target tweets get one entry per target.
    """
    path = Path(predictions_json_path)
    if not path.exists():
        return {}
    data = load_json(path)
    out: dict[EvalKey, dict[str, Any]] = {}
    for pred in data.get("predictions") or []:
        key = _eval_key(pred)
        if key is not None:
            out[key] = pred
    return out


@st.cache_data(show_spinner=False)
def load_human_probs_by_focal_user(
    json_path: str,
) -> dict[str, dict[EvalKey, dict[str, Any]]]:
    """Map focal -> (representative_tweet_id, target_entity) -> label record (incl. prob_*)."""
    path = Path(json_path)
    if not path.is_file():
        return {}
    data = load_json(path)
    out: dict[str, dict[EvalKey, dict[str, Any]]] = {}
    for user, block in (data.get("users") or {}).items():
        m: dict[EvalKey, dict[str, Any]] = {}
        for rec in block.get("labels") or []:
            key = _eval_key(rec)
            if key is not None:
                m[key] = rec
        out[user] = m
    return out


def _agent_prediction_argmax_label(pred: dict[str, Any]) -> str:
    scores: list[tuple[str, float]] = [
        ("neutral", float(pred.get("p_neutral") or 0.0)),
        ("endorsing", float(pred.get("p_endorsing") or 0.0)),
        ("disendorsing", float(pred.get("p_disendorsing") or 0.0)),
    ]
    return max(scores, key=lambda x: x[1])[0]


def _format_agent_probs(pred: dict[str, Any]) -> str:
    pn = float(pred.get("p_neutral") or 0.0)
    pe = float(pred.get("p_endorsing") or 0.0)
    pd = float(pred.get("p_disendorsing") or 0.0)
    return (
        f"p_endorsing={pe:.3f}, p_disendorsing={pd:.3f}, p_neutral={pn:.3f}"
    )


def _normalize_gt_label(val: Any) -> str | None:
    """Map clean_labels ground_truth_label to endorsing/disendorsing/neutral."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in GT_CLASSES:
        return s
    return None


def human_label_to_gt_class(label: str | None) -> str | None:
    """Map persisted human label → same 3-way as GT; None = excluded from classification eval."""
    if not label:
        return None
    low = label.strip().lower()
    if low == "wrong_target":
        return None
    m = {"endorse": "endorsing", "disendorse": "disendorsing", "neutral": "neutral"}
    return m.get(low)


def _pred_prob_vec_ordered(pred: dict[str, Any]) -> list[float]:
    """Order aligned with GT_CLASSES: endorsing, disendorsing, neutral."""
    return [
        float(pred.get("p_endorsing") or 0.0),
        float(pred.get("p_disendorsing") or 0.0),
        float(pred.get("p_neutral") or 0.0),
    ]


def _human_soft_prob_vec_ordered(rec: dict[str, Any]) -> list[float] | None:
    """Human 3-way softmax from `predictions_top5_jul2024_human_probs` (endorse/disendorse/neutral)."""
    pe = rec.get("prob_endorse")
    pd_ = rec.get("prob_disendorse")
    pn = rec.get("prob_neutral")
    if pe is None or pd_ is None or pn is None:
        return None
    return [float(pe), float(pd_), float(pn)]


def _human_gt_pred_probs_soft_only(
    rows: list[tuple[str, int, str, str]],
    human_probs_by_focal: dict[str, dict[EvalKey, dict[str, Any]]] | None,
    focal_filter: str | None = None,
) -> tuple[list[str], list[str], list[list[float]]]:
    """Ground truth, human discrete class (from human_probs `label`), and soft prob vector.

    Only includes evaluation rows where ``predictions_top5_jul2024_human_probs.json``
    has a record with all three ``prob_*`` fields for that focal and ``(tid, target_entity)``
    pair. Multi-target tweets are matched per target.
    """
    if not human_probs_by_focal:
        return [], [], []
    yt: list[str] = []
    yp: list[str] = []
    pr: list[list[float]] = []
    for focal, tid, target, gt in rows:
        if focal_filter is not None and focal != focal_filter:
            continue
        hp = human_probs_by_focal.get(focal, {}).get((int(tid), target))
        if hp is None:
            continue
        pv = _human_soft_prob_vec_ordered(hp)
        if pv is None:
            continue
        hc = human_label_to_gt_class(hp.get("label"))
        if hc is None:
            continue
        yt.append(gt)
        yp.append(hc)
        pr.append(pv)
    return yt, yp, pr


def _one_hot(gt_class: str) -> tuple[float, float, float]:
    i = GT_CLASSES.index(gt_class)
    return tuple(1.0 if k == i else 0.0 for k in range(3))


def multiclass_mean_brier(
    y_true_class: list[str], prob_vecs: list[list[float]]
) -> float:
    """Mean over rows of (1/3) * sum_k (p_k − one_hot[y])²."""
    n = len(y_true_class)
    if n == 0:
        return float("nan")
    total = 0.0
    for yt, pv in zip(y_true_class, prob_vecs, strict=False):
        oh = _one_hot(yt)
        total += sum((pv[k] - oh[k]) ** 2 for k in range(3)) / 3.0
    return total / n


def marginal_mean_brier(y_true_class: list[str]) -> tuple[float, list[float]]:
    """Const prediction = empirical GT class frequencies on this slice."""
    n = len(y_true_class)
    if n == 0:
        return float("nan"), []
    freq = {c: 0 for c in GT_CLASSES}
    for yt in y_true_class:
        if yt in freq:
            freq[yt] += 1
    bar = [freq[c] / n for c in GT_CLASSES]
    total = 0.0
    for yt in y_true_class:
        oh = _one_hot(yt)
        total += sum((bar[k] - oh[k]) ** 2 for k in range(3)) / 3.0
    return total / n, bar


def brier_skill_score(brier_model: float, brier_marginal: float) -> float | None:
    """1 − Brier_model / Brier_marginal. None if marginal≈0 or invalid."""
    if (
        brier_marginal is None
        or brier_model is None
        or brier_marginal <= 1e-15
        or brier_model != brier_model
        or brier_marginal != brier_marginal
    ):
        return None
    return 1.0 - brier_model / brier_marginal


def _metrics_from_predictions(
    y_true: list[str], y_pred: list[str], prob_vecs: list[list[float]]
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accuracy": float("nan"),
        "weighted_f1": float("nan"),
        "macro_f1": float("nan"),
        "f1_endorsing": float("nan"),
        "f1_disendorsing": float("nan"),
        "f1_neutral": float("nan"),
        "brier": float("nan"),
        "brier_marginal": float("nan"),
        "brier_skill": None,
        "weighted_auroc": float("nan"),
        "support": len(y_true),
    }
    n = len(y_true)
    if n == 0:
        return out
    lbls = list(GT_CLASSES)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", labels=lbls, zero_division=0)
    )
    out["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", labels=lbls, zero_division=0)
    )
    f1_per = f1_score(y_true, y_pred, average=None, labels=lbls, zero_division=0)
    for i, c in enumerate(GT_CLASSES):
        out[f"f1_{c}"] = float(f1_per[i])
    bri = multiclass_mean_brier(y_true, prob_vecs)
    bmar, _ = marginal_mean_brier(y_true)
    out["brier"] = bri
    out["brier_marginal"] = bmar
    out["brier_skill"] = brier_skill_score(bri, bmar)
    y_score_mx = np.asarray(prob_vecs, dtype=np.float64)
    # sklearn roc_auc_score(multi_class='ovr') requires `labels` in lex order; remap columns.
    lbls_ovr = sorted(lbls)
    if y_score_mx.shape == (n, len(lbls)):
        idx_perm = [lbls.index(c) for c in lbls_ovr]
        y_ovr = y_score_mx[:, idx_perm]
        try:
            out["weighted_auroc"] = float(
                roc_auc_score(
                    y_true,
                    y_ovr,
                    multi_class="ovr",
                    average="weighted",
                    labels=lbls_ovr,
                )
            )
        except ValueError:
            out["weighted_auroc"] = float("nan")
    return out


def _confusion_df(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(GT_CLASSES))
    return pd.DataFrame(cm, index=[f"GT: {c}" for c in GT_CLASSES], columns=list(GT_CLASSES))


# Summary metrics table: bold best value per column (within Predictor rows).
_SUMMARY_METRICS_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "weighted_auroc",
        "f1_endorsing",
        "f1_disendorsing",
        "f1_neutral",
        "brier_skill",
    }
)
_SUMMARY_METRICS_LOWER_IS_BETTER: frozenset[str] = frozenset({"brier", "brier_marginal"})


def _col_best_boolean_mask(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """True where numeric cell ties column-best (multiple True on ties)."""
    num = pd.to_numeric(series, errors="coerce")
    valid = num.notna()
    if not valid.any():
        return pd.Series(False, index=series.index)
    cmp_vals = num[valid]
    bound = cmp_vals.max() if higher_is_better else cmp_vals.min()
    out = pd.Series(False, index=series.index)
    out.loc[valid] = np.isclose(num[valid], bound, rtol=0.0, atol=1e-9)
    return out


def _summary_metrics_bold_mask(dfm_metrics: pd.DataFrame) -> pd.DataFrame:
    """Same columns/index as dfm_metrics; True = bold cell (Predictor excluded)."""
    mask = pd.DataFrame(False, index=dfm_metrics.index, columns=dfm_metrics.columns)
    for col in dfm_metrics.columns:
        if col == "model":
            continue
        if col == "support":
            continue
        if col in _SUMMARY_METRICS_HIGHER_IS_BETTER:
            mask[col] = _col_best_boolean_mask(dfm_metrics[col], higher_is_better=True)
        elif col in _SUMMARY_METRICS_LOWER_IS_BETTER:
            mask[col] = _col_best_boolean_mask(dfm_metrics[col], higher_is_better=False)
    return mask


def _styler_css_bold(mask: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return Styler.apply(axis=None) callable; bold cells where mask is True."""

    def _css(_underlying: pd.DataFrame) -> pd.DataFrame:
        m = mask.reindex(
            index=_underlying.index, columns=_underlying.columns, fill_value=False
        )
        css = pd.DataFrame("", index=_underlying.index, columns=_underlying.columns)
        css.values[:] = np.where(m.values.astype(bool), "font-weight: bold", "")
        return css

    return _css


def _brier_skill_float_from_metrics(m: dict[str, Any] | None) -> float:
    """Finite skill value or NaN."""
    if m is None:
        return float("nan")
    sk = m.get("brier_skill")
    if sk is None:
        return float("nan")
    try:
        x = float(sk)
    except (TypeError, ValueError):
        return float("nan")
    if x != x:
        return float("nan")
    return x


def _skill_grid_row_best_mask(df_skill_num: pd.DataFrame) -> pd.DataFrame:
    """Bold highest Brier skill per row (same focal; compare Human vs agents)."""
    mask = pd.DataFrame(False, index=df_skill_num.index, columns=df_skill_num.columns)
    for r in df_skill_num.index:
        row = df_skill_num.loc[r]
        valid = row.dropna()
        if len(valid) == 0:
            continue
        best = float(valid.max())
        mask.loc[r] = np.isclose(row.astype(float), best, rtol=0.0, atol=1e-9) & row.notna()
    return mask


def _agent_y_true_and_prob_matrix_for_run(
    rows: list[tuple[str, int, str, str]],
    focals: list[str],
    predictions_root: Path,
    run: str,
) -> tuple[list[str], np.ndarray] | None:
    """Ground-truth classes and (n × 3) prob matrix for rows where this run has a pred.

    Matches by ``(tid, target_entity)`` so multi-target tweets are scored per target.
    """
    pmap_by_focal: dict[str, dict[EvalKey, dict[str, Any]]] = {}
    for fu in focals:
        pmap_by_focal[fu] = load_agent_predictions_by_key(
            str((predictions_root / run / f"predictions_{fu}.json").resolve())
        )
    y_true: list[str] = []
    probs: list[list[float]] = []
    for focal, tid, target, gt in rows:
        pr = pmap_by_focal.get(focal, {}).get((int(tid), target))
        if pr is None:
            continue
        y_true.append(gt)
        probs.append(_pred_prob_vec_ordered(pr))
    if not y_true:
        return None
    return y_true, np.asarray(probs, dtype=np.float64)


def _human_y_true_and_prob_matrix(
    rows: list[tuple[str, int, str, str]],
    human_probs_by_focal: dict[str, dict[EvalKey, dict[str, Any]]] | None,
) -> tuple[list[str], np.ndarray] | None:
    """Ground truth vs soft human probs only (``predictions_top5_jul2024_human_probs.json``)."""
    yt, _yp, pr = _human_gt_pred_probs_soft_only(rows, human_probs_by_focal, None)
    if not yt:
        return None
    return yt, np.asarray(pr, dtype=np.float64)


def render_roc_ovr_curves_for_runs(
    *,
    rows: list[tuple[str, int, str, str]],
    focals: list[str],
    predictions_root: Path,
    prediction_runs: list[str],
    run_display_labels: list[str],
    human_probs_path: Path,
) -> None:
    """One figure: 3 stacked panels (OvR), Human + one curve per agent model."""
    if len(run_display_labels) != len(prediction_runs):
        run_display_labels = _unique_abbrev_labels_for_runs(prediction_runs)
    if not rows:
        return
    human_probs_by_focal: dict[str, dict[int, dict[str, Any]]] | None = None
    hp_path = human_probs_path.expanduser().resolve()
    if hp_path.is_file():
        human_probs_by_focal = load_human_probs_by_focal_user(str(hp_path))
    built_human = _human_y_true_and_prob_matrix(rows, human_probs_by_focal)
    if not prediction_runs and built_human is None:
        return
    st.markdown("##### ROC curves (one-vs-rest, probability scores)")
    st.caption(
        f"Native canvas ≈ **{ROC_OVR_FIGSIZE_W_IN:.1f} × {ROC_OVR_FIGSIZE_H_IN:.1f}** inches at "
        f"**{ROC_OVR_FIGURE_DPI:.0f} dpi** → **~"
        f"{int(ROC_OVR_FIGSIZE_W_IN * ROC_OVR_FIGURE_DPI)} × {int(ROC_OVR_FIGSIZE_H_IN * ROC_OVR_FIGURE_DPI)}** px. "
        "Each panel is binary “this class vs rest” using **predicted probability for that class**. "
        "Agents: **p_endorsing / p_disendorsing / p_neutral** per run. Human: soft "
        "**prob_endorse / prob_disendorse / prob_neutral** from "
        "`predictions_top5_jul2024_human_probs.json`; rows missing that triple omitted. Diagonal = random."
    )
    if not hp_path.is_file():
        st.caption(
            f"Note: human soft-prob file not found at `{hp_path}`; Human ROC track is omitted."
        )
    style_map, families = _assign_run_styles(prediction_runs)
    if any(len(runs) > 1 for _f, runs in families):
        st.caption(
            "**Color = base model family; line style = prompt/calibration variant.** "
            "Same-color curves are the same model with different prompt or "
            "calibration tweaks. Human is dotted black."
        )
    if any(len(runs) > len(_VARIANT_LINE_STYLES) for _f, runs in families):
        st.caption(
            f"⚠ Some model families have more than {len(_VARIANT_LINE_STYLES)} variants — "
            "line styles will repeat within those families."
        )
    with plt.rc_context(
        {
            "figure.dpi": ROC_OVR_FIGURE_DPI,
            "savefig.dpi": ROC_OVR_FIGURE_DPI,
        },
    ):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(ROC_OVR_FIGSIZE_W_IN, ROC_OVR_FIGSIZE_H_IN),
            dpi=ROC_OVR_FIGURE_DPI,
            layout="constrained",
        )
        plotted_ovr = [False, False, False]
        if built_human is not None:
            y_true_h, P_h = built_human
            for ci, _cname in enumerate(GT_CLASSES):
                ax = axes[ci]
                y_bin = np.array([1 if yt == GT_CLASSES[ci] else 0 for yt in y_true_h], dtype=np.int32)
                if np.sum(y_bin) == 0 or np.sum(y_bin) == len(y_bin):
                    continue
                scores = P_h[:, ci]
                try:
                    fpr, tpr, _ = roc_curve(y_bin, scores)
                    roc_auc = auc(fpr, tpr)
                except ValueError:
                    continue
                ax.plot(
                    fpr,
                    tpr,
                    color="black",
                    lw=ROC_OVR_LINEWIDTH,
                    ls=":",
                    zorder=10,
                    label=f"Human ({roc_auc:.3f})",
                )
                plotted_ovr[ci] = True
        for run_idx, run in enumerate(prediction_runs):
            color, linestyle = style_map[run]
            built = _agent_y_true_and_prob_matrix_for_run(
                rows, focals, predictions_root, run
            )
            if built is None:
                continue
            y_true, P = built
            for ci, _cname in enumerate(GT_CLASSES):
                ax = axes[ci]
                y_bin = np.array([1 if yt == GT_CLASSES[ci] else 0 for yt in y_true], dtype=np.int32)
                if np.sum(y_bin) == 0 or np.sum(y_bin) == len(y_bin):
                    continue
                scores = P[:, ci]
                try:
                    fpr, tpr, _ = roc_curve(y_bin, scores)
                    roc_auc = auc(fpr, tpr)
                except ValueError:
                    continue
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    linestyle=linestyle,
                    lw=ROC_OVR_LINEWIDTH,
                    zorder=1,
                    label=f"{run_display_labels[run_idx]} ({roc_auc:.3f})",
                )
                plotted_ovr[ci] = True
        for ci, cname in enumerate(GT_CLASSES):
            ax = axes[ci]
            ax.plot([0, 1], [0, 1], "k:", alpha=0.4, lw=ROC_OVR_DIAG_LINEWIDTH)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xticks(np.arange(0.0, 1.01, 0.1))
            ax.set_yticks(np.arange(0.0, 1.01, 0.1))
            ax.set_xlabel("False positive rate", fontsize=ROC_OVR_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("True positive rate", fontsize=ROC_OVR_AXIS_LABEL_FONTSIZE)
            ax.set_title(f'GT class: "{cname}" vs rest', fontsize=ROC_OVR_TITLE_FONTSIZE)
            ax.tick_params(axis="both", labelsize=ROC_OVR_TICK_FONTSIZE)
            ax.set_axisbelow(True)
            ax.grid(
                True,
                which="major",
                linestyle=ROC_OVR_GRID_LINESTYLE,
                linewidth=ROC_OVR_GRID_LINEWIDTH,
                color=ROC_OVR_GRID_COLOR,
                alpha=ROC_OVR_GRID_ALPHA,
            )
            if plotted_ovr[ci]:
                ax.legend(
                    fontsize=ROC_OVR_LEGEND_FONTSIZE,
                    loc="lower left",
                    bbox_to_anchor=(1.015, 0),
                    frameon=False,
                    handlelength=1.6,
                    borderaxespad=0.0,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid OvR ROC\n(both classes needed)",
                    ha="center",
                    va="center",
                    fontsize=ROC_OVR_AXIS_LABEL_FONTSIZE,
                    color="0.35",
                )
            ax.set_aspect("equal", adjustable="box")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


# Single-panel "precision @ 20% endorsing prior" figure: square axes box, legend on the right.
PRIOR_CURVE_FIGSIZE_W_IN: float = 6.5
PRIOR_CURVE_FIGSIZE_H_IN: float = 4.6
# Endorsing-prior re-weighting; if you change these, also update the caption / axis label.
ENDORSING_PRIOR_POS: float = 20.0
ENDORSING_PRIOR_NEG: float = 80.0


def render_precision_at_endorsing_prior_curve(
    *,
    rows: list[tuple[str, int, str, str]],
    focals: list[str],
    predictions_root: Path,
    prediction_runs: list[str],
    run_display_labels: list[str],
    human_probs_path: Path,
) -> None:
    """Single panel: precision under a fixed 20% endorsing prior, plotted vs recall.

    Binary task: ``endorsing vs rest``. For each predictor we sweep the threshold
    on its endorsing-probability score (Human: ``prob_endorse``; agents:
    ``p_endorsing``) and at every operating point compute ``(TPR, FPR)``. Those
    are then re-weighted to a fixed positive prevalence of 20% (negatives 80%):

        R = (TPR * 20) / (TPR * 20 + FPR * 80)

    ``R`` is the **precision (positive predictive value)** the predictor would
    attain if endorsing items appeared at a 20% base rate in deployment —
    independent of the empirical prevalence on this evaluation slice (which lets
    curves with different supports share the same axes). A random scorer has
    ``TPR == FPR`` at every threshold so ``R`` collapses to ``20 / 100 = 0.20``;
    that's the dashed baseline.
    """
    if len(run_display_labels) != len(prediction_runs):
        run_display_labels = _unique_abbrev_labels_for_runs(prediction_runs)
    if not rows:
        return
    human_probs_by_focal: dict[str, dict[int, dict[str, Any]]] | None = None
    hp_path = human_probs_path.expanduser().resolve()
    if hp_path.is_file():
        human_probs_by_focal = load_human_probs_by_focal_user(str(hp_path))
    built_human = _human_y_true_and_prob_matrix(rows, human_probs_by_focal)
    if not prediction_runs and built_human is None:
        return

    pos_class = "endorsing"
    pos_idx = GT_CLASSES.index(pos_class)
    pi_pos = ENDORSING_PRIOR_POS
    pi_neg = ENDORSING_PRIOR_NEG
    baseline_R = pi_pos / (pi_pos + pi_neg)

    st.markdown(
        "##### Precision at 20% endorsing prior vs recall "
        "(endorsing vs rest, probability scores)"
    )
    st.caption(
        "Binary task: **endorsing vs rest**, scored on each predictor's "
        "**p_endorsing** (Human uses **prob_endorse** from "
        "`predictions_top5_jul2024_human_probs.json`). "
        "**Recall (x)** = TPR = TP / (TP + FN) — fraction of endorsing items caught. "
        "**R (y)** = (TPR · 20) / (TPR · 20 + FPR · 80) — the precision the "
        "predictor would have if endorsing items appeared at a fixed **20% base "
        "rate** in the population (re-weighting positives to 20% and negatives "
        "to 80% regardless of the empirical prevalence on this slice). "
        "**How to read it:** *if I deploy this predictor on a stream where ~20% "
        "of items are endorsing, what fraction of the items it flags will "
        "actually be endorsing as I dial recall up?* Higher = better; the "
        "left edge (low recall) shows precision when the predictor is most "
        "selective, the right edge (recall = 1) shows precision once it accepts "
        "everything. Each curve is computed on **only the rows where that "
        "predictor has a score** (its own support; `n=…` in the legend). "
        f"Dashed horizontal line = random-classifier baseline R = {baseline_R:.2f} "
        "(since TPR = FPR at every threshold for a random scorer)."
    )
    if not hp_path.is_file():
        st.caption(
            f"Note: human soft-prob file not found at `{hp_path}`; Human curve omitted."
        )
    style_map, families = _assign_run_styles(prediction_runs)
    if any(len(runs) > 1 for _f, runs in families):
        st.caption(
            "**Color = base model family; line style = prompt/calibration variant.** "
            "Same-color curves are the same model with different prompt or "
            "calibration tweaks. Human is dotted black."
        )
    if any(len(runs) > len(_VARIANT_LINE_STYLES) for _f, runs in families):
        st.caption(
            f"⚠ Some model families have more than {len(_VARIANT_LINE_STYLES)} variants — "
            "line styles will repeat within those families."
        )

    def _compute_R_curve(
        y_true_classes: list[str], P: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """(tpr, R) sweeping the endorsing-score threshold; ``None`` if degenerate."""
        y_bin = np.asarray(
            [1 if yt == pos_class else 0 for yt in y_true_classes], dtype=np.int32
        )
        n_pos = int(np.sum(y_bin))
        if n_pos == 0 or n_pos == len(y_bin):
            return None
        scores = P[:, pos_idx]
        try:
            fpr, tpr, _ = roc_curve(y_bin, scores)
        except ValueError:
            return None
        denom = tpr * pi_pos + fpr * pi_neg
        # R undefined when no items are flagged positive (TPR = FPR = 0); leave as NaN
        # so matplotlib doesn't draw a misleading point at the origin.
        R = np.where(denom > 0, (tpr * pi_pos) / np.maximum(denom, 1e-12), np.nan)
        return tpr, R

    with plt.rc_context(
        {
            "figure.dpi": ROC_OVR_FIGURE_DPI,
            "savefig.dpi": ROC_OVR_FIGURE_DPI,
        },
    ):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(PRIOR_CURVE_FIGSIZE_W_IN, PRIOR_CURVE_FIGSIZE_H_IN),
            dpi=ROC_OVR_FIGURE_DPI,
            layout="constrained",
        )
        plotted_any = False

        if built_human is not None:
            y_true_h, P_h = built_human
            curve = _compute_R_curve(y_true_h, P_h)
            if curve is not None:
                tpr_h, R_h = curve
                ax.plot(
                    tpr_h,
                    R_h,
                    color="black",
                    lw=ROC_OVR_LINEWIDTH,
                    ls=":",
                    zorder=10,
                    label=f"Human (n={len(y_true_h)})",
                )
                plotted_any = True

        for run_idx, run in enumerate(prediction_runs):
            color, linestyle = style_map[run]
            built = _agent_y_true_and_prob_matrix_for_run(
                rows, focals, predictions_root, run
            )
            if built is None:
                continue
            y_true, P = built
            curve = _compute_R_curve(y_true, P)
            if curve is None:
                continue
            tpr_r, R_r = curve
            ax.plot(
                tpr_r,
                R_r,
                color=color,
                linestyle=linestyle,
                lw=ROC_OVR_LINEWIDTH,
                zorder=1,
                label=f"{run_display_labels[run_idx]} (n={len(y_true)})",
            )
            plotted_any = True

        ax.axhline(
            baseline_R,
            color="0.4",
            lw=ROC_OVR_DIAG_LINEWIDTH,
            ls="--",
            alpha=0.6,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.set_xticks(np.arange(0.0, 1.01, 0.05), minor=True)
        ax.set_yticks(np.arange(0.0, 1.01, 0.05), minor=True)
        ax.set_xlabel("Recall (TPR)", fontsize=ROC_OVR_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(
            "R = precision @ 20% endorsing prior",
            fontsize=ROC_OVR_AXIS_LABEL_FONTSIZE,
        )
        ax.set_title(
            f'Endorsing vs rest (random baseline R={baseline_R:.2f})',
            fontsize=ROC_OVR_TITLE_FONTSIZE,
        )
        ax.tick_params(axis="both", labelsize=ROC_OVR_TICK_FONTSIZE)
        ax.set_axisbelow(True)
        ax.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=0.6,
            color="0.55",
            alpha=0.7,
        )
        ax.grid(
            True,
            which="minor",
            linestyle=":",
            linewidth=0.4,
            color="0.75",
            alpha=0.5,
        )
        if plotted_any:
            ax.legend(
                fontsize=ROC_OVR_LEGEND_FONTSIZE,
                loc="lower left",
                bbox_to_anchor=(1.015, 0),
                frameon=False,
                handlelength=1.6,
                borderaxespad=0.0,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No valid endorsing-vs-rest curve\n(both classes needed)",
                ha="center",
                va="center",
                fontsize=ROC_OVR_AXIS_LABEL_FONTSIZE,
                color="0.35",
            )
        ax.set_aspect("equal", adjustable="box")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


def build_evaluation_rows_for_scope(
    clean_labels_dir: Path,
    focal_users: list[str],
    visible_ids_by_user: dict[str, set[str]],
) -> list[tuple[str, int, str, str]]:
    """(focal, tweet_id, target_entity, gt_class) for items in-scope with usable GT.

    Iterates ``clean_labels`` keyed by ``(tid, target_entity)`` so multi-target tweets
    are not silently collapsed. ``visible_ids_by_user`` is per-tid (= the set of tids
    in that focal's labeling pool); every (tid, target) clean-label row whose tid is in
    scope is included, even if only one of the targets was actually shown to the human
    labeler. Per-model support naturally drops back when a predictor has no record for
    a given (tid, target).
    """
    rows: list[tuple[str, int, str, str]] = []
    for focal in focal_users:
        by_key = load_clean_label_items_by_key(
            str(clean_labels_dir.resolve()), focal
        )
        visible = visible_ids_by_user.get(focal, set())
        for (tid, target), rec in by_key.items():
            if str(tid) not in visible:
                continue
            gt = _normalize_gt_label(rec.get("ground_truth_label"))
            if gt is None:
                continue
            rows.append((focal, int(tid), target, gt))
    return rows


def render_review_performance(
    *,
    clean_labels_dir: Path,
    predictions_root: Path,
    focal_username: str,
    visible_ids_by_user: dict[str, set[str]],
    labels_by_user: dict[str, dict[str, dict[str, Any]]],
    prediction_runs: list[str],
    run_display_labels: list[str],
    human_probs_path: Path = DEFAULT_HUMAN_PROBS_JSON,
) -> None:
    """Aggregate metrics vs clean_labels GT; confusion matrices per human / agent run."""
    if len(run_display_labels) != len(prediction_runs):
        run_display_labels = _unique_abbrev_labels_for_runs(prediction_runs)
    st.subheader("Performance vs ground truth (`clean_labels`)")
    st.caption(
        "GT = **ground_truth_label** (endorsing / disendorsing / neutral). "
        "**Eval rows are keyed by `(representative_tweet_id, target_entity)`** — multi-target "
        "tweets contribute one row per target; predictions are matched per target so a "
        "tweet's GT row is never paired with another target's score. "
        "Agents: argmax(**p_endorsing**, **p_disendorsing**, **p_neutral**). "
        "**Human** classification: **label** from ``predictions_top5_jul2024_human_probs.json`` "
        "(**wrong_target** excluded). **Human Brier / skill**: soft "
        "**prob_endorse / prob_disendorse / prob_neutral** from that file only; "
        "evaluation rows without all three probs are dropped from the Human row. "
        "**Brier** mean (1/3)∑ᵢ(pᵢ−target)² per row; "
        "**Brier marginal** = same metric if every row predicted the GT class-frequency vector; "
        "**Brier skill** = 1 − Brier/Brierₘ. **Weighted AUROC** = sklearn OvR ROC AUC per "
        "class, averaged with weights = **GT counts per class / n** (more frequent "
        "classes count more)."
    )

    scope = st.radio(
        "Evaluation slice",
        options=["pool_all_focal", "current_focal"],
        horizontal=True,
        format_func=lambda x: (
            "Pool all focal users" if x == "pool_all_focal" else f"`{focal_username}` only"
        ),
        key="review_perf_scope",
    )
    focals = (
        list(FOCAL_USERS) if scope == "pool_all_focal" else [focal_username]
    )

    if not prediction_runs:
        st.warning(
            f"No subfolders found under `{predictions_root}`. "
            "Add Cursor agent output directories there to score models."
        )

    rows = build_evaluation_rows_for_scope(
        clean_labels_dir=clean_labels_dir,
        focal_users=focals,
        visible_ids_by_user=visible_ids_by_user,
    )
    if not rows:
        st.warning("No overlapping clean-label GT rows for this slice and visible IDs.")
        return

    st.caption(f"**{len(rows)}** items with usable GT after filtering to labeling scope.")

    hp_path = human_probs_path.expanduser().resolve()
    human_probs_by_focal: dict[str, dict[int, dict[str, Any]]] = {}
    if hp_path.is_file():
        human_probs_by_focal = load_human_probs_by_focal_user(str(hp_path))
    else:
        st.warning(
            f"Human probability metrics need soft labels at `{hp_path}` (missing). "
            "Human row in the summary is omitted until that file exists."
        )

    yt_h, yp_h, pr_h = _human_gt_pred_probs_soft_only(rows, human_probs_by_focal, None)
    if hp_path.is_file() and not yt_h and rows:
        st.info(
            "No evaluation rows overlap **predictions_top5_jul2024_human_probs.json** with "
            "all three `prob_*` fields; Human summary metrics are empty for this slice."
        )

    human_row = _metrics_from_predictions(yt_h, yp_h, pr_h) if yt_h else None
    if yt_h and len(yt_h) < len(rows):
        st.caption(
            f"**Human** summary uses **{len(yt_h)}** rows with all three `prob_*` fields in "
            f"`predictions_top5_jul2024_human_probs.json` (of **{len(rows)}** GT rows here). "
            "Agent rows still use every prediction that overlaps the slice."
        )

    metric_rows: list[dict[str, Any]] = []
    if human_row and int(human_row.get("support") or 0) > 0:
        mr = dict(human_row)
        mr["model"] = "Human"
        metric_rows.append(mr)

    active_prediction_runs: list[str] = []
    active_run_display_labels: list[str] = []
    for i, run in enumerate(prediction_runs):
        pmap_by_focal: dict[str, dict[EvalKey, dict[str, Any]]] = {}
        for fu in focals:
            pmap_by_focal[fu] = load_agent_predictions_by_key(
                str((predictions_root / run / f"predictions_{fu}.json").resolve())
            )
        yt_v: list[str] = []
        yp_v: list[str] = []
        prv: list[list[float]] = []
        for focal, tid, target, gt in rows:
            pr = pmap_by_focal.get(focal, {}).get((int(tid), target))
            if pr is None:
                continue
            yt_v.append(gt)
            yp_v.append(_agent_prediction_argmax_label(pr))
            prv.append(_pred_prob_vec_ordered(pr))
        rdict = _metrics_from_predictions(yt_v, yp_v, prv)
        if int(rdict.get("support") or 0) == 0:
            continue
        rdict["model"] = run_display_labels[i]
        metric_rows.append(rdict)
        active_prediction_runs.append(run)
        active_run_display_labels.append(run_display_labels[i])

    n_excluded_support0 = len(prediction_runs) - len(active_prediction_runs)
    if n_excluded_support0 > 0:
        st.caption(
            f"Excluded **{n_excluded_support0}** agent run(s) with support 0 "
            "(no overlapping predictions vs GT for this evaluation slice)."
        )

    if metric_rows:
        dfm = pd.DataFrame(metric_rows)
        _summary_metric_col_order = [
            "model",
            "accuracy",
            "brier_skill",
            "weighted_auroc",
            "weighted_f1",
            "macro_f1",
            "f1_endorsing",
            "f1_disendorsing",
            "f1_neutral",
            "brier",
            "brier_marginal",
            "support",
        ]
        cols_ordered = [c for c in _summary_metric_col_order if c in dfm.columns]
        leftover = [c for c in dfm.columns if c not in cols_ordered]
        dfm = dfm[cols_ordered + leftover]
        st.markdown("##### Summary metrics")
        st.caption(
            "**Bold** = best value in that column (accuracy / Brier skill / weighted AUROC / F1: higher; "
            "Brier & marginal baseline Brier: lower). **Support** is not ranked."
        )

        bold_summary_internal = _summary_metrics_bold_mask(dfm)

        display = dfm.copy()

        display["accuracy"] = display["accuracy"].map(
            lambda v: f"{v * 100:.2f}%" if v == v else "—"
        )
        for c in [
            "weighted_auroc",
            "weighted_f1",
            "macro_f1",
            "f1_endorsing",
            "f1_disendorsing",
            "f1_neutral",
        ]:
            display[c] = display[c].map(lambda v: f"{v:.4f}" if v == v else "—")

        display["brier"] = display["brier"].map(lambda v: f"{v:.4f}" if v == v else "—")
        display["brier_marginal"] = display["brier_marginal"].map(
            lambda v: f"{v:.4f}" if v == v else "—"
        )

        def _fmt_skill_cell(val: Any) -> str:
            if val is None:
                return "—"
            try:
                x = float(val)
            except (TypeError, ValueError):
                return "—"
            if x != x:
                return "—"
            return f"{x:.4f}"

        display["brier_skill"] = display["brier_skill"].map(_fmt_skill_cell)
        rename = {
            "model": "Predictor",
            "accuracy": "Accuracy",
            "weighted_auroc": "Weighted AUROC",
            "weighted_f1": "Weighted F1",
            "macro_f1": "Macro F1",
            "f1_endorsing": "F₁ endorsing",
            "f1_disendorsing": "F₁ disendorsing",
            "f1_neutral": "F₁ neutral",
            "brier": "Brier",
            "brier_marginal": "Brier (marginal baseline)",
            "brier_skill": "Brier skill vs marginal",
            "support": "Support (n)",
        }
        display = display.rename(columns=rename)
        bold_summary_display = bold_summary_internal.rename(columns=rename)

        styled_summary = display.style.apply(
            _styler_css_bold(bold_summary_display), axis=None
        )
        st.dataframe(styled_summary, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No overlapping predictions to score (human had no usable labels and/or "
            "no model preds aligned with GT)."
        )

    # --- Focal × model: Brier skill matrix ----------------------------------
    if rows and (yt_h or active_prediction_runs):

        def _fmt_skill_cell(m: dict[str, Any] | None) -> str:
            if m is None:
                return "—"
            sk = m.get("brier_skill")
            try:
                x = float(sk) if sk is not None else None
            except (TypeError, ValueError):
                return "—"
            if x is None or x != x:
                return "—"
            return f"{x:.4f}"

        def _human_metrics_focal(fu: str) -> dict[str, Any] | None:
            yt_f, yp_f, pr_f = _human_gt_pred_probs_soft_only(
                rows, human_probs_by_focal, focal_filter=fu
            )
            if not yt_f:
                return None
            return _metrics_from_predictions(yt_f, yp_f, pr_f)

        def _agent_metrics_focal(fu: str, run: str) -> dict[str, Any] | None:
            pmap = load_agent_predictions_by_key(
                str((predictions_root / run / f"predictions_{fu}.json").resolve())
            )
            yt_v: list[str] = []
            yp_v: list[str] = []
            prv: list[list[float]] = []
            for focal, tid, target, gt in rows:
                if focal != fu:
                    continue
                pr = pmap.get((int(tid), target))
                if pr is None:
                    continue
                yt_v.append(gt)
                yp_v.append(_agent_prediction_argmax_label(pr))
                prv.append(_pred_prob_vec_ordered(pr))
            if not yt_v:
                return None
            return _metrics_from_predictions(yt_v, yp_v, prv)

        idx = list(focals)
        col_models: list[str] = []
        if yt_h:
            col_models.append("Human")
        col_models.extend(active_run_display_labels)
        skill_fmt: list[list[str]] = []
        skill_num: list[list[float]] = []
        for fu in idx:
            fmt_row: list[str] = []
            num_row: list[float] = []
            if yt_h:
                hm = _human_metrics_focal(fu)
                num_row.append(_brier_skill_float_from_metrics(hm))
                fmt_row.append(_fmt_skill_cell(hm))
            for run in active_prediction_runs:
                am = _agent_metrics_focal(fu, run)
                num_row.append(_brier_skill_float_from_metrics(am))
                fmt_row.append(_fmt_skill_cell(am))
            skill_fmt.append(fmt_row)
            skill_num.append(num_row)

        df_skill = pd.DataFrame(skill_fmt, index=idx, columns=col_models)
        df_skill_num = pd.DataFrame(skill_num, index=idx, columns=col_models)
        df_skill.index.name = "Focal user"
        bold_skill_inner = _skill_grid_row_best_mask(df_skill_num)

        ds_skill = df_skill.reset_index()
        bold_skill_pad = pd.DataFrame(False, index=ds_skill.index, columns=ds_skill.columns)
        focal_col = ds_skill.columns[0]
        bold_skill_pad[focal_col] = False
        for j, cm in enumerate(col_models):
            bold_skill_pad[cm] = bold_skill_inner[cm].values

        st.markdown("##### Focal user × model: Brier skill vs marginal")
        st.caption(
            "Rows = focal users in this evaluation slice; columns = Human (if any) + each "
            "agent run with non-zero support on this slice. "
            "Each cell is Brier skill vs marginal on that focal subset only "
            "(same definition as summary; Human = soft probs from human_probs JSON only). "
            "— means no overlapping usable GT rows for that cell. "
            "**Bold** = highest skill in that row (same focal; ties bold together)."
        )
        styled_skill = ds_skill.style.apply(
            _styler_css_bold(bold_skill_pad), axis=None
        )
        st.dataframe(styled_skill, use_container_width=True, hide_index=True)

    if rows:
        render_roc_ovr_curves_for_runs(
            rows=rows,
            focals=list(focals),
            predictions_root=predictions_root,
            prediction_runs=active_prediction_runs,
            run_display_labels=active_run_display_labels,
            human_probs_path=human_probs_path,
        )
        render_precision_at_endorsing_prior_curve(
            rows=rows,
            focals=list(focals),
            predictions_root=predictions_root,
            prediction_runs=active_prediction_runs,
            run_display_labels=active_run_display_labels,
            human_probs_path=human_probs_path,
        )

    st.markdown("##### Confusion matrices (rows = GT, columns = predicted)")
    if yt_h:
        with st.expander("Human", expanded=not active_prediction_runs):
            st.dataframe(_confusion_df(yt_h, yp_h), use_container_width=True)

    for i, run in enumerate(active_prediction_runs):
        pmap_by_focal_cm: dict[str, dict[EvalKey, dict[str, Any]]] = {}
        for fu in focals:
            pmap_by_focal_cm[fu] = load_agent_predictions_by_key(
                str((predictions_root / run / f"predictions_{fu}.json").resolve())
            )
        yt_c: list[str] = []
        yp_c: list[str] = []
        for focal, tid, target, gt in rows:
            pr = pmap_by_focal_cm.get(focal, {}).get((int(tid), target))
            if pr is None:
                continue
            yt_c.append(gt)
            yp_c.append(_agent_prediction_argmax_label(pr))
        if not yt_c:
            continue
        with st.expander(active_run_display_labels[i], expanded=False):
            st.dataframe(_confusion_df(yt_c, yp_c), use_container_width=True)


# ---------------------------------------------------------------------------
# Async save: a single background daemon thread consuming snapshot writes.
# Multiple queued snapshots are fine; the latest one always wins on disk
# because we serialize through one worker and write atomically.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def get_save_worker() -> tuple["queue.Queue[tuple[str, Path] | None]", list[str]]:
    """Return (queue, error_log). The worker writes pre-serialized JSON atomically."""
    q: "queue.Queue[tuple[str, Path] | None]" = queue.Queue()
    errors: list[str] = []

    def worker() -> None:
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return
            text, path = item
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                with tmp_path.open("w", encoding="utf-8") as f:
                    f.write(text)
                tmp_path.replace(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{utc_now_iso()} save failed: {exc!r}")
            finally:
                q.task_done()

    t = threading.Thread(target=worker, name="label-save-worker", daemon=True)
    t.start()
    return q, errors


def schedule_save(payload: dict[str, Any], path: Path) -> None:
    """Serialize in the caller thread (no deepcopy); worker only writes bytes."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    q, _ = get_save_worker()
    q.put((text, path))


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_sources(
    ground_truth_path: str,
    neighbors_path: str,
    graph_path: str,
) -> dict[str, Any]:
    """Load the three source files and return a normalized bundle."""
    gt = load_json(Path(ground_truth_path))
    nb = load_json(Path(neighbors_path))
    graph_data = load_json(Path(graph_path))

    graph: dict[str, list[str]] = {}
    for focal, neighbors in (graph_data.get("effective_neighbor_graph") or {}).items():
        graph[focal] = [n["username"] for n in neighbors]

    return {
        "ground_truth": gt,
        "neighbors": nb,
        "graph": graph,
        "ground_truth_path": ground_truth_path,
        "neighbors_path": neighbors_path,
        "graph_path": graph_path,
    }


def items_for_focal(
    bundle: dict[str, Any],
    focal: str,
    allowed_ids_by_user: dict[str, set[int]] | None = None,
) -> list[dict[str, Any]]:
    """Build the deterministic item list for a focal user.

    Combines that user's own targets and their neighborhood's targets,
    dedupes by representative_tweet_id (focal pool wins), tags each item
    with `_source` ("focal"/"neighbor"), sorts by representative_tweet_id,
    then shuffles with a seed = focal username.
    """
    gt_items = bundle["ground_truth"].get("items", [])
    nb_items = bundle["neighbors"].get("items", [])
    neighbor_usernames = set(bundle["graph"].get(focal, []))

    by_id: dict[int, dict[str, Any]] = {}

    for it in nb_items:
        if it.get("path_anchor_username") in neighbor_usernames:
            tagged = dict(it)
            tagged["_source"] = "neighbor"
            by_id[int(it["representative_tweet_id"])] = tagged

    for it in gt_items:
        if it.get("path_anchor_username") == focal:
            tagged = dict(it)
            tagged["_source"] = "focal"
            by_id[int(it["representative_tweet_id"])] = tagged

    ordered = [by_id[k] for k in sorted(by_id.keys())]
    allowed_ids = (allowed_ids_by_user or {}).get(focal)
    if allowed_ids:
        ordered = [
            item
            for item in ordered
            if int(item["representative_tweet_id"]) in allowed_ids
        ]
    Random(focal).shuffle(ordered)
    return ordered


# ---------------------------------------------------------------------------
# Labels: in-memory state and on-disk persistence
# ---------------------------------------------------------------------------


def load_labels(output_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {focal_user: {representative_tweet_id_str: record}}."""
    if not output_path.exists():
        return {u: {} for u in FOCAL_USERS}
    try:
        data = load_json(output_path)
    except json.JSONDecodeError:
        return {u: {} for u in FOCAL_USERS}
    out: dict[str, dict[str, dict[str, Any]]] = {u: {} for u in FOCAL_USERS}
    for user, user_block in (data.get("users") or {}).items():
        if user not in out:
            out[user] = {}
        for rec in user_block.get("labels", []):
            tid = rec.get("representative_tweet_id")
            if tid is None:
                continue
            out[user][str(tid)] = _sanitize_label_record(rec)
    return out


def filter_labels_by_visible_ids(
    labels_by_user: dict[str, dict[str, dict[str, Any]]],
    visible_ids_by_user: dict[str, set[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    filtered: dict[str, dict[str, dict[str, Any]]] = {}
    for user in FOCAL_USERS:
        visible_ids = visible_ids_by_user.get(user, set())
        filtered[user] = {
            tid: rec
            for tid, rec in labels_by_user.get(user, {}).items()
            if tid in visible_ids
        }
    return filtered


def build_payload(
    *,
    bundle: dict[str, Any],
    labels_by_user: dict[str, dict[str, dict[str, Any]]],
    output_path: Path,
    visible_ids_by_user: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    users_block: dict[str, Any] = {}
    for user in FOCAL_USERS:
        visible_ids = (visible_ids_by_user or {}).get(user)
        recs = [
            rec
            for tid, rec in labels_by_user.get(user, {}).items()
            if visible_ids is None or tid in visible_ids
        ]
        recs_sorted = sorted(
            recs, key=lambda r: int(r.get("representative_tweet_id") or 0)
        )
        users_block[user] = {
            "label_count": len(recs_sorted),
            "labels": [_sanitize_label_record(r) for r in recs_sorted],
        }
    return {
        "version": 3,
        "task": "top5_jul2024_focal_plus_neighbors",
        "label_options": LABEL_VALUES,
        "label_display": LABEL_DISPLAY,
        "ground_truth_source": bundle["ground_truth_path"],
        "neighbors_source": bundle["neighbors_path"],
        "neighbor_graph_source": bundle["graph_path"],
        "output_path": str(output_path),
        "last_updated_at": utc_now_iso(),
        "focal_users": FOCAL_USERS,
        "confidence_options": CONFIDENCE_OPTIONS,
        "confidence_prior_note": CONFIDENCE_PRIOR_NOTE,
        "users": users_block,
    }


def set_label(
    *,
    focal: str,
    item: dict[str, Any],
    label: str,
    bundle: dict[str, Any],
    output_path: Path,
    confidence_key: str,
    visible_ids_by_user: dict[str, set[str]],
) -> None:
    labels_by_user: dict[str, dict[str, dict[str, Any]]] = st.session_state[
        "labels_by_user"
    ]
    pct = st.session_state.get(confidence_key, DEFAULT_CONFIDENCE_PCT)
    if pct not in CONFIDENCE_OPTION_SET:
        pct = DEFAULT_CONFIDENCE_PCT
    record = {
        "representative_tweet_id": item["representative_tweet_id"],
        "target_entity": item.get("target_entity"),
        "longer_name": item.get("longer_name"),
        "url": item.get("url"),
        "_path_anchor_tweet_id": item.get("path_anchor_tweet_id"),
        "label": label,
        "confidence_correct_pct": int(pct),
        "labeled_at": utc_now_iso(),
    }
    labels_by_user.setdefault(focal, {})[str(item["representative_tweet_id"])] = record

    payload = build_payload(
        bundle=bundle,
        labels_by_user=labels_by_user,
        output_path=output_path,
        visible_ids_by_user=visible_ids_by_user,
    )
    schedule_save(payload, output_path)


def _on_confidence_change(
    focal: str,
    item: dict[str, Any],
    bundle: dict[str, Any],
    output_path: Path,
    confidence_key: str,
    visible_ids_by_user: dict[str, set[str]],
) -> None:
    """Persist confidence when the radio changes and a label already exists."""
    labels_by_user: dict[str, dict[str, dict[str, Any]]] = st.session_state[
        "labels_by_user"
    ]
    tid = str(item["representative_tweet_id"])
    rec = labels_by_user.get(focal, {}).get(tid)
    if rec is None or not rec.get("label"):
        return
    pct = st.session_state.get(confidence_key, DEFAULT_CONFIDENCE_PCT)
    if pct not in CONFIDENCE_OPTION_SET:
        pct = DEFAULT_CONFIDENCE_PCT
    rec["confidence_correct_pct"] = int(pct)
    rec["labeled_at"] = utc_now_iso()
    payload = build_payload(
        bundle=bundle,
        labels_by_user=labels_by_user,
        output_path=output_path,
        visible_ids_by_user=visible_ids_by_user,
    )
    schedule_save(payload, output_path)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def render_card(
    *,
    idx: int,
    item: dict[str, Any],
    current_label: str | None,
    current_confidence_pct: int | None,
    focal: str,
    bundle: dict[str, Any],
    output_path: Path,
    visible_ids_by_user: dict[str, set[str]],
    review_mode: bool,
    clean_by_key: dict[EvalKey, dict[str, Any]],
    prediction_runs: list[str],
    run_display_labels: list[str],
    pred_by_run: dict[str, dict[EvalKey, dict[str, Any]]],
) -> None:
    target = item.get("target_entity") or "(no target_entity)"
    longer = item.get("longer_name") or ""
    url = item.get("url") or ""
    tid = item["representative_tweet_id"]
    confidence_key = f"conf_{focal}_{tid}"
    item_key: EvalKey = (int(tid), str(item.get("target_entity") or ""))

    with st.container(border=True):
        st.markdown(f"### {idx}. {target}")
        if longer and longer != target:
            st.markdown(f"**Longer name:** {longer}")
        if url and not _is_x_com_url(url):
            st.markdown(f"**URL:** {url}")

        if review_mode:
            clean_rec = clean_by_key.get(item_key)
            _review_comparison_md_row(
                clean_rec=clean_rec,
                human_label=current_label,
                prediction_runs=prediction_runs,
                run_display_labels=run_display_labels,
                pred_by_run=pred_by_run,
                eval_key=item_key,
            )
            st.caption(
                f"{EMOJI_ENDORSING} endorsing · {EMOJI_NEUTRAL} neutral · "
                f"{EMOJI_DISENDORING} disendorsing · {EMOJI_HUMAN_WRONG_TARGET} wrong target (human)"
            )

            st.markdown("##### Ground truth (`clean_labels`)")
            if clean_rec:
                gt = clean_rec.get("ground_truth_label") or "(missing)"
                st.markdown(f"- **ground_truth_label:** `{gt}`")
                st.caption(
                    f"direction={clean_rec.get('direction')!r}, "
                    f"source={clean_rec.get('source')!r}, "
                    f"author={clean_rec.get('representative_tweet_author')!r}"
                )
                ctx = clean_rec.get("context")
                if ctx:
                    with st.expander("Context (clean labels)"):
                        st.write(ctx)
            else:
                st.warning("No row in clean_labels for this tweet id.")

            for display_label, run in zip(run_display_labels, prediction_runs, strict=True):
                pr = pred_by_run.get(run, {}).get(item_key)
                if not pr:
                    continue
                title = _truncate_display(display_label)
                with st.expander(f"{title} — probs & reasoning", expanded=False):
                    st.markdown(
                        f"**Argmax:** `{_agent_prediction_argmax_label(pr)}`  \n"
                        f"**Probs:** {_format_agent_probs(pr)}"
                    )
                    st.markdown(f"**Reasoning:** {pr.get('reasoning') or '—'}")

            st.markdown("##### Your human label (read-only)")
            if current_label:
                conf_note = ""
                if current_confidence_pct is not None:
                    conf_note = f" — confidence **{current_confidence_pct}%**"
                st.markdown(
                    f"`{current_label}` "
                    f"({LABEL_DISPLAY.get(current_label, current_label)}){conf_note}"
                )
            else:
                st.markdown("_No label saved for this item._")
            return

        if confidence_key not in st.session_state:
            if current_confidence_pct in CONFIDENCE_OPTION_SET:
                st.session_state[confidence_key] = current_confidence_pct
            else:
                st.session_state[confidence_key] = DEFAULT_CONFIDENCE_PCT

        st.caption(CONFIDENCE_PRIOR_NOTE)
        st.radio(
            "Confidence your label is correct",
            options=CONFIDENCE_OPTIONS,
            horizontal=True,
            format_func=lambda p: f"{p}%",
            key=confidence_key,
            on_change=_on_confidence_change,
            kwargs={
                "focal": focal,
                "item": item,
                "bundle": bundle,
                "output_path": output_path,
                "confidence_key": confidence_key,
                "visible_ids_by_user": visible_ids_by_user,
            },
        )

        if current_label:
            st.markdown(
                f"**Current label:** `{current_label}` "
                f"({LABEL_DISPLAY.get(current_label, current_label)})"
            )
        else:
            st.markdown("**Current label:** _none_")

        cols = st.columns(len(LABEL_OPTIONS))
        for col, (value, display) in zip(cols, LABEL_OPTIONS):
            is_active = current_label == value
            col.button(
                ("✓ " if is_active else "") + display,
                key=f"btn_{focal}_{tid}_{value}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                on_click=set_label,
                kwargs={
                    "focal": focal,
                    "item": item,
                    "label": value,
                    "bundle": bundle,
                    "output_path": output_path,
                    "confidence_key": confidence_key,
                    "visible_ids_by_user": visible_ids_by_user,
                },
            )


def _render_browse_item_page(
    *,
    total: int,
    items: list[dict[str, Any]],
    focal: str,
    bundle: dict[str, Any],
    output_path: Path,
    visible_ids_by_user: dict[str, set[str]],
    filter_choice: str,
    filter_to_label_value: dict[str, str],
    items_per_page: int,
    labels_by_user: dict[str, dict[str, dict[str, Any]]],
    review_mode: bool,
    clean_by_key: dict[EvalKey, dict[str, Any]],
    prediction_runs: list[str],
    run_display_labels: list[str],
    pred_by_run: dict[str, dict[EvalKey, dict[str, Any]]],
) -> None:
    """Paging + filtered item cards for the current focal."""
    if total == 0:
        st.info("No items for this focal user (or empty neighborhood).")
        return

    filtered_items: list[tuple[int, dict[str, Any]]] = []
    for idx, item in enumerate(items, start=1):
        tid_str = str(item["representative_tweet_id"])
        rec = labels_by_user.get(focal, {}).get(tid_str)
        current_label = rec.get("label") if rec else None

        if filter_choice == "Unlabeled" and current_label is not None:
            continue
        if filter_choice in filter_to_label_value:
            if current_label != filter_to_label_value[filter_choice]:
                continue

        filtered_items.append((idx, item))

    n_filtered = len(filtered_items)
    if n_filtered == 0:
        st.info(f"No items match filter '{filter_choice}'.")
        return

    n_pages = max(1, (n_filtered + items_per_page - 1) // items_per_page)
    page_key = f"endorse_label_page::{focal}::{filter_choice}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    if int(st.session_state[page_key]) > n_pages:
        st.session_state[page_key] = n_pages
    page = st.sidebar.number_input(
        "Page",
        min_value=1,
        max_value=n_pages,
        step=1,
        key=page_key,
        help=f"{n_filtered} items match this filter.",
    )
    start = (page - 1) * items_per_page
    page_slice = filtered_items[start : start + items_per_page]

    st.caption(
        f"Showing **{len(page_slice)}** of **{n_filtered}** matching items "
        f"(page **{page}** / **{n_pages}**)."
    )

    for idx, item in page_slice:
        tid_str = str(item["representative_tweet_id"])
        rec = labels_by_user.get(focal, {}).get(tid_str)
        current_label = rec.get("label") if rec else None
        cur_conf = rec.get("confidence_correct_pct") if rec else None
        if cur_conf is not None:
            try:
                cur_conf = int(cur_conf)
            except (TypeError, ValueError):
                cur_conf = None

        render_card(
            idx=idx,
            item=item,
            current_label=current_label,
            current_confidence_pct=cur_conf,
            focal=focal,
            bundle=bundle,
            output_path=output_path,
            visible_ids_by_user=visible_ids_by_user,
            review_mode=review_mode,
            clean_by_key=clean_by_key,
            prediction_runs=prediction_runs,
            run_display_labels=run_display_labels,
            pred_by_run=pred_by_run,
        )


def main() -> None:
    st.set_page_config(
        page_title="Endorsement target labeler",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Endorsement target labeler — top 5 + neighbors (Jul 2024)")

    # --- Sidebar: file paths ------------------------------------------------
    st.sidebar.header("Source files")
    gt_str = st.sidebar.text_input(
        "Ground-truth (focal) targets JSON",
        value=str(DEFAULT_GROUND_TRUTH),
    )
    nb_str = st.sidebar.text_input(
        "Neighbor targets JSON",
        value=str(DEFAULT_NEIGHBORS),
    )
    graph_str = st.sidebar.text_input(
        "Effective neighbor graph JSON",
        value=str(DEFAULT_NEIGHBOR_GRAPH),
    )
    out_str = st.sidebar.text_input(
        "Output labels JSON (single shared file)",
        value=str(DEFAULT_OUTPUT),
    )

    for label, p in (
        ("ground-truth", gt_str),
        ("neighbors", nb_str),
        ("graph", graph_str),
    ):
        if not Path(p).exists():
            st.sidebar.error(f"{label} file not found: {p}")
            st.stop()

    output_path = Path(out_str).expanduser().resolve()
    bundle_key = (gt_str, nb_str, graph_str)

    try:
        bundle = load_sources(gt_str, nb_str, graph_str)
    except json.JSONDecodeError as exc:
        st.sidebar.error(f"JSON parse error: {exc}")
        st.stop()

    allowed_ids_by_user = load_clean_label_allowed_ids(str(DEFAULT_CLEAN_LABELS_DIR))
    # Build per-focal item lists once per source bundle (not 10+ times per rerun).
    items_cache_key = "endorse_items_by_focal"
    items_bundle_key = "endorse_items_bundle_key"
    if (
        st.session_state.get(items_bundle_key) != bundle_key
        or items_cache_key not in st.session_state
    ):
        st.session_state[items_bundle_key] = bundle_key
        st.session_state[items_cache_key] = {
            u: items_for_focal(bundle, u, allowed_ids_by_user) for u in FOCAL_USERS
        }
    items_by_focal: dict[str, list[dict[str, Any]]] = st.session_state[items_cache_key]

    visible_ids_by_user = {
        user: {str(item["representative_tweet_id"]) for item in items_by_focal[user]}
        for user in FOCAL_USERS
    }

    # --- Load labels into session_state once per output path + source bundle
    state_key = f"labels_loaded::{output_path}::{bundle_key}"
    if not st.session_state.get(state_key):
        st.session_state["labels_by_user"] = filter_labels_by_visible_ids(
            load_labels(output_path),
            visible_ids_by_user,
        )
        st.session_state[state_key] = True
    labels_by_user: dict[str, dict[str, dict[str, Any]]] = st.session_state[
        "labels_by_user"
    ]

    all_cursor_run_folders = list_cursor_prediction_run_folders(
        str(CURSOR_PREDICTIONS_DIR.resolve())
    )

    # --- Sidebar: focal user picker -----------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("Focal user")
    focal = st.sidebar.selectbox(
        "Pick a focal user to label",
        options=FOCAL_USERS,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Review mode")
    review_mode = st.sidebar.checkbox(
        "Show ground truth & model predictions (read-only)",
        value=True,
        help=(
            "Shows rows from `clean_labels/`, a compact comparison table of "
            "ground truth / human / each agent run folder under "
            "`cursor_agent_predictions/` (emoji = argmax), "
            "then details. Label buttons and confidence are hidden; your saved "
            "label is read-only."
        ),
    )
    prediction_runs: list[str] = []
    clean_by_key: dict[EvalKey, dict[str, Any]] = {}
    pred_by_run: dict[str, dict[EvalKey, dict[str, Any]]] = {}
    if review_mode:
        prediction_runs = all_cursor_run_folders
        if all_cursor_run_folders:
            st.sidebar.caption(
                f"Using **{len(all_cursor_run_folders)}** run folder(s) under "
                "`cursor_agent_predictions/` (one column per folder, newest first)."
            )
        clean_by_key = load_clean_label_items_by_key(
            str(DEFAULT_CLEAN_LABELS_DIR.resolve()), focal
        )
        gt_clean_path = DEFAULT_CLEAN_LABELS_DIR / f"ground_truth_{focal}.json"
        if not gt_clean_path.exists():
            st.sidebar.error(f"Clean labels file not found: {gt_clean_path}")
        if not all_cursor_run_folders:
            st.sidebar.warning(
                f"No run folders found under `{CURSOR_PREDICTIONS_DIR}`. "
                "Add subfolders with `predictions_<focal>.json` files to compare models."
            )
        else:
            for run in prediction_runs:
                pred_path = CURSOR_PREDICTIONS_DIR / run / f"predictions_{focal}.json"
                pred_by_run[run] = load_agent_predictions_by_key(
                    str(pred_path.resolve())
                )
                if not pred_path.exists():
                    st.sidebar.warning(
                        f"Missing file for this focal user: `{pred_path.name}` in `{run}`"
                    )

    run_display_labels = _unique_abbrev_labels_for_runs(prediction_runs)

    items = items_by_focal[focal]
    total = len(items)

    # --- Sidebar: per-user progress overview --------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Per-user progress")
    overview_lines: list[str] = []
    for u in FOCAL_USERS:
        u_items = items_by_focal[u]
        labeled = sum(
            1
            for it in u_items
            if str(it["representative_tweet_id"]) in labels_by_user.get(u, {})
        )
        marker = " ◀" if u == focal else ""
        overview_lines.append(f"- **{u}** — {labeled} / {len(u_items)}{marker}")
    st.sidebar.markdown("\n".join(overview_lines))

    # --- Sidebar: current-user counts and filter ----------------------------
    counts = {v: 0 for v in LABEL_VALUES}
    for it in items:
        rec = labels_by_user.get(focal, {}).get(str(it["representative_tweet_id"]))
        if rec and rec.get("label") in counts:
            counts[rec["label"]] += 1
    labeled_total = sum(counts.values())
    unlabeled_total = total - labeled_total

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"`{focal}` progress")
    st.sidebar.progress(
        0 if total == 0 else labeled_total / total,
        text=f"{labeled_total} / {total} labeled",
    )
    st.sidebar.markdown(
        f"- **Endorse:** {counts['endorse']}\n"
        f"- **Disendorse:** {counts['disendorse']}\n"
        f"- **Neutral:** {counts['neutral']}\n"
        f"- **Wrong target:** {counts['wrong_target']}\n"
        f"- **Unlabeled:** {unlabeled_total}"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter")
    filter_choice = st.sidebar.radio(
        "Show",
        options=[
            "All",
            "Unlabeled",
            "Endorse",
            "Disendorse",
            "Neutral",
            "Wrong target",
        ],
        index=0,
    )
    filter_to_label_value = {
        "Endorse": "endorse",
        "Disendorse": "disendorse",
        "Neutral": "neutral",
        "Wrong target": "wrong_target",
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("Paging")
    items_per_page = st.sidebar.slider(
        "Items per page (fewer = faster UI)",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
    )

    # --- Sidebar: save status -----------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Save status")
    save_q, save_errors = get_save_worker()
    pending = save_q.qsize()
    st.sidebar.markdown(
        f"- **Output:** `{output_path}`\n"
        f"- **Pending writes:** {pending}\n"
        f"- **Save errors:** {len(save_errors)}"
    )
    if save_errors:
        with st.sidebar.expander("Show recent save errors"):
            for line in save_errors[-10:]:
                st.code(line)

    # --- Main pane ----------------------------------------------------------
    if review_mode:
        st.info(
            "**Review mode:** **Aggregate metrics** (default tab) has accuracy / Brier skill / "
            "weighted AUROC / F1 / Brier vs marginal baseline / one-vs-rest **ROC** curves "
            "+ a **precision-at-20%-endorsing-prior vs recall** curve "
            "(each scored on that model's own support); "
            "**Browse items** shows the compact **GT / Human / models** table "
            f"({EMOJI_ENDORSING} endorsing · {EMOJI_NEUTRAL} neutral · "
            f"{EMOJI_DISENDORING} disendorsing; {EMOJI_HUMAN_WRONG_TARGET} human wrong "
            "target), then details. Labeling controls are hidden."
        )
        metrics_tab, browse_tab = st.tabs(["Aggregate metrics", "Browse items"])
        with metrics_tab:
            render_review_performance(
                clean_labels_dir=DEFAULT_CLEAN_LABELS_DIR.resolve(),
                predictions_root=CURSOR_PREDICTIONS_DIR.resolve(),
                focal_username=focal,
                visible_ids_by_user=visible_ids_by_user,
                labels_by_user=labels_by_user,
                prediction_runs=prediction_runs,
                run_display_labels=run_display_labels,
            )
        with browse_tab:
            _render_browse_item_page(
                total=total,
                items=items,
                focal=focal,
                bundle=bundle,
                output_path=output_path,
                visible_ids_by_user=visible_ids_by_user,
                filter_choice=filter_choice,
                filter_to_label_value=filter_to_label_value,
                items_per_page=items_per_page,
                labels_by_user=labels_by_user,
                review_mode=review_mode,
                clean_by_key=clean_by_key,
                prediction_runs=prediction_runs,
                run_display_labels=run_display_labels,
                pred_by_run=pred_by_run,
            )
        return

    st.markdown(
        f"Labeling **{total}** targets attributed to **`{focal}`** or to one of "
        f"their **{len(bundle['graph'].get(focal, []))} neighbors**. The order "
        f"is shuffled deterministically per user. Source (focal vs neighbor) "
        f"and the model's predicted direction are hidden — your label drives "
        f"the evaluation.\n\n"
        f"**{CONFIDENCE_PRIOR_NOTE}**"
    )

    if total == 0:
        st.info("No items for this focal user (or empty neighborhood).")
        return

    _render_browse_item_page(
        total=total,
        items=items,
        focal=focal,
        bundle=bundle,
        output_path=output_path,
        visible_ids_by_user=visible_ids_by_user,
        filter_choice=filter_choice,
        filter_to_label_value=filter_to_label_value,
        items_per_page=items_per_page,
        labels_by_user=labels_by_user,
        review_mode=review_mode,
        clean_by_key=clean_by_key,
        prediction_runs=prediction_runs,
        run_display_labels=run_display_labels,
        pred_by_run=pred_by_run,
    )


if __name__ == "__main__":
    main()
