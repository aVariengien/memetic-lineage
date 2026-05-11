#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "scikit-learn>=1.3",
#   "pandas>=2.0",
#   "matplotlib>=3.8",
#   "numpy>=1.26",
# ]
# ///
# %%
"""VS-code style notebook (`#%%` cells): bar plots of model performance.

Three metrics per slice (Brier skill score, weighted AUROC, accuracy) drawn for
two slices of cursor agent runs:
  - non-calib runs (everything that doesn't contain "calib" in its display name),
    plus Human (soft probs from `predictions_top5_jul2024_human_probs.json`)
  - calib runs only (Human has no calibration variant, so it's excluded)

All eval logic mirrors `06_label_endorsement_targets_ui.py`:
  - Composite eval key `(representative_tweet_id, target_entity)` so multi-target
    tweets aren't silently collapsed (see source file lines 192-197).
  - GT from `clean_labels/ground_truth_<focal>.json` (`endorsing/disendorsing/neutral`).
  - Human classification = `label` from `predictions_top5_jul2024_human_probs.json`
    (`wrong_target` excluded). Human probs = `prob_endorse/prob_disendorse/prob_neutral`
    from the same file (rows missing the triple are dropped from Human).
  - Agent argmax = argmax(`p_endorsing`, `p_disendorsing`, `p_neutral`); same probs
    drive Brier and weighted AUROC.
  - Display names = `_RUN_FOLDER_TS_SUFFIX` strip + `-1/-2/...` on collisions.

Run with:
  uv run curation-bench/09_metrics_bar_plots.py
or open the file in VS Code / Cursor and run cells with the Python extension.
"""

from __future__ import annotations

# %% Imports + static config (same as 06_*.py)
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

CURSOR_PREDICTIONS_DIR = DATA_DIR / "labels" / "cursor_agent_predictions"
HUMAN_PROBS_JSON = DATA_DIR / "labels" / "predictions_top5_jul2024_human_probs.json"
CLEAN_LABELS_DIR = DATA_DIR / "curation_bench_clean_data" / "clean_labels"

FOCAL_USERS: list[str] = [
    "goblinodds",
    "daniellefong",
    "exgenesis",
    "archived_videos",
    "danielbrottman",
]

GT_CLASSES: tuple[str, ...] = ("endorsing", "disendorsing", "neutral")

# Dumb-baseline accuracy for the accuracy plot (share of `neutral` GT rows in the
# pooled 5-focal eval set). Provided by the user; not recomputed here.
ACCURACY_NEUTRAL_BASELINE: float = 0.8118

# Random-classifier baseline for weighted AUROC. AUROC is class-balance invariant:
# each binary OvR slice has random AUROC = 0.5, so any weighted average is also 0.5.
WEIGHTED_AUROC_RANDOM_BASELINE: float = 0.5

EvalKey = tuple[int, str]


def _eval_key(rec: dict[str, Any]) -> EvalKey | None:
    tid = rec.get("representative_tweet_id")
    tgt = rec.get("target_entity")
    if tid is None or tgt is None:
        return None
    return (int(tid), str(tgt))


def _normalize_gt_label(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s in GT_CLASSES else None


def _human_label_to_gt_class(label: str | None) -> str | None:
    """Map persisted human label → 3-way GT class. `wrong_target` excluded."""
    if not label:
        return None
    low = label.strip().lower()
    if low == "wrong_target":
        return None
    return {
        "endorse": "endorsing",
        "disendorse": "disendorsing",
        "neutral": "neutral",
    }.get(low)


def _pred_prob_vec_ordered(pred: dict[str, Any]) -> list[float]:
    """Order = GT_CLASSES (endorsing, disendorsing, neutral)."""
    return [
        float(pred.get("p_endorsing") or 0.0),
        float(pred.get("p_disendorsing") or 0.0),
        float(pred.get("p_neutral") or 0.0),
    ]


def _human_soft_prob_vec_ordered(rec: dict[str, Any]) -> list[float] | None:
    pe = rec.get("prob_endorse")
    pd_ = rec.get("prob_disendorse")
    pn = rec.get("prob_neutral")
    if pe is None or pd_ is None or pn is None:
        return None
    return [float(pe), float(pd_), float(pn)]


def _agent_argmax_label(pred: dict[str, Any]) -> str:
    scores = [
        ("neutral", float(pred.get("p_neutral") or 0.0)),
        ("endorsing", float(pred.get("p_endorsing") or 0.0)),
        ("disendorsing", float(pred.get("p_disendorsing") or 0.0)),
    ]
    return max(scores, key=lambda x: x[1])[0]


_RUN_FOLDER_TS_SUFFIX = re.compile(r"_\d{8}T\d{6}$")


def _abbrev_run_folder(name: str, max_len: int = 40) -> str:
    base = _RUN_FOLDER_TS_SUFFIX.sub("", name).strip() or name
    return base if len(base) <= max_len else base[: max_len - 1] + "…"


def _unique_abbrev_labels_for_runs(runs: list[str]) -> list[str]:
    """Display names: strip timestamps, suffix -1/-2/… on base-name collisions."""
    abbrevs = [_abbrev_run_folder(r) for r in runs]
    counts: dict[str, int] = {}
    for a in abbrevs:
        counts[a] = counts.get(a, 0) + 1
    seen: dict[str, int] = {}
    out: list[str] = []
    for a in abbrevs:
        if counts[a] == 1:
            out.append(a)
        else:
            seen[a] = seen.get(a, 0) + 1
            out.append(f"{a}-{seen[a]}")
    return out


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# %% Load human soft probs (composite-key, by focal user)
def load_human_probs_by_focal_user(
    json_path: Path,
) -> dict[str, dict[EvalKey, dict[str, Any]]]:
    if not json_path.is_file():
        return {}
    data = _load_json(json_path)
    out: dict[str, dict[EvalKey, dict[str, Any]]] = {}
    for user, block in (data.get("users") or {}).items():
        m: dict[EvalKey, dict[str, Any]] = {}
        for rec in block.get("labels") or []:
            key = _eval_key(rec)
            if key is not None:
                m[key] = rec
        out[user] = m
    return out


human_probs_by_focal = load_human_probs_by_focal_user(HUMAN_PROBS_JSON)
print(
    f"Loaded human probs for {sum(len(v) for v in human_probs_by_focal.values())} "
    f"(focal, tid, target) rows across {len(human_probs_by_focal)} focal users."
)


# %% Load clean labels (composite key) and build pooled eval rows
def load_clean_label_items_by_key(
    clean_labels_dir: Path, focal: str
) -> dict[EvalKey, dict[str, Any]]:
    path = clean_labels_dir / f"ground_truth_{focal}.json"
    if not path.exists():
        return {}
    data = _load_json(path)
    out: dict[EvalKey, dict[str, Any]] = {}
    for item in data.get("items") or []:
        key = _eval_key(item)
        if key is not None:
            out[key] = item
    return out


def build_pooled_eval_rows() -> list[tuple[str, int, str, str]]:
    """(focal, tid, target_entity, gt_class) for every clean_labels row with usable GT."""
    rows: list[tuple[str, int, str, str]] = []
    for focal in FOCAL_USERS:
        by_key = load_clean_label_items_by_key(CLEAN_LABELS_DIR, focal)
        for (tid, target), rec in by_key.items():
            gt = _normalize_gt_label(rec.get("ground_truth_label"))
            if gt is None:
                continue
            rows.append((focal, tid, target, gt))
    return rows


eval_rows = build_pooled_eval_rows()
n_total = len(eval_rows)
class_counts = {c: sum(1 for _f, _t, _tg, g in eval_rows if g == c) for c in GT_CLASSES}
print(f"Pooled eval rows (5 focal users): n = {n_total}")
print(
    "Class frequencies:  "
    + ", ".join(
        f"{c} = {class_counts[c]} ({class_counts[c] / n_total * 100:.2f}%)"
        for c in GT_CLASSES
    )
)


# %% Discover run folders + display names
def list_run_folders(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        (p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")),
        reverse=True,
    )


prediction_runs = list_run_folders(CURSOR_PREDICTIONS_DIR)
run_display_labels = _unique_abbrev_labels_for_runs(prediction_runs)
print(f"Discovered {len(prediction_runs)} run folder(s):")
for run, disp in zip(prediction_runs, run_display_labels, strict=True):
    print(f"  - {disp}   <-  {run}")


# %% Per-run prediction loaders
def load_agent_predictions_by_key(
    predictions_json_path: Path,
) -> dict[EvalKey, dict[str, Any]]:
    if not predictions_json_path.is_file():
        return {}
    data = _load_json(predictions_json_path)
    out: dict[EvalKey, dict[str, Any]] = {}
    for pred in data.get("predictions") or []:
        key = _eval_key(pred)
        if key is not None:
            out[key] = pred
    return out


def load_agent_preds_for_run(run: str) -> dict[str, dict[EvalKey, dict[str, Any]]]:
    out: dict[str, dict[EvalKey, dict[str, Any]]] = {}
    for fu in FOCAL_USERS:
        out[fu] = load_agent_predictions_by_key(
            CURSOR_PREDICTIONS_DIR / run / f"predictions_{fu}.json"
        )
    return out


# %% Metric computation (mirrors `_metrics_from_predictions` in 06_*.py)
def _multiclass_mean_brier(
    y_true_class: list[str], prob_vecs: list[list[float]]
) -> float:
    n = len(y_true_class)
    if n == 0:
        return float("nan")
    total = 0.0
    for yt, pv in zip(y_true_class, prob_vecs, strict=False):
        i_true = GT_CLASSES.index(yt)
        oh = [1.0 if k == i_true else 0.0 for k in range(3)]
        total += sum((pv[k] - oh[k]) ** 2 for k in range(3)) / 3.0
    return total / n


def _marginal_mean_brier(y_true_class: list[str]) -> float:
    n = len(y_true_class)
    if n == 0:
        return float("nan")
    freq = {c: 0 for c in GT_CLASSES}
    for yt in y_true_class:
        if yt in freq:
            freq[yt] += 1
    bar = [freq[c] / n for c in GT_CLASSES]
    total = 0.0
    for yt in y_true_class:
        i_true = GT_CLASSES.index(yt)
        oh = [1.0 if k == i_true else 0.0 for k in range(3)]
        total += sum((bar[k] - oh[k]) ** 2 for k in range(3)) / 3.0
    return total / n


def _brier_skill(brier_model: float, brier_marginal: float) -> float:
    if (
        brier_marginal is None
        or brier_marginal != brier_marginal
        or brier_marginal <= 1e-15
    ):
        return float("nan")
    if brier_model is None or brier_model != brier_model:
        return float("nan")
    return 1.0 - brier_model / brier_marginal


def _weighted_auroc(
    y_true: list[str], prob_vecs: list[list[float]]
) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    lbls = list(GT_CLASSES)
    # sklearn requires `labels` in lex order for OvR; remap columns accordingly.
    lbls_ovr = sorted(lbls)
    idx_perm = [lbls.index(c) for c in lbls_ovr]
    y_score_mx = np.asarray(prob_vecs, dtype=np.float64)
    if y_score_mx.shape != (n, len(lbls)):
        return float("nan")
    y_ovr = y_score_mx[:, idx_perm]
    try:
        return float(
            roc_auc_score(
                y_true, y_ovr, multi_class="ovr", average="weighted", labels=lbls_ovr
            )
        )
    except ValueError:
        return float("nan")


def metrics_for_predictions(
    y_true: list[str], y_pred: list[str], prob_vecs: list[list[float]]
) -> dict[str, float]:
    if not y_true:
        return {
            "accuracy": float("nan"),
            "weighted_auroc": float("nan"),
            "brier": float("nan"),
            "brier_marginal": float("nan"),
            "brier_skill": float("nan"),
            "support": 0,
        }
    bri = _multiclass_mean_brier(y_true, prob_vecs)
    bmar = _marginal_mean_brier(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_auroc": _weighted_auroc(y_true, prob_vecs),
        "brier": bri,
        "brier_marginal": bmar,
        "brier_skill": _brier_skill(bri, bmar),
        "support": len(y_true),
    }


# %% Compute metrics for Human + every run on the pooled eval slice
def collect_human_predictions(
    rows: list[tuple[str, int, str, str]],
    human_by_focal: dict[str, dict[EvalKey, dict[str, Any]]],
) -> tuple[list[str], list[str], list[list[float]]]:
    yt: list[str] = []
    yp: list[str] = []
    pr: list[list[float]] = []
    for focal, tid, target, gt in rows:
        rec = human_by_focal.get(focal, {}).get((int(tid), target))
        if rec is None:
            continue
        pv = _human_soft_prob_vec_ordered(rec)
        if pv is None:
            continue
        hc = _human_label_to_gt_class(rec.get("label"))
        if hc is None:
            continue
        yt.append(gt)
        yp.append(hc)
        pr.append(pv)
    return yt, yp, pr


def collect_agent_predictions(
    rows: list[tuple[str, int, str, str]],
    preds_by_focal: dict[str, dict[EvalKey, dict[str, Any]]],
) -> tuple[list[str], list[str], list[list[float]]]:
    yt: list[str] = []
    yp: list[str] = []
    pr: list[list[float]] = []
    for focal, tid, target, gt in rows:
        rec = preds_by_focal.get(focal, {}).get((int(tid), target))
        if rec is None:
            continue
        yt.append(gt)
        yp.append(_agent_argmax_label(rec))
        pr.append(_pred_prob_vec_ordered(rec))
    return yt, yp, pr


metric_records: list[dict[str, Any]] = []

yt_h, yp_h, pr_h = collect_human_predictions(eval_rows, human_probs_by_focal)
if yt_h:
    m = metrics_for_predictions(yt_h, yp_h, pr_h)
    m["model"] = "Human"
    m["is_calib"] = False
    metric_records.append(m)
    print(f"Human: support = {m['support']}")
else:
    print("Human: no usable rows (file missing or no overlap with `prob_*` triples).")

for run, disp in zip(prediction_runs, run_display_labels, strict=True):
    preds = load_agent_preds_for_run(run)
    yt, yp, pr = collect_agent_predictions(eval_rows, preds)
    m = metrics_for_predictions(yt, yp, pr)
    m["model"] = disp
    m["is_calib"] = "calib" in disp.lower()
    if m["support"] == 0:
        print(f"  skipping (support 0): {disp}")
        continue
    metric_records.append(m)

metrics_df = pd.DataFrame(metric_records)
print()
print(metrics_df.to_string(index=False))


# %% Bar plot helper
def plot_metric_bars(
    df: pd.DataFrame,
    *,
    title: str,
    pin_human_first: bool,
) -> None:
    """One figure with 3 stacked horizontal bar subplots (Brier skill, weighted AUROC, accuracy)."""
    if df.empty:
        print(f"[skip] {title}: no models")
        return

    # Sort: optionally pin Human first, then alphabetical by display name.
    ordered_models: list[str] = []
    if pin_human_first and "Human" in df["model"].values:
        ordered_models.append("Human")
    rest = sorted(m for m in df["model"] if m != "Human")
    ordered_models.extend(rest)
    df_ord = df.set_index("model").reindex(ordered_models).reset_index()

    n = len(df_ord)
    per_plot_h = max(2.4, 0.36 * n + 1.4)
    fig, axes = plt.subplots(
        3, 1, figsize=(9.5, per_plot_h * 3), constrained_layout=True
    )
    fig.suptitle(title, fontsize=13, weight="bold")

    metric_specs = [
        {
            "col": "brier_skill",
            "ax": axes[0],
            "title": "Brier skill score (vs marginal-frequency baseline)",
            "xlabel": "skill (1 − Brier/Brier_marginal)",
            "fmt": lambda v: f"{v:+.3f}" if v == v else "—",
            "baselines": [(0.0, "marginal baseline (skill = 0)")],
            "xlim": None,  # auto, with margin
        },
        {
            "col": "weighted_auroc",
            "ax": axes[1],
            "title": "Weighted AUROC (one-vs-rest, GT-class weighted)",
            "xlabel": "weighted AUROC",
            "fmt": lambda v: f"{v:.3f}" if v == v else "—",
            "baselines": [
                (
                    WEIGHTED_AUROC_RANDOM_BASELINE,
                    f"random baseline ({WEIGHTED_AUROC_RANDOM_BASELINE:.2f})",
                )
            ],
            "xlim": (0.4, 1.0),
        },
        {
            "col": "accuracy",
            "ax": axes[2],
            "title": "Accuracy (argmax vs ground truth)",
            "xlabel": "accuracy",
            "fmt": lambda v: f"{v * 100:.2f}%" if v == v else "—",
            "baselines": [
                (
                    ACCURACY_NEUTRAL_BASELINE,
                    f"always-`neutral` baseline ({ACCURACY_NEUTRAL_BASELINE * 100:.2f}%)",
                )
            ],
            "xlim": (0.0, 1.0),
        },
    ]

    y_pos = np.arange(n)
    for spec in metric_specs:
        ax = spec["ax"]
        vals = df_ord[spec["col"]].to_numpy(dtype=float)
        bars = ax.barh(y_pos, np.where(np.isnan(vals), 0.0, vals), color="steelblue")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_ord["model"].tolist(), fontsize=9)
        ax.invert_yaxis()  # first model at top
        ax.set_title(spec["title"], fontsize=10)
        ax.set_xlabel(spec["xlabel"], fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        for baseline_x, label in spec["baselines"]:
            ax.axvline(
                baseline_x,
                color="crimson",
                linestyle=":",
                linewidth=1.2,
                label=label,
            )
        ax.legend(loc="lower right", fontsize=7, frameon=False)

        if spec["xlim"] is not None:
            ax.set_xlim(*spec["xlim"])
        else:
            finite = vals[np.isfinite(vals)]
            if finite.size:
                lo = min(0.0, float(finite.min()))
                hi = max(0.0, float(finite.max()))
                pad = 0.05 * max(0.05, hi - lo)
                ax.set_xlim(lo - pad, hi + pad)

        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                "  " + spec["fmt"](v),
                va="center",
                ha="left",
                fontsize=7.5,
                color="0.2",
            )

    plt.show()


# %% Plot: non-calib runs (incl. Human)
non_calib_df = metrics_df[~metrics_df["is_calib"]].copy()
print(
    f"Non-calib slice: {len(non_calib_df)} rows  "
    f"({sorted(non_calib_df['model'].tolist())})"
)
plot_metric_bars(
    non_calib_df,
    title=f"Non-calib runs + Human  —  pooled 5 focal users (n_eval = {n_total})",
    pin_human_first=True,
)


# %% Plot: calib runs only
calib_df = metrics_df[metrics_df["is_calib"]].copy()
print(
    f"Calib slice: {len(calib_df)} rows  "
    f"({sorted(calib_df['model'].tolist())})"
)
plot_metric_bars(
    calib_df,
    title=f"Calib runs  —  pooled 5 focal users (n_eval = {n_total})",
    pin_human_first=False,
)

# %%
