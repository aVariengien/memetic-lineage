#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["tqdm"]
# ///
"""
Run parallel Cursor CLI agents to predict endorsement labels from user archives.

Each agent gets a clean isolated workspace containing:
  - The user's Twitter/X archive (year folders only, no combined file)
  - A stripped version of the ground truth JSON (no labels/context/direction)
  - The endorsement prompt definition

Output predictions are written to data/labels/cursor_agent_predictions/.
"""

import argparse
import asyncio
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ARCHIVES_DIR = SCRIPT_DIR / "data/user_archives"
LABELS_DIR = SCRIPT_DIR / "data/curation_bench_clean_data/clean_labels"
PROMPT_FILE = SCRIPT_DIR / "prompts/endorsement_prompt.md"
OUTPUT_DIR = SCRIPT_DIR / "data/labels/cursor_agent_predictions"

ALL_USERS = [
    "danielbrottman",
    "daniellefong",
    "exgenesis",
    "goblinodds",
    "archived_videos",
]

# Fields in each item that must be stripped before the agent sees the data
ITEM_FIELDS_TO_STRIP = {
    "path_anchor_username",
    "representative_tweet_author",
    "source",
    "ground_truth_label",
    "context",
    "direction",
}

# Top-level fields that reveal label statistics or internal details
TOP_LEVEL_FIELDS_TO_STRIP = {
    "ground_truth_rule",
    "stats",
    "source_files",
    "models",
    "neighbors_in_scope",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_ground_truth(data: dict) -> dict:
    """Return a copy of the ground truth JSON with all label-leaking fields removed."""
    stripped = {
        k: v
        for k, v in data.items()
        if k not in TOP_LEVEL_FIELDS_TO_STRIP
    }
    stripped["items"] = [
        {k: v for k, v in item.items() if k not in ITEM_FIELDS_TO_STRIP}
        for item in data.get("items", [])
    ]
    return stripped


def build_agent_prompt(focal_user: str, n_items: int) -> str:
    return f"""\
You are analyzing the Twitter/X archive of @{focal_user} to predict whether they would endorse, disendorse, or be neutral toward {n_items} named entities during July 2024.

Your primary objective is calibrated probability forecasting. The scoring metric is Brier skill score against a constant class-frequency baseline, so a weak clue should only move probabilities slightly. Top-1 accuracy is secondary.

## Files Available

- `archive/` — @{focal_user}'s Twitter/X archive up to the cutoff. The archive may not contain July 2024 target tweets. Use it only as pre-cutoff evidence about the user's tastes, values, prior endorsements, dislikes, and recurring communities.
- `endorsement_targets.json` — the entities to predict.
- `endorsement_prompt.md` — the exact standard for what counts as endorsement or disendorsement.

Do not use the web. Do not access files outside this workspace.

## Critical Dataset Fact

The target list is not a random set of objects. Many neutral examples are entities endorsed or disendorsed by people in @{focal_user}'s social neighborhood, not by @{focal_user}. Therefore:

- A target being interesting, niche, socially adjacent, or “the kind of thing this user might like” is weak evidence only.
- A target appearing in a nearby community is not enough to predict endorsement.
- Neutral is the default even for plausible-looking targets.

## Base Rates

Start each item from approximately:

- `p_neutral = 0.80`
- `p_endorsing = 0.18`
- `p_disendorsing = 0.02`

Disendorsement is very rare. Keep `p_disendorsing` near `0.01-0.03` unless there is specific evidence that @{focal_user} dislikes this target or closely similar targets.

## Evidence Strength Bands

Use these bands to stay calibrated:

### No evidence or only social-neighborhood plausibility
Use about:
- `p_neutral = 0.80-0.88`
- `p_endorsing = 0.10-0.18`
- `p_disendorsing = 0.01-0.03`

This includes targets that merely match the user's broad interests, friends, subculture, aesthetics, or common topics.

### Weak topical fit
Use about:
- `p_neutral = 0.70-0.80`
- `p_endorsing = 0.18-0.28`
- `p_disendorsing = 0.01-0.04`

Example: the user often discusses Buddhism, films, AI, books, or local events, and the target is in that domain, but you found no target-specific evidence.

### Moderate evidence
Use about:
- `p_neutral = 0.50-0.70`
- `p_endorsing = 0.28-0.48`
- `p_disendorsing = 0.02-0.08`

Example: the target or close aliases appear in the archive, or the user has repeatedly praised a very similar author/project/category, but you do not have a clear prior endorsement of this exact target.

### Strong endorsement evidence
Use about:
- `p_neutral = 0.20-0.45`
- `p_endorsing = 0.50-0.75`
- `p_disendorsing = 0.01-0.06`

Use this only when the archive shows clear positive evaluation, recommendation, repeated use, affection, or durable praise for this exact target or a near-identical alias.

### Strong disendorsement evidence
Use about:
- `p_neutral = 0.35-0.65`
- `p_endorsing = 0.02-0.15`
- `p_disendorsing = 0.25-0.55`

Use this only for clear dislike, warning, complaint, boredom, rejection, or durable negative stance by @{focal_user}. Do not infer disendorsement from absence of interest.

## Workflow

1. Read `endorsement_prompt.md` first. Use its strict definition of endorsement/disendorsement.
2. Read the structure of `endorsement_targets.json`.
3. Build a short profile of @{focal_user} from the archive:
   - repeated interests
   - things they clearly recommend
   - things they clearly dislike
   - how strong their language tends to be
   - domains where they often post thin reactions that should not count as durable endorsement
4. For each target:
   - Search exact `target_entity`, `longer_name`, obvious aliases, URLs, author names, and title fragments.
   - Treat exact or alias evidence as much stronger than broad topic fit.
   - If nothing is found, use the base rate or weak topical-fit band, not a high-confidence guess.
   - Ask: “Would I expect @{focal_user}, not someone adjacent to them, to still recommend or warn against this named thing months later?”
5. Before writing the final JSON, do a calibration pass:
   - Most rows should remain neutral-top.
   - Many neutral-looking but plausible targets should still have `p_neutral >= 0.75`.
   - Do not assign `p_endorsing > 0.35` from vibe/domain fit alone.
   - Do not assign `p_disendorsing > 0.08` without concrete negative evidence.
   - If a probability feels like a guess, shrink it back toward `0.80 / 0.18 / 0.02`.

## Output

Write `predictions.json` in the workspace root with exactly this structure:

```json
{
  "focal_user": "{focal_user}",
  "generated_at": "<ISO 8601 timestamp>",
  "predictions": [
    {
      "representative_tweet_id": 1803432686528213472,
      "target_entity": "Hank Green video on media literacy",
      "reasoning": "One sentence naming the evidence strength: exact archive evidence, close analogue, weak topic fit, or no evidence.",
      "p_neutral": 0.80,
      "p_endorsing": 0.18,
      "p_disendorsing": 0.02
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Per-agent runner
# ---------------------------------------------------------------------------


async def run_agent(
    username: str,
    workspace: Path,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    startup_lock: asyncio.Lock,
    progress: tqdm,
    model: str | None = None,
) -> dict:
    """Set up workspace and run the Cursor agent for one user. Returns a result dict."""
    async with semaphore:
        try:
            # ---- Workspace setup ----------------------------------------
            archive_src = ARCHIVES_DIR / username
            archive_dst = workspace / "archive"
            archive_dst.mkdir()

            # Copy each year folder (skip flat files like all_tweets_*.md)
            for item in sorted(archive_src.iterdir()):
                if item.is_dir() and item.name.isdigit():
                    shutil.copytree(item, archive_dst / item.name)

            # Strip labels and write endorsement_targets.json
            gt_file = LABELS_DIR / f"ground_truth_{username}.json"
            with open(gt_file) as f:
                gt_data = json.load(f)
            stripped = strip_ground_truth(gt_data)
            n_items = len(stripped.get("items", []))
            with open(workspace / "endorsement_targets.json", "w") as f:
                json.dump(stripped, f, indent=2)

            # Copy endorsement prompt
            shutil.copy(PROMPT_FILE, workspace / "endorsement_prompt.md")

            # ---- Build prompt -------------------------------------------
            prompt = build_agent_prompt(username, n_items)

            # ---- Run agent ----------------------------------------------
            cmd = [
                "agent",
                "-p", prompt,
                "--print",
                "--output-format", "stream-json",
                "--yolo",
                "--trust",
                "--sandbox", "enabled",
                "--workspace", str(workspace),
            ]
            if model:
                cmd += ["--model", model]

            # Serialize agent startups: the CLI writes ~/.cursor/cli-config.json
            # atomically (tmp → rename) at launch; concurrent startups race on
            # that rename and one of them gets ENOENT. Hold the lock just long
            # enough for the config write to settle, then release so other agents
            # can start while this one keeps running in parallel.
            async with startup_lock:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace),
                )
                await asyncio.sleep(4.0)  # wait for cli-config.json write to complete

            stdout, stderr = await proc.communicate()

            # ---- Always write logs --------------------------------------
            log_file = output_dir / f"agent_{username}.log"
            with open(log_file, "w", errors="replace") as f:
                f.write(f"=== stdout ===\n")
                f.write(stdout.decode(errors="replace"))
                if stderr:
                    f.write(f"\n=== stderr ===\n")
                    f.write(stderr.decode(errors="replace"))

            if proc.returncode != 0:
                err_msg = stderr.decode(errors="replace")[:800]
                tqdm.write(f"[ERROR] {username}: agent exited with code {proc.returncode}")
                tqdm.write(f"[ERROR] {username}: {err_msg}")
                tqdm.write(f"[ERROR] {username}: full log → {log_file.name}")
                progress.update(1)
                return {"username": username, "status": "error", "message": f"exit code {proc.returncode}"}

            # ---- Collect output -----------------------------------------
            pred_file = workspace / "predictions.json"
            if not pred_file.exists():
                tqdm.write(f"[ERROR] {username}: predictions.json not found after agent run")
                tqdm.write(f"[ERROR] {username}: full log → {log_file.name}")
                progress.update(1)
                return {"username": username, "status": "error", "message": "predictions.json missing"}

            out_file = output_dir / f"predictions_{username}.json"
            shutil.copy(pred_file, out_file)
            tqdm.write(f"[OK]    {username}: predictions written → {out_file.name}")
            progress.update(1)
            return {"username": username, "status": "success", "output": str(out_file)}

        except Exception as exc:
            tqdm.write(f"[ERROR] {username}: {exc}")
            progress.update(1)
            return {"username": username, "status": "error", "message": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run parallel Cursor CLI agents for endorsement prediction."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of agents to run in parallel (default: 5).",
    )
    parser.add_argument(
        "--users",
        nargs="+",
        default=ALL_USERS,
        metavar="USERNAME",
        help="Subset of users to process (default: all 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for output prediction files.",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Do not delete temporary workspaces after the run (useful for debugging).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to pass to the agent CLI (e.g. claude-sonnet-4, gpt-4o).",
    )
    args = parser.parse_args()

    # Validate users
    unknown = set(args.users) - set(ALL_USERS)
    if unknown:
        parser.error(f"Unknown user(s): {', '.join(sorted(unknown))}. Valid: {', '.join(ALL_USERS)}")

    # Build a per-run output subfolder: <output_dir>/<model>_<timestamp>/
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    model_slug = (args.model or "default").replace("/", "-").replace(":", "-")
    run_output_dir = args.output_dir / f"{model_slug}_{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Temp root for all workspaces this run
    tmp_root = Path(tempfile.mkdtemp(prefix=f"cursor_endorse_{run_id}_"))
    print(f"Workspace root : {tmp_root}")
    print(f"Output dir     : {run_output_dir}")
    print(f"Users          : {', '.join(args.users)}")
    print(f"Concurrency    : {args.concurrency}")
    print()

    semaphore = asyncio.Semaphore(args.concurrency)
    startup_lock = asyncio.Lock()

    with tqdm(total=len(args.users), desc="Agents", unit="user") as progress:
        tasks = []
        for username in args.users:
            workspace = tmp_root / username
            workspace.mkdir()
            tasks.append(
                run_agent(username, workspace, run_output_dir, semaphore, startup_lock,
                          progress, model=args.model)
            )
        results = await asyncio.gather(*tasks)

    # Summary
    print("\n=== Summary ===")
    ok = [r for r in results if r["status"] == "success"]
    err = [r for r in results if r["status"] != "success"]
    for r in ok:
        print(f"  ✓  {r['username']:20s} → {Path(r['output']).name}")
    for r in err:
        print(f"  ✗  {r['username']:20s}   {r.get('message', '')}")
    print(f"\n{len(ok)}/{len(results)} agents completed successfully.")

    if args.keep_workspaces:
        print(f"\nWorkspaces kept at: {tmp_root}")
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\nTemp workspaces cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())
