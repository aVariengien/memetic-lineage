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
You are analyzing the Twitter/X archive of @{focal_user} (data ends 2024-07-01) to predict the label for {n_items} (representative_tweet_id, target_entity) pairs from July 2024.

You are in autonomous agent mode — no user is available to answer questions or validate intermediate decisions. Carry out the task to completion in one shot.

## How the labels are produced — read carefully, this IS the task

Each item points at a `representative_tweet_id` that sits inside a July 2024 conversation path. An external LLM read that path and decided someone in it endorsed (or disendorsed) the target entity.

The ground-truth label is determined entirely by **who authored that representative tweet**:

- If `@{focal_user}` themselves authored the representative tweet → label is `endorsing` or `disendorsing` (the direction the LLM extracted).
- If anyone else (a neighbor in @{focal_user}'s social graph) authored it → label is `neutral`.

You DO NOT see who authored the representative tweet — that field is stripped. Your real job is to estimate, from the archive alone, the probability that @{focal_user} is the author.

This is **not a preference task**. @{focal_user} may genuinely love a target, but if a neighbor wrote the rep tweet about it, the label is `neutral`. Likewise, @{focal_user} may have authored a brand-new endorsement of something never mentioned in their archive — the label is still `endorsing`. Do not confuse "user likes X" with "user authored the rep tweet about X".

## Files Available

- `archive/` — @{focal_user}'s pre-cutoff conversation trees (year/month folders). Use it to learn the user's voice, recurring obsessions, idiosyncratic vocabulary, recommendation habits, and the cast of accounts they typically talk to / amplify.
- `endorsement_targets.json` — the items to predict, with fields `representative_tweet_id`, `target_entity`, `longer_name`, `url`, `path_anchor_tweet_id`.
- `endorsement_prompt.md` — the standard the external LLM used to decide what qualifies as endorsement / disendorsement.

The archive ends before the prediction window. This is by design: the benchmark is an **extrapolation** problem. The archive cannot directly tell you what @{focal_user} tweeted in July 2024. It can only tell you what kind of person they are, what they originate vs amplify, and whose voice this target most likely belongs to.

Do not use the web. Do not access files outside this workspace.

## Mental decomposition

For each item, decompose:

  p_focal_authored ≡ P(the rep_tweet was authored by @{focal_user})
  p_endorsing_given_focal   ≈ 0.90    (most extracted directions are endorsing)
  p_disendorsing_given_focal ≈ 0.10

Then:
  p_endorsing    = p_focal_authored * p_endorsing_given_focal
  p_disendorsing = p_focal_authored * p_disendorsing_given_focal
  p_neutral      = 1 - p_endorsing - p_disendorsing

If there is item-specific evidence that @{focal_user} dislikes this target (warning, complaint, durable rejection of it or a close category), shift `p_disendorsing_given_focal` upward — typically 0.4 to 0.7 — and recompute. Otherwise keep it near 0.10.

This decomposition is mandatory thinking even if you only output the three probabilities. It forces you to focus on the authorship question, which is what the label tracks.

## Base rate

Across the dataset, roughly 80% of items are non-focal-authored. With no information:

  p_focal_authored ≈ 0.20
  → p_neutral ≈ 0.80, p_endorsing ≈ 0.18, p_disendorsing ≈ 0.02

The Brier skill score is computed against exactly this baseline. Predicting the base rate everywhere gives a skill score of zero. To beat zero you must commit to deviations where you have signal. **Do not retreat to the base rate "to be safe"** — that is precisely the zero-skill policy.

## What is signal for `p_focal_authored`?

### Strong push toward focal-authored (p_focal_authored 0.45 – 0.75)
- The target appears **by name or close alias** in @{focal_user}'s archive as something they themselves brought up evaluatively ("I love X", "X is great", "you should read X"). Not "X was mentioned in a thread they replied to" — they originated the evaluation.
- The target sits inside @{focal_user}'s **idiosyncratic voice**: a niche / vocabulary / aesthetic they themselves bring up unprompted across the archive, not just things their tribe talks about.
- The user has a documented habit of producing exactly this kind of recommendation (e.g. a years-long pattern of rating reality shows, naming programming tools, recommending Buddhist books, etc.) and the target fits that habit.
- The target is so specific to the user's known niche that a neighbor would be unlikely to bring it up in @{focal_user}'s mentions.

### Weak update only (p_focal_authored 0.18 – 0.30)
- The target is broadly popular within @{focal_user}'s social tribe (e.g. canonical SSC/ACX posts, Claude/Anthropic, common LessWrong / TPOT / tech-twitter references). These are *exactly* the things neighbors quote at the user — so high tribe-fit is **not** evidence of focal authorship.
- The target is in the user's broad domain (films, books, AI, etc.) but you have no archive evidence of the user themselves naming it or a close analogue.
- The target appears in the archive only inside RTs or replies-to others, never as something the user originated.

### Counter-signal — push toward neighbor-authored (p_focal_authored 0.05 – 0.18)
- You can name a specific neighbor whose niche/voice this target obviously belongs to, more than to @{focal_user}.
- The target is the kind of viral / community content the user typically *receives* rather than originates.

## Workflow

1. Read `endorsement_prompt.md` and skim `endorsement_targets.json` so you know the shape of the task.
2. Build a profile of @{focal_user} centered on **authorship-relevant** features:
   - What do they originate vs amplify? Sample standalone tweets vs replies / RTs / quotes.
   - Their recurring "I recommend / I avoid" patterns — categories, confidence level, phrasings.
   - Their close associates and the things those associates are known for. When a target is in a specific neighbor's niche, that raises P(neighbor-authored).
   - The vocabulary and tone distinctively theirs.
3. For each target, ask in order:
   a. Does this target appear by name or close alias in @{focal_user}'s archive as something they brought up evaluatively? → strong update toward focal-authored.
   b. Is the target obviously in a specific neighbor's niche more than @{focal_user}'s? → weak update toward neighbor-authored.
   c. Is this just "the kind of thing the tribe likes"? → no update — stay near base rate.
   d. Is there explicit negative evidence for this target or its category? → raise `p_disendorsing_given_focal`.
4. Emit a per-item triple with concrete, **item-specific** reasoning that names the actual signal used.

## Calibration rules

- Most items remain neutral-top — that matches the 80% base rate. Don't force endorsement-top calls unless real signal supports it.
- When the archive shows clear, durable, repeated focal-user endorsement of the target by name, `p_endorsing` can go up to ~0.65 (i.e. `p_focal_authored ≈ 0.7`). Higher requires unambiguous evidence.
- When the archive shows clear focal-user disendorsement of the target or a tightly-related category, `p_disendorsing` can go up to ~0.5.
- High topic-fit alone is **not** strong signal — many such items are neighbor-authored (this is exactly the failure mode the benchmark is built to expose). Cap `p_endorsing` at ~0.30 when you only have topic fit.
- **No two reasoning sentences should be identical.** If you find yourself writing "no archive evidence; default base rate" for many items in a row, you have stopped doing the task — go back and apply the authorship checklist item-by-item.

## Output

Write `predictions.json` in the workspace root with exactly this structure:

```json
{{
  "focal_user": "{focal_user}",
  "generated_at": "<ISO 8601 timestamp>",
  "predictions": [
    {{
      "representative_tweet_id": 1803432686528213472,
      "target_entity": "Hank Green video on media literacy",
      "reasoning": "One sentence naming the AUTHORSHIP signal: archive-by-name match / specific-neighbor-niche / tribe-only fit / explicit negative evidence / etc.",
      "p_neutral": 0.80,
      "p_endorsing": 0.18,
      "p_disendorsing": 0.02
    }}
  ]
}}
```

Rules:
- Include every entry from `endorsement_targets.json`, in the same order.
- Probabilities must sum to 1.0.
- Reasoning is one sentence, no line breaks, focused on the *authorship* signal — not the user's general preference.
- Do not expose long chain-of-thought; only the concise authorship-evidence summary.
- Do not ask the user for validation. Carry the task to the end.
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
    parser.add_argument(
        "--model-suffix",
        default="",
        metavar="TEXT",
        help=(
            "Optional label for the output subfolder only (not passed to the agent). "
            "If set, the run folder is <model>-<suffix>_<timestamp> instead of "
            "<model>_<timestamp> (default: empty)."
        ),
    )
    args = parser.parse_args()

    # Validate users
    unknown = set(args.users) - set(ALL_USERS)
    if unknown:
        parser.error(f"Unknown user(s): {', '.join(sorted(unknown))}. Valid: {', '.join(ALL_USERS)}")

    # Build a per-run output subfolder:
    #   No suffix:  <output_dir>/<model>_<timestamp>/
    #   With suffix: <output_dir>/<model>-<suffix>_<timestamp>/
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    model_slug = (args.model or "default").replace("/", "-").replace(":", "-")
    if (s := args.model_suffix.strip()):
        suffix_safe = s.replace("/", "-").replace(":", "-")
        run_folder = f"{model_slug}-{suffix_safe}_{run_id}"
    else:
        run_folder = f"{model_slug}_{run_id}"
    run_output_dir = args.output_dir / run_folder
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Temp root for all workspaces this run
    tmp_root = Path(tempfile.mkdtemp(prefix=f"cursor_endorse_{run_id}_"))
    print(f"Workspace root : {tmp_root}")
    print(f"Output dir     : {run_output_dir}")
    print(f"Users          : {', '.join(args.users)}")
    print(f"Concurrency    : {args.concurrency}")
    if args.model_suffix.strip():
        print(f"Model suffix   : {args.model_suffix.strip()!r}")
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
