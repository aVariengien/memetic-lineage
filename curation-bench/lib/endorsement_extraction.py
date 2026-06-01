"""Endorsement-extraction orchestrator.

Single source of truth for the pipeline used by scripts 03 and 04
(and any future cohort/window variant). Given a list of candidate tweets
and a few output knobs, this module:

  1. Collapses candidates to unique maximal reply paths.
  2. Renders each path in parallel (resolves t.co URLs).
  3. Writes the rendered-paths JSON.
  4. Batches paths through the LLM in parallel for endorsement extraction.
     - Per-row validation failures retry the whole batch (see
       ``extract_endorsement_targets_batch``).
     - If a batch still fails after all retries, it is logged and skipped
       (the rest of the run continues).
  5. Enriches each target with its source path's anchor tweet/username.
  6. Writes the endorsement-targets JSON.

Callers supply only:
  * the candidate tweets (already filtered),
  * the output paths,
  * arbitrary metadata to merge into each output JSON,
  * model + concurrency knobs.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .curation_primitives import (
    collapse_to_maximal_unique_paths,
    create_client,
    extract_endorsement_targets_batch,
    infer_provider,
    now_utc_iso,
    persist_url_resolution_cache,
    render_sampled_item,
)


@dataclass
class PipelineResult:
    """Summary of a pipeline run, returned to the caller for printing/tests."""

    paths_path: Path
    targets_path: Path | None
    candidate_count: int
    unique_path_count: int
    rendered_path_count: int
    sampled_items: list[dict[str, Any]]
    enriched_results: list[dict[str, Any]] = field(default_factory=list)
    target_counts: Counter = field(default_factory=Counter)
    failed_batches: list[dict[str, Any]] = field(default_factory=list)


def _render_paths_in_parallel(
    unique_paths: list[dict[str, Any]],
    tweet_dict: Any,
    conversation_trees: Any,
    path_concurrency: int,
) -> list[dict[str, Any]]:
    """Render every unique path concurrently. Returns items in input order."""
    rendered_items: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=path_concurrency) as executor:
        futures = {
            executor.submit(render_sampled_item, item, tweet_dict, conversation_trees): index
            for index, item in enumerate(unique_paths)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Rendering paths",
        ):
            index = futures[future]
            try:
                rendered_item = future.result()
            except Exception as exc:
                tqdm.write(f"[error] render path index {index} failed: {exc}")
                continue
            if rendered_item is not None:
                rendered_items[index] = rendered_item

    return [rendered_items[index] for index in sorted(rendered_items)]


def _run_llm_batches(
    sampled_items: list[dict[str, Any]],
    *,
    client: Any,
    provider: str,
    model: str,
    batch_size: int,
    llm_concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all LLM batches concurrently. Per-batch failures are logged + skipped.

    Returns ``(flat_results, failed_batches)`` where ``failed_batches`` is a
    list of ``{"batch_index": int, "size": int, "error": str}`` records.
    """
    batches = [
        sampled_items[index:index + batch_size]
        for index in range(0, len(sampled_items), batch_size)
    ]
    completed: dict[int, list[dict[str, Any]]] = {}
    failed: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=llm_concurrency) as executor:
        futures = {
            executor.submit(
                extract_endorsement_targets_batch,
                client,
                provider,
                batch,
                model,
                f"batch {batch_index}",
            ): batch_index
            for batch_index, batch in enumerate(batches)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="LLM target extract",
        ):
            batch_index = futures[future]
            try:
                completed[batch_index] = future.result()
            except Exception as exc:
                tqdm.write(
                    f"[error] batch {batch_index} permanently failed after retries: {exc} "
                    f"(skipping {len(batches[batch_index])} paths)"
                )
                failed.append(
                    {
                        "batch_index": batch_index,
                        "size": len(batches[batch_index]),
                        "error": str(exc),
                    }
                )

    flat_results = [
        result for index in sorted(completed) for result in completed[index]
    ]
    return flat_results, failed


def _enrich_with_anchor(
    flat_results: list[dict[str, Any]],
    sampled_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach (path_anchor_tweet_id, path_anchor_username) to every result.

    Anchor = the candidate tweet whose path the LLM was looking at when it
    emitted that target. Falls back to scanning all paths if the
    representative_tweet_id isn't a path anchor itself.
    """
    tweet_id_to_path = {int(item["tweet_id"]): item for item in sampled_items}
    enriched: list[dict[str, Any]] = []
    for result in flat_results:
        rep_id = int(result["representative_tweet_id"])
        anchor_match = tweet_id_to_path.get(rep_id)
        anchor_username: str | None = None
        anchor_tweet_id: int | None = None
        if anchor_match is not None:
            anchor_username = anchor_match["username"]
            anchor_tweet_id = int(anchor_match["tweet_id"])
        else:
            for item in sampled_items:
                if rep_id in [int(t) for t in item.get("path_tweet_ids", [])]:
                    anchor_username = item["username"]
                    anchor_tweet_id = int(item["tweet_id"])
                    break
        enriched.append(
            {
                **result,
                "path_anchor_tweet_id": anchor_tweet_id,
                "path_anchor_username": anchor_username,
            }
        )
    return enriched


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def run_endorsement_pipeline(
    *,
    candidates: list[dict[str, Any]],
    tweet_dict: Any,
    conversation_trees: Any,
    paths_path: Path,
    targets_path: Path,
    paths_payload_extras: dict[str, Any] | None = None,
    targets_payload_extras: dict[str, Any] | None = None,
    paths_stats_extras: dict[str, Any] | None = None,
    targets_stats_extras: dict[str, Any] | None = None,
    model: str,
    batch_size: int = 12,
    llm_concurrency: int = 50,
    path_concurrency: int = 50,
    paths_only: bool = False,
    window_start: str | None = None,
    window_end: str | None = None,
) -> PipelineResult:
    """Run the full extraction pipeline.

    Args:
        candidates: Already-collected candidate tweets. Each dict must have
            ``tweet_id``, ``username``, ``created_at``, ``full_text``,
            ``reply_to_tweet_id``, ``quoted_tweet_id``.
        tweet_dict, conversation_trees: Loaded caches.
        paths_path, targets_path: Output JSON file paths.
        paths_payload_extras: Top-level keys merged into the paths JSON
            (e.g. ``cohort``, ``users``, ``window``).
        targets_payload_extras: Top-level keys merged into the targets JSON.
        paths_stats_extras: Extra keys merged under ``stats`` in the paths
            JSON (e.g. ``excluded_missing_parent``, ``per_user_candidate_counts``).
        targets_stats_extras: Extra keys merged under ``stats`` in the
            targets JSON.
        model: LLM model id.
        batch_size, llm_concurrency, path_concurrency: knobs.
        paths_only: If true, stop after writing the paths JSON.
        window_start, window_end: Inclusive / exclusive prediction-window
            bounds. When both are provided, ancestors that fall outside
            ``[window_start, window_end)`` (or that are missing from
            ``tweet_dict``) are stripped from every reply path before
            rendering. This guarantees ``path_text``, the LLM prompt, and
            ``representative_tweet_id`` validation only see in-window tweets.
            Set both to ``None`` to disable (legacy behaviour).

    Returns:
        ``PipelineResult`` with paths, counts, and the failed batches log.
    """
    paths_payload_extras = dict(paths_payload_extras or {})
    targets_payload_extras = dict(targets_payload_extras or {})
    paths_stats_extras = dict(paths_stats_extras or {})
    targets_stats_extras = dict(targets_stats_extras or {})

    if (window_start is None) != (window_end is None):
        raise ValueError(
            "window_start and window_end must both be provided or both be None"
        )

    if not candidates:
        raise ValueError("run_endorsement_pipeline received an empty candidate list")

    ordered_candidates = sorted(
        candidates, key=lambda item: (item["created_at"], item["tweet_id"])
    )
    unique_paths = collapse_to_maximal_unique_paths(
        ordered_candidates,
        tweet_dict,
        conversation_trees,
        window_start=window_start,
        window_end=window_end,
    )
    if window_start is not None and window_end is not None:
        print(
            f"Window-truncated reply paths to [{window_start}, {window_end}); "
            f"every ancestor strictly before {window_start} or missing from tweet_dict was dropped."
        )
    print(f"Unique maximal reply paths: {len(unique_paths):,}")
    if not unique_paths:
        raise ValueError("No unique maximal paths could be built from the candidates")

    sampled_items = _render_paths_in_parallel(
        unique_paths,
        tweet_dict,
        conversation_trees,
        path_concurrency=path_concurrency,
    )
    persist_url_resolution_cache(force=True)

    paths_stats = {
        "candidate_tweet_count": len(ordered_candidates),
        "unique_path_count": len(unique_paths),
        "rendered_path_count": len(sampled_items),
        "path_window_truncation": {
            "applied": window_start is not None and window_end is not None,
            "window_start_inclusive": window_start,
            "window_end_exclusive": window_end,
        },
        **paths_stats_extras,
    }
    paths_payload = {
        "generated_at": now_utc_iso(),
        **paths_payload_extras,
        "stats": paths_stats,
        "items": sampled_items,
    }
    _write_json(paths_path, paths_payload)
    print(f"Saved rendered paths to {paths_path}")

    if paths_only:
        return PipelineResult(
            paths_path=paths_path,
            targets_path=None,
            candidate_count=len(ordered_candidates),
            unique_path_count=len(unique_paths),
            rendered_path_count=len(sampled_items),
            sampled_items=sampled_items,
        )

    provider = infer_provider(model)
    client = create_client(model)

    flat_results, failed_batches = _run_llm_batches(
        sampled_items,
        client=client,
        provider=provider,
        model=model,
        batch_size=batch_size,
        llm_concurrency=llm_concurrency,
    )
    enriched_results = _enrich_with_anchor(flat_results, sampled_items)

    target_counts: Counter = Counter(result["direction"] for result in enriched_results)
    target_counts["total_targets"] = len(enriched_results)
    targets_per_anchor: Counter = Counter(
        result["path_anchor_username"]
        for result in enriched_results
        if result["path_anchor_username"]
    )

    targets_stats = {
        "candidate_count": len(ordered_candidates),
        "rendered_path_count": len(sampled_items),
        "batch_size": batch_size,
        "concurrency": llm_concurrency,
        "failed_batches": failed_batches,
        "targets_per_anchor_username": dict(targets_per_anchor),
        "path_window_truncation": {
            "applied": window_start is not None and window_end is not None,
            "window_start_inclusive": window_start,
            "window_end_exclusive": window_end,
        },
        **targets_stats_extras,
    }
    targets_payload = {
        "generated_at": now_utc_iso(),
        "model": model,
        **targets_payload_extras,
        "stats": targets_stats,
        "counts": dict(target_counts),
        "items": enriched_results,
    }
    _write_json(targets_path, targets_payload)
    print(f"Saved endorsement targets to {targets_path}")
    for label, count in sorted(target_counts.items()):
        print(f"  {label:<14} {count:>4}")
    if failed_batches:
        print(
            f"[warn] {len(failed_batches)} batch(es) skipped after retries; "
            f"see stats.failed_batches in {targets_path.name}"
        )

    return PipelineResult(
        paths_path=paths_path,
        targets_path=targets_path,
        candidate_count=len(ordered_candidates),
        unique_path_count=len(unique_paths),
        rendered_path_count=len(sampled_items),
        sampled_items=sampled_items,
        enriched_results=enriched_results,
        target_counts=target_counts,
        failed_batches=failed_batches,
    )
