"""curation-bench library package.

Two submodules:

- ``curation_primitives``: low-level helpers (URL resolution, LLM client,
  per-batch endorsement extraction, conversation tree helpers).
- ``endorsement_extraction``: high-level orchestrator that runs the full
  pipeline (collapse paths -> render -> LLM -> write JSON) for a set of
  candidate tweets.

Everything from both modules is re-exported here so callers can write
``from lib import run_endorsement_pipeline, normalize_username, ...``.
"""

from .curation_primitives import (
    ENDORSEMENT_PROMPT_PATH,
    TCO_URL_PATTERN,
    URL_RESOLUTION_CACHE_PATH,
    VALID_DIRECTIONS,
    build_extraction_user_prompt,
    call_model_with_retries,
    clean_rendered_tree,
    collapse_to_maximal_unique_paths,
    create_client,
    created_at_str,
    extract_endorsement_targets_batch,
    filtered_tree_for_path,
    get_reply_tree,
    has_present_parent,
    infer_provider,
    is_candidate_tweet,
    is_retweet,
    load_caches,
    load_endorsement_prompt,
    load_url_resolution_cache,
    make_header_renderer,
    normalize_username,
    now_utc_iso,
    parse_eligible_usernames,
    path_from_root,
    persist_url_resolution_cache,
    render_conversation_trees,
    render_sampled_item,
    render_target_path,
    resolve_short_url,
    resolve_urls_in_text,
    trim_tree_to_user,
    warm_url_cache,
)
from .endorsement_extraction import (
    PipelineResult,
    run_endorsement_pipeline,
)

__all__ = [
    "ENDORSEMENT_PROMPT_PATH",
    "TCO_URL_PATTERN",
    "URL_RESOLUTION_CACHE_PATH",
    "VALID_DIRECTIONS",
    "PipelineResult",
    "build_extraction_user_prompt",
    "call_model_with_retries",
    "clean_rendered_tree",
    "collapse_to_maximal_unique_paths",
    "create_client",
    "created_at_str",
    "extract_endorsement_targets_batch",
    "filtered_tree_for_path",
    "get_reply_tree",
    "has_present_parent",
    "infer_provider",
    "is_candidate_tweet",
    "is_retweet",
    "load_caches",
    "load_endorsement_prompt",
    "load_url_resolution_cache",
    "make_header_renderer",
    "normalize_username",
    "now_utc_iso",
    "parse_eligible_usernames",
    "path_from_root",
    "persist_url_resolution_cache",
    "render_conversation_trees",
    "render_sampled_item",
    "render_target_path",
    "resolve_short_url",
    "resolve_urls_in_text",
    "run_endorsement_pipeline",
    "trim_tree_to_user",
    "warm_url_cache",
]
