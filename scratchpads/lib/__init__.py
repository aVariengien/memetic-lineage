"""Lightweight package exports for `scratchpads.lib`.

Keep package import side effects minimal so callers can import individual helpers
without transitively requiring optional dependencies such as image-description or
LLM tooling.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "StrandSeed",
    "StrandBuildResult",
    "get_strand_seeds",
    "build_strand_single",
    "build_strands_phased",
    "RatedStrandResult",
    "rate_strand",
    "rate_strands_batch",
    "MediaDescription",
    "get_image_cache",
    "get_image_descriptions",
    "get_image_descriptions_batch",
    "parallel_map_to_dict",
    "parallel_map_to_dict_with_context",
    "batch_keys",
    "with_retry",
    "is_rate_limit_error",
    "is_transient_error",
    "load_caches",
    "get_quote_tweets_dict",
    "get_filtered_quote_tweets_dict",
    "generate_caches",
    "generate_filtered_quote_cache",
    "AccountConversationsResult",
    "get_account_tweets",
    "get_account_conversations",
    "save_account_conversations",
    "explore_account",
]

_EXPORT_TO_MODULE = {
    "StrandSeed": "strand_builder",
    "StrandBuildResult": "strand_builder",
    "get_strand_seeds": "strand_builder",
    "build_strand_single": "strand_builder",
    "build_strands_phased": "strand_builder",
    "RatedStrandResult": "strand_rater",
    "rate_strand": "strand_rater",
    "rate_strands_batch": "strand_rater",
    "MediaDescription": "image_describer",
    "get_image_cache": "image_describer",
    "get_image_descriptions": "image_describer",
    "get_image_descriptions_batch": "image_describer",
    "parallel_map_to_dict": "parallel",
    "parallel_map_to_dict_with_context": "parallel",
    "batch_keys": "parallel",
    "with_retry": "retry",
    "is_rate_limit_error": "retry",
    "is_transient_error": "retry",
    "load_caches": "strand_caches",
    "get_quote_tweets_dict": "strand_caches",
    "get_filtered_quote_tweets_dict": "strand_caches",
    "generate_caches": "strand_caches",
    "generate_filtered_quote_cache": "strand_caches",
    "AccountConversationsResult": "account_explorer",
    "get_account_tweets": "account_explorer",
    "get_account_conversations": "account_explorer",
    "save_account_conversations": "account_explorer",
    "explore_account": "account_explorer",
}


def __getattr__(name: str):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
