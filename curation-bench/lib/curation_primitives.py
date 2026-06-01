"""Shared primitives for curation-bench scripts.

Contains:
  - URL resolution (cache-backed, thread-safe, t.co -> final URL)
  - LLM client (OpenAI-compatible + Gemini) with retry/backoff
  - Endorsement target batch extraction (with `url` field, retry-on-validation)
  - Conversation path & tree helpers (path-from-root, filtered tree)

Pure library module (no CLI). Imported via `from lib import ...`.
The orchestrator that strings these together lives in
`lib/endorsement_extraction.py`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import requests
from tqdm import tqdm

if TYPE_CHECKING:
    from openai import OpenAI

LIB_DIR = Path(__file__).resolve().parent
CURATION_BENCH_DIR = LIB_DIR.parent
WORKSPACE_DIR = CURATION_BENCH_DIR.parent

# Use the workspace root on sys.path so we can import scratchpads as a package
# without colliding with our own `lib/` package name.
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from scratchpads.lib.conversation_explorer import (  # noqa: E402  # type: ignore[import-not-found]
    render_conversation_trees,
)
from scratchpads.lib.problem_analysis import (  # noqa: E402  # type: ignore[import-not-found]
    normalize_username,
    parse_eligible_usernames,
    parse_json_from_llm,
)
from scratchpads.lib.strand_caches import (  # noqa: E402  # type: ignore[import-not-found]
    load_caches,
)

URL_RESOLUTION_CACHE_PATH = CURATION_BENCH_DIR / "data" / "twitter_short_url_cache.json"
ENDORSEMENT_PROMPT_PATH = CURATION_BENCH_DIR / "prompts" / "endorsement_prompt.md"

MAX_API_RETRIES = 3
MAX_PARSE_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 2.0

TCO_URL_PATTERN = re.compile(r"https?://t\.co/[A-Za-z0-9]+")
URL_RESOLUTION_CACHE: dict[str, str] = {}
URL_RESOLUTION_CACHE_LOCK = Lock()
URL_CACHE_DIRTY_WRITES = 0
URL_CACHE_FLUSH_EVERY = 50

VALID_DIRECTIONS = {"endorsing", "disendorsing"}


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def created_at_str(tweet: dict[str, Any]) -> str:
    return str(tweet.get("created_at", ""))[:19]


def is_retweet(tweet: dict[str, Any]) -> bool:
    text = str(tweet.get("full_text", "") or "").lstrip()
    return text.startswith("RT @")


def is_candidate_tweet(tweet: dict[str, Any]) -> bool:
    return not is_retweet(tweet)


def has_present_parent(tweet: dict[str, Any], tweet_dict: Any) -> bool:
    parent_id = tweet.get("reply_to_tweet_id")
    if parent_id is None:
        return True
    return tweet_dict.get(parent_id) is not None or tweet_dict.get(str(parent_id)) is not None


def get_reply_tree(
    tweet_id: int,
    tweet: dict[str, Any],
    conversation_trees: Any,
) -> tuple[int, dict[str, Any] | None]:
    conv_id = tweet.get("conversation_id")
    if conv_id is not None:
        tree = conversation_trees.get(conv_id) or conversation_trees.get(str(conv_id))
        if tree:
            return int(conv_id), tree

    tree = conversation_trees.get(tweet_id) or conversation_trees.get(str(tweet_id))
    if tree:
        return int(tweet_id), tree

    return tweet_id, None


def path_from_root(tweet_id: int, tree: dict[str, Any] | None) -> list[int]:
    if not tree:
        return [tweet_id]

    parents = tree.get("parents", {})
    path = [tweet_id]
    seen = {tweet_id}
    current = tweet_id

    while True:
        parent = parents.get(current)
        if parent is None or parent in seen:
            break
        path.append(parent)
        seen.add(parent)
        current = parent

    path.reverse()
    return path


def filtered_tree_for_path(path_ids: list[int]) -> dict[str, Any]:
    children: dict[int, list[int]] = {}
    parents: dict[int, int] = {}

    for parent_id, child_id in zip(path_ids, path_ids[1:]):
        children[parent_id] = [child_id]
        parents[child_id] = parent_id

    return {
        "root": path_ids[0],
        "children": children,
        "parents": parents,
    }


def make_header_renderer():
    def render_header(tweet: dict[str, Any]) -> str:
        date_str = created_at_str(tweet)[:10]
        username = normalize_username(tweet.get("username", ""))
        tweet_id = tweet.get("tweet_id")
        tweet_id_int = int(tweet_id) if tweet_id is not None else -1
        return f"{tweet_id_int} [{date_str}] @{username}"

    return render_header


def clean_rendered_tree(text: str) -> str:
    cleaned = text.rstrip()
    if cleaned.endswith("==="):
        cleaned = cleaned[:-3].rstrip()
    return cleaned


# ---- URL resolution ---------------------------------------------------------


def load_url_resolution_cache() -> None:
    global URL_RESOLUTION_CACHE
    if not URL_RESOLUTION_CACHE_PATH.exists():
        return
    try:
        with URL_RESOLUTION_CACHE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            URL_RESOLUTION_CACHE = {
                str(short_url): str(resolved_url)
                for short_url, resolved_url in data.items()
            }
            print(f"Loaded {len(URL_RESOLUTION_CACHE):,} cached short URLs")
    except Exception as exc:
        tqdm.write(f"[error] Failed to load URL cache {URL_RESOLUTION_CACHE_PATH}: {exc}")


def persist_url_resolution_cache(force: bool = False) -> None:
    global URL_CACHE_DIRTY_WRITES

    with URL_RESOLUTION_CACHE_LOCK:
        if not force and URL_CACHE_DIRTY_WRITES < URL_CACHE_FLUSH_EVERY:
            return
        cache_snapshot = dict(URL_RESOLUTION_CACHE)
        URL_CACHE_DIRTY_WRITES = 0

    URL_RESOLUTION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = URL_RESOLUTION_CACHE_PATH.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(cache_snapshot, file, indent=2, ensure_ascii=False, sort_keys=True)
    temp_path.replace(URL_RESOLUTION_CACHE_PATH)


def update_url_resolution_cache(short_url: str, resolved_url: str) -> None:
    global URL_CACHE_DIRTY_WRITES

    should_flush = False
    with URL_RESOLUTION_CACHE_LOCK:
        previous = URL_RESOLUTION_CACHE.get(short_url)
        if previous == resolved_url:
            return
        URL_RESOLUTION_CACHE[short_url] = resolved_url
        URL_CACHE_DIRTY_WRITES += 1
        should_flush = URL_CACHE_DIRTY_WRITES >= URL_CACHE_FLUSH_EVERY

    if should_flush:
        persist_url_resolution_cache(force=True)


def resolve_short_url(url: str) -> str:
    with URL_RESOLUTION_CACHE_LOCK:
        cached = URL_RESOLUTION_CACHE.get(url)
        if cached is not None:
            return cached

    try:
        response = requests.get(url, allow_redirects=True, timeout=20, stream=True)
        resolved = response.url or url
        response.close()
    except Exception as exc:
        tqdm.write(f"[error] URL resolution failed for {url}: {exc}")
        resolved = url

    update_url_resolution_cache(url, resolved)
    return resolved


def resolve_urls_in_text(text: str) -> str:
    return TCO_URL_PATTERN.sub(lambda match: resolve_short_url(match.group(0)), text)


def warm_url_cache(urls: Iterable[str], concurrency: int, desc: str) -> None:
    """Resolve a set of t.co URLs in parallel so later sequential calls hit the cache."""
    unique = sorted({u for u in urls if u not in URL_RESOLUTION_CACHE})
    if not unique:
        return
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(resolve_short_url, url): url for url in unique}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            url = futures[future]
            try:
                future.result()
            except Exception as exc:
                tqdm.write(f"[error] warm URL {url} failed: {exc}")
    persist_url_resolution_cache(force=True)


# ---- LLM client -------------------------------------------------------------


def infer_provider(model: str) -> str:
    return "gemini" if model.startswith("gemini-") else "openai_compatible"


def create_client(model: str) -> Any:
    provider = infer_provider(model)
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise OSError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini models.")
        return {"provider": "gemini", "api_key": api_key}

    if "DEEPSEEK_API_KEY" not in os.environ:
        raise OSError("Set DEEPSEEK_API_KEY before running this script.")
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing optional dependency `openai`. Install it before running OpenAI-compatible models."
        ) from exc
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def call_openai_compatible_api(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def call_gemini_api(
    client: dict[str, str],
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": client["api_key"],
        },
        json={
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError(f"No Gemini candidates in response: {payload}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    content = "".join(text_parts).strip()
    if not content:
        raise ValueError(f"No Gemini text content in response: {payload}")
    return content


def call_model_with_retries(
    client: Any,
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    request_label: str,
) -> str:
    delay = INITIAL_RETRY_DELAY_SECONDS
    last_error: Exception | None = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            if provider == "gemini":
                return call_gemini_api(client, model, system_prompt, user_prompt)
            return call_openai_compatible_api(client, model, system_prompt, user_prompt)
        except Exception as exc:
            last_error = exc
            tqdm.write(
                f"[error] {request_label} API attempt {attempt}/{MAX_API_RETRIES} failed: {exc}"
            )
            if attempt == MAX_API_RETRIES:
                break
            time.sleep(delay)
            delay *= 2

    assert last_error is not None
    raise last_error


# ---- Endorsement extraction -------------------------------------------------


def load_endorsement_prompt() -> str:
    if not ENDORSEMENT_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing prompt file: {ENDORSEMENT_PROMPT_PATH}")
    return ENDORSEMENT_PROMPT_PATH.read_text(encoding="utf-8")


def build_extraction_user_prompt(items: list[dict[str, Any]]) -> str:
    parts = [
        "Extract endorsement and disendorsement targets from these paths.",
        "Return ONLY the targets you clearly see.",
        "Do not return placeholder rows for paths with no target.",
        "",
        "Schema:",
        (
            '{"targets": [{"representative_tweet_id": 123, "direction": "endorsing"|"disendorsing", '
            '"target_entity": "string", "longer_name": "string", '
            '"context": "1-2 sentences: what the target IS + where it came up; no stance, no @handle of author", '
            '"url": "string|null"}]}'
        ),
        "",
    ]

    for index, item in enumerate(items, start=1):
        parts.extend(
            [
                f"PATH {index}",
                f"path_target_tweet_id: {int(item['tweet_id'])}",
                f"username: {item['username']}",
                f"created_at: {item['created_at']}",
                "path:",
                "```text",
                item["path_text"].rstrip(),
                "```",
                "",
            ]
        )

    return "\n".join(parts).rstrip()


def _normalize_url_field(found: dict[str, Any]) -> str | None:
    """Accept either a string URL or null/missing. Empty strings -> None."""
    raw = found.get("url", None)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"url must be string or null, got {type(raw).__name__}: {raw!r}")
    cleaned = raw.strip()
    return cleaned or None


def _validate_target_rows(
    raw_targets: list[Any],
    valid_tweet_ids: set[int],
) -> list[dict[str, Any]]:
    """Strict validation. Raises ValueError on the first invalid row.

    Used inside the batch retry loop so any per-row defect (hallucinated
    tweet_id, missing field, bad direction, etc.) re-prompts the whole batch.
    """
    results: list[dict[str, Any]] = []
    for found in raw_targets:
        if not isinstance(found, dict):
            raise ValueError(f"Target rows must be objects: {found}")

        representative_tweet_id_raw = found.get("representative_tweet_id")
        if representative_tweet_id_raw is None:
            raise ValueError(f"Missing representative_tweet_id in target row: {found}")

        representative_tweet_id = int(representative_tweet_id_raw)
        if representative_tweet_id not in valid_tweet_ids:
            raise ValueError(
                f"representative_tweet_id {representative_tweet_id} is not present in the provided paths"
            )

        direction = found.get("direction")
        if direction not in VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction in target row: {found}")

        target_entity = str(found.get("target_entity", "")).strip()
        if not target_entity:
            raise ValueError(f"Missing target_entity in target row: {found}")

        longer_name = str(found.get("longer_name", "")).strip()
        if not longer_name:
            raise ValueError(f"Missing longer_name in target row: {found}")

        context = str(found.get("context", "")).strip()
        if not context:
            raise ValueError(f"Missing context in target row: {found}")

        url = _normalize_url_field(found)

        results.append(
            {
                "representative_tweet_id": representative_tweet_id,
                "direction": direction,
                "target_entity": target_entity,
                "longer_name": longer_name,
                "context": context,
                "url": url,
            }
        )

    return results


def extract_endorsement_targets_batch(
    client: Any,
    provider: str,
    items: list[dict[str, Any]],
    model: str,
    request_label: str,
) -> list[dict[str, Any]]:
    """Run one batch through the LLM. Returns list of validated target dicts.

    Each returned dict has:
      representative_tweet_id, direction, target_entity, longer_name, context, url

    Both JSON-parse failures AND per-row validation failures (e.g. the LLM
    hallucinating a tweet_id that wasn't in the input) trigger a retry of the
    whole batch up to MAX_PARSE_RETRIES. After max retries the last exception
    is raised so the orchestrator can decide to skip+log this batch.
    """
    system_prompt = load_endorsement_prompt()
    user_prompt = build_extraction_user_prompt(items)
    valid_tweet_ids = {
        int(tweet_id)
        for item in items
        for tweet_id in item.get("path_tweet_ids", [])
    }
    parse_delay = INITIAL_RETRY_DELAY_SECONDS
    last_error: Exception | None = None

    for attempt in range(1, MAX_PARSE_RETRIES + 1):
        raw_content = call_model_with_retries(
            client=client,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request_label=request_label,
        )
        try:
            parsed = parse_json_from_llm(raw_content)
            raw_targets = parsed.get("targets")
            if not isinstance(raw_targets, list):
                raise ValueError(f"Invalid model response shape: {parsed}")
            return _validate_target_rows(raw_targets, valid_tweet_ids)
        except Exception as exc:
            last_error = exc
            tqdm.write(
                f"[error] {request_label} attempt {attempt}/{MAX_PARSE_RETRIES} failed: {exc}"
            )
            if attempt == MAX_PARSE_RETRIES:
                break
            time.sleep(parse_delay)
            parse_delay *= 2

    assert last_error is not None
    raise last_error


# ---- Path rendering for extraction ------------------------------------------


def render_target_path(
    tweet_id: int,
    tweet: dict[str, Any],
    tweet_dict: Any,
    conversation_trees: Any,
) -> tuple[str, list[int]]:
    path_ids = _path_ids_for_tweet(tweet_id, tweet, conversation_trees)
    filtered_tree = filtered_tree_for_path(path_ids)
    rendered = render_conversation_trees(
        {path_ids[0]: filtered_tree},
        tweet_dict,
        render_header=make_header_renderer(),
    )
    return resolve_urls_in_text(clean_rendered_tree(rendered)), path_ids


def _path_ids_for_tweet(tweet_id: int, tweet: dict[str, Any], conversation_trees: Any) -> list[int]:
    _, tree = get_reply_tree(tweet_id, tweet, conversation_trees)
    return path_from_root(tweet_id, tree)


def _truncate_path_to_window(
    path_ids: list[int],
    tweet_dict: Any,
    window_start: str,
    window_end: str,
) -> list[int]:
    """Keep only the contiguous in-window suffix of ``path_ids`` ending at the leaf.

    ``path_ids`` is ordered root -> leaf. The leaf is the candidate (anchor)
    tweet, which is guaranteed to be in window by upstream selection. Reply
    timestamps are monotonic root -> leaf, so the path is structurally
    ``[ ..pre-window.. ] [ ..in-window.. ]`` with a single transition. We walk
    from the leaf backwards and stop at the first ancestor that is either
    pre-window or missing from ``tweet_dict`` (we cannot prove it is in window
    in that case; conservative).

    Returns the truncated list. The leaf is preserved iff it is in window.
    """
    cutoff = len(path_ids)
    for index in range(len(path_ids) - 1, -1, -1):
        ancestor_id = path_ids[index]
        ancestor = tweet_dict.get(ancestor_id) or tweet_dict.get(str(ancestor_id))
        ancestor_ts = created_at_str(ancestor) if ancestor else ""
        if ancestor_ts and window_start <= ancestor_ts < window_end:
            cutoff = index
        else:
            break
    return path_ids[cutoff:]


def collapse_to_maximal_unique_paths(
    candidates: list[dict[str, Any]],
    tweet_dict: Any,
    conversation_trees: Any,
    *,
    window_start: str | None = None,
    window_end: str | None = None,
) -> list[dict[str, Any]]:
    """Collapse candidates to unique maximal reply paths.

    If ``window_start`` and ``window_end`` are both provided, every ancestor
    that is pre-window (``created_at < window_start``) or missing from
    ``tweet_dict`` is dropped from each path before deduplication. This
    prevents pre-window content from ever reaching the rendered ``path_text``,
    the LLM prompt, or the validation set, which is the root-cause fix for
    the data leak documented in DATA_LEAK_DIAGNOSTIC.md.
    """
    unique_by_path: dict[tuple[int, ...], dict[str, Any]] = {}
    strict_prefixes: set[tuple[int, ...]] = set()
    apply_window = window_start is not None and window_end is not None

    for item in candidates:
        tweet_id = int(item["tweet_id"])
        tweet = tweet_dict.get(tweet_id)
        if not tweet:
            continue

        path_ids = _path_ids_for_tweet(tweet_id, tweet, conversation_trees)
        if apply_window:
            path_ids = _truncate_path_to_window(
                path_ids, tweet_dict, window_start, window_end  # type: ignore[arg-type]
            )
            if not path_ids:
                continue
        path_key = tuple(path_ids)
        if path_key in unique_by_path:
            continue

        unique_by_path[path_key] = {
            **item,
            "path_tweet_ids": path_ids,
        }
        for prefix_length in range(1, len(path_ids)):
            strict_prefixes.add(path_key[:prefix_length])

    maximal_paths = [
        item
        for path_key, item in unique_by_path.items()
        if path_key not in strict_prefixes
    ]
    return sorted(maximal_paths, key=lambda item: (item["created_at"], item["tweet_id"]))


def render_sampled_item(
    item: dict[str, Any],
    tweet_dict: Any,
    conversation_trees: Any,
) -> dict[str, Any] | None:
    tweet_id = int(item["tweet_id"])
    tweet = tweet_dict.get(tweet_id)
    if not tweet:
        return None

    existing_path_ids = item.get("path_tweet_ids")
    if isinstance(existing_path_ids, list) and existing_path_ids:
        path_ids = [int(path_id) for path_id in existing_path_ids]
        filtered_tree = filtered_tree_for_path(path_ids)
        rendered = render_conversation_trees(
            {path_ids[0]: filtered_tree},
            tweet_dict,
            render_header=make_header_renderer(),
        )
        path_text = resolve_urls_in_text(clean_rendered_tree(rendered))
    else:
        path_text, path_ids = render_target_path(tweet_id, tweet, tweet_dict, conversation_trees)

    return {
        **item,
        "path_tweet_ids": path_ids,
        "path_text": path_text,
    }


# ---- Tree filtering for the user-archive script -----------------------------


def trim_tree_to_user(
    tree: dict[str, Any],
    user_node_ids: set[int],
) -> dict[str, Any]:
    """Build a sub-tree containing user nodes, all their ancestors, and their direct children.

    Returns a tree dict (root, children, parents) with only visible nodes.
    """
    parents = tree.get("parents", {}) or {}
    children = tree.get("children", {}) or {}

    visible: set[int] = set(user_node_ids)
    for node in user_node_ids:
        cur = node
        while True:
            parent = parents.get(cur)
            if parent is None or parent in visible:
                break
            visible.add(parent)
            cur = parent
        for child in children.get(node, []):
            visible.add(child)

    new_parents = {n: p for n, p in parents.items() if n in visible and p in visible}
    new_children: dict[int, list[int]] = {}
    for n in visible:
        kept = [c for c in children.get(n, []) if c in visible]
        if kept:
            new_children[n] = kept

    original_root = tree.get("root")
    if original_root is not None and original_root in visible:
        root = original_root
    else:
        candidates = [n for n in visible if new_parents.get(n) is None]
        root = min(candidates) if candidates else (next(iter(visible)) if visible else None)

    return {
        "root": root,
        "children": new_children,
        "parents": new_parents,
    }


# ---- Re-export -------------------------------------------------------------

__all__ = [
    "ENDORSEMENT_PROMPT_PATH",
    "TCO_URL_PATTERN",
    "URL_RESOLUTION_CACHE_PATH",
    "VALID_DIRECTIONS",
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
    "trim_tree_to_user",
    "warm_url_cache",
]
