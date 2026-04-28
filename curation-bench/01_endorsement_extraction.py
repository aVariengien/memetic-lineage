#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "diskcache",
#   "groq",
#   "httpx",
#   "openai",
#   "pandas",
#   "python-dotenv",
#   "requests",
#   "tqdm",
# ]
# ///

"""Render reply paths and extract endorsement targets from them."""

import argparse
import json
import os
import random
import re
import sys
import time
from threading import Lock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
CURATION_BENCH_DIR = Path(__file__).resolve().parent
SCRATCHPADS_DIR = WORKSPACE_DIR / "scratchpads"

if str(SCRATCHPADS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCHPADS_DIR))

from lib.conversation_explorer import render_conversation_trees  # noqa: E402  # type: ignore[import-not-found]
from lib.problem_analysis import (  # noqa: E402
    normalize_username,
    parse_eligible_usernames,
    parse_json_from_llm,
)  # type: ignore[import-not-found]
from lib.strand_caches import load_caches  # noqa: E402  # type: ignore[import-not-found]

RAW_USER_DIRECTORY_PATH = SCRATCHPADS_DIR / "data" / "raw_copy_paste_user_directory.txt"
TWEET_ID_SUBSETS_PATH = SCRATCHPADS_DIR / "data" / "problem_resolution" / "tweet_id_subsets_aug2024.json"
ENDORSEMENT_PROMPT_PATH = CURATION_BENCH_DIR / "prompts" / "endorsement_prompt.md"
URL_RESOLUTION_CACHE_PATH = CURATION_BENCH_DIR / "data" / "twitter_short_url_cache.json"

THREE_MONTH_START = "2024-06-01 00:00:00"
THREE_MONTH_END = "2024-09-01 00:00:00"
DEFAULT_WINDOW_START = "2024-06-04 00:00:00"
DEFAULT_WINDOW_END = "2024-06-07 00:00:00"
ARCHIVE_UPLOAD_CUTOFF = pd.Timestamp("2025-09-01")

DEFAULT_TOP_N_USERS = 5
DEFAULT_SAMPLE_SIZE = None
DEFAULT_SAMPLE_SEED = 42
DEFAULT_PATH_CONCURRENCY = 50
DEFAULT_BATCH_SIZE = 12
DEFAULT_CONCURRENCY = 50
DEFAULT_MODEL = "gemini-3.1-flash-lite-preview" #"deepseek-chat"
MAX_API_RETRIES = 3
MAX_PARSE_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 2.0
TCO_URL_PATTERN = re.compile(r"https?://t\.co/[A-Za-z0-9]+")
URL_RESOLUTION_CACHE: dict[str, str] = {}
URL_RESOLUTION_CACHE_LOCK = Lock()
URL_CACHE_DIRTY_WRITES = 0
URL_CACHE_FLUSH_EVERY = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build root-to-target reply paths for sampled tweets from the top-50 "
            "Jun-Aug 2024 cohort and extract endorsement or disendorsement targets "
            "from the rendered paths."
        )
    )
    parser.add_argument("--start", default=DEFAULT_WINDOW_START, help="Inclusive window start.")
    parser.add_argument("--end", default=DEFAULT_WINDOW_END, help="Exclusive window end.")
    parser.add_argument("--top-n-users", type=int, default=DEFAULT_TOP_N_USERS)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Optional cap on the number of paths to process. Omit to use all candidates.",
    )
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--path-concurrency",
        type=int,
        default=DEFAULT_PATH_CONCURRENCY,
        help="Parallelism for rendering paths and resolving URLs.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Stop after generating the sampled path JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for sampled paths and endorsement target outputs.",
    )
    return parser.parse_args()


def now_utc() -> str:
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


def get_reply_tree(tweet_id: int, tweet: dict[str, Any], conversation_trees: Any) -> tuple[int, Any | None]:
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


def get_path_ids_for_tweet(
    tweet_id: int,
    tweet: dict[str, Any],
    conversation_trees: Any,
) -> list[int]:
    _, tree = get_reply_tree(tweet_id, tweet, conversation_trees)
    return path_from_root(tweet_id, tree)


def collapse_to_maximal_unique_paths(
    candidates: list[dict[str, Any]],
    tweet_dict: Any,
    conversation_trees: Any,
) -> list[dict[str, Any]]:
    unique_by_path: dict[tuple[int, ...], dict[str, Any]] = {}
    strict_prefixes: set[tuple[int, ...]] = set()

    for item in candidates:
        tweet_id = int(item["tweet_id"])
        tweet = tweet_dict.get(tweet_id)
        if not tweet:
            continue

        path_ids = get_path_ids_for_tweet(tweet_id, tweet, conversation_trees)
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


def render_target_path(
    tweet_id: int,
    tweet: dict[str, Any],
    tweet_dict: Any,
    conversation_trees: Any,
) -> tuple[str, list[int]]:
    path_ids = get_path_ids_for_tweet(tweet_id, tweet, conversation_trees)
    filtered_tree = filtered_tree_for_path(path_ids)
    rendered = render_conversation_trees(
        {path_ids[0]: filtered_tree},
        tweet_dict,
        render_header=make_header_renderer(),
    )
    return resolve_urls_in_text(clean_rendered_tree(rendered)), path_ids


def render_sampled_item(item: dict[str, Any], tweet_dict: Any, conversation_trees: Any) -> dict[str, Any] | None:
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


def build_extraction_user_prompt(items: list[dict[str, Any]]) -> str:
    parts = [
        "Extract endorsement and disendorsement targets from these paths.",
        "Return ONLY the targets you clearly see.",
        "Do not return placeholder rows for paths with no target.",
        "",
        "Schema:",
        '{"targets": [{"representative_tweet_id": 123, "direction": "endorsing"|"disendorsing", "target_entity": "string", "longer_name": "string", "context": "short explanation", "url": "resolved target URL or null"}]}',
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


def load_endorsement_prompt() -> str:
    if not ENDORSEMENT_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing prompt file: {ENDORSEMENT_PROMPT_PATH}")
    return ENDORSEMENT_PROMPT_PATH.read_text(encoding="utf-8")


def infer_provider(model: str) -> str:
    return "gemini" if model.startswith("gemini-") else "openai_compatible"


def create_client(model: str) -> Any:
    provider = infer_provider(model)
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini models.")
        return {"provider": "gemini", "api_key": api_key}

    if "DEEPSEEK_API_KEY" not in os.environ:
        raise EnvironmentError("Set DEEPSEEK_API_KEY before running this script.")
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def call_openai_compatible_api(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
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


def call_gemini_api(client: dict[str, str], model: str, system_prompt: str, user_prompt: str) -> str:
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


def extract_endorsement_targets_batch(
    client: Any,
    provider: str,
    items: list[dict[str, Any]],
    model: str,
    request_label: str,
) -> list[dict[str, Any]]:
    system_prompt = load_endorsement_prompt()
    user_prompt = build_extraction_user_prompt(items)
    parse_delay = INITIAL_RETRY_DELAY_SECONDS
    parsed: dict[str, Any] | None = None
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
            if not isinstance(parsed.get("targets"), list):
                raise ValueError(f"Invalid model response shape: {parsed}")
            break
        except Exception as exc:
            last_error = exc
            tqdm.write(
                f"[error] {request_label} parse attempt {attempt}/{MAX_PARSE_RETRIES} failed: {exc}"
            )
            if attempt == MAX_PARSE_RETRIES:
                break
            time.sleep(parse_delay)
            parse_delay *= 2

    if parsed is None:
        assert last_error is not None
        raise last_error

    valid_directions = {"endorsing", "disendorsing"}
    valid_tweet_ids = {
        int(tweet_id)
        for item in items
        for tweet_id in item.get("path_tweet_ids", [])
    }

    results: list[dict[str, Any]] = []
    for found in parsed["targets"]:
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
        if direction not in valid_directions:
            raise ValueError(f"Invalid direction in target row: {found}")

        target_entity = str(found.get("target_entity", "")).strip()
        if not target_entity:
            raise ValueError(f"Missing target_entity in target row: {found}")

        longer_name = str(found.get("longer_name", "")).strip()
        if not longer_name:
            raise ValueError(f"Missing longer_name in target row: {found}")

        context = str(found.get("context", "")).strip()
        if not context:
            context = str(found.get("reason", "")).strip()
        if not context:
            raise ValueError(f"Missing context/reason in target row: {found}")

        url_value = found.get("url")
        if url_value is None:
            normalized_url = None
        else:
            normalized_url = str(url_value).strip()
            if not normalized_url:
                raise ValueError(f"Invalid url in target row: {found}")
            if "t.co/" in normalized_url:
                raise ValueError(f"url must be resolved and not use t.co: {found}")

        results.append(
            {
                "representative_tweet_id": representative_tweet_id,
                "direction": direction,
                "target_entity": target_entity,
                "longer_name": longer_name,
                "context": context,
                "url": normalized_url,
            }
        )

    return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_url_resolution_cache()

    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.path_concurrency <= 0:
        raise ValueError("--path-concurrency must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.start >= args.end:
        raise ValueError("--start must be earlier than --end")
    tweet_dict, conversation_trees = load_caches(auto_generate=False)
    print(f"Loaded {len(tweet_dict):,} tweets and {len(conversation_trees):,} conversation trees")

    eligible_users = parse_eligible_usernames(RAW_USER_DIRECTORY_PATH, ARCHIVE_UPLOAD_CUTOFF)

    if not TWEET_ID_SUBSETS_PATH.exists():
        raise FileNotFoundError(
            f"Missing subset file: {TWEET_ID_SUBSETS_PATH}. "
            "This script expects the same precomputed subset used by 28_problem_resolution_3month.py."
        )

    with TWEET_ID_SUBSETS_PATH.open("r", encoding="utf-8") as file:
        subset_data = json.load(file)

    six_month_ids = [int(tweet_id) for tweet_id in subset_data["tweet_ids"]["six_month_to_sep_2024"]]

    three_month_tweet_count = 0
    user_counts: Counter[str] = Counter()
    window_candidate_pool: list[dict[str, Any]] = []
    excluded_missing_parent = 0
    for tweet_id in tqdm(six_month_ids, desc="Scanning subset for cohort and window"):
        tweet = tweet_dict.get(tweet_id)
        if not tweet:
            continue

        created_at = created_at_str(tweet)
        if not (THREE_MONTH_START <= created_at < THREE_MONTH_END):
            continue

        three_month_tweet_count += 1

        username = normalize_username(tweet.get("username", ""))
        if username not in eligible_users:
            continue

        user_counts[username] += 1

        if not (args.start <= created_at < args.end):
            continue

        if not is_candidate_tweet(tweet):
            continue

        if not has_present_parent(tweet, tweet_dict):
            excluded_missing_parent += 1
            continue

        window_candidate_pool.append(
            {
                "tweet_id": int(tweet_id),
                "username": username,
                "created_at": created_at,
                "reply_to_tweet_id": tweet.get("reply_to_tweet_id"),
                "quoted_tweet_id": tweet.get("quoted_tweet_id"),
                "full_text": tweet.get("full_text", ""),
            }
        )

    print(f"Three-month subset: {three_month_tweet_count:,} tweets")

    top_users = user_counts.most_common(args.top_n_users)
    top_usernames = {username for username, _ in top_users}

    print(f"Top {args.top_n_users} users from Jun-Aug 2024 cohort:")
    for index, (username, count) in enumerate(top_users, start=1):
        print(f"  {index:>2}. @{username:<22} {count:>5} tweets")

    candidates = [
        item
        for item in window_candidate_pool
        if item["username"] in top_usernames
    ]

    print(f"Window candidate tweets from top users: {len(candidates):,}")
    print(f"Excluded replies with missing parent tweet: {excluded_missing_parent:,}")
    if not candidates:
        raise ValueError("No candidate tweets found for the selected window.")

    ordered_candidates = sorted(candidates, key=lambda item: (item["created_at"], item["tweet_id"]))
    unique_paths = collapse_to_maximal_unique_paths(
        ordered_candidates,
        tweet_dict,
        conversation_trees,
    )
    print(f"Unique maximal paths from candidates: {len(unique_paths):,}")
    if not unique_paths:
        raise ValueError("No unique maximal paths found for the selected window.")

    if args.sample_size is None or len(unique_paths) <= args.sample_size:
        sampled = unique_paths
        if args.sample_size is None:
            print(f"Using all {len(sampled):,} unique paths because no sample_size was provided")
        else:
            print(
                f"Using all {len(sampled):,} unique paths because sample_size exceeds available paths"
            )
    else:
        rng = random.Random(args.sample_seed)
        sampled = sorted(
            rng.sample(unique_paths, args.sample_size),
            key=lambda item: (item["created_at"], item["tweet_id"]),
        )
        print(
            f"Sampled {len(sampled):,} unique paths from {len(unique_paths):,} available paths "
            f"with seed={args.sample_seed}"
        )

    rendered_items: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.path_concurrency) as executor:
        futures = {
            executor.submit(render_sampled_item, item, tweet_dict, conversation_trees): index
            for index, item in enumerate(sampled)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering unique paths"):
            index = futures[future]
            try:
                rendered_item = future.result()
                if rendered_item is not None:
                    rendered_items[index] = rendered_item
            except Exception as exc:
                tqdm.write(f"[error] render path item {index} failed: {exc}")
                raise

    sampled_items = [rendered_items[index] for index in sorted(rendered_items)]
    persist_url_resolution_cache(force=True)

    window_slug = (
        args.start[:10].replace("-", "")
        + "_"
        + args.end[:10].replace("-", "")
        + f"_sample{len(sampled_items)}"
    )
    sample_path = args.output_dir / f"{window_slug}_paths.json"
    results_path = args.output_dir / f"{window_slug}_endorsement_targets.json"

    with sample_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "generated_at": now_utc(),
                "window": {"start_inclusive": args.start, "end_exclusive": args.end},
                "three_month_cohort": {
                    "start_inclusive": THREE_MONTH_START,
                    "end_exclusive": THREE_MONTH_END,
                    "top_n_users": args.top_n_users,
                    "top_users": [{"username": username, "tweet_count": count} for username, count in top_users],
                },
                "sampling": {
                    "candidate_tweet_count": len(ordered_candidates),
                    "unique_path_count": len(unique_paths),
                    "sample_size_requested": args.sample_size,
                    "sample_size_actual": len(sampled_items),
                    "sample_seed": args.sample_seed,
                },
                "items": sampled_items,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved rendered paths to {sample_path}")

    if args.paths_only:
        print("Stopping after path generation because --paths-only was set")
        return

    provider = infer_provider(args.model)
    client = create_client(args.model)

    batches = [
        sampled_items[index:index + args.batch_size]
        for index in range(0, len(sampled_items), args.batch_size)
    ]
    completed: dict[int, list[dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                extract_endorsement_targets_batch,
                client,
                provider,
                batch,
                args.model,
                f"batch {batch_index}",
            ): batch_index
            for batch_index, batch in enumerate(batches)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM target extract"):
            batch_index = futures[future]
            try:
                completed[batch_index] = future.result()
            except Exception as exc:
                tqdm.write(f"[error] batch {batch_index} failed: {exc}")
                raise

    flat_results = [result for index in sorted(completed) for result in completed[index]]
    target_counts = Counter(result["direction"] for result in flat_results)
    target_counts["total_targets"] = len(flat_results)

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "generated_at": now_utc(),
                "model": args.model,
                "window": {"start_inclusive": args.start, "end_exclusive": args.end},
                "sampling": {
                    "candidate_count": len(ordered_candidates),
                    "sample_size_requested": args.sample_size,
                    "sample_size_actual": len(sampled_items),
                    "sample_seed": args.sample_seed,
                    "batch_size": args.batch_size,
                    "concurrency": args.concurrency,
                },
                "counts": dict(target_counts),
                "items": flat_results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved endorsement targets to {results_path}")
    for label, count in sorted(target_counts.items()):
        print(f"  {label:<14} {count:>4}")


if __name__ == "__main__":
    main()
