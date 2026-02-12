# %%
"""
Bootstrap analysis for problem-resolution study.

1. Build tweet-id subsets for last week of Aug 2024 and six months up to Sep 2024
2. Sample 50 top-level tweets for manual labeling
3. Find top 20 eligible users by tweet volume in the week subset
4. Collect their top-level tweets
5. Classify tweets as problem statements via DeepSeek
"""

# %%
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    SCRATCHPADS_DIR = Path(__file__).parent
except NameError:
    SCRATCHPADS_DIR = Path.cwd()
    if SCRATCHPADS_DIR.name != "scratchpads":
        SCRATCHPADS_DIR = SCRATCHPADS_DIR / "scratchpads"

if str(SCRATCHPADS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCHPADS_DIR))

from lib.strand_caches import load_caches, get_quote_tweets_dict  # noqa: E402

# %%
# === Configuration ===

OUTPUT_DIR = SCRATCHPADS_DIR / "data" / "problem_resolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_USER_DIRECTORY_PATH = SCRATCHPADS_DIR / "data" / "raw_copy_paste_user_directory.txt"
TWEET_ID_SUBSETS_PATH = OUTPUT_DIR / "tweet_id_subsets_aug2024.json"
MANUAL_SAMPLE_PATH = OUTPUT_DIR / "week_aug_last_top_level_uniform_sample_50_for_manual_labeling.json"
DEEPSEEK_OUTPUT_PATH = OUTPUT_DIR / "week_aug_last_top_level_problem_classification_deepseek.json"
PROBLEM_THREADS_PATH = OUTPUT_DIR / "problem_threads.json"
FEW_SHOT_OUTCOME_PATH = OUTPUT_DIR / "few_shot_outcome_classification.json"
OUTCOME_OUTPUT_PATH = OUTPUT_DIR / "problem_outcome_classification_deepseek.json"
USER_HISTORIES_PATH = OUTPUT_DIR / "user_tweet_histories.json"

WEEK_START = "2024-08-25 00:00:00"  # inclusive
WEEK_END = "2024-09-01 00:00:00"  # exclusive
SIX_MONTH_START = "2024-03-01 00:00:00"  # inclusive
SIX_MONTH_END = "2024-09-01 00:00:00"  # exclusive

ARCHIVE_UPLOAD_CUTOFF = pd.Timestamp("2025-09-01")
TOP_N_USERS = 20
DEEPSEEK_BATCH_SIZE = 100
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_PARALLEL_CALLS = 30

# %%
# === Helpers ===


def normalize_username(username: str) -> str:
    return username.strip().lstrip("@").lower() if username else ""


def is_top_level_original(tweet: dict) -> bool:
    """True if the tweet is neither a reply nor a retweet."""
    if tweet.get("reply_to_tweet_id") is not None:
        return False
    text = str(tweet.get("full_text", "") or "").lstrip()
    return not text.startswith("RT @")


def tweet_record(tweet_id: int, tweet: dict) -> dict:
    """Extract a minimal flat record from a raw tweet."""
    return {
        "tweet_id": int(tweet_id),
        "username": normalize_username(tweet.get("username", "")),
        "reply_to_tweet_id": tweet.get("reply_to_tweet_id"),
        "full_text": tweet.get("full_text", ""),
        "created_at": str(tweet.get("created_at")),
        "conversation_id": tweet.get("conversation_id"),
    }


def parse_eligible_usernames(path: Path, cutoff: pd.Timestamp) -> set[str]:
    """Parse the raw user directory; return usernames with archive uploaded before cutoff.

    Expected tab-separated fields:
    Display Name | @username | Tweets | Likes | Followers | Archive Date | Account Created At | Archive Uploaded At
    """
    eligible = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8 or not parts[1].strip().startswith("@"):
                continue
            uploaded_at = pd.to_datetime(parts[7].strip(), format="%d/%m/%Y", errors="coerce")
            if pd.notna(uploaded_at) and pd.Timestamp(uploaded_at) < cutoff:
                eligible.add(normalize_username(parts[1].strip()))
    print(f"Eligible users (archive before {cutoff.date()}): {len(eligible):,}")
    return eligible


def parse_json_from_llm(raw: str) -> dict:
    """Extract a JSON object from raw LLM output, with brace-matching fallback."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def classify_problem_batch(client, tweets: list[dict], few_shot: list[dict] | None = None) -> list[dict]:
    """Ask DeepSeek whether each tweet is a problem statement. Returns one result per input tweet."""
    system_prompt = (
        "You classify tweets as problem statements. A tweet is a problem statement if:\n"
        "1. It asks a question to the community (issue, complaint, request for help).\n"
        "2. The community could provide a solution (sharing resources, introductions, etc.).\n"
        "If the author is simply expressing a personal problem without community intent, it doesn't count.\n"
        "Problem tweets are rare (~5%). If unsure, answer false.\n"
        "Return strict JSON only."
    )
    if few_shot:
        examples = [
            {"tweet_id": int(e["tweet_id"]), "full_text": e.get("full_text", ""),
             "manual_is_problem": bool(e["manual_is_problem"]), "manual_notes": e.get("manual_notes", "")}
            for e in few_shot
        ]
        system_prompt += f"\n\nFew-shot labeled examples:\n{json.dumps(examples)}"

    payload = [{"tweet_id": int(t["tweet_id"]), "full_text": t.get("full_text", "")} for t in tweets]
    user_prompt = (
        "Classify each tweet. Return EXACTLY one result per input tweet.\n"
        'Schema: {"results": [{"tweet_id": 123, "is_problem": true, '
        '"confidence": "high"|"medium"|"low", "reason": "short explanation"}]}\n\n'
        f"Tweets:\n{json.dumps(payload)}"
    )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    parsed = parse_json_from_llm(response.choices[0].message.content or "")
    if not isinstance(parsed.get("results"), list):
        raise ValueError(f"Invalid model response shape: {parsed}")

    # Index by tweet_id, return in input order with fallback for missing ids
    by_id = {int(r["tweet_id"]): r for r in parsed["results"] if isinstance(r, dict) and "tweet_id" in r}
    results = []
    for t in tweets:
        tid = int(t["tweet_id"])
        found = by_id.get(tid, {})
        results.append({
            "tweet_id": tid,
            "is_problem": found.get("is_problem"),
            "confidence": str(found.get("confidence", "low")),
            "reason": str(found.get("reason", "missing from model output" if not found else "")),
        })
    return results


# %%
# === Load tweet caches ===

tweet_dict, conversation_trees = load_caches(auto_generate=False)
print(f"Loaded {len(tweet_dict):,} tweets, {len(conversation_trees):,} conversation trees")

# %%
# === Build or load tweet-id subsets by date range ===

if TWEET_ID_SUBSETS_PATH.exists():
    with TWEET_ID_SUBSETS_PATH.open("r") as f:
        subset_data = json.load(f)
    week_ids = [int(tid) for tid in subset_data["tweet_ids"]["week_last_aug_2024"]]
    six_month_ids = [int(tid) for tid in subset_data["tweet_ids"]["six_month_to_sep_2024"]]
else:
    week_ids, six_month_ids = [], []
    for tweet_id in tqdm(tweet_dict, desc="Filtering tweets by date"):
        created_at = str(tweet_dict[tweet_id].get("created_at", ""))[:19]
        if len(created_at) != 19:
            continue
        if WEEK_START <= created_at < WEEK_END:
            week_ids.append(int(tweet_id))
        if SIX_MONTH_START <= created_at < SIX_MONTH_END:
            six_month_ids.append(int(tweet_id))

    with TWEET_ID_SUBSETS_PATH.open("w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "tweet_ids": {"week_last_aug_2024": week_ids, "six_month_to_sep_2024": six_month_ids},
        }, f)

print(f"Week subset: {len(week_ids):,} tweets | Six-month subset: {len(six_month_ids):,} tweets")

# %%
# === Sample 50 top-level tweets for manual labeling ===

week_top_level_all = [
    tweet_record(tid, tweet_dict[tid])
    for tid in week_ids
    if tid in tweet_dict and is_top_level_original(tweet_dict[tid])
]

rng = random.Random(28)
sample = sorted(
    rng.sample(week_top_level_all, k=min(50, len(week_top_level_all))),
    key=lambda t: t["created_at"],
)

with MANUAL_SAMPLE_PATH.open("w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_range": {"start_inclusive": WEEK_START, "end_exclusive": WEEK_END},
        "sampling": {
            "method": "uniform_without_replacement",
            "population_size": len(week_top_level_all),
            "sample_size": len(sample),
            "seed": 28,
        },
        "labeling_schema": {
            "manual_is_problem": "true/false by annotator",
            "manual_notes": "optional rationale",
        },
        "samples": [{**row, "manual_is_problem": None, "manual_notes": ""} for row in sample],
    }, f, indent=2)

print(f"Saved {len(sample)} manual-label samples to {MANUAL_SAMPLE_PATH.name}")

# %%
# === Find top users by tweet volume (eligible users only) ===

eligible_users = parse_eligible_usernames(RAW_USER_DIRECTORY_PATH, ARCHIVE_UPLOAD_CUTOFF)

week_counts: Counter = Counter()
for tid in week_ids:
    tweet = tweet_dict.get(tid)
    if tweet:
        username = normalize_username(tweet.get("username", ""))
        if username in eligible_users:
            week_counts[username] += 1

top_users = week_counts.most_common(TOP_N_USERS)

print(f"\nTop {TOP_N_USERS} users (last week Aug 2024):")
for i, (user, count) in enumerate(top_users, 1):
    print(f"  {i:>2}. @{user:<22} {count:>5} tweets")

# %%
# === Collect top-level tweets for top users ===

top_usernames = {u for u, _ in top_users}
top_level_by_user: dict[str, list[dict]] = defaultdict(list)

for tid in week_ids:
    tweet = tweet_dict.get(tid)
    if not tweet:
        continue
    username = normalize_username(tweet.get("username", ""))
    if username in top_usernames and is_top_level_original(tweet):
        top_level_by_user[username].append(tweet_record(tid, tweet))

all_top_level: list[dict] = []
for username, _ in top_users:
    all_top_level.extend(sorted(top_level_by_user.get(username, []), key=lambda t: t["created_at"]))

print(f"Top-level original tweets from top {TOP_N_USERS} users: {len(all_top_level):,}")

# %%
# === Classify tweets as problem statements via DeepSeek ===

if not all_top_level:
    print("No tweets to classify.")
else:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    # Load few-shot examples from manual labels (rows where manual_is_problem was filled in)
    few_shot = []
    if MANUAL_SAMPLE_PATH.exists():
        with MANUAL_SAMPLE_PATH.open("r") as f:
            few_shot = [s for s in json.load(f).get("samples", []) if s.get("manual_is_problem") is not None]
        print(f"Few-shot examples: {len(few_shot)}")

    # Run classification in parallel batches
    batches = [all_top_level[i:i + DEEPSEEK_BATCH_SIZE] for i in range(0, len(all_top_level), DEEPSEEK_BATCH_SIZE)]
    completed: dict[int, list[dict]] = {}

    with ThreadPoolExecutor(max_workers=DEEPSEEK_PARALLEL_CALLS) as executor:
        futures = {
            executor.submit(classify_problem_batch, client, batch, few_shot): idx
            for idx, batch in enumerate(batches)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek classify"):
            completed[futures[future]] = future.result()

    flat_results = [r for idx in sorted(completed) for r in completed[idx]]

    with DEEPSEEK_OUTPUT_PATH.open("w") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": DEEPSEEK_MODEL,
            "total_tweets": len(all_top_level),
            "few_shot_count": len(few_shot),
            "results": flat_results,
        }, f, indent=2)

    print(f"Saved {len(flat_results):,} classifications to {DEEPSEEK_OUTPUT_PATH.name}")

# %%
# === Load problem tweet ids from classification results ===

with DEEPSEEK_OUTPUT_PATH.open("r") as f:
    classification_data = json.load(f)

problem_ids = [
    r["tweet_id"] for r in classification_data["results"]
    if r.get("is_problem") is True
]
print(f"Problem tweets: {len(problem_ids)}")

# %%
# === Load quote tweets cache ===

quote_tweets_dict = get_quote_tweets_dict()

# %%
# === Gather replies + quotes for each problem tweet, format as thread strings ===


def collect_descendant_ids(tree: dict, root_id: int) -> list[int]:
    """BFS walk of a conversation tree's children dict, returning all descendant tweet ids."""
    children_map = tree.get("children", {})
    descendants = []
    queue = [root_id]
    while queue:
        current = queue.pop(0)
        for child_id in children_map.get(current, []):
            descendants.append(child_id)
            queue.append(child_id)
    return descendants


def format_tweet_line(tweet: dict, label: str = "") -> str:
    prefix = f"[{label}] " if label else ""
    username = tweet.get("username", "?")
    created_at = str(tweet.get("created_at", ""))[:19]
    text = " ".join(str(tweet.get("full_text", "")).split())
    return f"{prefix}@{username} ({created_at}): {text}"


problem_threads: list[dict] = []

for tweet_id in problem_ids:
    root_tweet = tweet_dict.get(tweet_id)
    if not root_tweet:
        continue

    # Collect reply ids from the conversation tree (key = conversation_id = tweet_id for top-level)
    reply_ids = []
    tree = conversation_trees.get(tweet_id) or conversation_trees.get(str(tweet_id))
    if tree:
        reply_ids = collect_descendant_ids(tree, tweet_id)

    # Collect quote tweet ids
    quote_ids = (
        quote_tweets_dict.get(tweet_id, [])
        or quote_tweets_dict.get(str(tweet_id), [])
        or []
    )

    # Build labeled tweet records: (created_at, label, tweet_dict_entry)
    context_tweets = []
    for rid in reply_ids:
        t = tweet_dict.get(rid)
        if t:
            context_tweets.append((str(t.get("created_at", ""))[:19], "reply", t))
    for qid in quote_ids:
        t = tweet_dict.get(qid)
        if t:
            context_tweets.append((str(t.get("created_at", ""))[:19], "quote", t))

    # Sort oldest first
    context_tweets.sort(key=lambda x: x[0])

    # Format the thread string
    lines = [format_tweet_line(root_tweet, "problem")]
    for _, label, t in context_tweets:
        lines.append(format_tweet_line(t, label))
    thread_string = "\n".join(lines)

    problem_threads.append({
        "tweet_id": tweet_id,
        "reply_count": len(reply_ids),
        "quote_count": len([qid for qid in quote_ids if tweet_dict.get(qid)]),
        "thread": thread_string,
    })

print(f"Built {len(problem_threads)} problem threads")
print(f"Threads with replies: {sum(1 for t in problem_threads if t['reply_count'] > 0)}")
print(f"Threads with quotes: {sum(1 for t in problem_threads if t['quote_count'] > 0)}")

# %%
# === Save problem threads ===

with PROBLEM_THREADS_PATH.open("w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_problems": len(problem_threads),
        "threads": problem_threads,
    }, f, indent=2, ensure_ascii=False)

print(f"Saved to {PROBLEM_THREADS_PATH.name}")

# Preview first thread
if problem_threads:
    print(f"\n--- Example thread (tweet {problem_threads[0]['tweet_id']}) ---")
    print(problem_threads[0]["thread"])

# %%
# === Classify problem thread outcomes via DeepSeek ===


def classify_outcome_batch(client, threads: list[dict], few_shot: list[dict]) -> list[dict]:
    """Classify the outcome of each problem thread. Returns one result per thread."""
    system_prompt = (
        "You classify the outcome of problem threads on Twitter.\n"
        "Each thread starts with a problem tweet, followed by replies and quote tweets.\n"
        "Your job is to assess the community's reaction. Use exactly one of these labels:\n\n"
        "- resolved_by_community: The community provided help and there is evidence it worked. "
        "Maybe the author thanks someone, says they'll try it, or the author asked about advice and got an answer that addressed their problem.\n"
        "- serious_attempt: At least one reply makes a convincing, substantive attempt to help "
        "(not just sympathy or banter), but there is no clear evidence the question was answered.\n"
        "- unresolved_by_community: Either no replies, only surface-level/joke replies, "
        "self-resolved without community input, or the engagement didn't meaningfully address the problem.\n\n"
        "Return strict JSON only."
    )

    few_shot_payload = [
        {"tweet_id": int(e["tweet_id"]), "label": e["label"],
         "rationale": e.get("rationale", ""), "thread": e["thread"]}
        for e in few_shot
    ]
    system_prompt += f"\n\nFew-shot labeled examples:\n{json.dumps(few_shot_payload, ensure_ascii=False)}"

    payload = [{"tweet_id": int(t["tweet_id"]), "thread": t["thread"]} for t in threads]
    user_prompt = (
        "Classify each thread's outcome. Return EXACTLY one result per input thread.\n"
        'Schema: {"results": [{"tweet_id": 123, '
        '"label": "resolved_by_community"|"serious_attempt"|"unresolved_by_community", '
        '"confidence": "high"|"medium"|"low", '
        '"reason": "short explanation"}]}\n\n'
        f"Threads:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    parsed = parse_json_from_llm(response.choices[0].message.content or "")
    if not isinstance(parsed.get("results"), list):
        raise ValueError(f"Invalid model response shape: {parsed}")

    valid_labels = {"resolved_by_community", "serious_attempt", "unresolved_by_community"}
    by_id = {int(r["tweet_id"]): r for r in parsed["results"] if isinstance(r, dict) and "tweet_id" in r}
    results = []
    for t in threads:
        tid = int(t["tweet_id"])
        found = by_id.get(tid, {})
        label = found.get("label", "unresolved_by_community")
        if label not in valid_labels:
            label = "unresolved_by_community"
        results.append({
            "tweet_id": tid,
            "label": label,
            "confidence": str(found.get("confidence", "low")),
            "reason": str(found.get("reason", "missing from model output" if not found else "")),
        })
    return results


# Load few-shot examples
with FEW_SHOT_OUTCOME_PATH.open("r") as f:
    few_shot_outcome = json.load(f)["examples"]
print(f"Few-shot outcome examples: {len(few_shot_outcome)}")

# Filter out threads already in the few-shot set
few_shot_ids = {int(e["tweet_id"]) for e in few_shot_outcome}
threads_to_classify = [t for t in problem_threads if t["tweet_id"] not in few_shot_ids]
print(f"Threads to classify (excluding few-shot): {len(threads_to_classify)}")

# %%
# === Run outcome classification in parallel ===

from openai import OpenAI

outcome_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# Batch size of 5 since threads are much longer than individual tweets
OUTCOME_BATCH_SIZE = 5
outcome_batches = [
    threads_to_classify[i:i + OUTCOME_BATCH_SIZE]
    for i in range(0, len(threads_to_classify), OUTCOME_BATCH_SIZE)
]
completed_outcomes: dict[int, list[dict]] = {}

with ThreadPoolExecutor(max_workers=DEEPSEEK_PARALLEL_CALLS) as executor:
    futures = {
        executor.submit(classify_outcome_batch, outcome_client, batch, few_shot_outcome): idx
        for idx, batch in enumerate(outcome_batches)
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek outcome classify"):
        completed_outcomes[futures[future]] = future.result()

outcome_results = [r for idx in sorted(completed_outcomes) for r in completed_outcomes[idx]]

with OUTCOME_OUTPUT_PATH.open("w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": DEEPSEEK_MODEL,
        "total_threads": len(threads_to_classify),
        "few_shot_count": len(few_shot_outcome),
        "results": outcome_results,
    }, f, indent=2, ensure_ascii=False)

print(f"Saved {len(outcome_results)} outcome classifications to {OUTCOME_OUTPUT_PATH.name}")

# Summary
from collections import Counter as _Counter
label_counts = _Counter(r["label"] for r in outcome_results)
for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

# %%
# === Build per-user tweet histories ===

# Merge outcome labels: DeepSeek results + few-shot manual labels
outcome_by_id: dict[int, dict] = {}
for r in outcome_results:
    outcome_by_id[int(r["tweet_id"])] = r
for e in few_shot_outcome:
    outcome_by_id[int(e["tweet_id"])] = {
        "tweet_id": int(e["tweet_id"]),
        "label": e["label"],
        "confidence": "high",
        "reason": e.get("rationale", "manual few-shot label"),
    }

problem_id_set = set(problem_ids)

# Single pass over week_ids, bucket by username
user_tweets: dict[str, list] = defaultdict(list)
user_problems: dict[str, list] = defaultdict(list)

for tid in week_ids:
    tweet = tweet_dict.get(tid)
    if not tweet:
        continue
    username = normalize_username(tweet.get("username", ""))
    if username not in top_usernames:
        continue

    if tid in problem_id_set:
        outcome = outcome_by_id.get(tid, {})
        user_problems[username].append({
            **dict(tweet),
            "tweet_id": int(tid),
            "outcome_label": outcome.get("label", "unknown"),
            "outcome_confidence": outcome.get("confidence", ""),
            "outcome_reason": outcome.get("reason", ""),
        })
    else:
        user_tweets[username].append({
            "tweet_id": tid,
            "is_top_level": tweet.get("reply_to_tweet_id") is None,
            "created_at": str(tweet.get("created_at", ""))[:19],
        })

user_histories: dict[str, dict] = {}
for username in top_usernames:
    tweets_list = sorted(user_tweets.get(username, []), key=lambda t: t["created_at"])
    problems_list = sorted(user_problems.get(username, []), key=lambda t: t["created_at"])
    user_histories[username] = {
        "username": username,
        "tweet_count": len(tweets_list),
        "problem_count": len(problems_list),
        "tweets": tweets_list,
        "problems": problems_list,
    }

with USER_HISTORIES_PATH.open("w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "week_range": {"start_inclusive": WEEK_START, "end_exclusive": WEEK_END},
        "users": user_histories,
    }, f, indent=2, ensure_ascii=False)

print(f"Saved {len(user_histories)} user histories to {USER_HISTORIES_PATH.name}")
for username, h in sorted(user_histories.items(), key=lambda x: -x[1]["problem_count"]):
    print(f"  @{username:<22} {h['tweet_count']:>4} tweets, {h['problem_count']:>2} problems")

# %%
