# %%
"""
Problem-resolution pipeline for 3 months: June, July, August 2024.
Top 50 users by tweet volume. Reuses few-shot examples from the week study.
"""

# %%
import json
import os
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
from lib.problem_analysis import (  # noqa: E402
    normalize_username, is_top_level_original, tweet_record,
    parse_eligible_usernames, classify_problem_batch, classify_outcome_batch,
    collect_descendant_ids, format_tweet_line,
)

# %%
# === Configuration ===

WEEK_OUTPUT_DIR = SCRATCHPADS_DIR / "data" / "problem_resolution"
OUTPUT_DIR = SCRATCHPADS_DIR / "data" / "problem_resolution" / "3month"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_USER_DIRECTORY_PATH = SCRATCHPADS_DIR / "data" / "raw_copy_paste_user_directory.txt"

# Reuse few-shot files from the week study
FEW_SHOT_PROBLEM_PATH = WEEK_OUTPUT_DIR / "few_shot_problem_classification.json"
FEW_SHOT_OUTCOME_PATH = WEEK_OUTPUT_DIR / "few_shot_outcome_classification.json"

# Source: six-month subset built in notebook 28
TWEET_ID_SUBSETS_PATH = WEEK_OUTPUT_DIR / "tweet_id_subsets_aug2024.json"

# Outputs for this run
PROBLEM_CLASSIFICATION_PATH = OUTPUT_DIR / "problem_classification_deepseek.json"
PROBLEM_THREADS_PATH = OUTPUT_DIR / "problem_threads.json"
OUTCOME_CLASSIFICATION_PATH = OUTPUT_DIR / "outcome_classification_deepseek.json"
USER_HISTORIES_PATH = OUTPUT_DIR / "user_tweet_histories.json"

THREE_MONTH_START = "2024-06-01 00:00:00"  # inclusive
THREE_MONTH_END = "2024-09-01 00:00:00"  # exclusive

ARCHIVE_UPLOAD_CUTOFF = pd.Timestamp("2025-09-01")
TOP_N_USERS = 50
PROBLEM_BATCH_SIZE = 50
OUTCOME_BATCH_SIZE = 10
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_PARALLEL_CALLS = 30

# %%
# === Load tweet caches ===

tweet_dict, conversation_trees = load_caches(auto_generate=False)
print(f"Loaded {len(tweet_dict):,} tweets, {len(conversation_trees):,} conversation trees")

# %%
# === Filter six-month ids down to June-August 2024 ===

with TWEET_ID_SUBSETS_PATH.open("r") as f:
    subset_data = json.load(f)
six_month_ids = [int(tid) for tid in subset_data["tweet_ids"]["six_month_to_sep_2024"]]

three_month_ids = []
for tid in tqdm(six_month_ids, desc="Filtering to Jun-Aug"):
    tweet = tweet_dict.get(tid)
    if not tweet:
        continue
    created_at = str(tweet.get("created_at", ""))[:19]
    if THREE_MONTH_START <= created_at < THREE_MONTH_END:
        three_month_ids.append(tid)

print(f"Three-month subset (Jun-Aug 2024): {len(three_month_ids):,} tweets")

# %%
# === Find top 50 eligible users by tweet volume ===

eligible_users = parse_eligible_usernames(RAW_USER_DIRECTORY_PATH, ARCHIVE_UPLOAD_CUTOFF)

user_counts: Counter = Counter()
for tid in three_month_ids:
    tweet = tweet_dict.get(tid)
    if tweet:
        username = normalize_username(tweet.get("username", ""))
        if username in eligible_users:
            user_counts[username] += 1

top_users = user_counts.most_common(TOP_N_USERS)
top_usernames = {u for u, _ in top_users}

print(f"\nTop {TOP_N_USERS} users (Jun-Aug 2024):")
for i, (user, count) in enumerate(top_users, 1):
    print(f"  {i:>2}. @{user:<22} {count:>5} tweets")

# %%
# === Collect top-level tweets for top users ===

top_level_by_user: dict[str, list[dict]] = defaultdict(list)

for tid in three_month_ids:
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

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# Load few-shot examples for problem classification (from the week study)
few_shot_problem = []
if FEW_SHOT_PROBLEM_PATH.exists():
    with FEW_SHOT_PROBLEM_PATH.open("r") as f:
        few_shot_problem = json.load(f).get("examples", [])
    print(f"Problem few-shot examples: {len(few_shot_problem)}")
# %%
batches = [all_top_level[i:i + PROBLEM_BATCH_SIZE] for i in range(0, len(all_top_level), PROBLEM_BATCH_SIZE)]
completed: dict[int, list[dict]] = {}

with ThreadPoolExecutor(max_workers=DEEPSEEK_PARALLEL_CALLS) as executor:
    futures = {
        executor.submit(classify_problem_batch, client, batch, DEEPSEEK_MODEL, few_shot_problem): idx
        for idx, batch in enumerate(batches)
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek problem classify"):
        completed[futures[future]] = future.result()

flat_results = [r for idx in sorted(completed) for r in completed[idx]]

with PROBLEM_CLASSIFICATION_PATH.open("w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": DEEPSEEK_MODEL,
        "total_tweets": len(all_top_level),
        "few_shot_count": len(few_shot_problem),
        "results": flat_results,
    }, f, indent=2)

print(f"Saved {len(flat_results):,} classifications to {PROBLEM_CLASSIFICATION_PATH.name}")

# %%
# === Load problem tweet ids ===

with PROBLEM_CLASSIFICATION_PATH.open("r") as f:
    classification_data = json.load(f)

problem_ids = [r["tweet_id"] for r in classification_data["results"] if r.get("is_problem") is True]
print(f"Problem tweets: {len(problem_ids)}")

# %%
# === Load quote tweets cache ===

quote_tweets_dict = get_quote_tweets_dict()

# %%
# === Build problem threads ===

problem_threads: list[dict] = []

for tweet_id in problem_ids:
    root_tweet = tweet_dict.get(tweet_id)
    if not root_tweet:
        continue

    reply_ids = []
    tree = conversation_trees.get(tweet_id) or conversation_trees.get(str(tweet_id))
    if tree:
        reply_ids = collect_descendant_ids(tree, tweet_id)

    quote_ids = quote_tweets_dict.get(tweet_id, []) or quote_tweets_dict.get(str(tweet_id), []) or []

    context_tweets = []
    for rid in reply_ids:
        t = tweet_dict.get(rid)
        if t:
            context_tweets.append((str(t.get("created_at", ""))[:19], "reply", t))
    for qid in quote_ids:
        t = tweet_dict.get(qid)
        if t:
            context_tweets.append((str(t.get("created_at", ""))[:19], "quote", t))

    context_tweets.sort(key=lambda x: x[0])

    lines = [format_tweet_line(root_tweet, "problem")]
    for _, tag, t in context_tweets:
        lines.append(format_tweet_line(t, tag))

    problem_threads.append({
        "tweet_id": tweet_id,
        "reply_count": len(reply_ids),
        "quote_count": len([qid for qid in quote_ids if tweet_dict.get(qid)]),
        "thread": "\n".join(lines),
    })

print(f"Built {len(problem_threads)} problem threads")
print(f"  with replies: {sum(1 for t in problem_threads if t['reply_count'] > 0)}")
print(f"  with quotes: {sum(1 for t in problem_threads if t['quote_count'] > 0)}")

# %%
# === Save problem threads ===

with PROBLEM_THREADS_PATH.open("w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_problems": len(problem_threads),
        "threads": problem_threads,
    }, f, indent=2, ensure_ascii=False)

print(f"Saved to {PROBLEM_THREADS_PATH.name}")

# %%
# === Classify problem thread outcomes via DeepSeek ===

with FEW_SHOT_OUTCOME_PATH.open("r") as f:
    few_shot_outcome = json.load(f)["examples"]
print(f"Outcome few-shot examples: {len(few_shot_outcome)}")

few_shot_ids = {int(e["tweet_id"]) for e in few_shot_outcome}
non_few_shot = [t for t in problem_threads if t["tweet_id"] not in few_shot_ids]

# Threads with no replies/quotes are trivially unresolved — skip the API call
no_reply_results = [
    {"tweet_id": t["tweet_id"], "label": "unresolved_by_community",
     "confidence": "high", "reason": "no reply"}
    for t in non_few_shot if t["reply_count"] == 0 and t["quote_count"] == 0
]
threads_to_classify = [t for t in non_few_shot if t["reply_count"] > 0 or t["quote_count"] > 0]
print(f"Threads to classify: {len(threads_to_classify)} via DeepSeek, {len(no_reply_results)} trivially unresolved (no reply)")

outcome_batches = [
    threads_to_classify[i:i + OUTCOME_BATCH_SIZE]
    for i in range(0, len(threads_to_classify), OUTCOME_BATCH_SIZE)
]
completed_outcomes: dict[int, list[dict]] = {}

with ThreadPoolExecutor(max_workers=DEEPSEEK_PARALLEL_CALLS) as executor:
    futures = {
        executor.submit(classify_outcome_batch, client, batch, DEEPSEEK_MODEL, few_shot_outcome): idx
        for idx, batch in enumerate(outcome_batches)
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek outcome classify"):
        completed_outcomes[futures[future]] = future.result()

outcome_results = no_reply_results + [r for idx in sorted(completed_outcomes) for r in completed_outcomes[idx]]

with OUTCOME_CLASSIFICATION_PATH.open("w") as f:
    json.dump({
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": DEEPSEEK_MODEL,
        "total_threads": len(threads_to_classify),
        "trivially_unresolved": len(no_reply_results),
        "few_shot_count": len(few_shot_outcome),
        "results": outcome_results,
    }, f, indent=2, ensure_ascii=False)

print(f"Saved {len(outcome_results)} outcome classifications to {OUTCOME_CLASSIFICATION_PATH.name}")

label_counts = Counter(r["label"] for r in outcome_results)
for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

# %%
# === Build per-user tweet histories ===

# Merge outcome labels: DeepSeek results + few-shot manual labels
outcome_by_id: dict[int, dict] = {}
for r in outcome_results:
    outcome_by_id[int(r["tweet_id"])] = r
for e in few_shot_outcome:
    tid = int(e["tweet_id"])
    if tid in {p["tweet_id"] for p in problem_threads}:
        outcome_by_id[tid] = {
            "tweet_id": tid,
            "label": e["label"],
            "confidence": "high",
            "reason": e.get("rationale", "manual few-shot label"),
        }

problem_id_set = set(problem_ids)

# Single pass over three_month_ids, bucket by username
user_tweets: dict[str, list] = defaultdict(list)
user_problems: dict[str, list] = defaultdict(list)

for tid in three_month_ids:
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
    problems_list = sorted(user_problems.get(username, []), key=lambda t: str(t.get("created_at", "")))
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
        "range": {"start_inclusive": THREE_MONTH_START, "end_exclusive": THREE_MONTH_END},
        "users": user_histories,
    }, f, indent=2, ensure_ascii=False)

print(f"Saved {len(user_histories)} user histories to {USER_HISTORIES_PATH.name}")
for username, h in sorted(user_histories.items(), key=lambda x: -x[1]["problem_count"]):
    print(f"  @{username:<22} {h['tweet_count']:>4} tweets, {h['problem_count']:>2} problems")

# %%
