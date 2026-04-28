# %%
"""
Classify problem tweets as bounty-worthy using DeepSeek-Reasoner.

A problem is "bounty-worthy" if the poster would plausibly pay a small bounty
to get a good answer — e.g. recommendations, experience-sharing, specialised
expertise questions, or practical help that a trusted peer network could fulfil.
"""

# %%
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

try:
    SCRATCHPADS_DIR = Path(__file__).parent
except NameError:
    SCRATCHPADS_DIR = Path.cwd()
    if SCRATCHPADS_DIR.name != "scratchpads":
        SCRATCHPADS_DIR = SCRATCHPADS_DIR / "scratchpads"

if str(SCRATCHPADS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCHPADS_DIR))

# %%
# === Configuration ===

DATA_DIR = SCRATCHPADS_DIR / "data" / "problem_resolution" / "3month"
PROBLEM_CLASSIFICATION_PATH = DATA_DIR / "problem_classification_deepseek.json"
USER_HISTORIES_PATH = DATA_DIR / "user_tweet_histories.json"
BOUNTY_CLASSIFICATION_PATH = DATA_DIR / "bounty_classification_deepseek.json"

DEEPSEEK_MODEL = "deepseek-reasoner"
BATCH_SIZE = 10
PARALLEL_CALLS = 100

# %%
# === Few-shot examples (hand-crafted based on category definitions) ===
#
# Format: {"full_text": ..., "is_bounty": bool, "bounty_description": str | null, "reason": ...}

FEW_SHOT_EXAMPLES = [
    # --- BOUNTY: YES ---
    {
        "full_text": "Recommend me a book on the late USSR that covers both economics and ideology.",
        "is_bounty": True,
        "bounty_description": "Looking for a book recommendation on late Soviet economics and ideology from someone with deep knowledge of the period.",
        "reason": "Specific recommendation request unlikely to be answered well by a generic LLM — benefits from a trusted peer with real knowledge.",
    },
    {
        "full_text": "What's a good cheap CO2 monitor? Looking for something under $50 that actually works.",
        "is_bounty": True,
        "bounty_description": "Looking for a tested, affordable CO2 monitor recommendation from someone who has tried several.",
        "reason": "Product recommendation with practical constraints — peer experience is more trustworthy than aggregate reviews.",
    },
    {
        "full_text": "Did SSRIs genuinely work for you long-term, and what did that feel like?",
        "is_bounty": True,
        "bounty_description": "Looking for honest long-term personal experience with SSRIs — what changed, what stayed the same.",
        "reason": "Lived-experience question on health that a network of trusted peers can answer better than generic medical advice.",
    },
    {
        "full_text": "Has anyone moved from a hot climate to a cooler one and found they exercised more as a result?",
        "is_bounty": True,
        "bounty_description": "Looking for personal accounts of climate change affecting exercise habits — especially hot-to-cold moves.",
        "reason": "Experience-sharing question — peer data point is more useful than generic fitness content.",
    },
    {
        "full_text": "Why did transformers specifically revolutionize AI rather than prior architectures?",
        "is_bounty": True,
        "bounty_description": "Looking for a detailed, technically grounded explanation of why transformers succeeded where RNNs/CNNs didn't.",
        "reason": "Specialised expertise question — a concise expert answer is genuinely valuable.",
    },
    {
        "full_text": "My injuries heal unusually slowly — what should I investigate?",
        "is_bounty": True,
        "bounty_description": "Looking for informed leads on what blood work, deficiencies, or conditions cause slow healing — personal experience or medical knowledge welcome.",
        "reason": "Practical health problem with no obvious answer — peer network with relevant experience or expertise is ideal.",
    },
    {
        "full_text": "Anyone know how to get car insurance without a fixed address?",
        "is_bounty": True,
        "bounty_description": "Looking for a concrete solution or provider that handles car insurance for people without a permanent address.",
        "reason": "Practical logistics problem that someone has almost certainly solved — peer knowledge is the fastest path.",
    },
    {
        "full_text": "How do I communicate to my CTO that the data warehouse is critical infrastructure and needs proper maintenance?",
        "is_bounty": True,
        "bounty_description": "Looking for framing strategies or scripts to persuade a skeptical technical executive to invest in data infrastructure.",
        "reason": "Workplace communication problem with real stakes — people who have navigated this can give concrete advice.",
    },
    {
        "full_text": "People who are often on the move, how do you do your laundry? Looking for practical ideas beyond coin laundromats.",
        "is_bounty": True,
        "bounty_description": "Looking for practical laundry solutions for nomadic or frequently travelling people.",
        "reason": "Practical lifestyle problem with experience-based answers — a peer network of travellers is ideal.",
    },
    {
        "full_text": "Is non-destructive mind upload theoretically permitted by physics?",
        "is_bounty": True,
        "bounty_description": "Looking for a careful argument about whether physics rules out or allows non-destructive mind uploading.",
        "reason": "Specialised epistemic question at the intersection of physics and philosophy of mind — expert reasoning has real value.",
    },
    # --- BOUNTY: NO ---
    {
        "full_text": "The urge to cry is there but it needs a catalyst. Someone be mean to me.",
        "is_bounty": False,
        "bounty_description": None,
        "reason": "Social/emotional banter — asking for a playful reaction, not a service or information.",
    },
    {
        "full_text": "What youtube video are you showing to the aliens?",
        "is_bounty": False,
        "bounty_description": None,
        "reason": "Intellectual play / hypothetical game — fun community interaction, not a real request.",
    },
    {
        "full_text": "Suddenly stopped feeling depressed — can I get encouragement in the chat?",
        "is_bounty": False,
        "bounty_description": None,
        "reason": "Seeking social/emotional support in a casual way — not a problem a bounty would solve.",
    },
    {
        "full_text": "What five items would someone place in a summoning circle to summon you?",
        "is_bounty": False,
        "bounty_description": None,
        "reason": "Personality game / social ice-breaker — no real information or service being sought.",
    },
    {
        "full_text": "Is the ship of Theseus a situationship or a delusionship?",
        "is_bounty": False,
        "bounty_description": None,
        "reason": "Witty intellectual banter — framed as a question but the point is the joke, not an answer.",
    },
    {
        "full_text": "what youtube video are you showing to the aliens?",
        "is_bounty": False,
        "bounty_description": None,
        "reason": "Classic TPOT game prompt — communal fun, not a genuine request for help.",
    },
    {
        "full_text": "Ever met someone so hot your psyche registers them as a cartoon character?",
        "is_bounty": False,
        "bounty_description": None,
        "reason": "Observation/vibe tweet fishing for relatable responses — social not informational.",
    },
]

# %%
# === Load problem tweet ids ===

with PROBLEM_CLASSIFICATION_PATH.open("r") as f:
    classification_data = json.load(f)

problem_ids = {
    r["tweet_id"]
    for r in classification_data["results"]
    if r.get("is_problem") is True
}
print(f"Problem tweet ids: {len(problem_ids):,}")

# %%
# === Build tweet text lookup from user histories ===

with USER_HISTORIES_PATH.open("r") as f:
    histories_data = json.load(f)

problem_tweets: list[dict] = []
for uname, user in histories_data["users"].items():
    for p in user.get("problems", []):
        tid = int(p["tweet_id"])
        if tid in problem_ids:
            problem_tweets.append({
                "tweet_id": tid,
                "username": uname,
                "full_text": p.get("full_text", ""),
            })

print(f"Problem tweets loaded: {len(problem_tweets):,}")

# %%
# === Load existing results for resumability ===

existing_results: dict[int, dict] = {}
if BOUNTY_CLASSIFICATION_PATH.exists():
    with BOUNTY_CLASSIFICATION_PATH.open("r") as f:
        existing_data = json.load(f)
    for r in existing_data.get("results", []):
        existing_results[int(r["tweet_id"])] = r
    print(f"Resuming: {len(existing_results):,} already classified")

tweets_to_classify = [t for t in problem_tweets if int(t["tweet_id"]) not in existing_results]
print(f"Remaining to classify: {len(tweets_to_classify):,}")

# %%
# === DeepSeek-Reasoner bounty classification ===

from openai import OpenAI  # noqa: E402

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

# Build system prompt with few-shot examples
_FEW_SHOT_JSON = json.dumps(FEW_SHOT_EXAMPLES, ensure_ascii=False, indent=2)

SYSTEM_PROMPT = f"""You classify Twitter problem-posts as "bounty-worthy": would the poster plausibly pay a small bounty to get a genuinely useful answer?

BOUNTY-WORTHY — the post is a request where a trusted peer network adds real value:
• Recommendations: specific book/tool/product requests where peer curation beats generic search.
• Experience-sharing: "has anyone done X" or "did Y work for you" — lived experience has signal.
• Specialised expertise: deep technical, medical, legal, philosophical questions where an expert answer is genuinely valuable.
• Practical problems: actionable help needed (logistics, career, health, engineering).

NOT BOUNTY-WORTHY:
• Social/emotional banter: seeking laughs, reactions, or emotional play.
• Personality games / hypotheticals posed as fun prompts.
• Vibe tweets fishing for relatable replies.
• Casual community check-ins ("what are you all doing this weekend?").

When a post IS bounty-worthy, write a concise 1-sentence bounty_description framing what is being sought, e.g. "Looking for personal experience with X" or "Looking for a detailed argument why Y".

Few-shot labeled examples:
{_FEW_SHOT_JSON}

Return ONLY valid JSON — no markdown fences, no prose outside the JSON object.
Schema:
{{
  "results": [
    {{
      "tweet_id": <int>,
      "is_bounty": <bool>,
      "confidence": "high" | "medium" | "low",
      "bounty_description": <string or null>
    }}
  ]
}}"""


def _parse_json_from_text(raw: str) -> dict:
    """Extract a JSON object from raw text, with brace-matching fallback."""
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response. Raw:\n{raw[:500]}")
    return json.loads(match.group(0))


def classify_bounty_batch(tweets: list[dict]) -> list[dict]:
    """Ask DeepSeek-Reasoner whether each problem tweet is bounty-worthy."""
    payload = [
        {"tweet_id": int(t["tweet_id"]), "full_text": t.get("full_text", "")}
        for t in tweets
    ]
    user_prompt = (
        "Classify each tweet. Return EXACTLY one result per input tweet.\n\n"
        f"Tweets:\n{json.dumps(payload, ensure_ascii=False)}"
    )

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # Note: deepseek-reasoner does not support response_format json_object
    )

    raw = response.choices[0].message.content or ""
    parsed = _parse_json_from_text(raw)
    if not isinstance(parsed.get("results"), list):
        raise ValueError(f"Invalid model response shape: {parsed}")

    by_id = {
        int(r["tweet_id"]): r
        for r in parsed["results"]
        if isinstance(r, dict) and "tweet_id" in r
    }
    results = []
    for t in tweets:
        tid = int(t["tweet_id"])
        found = by_id.get(tid, {})
        results.append({
            "tweet_id": tid,
            "is_bounty": bool(found.get("is_bounty", False)),
            "confidence": str(found.get("confidence", "low")),
            "bounty_description": found.get("bounty_description") or None,
        })
    return results


# %%
# === Run classification in parallel batches ===

batches = [
    tweets_to_classify[i: i + BATCH_SIZE]
    for i in range(0, len(tweets_to_classify), BATCH_SIZE)
]
print(f"Batches: {len(batches)} × up to {BATCH_SIZE} tweets, {PARALLEL_CALLS} parallel workers")

completed_new: dict[int, list[dict]] = {}  # batch_idx -> results

with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as executor:
    futures = {
        executor.submit(classify_bounty_batch, batch): idx
        for idx, batch in enumerate(batches)
    }
    for future in tqdm(as_completed(futures), total=len(futures), desc="DeepSeek bounty classify"):
        idx = futures[future]
        try:
            completed_new[idx] = future.result()
        except Exception as exc:
            tqdm.write(f"Batch {idx} failed: {exc}")
            # Fallback: mark as not bounty with low confidence
            completed_new[idx] = [
                {"tweet_id": int(t["tweet_id"]), "is_bounty": False,
                 "confidence": "low", "bounty_description": None}
                for t in batches[idx]
            ]

new_results = [r for idx in sorted(completed_new) for r in completed_new[idx]]

# %%
# === Merge with existing results and save ===

all_results_by_id: dict[int, dict] = {**existing_results}
for r in new_results:
    all_results_by_id[int(r["tweet_id"])] = r

# Preserve original ordering (problem_tweets order)
all_results = [all_results_by_id[int(t["tweet_id"])] for t in problem_tweets if int(t["tweet_id"]) in all_results_by_id]

with BOUNTY_CLASSIFICATION_PATH.open("w") as f:
    json.dump(
        {
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": DEEPSEEK_MODEL,
            "total_problems": len(problem_tweets),
            "results": all_results,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

bounty_count = sum(1 for r in all_results if r.get("is_bounty"))
print(f"\nSaved {len(all_results):,} results to {BOUNTY_CLASSIFICATION_PATH.name}")
print(f"  bounty-worthy:  {bounty_count:,}  ({bounty_count / len(all_results):.1%})")
print(f"  not bounty:     {len(all_results) - bounty_count:,}")

# %%
# === Quick preview of bounty-worthy problems ===

bounties = [r for r in all_results if r.get("is_bounty")]
print(f"\n--- Sample bounty descriptions (first 20) ---")
tweet_text_by_id = {int(t["tweet_id"]): t["full_text"] for t in problem_tweets}
for r in bounties[:20]:
    tid = r["tweet_id"]
    text = tweet_text_by_id.get(tid, "?")[:120]
    desc = r.get("bounty_description") or ""
    conf = r.get("confidence", "?")
    print(f"\n[{conf}] {text}")
    print(f"  → {desc}")

# %%
