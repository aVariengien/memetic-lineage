# Scratchpads Agent Guidelines

## Architecture Principles

### Functional Pipeline
- Data flows through phases, always returns to top layer
- Functions return `Dict[int, T]` keyed by tweet_id
- No hidden state mutations - explicit cache passing and merging

### Phase-Level Parallelism
Parallelize each phase separately, not end-to-end. Benefits:
- Dedupe work before dispatching (images especially)
- Per-phase concurrency tuning based on resource type
- Shared cache between phases
- Clear progress tracking

**Concurrency targets:**
| Phase | Type | Workers | Reason |
|-------|------|---------|--------|
| Seeds (semantic search) | IO | 4 | Moderate API limits |
| Filter trees | CPU | 8 | Pure dict operations |
| Image descriptions | IO | 2 | Groq rate limits |
| Render | CPU | 1 | Fast, string ops |

### Caching
- Use diskcache for large persistent caches (tweet_dict, reply_trees, quote_tweets)
- Image cache uses CSV for simplicity and human-readability
- Always check cache before dispatching parallel work

## Quickstart: Get Oriented Fast

Use this when opening a fresh notebook/script and you want to understand the data shape quickly.

### Orientation Tricks
- Start with `scratchpads/23_serendipity_metric.py` for real usage, not abstractions
- Treat `load_caches()` as the default entrypoint for network-level analyses
- Remember `tweet_dict` and `conversation_trees` are disk-backed (`diskcache.Cache`), not in-memory dicts
- Use a single known tweet id to sanity-check structure before running expensive loops
- Build interaction history only when needed (`get_or_build_interaction_history(tweet_dict)`)
- Keep pair keys canonical with `make_pair_key(user_a, user_b)` (alphabetical lowercase)

### Minimal Cache Load Snippet

```python
from lib.strand_caches import load_caches, get_quote_tweets_dict
from lib.serendipity import get_or_build_interaction_history

# Core caches used in most analyses
tweet_dict, conversation_trees = load_caches(auto_generate=False)

# Optional cache for quote relationships
quote_tweets_dict = get_quote_tweets_dict()

# Optional derived cache (built from tweet_dict if missing)
pair_history = get_or_build_interaction_history(tweet_dict)

print(f"tweets: {len(tweet_dict):,}")
print(f"conversation trees: {len(conversation_trees):,}")
print(f"quote index roots: {len(quote_tweets_dict):,}")
print(f"user pairs in interaction history: {len(pair_history):,}")
```

### Cache Composition (Data Structures)

- `tweet_dict`: `Cache[tweet_id -> tweet_record_dict]`
  - Primary per-tweet store; each value is an enriched tweet dict
- `conversation_trees` (stored in `reply_trees.diskcache`): `Cache[root_or_conversation_id -> ConversationTree]`
  - `ConversationTree` shape:
    - `root: int`
    - `children: Dict[int, List[int]]` (parent tweet id -> child tweet ids)
    - `parents: Dict[int, int]` (child tweet id -> parent tweet id)
- `quote_tweets_dict`: `Cache[quoted_tweet_id -> List[quoting_tweet_id]]`
  - Reverse index for quote traversal
- `pair_history`: `Cache[(user_a, user_b) -> List[Interaction]]` (or plain dict right after build)
  - Pair keys are lowercase sorted tuples
  - Each `Interaction` contains `tweet_id`, `from_user`, `to_user`, `created_at`, `reply_to_tweet_id`, `full_text`

### Example Tweet Object (From `tweet_dict`)

Use this as the expected record shape reference:

```python
tweet_dict[574832598607790080]
{'tweet_id': 574832598607790080,
 'account_id': 9729972,
 'username': 'michaelgarfield',
 'account_display_name': 'Michael Garfield 🔮',
 'created_at': '2015-03-09 07:22:26+00',
 'full_text': '"Stories are for children...I think we\'re past stories. Now we\'re in something more like a game." – @rushkoff on transcending our myths',
 'retweet_count': 0,
 'favorite_count': 0,
 'reply_to_tweet_id': None,
 'reply_to_user_id': None,
 'reply_to_username': None,
 'quoted_tweet_id': None,
 'conversation_id': 574832598607790080,
 'avatar_media_url': 'https://pbs.twimg.com/profile_images/1768691462445715456/4RowAjho_normal.jpg',
 'archive_upload_id': 508}
```

### Error Handling
- Exponential backoff on external API calls (Groq, Supabase)
- For structured output failures (LLM), retry with higher temperature
- `parallel_map_to_dict` logs errors but continues processing

## Key Files

| File | Purpose |
|------|---------|
| `lib/strand_caches.py` | Diskcache loading for tweet_dict, reply_trees, quote_tweets |
| `lib/strand_builder.py` | Strand pipeline: seeds → trees → images → render |
| `lib/strand_rater.py` | LLM-based strand rating |
| `lib/image_describer.py` | Groq image descriptions with retry |
| `lib/parallel.py` | `parallel_map_to_dict` utility |
| `lib/retry.py` | `@with_retry` decorator |

## Conventions

- Type hints on all functions (signatures or short comments)
- No inline explanation comments unless asked
- Prefer `toolz` patterns where applicable
- Tests are integration-style, not exhaustive unit tests

## TODOs

- [ ] Add async variants of parallel utilities for true async IO
- [ ] Consolidate image cache to diskcache format
- [ ] Add structured logging instead of print statements

