# %%
"""
Analysis of problem-resolution data from the 3-month pipeline.
"""

# %%
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    SCRATCHPADS_DIR = Path(__file__).parent
except NameError:
    SCRATCHPADS_DIR = Path.cwd()
    if SCRATCHPADS_DIR.name != "scratchpads":
        SCRATCHPADS_DIR = SCRATCHPADS_DIR / "scratchpads"

DATA_DIR = SCRATCHPADS_DIR / "data" / "problem_resolution" / "3month"
USER_HISTORIES_PATH = DATA_DIR / "user_tweet_histories.json"

# %%
# === Load data ===

with USER_HISTORIES_PATH.open("r") as f:
    data = json.load(f)

users = data["users"]
print(f"Loaded {len(users)} users")

# Collect all problem tweets across users
all_problems = []
for username, user_data in users.items():
    for p in user_data.get("problems", []):
        all_problems.append(p)

total_tweets = sum(user_data["tweet_count"] for user_data in users.values())
top_level_tweets = sum(
    1 for user_data in users.values()
    for t in user_data.get("tweets", []) if t.get("is_top_level")
) + len(all_problems)  # problems are all top-level
print(f"Total tweets: {total_tweets:,}")
print(f"Top-level tweets: {top_level_tweets:,} ({top_level_tweets/total_tweets:.1%})")
no_reply = sum(1 for p in all_problems if p.get("outcome_reason") == "no reply")
print(f"Total problem tweets: {len(all_problems)} ({len(all_problems)/top_level_tweets:.1%} of top-level)")
print(f"  without any reply: {no_reply} ({no_reply/len(all_problems):.1%})")

# %%
# === Print few-shot outcome classification examples ===

FEW_SHOT_OUTCOME_PATH = DATA_DIR.parent / "few_shot_outcome_classification.json"
with FEW_SHOT_OUTCOME_PATH.open("r") as f:
    few_shot_data = json.load(f)

print("=" * 80)
print("FEW-SHOT OUTCOME CLASSIFICATION EXAMPLES")
print(f"Labels: {', '.join(few_shot_data['labeling_schema']['labels'])}")
print(f"Sampling: {few_shot_data['sampling']['method']}")
print("=" * 80)

for i, ex in enumerate(few_shot_data["examples"], 1):
    label = ex["label"]
    colors = {"resolved_by_community": "✅", "serious_attempt": "🔶", "unresolved_by_community": "❌"}
    icon = colors.get(label, "?")

    print(f"\n{'─' * 80}")
    print(f"Example {i}/{len(few_shot_data['examples'])}  {icon} {label}")
    print(f"Tweet ID: {ex['tweet_id']}")
    print(f"Rationale: {ex['rationale']}")
    print(f"{'─' * 40}")
    for line in ex["thread"].split("\n"):
        print(f"  {line}")

print(f"\n{'=' * 80}")

# %%
# === Distribution of likes and retweets on problem tweets ===

likes = np.array([p.get("favorite_count", 0) for p in all_problems])
rts = np.array([p.get("retweet_count", 0) for p in all_problems])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(likes[likes > 0], bins=np.logspace(0, np.log10(likes.max() + 1), 40),
             color="#4a90d9", edgecolor="white", linewidth=0.5)
axes[0].set_xscale("log")
axes[0].set_xlabel("Likes (log scale)")
axes[0].set_ylabel("Count")
axes[0].set_title("Likes Distribution (log scale, excluding 0)")

axes[1].hist(rts[rts > 0], bins=np.logspace(0, np.log10(rts.max() + 1), 40),
             color="#e07b53", edgecolor="white", linewidth=0.5)
axes[1].set_xscale("log")
axes[1].set_xlabel("Retweets (log scale)")
axes[1].set_ylabel("Count")
axes[1].set_title("Retweets Distribution (log scale, excluding 0)")

plt.tight_layout()
plt.show()

# %%
# === Summary statistics ===

print("Likes on problem tweets:")
print(f"  count:  {len(likes)}")
print(f"  zero:   {(likes == 0).sum()} ({(likes == 0).mean():.1%})")
print(f"  median: {np.median(likes):.0f}")
print(f"  mean:   {np.mean(likes):.1f}")
print(f"  p75:    {np.percentile(likes, 75):.0f}")
print(f"  p90:    {np.percentile(likes, 90):.0f}")
print(f"  p99:    {np.percentile(likes, 99):.0f}")
print(f"  max:    {likes.max()}")

print(f"\nRetweets on problem tweets:")
print(f"  count:  {len(rts)}")
print(f"  zero:   {(rts == 0).sum()} ({(rts == 0).mean():.1%})")
print(f"  median: {np.median(rts):.0f}")
print(f"  mean:   {np.mean(rts):.1f}")
print(f"  p75:    {np.percentile(rts, 75):.0f}")
print(f"  p90:    {np.percentile(rts, 90):.0f}")
print(f"  p99:    {np.percentile(rts, 99):.0f}")
print(f"  max:    {rts.max()}")

# %%
# === Problems per day and resolved per day — summary tables ===

import pandas as pd

df = pd.DataFrame(all_problems)
df["date"] = pd.to_datetime(df["created_at"]).dt.date
df["is_resolved"] = df["outcome_label"] == "resolved_by_community"
df["is_engaged"] = df["outcome_label"].isin({"resolved_by_community", "serious_attempt"})

# Daily aggregation
daily = df.groupby("date").agg(
    problems=("tweet_id", "count"),
    resolved=("is_resolved", "sum"),
).reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

print("=== Daily problems and resolved (first 15 days) ===")
print(daily.head(15).to_string(index=False))

print(f"\n=== Summary ===")
print(f"  Total days:           {len(daily)}")
print(f"  Total problems:       {daily['problems'].sum()}")
print(f"  Total resolved:       {int(daily['resolved'].sum())}")
n_users = len(df["username"].unique())
print(f"  Avg problems/day:     {daily['problems'].mean():.1f}")
print(f"  Avg problems/day/user:{daily['problems'].mean() / n_users:.2f}")
print(f"  Avg resolved/day:     {daily['resolved'].mean():.1f}")
print(f"  Overall resolve rate: {df['is_resolved'].mean():.1%}")
print(f"  Overall engaged rate: {df['is_engaged'].mean():.1%}  (resolved + serious attempt)")

# %%
# === Random examples: 3 problems + threads per category ===

PROBLEM_THREADS_PATH = DATA_DIR / "problem_threads.json"
with PROBLEM_THREADS_PATH.open("r") as f:
    problem_threads_data = json.load(f)

thread_by_tweet_id = {
    str(item["tweet_id"]): item.get("thread", "")
    for item in problem_threads_data.get("threads", [])
}

category_to_labels = {
    "resolved": {"resolved_by_community"},
    "engaged": {"resolved_by_community", "serious_attempt"},
    "unresolved": {"unresolved_by_community"},
}

# Compute estimated views here if this cell is run before the later views section.
if "estimated_views" not in df.columns:
    from lib.problem_analysis import predict_views

    df["estimated_views"] = df.apply(
        lambda row: predict_views(row.get("favorite_count", 0), row.get("retweet_count", 0)), axis=1
    )

rng = np.random.default_rng()
n_examples = 3

print("=" * 100)
print("RANDOM PROBLEM + THREAD EXAMPLES")
print("=" * 100)

for category_name, labels in category_to_labels.items():
    subset = df[df["outcome_label"].isin(labels)].copy()
    if subset.empty:
        print(f"\n[{category_name.upper()}] No examples found.")
        continue

    sample_n = min(n_examples, len(subset))
    sample_idx = rng.choice(subset.index.to_numpy(), size=sample_n, replace=False)
    sampled = subset.loc[sample_idx]

    print(f"\n{'-' * 100}")
    print(f"{category_name.upper()} — showing {sample_n} random examples")
    print(f"{'-' * 100}")

    for i, (_, row) in enumerate(sampled.iterrows(), 1):
        tweet_id = str(row["tweet_id"])
        thread_text = thread_by_tweet_id.get(tweet_id, "(thread not found)")

        print(f"\nExample {i}/{sample_n}")
        print(f"  user:      @{row['username']}")
        print(f"  tweet_id:  {tweet_id}")
        print(f"  outcome:   {row['outcome_label']}")
        print(f"  est_views: {row.get('estimated_views', np.nan):,.0f}")
        print(f"  problem:   {row.get('full_text', '')}")
        print("  thread:")
        for line in thread_text.split("\n"):
            print(f"    {line}")

# %%
# === Per-user summary table ===

# Get total tweet count per user (problems + non-problems)
user_tweet_counts = {
    username: user_data["tweet_count"]
    for username, user_data in users.items()
}

user_summary = df.groupby("username").agg(
    problems=("tweet_id", "count"),
    resolved=("is_resolved", "sum"),
    engaged=("is_engaged", "sum"),
).reset_index()
user_summary["tweet_count"] = user_summary["username"].str.lower().map(user_tweet_counts)
user_summary["problem_rate"] = user_summary["problems"] / user_summary["tweet_count"]
user_summary["resolve_rate"] = user_summary["resolved"] / user_summary["problems"]
user_summary["engaged_rate"] = user_summary["engaged"] / user_summary["problems"]
user_summary = user_summary.sort_values("problems", ascending=False).reset_index(drop=True)

print("=== Per-user problem counts (top 20 problem posters) ===")
fmt = user_summary.head(20).copy()
fmt["prob/day"] = (fmt["problems"] / 92).map("{:.2f}".format)
fmt["problem_rate"] = fmt["problem_rate"].map("{:.1%}".format)
fmt["resolve_rate"] = fmt["resolve_rate"].map("{:.1%}".format)
fmt["engaged_rate"] = fmt["engaged_rate"].map("{:.1%}".format)
print(fmt.to_string(index=False))

# %%
# === Box plots (boîte à moustaches) ===

# Compute problems per day per user (total problems / number of days in range)
n_days_in_range = (pd.Timestamp("2024-09-01") - pd.Timestamp("2024-06-01")).days  # 92 days
user_summary["problems_per_day"] = user_summary["problems"] / n_days_in_range

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Filter for users with ≥3 problems (for rate box plots)
enough_problems = user_summary["problems"] >= 3
n_filtered = enough_problems.sum()

# 1. Problems per user
axes[0, 0].boxplot(user_summary["problems"], vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#4a90d9", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2))
axes[0, 0].set_title("Problems per User\n(3-month total)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_xticks([])

# 2. Problems per day per user
axes[0, 1].boxplot(user_summary["problems_per_day"], vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#3498db", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2))
axes[0, 1].set_title("Problems per Day per User")
axes[0, 1].set_ylabel("Count / day")
axes[0, 1].set_xticks([])

# 3. Problem rate per user (problems / total tweets)
axes[0, 2].boxplot(user_summary["problem_rate"], vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#e07b53", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2))
axes[0, 2].set_title("Problem Rate per User\n(problems / tweets)")
axes[0, 2].set_ylabel("Rate")
axes[0, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
axes[0, 2].set_xticks([])

# 4. Resolve rate per user
axes[1, 0].boxplot(user_summary.loc[enough_problems, "resolve_rate"], vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#27ae60", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2))
axes[1, 0].set_title(f"Resolve Rate per User\n(≥3 problems, n={n_filtered})")
axes[1, 0].set_ylabel("Rate")
axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
axes[1, 0].set_xticks([])

# 5. Engaged rate per user (resolved + serious attempt)
axes[1, 1].boxplot(user_summary.loc[enough_problems, "engaged_rate"], vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#f39c12", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2))
axes[1, 1].set_title(f"Engaged Rate per User\n(resolved + serious, ≥3 problems)")
axes[1, 1].set_ylabel("Rate")
axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
axes[1, 1].set_xticks([])

# 6. Side-by-side comparison: resolve vs engaged
bp = axes[1, 2].boxplot(
    [user_summary.loc[enough_problems, "resolve_rate"],
     user_summary.loc[enough_problems, "engaged_rate"]],
    vert=True, patch_artist=True, medianprops=dict(color="red", linewidth=2),
)
bp["boxes"][0].set_facecolor("#27ae60"); bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor("#f39c12"); bp["boxes"][1].set_alpha(0.6)
axes[1, 2].set_xticklabels(["Resolved", "Engaged"], fontsize=10)
axes[1, 2].set_title("Resolved vs Engaged Rate")
axes[1, 2].set_ylabel("Rate")
axes[1, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.suptitle("Distribution Box Plots", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# === Timeline: problems and resolved per day with rolling average ===

window = 7  # 7-day rolling average
daily["problems_7d"] = daily["problems"].rolling(window, center=True, min_periods=1).mean()
daily["resolved_7d"] = daily["resolved"].rolling(window, center=True, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(daily["date"], daily["problems"], color="#4a90d9", alpha=0.3, label="Problems (daily)")
ax.bar(daily["date"], daily["resolved"], color="#27ae60", alpha=0.3, label="Resolved (daily)")
ax.plot(daily["date"], daily["problems_7d"], color="#2c5f8a", linewidth=2, label="Problems (7d avg)")
ax.plot(daily["date"], daily["resolved_7d"], color="#1a7a3a", linewidth=2, label="Resolved (7d avg)")

ax.set_xlabel("Date")
ax.set_ylabel("Count")
ax.set_title("Problem Tweets and Resolved per Day (all users)")
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# %%
# === Timeline for top 10 users with most problems ===

top10_users = user_summary.head(10)["username"].tolist()

fig, axes = plt.subplots(5, 2, figsize=(16, 20), sharex=True)
axes_flat = axes.flatten()

for i, username in enumerate(top10_users):
    ax = axes_flat[i]
    user_df = df[df["username"] == username].copy()
    user_daily = user_df.groupby("date").agg(
        problems=("tweet_id", "count"),
        resolved=("is_resolved", "sum"),
    ).reset_index()
    user_daily["date"] = pd.to_datetime(user_daily["date"])
    user_daily = user_daily.sort_values("date")

    # Reindex to fill missing days with 0
    full_range = pd.date_range(daily["date"].min(), daily["date"].max())
    user_daily = user_daily.set_index("date").reindex(full_range, fill_value=0).reset_index()
    user_daily.rename(columns={"index": "date"}, inplace=True)

    user_daily["problems_7d"] = user_daily["problems"].rolling(window, center=True, min_periods=1).mean()
    user_daily["resolved_7d"] = user_daily["resolved"].rolling(window, center=True, min_periods=1).mean()

    total_p = user_df.shape[0]
    total_r = int(user_df["is_resolved"].sum())

    ax.bar(user_daily["date"], user_daily["problems"], color="#4a90d9", alpha=0.3)
    ax.bar(user_daily["date"], user_daily["resolved"], color="#27ae60", alpha=0.3)
    ax.plot(user_daily["date"], user_daily["problems_7d"], color="#2c5f8a", linewidth=1.5)
    ax.plot(user_daily["date"], user_daily["resolved_7d"], color="#1a7a3a", linewidth=1.5)
    ax.set_title(f"@{username}  ({total_p} problems, {total_r} resolved)")
    ax.set_ylabel("Count")

# Shared legend
handles = [
    plt.Line2D([], [], color="#4a90d9", alpha=0.3, linewidth=6, label="Problems (daily)"),
    plt.Line2D([], [], color="#27ae60", alpha=0.3, linewidth=6, label="Resolved (daily)"),
    plt.Line2D([], [], color="#2c5f8a", linewidth=2, label="Problems (7d avg)"),
    plt.Line2D([], [], color="#1a7a3a", linewidth=2, label="Resolved (7d avg)"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Top 10 Users by Problem Count — Daily Timeline", fontsize=14, y=1.01)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# %%
# === Estimate views for each problem tweet ===

from lib.problem_analysis import predict_views

df["estimated_views"] = df.apply(
    lambda row: predict_views(row.get("favorite_count", 0), row.get("retweet_count", 0)), axis=1
)

print(f"Estimated views — median: {df['estimated_views'].median():.0f}, mean: {df['estimated_views'].mean():.0f}, "
      f"max: {df['estimated_views'].max():.0f}")

# %%
# === Views vs resolution: scatter + box plot by outcome ===

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: views vs resolved (jittered binary y)
resolved_mask = df["is_resolved"]
jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(df))

axes[0].scatter(df.loc[~resolved_mask, "estimated_views"].clip(lower=1), jitter[~resolved_mask.values],
                alpha=0.3, s=10, color="#e07b53", label="Not resolved")
axes[0].scatter(df.loc[resolved_mask, "estimated_views"].clip(lower=1), 1 + jitter[resolved_mask.values],
                alpha=0.3, s=10, color="#27ae60", label="Resolved")
axes[0].set_xscale("log")
axes[0].set_xlabel("Estimated Views (log scale)")
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(["Not resolved", "Resolved"])
axes[0].set_title("Estimated Views by Resolution Status")
axes[0].legend()

# Box plot: views by outcome label
outcome_labels = ["resolved_by_community", "serious_attempt", "unresolved_by_community"]
outcome_colors = ["#27ae60", "#f39c12", "#e07b53"]
views_by_outcome = [
    df.loc[df["outcome_label"] == label, "estimated_views"].clip(lower=1).values
    for label in outcome_labels
]

bp = axes[1].boxplot(views_by_outcome, vert=True, patch_artist=True,
                     medianprops=dict(color="red", linewidth=2))
for patch, color in zip(bp["boxes"], outcome_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[1].set_xticklabels(["Resolved", "Serious\nAttempt", "Unresolved"], fontsize=10)
axes[1].set_ylabel("Estimated Views")
axes[1].set_title("Estimated Views by Outcome Category")
axes[1].set_yscale("log")

plt.tight_layout()
plt.show()

# %%
# === Resolve rate by views bucket ===

df["views_bucket"] = pd.cut(
    df["estimated_views"],
    bins=[0, 100, 500, 1000, 5000, 10000, 50000, float("inf")],
    labels=["0-100", "100-500", "500-1K", "1K-5K", "5K-10K", "10K-50K", "50K+"],
)

bucket_stats = df.groupby("views_bucket", observed=True).agg(
    count=("tweet_id", "count"),
    resolved=("is_resolved", "sum"),
    engaged=("is_engaged", "sum"),
).reset_index()
bucket_stats["resolve_rate"] = bucket_stats["resolved"] / bucket_stats["count"]
bucket_stats["engaged_rate"] = bucket_stats["engaged"] / bucket_stats["count"]

print("=== Resolve / engaged rate by estimated views bucket ===")
fmt_b = bucket_stats.copy()
fmt_b["resolve_rate"] = fmt_b["resolve_rate"].map("{:.1%}".format)
fmt_b["engaged_rate"] = fmt_b["engaged_rate"].map("{:.1%}".format)
print(fmt_b.to_string(index=False))

fig, ax1 = plt.subplots(figsize=(10, 5))

x = range(len(bucket_stats))
ax1.bar(x, bucket_stats["count"], color="#4a90d9", alpha=0.5, label="Tweet count")
ax1.set_xlabel("Estimated Views Bucket")
ax1.set_ylabel("Number of Problem Tweets", color="#4a90d9")
ax1.set_xticks(x)
ax1.set_xticklabels(bucket_stats["views_bucket"], rotation=30)

ax2 = ax1.twinx()
ax2.plot(x, bucket_stats["engaged_rate"], color="#f39c12", marker="s", linewidth=2, label="Engaged rate")
ax2.plot(x, bucket_stats["resolve_rate"], color="#27ae60", marker="o", linewidth=2, label="Resolve rate")
ax2.set_ylabel("Rate", color="#333")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

fig.suptitle("Resolve & Engaged Rate vs Estimated Views", fontsize=13)
fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.88))
plt.tight_layout()
plt.show()

# %%
# === Estimate p: probability that a single view resolves the problem ===
# Model: P(resolved) ≈ p * #views  (first-order approx of 1-(1-p)^views)
# Per-user: p_user = resolved_count / sum(views of all problem tweets)

user_p = df.groupby("username").agg(
    problems=("tweet_id", "count"),
    resolved=("is_resolved", "sum"),
    engaged=("is_engaged", "sum"),
    total_views=("estimated_views", "sum"),
    mean_views=("estimated_views", "mean"),
).reset_index()
user_p["resolve_rate"] = user_p["resolved"] / user_p["problems"]
user_p["engaged_rate"] = user_p["engaged"] / user_p["problems"]
user_p["p_resolve"] = user_p["resolved"] / user_p["total_views"]
user_p["p_engaged"] = user_p["engaged"] / user_p["total_views"]
user_p = user_p.sort_values("problems", ascending=False).reset_index(drop=True)

print("=== Per-user p (probability a view resolves / engages) ===")
fmt_p = user_p.copy()
fmt_p["resolve_rate"] = fmt_p["resolve_rate"].map("{:.1%}".format)
fmt_p["engaged_rate"] = fmt_p["engaged_rate"].map("{:.1%}".format)
fmt_p["p_resolve"] = fmt_p["p_resolve"].map("{:.2e}".format)
fmt_p["p_engaged"] = fmt_p["p_engaged"].map("{:.2e}".format)
fmt_p["mean_views"] = fmt_p["mean_views"].map("{:.0f}".format)
fmt_p["total_views"] = fmt_p["total_views"].map("{:.0f}".format)
print(fmt_p.to_string(index=False))

# %%
# === Box plot of p_per_view + scatter: mean views vs p ===

user_p_filtered = user_p[user_p["problems"] >= 3].copy()

# Box plot: p_resolve vs p_engaged side by side
fig, ax = plt.subplots(figsize=(6, 5))
bp = ax.boxplot(
    [user_p_filtered["p_resolve"] * 1e6, user_p_filtered["p_engaged"] * 1e6],
    vert=True, patch_artist=True, medianprops=dict(color="red", linewidth=2),
)
bp["boxes"][0].set_facecolor("#27ae60"); bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor("#f39c12"); bp["boxes"][1].set_alpha(0.6)
ax.set_xticklabels(["p_resolve", "p_engaged"], fontsize=11)
ax.set_title(f"p per view (×10⁻⁶) across users  (≥3 problems, n={len(user_p_filtered)})")
ax.set_ylabel("p × 10⁻⁶")
plt.tight_layout()
plt.show()

# %%
# === Scatter: mean views vs p (one plot per metric) ===

def _scatter_p(ax, data, p_col, rate_col, ylabel, title, cbar_label):
    """Helper for p-vs-views scatter with labels on all points."""
    sc = ax.scatter(
        data["mean_views"], data[p_col] * 1e6,
        s=data["problems"] * 3, alpha=0.6,
        c=data[rate_col], cmap="RdYlGn", edgecolors="black", linewidth=0.3,
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Mean Estimated Views per Problem Tweet")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.85)

    # Arrow annotations only for a few hand-picked usernames
    highlight_users = {"visakanv", "eigenrobot", "eshear", "DanielleFong"}
    highlight_offsets = {
        "visakanv":     (20, 20),
        "eigenrobot":   (20, -25),
        "eshear":       (-30, 20),
        "DanielleFong": (20, -20),
    }

    for _, row in data.iterrows():
        xv = row["mean_views"]
        yv = row[p_col] * 1e6
        uname = row["username"]

        if uname in highlight_users:
            dx, dy = highlight_offsets[uname]
            ax.annotate(f"@{uname}", (xv, yv), fontsize=8,
                        xytext=(dx, dy), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.8, shrinkB=3),
                        ha="center", va="center", zorder=4)
        else:
            # Username inside the bubble
            ax.text(xv, yv, f"@{uname}", fontsize=8, ha="center", va="center",
                    alpha=0.7, zorder=3)


fig, axes = plt.subplots(2, 1, figsize=(12, 12))

_scatter_p(axes[0], user_p_filtered, "p_resolve", "resolve_rate",
           "p_resolve × 10⁻⁶", "p_resolve vs Mean Views  (dot size = #problems)", "Resolve rate")

_scatter_p(axes[1], user_p_filtered, "p_engaged", "engaged_rate",
           "p_engaged × 10⁻⁶", "p_engaged vs Mean Views  (dot size = #problems)", "Engaged rate")

plt.tight_layout()
plt.show()

# %%
# === Global p estimate ===

global_resolved = df["is_resolved"].sum()
global_engaged = df["is_engaged"].sum()
global_views = df["estimated_views"].sum()
global_p_resolve = global_resolved / global_views
global_p_engaged = global_engaged / global_views

print(f"=== Global p estimates ===")
print(f"  Total problems:   {len(df)}")
print(f"  Total resolved:   {int(global_resolved)}")
print(f"  Total engaged:    {int(global_engaged)}  (resolved + serious attempt)")
print(f"  Total est views:  {global_views:,.0f}")
print(f"\n  p_resolve:  {global_p_resolve:.2e}  (~1 in {1/global_p_resolve:,.0f} views)")
print(f"  p_engaged:  {global_p_engaged:.2e}  (~1 in {1/global_p_engaged:,.0f} views)")

# %%
# === Histograms (users with >5 posts): problems/day, views-to-resolve, views-to-engage ===

# Reuse per-user aggregates and keep only users active enough for stable estimates.
user_hist = user_p.merge(
    user_summary[["username", "tweet_count", "problems_per_day"]],
    on="username",
    how="left",
)
user_hist = user_hist[user_hist["tweet_count"] > 5].copy()

# Inverse per-view probabilities: expected views needed for one resolve / one engagement.
user_hist["views_per_resolve"] = np.where(
    user_hist["p_resolve"] > 0, 1 / user_hist["p_resolve"], np.nan
)
user_hist["views_per_engaged"] = np.where(
    user_hist["p_engaged"] > 0, 1 / user_hist["p_engaged"], np.nan
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

problems_per_day_values = user_hist["problems_per_day"].dropna()
views_per_resolved_values = user_hist["views_per_resolve"].replace([np.inf, -np.inf], np.nan).dropna()
views_per_engaged_values = user_hist["views_per_engaged"].replace([np.inf, -np.inf], np.nan).dropna()

metrics = [
    ("Problems / Day", "Problems / day", problems_per_day_values, "#4A90D9"),
    ("Views before 50% problem resolved", "Views", views_per_resolved_values, "#27AE60"),
    ("Views before 50% problem engaged", "Views", views_per_engaged_values, "#F39C12"),
]

for i, (title, xlabel, values, color) in enumerate(metrics):
    if len(values) == 0:
        axes[i].set_title(title)
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel("Users")
        continue

    if i == 0:
        bins = 20
        median_label = f"Median: {np.median(values):.2f}"
    else:
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 20)
        median_label = f"Median: {np.median(values):,.0f}"
        axes[i].set_xscale("log")

    axes[i].hist(
        values,
        bins=bins,
        color=color,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[i].axvline(
        np.median(values),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=median_label,
    )
    axes[i].legend()
    axes[i].set_title(title)
    axes[i].set_xlabel(xlabel)
    axes[i].set_ylabel("Users")

plt.suptitle("Community Archive Dataset", fontsize=18, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# === Export text files: all problems, and 1000 random problems with threads ===

EXPORT_DIR = DATA_DIR
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

all_problems_export_path = EXPORT_DIR / "all_problems_with_users.txt"
random_threads_export_path = EXPORT_DIR / "random_100_problems_with_threads.txt"
rng = np.random.default_rng(42)
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 1) Export all problems with user + basic metadata
with all_problems_export_path.open("w", encoding="utf-8") as f:
    f.write("All Problem Tweets with Users\n")
    f.write("=" * 100 + "\n\n")
    for i, row in enumerate(shuffled_df.itertuples(index=False), 1):
        f.write(f"{i}\n")
        f.write(f"@{row.username}\n")
        f.write(f"{getattr(row, 'full_text', '')}\n")

# 2) Export 1000 random problems with thread text
PROBLEM_THREADS_PATH = DATA_DIR / "problem_threads.json"
with PROBLEM_THREADS_PATH.open("r") as f:
    problem_threads_data = json.load(f)

thread_by_tweet_id = {
    str(item["tweet_id"]): item.get("thread", "")
    for item in problem_threads_data.get("threads", [])
}

sample_n = min(700, len(shuffled_df))
sampled_1000 = shuffled_df.head(sample_n).copy()

with random_threads_export_path.open("w", encoding="utf-8") as f:
    import re

    f.write("Random Sample of Problems with Threads\n")
    f.write("=" * 100 + "\n")
    f.write(f"sample_size: {sample_n}\n\n")
    for i, row in enumerate(sampled_1000.itertuples(index=False), 1):
        tweet_id = str(row.tweet_id)
        thread_text = thread_by_tweet_id.get(tweet_id, "(thread not found)")
        thread_text = re.sub(r"\s*\(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\):", ":", thread_text)
        f.write(f"[{i}]\n")
        f.write(f"@{row.username}\n")
        f.write(f"outcome: {row.outcome_label}\n")
        f.write("thread:\n")
        f.write(thread_text + "\n")

print(f"Exported all problems to: {all_problems_export_path}")
print(f"Exported random sample (n={sample_n}) with threads to: {random_threads_export_path}")

# %%
