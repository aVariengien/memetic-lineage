# %%
"""
Analysis of bounty-worthy problems: price distribution, resolution rates by
dollar group, and "views before 50% resolved" broken down by price bucket.
"""

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    SCRATCHPADS_DIR = Path(__file__).parent
except NameError:
    SCRATCHPADS_DIR = Path.cwd()
    if SCRATCHPADS_DIR.name != "scratchpads":
        SCRATCHPADS_DIR = SCRATCHPADS_DIR / "scratchpads"

if str(SCRATCHPADS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCHPADS_DIR))

from lib.problem_analysis import predict_views  # noqa: E402

BOUNTY_DIR = SCRATCHPADS_DIR / "data" / "problem_resolution" / "bounties"
BOUNTY_PRICE_PATH = BOUNTY_DIR / "bounty-price-prediction-eval-deepseek-bounty_classification_deepseek.json"
USER_HISTORIES_PATH = SCRATCHPADS_DIR / "data" / "problem_resolution" / "3month" / "user_tweet_histories.json"

N_DAYS = 92  # Jun–Aug 2024

# %%
# === Dollar groups spanning ~3 orders of magnitude ===

DOLLAR_BINS = [0, 10, 30, 100, 300, float("inf")]
DOLLAR_LABELS = ["< $10", "$10–$30", "$30–$100", "$100–$300", "$300+"]
DOLLAR_COLORS = ["#8ecae6", "#4a90d9", "#27ae60", "#f39c12", "#e07b53"]

# %%
# === Load and merge data ===

with BOUNTY_PRICE_PATH.open("r") as f:
    bounty_data = json.load(f)

with USER_HISTORIES_PATH.open("r") as f:
    histories_data = json.load(f)

# Build flat problem lookup (tweet_id → metadata)
problem_meta: dict[str, dict] = {}
for uname, user in histories_data["users"].items():
    for p in user.get("problems", []):
        problem_meta[str(p["tweet_id"])] = {
            "username": uname,
            "outcome_label": p.get("outcome_label", "unknown"),
            "favorite_count": p.get("favorite_count", 0) or 0,
            "retweet_count": p.get("retweet_count", 0) or 0,
            "created_at": str(p.get("created_at", ""))[:10],
            "full_text": p.get("full_text", ""),
        }

# Build main DataFrame
rows = []
for item in bounty_data["items"]:
    sid = item["story_id"]
    meta = problem_meta.get(sid, {})
    likes = meta.get("favorite_count", 0)
    rts = meta.get("retweet_count", 0)
    rows.append({
        "tweet_id": sid,
        "username": meta.get("username", "unknown"),
        "price_usd": item["point_estimate_usd"],
        "problem_description": item.get("problem_description", ""),
        "full_text": meta.get("full_text", ""),
        "outcome_label": meta.get("outcome_label", "unknown"),
        "favorite_count": likes,
        "retweet_count": rts,
        "created_at": meta.get("created_at", ""),
        "estimated_views": predict_views(likes, rts),
    })

df = pd.DataFrame(rows)
df["is_resolved"] = df["outcome_label"] == "resolved_by_community"
df["is_engaged"] = df["outcome_label"].isin({"resolved_by_community", "serious_attempt"})
df["dollar_group"] = pd.cut(
    df["price_usd"],
    bins=DOLLAR_BINS,
    labels=DOLLAR_LABELS,
    right=False,
)

print(f"Bounty problems: {len(df):,}  across {df['username'].nunique()} users")
print(f"Price — median: ${df['price_usd'].median():.0f}  mean: ${df['price_usd'].mean():.0f}"
      f"  min: ${df['price_usd'].min():.0f}  max: ${df['price_usd'].max():.0f}")
print(f"\nDollar group counts:")
print(df["dollar_group"].value_counts(sort=False).to_string())

# %%
# === Per-user aggregate ===

user_agg = df.groupby("username").agg(
    bounties=("tweet_id", "count"),
    total_price=("price_usd", "sum"),
    mean_price=("price_usd", "mean"),
    resolved=("is_resolved", "sum"),
    engaged=("is_engaged", "sum"),
    total_views=("estimated_views", "sum"),
).reset_index()

resolved_price = df[df["is_resolved"]].groupby("username")["price_usd"].sum().rename("resolved_price")
user_agg = user_agg.join(resolved_price, on="username").fillna({"resolved_price": 0})

user_agg["price_per_day"] = user_agg["total_price"] / N_DAYS
user_agg["resolved_price_per_day"] = user_agg["resolved_price"] / N_DAYS
user_agg["bounties_per_day"] = user_agg["bounties"] / N_DAYS
user_agg["resolve_rate"] = user_agg["resolved"] / user_agg["bounties"]
user_agg["engaged_rate"] = user_agg["engaged"] / user_agg["bounties"]
user_agg["p_resolve"] = np.where(user_agg["total_views"] > 0,
                                  user_agg["resolved"] / user_agg["total_views"], np.nan)
user_agg["views_per_resolve"] = np.where(user_agg["p_resolve"] > 0,
                                          1 / user_agg["p_resolve"], np.nan)
user_agg["p_engaged"] = np.where(user_agg["total_views"] > 0,
                                  user_agg["engaged"] / user_agg["total_views"], np.nan)
user_agg["views_per_engaged"] = np.where(user_agg["p_engaged"] > 0,
                                          1 / user_agg["p_engaged"], np.nan)

print("\n=== Per-user bounty summary (top 20 by bounties) ===")
fmt = user_agg.sort_values("bounties", ascending=False).head(20).copy()
fmt["mean_price"] = fmt["mean_price"].map("${:.0f}".format)
fmt["price_per_day"] = fmt["price_per_day"].map("${:.1f}/d".format)
fmt["resolve_rate"] = fmt["resolve_rate"].map("{:.0%}".format)
print(fmt[["username", "bounties", "mean_price", "price_per_day", "resolve_rate"]].to_string(index=False))

# %%
# === Per-user metric histograms (2×3 grid) ===

_user_hist = user_agg.copy()
_vpr = _user_hist["views_per_resolve"].replace([np.inf, -np.inf], np.nan).dropna()
_vpe = _user_hist["views_per_engaged"].replace([np.inf, -np.inf], np.nan).dropna()

_metrics = [
    ("bounties_per_day",   "Bounty problems / day",        "#4A90D9", False),
    ("mean_price",         "Avg bounty price per user ($)", "#9B59B6", False),
    ("resolve_rate",       "Resolve rate",                  "#27AE60", False),
    ("engaged_rate",       "Engaged rate",                  "#F39C12", False),
    (None,                 "Views before 50% resolved",     "#27AE60", True),   # views_per_resolve
    (None,                 "Views before 50% engaged",      "#F39C12", True),   # views_per_engaged
]
_series = [_user_hist["bounties_per_day"], _user_hist["mean_price"],
           _user_hist["resolve_rate"],     _user_hist["engaged_rate"],
           _vpr, _vpe]

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

for ax, (col, title, color, log_x), values in zip(axes.flatten(), _metrics, _series):
    values = values.dropna()
    if len(values) == 0:
        ax.set_title(title); continue

    if log_x:
        bins = list(np.logspace(np.log10(max(values.min(), 1)),
                                np.log10(values.max() + 1), 20))
        median_label = f"Median: {np.median(values):,.0f}"
        ax.set_xscale("log")
    else:
        bins = 20
        median_label = (f"Median: {np.median(values):.1%}"
                        if "rate" in title.lower()
                        else f"Median: {np.median(values):.2f}")

    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.5)
    ax.axvline(np.median(values), color="black", linestyle="--",
               linewidth=1.5, label=median_label)
    if "rate" in title.lower() and not log_x:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Users")

plt.suptitle("Per-User Bounty Metrics", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# === Avg bounty value per problem (normalised by global activity rate) ===
#
# global_avg_rate = mean across users of (bounty_problems / N_DAYS)
# per-user ratio  = (user_value / day) / global_avg_rate
#   → "how many $ of value does this user generate per unit of average activity?"
#   → ratio > 1 means the user generates more value than a typical user would
#     posting at the global average rate.
# Avg(ratio) = mean_user_value_per_day / global_avg_rate
#            = effective avg $ value per bounty problem (averaged over users)

global_avg_rate = user_agg["bounties_per_day"].mean()  # problems/day, averaged over users

user_agg["value_per_avg_problem"] = user_agg["price_per_day"] / global_avg_rate

avg_ratio = user_agg["value_per_avg_problem"].mean()
median_ratio = user_agg["value_per_avg_problem"].median()

print(f"\n=== Bounty activity & value ===")
print(f"  Global avg bounty problems / user / day:  {global_avg_rate:.3f}  "
      f"({global_avg_rate * N_DAYS:.1f} per user over {N_DAYS} days)")
print(f"\n  Avg( user_value/day  /  global_avg_rate ):")
print(f"    mean:   ${avg_ratio:.1f}   (≈ avg $ value per bounty problem, user-weighted)")
print(f"    median: ${median_ratio:.1f}")

# %%
# === Figure 1 — Bounty value per user per day: potential vs. realised ===
# Left:  all bounty-worthy problems (what Twitter *could* deliver)
# Right: only resolved problems    (what Twitter *actually* delivers)

ppd_all = user_agg["price_per_day"].dropna()
ppd_res = user_agg["resolved_price_per_day"].dropna()
ratio_vals = user_agg["value_per_avg_problem"].dropna()

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

for ax, values, color, title, xlabel in [
    (axes[0], ppd_all, "#4A90D9",
     "Potential bounty value\n(all bounty-worthy problems)", "$/day"),
    (axes[1], ppd_res, "#27AE60",
     "Realised bounty value\n(resolved problems only)", "$/day"),
    (axes[2], ratio_vals, "#9B59B6",
     "Value per avg-rate problem\n= ($/day) ÷ global avg problems/day", "$ per avg problem"),
]:
    median_v = values.median()
    ax.hist(values, bins=20, color=color, edgecolor="white", linewidth=0.5)
    ax.axvline(median_v, color="black", linestyle="--", linewidth=1.5,
               label=f"Median: ${median_v:.1f}")
    ax.legend(fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users")

plt.suptitle("Bounty Value per User per Day", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# === Figure 2 — Dollar group distribution + resolution rate ===

group_stats = df.groupby("dollar_group", observed=True).agg(
    count=("tweet_id", "count"),
    resolved=("is_resolved", "sum"),
    engaged=("is_engaged", "sum"),
    total_views=("estimated_views", "sum"),
).reset_index()
group_stats["resolve_rate"] = group_stats["resolved"] / group_stats["count"]
group_stats["engaged_rate"] = group_stats["engaged"] / group_stats["count"]
# views_per_resolve: expected views needed for one resolution (group-level p)
group_stats["views_per_resolve"] = np.where(
    group_stats["resolved"] > 0,
    group_stats["total_views"] / group_stats["resolved"],
    np.nan,
)
group_stats["views_per_engaged"] = np.where(
    group_stats["engaged"] > 0,
    group_stats["total_views"] / group_stats["engaged"],
    np.nan,
)

print("\n=== Resolution rate by dollar group ===")
fmt_g = group_stats.copy()
fmt_g["resolve_rate"] = fmt_g["resolve_rate"].map("{:.1%}".format)
fmt_g["engaged_rate"] = fmt_g["engaged_rate"].map("{:.1%}".format)
print(fmt_g.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: distribution (count per group)
x = range(len(group_stats))
axes[0].bar(x, group_stats["count"], color=DOLLAR_COLORS, edgecolor="white", linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(DOLLAR_LABELS, fontsize=10)
axes[0].set_xlabel("Bounty Price Group")
axes[0].set_ylabel("Number of Bounties")
axes[0].set_title("Distribution of Bounty Price Groups")
for xi, cnt in zip(x, group_stats["count"]):
    axes[0].text(xi, cnt + 5, str(cnt), ha="center", va="bottom", fontsize=9)

# Right: resolution & engagement rate per group (bars + twin y-axis lines)
ax_b = axes[1]
ax_r = ax_b.twinx()

bars = ax_b.bar(x, group_stats["count"], color=DOLLAR_COLORS, alpha=0.35,
                edgecolor="white", linewidth=0.5, label="Count")
ax_r.plot(x, group_stats["engaged_rate"], color="#f39c12", marker="s",
          linewidth=2, label="Engaged rate")
ax_r.plot(x, group_stats["resolve_rate"], color="#27ae60", marker="o",
          linewidth=2, label="Resolve rate")

ax_b.set_xticks(x)
ax_b.set_xticklabels(DOLLAR_LABELS, fontsize=10)
ax_b.set_xlabel("Bounty Price Group")
ax_b.set_ylabel("Number of Bounties", color="#555")
ax_r.set_ylabel("Rate", color="#333")
ax_r.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax_r.set_ylim(0, 1)

handles_b, labels_b = ax_b.get_legend_handles_labels()
handles_r, labels_r = ax_r.get_legend_handles_labels()
axes[1].legend(handles_b + handles_r, labels_b + labels_r, loc="upper right", fontsize=9)
axes[1].set_title("Resolve & Engaged Rate by Price Group")

plt.suptitle("Bounty Price Groups", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# === Figure 3 — Views before 50% resolved, per dollar group (small-multiple histograms) ===
#
# For each group we compute per-user views_per_resolve restricted to bounties
# in that group, matching the methodology of script 31.

fig, axes = plt.subplots(1, len(DOLLAR_LABELS), figsize=(20, 5), sharey=False)

for i, (label, color) in enumerate(zip(DOLLAR_LABELS, DOLLAR_COLORS)):
    ax = axes[i]
    group_df = df[df["dollar_group"] == label]

    # Per-user p within this group
    gu = group_df.groupby("username").agg(
        resolved=("is_resolved", "sum"),
        total_views=("estimated_views", "sum"),
        bounties=("tweet_id", "count"),
    ).reset_index()
    gu["p_resolve"] = np.where(gu["total_views"] > 0, gu["resolved"] / gu["total_views"], np.nan)
    gu["views_per_resolve"] = np.where(gu["p_resolve"] > 0, 1 / gu["p_resolve"], np.nan)

    values = gu["views_per_resolve"].replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) < 3:
        ax.set_title(f"{label}\n(n={len(values)} users, not enough data)")
        ax.set_xlabel("Views")
        ax.set_ylabel("Users")
        continue

    log_min = np.log10(max(values.min(), 1))
    log_max = np.log10(values.max() + 1)
    bins = list(np.logspace(log_min, log_max, min(15, len(values) + 1)))
    median_val = np.median(values)

    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.5)
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1.5,
               label=f"Median: {median_val:,.0f}")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_title(f"{label}\n(n={len(values)} users)")
    ax.set_xlabel("Views")
    ax.set_ylabel("Users")

plt.suptitle("Views before 50% Problem Resolved — by Bounty Price Group", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# === Figure 4 — Avg views before 50% resolved per dollar group (bar chart) ===

print("\n=== Views per resolve / engaged by dollar group ===")
fmt_v = group_stats[["dollar_group", "count", "resolved", "views_per_resolve", "views_per_engaged"]].copy()
fmt_v["views_per_resolve"] = fmt_v["views_per_resolve"].map(lambda v: f"{v:,.0f}" if v == v else "—")
fmt_v["views_per_engaged"] = fmt_v["views_per_engaged"].map(lambda v: f"{v:,.0f}" if v == v else "—")
print(fmt_v.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(group_stats))
width = 0.38

bars_r = ax.bar(x - width / 2, group_stats["views_per_resolve"], width,
                color=DOLLAR_COLORS, edgecolor="white", linewidth=0.5, label="Views / resolve")
bars_e = ax.bar(x + width / 2, group_stats["views_per_engaged"], width,
                color=DOLLAR_COLORS, edgecolor="white", linewidth=0.5, alpha=0.45, label="Views / engaged")

# Value labels on bars
for bar in list(bars_r) + list(bars_e):
    h = bar.get_height()
    if h == h:  # not NaN
        ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                f"{h:,.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(DOLLAR_LABELS, fontsize=11)
ax.set_xlabel("Bounty Price Group")
ax.set_ylabel("Views")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.legend(fontsize=10)
ax.set_title("Avg Views before 50% Resolved / Engaged — by Bounty Price Group", fontsize=13)

plt.tight_layout()
plt.show()

# %%
# === 10 random bounty problems: tweet, bounty description, price ===

sample = df.sample(10, random_state=None).reset_index(drop=True)

print("=" * 90)
print("10 RANDOM BOUNTY PROBLEMS")
print("=" * 90)

outcome_icon = {
    "resolved_by_community": "✅ resolved",
    "serious_attempt": "🔶 serious attempt",
    "unresolved_by_community": "❌ unresolved",
}

for i, row in sample.iterrows():
    icon = outcome_icon.get(row["outcome_label"], f"? {row['outcome_label']}")
    print(f"\n[{i + 1}/10]  ${row['price_usd']:.0f}  ·  @{row['username']}  ·  {icon}")
    print(f"  Tweet:  {row['full_text']}")
    print(f"  Bounty: {row['problem_description']}")

print("\n" + "=" * 90)

# %%
