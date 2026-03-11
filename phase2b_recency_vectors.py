"""
Phase 2B — Recency-Weighted Expertise Representation for Regime A (Short-Span Authors)

For each Regime A author:
  1. Compute recency weight per paper: w = exp(-lambda * (current_year - year))
  2. Aggregate weighted topic frequencies into a single expertise vector
  3. Normalize to unit L1 vector

Also computes:
  - recency_ratio       : pubs in last 5 years / total pubs (already known)
  - expertise_entropy   : Shannon entropy of topic distribution (specialist vs generalist)
  - dominant_topic      : highest-weight topic
  - weighted_year       : recency-weighted mean publication year (activity center of mass)

Two lambda values tested:
  - lambda = 0.1  : slow decay (10-year half-life ~7 years)
  - lambda = 0.3  : fast decay (favors very recent publications)
Default: lambda = 0.1 (recommended — preserves signal for emerging authors)

Outputs:
  - regime_a_expertise_vectors.parquet  ← (author_id, topic_name, weight_slow, weight_fast)
  - regime_a_author_features.csv        ← per-author aggregated features
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
TENSOR_2ND      = "author_topic_year_tensor_2nd_order.parquet"
REGIME_A_CSV    = "regime_a_authors.csv"

OUTPUT_VECTORS  = "regime_a_expertise_vectors.parquet"
OUTPUT_FEATURES = "regime_a_author_features.csv"

CURRENT_YEAR    = 2019   # dataset end year
RECENCY_WINDOW  = 5
LAMBDA_SLOW     = 0.1    # default: slow decay
LAMBDA_FAST     = 0.3    # comparison: fast decay

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("Loading data...")
tensor   = pd.read_parquet(TENSOR_2ND)
regime_a = pd.read_csv(REGIME_A_CSV)

regime_a_ids = set(regime_a["author_id"].tolist())
tensor_a     = tensor[tensor["author_id"].isin(regime_a_ids)].copy()

print(f"  -> Regime A authors          : {len(regime_a_ids):,}")
print(f"  -> Tensor entries (Regime A) : {len(tensor_a):,}")
print(f"  -> Unique topics             : {tensor_a['topic_name'].nunique()}")

all_topics   = sorted(tensor_a["topic_name"].unique())
topic_index  = {t: i for i, t in enumerate(all_topics)}
n_topics     = len(all_topics)

# Build author metadata lookup
author_meta = regime_a.set_index("author_id")[
    ["name", "total_pubs", "career_span", "active_years",
     "recency_ratio", "first_year", "last_year", "year_coverage"]
].to_dict("index")

# ── Step 2: Build recency-weighted vectors ────────────────────────────────────
print(f"\nBuilding recency-weighted expertise vectors...")
print(f"  Lambda slow = {LAMBDA_SLOW}  (half-life ≈ {np.log(2)/LAMBDA_SLOW:.1f} years)")
print(f"  Lambda fast = {LAMBDA_FAST}  (half-life ≈ {np.log(2)/LAMBDA_FAST:.1f} years)")

vector_rows  = []
feature_rows = []
skipped      = 0

for author_id, group in tqdm(tensor_a.groupby("author_id"), desc="Authors"):
    if not group["frequency"].sum():
        skipped += 1
        continue

    meta = author_meta.get(author_id, {})

    # ── Compute recency weights per year ──────────────────────────────────
    years       = group["year"].values
    frequencies = group["frequency"].values
    topics      = group["topic_name"].values

    delta_t      = CURRENT_YEAR - years          # years since publication
    w_slow       = np.exp(-LAMBDA_SLOW * delta_t)
    w_fast       = np.exp(-LAMBDA_FAST * delta_t)

    # ── Build weighted topic vectors ──────────────────────────────────────
    vec_slow = np.zeros(n_topics)
    vec_fast = np.zeros(n_topics)
    vec_raw  = np.zeros(n_topics)   # unweighted baseline

    for i, topic in enumerate(topics):
        idx = topic_index.get(topic)
        if idx is None:
            continue
        freq = frequencies[i]
        vec_slow[idx] += freq * w_slow[i]
        vec_fast[idx] += freq * w_fast[i]
        vec_raw[idx]  += freq

    # L1 normalize
    def l1_norm(v):
        s = v.sum()
        return v / s if s > 0 else v

    vec_slow_n = l1_norm(vec_slow)
    vec_fast_n = l1_norm(vec_fast)
    vec_raw_n  = l1_norm(vec_raw)

    # ── Save vector rows (sparse — only non-zero) ─────────────────────────
    for topic, idx in topic_index.items():
        if vec_slow_n[idx] > 0 or vec_fast_n[idx] > 0:
            vector_rows.append({
                "author_id"   : author_id,
                "topic_name"  : topic,
                "weight_slow" : round(float(vec_slow_n[idx]), 6),
                "weight_fast" : round(float(vec_fast_n[idx]), 6),
                "weight_raw"  : round(float(vec_raw_n[idx]),  6),
            })

    # ── Compute per-author features ───────────────────────────────────────
    # Dominant topic (slow decay)
    dominant_idx   = np.argmax(vec_slow_n)
    dominant_topic = all_topics[dominant_idx]

    # Expertise entropy (slow decay) — measures specialist vs generalist
    # Shannon entropy normalized by log(n_topics)
    nz = vec_slow_n[vec_slow_n > 0]
    entropy = float(-np.sum(nz * np.log(nz)) / np.log(n_topics)) if len(nz) > 1 else 0.0

    # Weighted mean publication year (activity center of mass)
    total_freq    = frequencies.sum()
    weighted_year = float(np.sum(years * frequencies) / total_freq) if total_freq > 0 else float(CURRENT_YEAR)

    # Recent pubs count
    recent_pubs   = int(frequencies[delta_t <= RECENCY_WINDOW].sum())
    total_tensor  = int(frequencies.sum())

    # Shift between raw and recency-weighted dominant topic
    raw_dominant_idx   = np.argmax(vec_raw_n)
    topic_shift        = int(dominant_idx != raw_dominant_idx)  # 1 if recency changes dominant topic

    feature_rows.append({
        "author_id"       : author_id,
        "name"            : meta.get("name", ""),
        "total_pubs"      : meta.get("total_pubs", 0),
        "career_span"     : meta.get("career_span", 0),
        "active_years"    : meta.get("active_years", 0),
        "recency_ratio"   : meta.get("recency_ratio", 0.0),
        "first_year"      : meta.get("first_year", None),
        "last_year"       : meta.get("last_year", None),
        "year_coverage"   : meta.get("year_coverage", "unknown"),
        "dominant_topic"  : dominant_topic,
        "expertise_entropy": round(entropy, 4),
        "weighted_year"   : round(weighted_year, 2),
        "recent_pubs"     : recent_pubs,
        "tensor_total"    : total_tensor,
        "topic_shift"     : topic_shift,
        "n_active_topics" : int((vec_slow_n > 0).sum()),
    })

print(f"  -> Authors processed : {len(feature_rows):,}")
print(f"  -> Authors skipped   : {skipped:,}  (no tensor entries)")
print(f"  -> Vector rows saved : {len(vector_rows):,}")

# ── Step 3: Build DataFrames ──────────────────────────────────────────────────
print("\nBuilding DataFrames...")
df_vectors  = pd.DataFrame(vector_rows)
df_features = pd.DataFrame(feature_rows)

# ── Step 4: Summary stats ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PHASE 2B SUMMARY")
print(f"{'='*60}")
print(f"  Regime A authors processed   : {len(df_features):,}")
print(f"  Unique expertise vectors     : {df_vectors['author_id'].nunique():,}")

print(f"\n  Expertise vector stats (slow decay, lambda={LAMBDA_SLOW}):")
print(f"    Avg active topics/author   : {df_features['n_active_topics'].mean():.2f}  (out of {n_topics})")
print(f"    Avg expertise entropy      : {df_features['expertise_entropy'].mean():.3f}  (0=specialist, 1=generalist)")
print(f"    Median weighted year       : {df_features['weighted_year'].median():.1f}")
print(f"    % recency ratio > 0.5      : {(df_features['recency_ratio'] > 0.5).mean()*100:.1f}%  ← emerging signal")
print(f"    % recency ratio = 0        : {(df_features['recency_ratio'] == 0).mean()*100:.1f}%  ← fully historical")
print(f"    % topic shift by recency   : {df_features['topic_shift'].mean()*100:.1f}%  ← recency changes dominant topic")

print(f"\n  Dominant topic distribution (top 5):")
top_topics = df_features["dominant_topic"].value_counts().head(5)
for topic, count in top_topics.items():
    print(f"    {topic:<50} : {count:,}  ({count/len(df_features)*100:.1f}%)")

# Entropy distribution
print(f"\n  Entropy distribution (specialization):")
print(f"    Low  (<0.3) specialist     : {(df_features['expertise_entropy'] < 0.3).mean()*100:.1f}%")
print(f"    Mid  (0.3-0.6)             : {((df_features['expertise_entropy'] >= 0.3) & (df_features['expertise_entropy'] <= 0.6)).mean()*100:.1f}%")
print(f"    High (>0.6) generalist     : {(df_features['expertise_entropy'] > 0.6).mean()*100:.1f}%")

# ── Step 5: Save ─────────────────────────────────────────────────────────────
print(f"\nSaving outputs...")
df_vectors.to_parquet(OUTPUT_VECTORS, index=False)
df_features.to_csv(OUTPUT_FEATURES, index=False)
print(f"  -> {OUTPUT_VECTORS}  ({len(df_vectors):,} rows)")
print(f"  -> {OUTPUT_FEATURES} ({len(df_features):,} rows)")

# ── Step 6: Sanity checks ─────────────────────────────────────────────────────
print(f"\nSanity check — top 5 emerging authors (recency_ratio > 0.7):")
emerging = df_features[df_features["recency_ratio"] > 0.7].sort_values(
    "recent_pubs", ascending=False).head(5)
print(f"\n  {'Name':<28} {'Pubs':>5} {'Span':>5} {'Recency':>8} {'Entropy':>8} {'Dominant Topic'}")
print("  " + "-" * 90)
for _, r in emerging.iterrows():
    print(f"  {r['name']:<28} {r['total_pubs']:>5} {r['career_span']:>5} "
          f"{r['recency_ratio']:>8.3f} {r['expertise_entropy']:>8.3f}  {r['dominant_topic']}")

print(f"\nSanity check — top 5 historical authors (recency_ratio = 0, high pubs):")
historical = df_features[df_features["recency_ratio"] == 0].sort_values(
    "total_pubs", ascending=False).head(5)
print(f"\n  {'Name':<28} {'Pubs':>5} {'Span':>5} {'WtdYear':>8} {'Entropy':>8} {'Dominant Topic'}")
print("  " + "-" * 90)
for _, r in historical.iterrows():
    print(f"  {r['name']:<28} {r['total_pubs']:>5} {r['career_span']:>5} "
          f"{r['weighted_year']:>8.1f} {r['expertise_entropy']:>8.3f}  {r['dominant_topic']}")

print(f"\nSanity check — topic shift by recency (authors where recency changes dominant topic):")
shifted = df_features[df_features["topic_shift"] == 1].sort_values(
    "recency_ratio", ascending=False).head(5)
print(f"\n  {'Name':<28} {'Recency':>8} {'Span':>5} {'Dominant Topic (recency-weighted)'}")
print("  " + "-" * 80)
for _, r in shifted.iterrows():
    print(f"  {r['name']:<28} {r['recency_ratio']:>8.3f} {r['career_span']:>5}  {r['dominant_topic']}")

print(f"\n✅ Phase 2B complete.")
print(f"   Regime A vectors → {OUTPUT_VECTORS}")
print(f"   Regime A features → {OUTPUT_FEATURES}")