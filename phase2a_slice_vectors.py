"""
Phase 2A — Temporal Expertise Representation for Regime B (Trajectory Authors)

For each Regime B author:
  1. Partition career into 5-year slices anchored to first_year
  2. Build slice-level topic vectors (normalized frequency over 2nd order topics)
  3. Compute temporal signals:
       - topic_continuity  : cosine similarity between consecutive slices
       - topic_drift       : 1 - continuity
       - productivity_slope: linear trend of publication count over slices
       - dominant_topic    : highest-weight topic per slice

Outputs:
  - slice_vectors.parquet        ← (author_id, slice_id, topic, weight)
  - author_temporal_features.csv ← per-author aggregated temporal signals
  - slice_metadata.csv           ← per-slice stats (year_start, year_end, pub_count)
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import linregress
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
TENSOR_2ND        = "author_topic_year_tensor_2nd_order.parquet"
REGIME_B_CSV      = "regime_b_authors.csv"

OUTPUT_SLICES     = "slice_vectors.parquet"
OUTPUT_FEATURES   = "author_temporal_features.csv"
OUTPUT_SLICE_META = "slice_metadata.csv"

SLICE_SIZE        = 5   # years per slice

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("Loading data...")
tensor    = pd.read_parquet(TENSOR_2ND)
regime_b  = pd.read_csv(REGIME_B_CSV)

# Normalize author_id format (tensor uses full URI, regime_b uses URI too)
regime_b_ids = set(regime_b["author_id"].tolist())

# Filter tensor to Regime B authors only
tensor_b = tensor[tensor["author_id"].isin(regime_b_ids)].copy()

print(f"  -> Regime B authors        : {len(regime_b_ids):,}")
print(f"  -> Tensor entries (Regime B): {len(tensor_b):,}")
print(f"  -> Unique topics (2nd order): {tensor_b['topic_name'].nunique()}")

# Get all unique topics for consistent vector dimensions
all_topics = sorted(tensor_b["topic_name"].unique())
topic_index = {t: i for i, t in enumerate(all_topics)}
n_topics = len(all_topics)
print(f"  -> Topic vocabulary size   : {n_topics}")

# ── Step 2: Build per-author slice vectors ────────────────────────────────────
print("\nBuilding 5-year slice vectors for Regime B authors...")

slice_rows   = []   # for slice_vectors.parquet
meta_rows    = []   # for slice_metadata.csv
feature_rows = []   # for author_temporal_features.csv

# Merge first_year from regime_b into tensor
author_first = regime_b.set_index("author_id")["first_year"].to_dict()
author_pubs  = regime_b.set_index("author_id")["total_pubs"].to_dict()

skipped = 0

for author_id, group in tqdm(tensor_b.groupby("author_id"), desc="Authors"):
    first_year = author_first.get(author_id)
    if first_year is None:
        skipped += 1
        continue

    # ── Assign each paper-year to a slice ──────────────────────────────────
    # Slice 0: first_year to first_year+4
    # Slice 1: first_year+5 to first_year+9, etc.
    group = group.copy()
    group["slice_id"] = ((group["year"] - first_year) // SLICE_SIZE).astype(int)
    group = group[group["slice_id"] >= 0]  # drop pre-career entries

    slices = sorted(group["slice_id"].unique())
    if len(slices) < 2:
        # Need at least 2 slices for trajectory modeling
        skipped += 1
        continue

    # ── Build topic vector per slice ───────────────────────────────────────
    slice_vectors = {}   # slice_id -> np.array

    for slice_id, sgroup in group.groupby("slice_id"):
        year_start = first_year + slice_id * SLICE_SIZE
        year_end   = year_start + SLICE_SIZE - 1
        pub_count  = sgroup["frequency"].sum()

        # Topic vector: normalized frequency
        vec = np.zeros(n_topics)
        for _, row in sgroup.iterrows():
            idx = topic_index.get(row["topic_name"])
            if idx is not None:
                vec[idx] += row["frequency"]

        # L1 normalize
        total = vec.sum()
        if total > 0:
            vec = vec / total

        slice_vectors[slice_id] = vec

        # Dominant topic
        dominant_idx   = np.argmax(vec)
        dominant_topic = all_topics[dominant_idx]

        meta_rows.append({
            "author_id"     : author_id,
            "slice_id"      : slice_id,
            "year_start"    : year_start,
            "year_end"      : year_end,
            "pub_count"     : int(pub_count),
            "dominant_topic": dominant_topic,
            "n_topics_active": int((vec > 0).sum()),
        })

        # Save slice vector rows
        for topic_name, idx in topic_index.items():
            if vec[idx] > 0:
                slice_rows.append({
                    "author_id" : author_id,
                    "slice_id"  : slice_id,
                    "year_start": year_start,
                    "year_end"  : year_end,
                    "topic_name": topic_name,
                    "weight"    : round(float(vec[idx]), 6),
                })

    # ── Compute temporal signals across slices ─────────────────────────────
    sorted_slice_ids = sorted(slice_vectors.keys())
    n_slices = len(sorted_slice_ids)

    # Topic continuity & drift between consecutive slices
    continuities = []
    drifts       = []
    for i in range(len(sorted_slice_ids) - 1):
        v1 = slice_vectors[sorted_slice_ids[i]]
        v2 = slice_vectors[sorted_slice_ids[i + 1]]
        # Handle zero vectors
        if v1.sum() == 0 or v2.sum() == 0:
            continue
        cos_sim = 1 - cosine(v1, v2)
        continuities.append(cos_sim)
        drifts.append(1 - cos_sim)

    avg_continuity = float(np.mean(continuities)) if continuities else 0.0
    avg_drift      = float(np.mean(drifts))       if drifts      else 0.0
    max_drift      = float(np.max(drifts))        if drifts      else 0.0

    # Productivity slope: linear regression of pub_count over slice index
    slice_pub_counts = []
    for sid in sorted_slice_ids:
        m = meta_rows[-1]  # last appended — may not be this author, use group
        sc = group[group["slice_id"] == sid]["frequency"].sum()
        slice_pub_counts.append(int(sc))

    if len(slice_pub_counts) >= 2:
        x = np.arange(len(slice_pub_counts))
        slope, intercept, r, p, se = linregress(x, slice_pub_counts)
        productivity_slope = float(slope)
        productivity_r2    = float(r ** 2)
    else:
        productivity_slope = 0.0
        productivity_r2    = 0.0

    # Topic breadth: avg number of active topics per slice
    topic_breadth = float(np.mean([
        (slice_vectors[sid] > 0).sum() for sid in sorted_slice_ids
    ]))

    feature_rows.append({
        "author_id"         : author_id,
        "n_slices"          : n_slices,
        "first_slice_year"  : first_year + sorted_slice_ids[0]  * SLICE_SIZE,
        "last_slice_year"   : first_year + sorted_slice_ids[-1] * SLICE_SIZE,
        "avg_continuity"    : round(avg_continuity, 4),
        "avg_drift"         : round(avg_drift, 4),
        "max_drift"         : round(max_drift, 4),
        "productivity_slope": round(productivity_slope, 4),
        "productivity_r2"   : round(productivity_r2, 4),
        "topic_breadth"     : round(topic_breadth, 4),
    })

print(f"  -> Authors processed : {len(feature_rows):,}")
print(f"  -> Authors skipped   : {skipped:,}  (< 2 slices)")
print(f"  -> Total slice rows  : {len(slice_rows):,}")
print(f"  -> Total meta rows   : {len(meta_rows):,}")

# ── Step 3: Build DataFrames ──────────────────────────────────────────────────
print("\nBuilding DataFrames...")
df_slices   = pd.DataFrame(slice_rows)
df_meta     = pd.DataFrame(meta_rows)
df_features = pd.DataFrame(feature_rows)

# Merge regime_b metadata into features
df_features = df_features.merge(
    regime_b[["author_id", "name", "total_pubs", "career_span",
               "active_years", "recency_ratio", "first_year", "last_year"]],
    on="author_id", how="left"
)

# ── Step 4: Print summary stats ───────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PHASE 2A SUMMARY")
print(f"{'='*60}")
print(f"  Authors with >= 2 slices     : {len(df_features):,}")
print(f"  Total slices created         : {len(df_meta):,}")
print(f"  Avg slices per author        : {df_meta.groupby('author_id').size().mean():.2f}")
print(f"  Max slices per author        : {df_meta.groupby('author_id').size().max()}")

print(f"\n  Temporal signal stats:")
print(f"    Avg topic continuity   : {df_features['avg_continuity'].mean():.3f}  (1=stable, 0=complete drift)")
print(f"    Avg topic drift        : {df_features['avg_drift'].mean():.3f}")
print(f"    Avg max drift          : {df_features['max_drift'].mean():.3f}")
print(f"    Avg productivity slope : {df_features['productivity_slope'].mean():.3f}")
print(f"    % with positive slope  : {(df_features['productivity_slope'] > 0).mean()*100:.1f}%  (growing)")
print(f"    % with negative slope  : {(df_features['productivity_slope'] < 0).mean()*100:.1f}%  (declining)")
print(f"    Avg topic breadth      : {df_features['topic_breadth'].mean():.2f} topics/slice")

# Continuity distribution
print(f"\n  Continuity distribution:")
print(f"    High (>0.7) stable     : {(df_features['avg_continuity'] > 0.7).mean()*100:.1f}%")
print(f"    Medium (0.3-0.7)       : {((df_features['avg_continuity'] >= 0.3) & (df_features['avg_continuity'] <= 0.7)).mean()*100:.1f}%")
print(f"    Low (<0.3) high drift  : {(df_features['avg_continuity'] < 0.3).mean()*100:.1f}%")

# ── Step 5: Save outputs ──────────────────────────────────────────────────────
print(f"\nSaving outputs...")
df_slices.to_parquet(OUTPUT_SLICES, index=False)
df_meta.to_csv(OUTPUT_SLICE_META, index=False)
df_features.to_csv(OUTPUT_FEATURES, index=False)
print(f"  -> {OUTPUT_SLICES}     ({len(df_slices):,} rows)")
print(f"  -> {OUTPUT_SLICE_META} ({len(df_meta):,} rows)")
print(f"  -> {OUTPUT_FEATURES}   ({len(df_features):,} rows)")

# ── Step 6: Sanity check — sample author trajectory ──────────────────────────
print(f"\nSanity check — sample author with highest continuity:")
top_cont = df_features.nlargest(1, "avg_continuity").iloc[0]
print(f"  Author : {top_cont['name']}  (span={top_cont['career_span']}, pubs={top_cont['total_pubs']})")
print(f"  Continuity={top_cont['avg_continuity']:.3f}, Drift={top_cont['avg_drift']:.3f}, Slope={top_cont['productivity_slope']:.3f}")
slices_sample = df_meta[df_meta["author_id"] == top_cont["author_id"]].sort_values("slice_id")
print(f"\n  {'Slice':>6} {'Years':>12} {'Pubs':>5} {'Dominant Topic':<45}")
print("  " + "-" * 72)
for _, r in slices_sample.iterrows():
    print(f"  {int(r['slice_id']):>6} {str(int(r['year_start']))+'-'+str(int(r['year_end'])):>12} "
          f"{int(r['pub_count']):>5} {r['dominant_topic']:<45}")

print(f"\nSanity check — sample author with highest drift:")
top_drift = df_features.nlargest(1, "max_drift").iloc[0]
print(f"  Author : {top_drift['name']}  (span={top_drift['career_span']}, pubs={top_drift['total_pubs']})")
print(f"  Continuity={top_drift['avg_continuity']:.3f}, MaxDrift={top_drift['max_drift']:.3f}, Slope={top_drift['productivity_slope']:.3f}")
slices_drift = df_meta[df_meta["author_id"] == top_drift["author_id"]].sort_values("slice_id")
print(f"\n  {'Slice':>6} {'Years':>12} {'Pubs':>5} {'Dominant Topic':<45}")
print("  " + "-" * 72)
for _, r in slices_drift.iterrows():
    print(f"  {int(r['slice_id']):>6} {str(int(r['year_start']))+'-'+str(int(r['year_end'])):>12} "
          f"{int(r['pub_count']):>5} {r['dominant_topic']:<45}")

print(f"\n✅ Phase 2A complete.")