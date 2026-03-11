"""
Phase 3 — Role-State Inference

Computes soft role scores (emerging, collaborating, supervising) for every author
by combining authorship position percentages with career and temporal signals.

Design principles:
  - No hard thresholds → continuous soft scores
  - All features z-score normalized before combining
  - Equal weighting within each role signal group
  - Softmax over 3 raw scores → probabilities summing to 1
  - Position signal downweighted for partial-coverage authors
  - Fully documented for paper methodology section

Inputs:
  - author_regime_map.csv              ← career metrics + regime + coverage flag
  - author_temporal_features.csv       ← Regime B: continuity, drift, slope
  - regime_a_author_features.csv       ← Regime A: recency, entropy, weighted_year
  - Upgraded6_Knowledge_graph...ttl    ← authorship position triples

Outputs:
  - author_role_scores.csv             ← soft role probabilities + dominant role
  - author_position_stats.csv          ← per-author authorship position percentages
"""

import pandas as pd
import numpy as np
from rdflib import Graph, Namespace, RDF
from scipy.special import softmax
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_TTL       = "Upgraded6_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
REGIME_MAP_CSV  = "author_regime_map.csv"
FEATURES_B_CSV  = "author_temporal_features.csv"
FEATURES_A_CSV  = "regime_a_author_features.csv"

OUTPUT_ROLES    = "author_role_scores.csv"
OUTPUT_POSITIONS= "author_position_stats.csv"

# Position signal reliability threshold
# Authors with < this many KG papers get downweighted position signal
MIN_RELIABLE_PAPERS = 5

# Softmax temperature: higher = softer (more uniform), lower = sharper
SOFTMAX_TEMP = 1.0

# ── Namespaces ─────────────────────────────────────────────────────────────────
EX = Namespace("http://expert-search.org/schema#")

# ── Step 1: Load KG and extract authorship positions ──────────────────────────
print(f"Loading KG from {INPUT_TTL}...")
g = Graph()
g.parse(INPUT_TTL, format="turtle")
g.bind("ex", EX)
print(f"  -> {len(g):,} triples loaded")

print("\nExtracting authorship position statistics...")

# Per-author position counters
author_pos_counts = {}  # author_id -> {first, middle, last, sole, total}

for authorship in tqdm(g.subjects(RDF.type, EX.Authorship), desc="Authorship nodes"):
    author_uri = g.value(authorship, EX.author)
    position   = g.value(authorship, EX.authorPosition)
    count      = g.value(authorship, EX.authorCount)

    if author_uri is None or position is None or count is None:
        continue

    try:
        pos   = int(str(position))
        total = int(str(count))
    except (ValueError, TypeError):
        continue

    aid = str(author_uri)
    if aid not in author_pos_counts:
        author_pos_counts[aid] = {"first":0,"middle":0,"last":0,"sole":0,"total":0}

    author_pos_counts[aid]["total"] += 1

    if total == 1:
        author_pos_counts[aid]["sole"]   += 1
        author_pos_counts[aid]["first"]  += 1   # sole counts as both
        author_pos_counts[aid]["last"]   += 1
    elif pos == 1:
        author_pos_counts[aid]["first"]  += 1
    elif pos == total:
        author_pos_counts[aid]["last"]   += 1
    else:
        author_pos_counts[aid]["middle"] += 1

print(f"  -> Authors with position data: {len(author_pos_counts):,}")

# ── Step 2: Build position stats DataFrame ────────────────────────────────────
pos_rows = []
for aid, counts in author_pos_counts.items():
    t = max(counts["total"], 1)
    pos_rows.append({
        "author_id"   : aid,
        "kg_papers"   : counts["total"],
        "n_first"     : counts["first"],
        "n_middle"    : counts["middle"],
        "n_last"      : counts["last"],
        "n_sole"      : counts["sole"],
        "pct_first"   : round(counts["first"]  / t, 4),
        "pct_middle"  : round(counts["middle"] / t, 4),
        "pct_last"    : round(counts["last"]   / t, 4),
        "pct_sole"    : round(counts["sole"]   / t, 4),
        # reliability: 1.0 if kg_papers >= threshold, scaled below
        "pos_reliable": min(1.0, counts["total"] / MIN_RELIABLE_PAPERS),
    })

df_pos = pd.DataFrame(pos_rows)
df_pos.to_csv(OUTPUT_POSITIONS, index=False)
print(f"  -> Position stats saved: {OUTPUT_POSITIONS} ({len(df_pos):,} rows)")

# ── Step 3: Load all author features ─────────────────────────────────────────
print("\nLoading author features...")
df_map = pd.read_csv(REGIME_MAP_CSV)
df_b   = pd.read_csv(FEATURES_B_CSV)
df_a   = pd.read_csv(FEATURES_A_CSV)

# Merge regime map with position stats
df = df_map.merge(df_pos, on="author_id", how="left")

# Fill missing position stats (authors with no KG papers)
for col in ["pct_first","pct_middle","pct_last","pct_sole","pos_reliable","kg_papers"]:
    df[col] = df[col].fillna(0.0)

# Merge Regime B temporal features
df_b_slim = df_b[["author_id","productivity_slope","avg_continuity","avg_drift"]].copy()
df = df.merge(df_b_slim, on="author_id", how="left")

# Merge Regime A recency features (entropy, weighted_year)
df_a_slim = df_a[["author_id","expertise_entropy","weighted_year","n_active_topics"]].copy()
df = df.merge(df_a_slim, on="author_id", how="left")

# For Regime A authors: productivity_slope = 0 (neutral)
df["productivity_slope"] = df["productivity_slope"].fillna(0.0)
df["avg_continuity"]     = df["avg_continuity"].fillna(0.5)   # neutral
df["avg_drift"]          = df["avg_drift"].fillna(0.5)

print(f"  -> Total authors in master table: {len(df):,}")

# ── Step 4: Feature engineering ───────────────────────────────────────────────
print("\nEngineering features...")

# Log-normalize skewed features
df["log_pubs"]  = np.log1p(df["total_pubs"])
df["log_span"]  = np.log1p(df["career_span"])
df["log_active"]= np.log1p(df["active_years"])

# Z-score normalize all features across full author population
def zscore(series):
    mu  = series.mean()
    std = series.std()
    if std == 0:
        return series * 0
    return (series - mu) / std

features = [
    "pct_first", "pct_middle", "pct_last",
    "recency_ratio", "log_span", "log_pubs",
    "log_active", "productivity_slope"
]

for f in features:
    df[f"z_{f}"] = zscore(df[f])

# ── Step 5: Role score computation ────────────────────────────────────────────
print("Computing role scores...")

"""
Signal rationale (documented for paper):

EMERGING:
  + pct_first       : primary author → main contributor → emerging expert
  + recency_ratio   : recent activity → currently active in the field
  + productivity_slope (positive) : growing output → career ascending
  - log_span        : short career → consistent with emerging stage
  - log_pubs        : lower total output → not yet established
  - pct_last        : rarely supervising → not yet senior

COLLABORATING:
  + pct_middle      : middle authorship → collaborative contributor
  + log_active      : sustained engagement → multiple active years
  (no strong career stage signal → collaborating is regime-neutral)

SUPERVISING:
  + pct_last        : last author → senior/PI role → supervising
  + log_span        : long career → established researcher
  + log_pubs        : high total output → established body of work
  + log_active      : many active years → sustained senior presence
  - recency_ratio   : lower recency → activity earlier in career
  - productivity_slope (negative) : declining output → later career
"""

# ── Emerging score ────────────────────────────────────────────────────────────
df["score_emerging"] = (
    + 1.0 * df["z_pct_first"]
    + 1.0 * df["z_recency_ratio"]
    + 0.5 * df["z_productivity_slope"]   # half weight — missing for Regime A
    - 0.5 * df["z_log_span"]
    - 0.5 * df["z_log_pubs"]
    - 0.5 * df["z_pct_last"]
)

# ── Collaborating score ───────────────────────────────────────────────────────
df["score_collaborating"] = (
    + 1.0 * df["z_pct_middle"]
    + 0.5 * df["z_log_active"]
)

# ── Supervising score ─────────────────────────────────────────────────────────
df["score_supervising"] = (
    + 1.0 * df["z_pct_last"]
    + 1.0 * df["z_log_span"]
    + 0.5 * df["z_log_pubs"]
    + 0.5 * df["z_log_active"]
    - 0.5 * df["z_recency_ratio"]
    - 0.5 * df["z_productivity_slope"]   # declining slope → senior
)

# ── Downweight position signal for partial-coverage authors ──────────────────
# For low pos_reliable authors, regress position scores toward 0 (neutral)
# This avoids falsely confident role labels from 1-2 paper samples
pos_weight = df["pos_reliable"].values

df["score_emerging"]     = df["score_emerging"]     * pos_weight + \
    (df["z_recency_ratio"] - 0.5*df["z_log_span"] - 0.5*df["z_log_pubs"]) * (1 - pos_weight)

df["score_collaborating"]= df["score_collaborating"]* pos_weight + \
    (0.5*df["z_log_active"]) * (1 - pos_weight)

df["score_supervising"]  = df["score_supervising"]  * pos_weight + \
    (df["z_log_span"] + 0.5*df["z_log_pubs"] + 0.5*df["z_log_active"] \
     - 0.5*df["z_recency_ratio"]) * (1 - pos_weight)

# ── Softmax → probabilities ───────────────────────────────────────────────────
raw_scores = df[["score_emerging","score_collaborating","score_supervising"]].values

probs = np.array([
    softmax(row / SOFTMAX_TEMP) for row in raw_scores
])

df["prob_emerging"]     = np.round(probs[:, 0], 4)
df["prob_collaborating"]= np.round(probs[:, 1], 4)
df["prob_supervising"]  = np.round(probs[:, 2], 4)

# Dominant role
role_labels = ["emerging", "collaborating", "supervising"]
df["dominant_role"] = [role_labels[np.argmax(p)] for p in probs]

# Confidence = max probability
df["role_confidence"] = np.round(probs.max(axis=1), 4)

# ── Step 6: Summary statistics ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PHASE 3 SUMMARY")
print(f"{'='*60}")
total = len(df)
print(f"  Total authors scored       : {total:,}")

for role in role_labels:
    n    = (df["dominant_role"] == role).sum()
    pct  = n / total * 100
    conf = df.loc[df["dominant_role"] == role, "role_confidence"].mean()
    print(f"  {role.capitalize():<15} : {n:>6,}  ({pct:5.1f}%)  avg confidence = {conf:.3f}")

print(f"\n  Regime breakdown by dominant role:")
for regime in ["A", "B"]:
    sub = df[df["regime"] == regime]
    print(f"\n  Regime {regime} ({len(sub):,} authors):")
    for role in role_labels:
        n   = (sub["dominant_role"] == role).sum()
        pct = n / len(sub) * 100
        print(f"    {role.capitalize():<15} : {n:>6,}  ({pct:5.1f}%)")

print(f"\n  Role score distributions (mean ± std of probabilities):")
for role in role_labels:
    col = f"prob_{role}"
    print(f"    {role.capitalize():<15} : {df[col].mean():.3f} ± {df[col].std():.3f}  "
          f"(min={df[col].min():.3f}, max={df[col].max():.3f})")

# ── Step 7: Save ─────────────────────────────────────────────────────────────
output_cols = [
    "author_id", "name", "regime", "year_coverage",
    "total_pubs", "career_span", "active_years", "recency_ratio",
    "pct_first", "pct_middle", "pct_last", "pct_sole",
    "kg_papers", "pos_reliable", "productivity_slope",
    "prob_emerging", "prob_collaborating", "prob_supervising",
    "dominant_role", "role_confidence"
]
df[output_cols].to_csv(OUTPUT_ROLES, index=False)
print(f"\n  -> {OUTPUT_ROLES} saved ({len(df):,} rows)")

# ── Step 8: Sanity checks ─────────────────────────────────────────────────────
print(f"\nSanity check — top 10 emerging authors:")
top_e = df.nlargest(10, "prob_emerging")
print(f"\n  {'Name':<28} {'Regime'} {'PctFirst':>9} {'Recency':>8} {'Span':>5} {'P(E)':>6} {'P(C)':>6} {'P(S)':>6}")
print("  " + "-" * 82)
for _, r in top_e.iterrows():
    print(f"  {r['name']:<28} {r['regime']:<7} {r['pct_first']:>9.3f} "
          f"{r['recency_ratio']:>8.3f} {r['career_span']:>5} "
          f"{r['prob_emerging']:>6.3f} {r['prob_collaborating']:>6.3f} {r['prob_supervising']:>6.3f}")

print(f"\nSanity check — top 10 supervising authors:")
top_s = df.nlargest(10, "prob_supervising")
print(f"\n  {'Name':<28} {'Regime'} {'PctLast':>8} {'Span':>5} {'Pubs':>6} {'P(E)':>6} {'P(C)':>6} {'P(S)':>6}")
print("  " + "-" * 82)
for _, r in top_s.iterrows():
    print(f"  {r['name']:<28} {r['regime']:<7} {r['pct_last']:>8.3f} "
          f"{r['career_span']:>5} {r['total_pubs']:>6} "
          f"{r['prob_emerging']:>6.3f} {r['prob_collaborating']:>6.3f} {r['prob_supervising']:>6.3f}")

print(f"\nSanity check — top 10 collaborating authors:")
top_c = df.nlargest(10, "prob_collaborating")
print(f"\n  {'Name':<28} {'Regime'} {'PctMid':>7} {'ActYrs':>7} {'P(E)':>6} {'P(C)':>6} {'P(S)':>6}")
print("  " + "-" * 72)
for _, r in top_c.iterrows():
    print(f"  {r['name']:<28} {r['regime']:<7} {r['pct_middle']:>7.3f} "
          f"{r['active_years']:>7} "
          f"{r['prob_emerging']:>6.3f} {r['prob_collaborating']:>6.3f} {r['prob_supervising']:>6.3f}")

print(f"\n✅ Phase 3 complete.")
print(f"   Role scores → {OUTPUT_ROLES}")
print(f"   Position stats → {OUTPUT_POSITIONS}")