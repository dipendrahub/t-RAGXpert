"""
Phase 1.1 — Author Regime Partition

Splits authors into:
  Regime B (trajectory-eligible):
      career_span  >= 6 years
      total_pubs   >= 8
      active_years >= 3

  Regime A (short-span / episodic):
      all others

Inputs:
  - Upgraded KG (for author metrics)
  - author_topic_year_tensor_2nd.parquet (for span/active_years from tensor)

Outputs:
  - regime_b_authors.csv   ← trajectory modeling candidates
  - regime_a_authors.csv   ← recency-only modeling candidates
  - author_regime_map.csv  ← full table with all metrics + regime label
"""

from rdflib import Graph, Namespace, RDF, URIRef
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_TTL      = "Upgraded6_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
TENSOR_2ND     = "author_topic_year_tensor_2nd_order.parquet"

OUTPUT_MAP     = "author_regime_map.csv"
OUTPUT_B       = "regime_b_authors.csv"
OUTPUT_A       = "regime_a_authors.csv"

# Eligibility thresholds
MIN_SPAN        = 6
MIN_PUBS        = 8
MIN_ACTIVE_YEARS = 3

DATASET_END_YEAR = 2019
RECENCY_WINDOW   = 5   # years for recency ratio

# ── Namespaces ─────────────────────────────────────────────────────────────────
EX      = Namespace("http://expert-search.org/schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# ── Step 1: Load KG author metrics ────────────────────────────────────────────
print(f"Loading KG from {INPUT_TTL}...")
g = Graph()
g.parse(INPUT_TTL, format="turtle")
g.bind("ex", EX)
g.bind("dcterms", DCTERMS)
print(f"  -> {len(g):,} triples loaded")

print("\nExtracting author metrics from KG...")
records = []
for author_uri in tqdm(g.subjects(RDF.type, EX.Author), desc="Authors"):
    name       = g.value(author_uri, DCTERMS.title)
    total_pubs = g.value(author_uri, EX.totalPubs)
    total_cit  = g.value(author_uri, EX.totalCitations)
    first_year = g.value(author_uri, EX.firstPublicationYear)
    last_year  = g.value(author_uri, EX.lastPublicationYear)
    active_yrs = g.value(author_uri, EX.activeYears)

    # Skip authors without year data
    if first_year is None or last_year is None:
        continue

    try:
        first      = int(str(first_year))
        last       = int(str(last_year))
        span       = last - first
        pubs       = int(str(total_pubs))   if total_pubs  else 0
        cit        = int(str(total_cit))    if total_cit   else 0
        act_yrs    = int(str(active_yrs))   if active_yrs  else 1
    except (ValueError, TypeError):
        continue

    records.append({
        "author_id"   : str(author_uri),
        "name"        : str(name) if name else "",
        "total_pubs"  : pubs,
        "total_cit"   : cit,
        "first_year"  : first,
        "last_year"   : last,
        "career_span" : span,
        "active_years": act_yrs,
    })

authors_df = pd.DataFrame(records)
print(f"  -> {len(authors_df):,} authors with complete metrics")

# ── Step 2: Compute recency ratio from tensor ─────────────────────────────────
print(f"\nComputing recency ratio from tensor (window={RECENCY_WINDOW} years)...")
tensor = pd.read_parquet(TENSOR_2ND)

# Total pubs per author from tensor
total_from_tensor = tensor.groupby("author_id")["frequency"].sum().reset_index()
total_from_tensor.columns = ["author_id", "tensor_total"]

# Recent pubs per author
recent = tensor[tensor["year"] >= DATASET_END_YEAR - RECENCY_WINDOW]
recent_counts = recent.groupby("author_id")["frequency"].sum().reset_index()
recent_counts.columns = ["author_id", "recent_pubs"]

# Merge and compute ratio
recency_df = total_from_tensor.merge(recent_counts, on="author_id", how="left")
recency_df["recent_pubs"]   = recency_df["recent_pubs"].fillna(0)
recency_df["recency_ratio"] = recency_df["recent_pubs"] / recency_df["tensor_total"]

authors_df = authors_df.merge(recency_df[["author_id", "recency_ratio"]],
                               on="author_id", how="left")
authors_df["recency_ratio"] = authors_df["recency_ratio"].fillna(0)

# ── Step 3: Apply regime partition ────────────────────────────────────────────
print("\nApplying regime partition...")

regime_b_mask = (
    (authors_df["career_span"]  >= MIN_SPAN) &
    (authors_df["total_pubs"]   >= MIN_PUBS) &
    (authors_df["active_years"] >= MIN_ACTIVE_YEARS)
)

authors_df["regime"] = "A"
authors_df.loc[regime_b_mask, "regime"] = "B"

regime_b = authors_df[authors_df["regime"] == "B"].copy()
regime_a = authors_df[authors_df["regime"] == "A"].copy()

# ── Step 4: Print summary ─────────────────────────────────────────────────────
total = len(authors_df)
print(f"\n{'='*60}")
print(f"REGIME PARTITION SUMMARY")
print(f"{'='*60}")
print(f"  Total authors with metrics     : {total:,}")
print(f"  Eligibility criteria:")
print(f"    career_span  >= {MIN_SPAN} years")
print(f"    total_pubs   >= {MIN_PUBS}")
print(f"    active_years >= {MIN_ACTIVE_YEARS}")
print(f"\n  Regime B (trajectory modeling) : {len(regime_b):,}  ({len(regime_b)/total*100:.1f}%)")
print(f"  Regime A (recency modeling)    : {len(regime_a):,}  ({len(regime_a)/total*100:.1f}%)")

print(f"\n  Regime B profile:")
print(f"    Median career span   : {regime_b['career_span'].median():.1f} years")
print(f"    Median total pubs    : {regime_b['total_pubs'].median():.0f}")
print(f"    Median active years  : {regime_b['active_years'].median():.1f}")
print(f"    Median recency ratio : {regime_b['recency_ratio'].median():.3f}")

print(f"\n  Regime A profile:")
print(f"    Median career span   : {regime_a['career_span'].median():.1f} years")
print(f"    Median total pubs    : {regime_a['total_pubs'].median():.0f}")
print(f"    Median active years  : {regime_a['active_years'].median():.1f}")
print(f"    Median recency ratio : {regime_a['recency_ratio'].median():.3f}")

# Stepwise filter impact
only_span  = (authors_df["career_span"]  >= MIN_SPAN).sum()
span_pubs  = ((authors_df["career_span"] >= MIN_SPAN) &
              (authors_df["total_pubs"]  >= MIN_PUBS)).sum()
print(f"\n  Stepwise filter impact:")
print(f"    span >= {MIN_SPAN}                          : {only_span:,}  ({only_span/total*100:.1f}%)")
print(f"    span >= {MIN_SPAN} AND pubs >= {MIN_PUBS}              : {span_pubs:,}  ({span_pubs/total*100:.1f}%)")
print(f"    span >= {MIN_SPAN} AND pubs >= {MIN_PUBS} AND active >= {MIN_ACTIVE_YEARS} : {len(regime_b):,}  ({len(regime_b)/total*100:.1f}%)")

# ── Step 5: Save outputs ──────────────────────────────────────────────────────
print(f"\nSaving outputs...")
authors_df.to_csv(OUTPUT_MAP, index=False)
regime_b.to_csv(OUTPUT_B, index=False)
regime_a.to_csv(OUTPUT_A, index=False)
print(f"  -> {OUTPUT_MAP}   ({len(authors_df):,} rows)")
print(f"  -> {OUTPUT_B}  ({len(regime_b):,} rows)")
print(f"  -> {OUTPUT_A}  ({len(regime_a):,} rows)")

# ── Step 6: Sanity check ─────────────────────────────────────────────────────
print(f"\nSanity check — sample Regime B authors (sorted by span):")
print(f"\n  {'Name':<30} {'Pubs':>5} {'Span':>5} {'ActYrs':>7} {'Recency':>8}")
print("  " + "-" * 60)
sample_b = regime_b.sort_values("career_span", ascending=False).head(10)
for _, row in sample_b.iterrows():
    print(f"  {row['name']:<30} {row['total_pubs']:>5} "
          f"{row['career_span']:>5} {row['active_years']:>7} "
          f"{row['recency_ratio']:>8.3f}")

print(f"\nSanity check — sample Regime A authors:")
print(f"\n  {'Name':<30} {'Pubs':>5} {'Span':>5} {'ActYrs':>7} {'Recency':>8}")
print("  " + "-" * 60)
sample_a = regime_a.sort_values("total_pubs", ascending=False).head(10)
for _, row in sample_a.iterrows():
    print(f"  {row['name']:<30} {row['total_pubs']:>5} "
          f"{row['career_span']:>5} {row['active_years']:>7} "
          f"{row['recency_ratio']:>8.3f}")

print(f"\n✅ Phase 1.1 complete.")
print(f"   Regime B → {OUTPUT_B}")
print(f"   Regime A → {OUTPUT_A}")
print(f"   Full map → {OUTPUT_MAP}")