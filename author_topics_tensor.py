"""
Phase 0.2 — Build Author–Topic–Year Tensor (dual granularity)

Builds TWO tensors in a single KG pass:
  - 2nd order topics → trajectory modeling (Phase 2, stable continuity)
  - 3rd order topics → query matching (Phase 4, fine-grained relevance)

Output files:
  author_topic_year_tensor_2nd.csv / .parquet  ← trajectory modeling
  author_topic_year_tensor_3rd.csv / .parquet  ← query matching

Each tensor row:
  author_id | year | topic_id | topic_name | frequency | binary | normalized

Weight definitions:
  binary     : 1 if author published on topic in that year
  frequency  : count of papers on that topic in that year
  normalized : frequency / total topics frequency per author-year (sums to 1)
"""

from rdflib import Graph, Namespace, RDF
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_TTL        = "Upgraded6_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
OUTPUT_2ND_CSV     = "author_topic_year_tensor_2nd_order.csv"
OUTPUT_2ND_PARQUET = "author_topic_year_tensor_2nd_order.parquet"
OUTPUT_3RD_CSV     = "author_topic_year_tensor_3rd_order.csv"
OUTPUT_3RD_PARQUET = "author_topic_year_tensor_3rd_order.parquet"

DATASET_YEAR_MIN = 1950
DATASET_YEAR_MAX = 2019

# ── Namespaces ─────────────────────────────────────────────────────────────────
EX      = Namespace("http://expert-search.org/schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_year(year_lit):
    raw = str(year_lit).strip()
    match = re.search(r'\d{4}', raw)
    if match:
        y = int(match.group())
        if DATASET_YEAR_MIN <= y <= DATASET_YEAR_MAX:
            return y
    return None

def build_dataframe(tensor, topic_names):
    """Convert (author, year, topic) -> count dict into a normalized DataFrame."""
    rows = []
    for (author_id, year, topic_id), freq in tensor.items():
        rows.append({
            "author_id"  : author_id,
            "year"       : year,
            "topic_id"   : topic_id,
            "topic_name" : topic_names.get(topic_id, topic_id.split("/")[-1]),
            "frequency"  : freq,
            "binary"     : 1,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # normalized weight per author-year
    totals = df.groupby(["author_id", "year"])["frequency"].transform("sum")
    df["normalized"] = df["frequency"] / totals
    return df

def print_stats(df, label):
    print(f"\n  [{label}] tensor stats:")
    print(f"    Rows            : {len(df):,}")
    print(f"    Unique authors  : {df['author_id'].nunique():,}")
    print(f"    Unique years    : {df['year'].nunique():,}")
    print(f"    Unique topics   : {df['topic_id'].nunique():,}")
    print(f"    Year range      : {df['year'].min()} – {df['year'].max()}")

# ── Step 1: Load KG ───────────────────────────────────────────────────────────
print(f"Loading KG from {INPUT_TTL}...")
g = Graph()
g.parse(INPUT_TTL, format="turtle")
g.bind("ex", EX)
g.bind("dcterms", DCTERMS)
print(f"  -> {len(g):,} triples loaded")

# ── Step 2: Build topic name lookup ───────────────────────────────────────────
print("\nBuilding topic name lookup...")
topic_names = {}
for uri, _, name_lit in g.triples((None, DCTERMS.title, None)):
    topic_names[str(uri)] = str(name_lit)
print(f"  -> {len(topic_names):,} names indexed")

# ── Step 3: Single-pass tensor build ─────────────────────────────────────────
print("\nBuilding tensors in single KG pass...")

tensor_2nd = defaultdict(int)   # (author, year, 2nd_topic) -> count
tensor_3rd = defaultdict(int)   # (author, year, 3rd_topic) -> count

papers_processed = 0
papers_skipped   = 0

for paper in tqdm(g.subjects(RDF.type, EX.Paper), desc="Papers"):

    # Year
    year_lit = g.value(paper, EX.publicationYear)
    if year_lit is None:
        papers_skipped += 1
        continue
    year = parse_year(year_lit)
    if year is None:
        papers_skipped += 1
        continue

    # Authors
    authors = list(g.objects(paper, EX.hasAuthor))
    if not authors:
        papers_skipped += 1
        continue

    # 2nd order topics
    topics_2nd = list(g.objects(paper, EX.hasSecondOrderTopic))

    # 3rd order topics
    topics_3rd = list(g.objects(paper, EX.hasThirdOrderTopic))

    if not topics_2nd and not topics_3rd:
        papers_skipped += 1
        continue

    for author in authors:
        author_id = str(author)

        for topic in topics_2nd:
            tensor_2nd[(author_id, year, str(topic))] += 1

        for topic in topics_3rd:
            tensor_3rd[(author_id, year, str(topic))] += 1

    papers_processed += 1

print(f"\n  -> Papers processed      : {papers_processed:,}")
print(f"  -> Papers skipped        : {papers_skipped:,}")
print(f"  -> 2nd order tensor size : {len(tensor_2nd):,} entries")
print(f"  -> 3rd order tensor size : {len(tensor_3rd):,} entries")

# ── Step 4: Build DataFrames ──────────────────────────────────────────────────
print("\nConverting to DataFrames...")
df_2nd = build_dataframe(tensor_2nd, topic_names)
df_3rd = build_dataframe(tensor_3rd, topic_names)

print_stats(df_2nd, "2nd order — trajectory modeling")
print_stats(df_3rd, "3rd order — query matching")

# ── Step 5: Save ─────────────────────────────────────────────────────────────
print(f"\nSaving 2nd order tensor...")
df_2nd.to_csv(OUTPUT_2ND_CSV, index=False)
df_2nd.to_parquet(OUTPUT_2ND_PARQUET, index=False)
print(f"  -> {OUTPUT_2ND_CSV}")
print(f"  -> {OUTPUT_2ND_PARQUET}")

print(f"\nSaving 3rd order tensor...")
df_3rd.to_csv(OUTPUT_3RD_CSV, index=False)
df_3rd.to_parquet(OUTPUT_3RD_PARQUET, index=False)
print(f"  -> {OUTPUT_3RD_CSV}")
print(f"  -> {OUTPUT_3RD_PARQUET}")

# ── Step 6: Sanity check ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SANITY CHECK")
print("=" * 60)

for df, label in [(df_2nd, "2nd order"), (df_3rd, "3rd order")]:
    print(f"\nTop 3 authors by tensor entries [{label}]:")
    top = (
        df.groupby("author_id")
        .agg(entries   =("frequency", "count"),
             years      =("year", "nunique"),
             topics     =("topic_id", "nunique"),
             total_pubs =("frequency", "sum"))
        .sort_values("entries", ascending=False)
        .head(3)
        .reset_index()
    )
    print(top.to_string(index=False))

    # Sample rows for top author
    sample_aid = top.iloc[0]["author_id"]
    print(f"\n  Sample rows for author {sample_aid} [{label}]:")
    sample = (df[df["author_id"] == sample_aid]
              .sort_values("year")
              .head(8))
    print(sample[["year", "topic_name", "frequency", "normalized"]]
          .to_string(index=False))

print("\n✅ Tensor build complete.")
print(f"   2nd order → use for: trajectory slicing, topic continuity, drift")
print(f"   3rd order → use for: query-author matching, fine-grained relevance")