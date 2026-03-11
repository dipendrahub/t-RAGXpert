"""
This file is to get details about the dataset. 
The task is to figure out how many time slices we can divide the corpus into.
Also, can we divide in such a way for all the authors
"""

"""
Dataset Analysis Script
Covers:
  A. Dataset Overview
  B. Author Distribution
"""

# import pandas as pd
# import ast
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np
# from collections import Counter

# # ── 1. Load data ──────────────────────────────────────────────────────────────
# authors_df = pd.read_csv("~/repo/Expert-Finding/papers_and_authors/authors.csv")
# papers_df  = pd.read_csv("~/repo/KG-Method/test_output_df_with_hierarchical_topics.csv", encoding="latin1", low_memory=False)

# # ── A. DATASET OVERVIEW ───────────────────────────────────────────────────────
# print("=" * 55)
# print("A. DATASET OVERVIEW")
# print("=" * 55)

# total_authors = len(authors_df)
# total_papers  = len(papers_df)
# print(f"Total number of authors : {total_authors:,}")
# print(f"Total number of papers  : {total_papers:,}")

# # Year range
# papers_df["year"] = pd.to_numeric(papers_df["year"], errors="coerce")
# year_min = int(papers_df["year"].min())
# year_max = int(papers_df["year"].max())
# print(f"Year range              : {year_min} – {year_max}")

# # Papers per year distribution
# papers_per_year = (
#     papers_df["year"]
#     .value_counts()
#     .sort_index()
#     .rename_axis("Year")
#     .reset_index(name="Papers")
# )
# print("\nPapers per year distribution:")
# print(papers_per_year.to_string(index=False))

# # ── B. AUTHOR DISTRIBUTION ────────────────────────────────────────────────────
# print("\n" + "=" * 55)
# print("B. AUTHOR DISTRIBUTION")
# print("=" * 55)

# n_pubs = pd.to_numeric(authors_df["n_pubs"], errors="coerce").dropna()
# median_pubs = n_pubs.median()
# print(f"Median publications per author : {median_pubs}")

# total = len(n_pubs)
# lt5   = (n_pubs < 5).sum()
# b5_20 = ((n_pubs >= 5) & (n_pubs <= 20)).sum()
# gte50 = (n_pubs >= 50).sum()

# print(f"% authors with <5 papers       : {lt5/total*100:.1f}%  ({lt5:,} authors)")
# print(f"% authors with 5–20 papers     : {b5_20/total*100:.1f}%  ({b5_20:,} authors)")
# print(f"% authors with ≥50 papers      : {gte50/total*100:.1f}%  ({gte50:,} authors)")

# # ── PLOTS ─────────────────────────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle("Dataset Analysis", fontsize=14, fontweight="bold")

# # Plot 1 – Papers per year
# ax1 = axes[0]
# ax1.bar(papers_per_year["Year"], papers_per_year["Papers"],
#         color="#4C72B0", edgecolor="white", width=0.8)
# ax1.set_title("Papers per Year", fontsize=12)
# ax1.set_xlabel("Year")
# ax1.set_ylabel("Number of Papers")
# ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
# ax1.grid(axis="y", linestyle="--", alpha=0.5)

# # Plot 2 – Histogram of n_pubs (clipped for readability)
# ax2 = axes[1]
# clip = 100  # clip tail for visibility
# clipped = n_pubs.clip(upper=clip)
# bins = np.arange(0, clip + 5, 5)
# ax2.hist(clipped, bins=bins, color="#DD8452", edgecolor="white")
# ax2.set_title("Distribution of Publications per Author\n(capped at 100 for readability)", fontsize=11)
# ax2.set_xlabel("Number of Publications")
# ax2.set_ylabel("Number of Authors")
# ax2.axvline(median_pubs, color="red", linestyle="--", linewidth=1.5, label=f"Median = {median_pubs:.0f}")
# ax2.legend()
# ax2.grid(axis="y", linestyle="--", alpha=0.5)

# # Band annotations
# for x, label, color in [(0, "<5", "#2ca02c"), (5, "5–20", "#1f77b4"), (50, "≥50", "#d62728")]:
#     ax2.axvspan(x, min(x + (5 if x == 0 else 15 if x == 5 else 50), clip),
#                 alpha=0.08, color=color, label=label)

# plt.tight_layout()
# plt.savefig("dataset_analysis.png", dpi=150, bbox_inches="tight")
# print("\nPlot saved → dataset_analysis.png")
# plt.show()

#########################################################################################################

"""
Temporal Analysis Script
Computes:
  1. Distribution of Career Span
  2. Distribution of First Publication Year
  3. Distribution of Last Publication Year
  4. Recency Ratio Distribution
  5. Publications per 5-Year Slice per Author
  6. Slice Stability Conditions (A, B, C)
  7. Emerging Expert Representativeness
"""

import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Load data ──────────────────────────────────────────────────────────────────
authors_df = pd.read_csv("~/repo/Expert-Finding/papers_and_authors/authors.csv")
papers_df  = pd.read_csv("~/repo/KG-Method/test_output_df_with_hierarchical_topics.csv", encoding="latin1", low_memory=False)
papers_df["year"] = pd.to_numeric(papers_df["year"], errors="coerce")

# DATASET_END_YEAR = int(papers_df["year"].max())
# RECENCY_WINDOW   = 3   # years for recency ratio
# SLICE_SIZE       = 3   # years per temporal slice

# print("=" * 60)
# print(f"Dataset end year : {DATASET_END_YEAR}")
# print(f"Recency window   : {RECENCY_WINDOW} years")
# print(f"Slice size       : {SLICE_SIZE} years")
# print("=" * 60)

# # ── Build author → paper years lookup ─────────────────────────────────────────
# # Parse the author list from papers
# author_years = defaultdict(list)   # author_id -> [year, year, ...]

# for _, row in papers_df.iterrows():
#     year = row["year"]
#     if pd.isna(year):
#         continue
#     year = int(year)
#     try:
#         authors = ast.literal_eval(row["authors"])
#     except Exception:
#         continue
#     for a in authors:
#         aid = str(a.get("id", "")).strip()
#         if aid:
#             author_years[aid].append(year)

# # Build per-author stats dataframe
# records = []
# for _, row in authors_df.iterrows():
#     aid = str(row["id"])
#     years = sorted(author_years.get(aid, []))

#     if not years:
#         # Fall back to n_pubs only — no year info available
#         first, last, span = np.nan, np.nan, np.nan
#         recency_ratio = np.nan
#         avg_pubs_per_slice = np.nan
#         n_slices = np.nan
#     else:
#         first = min(years)
#         last  = max(years)
#         span  = last - first

#         # Recency ratio
#         recent_count = sum(1 for y in years if y >= DATASET_END_YEAR - RECENCY_WINDOW)
#         recency_ratio = recent_count / len(years)

#         # Publications per 5-year slice
#         slice_start = (first // SLICE_SIZE) * SLICE_SIZE
#         slice_counts = defaultdict(int)
#         for y in years:
#             s = ((y - slice_start) // SLICE_SIZE) * SLICE_SIZE + slice_start
#             slice_counts[s] += 1
#         n_slices = len(slice_counts)
#         avg_pubs_per_slice = np.mean(list(slice_counts.values()))

#     records.append({
#         "id"               : aid,
#         "n_pubs"           : row["n_pubs"],
#         "first_year"       : first,
#         "last_year"        : last,
#         "career_span"      : span,
#         "recency_ratio"    : recency_ratio,
#         "n_slices"         : n_slices,
#         "avg_pubs_per_slice": avg_pubs_per_slice,
#     })

# stats = pd.DataFrame(records)
# valid = stats.dropna(subset=["career_span"])   # authors with year data

# # ══════════════════════════════════════════════════════════════════════════════
# # 1. CAREER SPAN
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n1. CAREER SPAN")
# print("-" * 40)
# span = valid["career_span"]
# print(f"  Median span          : {span.median():.1f} years")
# print(f"  Mean span            : {span.mean():.1f} years")
# print(f"  % span  < 5 years    : {(span < 5).mean()*100:.1f}%")
# print(f"  % span  5–10 years   : {((span >= 5) & (span <= 10)).mean()*100:.1f}%")
# print(f"  % span 11–20 years   : {((span > 10) & (span <= 20)).mean()*100:.1f}%")
# print(f"  % span > 20 years    : {(span > 20).mean()*100:.1f}%")

# # ══════════════════════════════════════════════════════════════════════════════
# # 2. FIRST PUBLICATION YEAR
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n2. FIRST PUBLICATION YEAR")
# print("-" * 40)
# fy = valid["first_year"]
# print(f"  Min first year   : {int(fy.min())}")
# print(f"  Max first year   : {int(fy.max())}")
# print(f"  Median           : {fy.median():.0f}")
# print(f"  % first year ≥ 2000 : {(fy >= 2000).mean()*100:.1f}%")
# print(f"  % first year ≥ 2010 : {(fy >= 2010).mean()*100:.1f}%")

# # ══════════════════════════════════════════════════════════════════════════════
# # 3. LAST PUBLICATION YEAR
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n3. LAST PUBLICATION YEAR")
# print("-" * 40)
# ly = valid["last_year"]
# cutoff = DATASET_END_YEAR - 10
# print(f"  Min last year    : {int(ly.min())}")
# print(f"  Max last year    : {int(ly.max())}")
# print(f"  Median           : {ly.median():.0f}")
# print(f"  % last year < {cutoff}   : {(ly < cutoff).mean()*100:.1f}%  ← potentially inactive")
# print(f"  % last year ≥ {DATASET_END_YEAR - 3} : {(ly >= DATASET_END_YEAR - 3).mean()*100:.1f}%  ← recently active")

# # ══════════════════════════════════════════════════════════════════════════════
# # 4. RECENCY RATIO
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n4. RECENCY RATIO  (pubs in last 5yr / total pubs)")
# print("-" * 40)
# rr = valid["recency_ratio"].dropna()
# print(f"  Median recency ratio : {rr.median():.3f}")
# print(f"  Mean recency ratio   : {rr.mean():.3f}")
# print(f"  % ratio near 0   (< 0.1)  : {(rr < 0.1).mean()*100:.1f}%  ← mostly historical")
# print(f"  % ratio mid      (0.1–0.5): {((rr >= 0.1) & (rr <= 0.5)).mean()*100:.1f}%  ← stable active")
# print(f"  % ratio high     (> 0.5)  : {(rr > 0.5).mean()*100:.1f}%  ← mostly recent / emerging")

# # ══════════════════════════════════════════════════════════════════════════════
# # 5. PUBLICATIONS PER 5-YEAR SLICE
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n5. PUBLICATIONS PER 5-YEAR SLICE")
# print("-" * 40)
# aps = valid["avg_pubs_per_slice"].dropna()
# print(f"  Median avg pubs/slice : {aps.median():.2f}")
# print(f"  Mean avg pubs/slice   : {aps.mean():.2f}")
# print(f"  % authors avg < 2/slice  : {(aps < 2).mean()*100:.1f}%  ← unstable for topic modeling")
# print(f"  % authors avg 2–5/slice  : {((aps >= 2) & (aps <= 5)).mean()*100:.1f}%  ← acceptable")
# print(f"  % authors avg > 5/slice  : {(aps > 5).mean()*100:.1f}%  ← well-represented")

# # ══════════════════════════════════════════════════════════════════════════════
# # 6. SLICE STABILITY CONDITIONS
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n6. SLICE STABILITY CONDITIONS")
# print("-" * 40)
# ns = valid["n_slices"].dropna()
# cond_a = (ns >= 2).mean() * 100
# cond_b = (aps >= 2).mean() * 100
# # Condition C: emerging authors (short career) not artificially split
# emerging_mask = valid["career_span"] <= SLICE_SIZE
# cond_c_ok = (valid.loc[emerging_mask, "n_slices"] == 1).mean() * 100

# print(f"  Condition A — % authors in ≥2 slices        : {cond_a:.1f}%  (want > 50%)")
# print(f"  Condition B — % authors avg ≥2 pubs/slice   : {cond_b:.1f}%  (want > 50%)")
# print(f"  Condition C — % emerging in exactly 1 slice : {cond_c_ok:.1f}%  (want > 70%)")

# if cond_a >= 50 and cond_b >= 50:
#     print("  ✅ 5-year slicing appears STABLE for this dataset")
# else:
#     print("  ⚠️  Consider 3-year slicing — 5-year slices may be too coarse")

# # ══════════════════════════════════════════════════════════════════════════════
# # 7. EMERGING EXPERT REPRESENTATIVENESS
# # ══════════════════════════════════════════════════════════════════════════════
# print("\n7. EMERGING EXPERT REPRESENTATIVENESS")
# print("-" * 40)
# recent_threshold = DATASET_END_YEAR - 5
# emerging = valid[
#     (valid["first_year"] >= recent_threshold) |
#     (valid["recency_ratio"] > 0.7)
# ]
# print(f"  Total authors with year data  : {len(valid):,}")
# print(f"  Emerging candidates (recent first_year OR high recency) : {len(emerging):,}  ({len(emerging)/len(valid)*100:.1f}%)")
# print(f"  Median career span (emerging) : {emerging['career_span'].median():.1f} years")
# print(f"  Median recency ratio (emerging): {emerging['recency_ratio'].median():.3f}")

# if len(emerging) / len(valid) >= 0.1:
#     print("  ✅ Sufficient emerging expert signal in dataset")
# else:
#     print("  ⚠️  Low emerging expert count — consider widening definition")

# # ══════════════════════════════════════════════════════════════════════════════
# # PLOTS
# # ══════════════════════════════════════════════════════════════════════════════
# fig = plt.figure(figsize=(18, 12))
# gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
# fig.suptitle("Temporal Analysis of Author Dataset", fontsize=15, fontweight="bold")

# COLOR = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# def hplot(ax, data, title, xlabel, color, bins=30, vline=None):
#     ax.hist(data.dropna(), bins=bins, color=color, edgecolor="white")
#     ax.set_title(title, fontsize=11)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Authors")
#     ax.grid(axis="y", linestyle="--", alpha=0.4)
#     if vline is not None:
#         ax.axvline(vline, color="red", linestyle="--", linewidth=1.5,
#                    label=f"Median={vline:.1f}")
#         ax.legend(fontsize=9)

# # 1 — Career span
# hplot(fig.add_subplot(gs[0, 0]), span, "1. Career Span", "Years", COLOR[0],
#       bins=40, vline=span.median())

# # 2 — First pub year
# hplot(fig.add_subplot(gs[0, 1]), fy, "2. First Publication Year", "Year", COLOR[1],
#       bins=30, vline=fy.median())

# # 3 — Last pub year
# hplot(fig.add_subplot(gs[0, 2]), ly, "3. Last Publication Year", "Year", COLOR[2],
#       bins=30, vline=ly.median())

# # 4 — Recency ratio
# ax4 = fig.add_subplot(gs[1, 0])
# ax4.hist(rr, bins=20, color=COLOR[3], edgecolor="white")
# ax4.axvline(rr.median(), color="red", linestyle="--", linewidth=1.5,
#             label=f"Median={rr.median():.2f}")
# ax4.set_title("4. Recency Ratio", fontsize=11)
# ax4.set_xlabel("Ratio (recent pubs / total)")
# ax4.set_ylabel("Authors")
# ax4.legend(fontsize=9)
# ax4.grid(axis="y", linestyle="--", alpha=0.4)

# # 5 — Avg pubs per slice
# hplot(fig.add_subplot(gs[1, 1]), aps, "5. Avg Pubs per 5-Year Slice",
#       "Avg Publications", COLOR[4], bins=30, vline=aps.median())

# # 6 — Number of slices per author
# ax6 = fig.add_subplot(gs[1, 2])
# ns.clip(upper=15).hist(ax=ax6, bins=15, color=COLOR[5], edgecolor="white")
# ax6.axvline(2, color="red", linestyle="--", linewidth=1.5, label="Min stable (2)")
# ax6.set_title("6. Number of Slices per Author\n(capped at 15)", fontsize=11)
# ax6.set_xlabel("Slices")
# ax6.set_ylabel("Authors")
# ax6.legend(fontsize=9)
# ax6.grid(axis="y", linestyle="--", alpha=0.4)

# plt.savefig("temporal_analysis.png", dpi=150, bbox_inches="tight")
# print("\nPlot saved → temporal_analysis_3years.png")
# plt.show()

"""
Temporal Analysis Script
Computes:
  1. Distribution of Career Span
  2. Distribution of First Publication Year
  3. Distribution of Last Publication Year
  4. Recency Ratio Distribution
  5. Publications per 5-Year Slice per Author
  6. Slice Stability Conditions (A, B, C)
  7. Emerging Expert Representativeness
"""


DATASET_END_YEAR = int(papers_df["year"].max())
RECENCY_WINDOW   = 5   # years for recency ratio
SLICE_SIZE       = 3   # years per temporal slice

print("=" * 60)
print(f"Dataset end year : {DATASET_END_YEAR}")
print(f"Recency window   : {RECENCY_WINDOW} years")
print(f"Slice size       : {SLICE_SIZE} years")
print("=" * 60)

# ── Build author → paper years lookup ─────────────────────────────────────────
# Parse the author list from papers
author_years = defaultdict(list)   # author_id -> [year, year, ...]

for _, row in papers_df.iterrows():
    year = row["year"]
    if pd.isna(year):
        continue
    year = int(year)
    try:
        authors = ast.literal_eval(row["authors"])
    except Exception:
        continue
    for a in authors:
        aid = str(a.get("id", "")).strip()
        if aid:
            author_years[aid].append(year)

# Build per-author stats dataframe
records = []
for _, row in authors_df.iterrows():
    aid = str(row["id"])
    years = sorted(author_years.get(aid, []))

    if not years:
        # Fall back to n_pubs only — no year info available
        first, last, span = np.nan, np.nan, np.nan
        recency_ratio = np.nan
        avg_pubs_per_slice = np.nan
        n_slices = np.nan
    else:
        first = min(years)
        last  = max(years)
        span  = last - first

        # Recency ratio
        recent_count = sum(1 for y in years if y >= DATASET_END_YEAR - RECENCY_WINDOW)
        recency_ratio = recent_count / len(years)

        # Publications per 5-year slice
        slice_start = (first // SLICE_SIZE) * SLICE_SIZE
        slice_counts = defaultdict(int)
        for y in years:
            s = ((y - slice_start) // SLICE_SIZE) * SLICE_SIZE + slice_start
            slice_counts[s] += 1
        n_slices = len(slice_counts)
        avg_pubs_per_slice = np.mean(list(slice_counts.values()))

    records.append({
        "id"               : aid,
        "n_pubs"           : row["n_pubs"],
        "first_year"       : first,
        "last_year"        : last,
        "career_span"      : span,
        "recency_ratio"    : recency_ratio,
        "n_slices"         : n_slices,
        "avg_pubs_per_slice": avg_pubs_per_slice,
    })

stats = pd.DataFrame(records)
valid = stats.dropna(subset=["career_span"])   # authors with year data

# ══════════════════════════════════════════════════════════════════════════════
# 1. CAREER SPAN
# ══════════════════════════════════════════════════════════════════════════════
print("\n1. CAREER SPAN")
print("-" * 40)
span = valid["career_span"]
print(f"  Median span          : {span.median():.1f} years")
print(f"  Mean span            : {span.mean():.1f} years")
print(f"  % span  < 5 years    : {(span < 5).mean()*100:.1f}%")
print(f"  % span  5–10 years   : {((span >= 5) & (span <= 10)).mean()*100:.1f}%")
print(f"  % span 11–20 years   : {((span > 10) & (span <= 20)).mean()*100:.1f}%")
print(f"  % span > 20 years    : {(span > 20).mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FIRST PUBLICATION YEAR
# ══════════════════════════════════════════════════════════════════════════════
print("\n2. FIRST PUBLICATION YEAR")
print("-" * 40)
fy = valid["first_year"]
print(f"  Min first year   : {int(fy.min())}")
print(f"  Max first year   : {int(fy.max())}")
print(f"  Median           : {fy.median():.0f}")
print(f"  % first year ≥ 2000 : {(fy >= 2000).mean()*100:.1f}%")
print(f"  % first year ≥ 2010 : {(fy >= 2010).mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 3. LAST PUBLICATION YEAR
# ══════════════════════════════════════════════════════════════════════════════
print("\n3. LAST PUBLICATION YEAR")
print("-" * 40)
ly = valid["last_year"]
cutoff = DATASET_END_YEAR - 10
print(f"  Min last year    : {int(ly.min())}")
print(f"  Max last year    : {int(ly.max())}")
print(f"  Median           : {ly.median():.0f}")
print(f"  % last year < {cutoff}   : {(ly < cutoff).mean()*100:.1f}%  ← potentially inactive")
print(f"  % last year ≥ {DATASET_END_YEAR - 3} : {(ly >= DATASET_END_YEAR - 3).mean()*100:.1f}%  ← recently active")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RECENCY RATIO
# ══════════════════════════════════════════════════════════════════════════════
print("\n4. RECENCY RATIO  (pubs in last 5yr / total pubs)")
print("-" * 40)
rr = valid["recency_ratio"].dropna()
print(f"  Median recency ratio : {rr.median():.3f}")
print(f"  Mean recency ratio   : {rr.mean():.3f}")
print(f"  % ratio near 0   (< 0.1)  : {(rr < 0.1).mean()*100:.1f}%  ← mostly historical")
print(f"  % ratio mid      (0.1–0.5): {((rr >= 0.1) & (rr <= 0.5)).mean()*100:.1f}%  ← stable active")
print(f"  % ratio high     (> 0.5)  : {(rr > 0.5).mean()*100:.1f}%  ← mostly recent / emerging")

# ══════════════════════════════════════════════════════════════════════════════
# 5. PUBLICATIONS PER 5-YEAR SLICE
# ══════════════════════════════════════════════════════════════════════════════
print("\n5. PUBLICATIONS PER 5-YEAR SLICE")
print("-" * 40)
aps = valid["avg_pubs_per_slice"].dropna()
print(f"  Median avg pubs/slice : {aps.median():.2f}")
print(f"  Mean avg pubs/slice   : {aps.mean():.2f}")
print(f"  % authors avg < 2/slice  : {(aps < 2).mean()*100:.1f}%  ← unstable for topic modeling")
print(f"  % authors avg 2–5/slice  : {((aps >= 2) & (aps <= 5)).mean()*100:.1f}%  ← acceptable")
print(f"  % authors avg > 5/slice  : {(aps > 5).mean()*100:.1f}%  ← well-represented")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SLICE STABILITY CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n6. SLICE STABILITY CONDITIONS")
print("-" * 40)
ns = valid["n_slices"].dropna()
cond_a = (ns >= 2).mean() * 100
cond_b = (aps >= 2).mean() * 100
# Condition C: emerging authors (short career) not artificially split
emerging_mask = valid["career_span"] <= SLICE_SIZE
cond_c_ok = (valid.loc[emerging_mask, "n_slices"] == 1).mean() * 100

print(f"  Condition A — % authors in ≥2 slices        : {cond_a:.1f}%  (want > 50%)")
print(f"  Condition B — % authors avg ≥2 pubs/slice   : {cond_b:.1f}%  (want > 50%)")
print(f"  Condition C — % emerging in exactly 1 slice : {cond_c_ok:.1f}%  (want > 70%)")

if cond_a >= 50 and cond_b >= 50:
    print("  ✅ 5-year slicing appears STABLE for this dataset")
else:
    print("  ⚠️  Consider 3-year slicing — 5-year slices may be too coarse")

# ══════════════════════════════════════════════════════════════════════════════
# 7. EMERGING EXPERT REPRESENTATIVENESS
# ══════════════════════════════════════════════════════════════════════════════
print("\n7. EMERGING EXPERT REPRESENTATIVENESS")
print("-" * 40)
recent_threshold = DATASET_END_YEAR - 5
emerging = valid[
    (valid["first_year"] >= recent_threshold) |
    (valid["recency_ratio"] > 0.7)
]
print(f"  Total authors with year data  : {len(valid):,}")
print(f"  Emerging candidates (recent first_year OR high recency) : {len(emerging):,}  ({len(emerging)/len(valid)*100:.1f}%)")
print(f"  Median career span (emerging) : {emerging['career_span'].median():.1f} years")
print(f"  Median recency ratio (emerging): {emerging['recency_ratio'].median():.3f}")

if len(emerging) / len(valid) >= 0.1:
    print("  ✅ Sufficient emerging expert signal in dataset")
else:
    print("  ⚠️  Low emerging expert count — consider widening definition")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("", fontsize=30, fontweight="bold")
plt.rcParams.update({
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.labelsize': 15,   # ← controls xlabel and ylabel
})

COLOR = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

def hplot(ax, data, title, xlabel, color, bins=30, vline=None):
    ax.hist(data.dropna(), bins=bins, color=color, edgecolor="white")
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Authors")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", linewidth=1.5,
                   label=f"Median={vline:.1f}")
        ax.legend(fontsize=18)

# 1 — Career span
hplot(fig.add_subplot(gs[0, 0]), span, "1. Career Span", "Years", COLOR[0],
      bins=40, vline=span.median())

# 2 — First pub year
hplot(fig.add_subplot(gs[0, 1]), fy, "2. First Publication Year", "Year", COLOR[1],
      bins=30, vline=fy.median())

# 3 — Last pub year
hplot(fig.add_subplot(gs[0, 2]), ly, "3. Last Publication Year", "Year", COLOR[2],
      bins=30, vline=ly.median())

# 4 — Recency ratio
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(rr, bins=20, color=COLOR[3], edgecolor="white")
ax4.axvline(rr.median(), color="red", linestyle="--", linewidth=1.5,
            label=f"Median={rr.median():.2f}")
ax4.set_title("4. Recency Ratio", fontsize=24)
ax4.set_xlabel("Ratio (recent pubs / total)")
ax4.set_ylabel("Authors")
ax4.legend(fontsize=15)
ax4.grid(axis="y", linestyle="--", alpha=0.4)

# 5 — Avg pubs per slice
hplot(fig.add_subplot(gs[1, 1]), aps, "5. Avg Pubs per 3-Year Slice",
      "Avg Publications", COLOR[4], bins=30, vline=aps.median())

# 6 — Number of slices per author
ax6 = fig.add_subplot(gs[1, 2])
ns.clip(upper=15).hist(ax=ax6, bins=15, color=COLOR[5], edgecolor="white")
ax6.axvline(2, color="red", linestyle="--", linewidth=1.5, label="Min stable (2)")
ax6.set_title("6. Number of Slices per Author\n(capped at 15, 3-yr windows)", fontsize=24)
ax6.set_xlabel("Slices")
ax6.set_ylabel("Authors")
ax6.legend(fontsize=15)
ax6.grid(axis="y", linestyle="--", alpha=0.4)

plt.savefig("temporal_analysis_3yr1.png", dpi=300, bbox_inches="tight")
print("\nPlot saved → temporal_analysis_3yr.png")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 8. CORE TRAJECTORY POPULATION (Temporal Eligible Authors)
# ══════════════════════════════════════════════════════════════════════════════
print("\n8. CORE TRAJECTORY POPULATION  (span ≥ 6 AND pubs ≥ 8)")
print("-" * 40)

total_with_data = len(valid)
core       = valid[(valid["career_span"] >= 6) & (valid["n_pubs"] >= 8)]
short_span = valid[~valid.index.isin(core.index)]

pct_core  = len(core)  / total_with_data * 100
pct_short = len(short_span) / total_with_data * 100

print(f"  Total authors with year data              : {total_with_data:,}")
print(f"  Regime B  Core  (span>=6 & pubs>=8)       : {len(core):,}  ({pct_core:.1f}%)  <- full trajectory modeling")
print(f"  Regime A  Short (rest)                    : {len(short_span):,}  ({pct_short:.1f}%)  <- recency-only modeling")

print(f"\n  Regime B - Core group profile:")
print(f"    Median career span     : {core['career_span'].median():.1f} years")
print(f"    Median total pubs      : {core['n_pubs'].median():.0f}")
print(f"    Median recency ratio   : {core['recency_ratio'].median():.3f}")
print(f"    Median slices          : {core['n_slices'].median():.1f}")

print(f"\n  Regime A - Short-span group profile:")
print(f"    Median career span     : {short_span['career_span'].median():.1f} years")
print(f"    Median total pubs      : {short_span['n_pubs'].median():.0f}")
print(f"    Median recency ratio   : {short_span['recency_ratio'].median():.3f}")

if pct_core >= 15:
    print(f"\n  OK - Core population ({pct_core:.1f}%) is sufficient for trajectory modeling")
else:
    print(f"\n  WARNING - Core population ({pct_core:.1f}%) is small - verify eligibility thresholds")

# Regime breakdown pie chart
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("", fontsize=24, fontweight="bold")

# Pie
axes2[0].pie(
    [len(core), len(short_span)],
    labels=[f"Regime B\nCore\n({pct_core:.1f}%)", f"Regime A\nShort-span\n({pct_short:.1f}%)"],
    colors=["#4C72B0", "#DD8452"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
axes2[0].set_title("Regime Split", fontsize=24)
plt.rcParams.update({'xtick.labelsize': 13, 'ytick.labelsize': 13})

# Career span coloured by regime
ax_span = axes2[1]
ax_span.hist(short_span["career_span"].dropna(), bins=40, color="#DD8452",
             alpha=0.7, label=f"Regime A (n={len(short_span):,})", edgecolor="white")
ax_span.hist(core["career_span"].dropna(), bins=40, color="#4C72B0",
             alpha=0.8, label=f"Regime B (n={len(core):,})", edgecolor="white")
ax_span.axvline(6, color="red", linestyle="--", linewidth=1.5, label="Threshold (6 yrs)")
ax_span.set_title("Career Span by Regime", fontsize=24)
ax_span.set_xlabel("Career Span (years)")
ax_span.set_ylabel("Authors")
ax_span.legend(fontsize=15)
ax_span.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("regime_partition1.png", dpi=300, bbox_inches="tight")
print("\nPlot saved -> regime_partition.png")
plt.show()