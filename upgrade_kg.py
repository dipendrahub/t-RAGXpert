# """
# KG Upgrade — Add author-level metrics using Author.csv directly.

# Adds per author:
#   ex:totalPubs            (from n_pubs     in Author.csv)
#   ex:totalCitations       (from n_citation in Author.csv)
#   ex:firstPublicationYear (computed from papers.csv)
#   ex:lastPublicationYear  (computed from papers.csv)
# """

# from rdflib import Graph, Literal, RDF, URIRef, Namespace, XSD
# import pandas as pd
# import ast
# from collections import defaultdict
# from tqdm import tqdm

# # ── Config ─────────────────────────────────────────────────────────────────────
# INPUT_TTL   = "/Utilisateurs/dsharm01/repo/KG-Method/New_knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
# OUTPUT_TTL  = "Upgraded_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
# authors_df = pd.read_csv("~/repo/Expert-Finding/papers_and_authors/authors.csv")
# papers_df  = pd.read_csv("~/repo/KG-Method/test_output_df_with_hierarchical_topics.csv", encoding="latin1", low_memory=False)

# # ── Namespaces ─────────────────────────────────────────────────────────────────
# EX      = Namespace("http://expert-search.org/schema#")
# DCTERMS = Namespace("http://purl.org/dc/terms/")

# # ── Step 1: Load Author.csv ────────────────────────────────────────────────────
# print("Loading Author.csv...")
# authors_df["n_pubs"]     = pd.to_numeric(authors_df["n_pubs"],     errors="coerce").fillna(0).astype(int)
# authors_df["n_citation"] = pd.to_numeric(authors_df["n_citation"], errors="coerce").fillna(0).astype(int)
# print(f"  -> {len(authors_df):,} authors loaded")

# # ── Step 2: Compute first/last year from papers.csv ───────────────────────────
# print("Computing first/last publication year from papers.csv...")
# papers_df["year"] = pd.to_numeric(papers_df["year"], errors="coerce")

# author_years = defaultdict(list)

# for _, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Scanning papers"):
#     year = row["year"]
#     if pd.isna(year):
#         continue
#     try:
#         authors_list = ast.literal_eval(row["authors"])
#     except Exception:
#         continue
#     for a in authors_list:
#         if not isinstance(a, dict):
#             continue
#         a_id = a.get("id")
#         if a_id:
#             author_years[str(a_id)].append(int(year))

# print(f"  -> Year data found for {len(author_years):,} authors")

# # ── Step 3: Load KG ───────────────────────────────────────────────────────────
# print(f"\nLoading KG from {INPUT_TTL}...")
# g = Graph()
# g.parse(INPUT_TTL, format="turtle")
# g.bind("ex", EX)
# g.bind("dcterms", DCTERMS)
# print(f"  -> {len(g):,} triples loaded")

# # ── Step 4: Remove stale values (safe re-run) ─────────────────────────────────
# for pred in [EX.totalPubs, EX.totalCitations,
#              EX.firstPublicationYear, EX.lastPublicationYear]:
#     for s, _, o in list(g.triples((None, pred, None))):
#         g.remove((s, pred, o))

# # ── Step 5: Inject metrics ────────────────────────────────────────────────────
# print("\nInjecting author metrics...")
# injected = 0

# for _, row in tqdm(authors_df.iterrows(), total=len(authors_df), desc="Authors"):
#     aid        = str(row["id"])
#     author_uri = URIRef(f"http://expert-search.org/author/{aid}")

#     if (author_uri, RDF.type, EX.Author) not in g:
#         continue

#     # Directly from Author.csv — no recomputation needed
#     g.add((author_uri, EX.totalPubs,
#            Literal(row["n_pubs"], datatype=XSD.integer)))
#     g.add((author_uri, EX.totalCitations,
#            Literal(row["n_citation"], datatype=XSD.integer)))

#     # First/last year from papers.csv
#     years = author_years.get(aid, [])
#     if years:
#         g.add((author_uri, EX.firstPublicationYear,
#                Literal(min(years), datatype=XSD.gYear)))
#         g.add((author_uri, EX.lastPublicationYear,
#                Literal(max(years), datatype=XSD.gYear)))

#     injected += 1

# print(f"  -> Injected metrics for {injected:,} author nodes")
# print(f"  -> Total triples now: {len(g):,}")

# # ── Step 6: Serialize ─────────────────────────────────────────────────────────
# print(f"\nSerializing to {OUTPUT_TTL}...")
# g.serialize(OUTPUT_TTL, format="turtle")
# print("Done.")

# # ── Step 7: Sanity check ──────────────────────────────────────────────────────
# print("\nSanity check — 10 sample authors:")
# q = """
# PREFIX ex: <http://expert-search.org/schema#>
# PREFIX dcterms: <http://purl.org/dc/terms/>
# SELECT ?name ?pubs ?citations ?first ?last WHERE {
#     ?a a ex:Author ;
#        dcterms:title ?name ;
#        ex:totalPubs ?pubs ;
#        ex:totalCitations ?citations ;
#        ex:firstPublicationYear ?first ;
#        ex:lastPublicationYear ?last .
# } LIMIT 10
# """
# print(f"  {'Name':<30} {'Pubs':>5} {'Cit':>8} {'First':>6} {'Last':>6}")
# print("  " + "-" * 60)
# for row in g.query(q):
#     print(f"  {str(row.name):<30} {str(row.pubs):>5} {str(row.citations):>8} "
#           f"{str(row.first):>6} {str(row.last):>6}")


"""
KG Upgrade — Add author-level metrics using the KG itself as source of truth.

  ex:totalPubs            <- Author.csv  n_pubs
  ex:totalCitations       <- Author.csv  n_citation
  ex:firstPublicationYear <- SPARQL: MIN(year) over author's papers in KG
  ex:lastPublicationYear  <- SPARQL: MAX(year) over author's papers in KG
"""



# from rdflib import Graph, Literal, RDF, URIRef, Namespace, XSD
# import pandas as pd
# import numpy as np
# import re
# from collections import defaultdict
# from tqdm import tqdm

# DATASET_YEAR_MIN = 1950   # ignore papers before this year
# DATASET_YEAR_MAX = 2019   # ignore papers after this year
# MAX_CAREER_SPAN  = 50     # a human cannot realistically publish for >50 years

# # ── Config ─────────────────────────────────────────────────────────────────────
# INPUT_TTL   = "/Utilisateurs/dsharm01/repo/KG-Method/New_knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
# OUTPUT_TTL  = "Upgraded5_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
# authors_df = pd.read_csv("~/repo/Expert-Finding/papers_and_authors/authors.csv")
# papers_df  = pd.read_csv("~/repo/KG-Method/test_output_df_with_hierarchical_topics.csv", encoding="latin1", low_memory=False)


# # ── Namespaces ─────────────────────────────────────────────────────────────────
# EX      = Namespace("http://expert-search.org/schema#")
# DCTERMS = Namespace("http://purl.org/dc/terms/")

# def parse_year(year_lit):
#     """
#     Robustly parse a year literal regardless of xsd type or prefix.
#     Handles: "2010", "+2010", "2010-05", gYear quirks, etc.
#     Returns int year or None.
#     """
#     raw = str(year_lit).strip()
#     # Extract first 4-digit number found
#     match = re.search(r'\d{4}', raw)
#     if match:
#         return int(match.group())
#     return None

# # ── Step 1: Load Author.csv ────────────────────────────────────────────────────
# print("Loading Author.csv...")
# authors_df["n_pubs"]     = pd.to_numeric(authors_df["n_pubs"],     errors="coerce").fillna(0).astype(int)
# authors_df["n_citation"] = pd.to_numeric(authors_df["n_citation"], errors="coerce").fillna(0).astype(int)
# print(f"  -> {len(authors_df):,} authors loaded")

# # ── Step 2: Load KG ───────────────────────────────────────────────────────────
# print(f"\nLoading KG from {INPUT_TTL}...")
# g = Graph()
# g.parse(INPUT_TTL, format="turtle")
# g.bind("ex", EX)
# g.bind("dcterms", DCTERMS)
# print(f"  -> {len(g):,} triples loaded")

# # ── Step 3: Extract author -> publication years ───────────────────────────────
# print("\nExtracting author years from KG triples...")

# author_years = defaultdict(set)   # use set to get distinct years automatically

# for paper, _, author in tqdm(g.triples((None, EX.hasAuthor, None)),
#                               desc="Scanning paper-author links"):
#     for _, _, year_lit in g.triples((paper, EX.publicationYear, None)):
#         year = parse_year(year_lit)
#         if year is not None:
#             author_years[str(author)].add(year)

# print(f"  -> Raw year data found for {len(author_years):,} authors")

# # Apply career realism filter
# author_first       = {}
# author_last        = {}
# author_active_yrs  = {}
# filtered_out       = 0

# for aid, years in author_years.items():
#     years_sorted = sorted(years)
#     first = years_sorted[0]

#     # Only keep years within [first, first + MAX_CAREER_SPAN]
#     realistic_years = [y for y in years_sorted if y <= first + MAX_CAREER_SPAN]

#     if not realistic_years:
#         filtered_out += 1
#         continue

#     author_first[aid]      = first
#     author_last[aid]       = max(realistic_years)
#     author_active_yrs[aid] = len(realistic_years)   # distinct active years

# print(f"  -> After realism filter : {len(author_first):,} authors retained")
# print(f"  -> Filtered out         : {filtered_out} authors")

# spans = [author_last[a] - author_first[a] for a in author_first]
# print(f"\n  Span stats  : min={min(spans)}, max={max(spans)}, "
#       f"median={int(np.median(spans))}, mean={np.mean(spans):.1f}")
# print(f"  Span >= 5   : {sum(s >= 5  for s in spans):,}  ({sum(s >= 5  for s in spans)/len(spans)*100:.1f}%)")
# print(f"  Span >= 10  : {sum(s >= 10 for s in spans):,}  ({sum(s >= 10 for s in spans)/len(spans)*100:.1f}%)")

# # ── Step 4: Remove stale triples ──────────────────────────────────────────────
# print("\nRemoving stale triples...")
# removed = 0
# for pred in [EX.totalPubs, EX.totalCitations,
#              EX.firstPublicationYear, EX.lastPublicationYear, EX.activeYears]:
#     triples = list(g.triples((None, pred, None)))
#     for s, p, o in triples:
#         g.remove((s, p, o))
#     removed += len(triples)
# print(f"  -> Removed {removed} stale triples")

# # ── Step 5: Inject metrics ────────────────────────────────────────────────────
# print("\nInjecting author metrics...")
# injected = 0
# no_years = 0

# for _, row in tqdm(authors_df.iterrows(), total=len(authors_df), desc="Injecting"):
#     aid        = str(row["id"])
#     author_uri = URIRef(f"http://expert-search.org/author/{aid}")

#     if (author_uri, RDF.type, EX.Author) not in g:
#         continue

#     g.add((author_uri, EX.totalPubs,
#            Literal(row["n_pubs"], datatype=XSD.integer)))
#     g.add((author_uri, EX.totalCitations,
#            Literal(row["n_citation"], datatype=XSD.integer)))

#     uri_str = str(author_uri)
#     if uri_str in author_first:
#         g.add((author_uri, EX.firstPublicationYear,
#                Literal(author_first[uri_str], datatype=XSD.integer)))
#         g.add((author_uri, EX.lastPublicationYear,
#                Literal(author_last[uri_str], datatype=XSD.integer)))
#         g.add((author_uri, EX.activeYears,
#                Literal(author_active_yrs[uri_str], datatype=XSD.integer)))
#     else:
#         no_years += 1

#     injected += 1

# print(f"  -> Injected for     : {injected:,} author nodes")
# print(f"  -> No year data     : {no_years:,} authors")
# print(f"  -> Total triples    : {len(g):,}")

# # ── Step 6: Serialize ─────────────────────────────────────────────────────────
# print(f"\nSerializing to {OUTPUT_TTL}...")
# g.serialize(OUTPUT_TTL, format="turtle")
# print("Done.")

# # ── Step 7: Sanity check — top 10 by span ────────────────────────────────────
# print("\nSanity check — 10 authors with longest career span:")
# print(f"\n  {'Name':<30} {'Pubs':>5} {'Cit':>8} {'First':>6} {'Last':>6} {'Span':>5} {'ActYrs':>7}")
# print("  " + "-" * 75)

# samples = []
# for author_uri in g.subjects(RDF.type, EX.Author):
#     first    = g.value(author_uri, EX.firstPublicationYear)
#     last     = g.value(author_uri, EX.lastPublicationYear)
#     act_yrs  = g.value(author_uri, EX.activeYears)
#     if first is None or last is None:
#         continue
#     try:
#         span = int(str(last)) - int(str(first))
#     except ValueError:
#         continue
#     name = g.value(author_uri, DCTERMS.title)
#     pubs = g.value(author_uri, EX.totalPubs)
#     cit  = g.value(author_uri, EX.totalCitations)
#     samples.append((span, str(name or ""), str(pubs or 0), str(cit or 0),
#                     str(first), str(last), str(act_yrs or 0)))

# samples.sort(reverse=True)
# for span, name, pubs, cit, first, last, act in samples[:10]:
#     print(f"  {name:<30} {pubs:>5} {cit:>8} {first:>6} {last:>6} {span:>5} {act:>7}")

# # ── Step 8: Regime summary ────────────────────────────────────────────────────
# print("\nRegime summary:")
# with_years = len(samples)

# # Regime B: span>=6 AND total_pubs>=8 AND active_years>=3
# regime_b     = sum(1 for s, n, p, c, f, l, a in samples
#                    if s >= 6 and int(p) >= 8 and int(a) >= 3)

# # Show impact of each condition individually for transparency
# only_span    = sum(1 for s, n, p, c, f, l, a in samples if s >= 6)
# span_pubs    = sum(1 for s, n, p, c, f, l, a in samples if s >= 6 and int(p) >= 8)

# print(f"  -> Authors with year data                        : {with_years:,}")
# print(f"  -> span >= 6                                     : {only_span:,}  ({only_span/with_years*100:.1f}%)")
# print(f"  -> span >= 6 AND pubs >= 8                       : {span_pubs:,}  ({span_pubs/with_years*100:.1f}%)")
# print(f"  -> span >= 6 AND pubs >= 8 AND active_years >= 3 : {regime_b:,}  ({regime_b/with_years*100:.1f}%)  ← Regime B")
# print(f"  -> Regime A (short-span / episodic)              : {with_years-regime_b:,}  ({(with_years-regime_b)/with_years*100:.1f}%)")


"""
KG Upgrade — Add authorship position to existing KG.

For each paper-author pair, adds:
  ex:authorPosition  (xsd:integer, 1-based)
  ex:authorCount     (xsd:integer, total authors on paper)

Also adds convenience role triples directly on the paper:
  ex:hasFirstAuthor  -> author URI  (emerging expert signal)
  ex:hasLastAuthor   -> author URI  (supervising expert signal)

These are added as new triples alongside existing ex:hasAuthor triples.
ex:hasAuthor is preserved for backward compatibility.

Role mapping (CS convention):
  position = 1              -> first author  (emerging)
  position = 2...(n-1)      -> middle author (collaborating)
  position = n (last)       -> last author   (supervising)
  position = 1 AND n = 1    -> sole author   (treated as both first and last)
"""

from rdflib import Graph, Literal, RDF, URIRef, Namespace, XSD
import pandas as pd
import ast
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

INPUT_TTL   = "Upgraded5_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
OUTPUT_TTL  = "Upgraded6_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
papers_df  = pd.read_csv("~/repo/KG-Method/test_output_df_with_hierarchical_topics.csv", encoding="latin1", low_memory=False)

# ── Namespaces ─────────────────────────────────────────────────────────────────
EX      = Namespace("http://expert-search.org/schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# ── Step 1: Load KG ───────────────────────────────────────────────────────────
print(f"Loading KG from {INPUT_TTL}...")
g = Graph()
g.parse(INPUT_TTL, format="turtle")
g.bind("ex", EX)
g.bind("dcterms", DCTERMS)
print(f"  -> {len(g):,} triples loaded")

# ── Step 2: Remove stale position triples (safe re-run) ───────────────────────
print("\nRemoving stale position triples...")
removed = 0
for pred in [EX.authorPosition, EX.authorCount,
             EX.hasFirstAuthor, EX.hasLastAuthor]:
    triples = list(g.triples((None, pred, None)))
    for s, p, o in triples:
        g.remove((s, p, o))
    removed += len(triples)
print(f"  -> Removed {removed} stale triples")

# ── Step 3: Load papers CSV and inject position triples ───────────────────────
# print(f"\nLoading {PAPERS_CSV}...")
# papers_df = pd.read_csv(PAPERS_CSV)
print(f"  -> {len(papers_df):,} papers loaded")

injected     = 0
skipped      = 0
no_authors   = 0

# Track role counts for summary
first_author_count  = 0
middle_author_count = 0
last_author_count   = 0
sole_author_count   = 0

print("\nInjecting author position triples...")
for _, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Papers"):
    paper_id  = str(row.get("id", ""))
    paper_uri = URIRef(f"http://expert-search.org/paper/{paper_id}")

    # Skip if paper not in KG
    if (paper_uri, RDF.type, EX.Paper) not in g:
        skipped += 1
        continue

    authors_raw = row.get("authors")
    if pd.isna(authors_raw):
        no_authors += 1
        continue

    try:
        authors_list = ast.literal_eval(authors_raw)
    except Exception:
        no_authors += 1
        continue

    if not authors_list:
        no_authors += 1
        continue

    n = len(authors_list)

    # Add total author count on paper
    g.add((paper_uri, EX.authorCount,
           Literal(n, datatype=XSD.integer)))

    for pos, a in enumerate(authors_list, start=1):
        if not isinstance(a, dict):
            continue

        a_id   = a.get("id")
        a_name = a.get("name")
        if not a_id and not a_name:
            continue

        import re, hashlib
        def slugify(text, maxlen=80):
            s = re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")
            if not s:
                s = hashlib.sha1(str(text).encode()).hexdigest()[:8]
            return s[:maxlen]

        aid        = str(a_id) if a_id else slugify(a_name)
        author_uri = URIRef(f"http://expert-search.org/author/{aid}")

        # Add position triple: paper -> (author, position)
        # Using RDF reification-lite: separate position triple keyed by paper+author
        pos_uri = URIRef(f"http://expert-search.org/authorship/{paper_id}_{aid}")
        g.add((pos_uri, RDF.type,          EX.Authorship))
        g.add((pos_uri, EX.paper,          paper_uri))
        g.add((pos_uri, EX.author,         author_uri))
        g.add((pos_uri, EX.authorPosition, Literal(pos, datatype=XSD.integer)))
        g.add((pos_uri, EX.authorCount,    Literal(n,   datatype=XSD.integer)))

        # Convenience role triples on paper
        if pos == 1:
            g.add((paper_uri, EX.hasFirstAuthor, author_uri))
            if n == 1:
                # Sole author — counts as both first and last
                g.add((paper_uri, EX.hasLastAuthor, author_uri))
                sole_author_count += 1
            else:
                first_author_count += 1
        elif pos == n:
            g.add((paper_uri, EX.hasLastAuthor, author_uri))
            last_author_count += 1
        else:
            middle_author_count += 1

        injected += 1

print(f"\n  -> Authorship nodes created : {injected:,}")
print(f"  -> Papers skipped (not in KG): {skipped:,}")
print(f"  -> Papers with no authors   : {no_authors:,}")
print(f"  -> Total triples now        : {len(g):,}")

print(f"\n  Role distribution:")
print(f"    First author (emerging)      : {first_author_count:,}")
print(f"    Middle author (collaborating): {middle_author_count:,}")
print(f"    Last author (supervising)    : {last_author_count:,}")
print(f"    Sole author                  : {sole_author_count:,}")

# ── Step 4: Serialize ─────────────────────────────────────────────────────────
print(f"\nSerializing to {OUTPUT_TTL}...")
g.serialize(OUTPUT_TTL, format="turtle")
print("Done.")

# ── Step 5: Sanity check ──────────────────────────────────────────────────────
print("\nSanity check — sample authorship nodes:")
q = """
PREFIX ex: <http://expert-search.org/schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT ?paperTitle ?authorName ?pos ?total WHERE {
    ?auth a ex:Authorship ;
          ex:paper ?paper ;
          ex:author ?author ;
          ex:authorPosition ?pos ;
          ex:authorCount ?total .
    ?paper dcterms:title ?paperTitle .
    ?author dcterms:title ?authorName .
}
ORDER BY ?paperTitle ?pos
LIMIT 15
"""
print(f"\n  {'Paper':<40} {'Author':<25} {'Pos':>4} {'Total':>6}")
print("  " + "-" * 80)
for row in g.query(q):
    title = str(row.paperTitle)[:38]
    print(f"  {title:<40} {str(row.authorName):<25} "
          f"{str(row.pos):>4} {str(row.total):>6}")