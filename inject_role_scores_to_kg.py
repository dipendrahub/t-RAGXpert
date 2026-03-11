"""
Phase 4 — Step 1: Inject Role Scores into Knowledge Graph

Adds 4 triples per author node:
    ex:probEmerging      xsd:decimal
    ex:probCollaborating xsd:decimal
    ex:probSupervising   xsd:decimal
    ex:dominantRole      xsd:string

Input : Upgraded6_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl
        author_role_scores.csv
Output: Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl

After this, the KG is the single source of truth for:
    - Author metrics (span, pubs, citations, active years)
    - Authorship positions (pct_first, pct_middle, pct_last)
    - Topic assignments (1st, 2nd, 3rd order)
    - Temporal signals (first/last year, active years)
    - Role state (emerging / collaborating / supervising + soft probabilities)
"""

from rdflib import Graph, Namespace, URIRef, Literal, RDF, XSD
import pandas as pd
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_TTL   = "Upgraded6_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
OUTPUT_TTL  = "Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
ROLE_CSV    = "author_role_scores.csv"

# ── Namespace ──────────────────────────────────────────────────────────────────
EX = Namespace("http://expert-search.org/schema#")

# ── Load KG ───────────────────────────────────────────────────────────────────
print(f"Loading KG from {INPUT_TTL}...")
g = Graph()
g.parse(INPUT_TTL, format="turtle")
g.bind("ex", EX)
print(f"  -> {len(g):,} triples loaded")

# ── Load role scores ──────────────────────────────────────────────────────────
print(f"\nLoading role scores from {ROLE_CSV}...")
df = pd.read_csv(ROLE_CSV)
print(f"  -> {len(df):,} authors with role scores")

# ── Remove stale role triples if re-running ───────────────────────────────────
print("\nRemoving stale role triples (if any)...")
stale = 0
for pred in [EX.probEmerging, EX.probCollaborating,
             EX.probSupervising, EX.dominantRole]:
    triples = list(g.triples((None, pred, None)))
    for t in triples:
        g.remove(t)
        stale += 1
print(f"  -> Removed {stale} stale triples")

# ── Inject role triples ───────────────────────────────────────────────────────
print("\nInjecting role score triples...")
injected   = 0
not_found  = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Authors"):
    author_uri = URIRef(str(row["author_id"]))

    # Only inject if author exists in KG
    if (author_uri, RDF.type, EX.Author) not in g:
        not_found += 1
        continue

    g.add((author_uri, EX.probEmerging,
           Literal(round(float(row["prob_emerging"]), 4),     datatype=XSD.decimal)))
    g.add((author_uri, EX.probCollaborating,
           Literal(round(float(row["prob_collaborating"]), 4), datatype=XSD.decimal)))
    g.add((author_uri, EX.probSupervising,
           Literal(round(float(row["prob_supervising"]), 4),   datatype=XSD.decimal)))
    g.add((author_uri, EX.dominantRole,
           Literal(str(row["dominant_role"]),                  datatype=XSD.string)))

    injected += 1

print(f"  -> Authors injected  : {injected:,}")
print(f"  -> Authors not in KG : {not_found:,}")
print(f"  -> Total triples now : {len(g):,}  (+{injected*4:,} new)")

# ── Serialise ─────────────────────────────────────────────────────────────────
print(f"\nSerializing to {OUTPUT_TTL}...")
g.serialize(destination=OUTPUT_TTL, format="turtle")
print("Done.")

# ── Sanity check via SPARQL ───────────────────────────────────────────────────
print("\nSanity check — SPARQL query for top 5 emerging authors:")
sparql_emerging = """
PREFIX ex: <http://expert-search.org/schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT ?name ?probE ?probC ?probS ?role
WHERE {
    ?author a ex:Author ;
            dcterms:title      ?name ;
            ex:probEmerging     ?probE ;
            ex:probCollaborating ?probC ;
            ex:probSupervising  ?probS ;
            ex:dominantRole     ?role .
    FILTER(?role = "emerging")
}
ORDER BY DESC(?probE)
LIMIT 5
"""
results = g.query(sparql_emerging)
print(f"\n  {'Name':<30} {'P(E)':>6} {'P(C)':>6} {'P(S)':>6} {'Role'}")
print("  " + "-" * 65)
for row in results:
    print(f"  {str(row.name):<30} {float(row.probE):>6.3f} "
          f"{float(row.probC):>6.3f} {float(row.probS):>6.3f} "
          f"{str(row.role)}")

print("\nSanity check — SPARQL query for top 5 supervising authors:")
sparql_supervising = """
PREFIX ex: <http://expert-search.org/schema#>
PREFIX dcterms: <http://purl.org/dc/terms/>
SELECT ?name ?probE ?probC ?probS ?role
WHERE {
    ?author a ex:Author ;
            dcterms:title      ?name ;
            ex:probEmerging     ?probE ;
            ex:probCollaborating ?probC ;
            ex:probSupervising  ?probS ;
            ex:dominantRole     ?role .
    FILTER(?role = "supervising")
}
ORDER BY DESC(?probS)
LIMIT 5
"""
results = g.query(sparql_supervising)
print(f"\n  {'Name':<30} {'P(E)':>6} {'P(C)':>6} {'P(S)':>6} {'Role'}")
print("  " + "-" * 65)
for row in results:
    print(f"  {str(row.name):<30} {float(row.probE):>6.3f} "
          f"{float(row.probC):>6.3f} {float(row.probS):>6.3f} "
          f"{str(row.role)}")

print("\nSanity check — role distribution via SPARQL:")
sparql_dist = """
PREFIX ex: <http://expert-search.org/schema#>
SELECT ?role (COUNT(?author) AS ?count)
WHERE {
    ?author a ex:Author ;
            ex:dominantRole ?role .
}
GROUP BY ?role
ORDER BY DESC(?count)
"""
results = g.query(sparql_dist)
print(f"\n  {'Role':<20} {'Count':>8}")
print("  " + "-" * 30)
for row in results:
    print(f"  {str(row[0]):<20} {int(row[1]):>8,}")

print(f"\n✅ Role scores injected into KG.")
print(f"   Output: {OUTPUT_TTL}")