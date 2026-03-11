# import pandas as pd
# import ast

# authors_df = pd.read_csv("~/repo/Expert-Finding/papers_and_authors/authors.csv")
# papers_df  = pd.read_csv("~/repo/KG-Method/test_output_df_with_hierarchical_topics.csv", encoding="latin1", low_memory=False)
# papers_df["id"] = papers_df["id"].astype(str)

# # Sample paper IDs from papers.csv
# print("=== Sample paper IDs from papers.csv ===")
# print(papers_df["id"].head(5).tolist())

# # Sample pub IDs from Author.csv pubs field
# sample_author = authors_df.iloc[0]
# print("\n=== Sample pubs field from Author.csv (first author) ===")
# print(sample_author["pubs"][:200])

# # Parse and extract IDs
# pubs = ast.literal_eval(sample_author["pubs"])
# pub_ids = [str(p["i"]) for p in pubs if isinstance(p, dict)]
# print("\n=== Extracted pub IDs ===")
# print(pub_ids[:5])

# # Check overlap
# paper_id_set = set(papers_df["id"].tolist())
# matches = [pid for pid in pub_ids if pid in paper_id_set]
# print(f"\n=== Overlap: {len(matches)} of {len(pub_ids)} pub IDs found in papers.csv ===")
# print("Matched:", matches[:3])
# print("Unmatched sample:", [p for p in pub_ids if p not in paper_id_set][:3])


"""
Diagnostic — check why author_years lookup is failing.
Run this locally to identify the key mismatch.
"""

from rdflib import Graph, Namespace, RDF, URIRef
import pandas as pd

EX      = Namespace("http://expert-search.org/schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_TTL   = "/Utilisateurs/dsharm01/repo/KG-Method/New_knowledge_graph_with_hierarchical_topics_full_dataset.ttl"

authors_df = pd.read_csv("~/repo/Expert-Finding/papers_and_authors/authors.csv")


# Load
print("Loading KG...")
g = Graph()
g.parse(INPUT_TTL, format="turtle")



# Run SPARQL
print("Running SPARQL...")
sparql_years = """
PREFIX ex: <http://expert-search.org/schema#>
SELECT ?author (MIN(?year) AS ?firstYear) (MAX(?year) AS ?lastYear)
WHERE {
    ?paper a ex:Paper ;
           ex:hasAuthor ?author ;
           ex:publicationYear ?year .
}
GROUP BY ?author
"""
results = list(g.query(sparql_years))
print(f"SPARQL returned {len(results)} rows")

# Show raw key format from SPARQL
print("\n=== Sample SPARQL keys (str(row.author)) ===")
for row in results[:5]:
    print(f"  repr: {repr(str(row.author))}")

# Show what we build as author_uri
print("\n=== Sample author URIs we build ===")
for _, row in authors_df.head(5).iterrows():
    aid = str(row["id"])
    uri = URIRef(f"http://expert-search.org/author/{aid}")
    print(f"  repr: {repr(str(uri))}")

# Direct match test
author_years = {str(row.author): {"first": int(str(row.firstYear)),
                                   "last":  int(str(row.lastYear))}
                for row in results}

print("\n=== Match test for first 5 authors in Author.csv ===")
for _, row in authors_df.head(5).iterrows():
    aid = str(row["id"])
    uri_str = f"http://expert-search.org/author/{aid}"
    hit = author_years.get(uri_str)
    print(f"  author {aid} -> key '{uri_str}' -> {hit}")