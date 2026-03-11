#!/usr/bin/env python3
"""query_authors.py

Load a TTL knowledge graph (prefer `Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl`),
accept a list of author names or URIs, and print/emit structured information per author:
 - papers (id, title)
 - coauthors (name, uri)
 - organizations
 - subtopics
 - main topics (via subtopic -> mainTopic)
 - role state (emerging / collaborating / supervising + probabilities) [NEW — Phase 4]

Also emits a RAG-ready text snippet summarizing the author's context,
enriched with role information for the LLM reranker.

PHASE 4 CHANGES:
  - get_rag_texts_for_ids() now accepts an optional pre-loaded graph `g`
    to avoid reloading the KG on every query call.
  - rag_text_for_author() now appends role label and probabilities
    when role data is present in the summary.
  - author_summary() now extracts role scores from the KG if available.

Usage examples:
  python3 query_authors.py --graph Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl --authors "Alice Smith,Bob Lee"
  python3 query_authors.py --graph Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl --authors-file authors.txt --output authors_info.json
"""
import argparse
import json
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, DCTERMS
from pathlib import Path
from collections import defaultdict

EX = Namespace('http://expert-search.org/schema#')

ROLE_LABEL_MAP = {
    "emerging"     : "Emerging Expert",
    "collaborating": "Collaborating Expert",
    "supervising"  : "Supervising Expert",
}


def load_graph(path):
    g = Graph()
    g.bind('ex', EX)
    g.bind('dct', DCTERMS)
    if Path(path).exists():
        g.parse(path, format='turtle')
        print(f"Loaded graph {path} (triples={len(g)})")
    else:
        raise FileNotFoundError(path)
    return g


def find_author_uris(g, author_input):
    """
    Accepts a numeric ID, full URI string, or author name.
    Returns list of matching URIRef objects.
    """
    if isinstance(author_input, str) and (
        author_input.startswith('http://') or author_input.startswith('https://') or '/author/' in author_input.lower()
    ):
        return [URIRef(author_input)]

    target = str(author_input).strip().lower()
    found = []
    for s in g.subjects(DCTERMS.title, None):
        for lit in g.objects(s, DCTERMS.title):
            if isinstance(lit, Literal) and lit.value and lit.value.strip().lower() == target:
                found.append(s)
                break
    return found


def paper_info(g, paper_uri):
    title = None
    for t in g.objects(paper_uri, DCTERMS.title):
        title = str(t)
        break
    return {'uri': str(paper_uri), 'title': title}


def author_summary(g, author_uri):
    """
    Build a complete summary dict for an author URI.
    Now includes role state from Phase 4 KG triples if available.
    """
    res = {
        'author_uri'  : str(author_uri),
        'name'        : None,
        'orgs'        : [],
        'papers'      : [],
        'coauthors'   : [],
        'subtopics'   : [],
        'main_topics' : [],
        'role'        : None,   # ← NEW: role state from KG
    }

    # Name
    for t in g.objects(author_uri, DCTERMS.title):
        res['name'] = str(t)
        break

    # Organizations
    for org in g.objects(author_uri, EX.affiliatedWith):
        title = None
        for tt in g.objects(org, DCTERMS.title):
            title = str(tt)
            break
        res['orgs'].append({'uri': str(org), 'title': title})

    # Papers
    papers = list(g.subjects(EX.hasAuthor, author_uri))
    for p in papers:
        res['papers'].append(paper_info(g, p))

    # Subtopics from papers
    st_set = set()
    for p in papers:
        for st in g.objects(p, EX.hasSubTopic):
            if st not in st_set:
                st_set.add(st)
                st_title = None
                for tt in g.objects(st, DCTERMS.title):
                    st_title = str(tt)
                    break
                res['subtopics'].append({'uri': str(st), 'title': st_title})

                # Main topics that have this subtopic
                for main in g.subjects(EX.hasSubTopic, st):
                    if (main, RDF.type, EX.Topic) not in g:
                        continue
                    main_title = None
                    for mt in g.objects(main, DCTERMS.title):
                        main_title = str(mt)
                        break
                    mt_entry = {'uri': str(main), 'title': main_title}
                    if mt_entry not in res['main_topics']:
                        res['main_topics'].append(mt_entry)

    # Coauthors with shared paper counts
    co_counts = defaultdict(int)
    for p in papers:
        for co in g.objects(p, EX.hasAuthor):
            if co == author_uri:
                continue
            co_counts[co] += 1

    for co, cnt in co_counts.items():
        name = None
        for tt in g.objects(co, DCTERMS.title):
            name = str(tt)
            break
        res['coauthors'].append({'uri': str(co), 'name': name, 'count': cnt})

    # Subtopics directly linked to author (inferred)
    for st in g.objects(author_uri, EX.hasSubTopic):
        if str(st) not in {s['uri'] for s in res['subtopics']}:
            st_title = None
            for tt in g.objects(st, DCTERMS.title):
                st_title = str(tt)
                break
            res['subtopics'].append({'uri': str(st), 'title': st_title})

            for main in g.subjects(EX.hasSubTopic, st):
                main_title = None
                for mt in g.objects(main, DCTERMS.title):
                    main_title = str(mt)
                    break
                mt_entry = {'uri': str(main), 'title': main_title}
                if mt_entry not in res['main_topics']:
                    res['main_topics'].append(mt_entry)

    # ── NEW: Role state from Phase 4 KG triples ───────────────────────────────
    dominant_role = None
    for obj in g.objects(author_uri, EX.dominantRole):
        dominant_role = str(obj)
        break

    if dominant_role:
        prob_e, prob_c, prob_s = 0.0, 0.0, 0.0
        for obj in g.objects(author_uri, EX.probEmerging):
            prob_e = float(obj)
            break
        for obj in g.objects(author_uri, EX.probCollaborating):
            prob_c = float(obj)
            break
        for obj in g.objects(author_uri, EX.probSupervising):
            prob_s = float(obj)
            break

        res['role'] = {
            'dominant_role'     : dominant_role,
            'label'             : ROLE_LABEL_MAP.get(dominant_role, dominant_role.capitalize()),
            'prob_emerging'     : prob_e,
            'prob_collaborating': prob_c,
            'prob_supervising'  : prob_s,
        }

    return res


def rag_text_for_author(summary):
    """
    Build RAG context string from author summary.
    Now appends role label and probabilities if role data is available.
    The LLM reranker will see the role information during author validation.
    """
    parts = []

    if summary.get('name'):
        parts.append(f"Author: {summary['name']}")

    if summary.get('papers'):
        titles = [p.get('title') or p.get('uri') for p in summary['papers']]
        parts.append("Papers: " + "; ".join(titles[:10]))

    if summary.get('subtopics'):
        parts.append("Subtopics: " + ", ".join([s.get('title') or s.get('uri') for s in summary['subtopics']]))

    if summary.get('coauthors'):
        parts.append("Coauthors: " + ", ".join([
            f"{c.get('name') or c.get('uri')} ({c.get('count')})"
            for c in summary['coauthors'][:10]
        ]))

    # ── NEW: Append role state ─────────────────────────────────────────────────
    role = summary.get('role')
    if role:
        dominant = role['dominant_role']
        prob     = role[f"prob_{dominant}"]
        parts.append(
            f"Expert Role: {role['label']} (confidence={prob:.3f})\n"
            f"Role Scores: Emerging={role['prob_emerging']:.3f} | "
            f"Collaborating={role['prob_collaborating']:.3f} | "
            f"Supervising={role['prob_supervising']:.3f}"
        )

    return "\n".join(parts)


def get_rag_texts_for_ids(author_ids, graph_path=None, g=None,
                           min_coauthored=2, max_paper_authors=50):
    """
    Return list of RAG text strings for the provided author IDs (keeps input order).
    If an author is not found, an empty string is returned at that position.

    PHASE 4 CHANGE: Accepts an optional pre-loaded graph `g`.
    If `g` is provided, it is used directly and the KG is NOT reloaded.
    If `g` is None, the KG is loaded from `graph_path` (original behaviour).

    Parameters:
    -----------
    author_ids    : list of int or str — numeric IDs or full URIs
    graph_path    : str  — path to TTL file (used only if g is None)
    g             : rdflib.Graph — pre-loaded graph (recommended for pipeline use)
    min_coauthored: int  — minimum shared papers to include a coauthor
    max_paper_authors: int — unused (kept for API compatibility)

    Returns:
    --------
    (list of str, dict) — RAG texts aligned with author_ids, and id->text dict
    """
    # Use pre-loaded graph if provided, otherwise load from file
    if g is None:
        if graph_path is None:
            raise ValueError("Either g (pre-loaded graph) or graph_path must be provided.")
        g = load_graph(graph_path)

    results        = []
    id_context_dict = {}

    for aid in author_ids:
        # Accept either numeric id or full URI
        if isinstance(aid, int) or (isinstance(aid, str) and str(aid).isdigit()):
            uri_str = f"http://expert-search.org/author/{aid}"
        else:
            uri_str = str(aid)

        uris = find_author_uris(g, uri_str)
        if not uris:
            results.append("")
            continue

        u       = uris[0]
        summary = author_summary(g, u)

        # Filter coauthors by minimum shared paper threshold
        if summary.get('coauthors'):
            summary['coauthors'] = [
                c for c in summary['coauthors']
                if c.get('count', 0) >= min_coauthored
            ]

        rag = rag_text_for_author(summary)
        results.append(rag)
        id_context_dict[aid] = rag

    return results, id_context_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', default='Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl',
                        help='TTL file (Upgraded7 recommended)')
    parser.add_argument('--authors', help='Comma-separated author names or URIs')
    parser.add_argument('--authors-file', help='File with one author name/URI per line')
    parser.add_argument('--output', help='Output JSON file (if omitted, print to stdout)')
    parser.add_argument('--rag-dir', help='Optional directory to write RAG text files per author')
    parser.add_argument('--min-coauthored', type=int, default=2,
                        help='Minimum number of shared papers to include a coauthor (default 2)')
    args = parser.parse_args()

    g = load_graph(args.graph)

    inputs = []
    if args.authors:
        inputs += [a.strip() for a in args.authors.split(',') if a.strip()]
    if args.authors_file:
        p = Path(args.authors_file)
        if p.exists():
            inputs += [l.strip() for l in p.read_text().splitlines() if l.strip()]

    if not inputs:
        print('No authors provided; use --authors or --authors-file')
        return

    results = []
    for inp in inputs:
        uris = find_author_uris(g, inp)
        if not uris:
            print(f"No author node found for '{inp}'")
            continue
        for u in uris:
            summary = author_summary(g, u)
            if summary.get('coauthors'):
                summary['coauthors'] = [
                    c for c in summary['coauthors']
                    if c.get('count', 0) >= args.min_coauthored
                ]
            summary['rag_text'] = rag_text_for_author(summary)
            results.append(summary)

            if args.rag_dir:
                Path(args.rag_dir).mkdir(parents=True, exist_ok=True)
                safe_name = (summary.get('name') or u.split('/')[-1]).replace(' ', '_')
                Path(args.rag_dir, f"{safe_name}.txt").write_text(summary['rag_text'])

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"Wrote results to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()