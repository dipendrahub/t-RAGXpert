#!/usr/bin/env python3
"""query_authors.py

Load a TTL knowledge graph (prefer `knowledge_graph_inferred.ttl`), accept a list
of author names or URIs, and print/emit structured information per author:
 - papers (id, title)
 - coauthors (name, uri)
 - organizations
 - subtopics
 - main topics (via subtopic -> mainTopic)
Also can emit a simple RAG-ready text snippet summarizing the author's context.

Usage examples:
  python3 query_authors.py --graph knowledge_graph_inferred.ttl --authors "Alice Smith,Bob Lee"
  python3 query_authors.py --graph knowledge_graph_inferred.ttl --authors-file authors.txt --output authors_info.json
"""
import argparse
import json
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF
from pathlib import Path

EX = Namespace('http://expert-search.org/schema#')
from rdflib.namespace import DCTERMS


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
    # If looks like a URI, return as URIRef
    from rdflib import URIRef
    if isinstance(author_input, str) and (
        author_input.startswith('http://') or author_input.startswith('https://') or '/author/' in author_input.lower()
    ):
        return [URIRef(author_input)]

    target = str(author_input).strip().lower()
    found = []
    # Search any subject that has a dcterms:title matching the author name (avoids namespace/type mismatches)
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
    res = {'author_uri': str(author_uri), 'name': None, 'orgs': [], 'papers': [], 'coauthors': [], 'subtopics': [], 'main_topics': []}
    for t in g.objects(author_uri, DCTERMS.title):
        res['name'] = str(t)
        break

    # organizations
    for org in g.objects(author_uri, EX.affiliatedWith):
        title = None
        for tt in g.objects(org, DCTERMS.title):
            title = str(tt)
            break
        res['orgs'].append({'uri': str(org), 'title': title})

    # papers
    papers = list(g.subjects(EX.hasAuthor, author_uri))
    for p in papers:
        res['papers'].append(paper_info(g, p))

    # Collect subtopics from the author's papers (paper -> hasSubTopic)
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

                # main topics that have this subtopic (only include nodes typed as Topic)
                for main in g.subjects(EX.hasSubTopic, st):
                    if (main, RDF.type, EX.Topic) not in g:
                        continue
                    main_title = None
                    for mt in g.objects(main, DCTERMS.title):
                        main_title = str(mt)
                        break
                    # avoid duplicates
                    mt_entry = {'uri': str(main), 'title': main_title}
                    if mt_entry not in res['main_topics']:
                        res['main_topics'].append(mt_entry)

    # coauthors: count how many papers each coauthor shares with the author
    from collections import defaultdict
    co_counts = defaultdict(int)
    for p in papers:
        for co in g.objects(p, EX.hasAuthor):
            if co == author_uri:
                continue
            co_counts[co] += 1

    # default min_coauthored will be supplied by caller in the summary flow
    # here we just build a preliminary coauthor list with counts (the caller will filter)
    for co, cnt in co_counts.items():
        name = None
        for tt in g.objects(co, DCTERMS.title):
            name = str(tt)
            break
        res['coauthors'].append({'uri': str(co), 'name': name, 'count': cnt})

    # subtopics directly linked to author (inferred) - add if not already present
    for st in g.objects(author_uri, EX.hasSubTopic):
        if str(st) not in {s['uri'] for s in res['subtopics']}:
            st_title = None
            for tt in g.objects(st, DCTERMS.title):
                st_title = str(tt)
                break
            res['subtopics'].append({'uri': str(st), 'title': st_title})

            # main topics that have this subtopic
            for main in g.subjects(EX.hasSubTopic, st):
                main_title = None
                for mt in g.objects(main, DCTERMS.title):
                    main_title = str(mt)
                    break
                mt_entry = {'uri': str(main), 'title': main_title}
                if mt_entry not in res['main_topics']:
                    res['main_topics'].append(mt_entry)

    return res


def rag_text_for_author(summary):
    parts = []
    if summary.get('name'):
        parts.append(f"Author: {summary['name']}")
    # if summary.get('orgs'):
    #     parts.append("Orgs: " + ", ".join([o.get('title') or o.get('uri') for o in summary['orgs']]))
    if summary.get('papers'):
        titles = [p.get('title') or p.get('uri') for p in summary['papers']]
        parts.append("Papers: " + "; ".join(titles[:10]))
    if summary.get('subtopics'):
        parts.append("Subtopics: " + ", ".join([s.get('title') or s.get('uri') for s in summary['subtopics']]))
    # if summary.get('main_topics'):
    #     parts.append("Main topics: " + ", ".join([m.get('title') or m.get('uri') for m in summary['main_topics']]))
    if summary.get('coauthors'):
        # coauthors now include counts; show only those that meet the caller threshold filter
        parts.append("Coauthors: " + ", ".join([f"{c.get('name') or c.get('uri')} ({c.get('count')})" for c in summary['coauthors'][:10]]))
    return "\n".join(parts)


def get_rag_texts_for_ids(author_ids, graph_path='knowledge_graph_inferred.ttl', min_coauthored=2, max_paper_authors=50):
    """Return list of RAG text strings for the provided author ids (keeps input order).

    `author_ids` may be numeric ids or full URIs. If an author is not found, an empty
    string is returned at that position.
    """
    g = load_graph(graph_path)
    results = []
    id_context_dict = {}
    for aid in author_ids:
        # accept either numeric id or full URI
        if isinstance(aid, int) or (isinstance(aid, str) and aid.isdigit()):
            uri_str = f"http://expert-search.org/author/{aid}"
        else:
            uri_str = str(aid)

        uris = find_author_uris(g, uri_str)
        if not uris:
            results.append("")
            continue
        # prefer first match
        u = uris[0]
        summary = author_summary(g, u)
        # filter coauthors
        if summary.get('coauthors'):
            filtered = [c for c in summary['coauthors'] if c.get('count', 0) >= min_coauthored]
            summary['coauthors'] = filtered
        rag = rag_text_for_author(summary)
        results.append(rag)
        id_context_dict[aid] = rag
    return results, id_context_dict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', default='knowledge_graph_inferred.ttl', help='TTL file (inferred preferred)')
    parser.add_argument('--authors', help='Comma-separated author names or URIs')
    parser.add_argument('--authors-file', help='File with one author name/URI per line')
    parser.add_argument('--output', help='Output JSON file (if omitted, print to stdout)')
    parser.add_argument('--rag-dir', help='Optional directory to write RAG text files per author')
    parser.add_argument('--min-coauthored', type=int, default=2, help='Minimum number of shared papers to include a coauthor (default 2)')
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
            # filter coauthors by threshold
            if summary.get('coauthors'):
                filtered = [c for c in summary['coauthors'] if c.get('count', 0) >= args.min_coauthored]
                summary['coauthors'] = filtered
            # add RAG text
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
    import argparse
    main()
