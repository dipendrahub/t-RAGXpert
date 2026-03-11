"""
inference_explain.py
--------------------
Runs ONE pipeline (baseline or proposed) on a single query.
For the top 3 experts, generates two explanations per author:
  1. relevance_explanation  — why the author is topically relevant (papers + topics)
  2. temporal_role_explanation — career trajectory, role label, and role probabilities

Everything is saved in a single combined JSON file.

Usage:
    python inference_explain.py --pipeline baseline --query "gibbs sampling"
    python inference_explain.py --pipeline proposed --query "gibbs sampling"

Output (saved to inference_results/):
    combined_explanation_{pipeline}_{query}.json
        ├── query
        ├── expanded_query
        ├── pipeline
        ├── retrieved_papers          <- top-3 paper titles + scores
        ├── decision_trace            <- how candidates were ranked
        └── top3_authors
            ├── author_1
            │   ├── kg_metadata
            │   ├── matched_papers
            │   ├── relevance_explanation
            │   └── temporal_role_explanation
            ├── author_2  ...
            └── author_3  ...
"""

import argparse
import json
import os
import warnings
import logging
from ast import literal_eval
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from rdflib import Graph, Namespace, URIRef
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("rdflib").setLevel(logging.ERROR)

import filter_data
import query_authors
from Utils import GetDocuments

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs("inference_results", exist_ok=True)

# ── Argument parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pipeline",
    choices=["baseline", "proposed"],
    default="proposed",
    help="Which pipeline to run: 'baseline' (citation-ranked) or 'proposed' (role-diverse).",
)
parser.add_argument(
    "--query",
    type=str,
    default="gibbs sampling",
    help="The query to run the pipeline on.",
)
args     = parser.parse_args()
PIPELINE = args.pipeline
QUERY    = args.query
SLUG     = QUERY.replace(" ", "_")

print(f"\n{'='*60}")
print(f"  Pipeline : {PIPELINE.upper()}")
print(f"  Query    : {QUERY}")
print(f"{'='*60}\n")

# ── Constants ─────────────────────────────────────────────────────────────────
TOP_N        = 3
KG_UPGRADED7 = "Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"
KG_ORIGINAL  = "/Utilisateurs/dsharm01/repo/KG-Method/knowledge_graph_inferred_full_dataset.ttl"

# ── Load models ───────────────────────────────────────────────────────────────
print("[1/6] Loading models...")
qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

config   = GetDocuments.read_json_file("llama_config.json")
llm_pipe = pipeline(
    "text-generation",
    model=config["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=config["token"],
)
print(f"      LLaMA running on: {llm_pipe.model.hf_device_map}")

reranker_model_name = "Qwen/Qwen3-Reranker-4B"
reranker_tokenizer  = AutoTokenizer.from_pretrained(reranker_model_name, padding_side="left")
reranker_model      = AutoModelForCausalLM.from_pretrained(
    reranker_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker_model = reranker_model.to(device).eval()

# ── Load data ─────────────────────────────────────────────────────────────────
print("[2/6] Loading data...")
papers_df  = pd.read_csv("~/repo/Expert-Finding/papers_df_with_topics_qwen_separate_sentences.csv")
authors_df = pd.read_csv("~/repo/Expert-Finding/papers_and_authors/authors.csv")

# ── Load KG ───────────────────────────────────────────────────────────────────
# Upgraded7 is the single source of truth for all role scores and career metrics.
print("[3/6] Loading knowledge graph...")
kg_with_roles = Graph()
kg_with_roles.parse(KG_UPGRADED7, format="turtle")
print(f"      Upgraded7 KG loaded: {len(kg_with_roles):,} triples")

EX = Namespace("http://expert-search.org/schema#")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def expand_query(query: str) -> str:
    prompt = (
        "Please define and clarify the search query below for better search results. "
        "Return only the definition as a single concise paragraph. "
        "Do not repeat the instructions or include the input query in your output.\n\n"
        "---INPUT---\n"
        f"{query}\n"
        "---END INPUT---\n\n"
        "OUTPUT:"
    )
    gen      = llm_pipe(prompt, max_new_tokens=25)[0]["generated_text"]
    response = gen.split("OUTPUT:")[-1].strip() if "OUTPUT:" in gen else gen.strip()
    return response.split("\n\n")[0].strip()


def get_author_metadata_from_kg(author_id: int) -> dict:
    """Pull all career metrics + role scores for one author from Upgraded7 KG."""
    uri  = URIRef(f"http://expert-search.org/author/{author_id}")
    meta = {
        "author_id":          author_id,
        "total_citations":    None,
        "total_pubs":         None,
        "first_year":         None,
        "last_year":          None,
        "active_years":       None,
        "career_span":        None,
        "dominant_role":      None,
        "prob_emerging":      None,
        "prob_collaborating": None,
        "prob_supervising":   None,
    }
    for pred, obj in kg_with_roles.predicate_objects(uri):
        local = pred.split("#")[-1] if "#" in pred else pred.split("/")[-1]
        val   = str(obj)
        if   local == "totalCitations":       meta["total_citations"]    = int(val)
        elif local == "totalPubs":            meta["total_pubs"]         = int(val)
        elif local == "firstPublicationYear": meta["first_year"]         = int(val)
        elif local == "lastPublicationYear":  meta["last_year"]          = int(val)
        elif local == "activeYears":          meta["active_years"]       = int(val)
        elif local == "dominantRole":         meta["dominant_role"]      = val
        elif local == "probEmerging":         meta["prob_emerging"]      = float(val)
        elif local == "probCollaborating":    meta["prob_collaborating"] = float(val)
        elif local == "probSupervising":      meta["prob_supervising"]   = float(val)

    if meta["first_year"] and meta["last_year"]:
        meta["career_span"] = meta["last_year"] - meta["first_year"]
    return meta


def get_citation_count_from_df(author_id: int) -> int:
    row = authors_df[authors_df["id"] == author_id]
    return int(row.iloc[0].get("n_citation", 0)) if not row.empty else 0


def get_role_pool_from_kg(expert_ids: list) -> dict:
    """
    Partition candidates by dominant role from Upgraded7 KG.
    Returns dict: role -> [(author_id, role_probability)], sorted by prob desc.
    """
    prob_pred_map = {
        "emerging":     "probEmerging",
        "collaborating":"probCollaborating",
        "supervising":  "probSupervising",
    }
    role_groups = defaultdict(list)
    for aid in expert_ids:
        uri      = URIRef(f"http://expert-search.org/author/{aid}")
        preds    = {
            (p.split("#")[-1] if "#" in p else p.split("/")[-1]): str(o)
            for p, o in kg_with_roles.predicate_objects(uri)
        }
        dominant = preds.get("dominantRole", "supervising")
        prob_key = prob_pred_map.get(dominant, "probSupervising")
        prob     = float(preds.get(prob_key, 0.0))
        role_groups[dominant].append((aid, prob))

    for role in role_groups:
        role_groups[role].sort(key=lambda x: x[1], reverse=True)
    return dict(role_groups)


def equal_quota_ranking(role_groups: dict, total: int) -> list:
    """
    Equal-quota: floor(N/3) from each role. Redistribute remainder from underfull groups.
    Merge order: emerging -> collaborating -> supervising.
    """
    roles     = ["emerging", "collaborating", "supervising"]
    quota     = total // 3
    buckets   = {r: [aid for aid, _ in role_groups.get(r, [])] for r in roles}
    selected  = {r: [] for r in roles}
    shortfall = 0

    for r in roles:
        take        = min(quota, len(buckets[r]))
        selected[r] = buckets[r][:take]
        shortfall  += quota - take

    if shortfall > 0:
        for r in roles:
            already = len(selected[r])
            give    = min(len(buckets[r]) - already, shortfall)
            if give > 0:
                selected[r] += buckets[r][already: already + give]
                shortfall   -= give
            if shortfall == 0:
                break

    ranked = []
    for r in roles:
        ranked.extend(selected[r])
    return ranked


def get_paper_titles_for_author(author_id: int, relevant_docs_df: pd.DataFrame) -> list:
    """Return titles of retrieved papers that this author co-authored."""
    titles = []
    for row in relevant_docs_df.itertuples():
        try:
            for a in literal_eval(row.authors):
                if literal_eval(a["id"]) == author_id:
                    titles.append(str(row.title))
                    break
        except Exception:
            continue
    return titles


# =============================================================================
# EXPLANATION PROMPTS — two separate prompts per author
# =============================================================================

def format_relevance_prompt(query: str, author_context: str, paper_titles: list) -> str:
    """
    Prompt 1: Topical relevance.
    Focuses on WHY the author's topics and papers match the query.
    """
    titles_str = (
        "\n".join(f"  - {t}" for t in paper_titles)
        if paper_titles else "  (none found in retrieved set)"
    )
    system = (
        "<|im_start|>system\n"
        "You are an expert assistant for Academic Expert Finding.\n"
        "Your task is to explain why an author is TOPICALLY RELEVANT to a given Query.\n"
        "Focus only on topic alignment and paper evidence. Do not discuss career stage or role.\n"
        "<|im_end|>\n"
    )
    user = (
        "<|im_start|>user\n"
        "Task: Explain why the author is relevant to the Query based on their topics and papers.\n\n"
        "Constraints:\n"
        "- Output must be valid JSON only (no surrounding text or markdown fences).\n"
        "- Only use the provided Author Information and Matched Papers.\n"
        "- Provide explicit `confidence` (0.0-1.0).\n\n"
        "Expected JSON format:\n"
        "{\n"
        "  \"author_name\": \"...\",\n"
        "  \"relevance_summary\": \"...\",\n"
        "  \"key_topics\": [\"...\"],\n"
        "  \"supporting_papers\": [\"...\"],\n"
        "  \"confidence\": 0.0\n"
        "}\n\n"
        f"Author Information:\n{author_context}\n\n"
        f"Matched Papers from Retrieved Set:\n{titles_str}\n\n"
        f"Query:\n{query}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return system + user


def format_temporal_role_prompt(query: str, author_context: str, meta: dict) -> str:
    """
    Prompt 2: Temporal and role-based explanation.
    Focuses on career stage, activity window, and role classification.
    """
    role_block = (
        f"  Dominant role      : {meta['dominant_role']}\n"
        f"  P(emerging)        : {meta['prob_emerging']}\n"
        f"  P(collaborating)   : {meta['prob_collaborating']}\n"
        f"  P(supervising)     : {meta['prob_supervising']}\n"
        f"  Career span        : {meta['career_span']} years "
        f"({meta['first_year']} - {meta['last_year']})\n"
        f"  Active years       : {meta['active_years']}\n"
        f"  Total publications : {meta['total_pubs']}\n"
        f"  Total citations    : {meta['total_citations']}"
    )
    system = (
        "<|im_start|>system\n"
        "You are an expert assistant for Academic Expert Finding.\n"
        "Your task is to explain an author's CAREER STAGE and ROLE PROFILE "
        "in the context of a given Query.\n"
        "Focus on temporal activity, career trajectory, and what their role label "
        "(emerging / collaborating / supervising) means for their relevance.\n"
        "<|im_end|>\n"
    )
    user = (
        "<|im_start|>user\n"
        "Task: Explain the author's temporal profile and role in the context of the Query.\n\n"
        "Constraints:\n"
        "- Output must be valid JSON only (no surrounding text or markdown fences).\n"
        "- Only use the provided Role & Career Metrics.\n"
        "- Provide explicit `confidence` (0.0-1.0).\n\n"
        "Expected JSON format:\n"
        "{\n"
        "  \"author_name\": \"...\",\n"
        "  \"career_summary\": \"...\",\n"
        "  \"role_interpretation\": \"...\",\n"
        "  \"temporal_activity_note\": \"...\",\n"
        "  \"confidence\": 0.0\n"
        "}\n\n"
        f"Role and Career Metrics:\n{role_block}\n\n"
        f"Author Information:\n{author_context}\n\n"
        f"Query:\n{query}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return system + user


def generate_llm_output(prompt: str) -> dict:
    """Run the reranker model in generation mode and parse JSON output."""
    inputs = reranker_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = reranker_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            eos_token_id=reranker_tokenizer.eos_token_id,
        )

    generated = reranker_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Strip markdown fences if model wraps output
    if generated.startswith("```"):
        parts     = generated.split("```")
        generated = parts[1].lstrip("json").strip() if len(parts) > 1 else generated

    try:
        return json.loads(generated)
    except json.JSONDecodeError:
        return {"raw_output": generated}


# =============================================================================
# STEP 1 — Query expansion + document retrieval
# =============================================================================
print("[4/6] Query expansion + document retrieval...")

expanded_query  = expand_query(QUERY)
print(f"  Original : {QUERY}")
print(f"  Expanded : {expanded_query}")

query_embedding = qwen_model.encode([expanded_query])[0].tolist()

similar_docs  = filter_data.retrieve_similar_documents_qwen_weighted(
    QUERY, qwen_model, query_embedding, papers_df, top_k=25, threshold=0.3
)
relevant_docs = filter_data.rerank_documents_deepseek_pairwise(
    expanded_query, similar_docs, top_k=15, batch_size=8
)
print(f"  Retrieved {len(relevant_docs)} papers after reranking.")

top3_papers = []
for i, row in relevant_docs.head(3).iterrows():
    entry = {
        "rank":  i + 1,
        "title": str(row.get("title", "N/A")),
        "score": round(float(row.get("reranker_score", 0)), 4),
    }
    top3_papers.append(entry)
    print(f"  [{i+1}] {entry['title'][:80]}  (score={entry['score']})")

# Deduplicated candidate author pool
expert_ids = []
for row in relevant_docs.itertuples():
    try:
        for expert in literal_eval(row.authors):
            expert_ids.append(literal_eval(expert["id"]))
    except Exception:
        continue

seen, unique_expert_ids = set(), []
for eid in expert_ids:
    if eid not in seen:
        seen.add(eid)
        unique_expert_ids.append(eid)

print(f"  Candidate pool: {len(unique_expert_ids)} unique authors")

# =============================================================================
# STEP 2 — Candidate ranking (pipeline branch)
# =============================================================================
print(f"\n[5/6] Candidate ranking ({PIPELINE.upper()})...")

if PIPELINE == "baseline":
    ranked_ids = filter_data.rank_experts_by_citations(unique_expert_ids, authors_df)

    decision_trace = {
        "ranking_method": "citation_count_descending",
        "explanation":    (
            "Candidates ranked by total citation count (descending). "
            "Role labels are shown for transparency but were NOT used in ranking."
        ),
        "n_candidates":     len(unique_expert_ids),
        "top10_candidates": [],
    }
    for aid in ranked_ids[:10]:
        meta  = get_author_metadata_from_kg(aid)
        cites = meta["total_citations"] or get_citation_count_from_df(aid)
        decision_trace["top10_candidates"].append({
            "author_id":          aid,
            "citations":          cites,
            "career_span_years":  meta["career_span"],
            "dominant_role":      meta["dominant_role"],
            "prob_emerging":      meta["prob_emerging"],
            "prob_collaborating": meta["prob_collaborating"],
            "prob_supervising":   meta["prob_supervising"],
        })

    print("  Top-5 by citations:")
    for e in decision_trace["top10_candidates"][:5]:
        print(f"    ID={e['author_id']}  citations={e['citations']}  "
              f"role={e['dominant_role']}  span={e['career_span_years']}yr")

else:
    role_groups = get_role_pool_from_kg(unique_expert_ids)
    pool_summary = {role: len(members) for role, members in role_groups.items()}
    ranked_ids   = equal_quota_ranking(role_groups, len(unique_expert_ids))

    decision_trace = {
        "ranking_method":  "equal_quota_role_diverse",
        "explanation":     (
            "Candidates partitioned into 3 role groups (emerging / collaborating / supervising) "
            "using dominant role from Upgraded7 KG. Each group ranked internally by role "
            "probability. Equal quota = floor(N/3) selected per group; shortfall redistributed "
            "from underfull groups. Merge order: emerging -> collaborating -> supervising."
        ),
        "n_candidates":           len(unique_expert_ids),
        "pool_role_distribution": pool_summary,
        "quota_per_role":         len(unique_expert_ids) // 3,
        "role_groups_detail": {
            role: [{"author_id": aid, "role_prob": round(p, 4)} for aid, p in members]
            for role, members in role_groups.items()
        },
        "top10_candidates": [],
    }
    for aid in ranked_ids[:10]:
        meta = get_author_metadata_from_kg(aid)
        decision_trace["top10_candidates"].append({
            "author_id":          aid,
            "dominant_role":      meta["dominant_role"],
            "prob_emerging":      meta["prob_emerging"],
            "prob_collaborating": meta["prob_collaborating"],
            "prob_supervising":   meta["prob_supervising"],
            "career_span_years":  meta["career_span"],
            "total_citations":    meta["total_citations"],
        })

    print(f"  Pool role distribution : {pool_summary}")
    print(f"  Quota per role         : {decision_trace['quota_per_role']}")
    print("  Top-5 after equal-quota:")
    for e in decision_trace["top10_candidates"][:5]:
        print(f"    ID={e['author_id']}  role={e['dominant_role']}  "
              f"span={e['career_span_years']}yr  cites={e['total_citations']}")

# RAG context + LLM author reranking
if PIPELINE == "baseline":
    author_contexts, id_context_dict = query_authors.get_rag_texts_for_ids(
        ranked_ids,
        graph_path=KG_ORIGINAL,
        min_coauthored=2,
        max_paper_authors=50,
    )
else:
    author_contexts, id_context_dict = query_authors.get_rag_texts_for_ids(
        ranked_ids,
        g=kg_with_roles,
        min_coauthored=2,
        max_paper_authors=50,
    )

final_ids = filter_data.rerank_authors_deepseek_pairwise(
    QUERY, ranked_ids, author_contexts, top_k=None
)
print(f"\n  Final top-{TOP_N} after LLM reranking: {final_ids[:TOP_N]}")

# =============================================================================
# STEP 3 — Per-author explanations (top 3)
# =============================================================================
print(f"\n[6/6] Generating explanations for top {TOP_N} authors...")

top3_authors = []

for rank, author_id in enumerate(final_ids[:TOP_N], start=1):
    print(f"\n  Author {rank}/{TOP_N}: ID={author_id}")

    ctx_index    = ranked_ids.index(author_id) if author_id in ranked_ids else 0
    author_ctx   = author_contexts[ctx_index] if ctx_index < len(author_contexts) else ""
    paper_titles = get_paper_titles_for_author(author_id, relevant_docs)
    meta         = get_author_metadata_from_kg(author_id)

    print(f"    Citations : {meta['total_citations']}  |  "
          f"Span: {meta['career_span']}yr ({meta['first_year']}-{meta['last_year']})  |  "
          f"Role: {meta['dominant_role']} "
          f"(E={meta['prob_emerging']:.2f}/C={meta['prob_collaborating']:.2f}/S={meta['prob_supervising']:.2f})")
    print(f"    Matched papers: {len(paper_titles)}")

    # -- Explanation 1: Topical relevance --
    print(f"    Generating relevance explanation...")
    rel_prompt   = format_relevance_prompt(QUERY, author_ctx, paper_titles)
    rel_explain  = generate_llm_output(rel_prompt)

    # -- Explanation 2: Temporal + role --
    print(f"    Generating temporal/role explanation...")
    tr_prompt    = format_temporal_role_prompt(QUERY, author_ctx, meta)
    tr_explain   = generate_llm_output(tr_prompt)

    if isinstance(rel_explain, dict) and "relevance_summary" in rel_explain:
        print(f"    Relevance : {rel_explain['relevance_summary'][:120]}...")
    if isinstance(tr_explain, dict) and "role_interpretation" in tr_explain:
        print(f"    Role/Temp : {tr_explain['role_interpretation'][:120]}...")

    top3_authors.append({
        "rank":        rank,
        "author_id":   author_id,
        "kg_metadata": {
            "total_citations":    meta["total_citations"],
            "total_pubs":         meta["total_pubs"],
            "career_span_years":  meta["career_span"],
            "first_year":         meta["first_year"],
            "last_year":          meta["last_year"],
            "active_years":       meta["active_years"],
            "dominant_role":      meta["dominant_role"],
            "prob_emerging":      meta["prob_emerging"],
            "prob_collaborating": meta["prob_collaborating"],
            "prob_supervising":   meta["prob_supervising"],
        },
        "matched_papers_in_retrieved_set": paper_titles,
        "relevance_explanation":      rel_explain,
        "temporal_role_explanation":  tr_explain,
    })

# =============================================================================
# SAVE COMBINED OUTPUT
# =============================================================================
combined = {
    "query":            QUERY,
    "expanded_query":   expanded_query,
    "pipeline":         PIPELINE,
    "retrieved_papers": top3_papers,
    "decision_trace":   decision_trace,
    "top3_authors":     top3_authors,
}

output_path = f"inference_results/combined_explanation_{PIPELINE}_{SLUG}.json"
with open(output_path, "w") as f:
    json.dump(combined, f, indent=4)

print(f"\n  Saved -> {output_path}")
print(f"\n{'='*60}")
print(f"  Done. Pipeline={PIPELINE.upper()}  Query='{QUERY}'")
print(f"  Top-3 authors: {[a['author_id'] for a in top3_authors]}")
print(f"{'='*60}\n")