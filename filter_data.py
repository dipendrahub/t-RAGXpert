import json
import random
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import regex as re
import pandas as pd
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
import Levenshtein
from rdflib import Graph, Namespace, URIRef


# ── Namespaces ─────────────────────────────────────────────────────────────────
EX      = Namespace("http://expert-search.org/schema#")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# Load Qwen3 Reranker (once)
reranker_model_name = "Qwen/Qwen3-Reranker-4B"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained(
    reranker_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reranker_model = reranker_model.to(device).eval()

# Token IDs for "yes" and "no"
token_yes_id = reranker_tokenizer.convert_tokens_to_ids("yes")
token_no_id = reranker_tokenizer.convert_tokens_to_ids("no")

# Prompt template parts
system_prompt = (
    "<|im_start|>system\n"
    "Judge whether the Document is relevant to the Query. "
    "The answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
)
suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_prompt(query: str, document: str) -> str:
    return f"{system_prompt}Query: {query}\nDocument: {document}{suffix_prompt}"


def rerank_documents_qwen(query: str, docs_df, top_k: int = 100, batch_size: int = 32):

    torch.cuda.empty_cache()
    scores = []

    docs = []
    for row in docs_df.itertuples(index=False, name=None):
        try:
            title = row[docs_df.columns.get_loc('title')] if 'title' in docs_df.columns else ""
            abstract = row[docs_df.columns.get_loc('abstract')] if 'abstract' in docs_df.columns else ""
        except Exception:
            title = row[0] if len(row) > 0 else ""
            abstract = row[1] if len(row) > 1 else ""

        title = title or ""
        abstract = abstract or ""
        docs.append(f"{str(title).strip()} {str(abstract).strip()}")

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        prompts = [format_prompt(query, d) for d in batch_docs]

        inputs = reranker_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            last_logits = outputs.logits[:, -1, :]

        yes_logits = last_logits[:, token_yes_id]
        no_logits = last_logits[:, token_no_id]
        pair_logits = torch.stack([no_logits, yes_logits], dim=1)
        probs = F.softmax(pair_logits, dim=1)[:, 1]

        scores.extend(probs.cpu().numpy().tolist())

    reranked_df = docs_df.copy()
    reranked_df["reranker_score"] = scores
    reranked_df = reranked_df.sort_values("reranker_score", ascending=False).reset_index(drop=True)
    return reranked_df.head(top_k)


def extract_float_from_text(text):
    match = re.search(r"\b(100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)\b", text)
    if match:
        return float(match.group(1))
    return None


def score_docs_with_llm(prompt, documents, pipe, threshold=70):
    relevant_docs = []

    for doc in tqdm(documents, desc="Scoring docs with LLM"):
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")

        message = [
            {"role": "system", "content": "You are an academic research assistant."},
            {"role": "user", "content": f"User Query: {prompt}\n\nDocument Abstract: {abstract}\n\nTask: On a scale of 1 to 100, how relevant is this document to the user query? Respond only in digits, do not write sentence."}
        ]

        response = pipe(message, max_new_tokens=10)

        generated_text = response[0].get("generated_text", [])
        if isinstance(generated_text, list) and len(generated_text) > 0:
            last_message = generated_text[-1]
            if last_message.get("role") == "assistant":
                raw_output = last_message.get("content", "").strip()
            else:
                raise ValueError("Assistant response missing in generated text.")
        else:
            raise ValueError("Unexpected response format.")

        score = int(extract_float_from_text(raw_output))

        if score >= threshold:
            relevant_docs.append(doc)

    print(f"Filtered {len(relevant_docs)} relevant documents out of {len(documents)}")
    return relevant_docs


def pairwise_compare(pipe, query, doc1, doc2):
    doc1_abstract = doc1['abstract']
    doc2_abstract = doc2['abstract']
    message = [
        {"role": "system", "content": "You are an academic research assistant."},
        {"role": "user", "content": f"Query: {query}\n\nDocument1: {doc1_abstract}\n\nDocument2: {doc2_abstract}\n\nTask: Which document is more relevant to the query? ONLY output '1' if Document 1 is more relevant, or '2' if Document 2 is more relevant. Output only a single character ('1' or '2') and nothing else. Do not explain your answer."}
    ]

    response = pipe(message, max_new_tokens=5)

    if isinstance(response, list) and len(response) > 0:
        generated_text = response[0].get("generated_text", "1")
        if isinstance(generated_text, list) and len(generated_text) > 0:
            output = generated_text[-1].get("content", "1").strip()
        else:
            output = str(generated_text).strip()
    else:
        output = "1"
        print("Default fallback activated in LLM similarity ranking step")
    return 1 if output == "1" else 2


def merge_sort_with_llm(pipe, query, docs):
    if len(docs) <= 1:
        return docs

    mid = len(docs) // 2
    left = merge_sort_with_llm(pipe, query, docs[:mid])
    right = merge_sort_with_llm(pipe, query, docs[mid:])

    return merge(left, right, pipe, query)


def merge(left, right, pipe, query):
    result = []
    while left and right:
        if pairwise_compare(pipe, query, left[0], right[0]) == 1:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))

    result.extend(left or right)
    return result


def rerank_docs_with_llm(query, documents, pipe, batch_size=50):
    if isinstance(documents, pd.DataFrame):
        is_dataframe = True
        docs_list = documents.to_dict(orient="records")
    elif isinstance(documents, list):
        is_dataframe = False
        docs_list = documents
    else:
        raise TypeError("Input must be a pandas DataFrame or a list of dicts.")

    ranked_batches = []
    for i in tqdm(range(0, len(docs_list), batch_size), desc="Processing batches for documents re-ranking with llm"):
        batch = docs_list[i:i + batch_size]
        ranked_batch = merge_sort_with_llm(pipe, query, batch)
        ranked_batches.append(ranked_batch)

    all_ranked = sum(ranked_batches, [])
    final_ranked_docs = merge_sort_with_llm(pipe, query, all_ranked)

    print(f"Re-ranked {len(final_ranked_docs)} documents.")

    return pd.DataFrame(final_ranked_docs) if is_dataframe else final_ranked_docs


def contains_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    return any(word in stop_words for word in words)


def filter_relevant_docs_with_cosine_sim(query_embedding, doc_embeddings, documents, threshold, dataset):
    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array([np.array(emb) for emb in doc_embeddings])
    similarities = cosine_similarity(query_embedding, float(doc_embeddings))[0]

    if dataset == 'people_list':
        relevant_docs = [doc for doc, sim in zip(documents, similarities) if sim >= threshold]
    elif dataset == 'distributed':
        similarities = np.array(similarities)
        relevant_docs = documents[similarities >= threshold]

    print(f"Filtered {len(relevant_docs)} relevant documents out of {len(documents)}")
    return relevant_docs


def filter_cosine_and_select_top_k_docs(query_embedding, doc_embeddings, documents_df, threshold, top_k):
    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array([np.array(emb) for emb in doc_embeddings])

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    mask = similarities >= threshold
    filtered_docs = documents_df[mask].copy()
    filtered_docs["cosine_sim"] = similarities[mask]

    top_k_docs = filtered_docs.sort_values(by="cosine_sim", ascending=False).head(top_k)

    print(f"Filtered {len(filtered_docs)} docs above threshold {threshold}")
    print(f"Returning top {len(top_k_docs)} docs sorted by cosine similarity")

    return top_k_docs


def combine_filtered_documents_with_labels(cosine_docs, llm_docs):
    doc_dict = {}

    for doc in cosine_docs:
        title = doc.get("title", "").strip()
        if title not in doc_dict:
            doc["selected_by"] = ["cosine"]
            doc_dict[title] = doc
        else:
            doc_dict[title]["selected_by"].append("cosine")

    for doc in llm_docs:
        title = doc.get("title", "").strip()
        if title not in doc_dict:
            doc["selected_by"] = ["llm"]
            doc_dict[title] = doc
        else:
            if "selected_by" not in doc_dict[title]:
                doc_dict[title]["selected_by"] = []
            if "llm" not in doc_dict[title]["selected_by"]:
                doc_dict[title]["selected_by"].append("llm")

    combined_docs = list(doc_dict.values())
    print(f"Combined relevant documents: {len(combined_docs)}")
    return combined_docs


def rerank_docs_by_citations(documents):
    if isinstance(documents, pd.DataFrame):
        if 'n_citation' not in documents.columns:
            raise ValueError("'n_citation' column not found in the DataFrame.")
        documents_sorted = documents.sort_values(by="n_citation", ascending=False).reset_index(drop=True)
        return documents_sorted

    elif isinstance(documents, list):
        if not all('n_citation' in doc for doc in documents):
            raise ValueError("'n_citation' field missing in one or more documents.")
        documents_sorted = sorted(documents, key=lambda x: x['n_citation'], reverse=True)
        return documents_sorted

    else:
        raise TypeError("Input must be a pandas DataFrame or a list of dicts.")


def rank_experts_by_citations(expert_ids, authors_df):
    """
    Original citation-based ranking — kept as BASELINE for comparison.
    """
    authors_df = authors_df[authors_df['tags'].notna()]
    filtered_df = authors_df[authors_df['id'].isin(expert_ids)]
    ranked_df = filtered_df.sort_values(by='n_citation', ascending=False)
    ranked_experts = ranked_df['id'].tolist()
    return ranked_experts


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Role-Diversity Ranking (NEW)
# ══════════════════════════════════════════════════════════════════════════════

_ROLE_QUERY = """
PREFIX ex: <http://expert-search.org/schema#>
SELECT ?author ?probE ?probC ?probS ?role
WHERE {{
    VALUES ?author {{ {uri_list} }}
    ?author ex:probEmerging      ?probE ;
            ex:probCollaborating ?probC ;
            ex:probSupervising   ?probS ;
            ex:dominantRole      ?role .
}}
"""

def _fetch_role_scores_from_kg(g, author_uris):
    """
    Query KG via SPARQL for role scores of a list of author URIs.
    Returns dict: uri_str -> {prob_emerging, prob_collaborating,
                               prob_supervising, dominant_role}
    """
    if not author_uris:
        return {}

    uri_list = " ".join(f"<{u}>" for u in author_uris)
    query    = _ROLE_QUERY.format(uri_list=uri_list)
    results  = g.query(query)

    scores = {}
    for row in results:
        scores[str(row.author)] = {
            "prob_emerging"     : float(row.probE),
            "prob_collaborating": float(row.probC),
            "prob_supervising"  : float(row.probS),
            "dominant_role"     : str(row.role),
        }
    return scores


def rank_experts_by_role_diversity(expert_ids, authors_df, g):
    """
    Role-diverse replacement for rank_experts_by_citations().

    Queries role scores from the KG (Upgraded7) via SPARQL and applies
    equal-quota candidate selection across emerging / collaborating / supervising.

    Algorithm:
      1. Deduplicate expert_ids
      2. Filter to authors with tags (same as citation baseline)
      3. Query KG for role scores
      4. Split into 3 groups by dominant_role
      5. Within each group rank by role probability (descending)
      6. Take floor(N/3) from each group (equal quota)
      7. Redistribute remainder slots if a group is underfull
      8. Return merged list: emerging → collaborating → supervising

    Parameters:
    -----------
    expert_ids  : list of int — author IDs from candidate pool (may have duplicates)
    authors_df  : DataFrame with columns 'id', 'n_citation', 'tags'
    g           : rdflib.Graph — Upgraded7 KG with role score triples

    Returns:
    --------
    list of int — deduplicated, role-diverse ranked author IDs
    """

    # ── 1. Deduplicate ────────────────────────────────────────────────────────
    seen       = set()
    unique_ids = []
    for eid in expert_ids:
        if eid not in seen:
            seen.add(eid)
            unique_ids.append(eid)

    # ── 2. Filter to authors with tags (matches citation baseline) ────────────
    authors_df_filtered = authors_df[authors_df["tags"].notna()]
    valid_ids  = set(authors_df_filtered["id"].tolist())
    unique_ids = [eid for eid in unique_ids if eid in valid_ids]

    if not unique_ids:
        return []

    # ── 3. Query KG for role scores ───────────────────────────────────────────
    author_uris = [f"http://expert-search.org/author/{eid}" for eid in unique_ids]
    role_scores = _fetch_role_scores_from_kg(g, author_uris)

    # ── 4. Split into role groups ─────────────────────────────────────────────
    groups = defaultdict(list)  # role -> [(author_id, role_prob)]

    for eid in unique_ids:
        uri    = f"http://expert-search.org/author/{eid}"
        scores = role_scores.get(uri)

        if scores is None:
            # Author not in KG role scores → assign to collaborating as neutral
            groups["collaborating"].append((eid, 0.333))
            continue

        role = scores["dominant_role"]
        prob = scores[f"prob_{role}"]
        groups[role].append((eid, prob))

    # ── 5. Rank within each group by role probability (descending) ────────────
    for role in groups:
        groups[role].sort(key=lambda x: x[1], reverse=True)

    # ── 6. Equal quota ────────────────────────────────────────────────────────
    N          = len(unique_ids)
    base_quota = N // 3
    remainder  = N % 3   # 0, 1, or 2 extra slots

    role_order = ["emerging", "collaborating", "supervising"]
    quotas     = {role: base_quota for role in role_order}

    # Give remainder slots to roles with most candidates first
    sorted_by_size = sorted(role_order, key=lambda r: len(groups[r]), reverse=True)
    for i in range(remainder):
        quotas[sorted_by_size[i]] += 1

    # ── 7. Fill quotas — redistribute from underfull groups ──────────────────
    selected  = {role: [] for role in role_order}
    shortfall = {}

    # First pass: fill up to quota
    for role in role_order:
        available = groups[role]
        take      = min(quotas[role], len(available))
        selected[role] = [eid for eid, _ in available[:take]]
        gap = quotas[role] - take
        if gap > 0:
            shortfall[role] = gap

    # Second pass: collect unused candidates and fill shortfalls
    if shortfall:
        extras = []
        for role in role_order:
            if role not in shortfall:
                used  = len(selected[role])
                avail = groups[role]
                extras.extend([eid for eid, _ in avail[used:]])

        for role in shortfall:
            needed = shortfall[role]
            fill   = extras[:needed]
            selected[role].extend(fill)
            extras  = extras[needed:]

    # ── 8. Merge: emerging → collaborating → supervising ─────────────────────
    final_ids = []
    added     = set()
    for role in role_order:
        for eid in selected[role]:
            if eid not in added:
                final_ids.append(eid)
                added.add(eid)

    # Print role distribution for transparency
    role_dist = defaultdict(int)
    for eid in final_ids:
        uri = f"http://expert-search.org/author/{eid}"
        s   = role_scores.get(uri)
        if s:
            role_dist[s["dominant_role"]] += 1
        else:
            role_dist["unknown"] += 1

    print(f"  Role-diverse candidate pool ({len(final_ids)} authors): "
          f"emerging={role_dist.get('emerging',0)}, "
          f"collaborating={role_dist.get('collaborating',0)}, "
          f"supervising={role_dist.get('supervising',0)}")

    return final_ids


def enrich_rag_text_with_role(rag_text, author_id, g):
    """
    Appends role label and probability scores to an existing RAG context string.
    Called after get_rag_texts_for_ids() so the LLM reranker is role-aware.

    Example appended text:
        Expert Role: Emerging Expert (confidence=0.998)
        Role Scores: Emerging=0.998 | Collaborating=0.001 | Supervising=0.001

    Parameters:
    -----------
    rag_text  : str  — existing RAG context from query_authors.rag_text_for_author()
    author_id : int  — numeric author ID
    g         : rdflib.Graph — Upgraded7 KG

    Returns:
    --------
    str — enriched RAG context
    """
    uri    = f"http://expert-search.org/author/{author_id}"
    scores = _fetch_role_scores_from_kg(g, [uri])
    s      = scores.get(uri)

    if s is None:
        return rag_text   # no role data → return unchanged

    role_label_map = {
        "emerging"     : "Emerging Expert",
        "collaborating": "Collaborating Expert",
        "supervising"  : "Supervising Expert",
    }

    label = role_label_map.get(s["dominant_role"], s["dominant_role"].capitalize())
    prob  = s[f"prob_{s['dominant_role']}"]

    role_line = (
        f"Expert Role: {label} (confidence={prob:.3f})\n"
        f"Role Scores: Emerging={s['prob_emerging']:.3f} | "
        f"Collaborating={s['prob_collaborating']:.3f} | "
        f"Supervising={s['prob_supervising']:.3f}"
    )

    return rag_text + "\n" + role_line


def compute_role_diversity_score(ranked_author_ids, g, top_k=10):
    """
    Computes normalized Shannon entropy of role distribution in top-K ranked list.
    Used as secondary evaluation metric to quantify seniority bias correction.

    Returns:
        0.0 → all authors have same role (maximum bias)
        1.0 → perfectly equal distribution across 3 roles

    Parameters:
    -----------
    ranked_author_ids : list of int
    g                 : rdflib.Graph
    top_k             : int

    Returns:
    --------
    float — normalized Shannon entropy [0, 1]
    """
    ids_to_check = ranked_author_ids[:top_k]
    if not ids_to_check:
        return 0.0

    uris        = [f"http://expert-search.org/author/{eid}" for eid in ids_to_check]
    role_scores = _fetch_role_scores_from_kg(g, uris)

    role_counts = defaultdict(int)
    total       = 0

    for eid in ids_to_check:
        uri = f"http://expert-search.org/author/{eid}"
        s   = role_scores.get(uri)
        if s:
            role_counts[s["dominant_role"]] += 1
            total += 1

    if total == 0:
        return 0.0

    entropy = 0.0
    for role in ["emerging", "collaborating", "supervising"]:
        p = role_counts.get(role, 0) / total
        if p > 0:
            entropy -= p * np.log(p)

    # Normalize by log(3) so max entropy = 1.0
    return round(entropy / np.log(3), 4)


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING FUNCTIONS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_similar_documents_qwen(qwen_model, query_embedding, papers_df, top_k=25, threshold=0.3):
    doc_embeddings = np.vstack(papers_df["embeddings"].apply(ast.literal_eval).values)
    query_vec = np.array(query_embedding).reshape(1, -1)

    with torch.no_grad():
        sims = qwen_model.similarity(torch.tensor(query_vec), torch.tensor(doc_embeddings))
    sims = sims.cpu().numpy().flatten()

    papers_df = papers_df.copy()
    papers_df["similarity"] = sims
    filtered = papers_df[papers_df["similarity"] >= threshold]
    top_docs = filtered.nlargest(top_k, "similarity")

    return top_docs.reset_index(drop=True)


def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0


def bm25_scores(bm25, query_tokens):
    return [bm25.score(query_tokens, i) for i in range(bm25.N)]


def Levenshtein_similarity(str1, str2):
    if not str1 or not str2:
        return 0.0
    distance = Levenshtein.distance(str1, str2)
    max_len = max(len(str1), len(str2))
    return 1.0 - (distance / max_len)


def _minmax(a, eps=1e-8):
    a = np.array(a, dtype=np.float32)
    lo = a.min()
    hi = a.max()
    if hi - lo < eps:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo + eps)


def retrieve_similar_documents_qwen_weighted(query, qwen_model, query_embedding, papers_df, top_k=25, threshold=0.3):
    candidate_multiplier = 3

    emb_values = papers_df["embeddings"].values
    emb_list = [np.array(ast.literal_eval(x), dtype=np.float32) for x in emb_values]
    n_docs = len(emb_list)

    query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    try:
        model_device = next(qwen_model.parameters()).device
    except Exception:
        model_device = torch.device("cpu")

    query_tensor = torch.from_numpy(query_vec).to(model_device)
    sims = np.empty(n_docs, dtype=np.float32)
    sim_batch_size = 2048

    for start in range(0, n_docs, sim_batch_size):
        end = min(start + sim_batch_size, n_docs)
        chunk = np.stack(emb_list[start:end])
        chunk_tensor = torch.from_numpy(chunk).to(model_device)
        with torch.no_grad():
            sims_chunk = qwen_model.similarity(query_tensor, chunk_tensor)
        sims[start:end] = sims_chunk.cpu().numpy().flatten()

    n_docs = len(sims)
    if n_docs == 0:
        return papers_df.copy().assign(weighted_similarity=np.array([]))

    candidate_k = min(n_docs, max(top_k, top_k * candidate_multiplier))
    if candidate_k < n_docs:
        top_candidate_idx = np.argpartition(-sims, candidate_k - 1)[:candidate_k]
        top_candidate_idx = top_candidate_idx[np.argsort(-sims[top_candidate_idx])]
    else:
        top_candidate_idx = np.argsort(-sims)

    jaccard_sims = np.zeros(n_docs, dtype=np.float32)
    doc_texts = (papers_df.get("title", "").fillna("").astype(str).str.strip() + " " +
                 papers_df.get("abstract", "").fillna("").astype(str).str.strip()).values

    for idx in top_candidate_idx:
        jaccard_sims[idx] = jaccard_similarity(query, doc_texts[idx])

    levenshtein_sims = np.zeros(n_docs, dtype=np.float32)
    for idx in top_candidate_idx:
        levenshtein_sims[idx] = Levenshtein_similarity(query, doc_texts[idx])

    alpha = 0.5
    beta = 0.35
    gamma = 0.15
    combined_sims = (alpha * _minmax(sims)) + (beta * _minmax(jaccard_sims)) + (gamma * _minmax(levenshtein_sims))

    papers_df = papers_df.copy()
    papers_df["weighted_similarity"] = combined_sims

    filtered = papers_df[papers_df["weighted_similarity"] >= threshold]
    if filtered.empty:
        return papers_df.nlargest(top_k, "weighted_similarity")

    return filtered.nlargest(top_k, "weighted_similarity").reset_index(drop=True)


def format_author_prompt(query, author_context):
    system_prompt = (
        "<|im_start|>system\n"
        "You are a helpful AI that determines whether an author is relevant to a given Query. "
        "An author is considered relevant if their Papers or Topics fall within the scope of the Query — even if the Papers/Topics is broad, general or interdisciplinary. "
        "Only answer with \"yes\" or \"no\". Do not provide explanations.<|im_end|>\n<|im_start|>user\n"
    )

    suffix_prompt = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    return (
        f"{system_prompt}"
        f"Query: {query}\n\n"
        f"Author Information:\n"
        f"{author_context}\n"
        f"{suffix_prompt}"
    )


def qwen3_binary_relevance(prompt):
    inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = reranker_model(**inputs)
        logits = outputs.logits[0, -1]

    yes_logit = logits[token_yes_id]
    no_logit = logits[token_no_id]
    probs = F.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
    return probs[1].item() > 0.3


def rerank_authors_qwen(query, author_ids, author_contexts):
    token_yes_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    token_no_id = reranker_tokenizer.convert_tokens_to_ids("no")
    ranked_authors = []

    i = 0
    for author_id in author_ids:
        prompt = format_author_prompt(query, author_contexts[i])

        inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            logits = outputs.logits[0, -1]

        yes_logit = logits[token_yes_id]
        no_logit = logits[token_no_id]
        probs = torch.softmax(torch.tensor([no_logit, yes_logit], device=device), dim=0)
        yes_prob = probs[1].item()

        ranked_authors.append((author_id, yes_prob))
        i += 1

    ranked_authors.sort(key=lambda x: x[1], reverse=True)
    return [author_id for author_id, _ in ranked_authors]


def format_explanation_prompt(query, author_context):
    system_prompt = (
        "<|im_start|>system\n"
        "You are an expert assistant for Academic Expert Finding.\n\n"
        "Your task is to EXPLAIN If and WHY an author is relevant to a given Query.\n"
        "<|im_end|>\n"
    )

    user_prompt = (
        "<|im_start|>user\n"
        "Task: Given the Query and the Author Information, produce a short explanation and structured evidence (for each author) showing why the author is relevant.\n\n"
        "Constraints:\n"
        "- Output must be valid JSON only (no surrounding text).\n"
        "- Do not invent information; only use the provided Author Information.\n"
        "- Provide explicit `confidence` (0–1) based only on available evidence.\n\n"
        "Author Information:\n"
        f"{author_context}\n\n"
        "Query:\n"
        f"{query}\n"
        "<|im_end|>\n"
    )
    assistant_prompt = "<|im_start|>assistant\n"

    example = (
        "<|im_start|>system\n"
        "EXAMPLE\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "{\"Author Information:\n"
        "Author Information:\n"
        "Author ID: A12345\n"
        "Name: Dr. Jane Smith\n"
        "Papers: ['Deep Learning for Natural Language Processing', 'Advances in Computer Vision']\n"
        "Topics: ['Machine Learning', 'Artificial Intelligence']\n\n"
        "Query:\n"
        "Natural Language Processing\n"
        "Reasoning: Dr. Jane Smith has authored multiple papers on Natural Language Processing, including 'Deep Learning for Natural Language Processing'. Her expertise in Machine Learning and Artificial Intelligence further supports her relevance to the query. I am 90% confident in this assessment.\n"
        "}<|im_end|>\n"
    )

    return system_prompt + user_prompt + assistant_prompt


def generate_author_explanation(query, author_context, relevant_ranked_documents):
    prompt = format_explanation_prompt(query=query, author_context=author_context)

    inputs = reranker_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = reranker_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
            eos_token_id=reranker_tokenizer.eos_token_id
        )

    generated_text = reranker_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    try:
        explanation = json.loads(generated_text)
    except json.JSONDecodeError:
        explanation = {"raw_output": generated_text}
    return explanation


import math
from collections import Counter


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)

        self.doc_len = []
        self.avgdl = 0
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)

        self._initialize()

    def _initialize(self):
        total_len = 0
        for doc in self.corpus:
            freqs = Counter(doc)
            self.term_freqs.append(freqs)
            self.doc_len.append(len(doc))
            total_len += len(doc)

            for term in freqs:
                self.doc_freqs[term] += 1

        self.avgdl = total_len / self.N if self.N > 0 else 0

    def idf(self, term):
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens, doc_index):
        score = 0.0
        freqs = self.term_freqs[doc_index]
        dl = self.doc_len[doc_index]

        for term in query_tokens:
            if term not in freqs:
                continue

            tf = freqs[term]
            idf = self.idf(term)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)

            score += idf * (numerator / denominator)

        return score


def format_pairwise_prompt_deepseek(query, doc_a, doc_b):
    return f"""
    You are a relevance judge for expert search.

    Query:
    {query}

    Candidate A:
    {doc_a}

    Candidate B:
    {doc_b}

    Which candidate is more relevant to the query?
    Answer with only "A" or "B".
    """.strip()


def rerank_documents_deepseek_pairwise(query: str, docs_df, top_k: int = 15, batch_size: int = 8):
    torch.cuda.empty_cache()

    docs = []
    for row in docs_df.itertuples(index=False, name=None):
        try:
            title = row[docs_df.columns.get_loc("title")] if "title" in docs_df.columns else ""
            abstract = row[docs_df.columns.get_loc("abstract")] if "abstract" in docs_df.columns else ""
        except Exception:
            title = row[0] if len(row) > 0 else ""
            abstract = row[1] if len(row) > 1 else ""

        docs.append(f"{str(title).strip()} {str(abstract).strip()}")

    docs = docs[:top_k]
    indices = list(range(len(docs)))

    ranked = [indices[0]]

    for idx in indices[1:]:
        inserted = False

        for pos in range(len(ranked)):
            doc_a = docs[idx]
            doc_b = docs[ranked[pos]]

            prompt = format_pairwise_prompt_deepseek(query, doc_a, doc_b)

            inputs = reranker_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = reranker_model.generate(
                    **inputs, max_new_tokens=1, do_sample=False, temperature=0.0,
                )

            decision = reranker_tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
            ).strip()

            if decision == "A":
                ranked.insert(pos, idx)
                inserted = True
                break

        if not inserted:
            ranked.append(idx)

    reranked_df = docs_df.iloc[ranked].reset_index(drop=True)
    return reranked_df


def format_author_pairwise_prompt_deepseek(query, author_a_context, author_b_context):
    return f"""
    You are a relevance judge for expert search.

    Query:
    {query}

    Author A:
    {author_a_context}

    Author B:
    {author_b_context}

    Which author is more relevant to the query?
    Answer with only "A" or "B".
    """.strip()


def rerank_authors_deepseek_pairwise(query, author_ids, author_contexts, top_k: int = None):
    torch.cuda.empty_cache()

    assert len(author_ids) == len(author_contexts)

    if top_k is not None:
        author_ids = author_ids[:top_k]
        author_contexts = author_contexts[:top_k]

    indices = list(range(len(author_ids)))
    ranked = [indices[0]]

    for idx in indices[1:]:
        inserted = False

        for pos in range(len(ranked)):
            a_ctx = author_contexts[idx]
            b_ctx = author_contexts[ranked[pos]]

            prompt = format_author_pairwise_prompt_deepseek(query, a_ctx, b_ctx)

            inputs = reranker_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = reranker_model.generate(
                    **inputs, max_new_tokens=1, do_sample=False, temperature=0.0,
                )

            decision = reranker_tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
            ).strip()

            if decision == "A":
                ranked.insert(pos, idx)
                inserted = True
                break

        if not inserted:
            ranked.append(idx)

    return [author_ids[i] for i in ranked]