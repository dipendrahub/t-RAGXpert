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
# from transformers import Ministral3ForCausalLM



# Load Qwen3 Reranker (once)
reranker_model_name = "Qwen/Qwen3-Reranker-4B"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name, padding_side='left')
# reranker_model = AutoModelForCausalLM.from_pretrained(reranker_model_name, torch_dtype=torch.float16)
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
    """
    Reranks a DataFrame of documents (with 'title' and 'abstract' columns) based on relevance to the query.
    This version batches tokenization and model calls to reduce per-document overhead while preserving
    the original scoring logic (probability of "yes").

    Parameters:
    - query: the user query string
    - docs_df: DataFrame with 'title' and 'abstract' columns
    - top_k: number of top results to return
    - batch_size: how many documents to process per model call (tune for memory)
    """
    scores = []

    # Build document strings once (use itertuples for speed)
    docs = []
    for row in docs_df.itertuples(index=False, name=None):
        # when name=None, row is a plain tuple in column order; handle safely
        try:
            # assume columns are at least ['title', 'abstract'] or similar order
            title = row[docs_df.columns.get_loc('title')] if 'title' in docs_df.columns else ""
            abstract = row[docs_df.columns.get_loc('abstract')] if 'abstract' in docs_df.columns else ""
        except Exception:
            # fallback: access by index positions (title first, abstract second)
            title = row[0] if len(row) > 0 else ""
            abstract = row[1] if len(row) > 1 else ""

        title = title or ""
        abstract = abstract or ""
        docs.append(f"{str(title).strip()} {str(abstract).strip()}")

    # Process in batches to reduce overhead
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        prompts = [format_prompt(query, d) for d in batch_docs]

        # Tokenize batch once with padding and move to device
        inputs = reranker_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            # logits shape: (batch, seq_len, vocab)
            last_logits = outputs.logits[:, -1, :]  # (batch, vocab)

        # Extract yes/no logits and compute probability of 'yes' per example
        yes_logits = last_logits[:, token_yes_id]
        no_logits = last_logits[:, token_no_id]
        pair_logits = torch.stack([no_logits, yes_logits], dim=1)  # (batch, 2)
        probs = F.softmax(pair_logits, dim=1)[:, 1]  # probability of 'yes'

        scores.extend(probs.cpu().numpy().tolist())

    # Add and sort scores (preserve original output type)
    reranked_df = docs_df.copy()
    reranked_df["reranker_score"] = scores
    reranked_df = reranked_df.sort_values("reranker_score", ascending=False).reset_index(drop=True)
    return reranked_df.head(top_k)


def extract_float_from_text(text):
    """Extract the first float or integer from a string, between 1 and 100."""
    match = re.search(r"\b(100(?:\.0+)?|[1-9]?\d(?:\.\d+)?)\b", text)
    if match:
        return float(match.group(1))
    return None


def score_docs_with_llm(prompt, documents, pipe, threshold=70):


    """
    Uses Llama (via pipe) to score document relevance against the user query.
    Returns only those documents with relevance score >= threshold.
    """
    relevant_docs = []

    for doc in tqdm(documents, desc="Scoring docs with LLM"):
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")

        message = [
            {"role": "system", "content": "You are an academic research assistant."},
            {"role": "user", "content": f"User Query: {prompt}\n\nDocument Abstract: {abstract}\n\nTask: On a scale of 1 to 100, how relevant is this document to the user query? Respond only in digits, do not write sentence."}
        ]

        #try:
        response = pipe(message, max_new_tokens=10)
        #print(f"Raw LLM Response: {response}")

        # Extract the assistant's response correctly
        generated_text = response[0].get("generated_text", [])
        if isinstance(generated_text, list) and len(generated_text) > 0:
            last_message = generated_text[-1]  # Extract last message
            if last_message.get("role") == "assistant":
                raw_output = last_message.get("content", "").strip()
            else:
                raise ValueError("Assistant response missing in generated text.")
        else:
            raise ValueError("Unexpected response format.")

        # Convert extracted response to a float
        score = int(extract_float_from_text(raw_output))  # Ensure integer (1–100 scale)
        #print(f"Doc: {title}\nScore: {score}\n")

        if score >= threshold:
            relevant_docs.append(doc)

        # except Exception as e:
        #     print(f"Error processing doc '{title}': {e}")

    print(f"Filtered {len(relevant_docs)} relevant documents out of {len(documents)}")
    return relevant_docs



def pairwise_compare(pipe, query, doc1, doc2):
    doc1_abstract = doc1['abstract']
    doc2_abstract = doc2['abstract']
    """Asks the LLM to compare two documents and determine which is more relevant."""
    message = [
        {"role": "system", "content": "You are an academic research assistant."},
        {"role": "user", "content": f"Query: {query}\n\nDocument1: {doc1_abstract}\n\nDocument2: {doc2_abstract}\n\nTask: Which document is more relevant to the query? ONLY output '1' if Document 1 is more relevant, or '2' if Document 2 is more relevant. Output only a single character ('1' or '2') and nothing else. Do not explain your answer."}
    ]
    
    response = pipe(message, max_new_tokens=5)
    #print(response)
    
    if isinstance(response, list) and len(response) > 0:
        generated_text = response[0].get("generated_text", "1")
        if isinstance(generated_text, list) and len(generated_text) > 0:
            output = generated_text[-1].get("content", "1").strip()
        else:
            output = str(generated_text).strip()
    else:
        output = "1"  # Default fallback in case of unexpected response format
        print("Default fallback activated in LLM similarity ranking step")
    return 1 if output == "1" else 2


def merge_sort_with_llm(pipe, query, docs):
    """Sorts documents using LLM-guided comparisons."""
    if len(docs) <= 1:
        return docs
    
    mid = len(docs) // 2
    left = merge_sort_with_llm(pipe, query, docs[:mid])
    right = merge_sort_with_llm(pipe, query, docs[mid:])
    
    return merge(left, right, pipe, query)


def merge(left, right, pipe, query):
    """Merges two sorted lists based on LLM comparisons."""
    result = []
    while left and right:
        if pairwise_compare(pipe, query, left[0], right[0]) == 1:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    
    result.extend(left or right)
    return result


def rerank_docs_with_llm(query, documents, pipe, batch_size=50):
    """
    Re-ranks documents (DataFrame or list of dicts) based on their relevance to the given query
    using batch processing and pairwise comparisons. Returns the same type as input.
    """
    # Detect input type
    if isinstance(documents, pd.DataFrame):
        is_dataframe = True
        docs_list = documents.to_dict(orient="records")
    elif isinstance(documents, list):
        is_dataframe = False
        docs_list = documents
    else:
        raise TypeError("Input must be a pandas DataFrame or a list of dicts.")

    # Create and rerank in batches
    ranked_batches = []
    for i in tqdm(range(0, len(docs_list), batch_size), desc="Processing batches for documents re-ranking with llm"):
        batch = docs_list[i:i + batch_size]
        ranked_batch = merge_sort_with_llm(pipe, query, batch)
        ranked_batches.append(ranked_batch)

    # Merge and re-rank all documents
    all_ranked = sum(ranked_batches, [])
    final_ranked_docs = merge_sort_with_llm(pipe, query, all_ranked)

    print(f"Re-ranked {len(final_ranked_docs)} documents.")

    # Return in the same format as input
    return pd.DataFrame(final_ranked_docs) if is_dataframe else final_ranked_docs


def contains_stopwords(text):
    stop_words = set(stopwords.words('english'))  # Load stop words set
    words = text.lower().split()  # Convert to lowercase and split into words

    return any(word in stop_words for word in words)  # Check if any word is a stop word


def filter_relevant_docs_with_cosine_sim(query_embedding, doc_embeddings, documents, threshold, dataset):
    """
    Compute cosine similarity between the query embedding and each document embedding.
    Return a filtered list of documents that meet the similarity threshold.
    """
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Ensure doc_embeddings is a properly shaped NumPy array
    doc_embeddings = np.array([np.array(emb) for emb in doc_embeddings])
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, float(doc_embeddings))[0]
    #print("similarities", similarities)
    
    # Filter documents based on threshold
    if dataset == 'people_list':
        relevant_docs = [doc for doc, sim in zip(documents, similarities) if sim >= threshold]
    elif dataset == 'distributed':
        similarities = np.array(similarities)
        relevant_docs = documents[similarities >= threshold]
    
    print(f"Filtered {len(relevant_docs)} relevant documents out of {len(documents)}")
    return relevant_docs


def filter_cosine_and_select_top_k_docs(query_embedding, doc_embeddings, documents_df, threshold, top_k):
    """
    1. Compute cosine similarity between query and each document embedding.
    2. Keep docs with sim >= threshold.
    3. Return top_k docs sorted by cosine sim.
    """

    query_embedding = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array([np.array(emb) for emb in doc_embeddings])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Filter docs with sim >= threshold
    mask = similarities >= threshold
    filtered_docs = documents_df[mask].copy()  # Keep filtered docs as DataFrame
    filtered_docs["cosine_sim"] = similarities[mask]

    # Sort by cosine similarity descending and select top_k
    top_k_docs = filtered_docs.sort_values(by="cosine_sim", ascending=False).head(top_k)

    print(f"Filtered {len(filtered_docs)} docs above threshold {threshold}")
    print(f"Returning top {len(top_k_docs)} docs sorted by cosine similarity")
    
    return top_k_docs


def combine_filtered_documents_with_labels(cosine_docs, llm_docs):
    """
    Combine documents from cosine and LLM filters using OR logic.
    Adds a 'selected_by' field indicating which filter(s) selected the document.
    """
    doc_dict = {}

    # Add cosine-filtered docs
    for doc in cosine_docs:
        title = doc.get("title", "").strip()
        if title not in doc_dict:
            doc["selected_by"] = ["cosine"]
            doc_dict[title] = doc
        else:
            doc_dict[title]["selected_by"].append("cosine")

    # Add LLM-filtered docs
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
    """
    Reranks documents by their 'n_citation' field in descending order.
    Works for both pandas DataFrame and list of dicts.
    Returns the same type as input.
    """
    # Detect input type
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
    Rank the given expert_ids based on their n_citation count in authors_df (highest first).
    
    Parameters:
    - expert_ids (list): List of author IDs to rank.
    - authors_df (DataFrame): DataFrame with columns 'id' and 'n_citation'.
    
    Returns:
    - ranked_experts (list): List of expert_ids ranked by citation count.
    """
    # print(expert_ids)
    #print(authors_df['tags'])
    authors_df = authors_df[authors_df['tags'].notna()] #get rid of NaN tags 
    #print('filtered_non_na_df',authors_df)
    # Filter authors_df to only rows matching our expert_ids
    filtered_df = authors_df[authors_df['id'].isin(expert_ids)]
    #print(filtered_df)

    # Sort by n_citation descending
    ranked_df = filtered_df.sort_values(by='n_citation', ascending=False)
    #print(ranked_df)
    # Get ranked list of ids
    ranked_experts = ranked_df['id'].tolist()

    return ranked_experts


def retrieve_similar_documents_qwen(qwen_model, query_embedding, papers_df, top_k=25, threshold=0.3):
    """
    Retrieves top_k similar documents using qwen_model.similarity()
    """
    # Prepare document embeddings
    doc_embeddings = np.vstack(papers_df["embeddings"].apply(ast.literal_eval).values)  # (N, D)
    query_vec = np.array(query_embedding).reshape(1, -1)  # (1, D)

    # Compute similarity using qwen_model.similarity()
    with torch.no_grad():
        sims = qwen_model.similarity(torch.tensor(query_vec), torch.tensor(doc_embeddings))  # shape: (1, N)
    sims = sims.cpu().numpy().flatten()

    # Filter by threshold and select top_k
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
    """
    Returns BM25 scores for query against all documents.
    """
    return [
        bm25.score(query_tokens, i)
        for i in range(bm25.N)
    ]


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
    """
    Optimized retrieval that reduces computation by:
    - Computing qwen_model.similarity once on batched tensors moved to the model/device.
    - Selecting only the top candidates by the model similarity (fast argpartition),
      and computing Jaccard similarity only for those candidates.
    - Combining similarities via weighted sum and returning top_k results.

    New parameter `candidate_multiplier` controls how many top model-sim candidates
    we compute the (more expensive) Jaccard for. Set to 3 by default.
    """
    candidate_multiplier = 3

    # Prepare document embeddings (parse lazily to avoid large allocations)
    emb_values = papers_df["embeddings"].values
    # Parse embeddings into list of numpy arrays (float32)
    emb_list = [np.array(ast.literal_eval(x), dtype=np.float32) for x in emb_values]
    n_docs = len(emb_list)

    query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    # Try to infer the device where the model lives; fallback to CPU
    try:
        model_device = next(qwen_model.parameters()).device
    except Exception:
        model_device = torch.device("cpu")

    # Compute similarity in chunks to avoid moving the whole corpus to device at once
    query_tensor = torch.from_numpy(query_vec).to(model_device)
    sims = np.empty(n_docs, dtype=np.float32)
    sim_batch_size = 2048  # tune this based on memory; smaller is safer

    for start in range(0, n_docs, sim_batch_size):
        end = min(start + sim_batch_size, n_docs)
        chunk = np.stack(emb_list[start:end])  # (batch, D)
        chunk_tensor = torch.from_numpy(chunk).to(model_device)
        with torch.no_grad():
            sims_chunk = qwen_model.similarity(query_tensor, chunk_tensor)
        sims[start:end] = sims_chunk.cpu().numpy().flatten()

    n_docs = len(sims)
    if n_docs == 0:
        return papers_df.copy().assign(weighted_similarity=np.array([]))

    # Choose top candidates by model similarity to limit expensive Jaccard calculations
    candidate_k = min(n_docs, max(top_k, top_k * candidate_multiplier))
    # fast selection of top candidate indices
    if candidate_k < n_docs:
        top_candidate_idx = np.argpartition(-sims, candidate_k - 1)[:candidate_k]
        # sort those indices by descending sim
        top_candidate_idx = top_candidate_idx[np.argsort(-sims[top_candidate_idx])]
    else:
        top_candidate_idx = np.argsort(-sims)

    # Compute Jaccard only for chosen candidates
    jaccard_sims = np.zeros(n_docs, dtype=np.float32)
    # Pre-build doc_texts array to avoid repeated .get calls
    doc_texts = (papers_df.get("title", "").fillna("").astype(str).str.strip() + " " +
                 papers_df.get("abstract", "").fillna("").astype(str).str.strip()).values

    for idx in top_candidate_idx:
        jaccard_sims[idx] = jaccard_similarity(query, doc_texts[idx])

    # Compute BM25 similarity only for chosen candidates to save time/memory
    # candidate_texts = [doc_texts[idx] for idx in top_candidate_idx]
    # bm25_candidate = BM25([doc_text.split() for doc_text in candidate_texts])
    # bm25_query = query.split()
    # bm25_sims = np.zeros(n_docs, dtype=np.float32)
    # bm25_candidate uses indices 0..len(candidate_texts)-1, map back to original indices
    # for local_i, idx in enumerate(top_candidate_idx):
    #     bm25_sims[idx] = bm25_candidate.score(bm25_query, local_i)

    # Compute Levenshtein similarity only for chosen candidates to save time/memory
    levenshtein_sims = np.zeros(n_docs, dtype=np.float32)
    for idx in top_candidate_idx:
        levenshtein_sims[idx] = Levenshtein_similarity(query, doc_texts[idx])

    # Combine similarities (weighted sum)
    alpha = 0.5
    beta = 0.35
    gamma = 0.15
    combined_sims = (alpha * _minmax(sims)) + (beta * _minmax(jaccard_sims)) + (gamma * _minmax(levenshtein_sims))

    # Attach combined score and return filtered top_k rows
    papers_df = papers_df.copy()
    papers_df["weighted_similarity"] = combined_sims

    # Filter by threshold first, then select top_k
    filtered = papers_df[papers_df["weighted_similarity"] >= threshold]
    if filtered.empty:
        # If nothing passes threshold, still return the top_k by weighted score
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
        logits = outputs.logits[0, -1]  # Last token

    yes_logit = logits[token_yes_id]
    no_logit = logits[token_no_id]
    probs = F.softmax(torch.tensor([no_logit, yes_logit]), dim=0)
    #print("probability ", probs)
    return probs[1].item() > 0.3  # True if "yes" is more probable


def rerank_authors_qwen(query, author_ids, author_contexts):
    """
    Returns a list of (author_id, yes_probability) sorted in descending order of relevance.
    """
    token_yes_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    token_no_id = reranker_tokenizer.convert_tokens_to_ids("no")
    ranked_authors = []

    i = 0
    for author_id in author_ids:
        # context = knowledge_graph.get_author_context(author_id, rdf_file_path)
        # context = knowledge_graph.enrich_author_context_with_coauthors(author_id, rdf_file_path)
        prompt = format_author_prompt(query, author_contexts[i])

        inputs = reranker_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = reranker_model(**inputs)
            logits = outputs.logits[0, -1]

        # Compute softmax on the last token logits for "yes" and "no"
        yes_logit = logits[token_yes_id]
        no_logit = logits[token_no_id]
        probs = torch.softmax(torch.tensor([no_logit, yes_logit], device=device), dim=0)
        yes_prob = probs[1].item()

        ranked_authors.append((author_id, yes_prob))
        i += 1

    # Sort by probability of "yes" descending
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

    # Example (strict) — the model should follow this structure exactly
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



def generate_author_explanation(
    query,
    author_context,
    relevant_ranked_documents
):
    """
    Generates a structured explanation (free text + evidence) for an author.

    Returns:
        dict: Parsed JSON explanation
    """

    prompt = format_explanation_prompt(
        query=query,
        author_context=author_context
        )

    inputs = reranker_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
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

    # --- Robust JSON parsing ---
    try:
        explanation = json.loads(generated_text)
    except json.JSONDecodeError:
        explanation = {
            "raw_output": generated_text
        }
    return explanation



import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        corpus: List[List[str]]  -> tokenized documents
        """
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


def rerank_documents_deepseek_pairwise(
    query: str,
    docs_df,
    top_k: int = 15,
    batch_size: int = 8,
):
    """
    Pairwise reranking using DeepSeek-R1-0528-Qwen3-8B.

    Uses insertion-based pairwise ranking:
    - Compare documents A vs B
    - Model outputs "A" or "B"
    """

    torch.cuda.empty_cache()

    # ---- Build document texts once ----
    docs = []
    for row in docs_df.itertuples(index=False, name=None):
        try:
            title = row[docs_df.columns.get_loc("title")] if "title" in docs_df.columns else ""
            abstract = row[docs_df.columns.get_loc("abstract")] if "abstract" in docs_df.columns else ""
        except Exception:
            title = row[0] if len(row) > 0 else ""
            abstract = row[1] if len(row) > 1 else ""

        docs.append(f"{str(title).strip()} {str(abstract).strip()}")

    # ---- Only rerank top_k candidates ----
    docs = docs[:top_k]
    indices = list(range(len(docs)))

    ranked = [indices[0]]  # start with first doc

    # ---- Pairwise insertion ranking ----
    for idx in indices[1:]:
        inserted = False

        for pos in range(len(ranked)):
            doc_a = docs[idx]
            doc_b = docs[ranked[pos]]

            prompt = format_pairwise_prompt_deepseek(query, doc_a, doc_b)

            inputs = reranker_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = reranker_model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.0,
                )

            decision = reranker_tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            if decision == "A":
                ranked.insert(pos, idx)
                inserted = True
                break

        if not inserted:
            ranked.append(idx)

    # ---- Build reranked DataFrame ----
    reranked_df = docs_df.iloc[ranked].reset_index(drop=True)

    # Optional: add rank column
    # reranked_df["pairwise_rank"] = range(1, len(reranked_df) + 1)

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


def rerank_authors_deepseek_pairwise(
    query,
    author_ids,
    author_contexts,
    top_k: int = None,
):
    """
    Pairwise author reranking using DeepSeek-R1-0528-Qwen3-8B.

    Parameters:
    - query: search query
    - author_ids: list of author IDs
    - author_contexts: list of text contexts aligned with author_ids
    - top_k: optional cutoff
    """

    torch.cuda.empty_cache()

    assert len(author_ids) == len(author_contexts)

    # Optional cutoff before expensive pairwise ranking
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
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = reranker_model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.0,
                )

            decision = reranker_tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()

            if decision == "A":
                ranked.insert(pos, idx)
                inserted = True
                break

        if not inserted:
            ranked.append(idx)

    return [author_ids[i] for i in ranked]