import pandas as pd
import json
from ast import literal_eval
from collections import defaultdict
from tqdm import tqdm
import requests
import json
from Utils import GetDocuments
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import nltk
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from nltk.corpus import stopwords
import filter_data, evaluation
import query_authors
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("rdflib").setLevel(logging.ERROR)
import ast
from nltk.tokenize import sent_tokenize
import regex as re

# ── Load embedding model ───────────────────────────────────────────────────────
qwen_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

# ── Configure LLaMA pipeline ───────────────────────────────────────────────────
config = GetDocuments.read_json_file("llama_config.json")
pipe = pipeline(
    "text-generation",
    model=config["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=config["token"],
)
print("LLama is running on ", pipe.model.hf_device_map)

# ── TF-IDF vectorizer ─────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ── Load Upgraded7 KG with role scores (once at startup) ──────────────────────

print("Loading Upgraded KG with role scores...")
kg_with_roles = Graph()
kg_with_roles.parse(
    "Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl",
    format="turtle"
)
print(f"  -> KG loaded: {len(kg_with_roles):,} triples")


def extract_keywords_from_query(prompt):
    if pipe is None:
        raise RuntimeError("LLM pipeline is not available.")

    message = [
        {"role": "system",
         "content": "From now on, extract any keyword from my prompt that resembles a topic. Output only the extracted topic (1-4 words) in the format: [topic]. Do not include any additional words, explanations, or variations. Maintain this format strictly in all responses."},
        {"role": "user", "content": prompt}
    ]

    output = [pipe(message, max_new_tokens=10)]
    prompt_keyword = output[0][0]['generated_text'][2]['content']

    if '[' in prompt_keyword:
        prompt_keyword = prompt_keyword.replace("[", "").replace("]", "")

    if prompt_keyword is None:
        print("Output is Null.")
        return extract_keywords_from_query(prompt)

    filtered_keywords = remove_stopwords(prompt_keyword)
    return filtered_keywords


def contains_stopwords(text):
    words = text.lower().split()
    return any(word in stop_words for word in words)


def remove_stopwords(text):
    if isinstance(text, str):
        tokens = text.split()
    else:
        tokens = text
    return ' '.join([word for word in tokens if word.lower() not in stop_words])


def get_embedding_from_specter(title, abstract, prompt, target):
    if title is None:
        title = ""
    if abstract is None:
        abstract = ""
    if target == 'paper':
        paper_text = title + " " + abstract
        embedding = qwen_model.encode([paper_text])[0]
    else:
        embedding = qwen_model.encode([prompt])[0]
    return embedding.tolist()


def simple_sentence_split(text):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())


def get_avg_sentence_embedding_from_qwen(title, abstract):
    paragraph = f"{title} {abstract}".strip()

    if not paragraph:
        return np.zeros(qwen_model.get_dimension()).tolist()

    sentences = simple_sentence_split(paragraph)
    sentence_embeddings = qwen_model.encode(sentences)
    avg_embedding = np.mean(sentence_embeddings, axis=0)

    return avg_embedding.tolist()


def get_embedding_from_sbert(title, abstract, prompt, target):
    if title is None:
        title = ""
    if abstract is None:
        abstract = []

    if target == 'paper':
        title_embedding = sbert_model.encode([title])[0]

        abstract_embeddings = []
        for sent in abstract:
            if sent.strip():
                sent_embedding = sbert_model.encode([sent])[0]
                abstract_embeddings.append(sent_embedding)

        if abstract_embeddings:
            abstract_avg_embedding = np.mean(abstract_embeddings, axis=0)
        else:
            abstract_avg_embedding = np.zeros(sbert_model.get_sentence_embedding_dimension())

        combined_embedding = ((title_embedding + abstract_avg_embedding) / 2).tolist()

    else:
        combined_embedding = sbert_model.encode([prompt])[0].tolist()

    return combined_embedding


def create_binary_validation_prompt(query, author_context):
    prompt = f"""

    Given the query and author information below, answer ONLY with "Yes" or "No". No explanations, no other text.

    Query: {query}

    Author information: 
    - Papers: {', '.join(paper['title'] for paper in author_context['papers']) if author_context['papers'] else 'None'}
    - Topics: {', '.join(author_context['topics']) if author_context['topics'] else 'None'}

    Is this author relevant to the query?

    Answer:"""

    return prompt


def llm_binary_relevance(llm_pipe, prompt):
    response = llm_pipe(prompt, max_new_tokens=3)[0]['generated_text'].strip().lower()
    if "answer: yes" in response or "answer: yes." in response:
        return True
    elif "answer: no" in response or "answer: no." in response:
        return False
    else:
        return False


def validate_authors_binary(llm_pipe, query, top_authors, rdf_file_path):
    relevant_authors = []
    for author_id in top_authors:
        context = knowledge_graph.get_author_context(author_id, rdf_file_path)
        prompt = create_binary_validation_prompt(query, context)
        is_relevant = llm_binary_relevance(llm_pipe, prompt)
        if is_relevant:
            relevant_authors.append(author_id)
    return relevant_authors


def expand_query_with_llm(query):
    instruction_sentence = "Do not include this sentence in the response."

    prompt = (
        "Please define and clarify the search query below for better search results. "
        "Return only the definition query as a single concise paragraph. "
        "Do not repeat the instructions or include the input query in your output.\n\n"
        "---INPUT---\n"
        f"{query}\n"
        "---END INPUT---\n\n"
        "OUTPUT:"
    )

    gen = pipe(prompt, max_new_tokens=25)[0]['generated_text']

    if 'OUTPUT:' in gen:
        response = gen.split('OUTPUT:')[-1].strip()
    else:
        response = gen.strip()

    if instruction_sentence in response:
        response = response.replace(instruction_sentence, '').strip()

    response = response.split('\n\n')[0].strip()

    return response


def main():
    time1_start = time.time()
    print("Starting Expert Search Process...")

    # ── Load data ──────────────────────────────────────────────────────────────
    time_to_load_data_start = time.time()
    papers_df = pd.read_csv("~/repo/Expert-Finding/papers_df_with_topics_qwen_separate_sentences.csv")
    time_to_load_data_end = time.time()
    print(f"Time taken to load data: {time_to_load_data_end - time_to_load_data_start} seconds")

    time_to_load_authors_start = time.time()
    authors_df = pd.read_csv("~/repo/Expert-Finding/papers_and_authors/authors.csv")
    time_to_load_authors_end = time.time()
    print(f"Time taken to load authors data: {time_to_load_authors_end - time_to_load_authors_start} seconds")

    #KG path
    rdf_file = "Upgraded7_Knowledge_graph_with_hierarchical_topics_full_dataset.ttl"

    # ── Metric accumulators ────────────────────────────────────────────────────
    map10_scores          = []
    mrr10_scores          = []
    mp5_scores            = []
    mp10_scores           = []
    ndcg5_scores          = []
    ndcg10_scores         = []
    role_diversity_scores = []   

    prompt_keywords = [['cluster analysis'], ['Bayesian statistics'], ['world wide web'], ['Novelty detection'],
    ['Image segmentation'], ['kernel density estimation'], ['gibbs sampling'], ['semantic grid'],
    ['Parallel algorithm'], ['learning to rank'], ['user interface'], ['Knowledge extraction'],
    ['Monte Carlo method'], ['relational database'], ['belief propagation'], ['Computational biology'],
    ['Convex optimization'], ['activity recognition'], ['interpolation'], ['Web 2.0'],
    ['Dimensionality reduction'], ['wearable computer'], ['wavelet transform'], ['Network theory'],
    ['Facial recognition system']]

    time2_start = time.time()

    for prompt_keyword in tqdm(prompt_keywords, desc="Processing Queries.."):
        embeddings = []
        topics = []
        kg_docs = []
        prompt_keyword = prompt_keyword[0]

        # ── Query expansion ────────────────────────────────────────────────────
        time_to_expand_query_start = time.time()
        prompt = expand_query_with_llm(prompt_keyword)
        time_to_expand_query_end = time.time()
        print(f"Time taken to expand query: {time_to_expand_query_end - time_to_expand_query_start} seconds")
        print("Prompt Keyword: ", prompt_keyword)
        print("Synthetic Prompt: ", prompt)

        # ── Query embedding ────────────────────────────────────────────────────
        query_embedding = get_embedding_from_specter(None, None, prompt, 'prompt')

        # ── Document retrieval ─────────────────────────────────────────────────
        time_to_get_weighted_similarity_start = time.time()
        print("Computing Cosine Similarity...")
        weighted_similar_docs = filter_data.retrieve_similar_documents_qwen_weighted(
            prompt_keyword, qwen_model, query_embedding, papers_df, top_k=25, threshold=0.3
        )
        print(f"No. of (cosine) similar documents: {len(weighted_similar_docs)}")
        time_to_get_weighted_similarity_end = time.time()
        print(f"Time taken to compute weighted similarity: {time_to_get_weighted_similarity_end - time_to_get_weighted_similarity_start} seconds")

        # ── Paper reranking ────────────────────────────────────────────────────
        print("Fetching Final Relevant Documents..")
        relevant_ranked_documents = []
        if len(weighted_similar_docs) > 1:
            time_to_rerank_papers_start = time.time()
            relevant_ranked_documents = filter_data.rerank_documents_qwen(prompt, weighted_similar_docs, top_k=15)
            # relevant_ranked_documents = filter_data.rerank_documents_deepseek_pairwise(
            #     prompt,
            #     weighted_similar_docs,
            #     top_k=15,
            #     batch_size=8,
            # )
            time_to_rerank_papers_end = time.time()
            print(f"Time taken to rerank papers: {time_to_rerank_papers_end - time_to_rerank_papers_start} seconds")
        else:
            print("**Number of documents passed through the cosine similarity threshold is lesser than 2**")
            continue

        print(f"Ranked Documents by relevance to query {relevant_ranked_documents}")

        # ── Extract candidate authors from ranked papers ────────────────────────
        candidate_experts = []
        for row in relevant_ranked_documents.itertuples():
            candidate_experts.append(row.authors)

        expert_ids, expert_names = [], []
        for expert_str in candidate_experts:
            experts_list = literal_eval(expert_str)
            for expert in experts_list:
                expert_ids.append(literal_eval(expert["id"]))
                expert_names.append(expert["name"])

        # ── CHANGED: role-diverse ranking instead of citation ranking ──────────
        # OLD: ranked_expert_ids = filter_data.rank_experts_by_citations(expert_ids, authors_df)
        ranked_expert_ids = filter_data.rank_experts_by_role_diversity(
            expert_ids, authors_df, kg_with_roles
        )

        predicted_ids = ranked_expert_ids

        # ── Get RAG contexts from Upgraded7 KG ────────────────────────────────
        author_contexts, id_context_dict = query_authors.get_rag_texts_for_ids(
            predicted_ids,
            g=kg_with_roles,        # pass pre-loaded graph — avoids reloading KG per query
            min_coauthored=2,
            max_paper_authors=50
        )
        # Role label + probabilities are now embedded in each RAG context
        # by query_authors.rag_text_for_author() — no separate enrichment step needed.

        # LLM author reranking (receives role-enriched contexts from query_authors)
        time_to_rerank_authors_start = time.time()
        relevant_author_ids = filter_data.rerank_authors_qwen(prompt_keyword, predicted_ids, author_contexts)
        # relevant_author_ids = filter_data.rerank_authors_deepseek_pairwise(
        #     prompt_keyword,
        #     predicted_ids,
        #     author_contexts,        # already contains role info from query_authors
        #     top_k=None
        # )
        time_to_rerank_authors_end = time.time()
        print(f"Time taken to rerank authors: {time_to_rerank_authors_end - time_to_rerank_authors_start} seconds")

        print("Predicted Ranked Expert IDs:", predicted_ids)
        print("Relevant authors after LLM validation:", relevant_author_ids)

        # ── CHANGED: compute role diversity score ──────────────────────────────
        diversity_score = filter_data.compute_role_diversity_score(
            relevant_author_ids, kg_with_roles, top_k=10
        )
        print(f"Role diversity score (top-10): {diversity_score:.4f}")

        # ── Ground truth ───────────────────────────────────────────────────────
        print('prompt keyword', prompt_keyword)
        ground_truth_ids = evaluation.get_ground_truth_experts(
            authors_df, prompt_keyword, top_k=None
        )
        print("Ground Truth IDs:", len(ground_truth_ids))

        # ── Evaluation ─────────────────────────────────────────────────────────
        results = evaluation.evaluate_expert_ranking(relevant_author_ids, ground_truth_ids)
        print("Evaluation Results:", results)

        map10_scores.append(results['MAP@10'])
        mrr10_scores.append(results['MRR@10'])
        mp5_scores.append(float(results['MP@10']))
        mp10_scores.append(float(results['MP@5']))
        ndcg5_scores.append(float(results['NDCG@10']))
        ndcg10_scores.append(float(results['NDCG@5']))
        role_diversity_scores.append(diversity_score)   

    # ── Final averages ─────────────────────────────────────────────────────────
    print("Average MAP@10:",             np.mean(map10_scores))
    print("Average MRR@10:",             np.mean(mrr10_scores))
    print("Average MP@10:",              np.mean(mp10_scores))
    print("Average MP@5:",               np.mean(mp5_scores))
    print("Average NDCG@10:",            np.mean(ndcg10_scores))
    print("Average NDCG@5:",             np.mean(ndcg5_scores))
    print("Average Role Diversity@10:",  np.mean(role_diversity_scores))   

    end_time = time.time()
    print(f"Total Time taken: {end_time - time1_start} seconds")
    print(f"Net Time taken: {end_time - time2_start} seconds")


if __name__ == "__main__":
    main()
