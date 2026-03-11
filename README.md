# t-RAGXpert: Career-aware Temporal Modeling for Unbiased Expert Finding

A knowledge-graph-based pipeline for academic expert finding that models expertise as a temporally evolving, role-aware phenomenon. The system combines a scholarly knowledge graph (6.75M RDF triples), temporal expertise representations, dynamic career-role inference, and LLM-based reranking to retrieve and explain ranked expert recommendations.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `temporal_ragxpert.py` | Main pipeline entry point (proposed system) |
| `filter_data.py` | Candidate ranking, RAG context construction, and role diversity scoring |
| `query_authors.py` | Builds per-author KG context for LLM reranking |
| `evaluation.py` | Computes MAP@10, MRR@10, MP@5/10, NDCG@5/10, and Role Diversity@K |
| `inference_explain.py` | Single-query inference with per-author relevance and role explanations |
| `data.py` | Data loading and preprocessing utilities |
| `Utils.py` | Shared utility functions |
| `author_topics_tensor.py` | Builds author–topic–year tensors from paper abstracts |
| `upgrade_kg.py` | Augments the base KG with career metrics and authorship position statistics |
| `inject_role_scores_to_kg.py` | Injects inferred role probability triples into the KG |
| `phase1_regime_partition.py` | Partitions authors into Regime B (core) and Regime A (episodic) |
| `phase2a_slice_vectors.py` | Computes 5-year overlapping slice vectors for Regime B authors |
| `phase2b_recency_vectors.py` | Computes recency-weighted aggregation vectors for Regime A authors |
| `phase3_role_inference.py` | Infers soft role probabilities (emerging / collaborating / supervising) |
| `regime_partition.py` | Regime partitioning utilities |
| `debug_code_data.py` | Debugging and data inspection utilities |
| `modelBatch.sh` | SLURM batch script for running pipeline phases on a compute cluster |

---

## 🔄 Pipeline Phases
```text
Phase 0  upgrade_kg.py                Build and augment the knowledge graph
Phase 1  phase1_regime_partition.py   Partition authors by career profile
Phase 2a phase2a_slice_vectors.py     Temporal slice vectors (Regime B)
Phase 2b phase2b_recency_vectors.py   Recency-weighted vectors (Regime A)
Phase 3  phase3_role_inference.py     Role-state inference → inject into KG
Phase 4  temporal_ragxpert.py         Expert ranking and retrieval
Phase 5  evaluation.py                Evaluate against ground truth
Phase 6  inference_explain.py         Single-query inference with explanations
```

---

## 🚀 Quick Start
```bash
# Run the full proposed pipeline on a query
python temporal_ragxpert.py 

# Run single-query inference with explanations
python inference_explain.py --pipeline proposed --query "Dimensionality Reduction"
```


