#!/bin/bash
# SLURM OPTIONS
#SBATCH --partition=gpu-a6000  # Partition is a queue for jobs
#SBATCH --time=48:00:00         # Time limit for the job
#SBATCH --job-name=temporal_25 # Name of your jobs
#SBATCH --error=error_logs/job-%j.err
#SBATCH --output=output_logs/job-%j.out
#SBATCH --nodes=1            # Number of nodes you want to run your process on
#SBATCH --ntasks-per-node=16 # Number of CPU cores
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1        # Number of GPUs

# Define timestamps
#timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# Rename logs with timestamp
#mv error_logs/job-$SLURM_JOB_ID.err error_logs/job-$SLURM_JOB_ID-$timestamp.err
#mv output_logs/job-$SLURM_JOB_ID.out output_logs/job-$SLURM_JOB_ID-$timestamp.out

PYTHON_VERSION=3.11 
ENVIRONMENT_NAME="expert-search"

module load Anaconda3
# You need this to be able to use the 'conda' command. I am not satisfied with this solution, I'll try to find a way so that you won't have to set it in the future.
source /opt/easybuild/software/Anaconda3/2024.02-1/etc/profile.d/conda.sh


# if the environment does not exist, create it
if ! conda info --envs | grep -q "^${ENVIRONMENT_NAME}"; then
  conda create -n ${ENVIRONMENT_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENVIRONMENT_NAME}

# # Install python packages, either individually or using a requirements.txt file.
# pip install -r ~/repo/Expert-Finding/requirements.txt
# pip install pyvis rdflib kglab owlrl 
# pip install pyvis jinja2
# python3 -m pip uninstall -y flash-attn flash_attn
# pip install python-levenshtein
# pip install -U transformers accelerate torch sentencepiece
# pip install transformers==5.0.0rc0
# pip install mistral-common --upgrade
# pip install -U transformers accelerate
# pip uninstall transformers -y
# pip install --user transformers==4.46.2 accelerate
# pip install --user git+https://github.com/huggingface/transformers.git accelerate
# pip install -U FlagEmbedding
# pip install faiss-cpu
# pip install rank-bm25
# pip install --upgrade sentence-transformers
# pip install --upgrade datasets pyarrow typing-extensions
# pip install --upgrade transformers peft --break-system-packages
# pip install python-Levenshtein
# pip install --upgrade FlagEmbedding




# python3 upgrade_kg.py #Litsearch_KnowledgeGraph_first_secon_third_order_topics.py #RAGXDoc.py  #KG_Relation_Extractor.py #RAGXDoc.py

# python3 temporal_ragxpert.py #inject_role_scores_to_kg.py   #phase3_role_inference.py #phase2b_recency_vectors.py  #phase1_regime_partition.py #regime_partition.py #author_topics_tensor.py   #debug_code_data.py

# python3 inference_explain.py --pipeline baseline --query "Dimensionality Reduction"

# python3 inference_explain.py --pipeline proposed --query "Dimensionality Reduction"

python3 data.py