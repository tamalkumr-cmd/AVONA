# eval_local.py
import time
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# 1. Load Local Embedding Model
print("Loading local embedding model for semantic evaluation...")
evaluator = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define Benchmark Dataset (Questions, Contexts, Generated Answers, Ground Truth)
benchmark_data = [
    {
        "question": "What is the secret server override passcode?",
        "context": "The secret server override passcode for maintenance mode is ALPHA-992-DELTA.",
        "generated_answer": "The server override passcode for maintenance mode is ALPHA-992-DELTA.",
        "ground_truth": "ALPHA-992-DELTA"
    },
    {
        "question": "What edge computing hardware does Project Orion utilize?",
        "context": "Project Orion is a custom autonomous drone developed in 2026, equipped with an RTX-powered onboard edge computing unit.",
        "generated_answer": "Project Orion utilizes an RTX-powered edge computing unit on board.",
        "ground_truth": "RTX-powered onboard edge computing unit"
    },
    {
        "question": "What is the daily lunch reimbursement limit during client travel?",
        "context": "Expense Limit: Daily lunch meal reimbursement during client travel is capped at $45 per person without prior approval.",
        "generated_answer": "The daily meal reimbursement is $45 per person.",
        "ground_truth": "$45 per person"
    },
    {
        "question": "Which database engine powers the primary cluster?",
        "context": "Database Architecture: Primary cluster runs PostgreSQL 16 on AWS Aurora with cross-region read replicas.",
        "generated_answer": "The database cluster runs on MySQL on Google Cloud.", # Simulated hallucination test
        "ground_truth": "PostgreSQL 16 on AWS Aurora"
    }
]

# 3. Compute Quantitative Metrics
results = []
print("Evaluating test samples locally...\n")

for row in benchmark_data:
    q = row["question"]
    ctx = row["context"]
    gen = row["generated_answer"]
    truth = row["ground_truth"]
    
    # A. Context Hit Rate (Is the ground truth present in the retrieved chunk?)
    context_hit = 1.0 if truth.lower() in ctx.lower() else 0.0
    
    # B. Fact Containment / Grounding (Did the model output the ground truth?)
    fact_contained = 1.0 if truth.lower() in gen.lower() else 0.0
    
    # C. Semantic Cosine Similarity (BERTScore equivalent using embeddings)
    emb_gen = evaluator.encode(gen, convert_to_tensor=True)
    emb_truth = evaluator.encode(truth, convert_to_tensor=True)
    semantic_sim = util.cos_sim(emb_gen, emb_truth).item()
    
    results.append({
        "Question": q,
        "Semantic Similarity": round(max(0.0, semantic_sim), 3),
        "Fact Containment": fact_contained,
        "Context Hit": context_hit
    })

# 4. Generate Performance Report
df = pd.DataFrame(results)

print("=" * 80)
print("                   LOCAL QUANTITATIVE RAG BENCHMARK")
print("=" * 80)
print(df.to_string(index=False))
print("-" * 80)
print(f"Average Semantic Similarity : {df['Semantic Similarity'].mean():.3f} / 1.000")
print(f"Fact Containment Rate       : {df['Fact Containment'].mean() * 100:.1f}%")
print(f"Retrieval Context Hit Rate  : {df['Context Hit'].mean() * 100:.1f}%")
print("=" * 80)