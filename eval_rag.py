# eval_rag.py
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# 1. Construct Ground Truth Test Dataset
eval_samples = {
    "question": [
        "What is the server maintenance mode override passcode?",
        "What edge computing hardware does Project Orion utilize?",
        "What is the daily lunch reimbursement limit during client travel?",
        "Which database engine powers the primary cluster?"
    ],
    "contexts": [
        ["The secret server override passcode for maintenance mode is ALPHA-992-DELTA."],
        ["Project Orion is a custom autonomous drone developed by the team in 2026, equipped with an RTX-powered onboard edge computing unit."],
        ["Expense Limit: Daily lunch meal reimbursement during client travel is capped at $45 per person without prior approval."],
        ["Database Architecture: Primary cluster runs PostgreSQL 16 on AWS Aurora with cross-region read replicas."]
    ],
    "answer": [
        "The maintenance mode override passcode is ALPHA-992-DELTA.",
        "Project Orion uses an RTX-powered onboard edge computing unit.",
        "The daily lunch reimbursement limit is $45 per person without prior approval.",
        "The primary cluster runs PostgreSQL 16 on AWS Aurora."
    ],
    "ground_truth": [
        "ALPHA-992-DELTA",
        "RTX-powered edge computing unit",
        "$45 per person",
        "PostgreSQL 16 on AWS Aurora"
    ]
}

eval_dataset = Dataset.from_dict(eval_samples)

print("Running Quantitative RAG Evaluation...")
# 2. Compute Metric Scores
score_results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ]
)

# 3. Format and Display Report
df_report = score_results.to_pandas()
print("\n" + "=" * 60)
print("       RAG QUANTITATIVE BENCHMARK REPORT")
print("=" * 60)
print(df_report[["question", "faithfulness", "answer_relevancy", "context_precision"]])

print("\n--- Overall Metric Averages ---")
for metric, val in score_results.items():
    print(f"{metric:<20}: {val:.4f}")