# run_gguf.py
from llama_cpp import Llama

# Match the exact filename produced by conversion:
MODEL_PATH = "./smollm2-360m-q8.gguf"

print("Loading quantized GGUF model into memory...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=30  # Offloads layers to RTX 3050 CUDA
)

print("\nGenerating response...\n")
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a concise, helpful technical assistant."},
        {"role": "user", "content": "Explain what a KV Cache is in 2 short bullet points."}
    ],
    temperature=0.7,
    max_tokens=150
)

print("=" * 40)
print(response["choices"][0]["message"]["content"])
print("=" * 40)