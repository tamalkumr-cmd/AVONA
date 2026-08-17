# benchmark.py
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "./my_finetuned_slm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("1. Loading model to GPU for benchmarking...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map=DEVICE)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).merge_and_unload()
model.eval()

test_prompt = "Explain how backpropagation works in neural networks in detail."
messages = [{"role": "user", "content": test_prompt}]
formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

# Warmup pass
_ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()

print("2. Running generation benchmark (150 tokens)...")
start_time = time.perf_counter()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        min_new_tokens=150,
        do_sample=False,
        use_cache=True
    )

torch.cuda.synchronize()
end_time = time.perf_counter()

# Calculations
total_time = end_time - start_time
tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
tokens_per_sec = tokens_generated / total_time
vram_used = torch.cuda.max_memory_allocated() / (1024 ** 2)

print("\n" + "=" * 50)
print("           LIVE BENCHMARK RESULTS")
print("=" * 50)
print(f"Device                   : {torch.cuda.get_device_name(0)}")
print(f"Tokens Generated         : {tokens_generated} tokens")
print(f"Total Inference Time     : {total_time:.3f} seconds")
print(f"Generation Throughput    : {tokens_per_sec:.2f} tokens/second")
print(f"Latency per Token        : {(total_time / tokens_generated) * 1000:.2f} ms/token")
print(f"Peak VRAM Consumption    : {vram_used:.1f} MB")
print("=" * 50)