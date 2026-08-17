# merge_model.py
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "./my_finetuned_slm"
MERGED_DIR = "./smollm2_360m_merged"

print("1. Loading base model and adapters...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
print("2. Fusing LoRA adapters into base weights...")
merged_model = model.merge_and_unload()

# Untie embeddings and duplicate weights so llama.cpp finds output.weight
print("3. Untying embeddings for GGUF output.weight compatibility...")
merged_model.config.tie_word_embeddings = False
merged_model.lm_head.weight = torch.nn.Parameter(
    merged_model.model.embed_tokens.weight.clone()
)

print(f"4. Saving merged model to {MERGED_DIR}...")
merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)

# Sanitize tokenizer_config.json for llama.cpp converter compatibility
config_path = os.path.join(MERGED_DIR, "tokenizer_config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "extra_special_tokens" in cfg and isinstance(cfg["extra_special_tokens"], list):
        del cfg["extra_special_tokens"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

print("Merge and configuration complete!")