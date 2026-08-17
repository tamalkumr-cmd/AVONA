# finetune.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
OUTPUT_DIR = "./my_finetuned_slm"

print(f"1. Loading tokenizer and model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model in FP16 (Takes ~720MB VRAM)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda"
)

# 2. Configure LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                       # Rank
    lora_alpha=32,              # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# 3. Load Instruction Dataset (Multi-turn conversations)
print("2. Downloading instruction dataset...")
dataset = load_dataset("trl-lib/Capybara", split="train[:3000]")

# 4. Training Configuration (Tailored for RTX 3050)
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4, # Effective batch size = 16
    learning_rate=2e-4,            # LoRA uses higher LR than full fine-tuning
    num_train_epochs=1,
    logging_steps=20,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

# 5. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

print("\n3. Starting LoRA Fine-Tuning on RTX 3050...")
trainer.train()

print(f"\n4. Saving fine-tuned LoRA adapters to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete!")