import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

# 1. 4-bit Quantization Config for RTX 3050 (4GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 2. Load Base Model in BF16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA Configuration (Attention + MLP projections)
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 4. Load & Pre-Tokenize Trajectory Dataset
raw_dataset = load_dataset("json", data_files="data/agent_trajectories.jsonl")

def tokenize_trajectories(batch):
    texts = [tokenizer.apply_chat_template(conv, tokenize=False) for conv in batch["messages"]]
    tokenized = tokenizer(texts, truncation=True, max_length=768)
    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    return tokenized

tokenized_dataset = raw_dataset["train"].map(
    tokenize_trajectories,
    batched=True,
    remove_columns=raw_dataset["train"].column_names
)

# 5. Standard Training Arguments
training_args = TrainingArguments(
    output_dir="./agent_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    num_train_epochs=3,
    bf16=True,
    fp16=False,
    logging_steps=10,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    save_strategy="epoch"
)

# 6. Standard Trainer with Dynamic Padding
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
)

print("Training Broad-Knowledge Coding Agent on RTX 3050...")
trainer.train()

# Save final adapter weights and tokenizer
model.save_pretrained("./agent_adapter")
tokenizer.save_pretrained("./agent_adapter")
print("Saved broad-capability adapter to ./agent_adapter")