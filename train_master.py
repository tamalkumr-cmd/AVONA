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

# 1. 4-bit Quantization Config (Optimized for 4GB RTX 3050)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 2. Base Model Setup
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 3. High-Rank Full LoRA Adaptation
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

# 4. Response Loss Masking Pre-processor
raw_dataset = load_dataset("json", data_files="data/master_train.jsonl")
response_template = "<|im_start|>assistant\n"
response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

def tokenize_and_mask(batch):
    input_ids_list = []
    labels_list = []
    
    for conv in batch["messages"]:
        full_text = tokenizer.apply_chat_template(conv, tokenize=False)
        enc = tokenizer(full_text, truncation=True, max_length=1024)
        input_ids = enc["input_ids"]
        labels = [-100] * len(input_ids)
        
        # Unmask only tokens corresponding to assistant turns
        for i in range(len(input_ids) - len(response_token_ids) + 1):
            if input_ids[i:i+len(response_token_ids)] == response_token_ids:
                start_idx = i + len(response_token_ids)
                # Find end of turn (<|im_end|>)
                end_idx = start_idx
                while end_idx < len(input_ids) and input_ids[end_idx] != tokenizer.encode("<|im_end|>")[0]:
                    end_idx += 1
                for j in range(start_idx, min(end_idx + 1, len(input_ids))):
                    labels[j] = input_ids[j]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}

tokenized_dataset = raw_dataset["train"].map(
    tokenize_and_mask,
    batched=True,
    remove_columns=raw_dataset["train"].column_names
)

# 5. Advanced Training Arguments
training_args = TrainingArguments(
    output_dir="./agent_adapter",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,                  # Replaced warmup_ratio with warmup_steps
    num_train_epochs=3,
    bf16=True,
    fp16=False,
    logging_steps=20,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
)

print("Training Master Model (Math + Facts + Code + C++) on RTX 3050...")
trainer.train()

model.save_pretrained("./agent_adapter")
tokenizer.save_pretrained("./agent_adapter")
print("Saved fully scaled master adapter to ./agent_adapter")