# chat_slm.py
import threading
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "./my_finetuned_slm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading base model and fine-tuned LoRA weights...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map=DEVICE
)

# Merge LoRA adapters for maximum inference speed
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model = model.merge_and_unload()  # Fuses adapter weights directly into base layers
model.eval()

def extract_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return " ".join(text_parts)
    elif isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)

def respond(message, history):
    messages = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "user")
            raw_content = item.get("content", "")
            messages.append({"role": role, "content": extract_text(raw_content)})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.append({"role": "user", "content": extract_text(item[0])})
            messages.append({"role": "assistant", "content": extract_text(item[1])})

    messages.append({"role": "user", "content": extract_text(message)})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Set up background token streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True  # Enables KV cache for faster generation
    )

    # Run generation on a background thread so tokens stream continuously
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream text live into Gradio UI
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# Launch Gradio interface
gr.ChatInterface(
    respond,
    title="💬 Real-Time Streaming Chatbot (SmolLM2-360M + LoRA)",
    description="Optimized with KV Cache, LoRA Weight Fusion, and Live Token Streaming."
).launch()