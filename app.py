# app.py
import gradio as gr
import tiktoken
import torch
from model import CustomGPT

N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
BLOCK_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "tinystories_llm.pt"

enc = tiktoken.get_encoding("gpt2")
model = CustomGPT(
    vocab_size=enc.n_vocab,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout=0.0
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

def generate_story(prompt, max_tokens, temperature, top_k):
    if not prompt.strip():
        return "Please enter a starting prompt."
    
    tokens = enc.encode(prompt)
    input_tensor = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k)
        )[0].tolist()
        
    return enc.decode(output)

demo = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(lines=3, placeholder="Once upon a time, Lily found a tiny red button...", label="Prompt"),
        gr.Slider(20, 200, value=100, step=10, label="Max Tokens"),
        gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Temperature"),
        gr.Slider(10, 100, value=40, step=5, label="Top-K")
    ],
    outputs=gr.Textbox(lines=6, label="Generated Output"),
    title="🤖 My First Custom LLM (Trained from Scratch)",
    description="Running locally on NVIDIA RTX 3050 with PyTorch & CUDA."
)

if __name__ == "__main__":
    demo.launch()