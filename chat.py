# chat.py
import torch
import tiktoken
from model import CustomGPT

# Configuration matching the saved checkpoint
BLOCK_SIZE = 128
N_EMBD = 256
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0  # Dropout should be 0 during inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "mini_llm_checkpoint.pt"

# 1. Load Tokenizer
enc = tiktoken.get_encoding("gpt2")

# 2. Instantiate Architecture and Load Weights
model = CustomGPT(
    vocab_size=enc.n_vocab,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT
).to(DEVICE)

try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint '{CHECKPOINT_PATH}' successfully on {DEVICE.upper()}!\n")
except FileNotFoundError:
    print(f"Error: Could not find '{CHECKPOINT_PATH}'. Please run train.py first.")
    exit()

# 3. Interactive CLI Loop
print("=" * 60)
print(" Mini-LLM Interactive Terminal (Type 'quit' or 'exit' to stop)")
print("=" * 60)

while True:
    try:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chat...")
            break

        # Encode prompt
        input_tokens = enc.encode(user_input)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Generate tokens
        print("LLM: ", end="", flush=True)
        with torch.no_grad():
            output_tokens = model.generate(
                input_tensor,
                max_new_tokens=80,
                temperature=0.7,  # Lower = more deterministic, Higher = more creative
                top_k=40          # Limits sampling to top 40 probable tokens
            )[0].tolist()

        # Print only the newly generated continuation
        new_tokens = output_tokens[len(input_tokens):]
        print(enc.decode(new_tokens))

    except KeyboardInterrupt:
        print("\nSession ended.")
        break