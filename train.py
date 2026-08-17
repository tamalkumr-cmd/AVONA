# train.py
import torch
from data import load_and_tokenize_data
from model import CustomGPT

BATCH_SIZE = 16
BLOCK_SIZE = 128
N_EMBD = 384       # Scaled up for better expressive capacity
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1
LEARNING_RATE = 4e-4
MAX_STEPS = 2500   # Run ~2500 steps (takes ~4-5 mins on RTX 3050)
EVAL_INTERVAL = 250
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_data, val_data, enc = load_and_tokenize_data(sample_limit=15000)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model = CustomGPT(
    vocab_size=enc.n_vocab,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params / 1e6:.2f}M parameters on {DEVICE.upper()}\n")

for step in range(1, MAX_STEPS + 1):
    model.train()
    xb, yb = get_batch("train")

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == 'cuda')):
        logits, loss = model(xb, yb)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    if step % EVAL_INTERVAL == 0 or step == 1:
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch("val")
            _, val_loss = model(x_val, y_val)
        print(f"Step {step:4d}/{MAX_STEPS} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

torch.save(model.state_dict(), "tinystories_llm.pt")
print("\nTrained model saved to 'tinystories_llm.pt'!")