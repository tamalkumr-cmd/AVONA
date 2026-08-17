# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """
    Computes Q, K, V linear projections, scaled dot-product attention,
    applies the causal mask (no looking at future tokens), and projects back.
    """
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Combined Q, K, V projections in one linear layer for speed
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # Lower-triangular causal mask buffer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                          .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch, Sequence Length, Embedding Dim

        # 1. Project to Q, K, V and split across heads
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 3. Apply causal mask (future positions set to -infinity)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # 4. Weighted aggregation of values: Att @ V
        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble head outputs

        return self.c_proj(y)


class MLP(nn.Module):
    """2-layer Feed-Forward Network with GELU activation."""
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with residual skip connections."""
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # Attention + Residual Skip
        x = x + self.mlp(self.ln2(x))  # MLP + Residual Skip
        return x


class CustomGPT(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=256, n_head=4, n_layer=4, block_size=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share embedding weights with final classification head
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        # Token + Positional embeddings
        tok_emb = self.token_embedding(idx) # (B, T, n_embd)
        pos_emb = self.pos_embedding(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb

        # Pass through Transformer blocks
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to calculate standard cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Autoregressive generation loop with temperature and top-k sampling."""
        for _ in range(max_new_tokens):
            # Crop to context window if sequence exceeds block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature # Focus on the last token

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx