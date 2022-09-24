import torch
import torch.nn as nn
import torch.functional as F

class GPT2(nn.Module):
    """A decoder-only Transformer like GPT-2"""

    def __init__(self, n_layer: int = 12, n_heads: int = 12, emb_dim: int = 768, vocab_size: int = 50257, block_len: int = 1024, drop: float = 0.1):
        super().__init__()

        # Calculate head dimension
        head_dim = self.emb_dim // self.n_heads

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(block_len, emb_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(n_heads, emb_dim, head_dim, drop, drop, block_len) for _ in range(n_layer)])

        # LM head producing logits
        self.head = nn.Linear(emb_dim, vocab_size)

        # Embedding dropout
        self.emb_drop = nn.Dropout(drop)

        # Final LayerNorm before logits
        self.final_ln = nn.LayerNorm(emb_dim)

    def forward(self, token_ids):
        B, S = token_ids.shape
        pos_idx = torch.arange(0, S).unsqueeze(0) # (1, S)

        # Create input embeddings by adding token + position embeddings
        token_embs = self.token_emb(token_ids)
        pos_embs = self.pos_emb(pos_idx)
        x = self.emb_drop(token_embs + pos_embs)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.head(x)

        return logits


class TransformerBlock(nn.Module):
    """A single Transformer block with MHA and MLP"""

    def __init__(self, n_heads: int, emb_dim: int, head_dim: int, attn_drop: float, resid_drop: float, block_len: int):
        super().__init__()

        # Layer norms and MHA
        self.ln_1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(n_heads, emb_dim, head_dim, attn_drop, resid_drop, block_len)
        self.ln_2 = nn.LayerNorm(emb_dim)

        # MLP
        self.fc_1 = nn.Linear(emb_dim, 4 * emb_dim)
        self.fc_2 = nn.Linear(4 * emb_dim, emb_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(resid_drop)

        self.mlp = lambda x: self.fc_2(self.drop(self.act(self.fc_1(x))))

    def forward(self, x):
        # Attention
        x = x + self.attn(self.ln_1(x))

        # MLP
        x = x + self.mlp(self.ln_2(x))

        return x



class MultiHeadAttention(nn.Moduel):
    """Implementation of a multi-head self-attention layer"""

    def __init__(self, n_heads: int, emb_dim: int, head_dim: int, attn_drop: float, resid_drop: float, seq_len: int):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = head_dim

        # Use one linear layer for q, k, v for all heads
        self.attn_proj = nn.Linear(emb_dim, 3 * emb_dim)

        # Output projection after multi-head self-attention
        self.out_proj = nn.Linear(emb_dim, emb_dim)

        # Regularisation
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        # Attention mask
        self.attn_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

    def forward(self, x):
        # Batch size, sequence length, embedding dimension (emb_dim)
        N, S, E = x.shape

        proj = self.attn_proj(x) # (N, S, 3 * E)
        q, k, v = proj.chunk(3, dim=2) # (N, S, E)

        # Split into heads
        q = q.view(N, S, self.n_heads, self.head_dim).transpose(1, 2) # (N, H, S, D)
        k = k.view(N, S, self.n_heads, self.head_dim).transpose(1, 2) # (N, H, S, D)
        v = v.view(N, S, self.n_heads, self.head_dim).transpose(1, 2) # (N, H, S, D)

        # Calculate scaled dot product attention
        attn = q @ k.transpose(2, 3) # (N, H, S, S)
        attn = attn * (1. / torch.sqrt(self.head_dim))

        # Causual attention
        attn = attn.masked_fill_(self.attn_mask == 0, float('-inf'))
        attn_scores = F.softmax(attn, dim=-1) # (N, H, S, S)
        attn_scores = self.attn_drop(attn_scores)

        # Combine values using attention scores
        out = attn_scores @ v # (N, H, S, D)

        # Concatenate heads
        out = out.transpose(1, 2) # (N, S, H, D)
        out = out.view(N, S, E) # (N, S, E)

        # Output projection
        out = self.out_proj(out) # (N, S, E)

        # Output regularisation
        out = self.resid_drop(out)

        return out

    

