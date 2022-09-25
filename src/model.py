from typing import NamedTuple
import torch
import torch.nn as nn
import torch.functional as F


class GPT2Config(NamedTuple):
    n_layer: int = 12
    n_heads: int = 12
    emb_dim: int = 768
    vocab_size: int = 50_257
    block_size: int = 1024

    attn_drop: float = 0.1
    resid_drop: float = 0.1
    emb_drop: float = 0.1


class GPT2(nn.Module):
    """A decoder-only Transformer like GPT-2"""

    def __init__(self, config: GPT2Config):
        super().__init__()

        # Calculate head dimension
        self.head_dim = config.emb_dim // config.n_heads

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        self.wpe = nn.Embedding(config.block_size, config.emb_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # LM head producing logits
        self.head = nn.Linear(config.emb_dim, config.vocab_size)

        # Embedding dropout
        self.emb_drop = nn.Dropout(config.attn_drop)

        # Final LayerNorm before logits
        self.final_ln = nn.LayerNorm(config.emb_dim)

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.emb_dim),
            wpe=nn.Embedding(config.block_size, config.emb_dim),
            drop=nn.Dropout(config.emb_drop),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.emb_dim),
        ))

        # Final linear layer for token prediction
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, token_ids):
        B, S = token_ids.shape
        pos_idx = torch.arange(0, S).unsqueeze(0)  # (1, S)

        # Create input embeddings
        token_embs = self.transformer.wte(token_ids)
        pos_embs = self.transformer.wpe(pos_idx)
        x = self.transformer.drop(token_embs + pos_embs)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_pretrained(cls):
        from transformers import GPT2LMHeadModel

        # Create GPT-2 model
        config = GPT2Config()
        model = GPT2(config)
        sd = model.state_dict()

        # Get HF GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()

        # Weights to ignore
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]

        # OpenAI uses Conv1D instead of a Linear layer
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Copy weights into our model
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


class TransformerBlock(nn.Module):
    """A single Transformer block with MHA and MLP"""

    def __init__(self, config: GPT2Config):
        super().__init__()

        # Layer norms and MHA
        self.ln_1 = nn.LayerNorm(config.emb_dim)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_dim)

        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.emb_dim, 4 * config.emb_dim),
            c_proj=nn.Linear(4 * config.emb_dim, config.emb_dim),
            act=nn.GELU(),
            dropout=nn.Dropout(config.resid_drop)
        ))

        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        # Attention
        x = x + self.attn(self.ln_1(x))

        # MLP
        x = x + self.mlpf(self.ln_2(x))

        return x


class MultiHeadAttention(nn.Module):
    """Implementation of a multi-head self-attention layer"""

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads

        # Use one linear layer for q, k, v for all heads
        self.c_attn = nn.Linear(config.emb_dim, 3 * config.emb_dim)

        # Output projection after multi-head self-attention
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim)

        # Regularisation
        self.attn_drop = nn.Dropout(config.attn_drop)
        self.resid_drop = nn.Dropout(config.resid_drop)

        # Causal attention mask
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # Batch size, sequence length, embedding dimension (emb_dim)
        N, S, E = x.shape

        proj = self.c_attn(x)  # (N, S, 3 * E)
        q, k, v = proj.chunk(3, dim=2)  # (N, S, E)

        # Split into heads
        q = q.view(N, S, self.n_heads, self.head_dim).transpose(1, 2)  # (N, H, S, D)
        k = k.view(N, S, self.n_heads, self.head_dim).transpose(1, 2)  # (N, H, S, D)
        v = v.view(N, S, self.n_heads, self.head_dim).transpose(1, 2)  # (N, H, S, D)

        # Calculate scaled dot product attention
        attn = q @ k.transpose(2, 3)  # (N, H, S, S)
        attn = attn * (1. / torch.sqrt(self.head_dim))

        # Causual attention
        attn = attn.masked_fill_(self.attn_mask == 0, float('-inf'))
        attn_scores = F.softmax(attn, dim=-1)  # (N, H, S, S)
        attn_scores = self.attn_drop(attn_scores)

        # Combine values using attention scores
        out = attn_scores @ v  # (N, H, S, D)

        # Concatenate heads
        out = out.transpose(1, 2)  # (N, S, H, D)
        out = out.view(N, S, E)  # (N, S, E)

        # Output projection
        out = self.c_proj(out)  # (N, S, E)

        # Output regularisation
        out = self.resid_drop(out)

        return out


if __name__ == '__main__':
    config = GPT2Config()
    model = GPT2.from_pretrained()