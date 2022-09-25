import math
import torch
import torch.nn as nn
from typing import NamedTuple
from torch.nn import functional as F


class GPT2Config(NamedTuple):
    n_layer: int = 12
    n_heads: int = 12
    emb_dim: int = 768
    vocab_size: int = 50_257
    block_size: int = 1024

    attn_drop: float = 0.1
    resid_drop: float = 0.1
    emb_drop: float = 0.1

    @classmethod
    def from_model_type(cls, model_type: str):
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']

        model_configs = {
            'gpt2': dict(n_layer=12, n_heads=12, emb_dim=768),
            'gpt2-medium': dict(n_layer=24, n_heads=16, emb_dim=1024),
            'gpt2-large': dict(n_layer=36, n_heads=20, emb_dim=1280),
            'gpt2-xl': dict(n_layer=48, n_heads=25, emb_dim=1600),
        }

        config = GPT2Config(**model_configs[model_type])
        return config


class GPT2(nn.Module):
    """A decoder-only Transformer like GPT-2"""

    def __init__(self, config: GPT2Config, model_type='gpt2'):
        super().__init__()

        self.config = config

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

        # Parameter count
        num_params = sum(p.numel() for p in self.transformer.parameters())
        print(f'Model: {model_type}, number of parameters: {num_params / 1e6:.2f}M')

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
    def from_pretrained(cls, model_type='gpt2'):
        from transformers import GPT2LMHeadModel

        # Create GPT-2 model
        config = GPT2Config.from_model_type(model_type)
        model = GPT2(config, model_type=model_type)
        sd = model.state_dict()

        # Get HF GPT-2 model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Truncate sequence to block_size
            idx_trunc = idx if idx.size(1) <= self.config.block_size else idx[:, -self.block_size:]

            # Get logits
            logits = self(idx_trunc)

            # Create probability distribution over next token
            logits = logits[:, -1, :] / temp

            # Only consider top_k options
            if top_k is not None:
                prob_thresh = torch.topk(logits, top_k).values.min()
                logits[logits < prob_thresh] = float('-inf')

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx


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

        # Calculate scaled dot product attention (scaled by head dimension)
        attn = q @ k.transpose(2, 3)  # (N, H, S, S)
        attn = attn * (1. / math.sqrt(k.size(-1)))

        # Causual attention
        attn = attn.masked_fill_(self.bias[:, :, :S, :S] == 0, float('-inf'))
        attn_scores = F.softmax(attn, dim=-1)  # (N, H, S, S)
        attn_scores = self.attn_drop(attn_scores)

        # Combine values using attention scores
        out = attn_scores @ v  # (N, H, S, D)

        # Concatenate heads
        out = out.transpose(1, 2)  # (N, S, H, D)
        out = out.contiguous().view(N, S, E)  # (N, S, E)

        # Output projection
        out = self.c_proj(out)  # (N, S, E)

        # Output regularisation
        out = self.resid_drop(out)

        return out
