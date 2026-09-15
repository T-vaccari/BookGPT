import torch
import torch.nn as nn
from torch.nn import functional as F
from gfx1010_attention import scaled_dot_product_attention


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
   def __init__(self, n_embd, n_head, block_size, attention_implementation):
      super().__init__()
      assert n_embd % n_head == 0
      self.n_head    = n_head
      self.n_embd    = n_embd
      self.head_size = n_embd // n_head
      self.attention_implementation = attention_implementation

      self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # packed Q, K, V
      self.c_proj = nn.Linear(n_embd, n_embd)       # output projection
      self.tril: torch.Tensor
      self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

   def forward(self, x):
      B, T, C = x.shape

      # 1. compute Q, K, V for all heads in one matmul, then split
      q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # each (B, T, n_embd)

      # 2. reshape into (B, n_head, T, head_size)
      q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
      k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
      v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

      # 3. scaled dot-product attention
      y = scaled_dot_product_attention(
         q,
         k,
         v,
         is_causal=True,
         implementation=self.attention_implementation,
      )

      # 4. re-assemble heads
      y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
      return self.c_proj(y)


class Block(nn.Module):
   def __init__(self, n_embd, n_head, block_size, dropout, attention_implementation):
      super().__init__()
      self.sa_head = MultiHeadAttention(
         n_embd,
         n_head,
         block_size,
         attention_implementation,
      )
      self.ffwd    = FeedForward(n_embd, dropout)
      self.ln1     = nn.LayerNorm(n_embd)
      self.ln2     = nn.LayerNorm(n_embd)

   def forward(self, x):
      x = x + self.sa_head(self.ln1(x))
      x = x + self.ffwd(self.ln2(x))
      return x


class GPT(nn.Module):
   def __init__(
      self,
      vocab_size,
      n_embd,
      n_head,
      block_size,
      n_blocks=4,
      dropout=0.1,
      attention_implementation="auto",
   ):
      super().__init__()
      self.block_size = block_size

      self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
      self.position_embedding_table = nn.Embedding(block_size, n_embd)
      self.blocks = nn.ModuleList([
         Block(n_embd, n_head, block_size, dropout, attention_implementation)
         for _ in range(n_blocks)
      ])
      self.layer_norm               = nn.LayerNorm(n_embd)
      self.lm_head                  = nn.Linear(n_embd, vocab_size)

   def forward(self, idx, targets=None):
      B, T = idx.shape

      tok_emb = self.token_embedding_table(idx)                                    # (B, T, n_embd)
      pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
      x = tok_emb + pos_emb                                                        # (B, T, n_embd)
      for block in self.blocks:
         x = block(x)
      x = self.layer_norm(x)
      logits = self.lm_head(x)                                                     # (B, T, vocab_size)

      loss = None
      if targets is not None:
         B, T, C = logits.shape
         loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

      return logits, loss

   def generate(self, idx, max_new_tokens, temperature=1.0, stop_token=None, decode=None):
      for _ in range(max_new_tokens):
        idx_cond  = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits    = logits[:, -1, :]

        if temperature <= 0:
            idx_next = logits.argmax(dim=-1, keepdim=True)   # greedy, nessuna softmax
        else:
            probs    = F.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

        if stop_token is not None and decode is not None:
            if decode([idx_next.item()]) == stop_token:
                break

      return idx
