import torch
import torch.nn as nn
from torch.nn import functional as F
from nano_medgpt.src.datasets.utils import *

torch.manual_seed(435123)

# hyperparamters
vocab_size = 37
dropout = 0.2
context_length = 8
num_heads = 1
n_embed = 4
n_layers = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length))) # not a model parameter. 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention affinities
        w = q @ k.transpose(-2, -1) * C**-0.5 # matmul, scale
        w = w.masked_fill(self.tril[:T, :T] ==0, float('-inf')) # mask
        w = F.softmax(w, dim=-1) # softmax
        w = self.dropout(w)

        # weighted aggregation
        v = self.value(x)
        out = w @ v # matmul
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ linear layer with RELU """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), # from ATTN paper - inner layer of n_embed (512 -> 2048) has a multiplier of x4
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block """
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed//num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_length, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads=num_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape 

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
    
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to last context_length tokens
            idx_cond = idx[:, -context_length:] 

            # forward
            logits, loss = self(idx_cond)

            # last timestep
            logits = logits[:, -1, :]

            # get all probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled idx to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


