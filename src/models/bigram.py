import torch
import torch.nn as nn
from torch.nn import functional as F
from nano_medgpt.src.datasets.utils import *

torch.manual_seed(435123)
batch_size = 32
context_length = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
    
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
            # forward
            logits, loss = self(idx)

            # last timestep
            logits = logits[:, -1, :]

            # get all probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled idx to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


