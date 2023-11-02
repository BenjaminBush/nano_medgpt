import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import pickle
import argparse
from azureml.core.run import Run
import glob
import os

torch.manual_seed(456123)

# training hyperparamters
vocab_size = 37
n_epochs = 500 # 5000
eval_iters = 20 # 200
eval_interval = 5 # 500
batch_size = 64
context_length = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------

# model hyperparamters
dropout = 0.2
num_heads = 6
n_embed = 384
n_layers = 6
# -----------------------------------------------------

################################ UTILS #############################
char_vocab = list("0123456789abcdefghijklmnopqrstuvwxyz ")
stoi = {}
itos = {}
i = 0
for c in char_vocab:
    stoi[c] = i
    itos[i] = c
    i+=1
    
# Encode: string -> int
encode = lambda s: [stoi[c] for c in s]

# Decode: int -> string
decode = lambda i: ''.join([itos[j] for j in i])

################################ GPT.PY   ###########################
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

################################ TRAIN.PY ###########################
def generate_text(m, max_new_tokens=100):
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_idx = m.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(generated_idx)

def _get_batch(df):
    ix = torch.randint(len(df)-context_length, (batch_size,))
    x = torch.stack([df[i:i+context_length] for i in ix])
    y = torch.stack([df[i+1:i+context_length+1] for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


def get_batch(data_folder, split="train"):
    part_selected = 0
    if split == "train":
        part_selected = 0 # random.randint(0, 5)
    else:
        part_selected = 6

    fpath = data_folder + "part_" + str(part_selected) + "_encoded.pt"
    df = torch.load(fpath)
    return _get_batch(df)


@torch.no_grad()
def estimate_loss(data_folder, model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data_folder = data_folder, split=split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(data_folder = None, make_encodings=False):
    if make_encodings:
        make_encodings()

    # instantiate the model and the optimizer
    model = GPT()
    m = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = 1e9

    for i in range(n_epochs):
        # First, let's check to see if we are at an eval step
        if i % eval_interval == 0:
            losses = estimate_loss(data_folder, model)
            print("Step number {}: Training Loss: {}, Validation Loss:{}".format(i, losses['train'], losses['val']))
        
        # Check for model saving condition
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            # torch.save(model.state_dict(), MODEL_SAVE_PATH)
            torch.save(model.state_dict(), './outputs/model.pth')

        # Main training loop
        xb, yb = get_batch(data_folder=data_folder, split="train")
        logits, loss = m(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    
    print("After training, the best validation loss is {}".format(best_val_loss))


parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, help='dataset')
args = parser.parse_args()

os.makedirs('./outputs', exist_ok=True)
data_folder = args.data_folder

run = Run.get_context() 

train_model(data_folder)
