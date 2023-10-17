import torch
from torch.nn import functional as F
from src.datasets.utils import *
from src.models.bigram import BigramLM
from src.models.gpt import *
import random
import pickle

torch.manual_seed(456123)

# hyperparamters
vocab_size = 37
n_epochs = 100
eval_iters = 5
eval_interval = 5
batch_size = 8
context_length = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------


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

# Wrapper for _get_batch method to load the encodings
def get_batch(dataset="train"):
    """ 
    train ~ parts 0-5 (only part 0 for now)
    val   ~ part 6
    """
    part_selected = 0
    if dataset == "train":
        part_selected = 0 # random.randint(0, 5)
    else:
        part_selected = 6
    
    fpath = PROCESSED_DATA_PATH + "part_" + str(part_selected) + "_encoded.pt"
    df = torch.load(fpath)
    return df

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(make_encodings=False):
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
            losses = estimate_loss(model)
            print("Step number {}: Training Loss: {}, Validation Loss:{}".format(i, losses['train'], losses['val']))
        
        # Check for model saving condition
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Main training loop
        xb, yb = get_batch("train")
        logits, loss = m(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    
    print("After training, the best validation loss is {}".format(best_val_loss))

if __name__ == "__main__":
    train_model()
    model = torch.load(MODEL_SAVE_PATH)
    generate_text(model, max_new_tokens = 10000)