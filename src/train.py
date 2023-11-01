import torch
from torch.nn import functional as F
from nano_medgpt.src.datasets.utils import *
from nano_medgpt.src.models.bigram import BigramLM
from nano_medgpt.src.models.gpt import *
import random
import pickle
import argparse
from azureml.core.run import Run
import glob

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


def get_batch(data_folder, split="train"):
    part_selected = 0
    if split == "train":
        part_selected = 0 # random.randint(0, 5)
    else:
        part_selected = 6

    fpath = data_folder + "part_" + str(part_selected) + "_encoded.pt"
    df = torch.load(fpath)
    return df


# Wrapper for _get_batch method to load the encodings
def local_get_batch(dataset="train"):
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
            losses = estimate_loss(model)
            print("Step number {}: Training Loss: {}, Validation Loss:{}".format(i, losses['train'], losses['val']))
        
        # Check for model saving condition
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            # torch.save(model.state_dict(), MODEL_SAVE_PATH)
            torch.save(model.state_dict(), './outputs/model.pth')

        # Main training loop
        xb, yb = get_batch(data_folder, "train")
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
# if __name__ == "__main__":
#     train_model()
#     model = torch.load(MODEL_SAVE_PATH)
#     generate_text(model, max_new_tokens = 10000)