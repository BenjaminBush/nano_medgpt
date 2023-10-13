import torch
from torch.nn import functional as F
from src.datasets.utils import *
from src.models.bigram import BigramLM
from src.models.gpt import *

torch.manual_seed(435123)

# hyperparamters
vocab_size = 37
n_epochs = 5000
batch_size = 64
context_length = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------


def generate_text(m, max_new_tokens=100):
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_idx = m.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    return decode(generated_idx)

def get_batch(df):
    ix = torch.randint(len(df)-context_length, (batch_size,))
    x = torch.stack([df[i:i+context_length] for i in ix])
    y = torch.stack([df[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def train_model():
    train, test = train_test_split()

    # instantiate the model and the optimizer
    model = GPT()
    m = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(n_epochs):
        xb, yb = get_batch(train)
        logits, loss = m(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    print(loss.item())
    print(generate_text(model))

if __name__ == "__main__":
    train_model()