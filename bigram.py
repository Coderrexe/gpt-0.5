import torch
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32  # number of independent sequences we process in parallel
block_size = 8  # maximum context length for predictions
max_iters = 3000
eval_interval = 300
# A typical good learning rate is 3e-4, but for small networks like this 1e-2 is fine.
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# ----------------

torch.manual_seed(42)

# Read in the data and inspect it.
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Here are all the unique characters that occur in this text.
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers.
char2idx = {char:idx for idx, char in enumerate(chars)}
idx2char = {idx:char for idx, char in enumerate(chars)}
# Encoder: takes a string, outputs a list of integers.
encode = lambda s: [char2idx[char] for char in s]
# Decoder: takes a list of integers, outputs a string.
decode = lambda l: "".join(idx2char[i] for i in l)

# Let's now encode the entire text dataset and store it into a torch.Tensor.
data = torch.tensor(encode(text), dtype=torch.long)
# Split data into training (90%) and validation (10%).
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading.
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y.
    data = train_data if split == "train" else val_data
    # Generate `batch_size` number of contexts and targets.
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y


# Calculate loss as a mean of many iterations.
# torch.no_grad tells PyTorch that we will not call .backward() in this function.
# This makes PyTorch a lot more efficient with memory use because it doesn't have to store
# any intermediate variables.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers.
        logits = self.token_embedding_table(idx)  # (batch_size, block_size, vocab_size)
        
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context.
        for _ in range(max_new_tokens):
            # Get the predictions.
            logits, loss = self(idx)
            # Focus only on the last time step.
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities.
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution.
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append the newly sampled index to the running sequence.
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


model = BigramLanguageModel(vocab_size)
model = model.to(device)

# Create PyTorch optimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets.
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # Sample a random batch of data.
    xb, yb = get_batch("train")

    # Evaluate the loss.
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(out)
