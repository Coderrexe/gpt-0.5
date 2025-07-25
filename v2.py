import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
# A typical good learning rate is 3e-4, but for small networks like this higher LR is fine.
learning_rate = 1e-3
eval_iters = 200
n_embd = 32 # number of embedding dimensions

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(1337)

# Read in the data and inspect it.
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

data = torch.tensor(encode(text), dtype=torch.long)
# Split data into training (90%) and validation (10%).
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # Generate a small batch of data of inputs x and targets y.
    data = train_data if split == "train" else val_data
    # Generate `batch_size` number of contexts and targets.
    indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y


# calculate loss as a mean of many iterations.
# torch.no_grad means we won't call .backward() in this function, so PyTorch
# can be  more efficient with memory since it doesn't have to store intermediate variables.
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


class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # what do I offer to others?
        self.query = nn.Linear(n_embd, head_size, bias=False) # what am I looking for?
        self.value = nn.Linear(n_embd, head_size, bias=False) # what should be returned if someone attends to me?
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    

class FeedForward(nn.Module):
    """ simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension
        # n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # Residual connection – better for optimisation
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # we need position embedding, because in self-attention, the tokens have no idea where
        # they are in the sequence, since self-attention and aggregation can be thought of as
        # a directed graph where all previous nodes point to the next token node.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers.
        tok_emb = self.token_embedding_table(idx)  # (batch_size, block_size, vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predictions.
            logits, loss = self(idx_cond)
            # Focus only on the last time step.
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities.
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution.
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append the newly sampled index to the running sequence.
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


model = BigramLanguageModel()
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample a random batch of data.
    xb, yb = get_batch("train")

    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(out)
