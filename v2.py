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
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
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


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # we need position embedding, because in self-attention, the tokens have no idea where
        # they are in the sequence, since self-attention and aggregation can be thought of as
        # a directed graph where all previous nodes point to the next token node.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers.
        tok_emb = self.token_embedding_table(idx)  # (batch_size, block_size, vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_head(x) # apply 1 head of self-attention
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

"""
Step 0: train loss 4.7305, val loss 4.7241
Step 300: train loss 2.8110, val loss 2.8249
Step 600: train loss 2.5434, val loss 2.5682
Step 900: train loss 2.4932, val loss 2.5088
Step 1200: train loss 2.4863, val loss 2.5035
Step 1500: train loss 2.4665, val loss 2.4921
Step 1800: train loss 2.4683, val loss 2.4936
Step 2100: train loss 2.4696, val loss 2.4846
Step 2400: train loss 2.4638, val loss 2.4879
Step 2700: train loss 2.4738, val loss 2.4911
Step 3000: train loss 2.4613, val loss 2.4897
Step 3300: train loss 2.4689, val loss 2.4793
Step 3600: train loss 2.4554, val loss 2.4919
Step 3900: train loss 2.4682, val loss 2.4906
Step 4200: train loss 2.4634, val loss 2.4882
Step 4500: train loss 2.4563, val loss 2.4804
Step 4800: train loss 2.4557, val loss 2.4852
Step 5100: train loss 2.4605, val loss 2.4926
Step 5400: train loss 2.4614, val loss 2.4932
Step 5700: train loss 2.4684, val loss 2.4781
Step 6000: train loss 2.4651, val loss 2.4672
Step 6300: train loss 2.4586, val loss 2.4858
Step 6600: train loss 2.4696, val loss 2.4893
Step 6900: train loss 2.4482, val loss 2.4910
Step 7200: train loss 2.4522, val loss 2.4827
Step 7500: train loss 2.4559, val loss 2.4896
Step 7800: train loss 2.4423, val loss 2.4936
Step 8100: train loss 2.4592, val loss 2.4923
Step 8400: train loss 2.4537, val loss 2.4865
Step 8700: train loss 2.4567, val loss 2.4805
Step 9000: train loss 2.4510, val loss 2.4840
Step 9300: train loss 2.4535, val loss 2.4868
Step 9600: train loss 2.4609, val loss 2.4908
Step 9900: train loss 2.4530, val loss 2.4864
Step 10200: train loss 2.4458, val loss 2.4902
Step 10500: train loss 2.4637, val loss 2.4865
Step 10800: train loss 2.4474, val loss 2.4911
Step 11100: train loss 2.4600, val loss 2.4898
Step 11400: train loss 2.4519, val loss 2.4950
Step 11700: train loss 2.4588, val loss 2.4881
Step 12000: train loss 2.4433, val loss 2.4878
Step 12300: train loss 2.4604, val loss 2.4895
Step 12600: train loss 2.4548, val loss 2.4948
Step 12900: train loss 2.4629, val loss 2.4802
Step 13200: train loss 2.4607, val loss 2.5007
Step 13500: train loss 2.4614, val loss 2.4952
Step 13800: train loss 2.4599, val loss 2.4817
Step 14100: train loss 2.4607, val loss 2.4879
Step 14400: train loss 2.4525, val loss 2.4945
Step 14700: train loss 2.4619, val loss 2.4917
Step 15000: train loss 2.4550, val loss 2.5013
Step 15300: train loss 2.4623, val loss 2.4926
Step 15600: train loss 2.4576, val loss 2.4900
Step 15900: train loss 2.4505, val loss 2.5004
Step 16200: train loss 2.4572, val loss 2.4964
Step 16500: train loss 2.4529, val loss 2.4839
Step 16800: train loss 2.4461, val loss 2.4953
Step 17100: train loss 2.4500, val loss 2.4875
Step 17400: train loss 2.4530, val loss 2.4928
Step 17700: train loss 2.4556, val loss 2.4925
Step 18000: train loss 2.4581, val loss 2.4979
Step 18300: train loss 2.4585, val loss 2.4991
Step 18600: train loss 2.4564, val loss 2.4892
Step 18900: train loss 2.4564, val loss 2.4995
Step 19200: train loss 2.4464, val loss 2.4883
Step 19500: train loss 2.4562, val loss 2.4906
Step 19800: train loss 2.4565, val loss 2.5012
2.4440457820892334

Foasth prse tize herst el
O u frnie hy:


Hak, CORineg agnthe t rr Masearor charnge?
Tyoucre thy, chouspo in mpery way avend ouburser sickns bekncard dhiceny

He tw el fe oupise he, le stselownthers;
Nom w
T:
TIONTouly m hofaruks, g he itheland's oe, oghithet f, badogienthofathatey foueay wad,
ureisold array n
ICoyockield, murs, in mamybalorthyongmyoorord Vofetthindy hak shil brveseay alsteanerm to, oupomp! wee d pre h, gavit gin Thean apsts lathind my d erouerse IOLUEDid n: athicorire.
II IS:
I
"""
