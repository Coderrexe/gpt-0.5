import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
# A typical good learning rate is 3e-4, but for small networks like this higher LR is fine.
learning_rate = 3e-4
eval_iters = 200
n_embd = 384 # number of embedding dimensions
n_head = 6
n_layer = 6
dropout = 0.2

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
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores (affinities)
        # algorithm given in the original paper
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """ simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
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


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # we need position embedding, because in self-attention, the tokens have no idea where
        # they are in the sequence, since self-attention and aggregation can be thought of as
        # a directed graph where all previous nodes point to the next token node.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
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


model = GPT()
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
loss: ~1.46
trained for 15 minutes on A100

The top in a world by susphoring grace.

LUCIO:
We muse hath resistes him so sovere: son't his other wrough
stands of coverent sh'd: he has here, and stand it
and poor exceeder or a Henry's last, stay
not in faith, forewell's base of graves, thanks, happy comparel,
warmentfully: may as face by the courst, that strangth
errise hath breathed. Hastings come to Valenting.

HERMIONE:
Well have been bolly poor late
Is the lords.

ABELLA:
Let's found: I will kind him;
I do braw'sy him business wherein far his face.

LUCENTIO:
He is last afford: make him diseably to London,
Take him great Hastings, boldness in his natic keeps,
To oftragn lost me ready glust through the house.
Why chose that I dares it be a Montague.

MONTAGUE:
Woe's Claudly Haste of his own at last the Volscient,
And seen'd helpit: bearn to do it be, and most hop,
Miscause's more conterar than without this lambs
Shall down appla fortune flight flowers.

FRIAR LAUAURENCE:
His son, do your morself, that leaven your honours
Sufferable in more and suffer five.
A horse! High-graced York rights. And bother Montague
That the caapter, that I soughd him; such a chooson
Woes, that they have splight that care
Fades the respect to her spult: betfore him,
Un tell him up hine, or hope, that throw'st thou carry
apied sing with wear over the plenting long stamper
That doth butcherity. For love, what arful was an soldier
That last twain of all and Romeo runly Froth.

VALHASINA:
Nobleman; go, then both groans to us.

AUFIDIUS:
O those prepation!

AUFIDIUS:
It is: ever crimty be a house.

Second Citizen:
We give heed.

All Clarence, that makes not know work. The may say speak way.
How is my sorrow to strange on the fares
That which to play some called Margaret
The state town outward's wife, as the foul sleep;
Trickly of from thy blod'sty day blows here,
And pratess that chrospiles stalk falls up,
The world's hollow princhment, which should a bankind,
At till naKaina-daughter tae truth,
Craged from lares an oar that rems' stol-eat with blass.
Those is sometimes well call the Tale, the rod,
Submished his truth; Right states; but for ourselves,
Claud not thy hand, addingness.
No, there all conslent here pue the fault that yokUCHastisful
From servant and folling 'em how that: be drunk,
Set was halt be else, I will betwixt thee three with Tewar:
I am their man before a vile bad amiss'd
And thought have shorn'd the back-flowed of mine,
And ne'er than this, they leave spectiff.
I am to sure,
To maintain on what rash thy dam of suddise!
Thyself thee pays wither edge.
God I am speak to-morrow's like, to me speak,
Am
Dash your deliverance, nitted tongue to study.
But if you were could not love, if you such commands,
Your ignoration lightnifies
Sufficed hath granted a sacret
Divine: minute hath too should be assured,
Unless, heaven to themselvish, as I am,
Hance my father bend to them speak;
His the business' hath themselves;
For his ordance: bow his hand, hell pluck my pet!
What it brace there of his oath?
Rather? Where, whilst thou garling feet Bark? aim? stay;
So if He and him come, and make his mostake
You forbid you had stoopp'd your grace.

Servant:
He may once it indeces:
See do it between.

Provost:
Ah, sir! shall it stay the heavy nights.

PeRDINA:
Behind-foot, sir; three manner did he remiss
no slain up is disconful: slave you breast-wish.

HUMIO:
Why more lose on than ime well so fofter townd
you.

LUCIO:
How find, I must by our son?

PRINCE EDWARD:
Not I, sir?

PETHUMIO:
Base my fa-lor; I have ports to guilty:
It string is remorse: seldiers, thou retirest that Titus;
And I will have my close father- place:
I have well kings your husband; he will flow.

FRIAR:
I am nother for your highness' remain.
"""