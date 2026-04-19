import torch
import torch.nn as nn
from torch.nn import functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(63459439)
torch.set_float32_matmul_precision('high')

# hyper=parameters
block_size = 64
batch_size = 256
training_loops = 1000
n_embed = 600
head_size = 16
learning_rate = 3e-4
B, T, C = 4, 8, 32
n_layer = 6
n_head = 6
dropout = 0.3

with open("wizard of oz.txt", 'r', encoding='utf-8') as f:
    text = f.read()

int_to_char = sorted(set(text))
char_to_int = { char:i for i, char in enumerate(int_to_char) }
vocab_size = len(int_to_char)

#used to encode and decode text
def encode(s):
    res = []
    for i in s:
        res.append(char_to_int[i])
    return res

def decode(arr):
    res = ""
    for i in arr:
        res += int_to_char[i]
    return res


#split the data into train and test
data = torch.tensor(encode(text), dtype=torch.long)

split = int(len(data) * 0.8)
val_data = data[split:].to(device=device)
train_data = data[:split].to(device=device)

    

#gets random batch of data from the text
def get_batch(batch_type: str):
    data = val_data if batch_type == 'test' else train_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #B, T, C
        q = self.query(x) #B, T, C

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 #scaled
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), 
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), 
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tkn_emd = self.token_embedding_table(idx) # (B, T, C) C in this case is n_embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tkn_emd + pos_emb # (B, T, C)
        x = self.blocks(x) # B, T, C
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else: 
            loss = None
        
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx tot the last block_size token
            logits, loss = self(idx[:, -block_size:])
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
# instantiate model
model = BigramModel()
model=model.to(device)
model = torch.compile(model)
xb, yb = get_batch('train')
xb, yb = xb.to(device), yb.to(device)
logits, loss = model(xb, yb)


#initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

#training loop
for i in range(training_loops):
    # get batch of data
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    # get the loss
    logits, loss = model(xb, yb)
    #zero out gradients from prev step
    optimizer.zero_grad(set_to_none=True)
    #getting the loss for all params
    loss.backward()
    # using the gradient to update params
    optimizer.step()

print('loss:', loss.item())


#used to kick off generation
idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    
#generate text after training
generated = decode(model.generate(idx, max_new_tokens=10000)[0].tolist())

# write generated text to file
with open("example.txt", "w") as f:
    f.write(generated)