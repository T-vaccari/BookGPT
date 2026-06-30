import os
# must be set before CUDA context initializes — fixes allocator fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import pickle
import numpy as np
import torch
from model import GPT

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
batch_size    = 128           
block_size    = 256
max_iters     = 10000
eval_interval = 250
eval_iters    = 100
learning_rate = 3e-4
n_embd        = 128
n_head        = 4
n_blocks      = 4
dropout       = 0.1          
device        = 'cuda'
ckpt_dir      = 'checkpoints'
data_dir      = 'data'
RESUME = False
# ---------------------------------------------------------------------------

torch.manual_seed(1337)
os.makedirs(ckpt_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Vocab — loaded from meta.pkl, built by prepare.py from the full corpus
# ---------------------------------------------------------------------------
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos, vocab_size = meta['stoi'], meta['itos'], meta['vocab_size']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)
print(f"vocab_size: {vocab_size}")

# ---------------------------------------------------------------------------
# Data — memory mapped, dtype derived from vocab_size to match prepare.py
# ---------------------------------------------------------------------------
dtype      = np.uint8 if vocab_size <= 256 else np.uint16
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=dtype, mode='r')
val_data   = np.memmap(os.path.join(data_dir, 'val.bin'),   dtype=dtype, mode='r')
print(f"train: {len(train_data):,} chars | val: {len(val_data):,} chars")


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))       for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model, optimizer, fp16 scaler
# ---------------------------------------------------------------------------
model     = GPT(vocab_size, n_embd, n_head, block_size, n_blocks, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler    = torch.cuda.amp.GradScaler()   # fp16 mixed precision; bf16 unreliable on gfx1010, torch.amp.GradScaler not exported on this dev build

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(path, step, best_val_loss):
    tmp = path + '.tmp'
    torch.save({
        'model':         model.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'scaler':        scaler.state_dict(),
        'step':          step,
        'best_val_loss': best_val_loss,
        'config': {
            'vocab_size': vocab_size,
            'n_embd':     n_embd,
            'n_head':     n_head,
            'block_size': block_size,
            'n_blocks':   n_blocks,
            'dropout':    dropout,
        },
    }, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_loss = float('inf')
start_step    = 0

if RESUME:
    ckpt = torch.load(f'{ckpt_dir}/ckpt_last.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_step    = ckpt['step']
    best_val_loss = ckpt['best_val_loss']
    print(f"Resuming from step {start_step} | best val {best_val_loss:.4f}")

for step in range(start_step, max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        save_checkpoint(f'{ckpt_dir}/ckpt_last.pt', step, best_val_loss)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            save_checkpoint(f'{ckpt_dir}/ckpt_best.pt', step, best_val_loss)
            print(f"            → best val {best_val_loss:.4f} saved")

    xb, yb = get_batch('train')

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

print("Done.")