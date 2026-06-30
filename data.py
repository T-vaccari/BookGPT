import os, glob, pickle
import numpy as np

in_dir      = 'file'
exclude     = {'shakespeare.txt'}
out_dir     = 'data'
val_frac    = 0.1
chunk_chars = 1_000_000

paths = sorted(
    p for p in glob.glob(os.path.join(in_dir, '*.txt'))
    if os.path.basename(p) not in exclude
)
assert paths, f"no .txt files found in {in_dir}/"

texts = []
for p in paths:
    with open(p, 'r', encoding='utf-8') as f:
        texts.append((os.path.basename(p), f.read()))
    print(f"  {texts[-1][0]:45s} {len(texts[-1][1]):>10,} chars")

full_text = "\n\n".join(t for _, t in texts)
print(f"\ncombined corpus: {len(full_text):,} chars from {len(paths)} files")

# vocab from everything, before any splitting
chars      = sorted(list(set(full_text)))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}
print(f"vocab: {vocab_size} unique chars")

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

# split EACH file individually, then concatenate the pieces —
# see note below for why this matters
train_parts, val_parts = [], []
for _, t in texts:
    n_val = int(len(t) * val_frac)
    train_parts.append(t[:-n_val])
    val_parts.append(t[-n_val:])

train_text = "\n\n".join(train_parts)
val_text   = "\n\n".join(val_parts)

dtype = np.uint8 if vocab_size <= 256 else np.uint16

def write_bin(text, path):
    arr = np.memmap(path, dtype=dtype, mode='w+', shape=(len(text),))
    for start in range(0, len(text), chunk_chars):
        chunk = text[start:start + chunk_chars]
        arr[start:start + len(chunk)] = [stoi[c] for c in chunk]
    arr.flush()

write_bin(train_text, os.path.join(out_dir, 'train.bin'))
write_bin(val_text,   os.path.join(out_dir, 'val.bin'))

print(f"\ntrain: {len(train_text):,} chars | val: {len(val_text):,} chars | dtype: {dtype}")