import os
import pickle
import torch
from model import GPT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
device          = 'cuda'
ckpt_dir        = 'checkpoints'
data_dir        = 'data'
max_new_tokens  = 500
temperature     = 1.0
# ---------------------------------------------------------------------------

# Vocab — stesso meta.pkl del training, mai ricostruito da zero
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def encode(s):
    try:
        return [stoi[c] for c in s]
    except KeyError as e:
        raise ValueError(
            f"Carattere {e} assente dal vocabolario ({len(stoi)} simboli). Rimuovilo dal contesto."
        )

decode = lambda l: ''.join(itos[i] for i in l)

# Miglior checkpoint — train.py sovrascrive ckpt_best.pt solo quando il val
# loss migliora, quindi è già "il migliore disponibile" per costruzione
ckpt   = torch.load(os.path.join(ckpt_dir, 'ckpt_best.pt'), map_location=device, weights_only=True)
config = ckpt['config']

model = GPT(
    vocab_size = config['vocab_size'],
    n_embd     = config['n_embd'],
    n_head     = config['n_head'],
    block_size = config['block_size'],
    n_blocks   = config['n_blocks'],
    dropout    = config['dropout'],
).to(device)

model.load_state_dict(ckpt['model'])
model.eval()   # disattiva dropout — senza questo si genera con rumore casuale ancora attivo

print(f"Checkpoint: step {ckpt['step']} | best val loss {ckpt['best_val_loss']:.4f}")
print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")
print(f"max_new_tokens={max_new_tokens}  temperature={temperature}")
print("Comandi: /tokens N   /temp X   /quit\n")


# ---------------------------------------------------------------------------
# Generazione — loop manuale, NON model.generate()
# (model.generate() si fermerebbe al primo '\n', sbagliato per prosa libera)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(idx, max_new_tokens, temperature):
    for _ in range(max_new_tokens):
        idx_cond  = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits    = logits[:, -1, :]

        if temperature <= 0:
            idx_next = logits.argmax(dim=-1, keepdim=True)   # greedy — nessuna divisione, nessuna softmax
        else:
            logits   = logits / temperature
            probs    = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ---------------------------------------------------------------------------
# Loop interattivo
# ---------------------------------------------------------------------------
while True:
    context = input(">>> ").strip()

    if context in ('/quit', '/exit'):
        break

    if context.startswith('/tokens '):
        try:
            max_new_tokens = int(context.split(' ', 1)[1])
            print(f"max_new_tokens = {max_new_tokens}")
        except ValueError:
            print("Uso: /tokens 500")
        continue

    if context.startswith('/temp '):
        try:
            temperature = float(context.split(' ', 1)[1])
            print(f"temperature = {temperature}")
        except ValueError:
            print("Uso: /temp 0.8")
        continue

    if context == '':
        context = '\n'   # serve almeno un carattere per avviare la generazione

    try:
        idx = torch.tensor([encode(context)], dtype=torch.long, device=device)
    except ValueError as e:
        print(e)
        continue

    out = generate(idx, max_new_tokens, temperature)
    print(decode(out[0].tolist()))
    print()