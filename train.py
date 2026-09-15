import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import build_model


ROOT = Path(__file__).parent
RUN = ROOT / "checkpoints" / "llm-library-bpe-5m"
DATA = ROOT / "data" / "italian-books-bpe-v1"
MODEL_CONFIG = {
    "vocab_size": 2048,
    "context_length": 256,
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 4,
    "d_ff": 1024,
    "rope_theta": 10000.0,
}


def load_tokens(split):
    return np.memmap(DATA / f"{split}.bin", dtype=np.uint16, mode="r")


def batch(tokens, batch_size, device):
    starts = torch.randint(len(tokens) - MODEL_CONFIG["context_length"] - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(np.asarray(tokens[i:i + MODEL_CONFIG["context_length"]], dtype=np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(np.asarray(tokens[i + 1:i + MODEL_CONFIG["context_length"] + 1], dtype=np.int64)) for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def validation_loss(model, tokens, batch_size, device, batches=32):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = batch(tokens, batch_size, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def save(path, model, optimizer, step, best_val_loss, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
        "best_val_loss": best_val_loss, "config": MODEL_CONFIG, "train_args": vars(args),
    }, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(1337)
    train_tokens, val_tokens = load_tokens("train"), load_tokens("val")
    model = build_model(MODEL_CONFIG, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    start_step, best_val_loss = 0, float("inf")
    resume_path = RUN / "last.pt" if args.resume == "auto" else Path(args.resume)
    if args.resume != "none" and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step, best_val_loss = int(checkpoint["step"]), float(checkpoint["best_val_loss"])
        print(f"resumed step={start_step} best_val_loss={best_val_loss:.4f}", flush=True)
    print(f"parameters={parameter_count} device={device} train_tokens={len(train_tokens)} val_tokens={len(val_tokens)}", flush=True)
    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(args.grad_accumulation):
            x, y = batch(train_tokens, args.batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / args.grad_accumulation
            loss.backward()
            loss_value += loss.item()
        progress = step / max(1, args.max_steps - 1)
        optimizer.param_groups[0]["lr"] = args.learning_rate * 0.1 + args.learning_rate * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed = step + 1
        if completed % args.eval_every == 0 or completed == args.max_steps:
            value = validation_loss(model, val_tokens, args.batch_size, device)
            print(json.dumps({"step": completed, "train_loss": loss_value, "val_loss": value}), flush=True)
            if value < best_val_loss:
                best_val_loss = value
                save(RUN / "best.pt", model, optimizer, completed, best_val_loss, args)
        if completed % args.checkpoint_every == 0 or completed == args.max_steps:
            save(RUN / "last.pt", model, optimizer, completed, best_val_loss, args)


if __name__ == "__main__":
    main()
