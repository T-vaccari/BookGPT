from pathlib import Path

import torch


def load_checkpoint(path, device="cpu"):
    return torch.load(Path(path), map_location=device, weights_only=True)


def list_checkpoints(root):
    root = Path(root)
    checkpoints = []
    for path in sorted(root.glob("*/best.pt")):
        checkpoint = load_checkpoint(path)
        step = int(checkpoint.get("step", 0))
        if step <= 0:
            continue
        checkpoints.append({
            "name": path.parent.name,
            "path": path,
            "step": step,
            "best_val_loss": float(checkpoint["best_val_loss"]),
            "config": checkpoint["config"],
        })
    return checkpoints


def resolve_checkpoint(value, root, purpose):
    path = Path(value).expanduser()
    if path.is_file():
        return path
    filename = "best.pt" if purpose == "inference" else "last.pt"
    path = Path(root) / value / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")
    return path


def format_model_specs(checkpoint, parameter_count, device):
    config = checkpoint["config"]
    loss = float(checkpoint["best_val_loss"])
    return (
        f"context={config['block_size']} | embedding={config['n_embd']} | "
        f"heads={config['n_head']} | layers={config['n_blocks']} | "
        f"dropout={config['dropout']} | vocab={config['vocab_size']}\n"
        f"parameters={parameter_count:,} | step={int(checkpoint['step'])} | "
        f"best_val_loss={loss:.4f} | device={device}"
    )
