import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from contextlib import nullcontext
import math
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from checkpoint_utils import format_model_specs, list_checkpoints, load_checkpoint
from model import GPT


# ## Hyperparameters --##
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
BLOCK_SIZE = 1024
N_EMBD = 384
N_HEAD = 12
N_BLOCKS = 8
DROPOUT = 0.15

MAX_ITERS = 10000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WARMUP_STEPS = 300
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250
EVAL_ITERS = 25
EVAL_BATCH_SIZE = 16


def choose_training():
    checkpoint_root = "checkpoints"
    runs = [
        checkpoint for checkpoint in list_checkpoints(checkpoint_root)
        if (checkpoint["path"].parent / "last.pt").is_file()
    ]

    print("0. Nuovo training")
    for index, checkpoint in enumerate(runs, 1):
        config = checkpoint["config"]
        print(
            f"{index}. Riprendi {checkpoint['name']} | step {checkpoint['step']} | "
            f"context {config['block_size']} | embedding {config['n_embd']}"
        )

    while True:
        try:
            choice = int(input("> "))
            if choice == 0:
                run_name = (
                    f"it-ctx{BLOCK_SIZE}-d{N_EMBD}-h{N_HEAD}-l{N_BLOCKS}"
                )
                resume = None
                break
            elif 1 <= choice <= len(runs):
                run_name = None
                resume = runs[choice - 1]["name"]
                break
        except ValueError:
            pass
        print("Scelta non valida")

    return SimpleNamespace(
        run_name=run_name, resume=resume, checkpoint_root=checkpoint_root,
        data_dir="data/italian-books", device="cuda",
        attention_implementation="gfx1010",
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        block_size=BLOCK_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
        n_blocks=N_BLOCKS, dropout=DROPOUT, max_iters=MAX_ITERS,
        learning_rate=LEARNING_RATE, min_learning_rate=MIN_LEARNING_RATE,
        warmup_steps=WARMUP_STEPS, adamw_betas=ADAMW_BETAS,
        adamw_eps=ADAMW_EPS, weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP, eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS, eval_batch_size=EVAL_BATCH_SIZE,
    )


def model_config(args, vocab_size, checkpoint=None):
    if checkpoint is not None:
        return checkpoint["config"]
    return {
        "vocab_size": vocab_size, "n_embd": args.n_embd,
        "n_head": args.n_head, "block_size": args.block_size,
        "n_blocks": args.n_blocks, "dropout": args.dropout,
    }


def main():
    args = choose_training()
    torch.manual_seed(1337)

    checkpoint = None
    best_checkpoint = None
    if args.resume:
        checkpoint_dir = os.path.join(args.checkpoint_root, args.resume)
        checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
        checkpoint = load_checkpoint(checkpoint_path, args.device)
        best_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.isfile(best_path):
            best_checkpoint = load_checkpoint(best_path, args.device)
    else:
        checkpoint_dir = os.path.join(args.checkpoint_root, args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as file:
        meta = pickle.load(file)
    vocab_size = meta["vocab_size"]
    config = model_config(args, vocab_size, checkpoint)
    if config["vocab_size"] != vocab_size:
        raise ValueError("Il vocabolario non corrisponde al checkpoint selezionato")

    dtype = np.uint8 if vocab_size <= 256 else np.uint16
    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=dtype, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=dtype, mode="r")
    block_size = config["block_size"]

    def get_batch(split, batch_size):
        data = train_data if split == "train" else val_data
        indices = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in indices])
        y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in indices])
        return x.to(args.device), y.to(args.device)

    model = GPT(
        **config,
        attention_implementation=args.attention_implementation,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=args.adamw_betas,
        eps=args.adamw_eps,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))

    start_step = 0
    best_val_loss = float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        for group in optimizer.param_groups:
            group["betas"] = args.adamw_betas
            group["eps"] = args.adamw_eps
            group["weight_decay"] = args.weight_decay
        start_step = int(checkpoint["step"]) + 1
        losses = [float(checkpoint["best_val_loss"])]
        if best_checkpoint is not None:
            losses.append(float(best_checkpoint["best_val_loss"]))
        best_val_loss = min(losses)
        print(f"Ripresa da: {checkpoint_path}")

    specs = dict(checkpoint) if checkpoint is not None else {"config": config, "step": 0}
    specs["best_val_loss"] = best_val_loss
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(format_model_specs(specs, parameters, args.device))

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=torch.float16) if args.device.startswith("cuda") else nullcontext()

    def learning_rate_at(step):
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / (
            args.max_iters - args.warmup_steps
        )
        coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            args.min_learning_rate
            + coefficient * (args.learning_rate - args.min_learning_rate)
        )

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        output = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for index in range(args.eval_iters):
                x, y = get_batch(split, args.eval_batch_size)
                with autocast_context():
                    _, loss = model(x, y)
                losses[index] = loss.item()
            output[split] = losses.mean()
        model.train()
        return output

    def save_checkpoint(path, step, current_best_loss):
        temporary_path = f"{path}.tmp"
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "step": step,
            "best_val_loss": current_best_loss, "config": config,
            "training_config": {
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_iters": args.max_iters,
                "eval_interval": args.eval_interval, "eval_iters": args.eval_iters,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "min_learning_rate": args.min_learning_rate,
                "warmup_steps": args.warmup_steps,
                "adamw_betas": args.adamw_betas,
                "adamw_eps": args.adamw_eps,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "attention_implementation": args.attention_implementation,
            },
        }, temporary_path)
        os.replace(temporary_path, path)

    for step in range(start_step, args.max_iters):
        if step % args.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = float(losses["val"])
                save_checkpoint(os.path.join(checkpoint_dir, "best.pt"), step, best_val_loss)
                print(f"best val {best_val_loss:.4f} salvato")
            save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), step, best_val_loss)

        learning_rate = learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation_steps):
            x, y = get_batch("train", args.batch_size)
            with autocast_context():
                _, loss = model(x, y)
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    print("Done.")


if __name__ == "__main__":
    main()
