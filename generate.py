import os
import pickle

import torch

from checkpoint_utils import format_model_specs, list_checkpoints, load_checkpoint
from model import GPT


CHECKPOINT_ROOT = "checkpoints"
DATA_DIR = "data/italian-books"
DEVICE = "cuda"


def choose_checkpoint():
    checkpoints = list_checkpoints(CHECKPOINT_ROOT)
    if not checkpoints:
        raise FileNotFoundError("Nessun checkpoint addestrato disponibile")

    print("Scegli il modello:")
    for index, checkpoint in enumerate(checkpoints, 1):
        config = checkpoint["config"]
        print(
            f"{index}. {checkpoint['name']} | step {checkpoint['step']} | "
            f"loss {checkpoint['best_val_loss']:.4f} | context {config['block_size']} | "
            f"embedding {config['n_embd']} | {config['n_blocks']} layer"
        )

    while True:
        try:
            return checkpoints[int(input("> ")) - 1]["path"]
        except (ValueError, IndexError):
            print("Scelta non valida")


def main():
    checkpoint_path = choose_checkpoint()
    checkpoint = load_checkpoint(checkpoint_path, DEVICE)
    config = checkpoint["config"]

    with open(os.path.join(DATA_DIR, "meta.pkl"), "rb") as file:
        meta = pickle.load(file)
    stoi, itos = meta["stoi"], meta["itos"]
    if meta["vocab_size"] != config["vocab_size"]:
        raise ValueError("Il vocabolario non corrisponde al checkpoint selezionato")

    def encode(text):
        try:
            return [stoi[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Carattere {error} assente dal vocabolario") from error

    def decode(tokens):
        return "".join(itos[token] for token in tokens)

    model = GPT(**config).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Checkpoint: {checkpoint_path}")
    print(format_model_specs(checkpoint, parameter_count, DEVICE))

    max_new_tokens = 500
    temperature = 1.0
    print("Comandi: /tokens N   /temp X   /quit\n")

    while True:
        context = input(">>> ").strip()
        if context in ("/quit", "/exit"):
            break
        if context.startswith("/tokens "):
            try:
                max_new_tokens = int(context.split(" ", 1)[1])
                print(f"max_new_tokens = {max_new_tokens}")
            except ValueError:
                print("Uso: /tokens 500")
            continue
        if context.startswith("/temp "):
            try:
                temperature = float(context.split(" ", 1)[1])
                print(f"temperature = {temperature}")
            except ValueError:
                print("Uso: /temp 0.8")
            continue
        if not context:
            context = "\n"

        try:
            indices = torch.tensor([encode(context)], dtype=torch.long, device=DEVICE)
        except ValueError as error:
            print(error)
            continue

        with torch.no_grad():
            output = model.generate(indices, max_new_tokens, temperature=temperature)
        print(decode(output[0].tolist()))
        print()


if __name__ == "__main__":
    main()
