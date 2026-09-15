import argparse
from pathlib import Path

import torch

from llm_library.bpe.tokenizer import Tokenizer
from model import build_model


ROOT = Path(__file__).parent
TOKENIZER = ROOT / "tokenizer" / "italian-books-bpe-v1"
EOT = "<|endoftext|>"
PROMPTS = ["Il", "La storia", "Nella notte", "Quando"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/llm-library-bpe-5m/best.pt")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", action="append")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    tokenizer = Tokenizer.from_files(TOKENIZER / "vocab.pkl", TOKENIZER / "merges.pkl", [EOT])
    model = build_model(checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    eot_id = tokenizer.reversed_vocab[EOT.encode("utf-8")]
    for prompt in args.prompt or PROMPTS:
        ids = tokenizer.encode(prompt)
        for _ in range(args.tokens):
            x = torch.tensor([ids[-checkpoint["config"]["context_length"]:]], device=device)
            with torch.no_grad():
                token = int(torch.argmax(model(x)[0, -1]).item())
            if token == eot_id:
                break
            ids.append(token)
        print(tokenizer.decode(ids))
        print("\n---\n")


if __name__ == "__main__":
    main()
