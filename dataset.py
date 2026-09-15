import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from llm_library.bpe.bpe_training import train_bpe
from llm_library.bpe.tokenizer import Tokenizer


ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus" / "italian-books"
DATA = ROOT / "data" / "italian-books-bpe-v1"
TOKENIZER = ROOT / "tokenizer" / "italian-books-bpe-v1"
EOT = "<|endoftext|>"
VOCAB_SIZE = 2048


def split_documents():
    train, val, documents = [], [], []
    for path in sorted(CORPUS.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        split = max(1, int(len(text) * 0.9))
        train.append(text[:split] + EOT)
        val.append(text[split:] + EOT)
        documents.append({"path": path.name, "characters": len(text), "train_characters": split})
    if not train:
        raise FileNotFoundError(f"Nessun .txt in {CORPUS}")
    return train, val, documents


def encode_to_bin(tokenizer, text, path):
    tokens = np.asarray(tokenizer.encode(text), dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(path)
    return int(tokens.size)


def prepare():
    train, val, documents = split_documents()
    TOKENIZER.mkdir(parents=True, exist_ok=True)
    train_text = DATA / "train-for-bpe.txt"
    train_text.parent.mkdir(parents=True, exist_ok=True)
    train_text.write_text("".join(train), encoding="utf-8")
    vocab, merges = train_bpe(train_text, VOCAB_SIZE, [EOT])
    with (TOKENIZER / "vocab.pkl").open("wb") as handle:
        pickle.dump(vocab, handle)
    with (TOKENIZER / "merges.pkl").open("wb") as handle:
        pickle.dump(merges, handle)
    tokenizer = Tokenizer(vocab, merges, [EOT])
    manifest = {
        "version": "italian-books-bpe-v1",
        "vocab_size": VOCAB_SIZE,
        "special_tokens": [EOT],
        "documents": documents,
        "train_tokens": encode_to_bin(tokenizer, "".join(train), DATA / "train.bin"),
        "val_tokens": encode_to_bin(tokenizer, "".join(val), DATA / "val.bin"),
    }
    (TOKENIZER / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
