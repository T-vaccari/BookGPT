#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$ROOT/checkpoints/llm-library-bpe-5m"
LOG="$RUN/training.log"
PYTHON="${PYTHON:-/home/tom/miniforge3/envs/ml-gfx1010-rc/bin/python}"
mkdir -p "$RUN"

if [[ ! -f "$ROOT/tokenizer/italian-books-bpe-v1/vocab.pkl" ]]; then
  "$PYTHON" dataset.py --prepare
fi
"$PYTHON" train.py --resume auto 2>&1 | tee -a "$LOG"
"$PYTHON" generate.py --checkpoint checkpoints/llm-library-bpe-5m/best.pt 2>&1 | tee -a "$LOG"
touch "$RUN/COMPLETED"

if [[ "${SHUTDOWN_ON_SUCCESS:-0}" == "1" ]]; then
  sudo -n /usr/bin/systemctl poweroff
fi
