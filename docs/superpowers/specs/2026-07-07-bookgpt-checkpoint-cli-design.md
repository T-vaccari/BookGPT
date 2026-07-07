# BookGPT checkpoint CLI design

## Scope

Keep the repository focused on model training and inference. Raw books, converted
text, prepared binary datasets, conversion scripts, resumable checkpoints, and
the untrained step-zero run remain local and ignored.

## Repository layout

- `model.py`: model architecture.
- `train.py`: new training and checkpoint resume CLI.
- `generate.py`: checkpoint discovery and interactive inference CLI.
- `checkpoint_utils.py`: shared checkpoint discovery, loading, naming, and
  model-spec formatting.
- `checkpoints/<run-name>/best.pt`: tracked inference checkpoint.
- `checkpoints/<run-name>/last.pt`: ignored resumable checkpoint.
- `tests/`: checkpoint CLI tests using small synthetic checkpoints.

The two trained runs are renamed from their embedded configuration:

- `it-ctx256-d256-h8-l5`
- `it-ctx512-d256-h8-l4`

## CLI behavior

`generate.py --list-checkpoints` lists available trained runs.
`generate.py --checkpoint <name>` loads that run's `best.pt`. A direct
checkpoint path is also accepted. Startup prints architecture, parameter count,
context size, step, validation loss, and device.

`train.py --resume <name-or-path>` restores model, optimizer, scaler, step,
and best validation loss from `last.pt` when a run name is supplied. Model
architecture always comes from the checkpoint to prevent incompatible resumes.

## Versioning

Git tracks source, tests, and the two trained `best.pt` files using normal Git.
It ignores `last.pt`, the step-zero run, corpora, ODT files, prepared datasets,
`prepare.py`, `convert_odt.py`, caches, and temporary scripts.

Historical batch size and learning rate cannot be reconstructed reliably because
the old checkpoints did not store them. Only embedded architecture, step, loss,
optimizer, and scaler state are treated as authoritative.
