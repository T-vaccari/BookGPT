# BookGPT Checkpoint CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize BookGPT and add explicit checkpoint selection for inference and resumable training.

**Architecture:** Keep the small repository flat. Put checkpoint discovery and metadata formatting in `checkpoint_utils.py`; `generate.py` and `train.py` remain executable scripts and consume that shared API.

**Tech Stack:** Python 3, PyTorch/ROCm, argparse, unittest.

---

### Task 1: Checkpoint discovery

**Files:**
- Create: `checkpoint_utils.py`
- Create: `tests/test_checkpoint_utils.py`

- [ ] Write tests proving that run names resolve to `best.pt` for inference, to `last.pt` for resume, direct paths remain valid, and only trained `best.pt` files are listed.
- [ ] Run `python -m unittest tests.test_checkpoint_utils -v` and confirm the missing module failure.
- [ ] Implement `list_checkpoints(root)`, `resolve_checkpoint(value, root, purpose)`, `load_checkpoint(path, device)`, and `format_model_specs(checkpoint, parameter_count, device)`.
- [ ] Re-run the tests and confirm they pass.

### Task 2: Inference CLI

**Files:**
- Modify: `generate.py`
- Create: `tests/test_generate_cli.py`

- [ ] Write subprocess tests for `--list-checkpoints`, required `--checkpoint`, and argument parsing without starting the interactive loop.
- [ ] Run the focused tests and confirm failure because the CLI does not exist.
- [ ] Add `--checkpoint`, `--list-checkpoints`, `--data-dir`, `--device`, `--max-new-tokens`, and `--temperature`; load architecture from checkpoint metadata and print complete model specifications.
- [ ] Re-run the focused tests and confirm they pass.

### Task 3: Resume CLI

**Files:**
- Modify: `train.py`
- Create: `tests/test_train_cli.py`

- [ ] Write tests proving `--resume <run>` chooses `last.pt`, checkpoint architecture overrides new-run architecture, and CLI parsing does not initialize CUDA during import.
- [ ] Run the focused tests and confirm failure.
- [ ] Move execution under `main()`, add `--run-name`, `--resume`, `--data-dir`, `--device`, and training hyperparameter flags; restore model, optimizer, scaler, step, and best validation loss on resume.
- [ ] Save all training hyperparameters in future checkpoint metadata and print model specifications before training.
- [ ] Re-run the focused tests and confirm they pass.

### Task 4: Repository layout and checkpoint migration

**Files:**
- Modify: `.gitignore`
- Move: `checkpoints_it_nativo/context1/ckpt_best.pt` to `checkpoints/it-ctx256-d256-h8-l5/best.pt`
- Move: `checkpoints_it_nativo/context1/ckpt_last.pt` to `checkpoints/it-ctx256-d256-h8-l5/last.pt`
- Move: `checkpoints_it_nativo/context512/ckpt_best.pt` to `checkpoints/it-ctx512-d256-h8-l4/best.pt`
- Move: `checkpoints_it_nativo/context512/ckpt_last.pt` to `checkpoints/it-ctx512-d256-h8-l4/last.pt`

- [ ] Ignore corpora, ODT files, prepared datasets, `prepare.py`, `convert_odt.py`, temporary scripts, caches, all `last.pt`, and the old step-zero run.
- [ ] Move the two trained runs without changing checkpoint bytes.
- [ ] Confirm `git status --short --ignored` shows only source, tests, documentation, and two `best.pt` files as versionable.

### Task 5: Verification

**Files:**
- No new files.

- [ ] Run `python -m unittest discover -s tests -v` in the `ml` Conda environment.
- [ ] Run `python generate.py --list-checkpoints` and verify both trained runs and their metadata.
- [ ] Load each `best.pt` on CPU, instantiate `GPT`, and verify `load_state_dict` succeeds.
- [ ] Compare hashes before and after migration.
- [ ] Review `git diff --check`, `git status --short`, and staged file sizes before committing.
