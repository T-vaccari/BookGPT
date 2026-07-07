import tempfile
import unittest
from pathlib import Path

import torch

from checkpoint_utils import format_model_specs, list_checkpoints, resolve_checkpoint


CONFIG = {
    "vocab_size": 140,
    "n_embd": 256,
    "n_head": 8,
    "block_size": 256,
    "n_blocks": 5,
    "dropout": 0.15,
}


class CheckpointUtilsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def save(self, run, filename, step):
        directory = self.root / run
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        torch.save({"config": CONFIG, "step": step, "best_val_loss": 1.25}, path)
        return path

    def test_lists_only_trained_best_checkpoints(self):
        self.save("trained", "best.pt", 100)
        self.save("untrained", "best.pt", 0)
        self.save("resume-only", "last.pt", 200)

        checkpoints = list_checkpoints(self.root)

        self.assertEqual([item["name"] for item in checkpoints], ["trained"])
        self.assertEqual(checkpoints[0]["step"], 100)

    def test_resolves_run_name_by_purpose(self):
        best = self.save("trained", "best.pt", 100)
        last = self.save("trained", "last.pt", 120)

        self.assertEqual(resolve_checkpoint("trained", self.root, "inference"), best)
        self.assertEqual(resolve_checkpoint("trained", self.root, "resume"), last)

    def test_accepts_direct_checkpoint_path(self):
        path = self.save("trained", "custom.pt", 100)
        self.assertEqual(resolve_checkpoint(str(path), self.root, "inference"), path)

    def test_formats_embedded_model_specs(self):
        checkpoint = {"config": CONFIG, "step": 100, "best_val_loss": 1.25}
        output = format_model_specs(checkpoint, 12_345, "cpu")
        self.assertIn("context=256", output)
        self.assertIn("embedding=256", output)
        self.assertIn("heads=8", output)
        self.assertIn("layers=5", output)
        self.assertIn("parameters=12,345", output)
        self.assertIn("step=100", output)
        self.assertIn("best_val_loss=1.2500", output)
        self.assertIn("device=cpu", output)


if __name__ == "__main__":
    unittest.main()
