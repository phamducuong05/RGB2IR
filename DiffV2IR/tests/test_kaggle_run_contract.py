import unittest
import re
import json
from pathlib import Path


RUNBOOK = Path(__file__).resolve().parents[1] / "kaggle_run.py"


class KaggleRunContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNBOOK.read_text(encoding="utf-8")

    def test_exposes_half_open_range_configuration(self):
        for pattern in (
            r"START_INDEX\s*=",
            r"END_INDEX\s*=",
            r"DATASET_SLUG_PREFIX\s*=",
            r"KAGGLE_USERNAME_SECRET\s*=",
            r"KAGGLE_KEY_SECRET\s*=",
            r"select_key_range\(",
            r"PART_VAL_TXT",
            r"PART_OUTPUT",
        ):
            with self.subTest(pattern=pattern):
                self.assertRegex(self.source, re.compile(pattern))

    def test_range_examples_are_documented(self):
        self.assertIn("[0, 1714)", self.source)
        self.assertIn("[1714, 3428)", self.source)
        self.assertIn("[3428, 5142)", self.source)

    def test_inference_commands_use_the_selected_subset(self):
        self.assertNotIn("--val-txt    {INPUT_FLIR}/align_validation.txt", self.source)
        self.assertEqual(self.source.count("--val-txt    {PART_VAL_TXT}"), 1)
        self.assertEqual(self.source.count("--output     {OUTPUT}"), 1)

    def test_runs_only_one_selected_range_inference(self):
        self.assertEqual(
            self.source.count('sh(f"""python infer_flir.py'),
            1,
        )
        self.assertNotIn("--limit 5", self.source)
        self.assertNotIn("--limit 20", self.source)
        self.assertNotIn("--metrics-only", self.source)

    def test_accepts_an_attached_weights_dataset(self):
        self.assertRegex(
            self.source,
            r'WEIGHTS_DATASET_DIR\s*=\s*"/kaggle/input/diffv2ir-model-weights-v1"',
        )
        for token in (
            '"FLIR.ckpt"',
            '"blip-image-captioning-base"',
            '"clip-vit-large-patch14"',
            "WEIGHTS_DATASET_DIR",
            "CLIP_MODEL",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_inference_commands_pass_the_selected_clip_model(self):
        self.assertEqual(self.source.count("--clip-version {CLIP_MODEL}"), 1)

    def test_packaging_has_an_exact_prediction_gate(self):
        for token in (
            "validate_predictions(",
            "missing_prediction_keys",
            "extra_prediction_files",
            "manifest.json",
            "dataset-metadata.json",
            "PACKAGE_DIR",
        ):
            with self.subTest(token=token):
                self.assertIn(token, self.source)

    def test_upload_is_private_and_uses_kaggle_directory_zip(self):
        self.assertIn("write_kaggle_credentials(", self.source)
        self.assertIn("kaggle datasets create", self.source)
        self.assertIn("--dir-mode zip", self.source)
        upload_section = self.source.split("CELL 8", 1)[-1]
        self.assertNotIn("--public", upload_section)

    def test_notebook_is_valid_and_contains_the_shard_contract(self):
        notebook_path = RUNBOOK.with_name("kaggle_run.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for token in (
            "START_INDEX",
            "END_INDEX",
            "WEIGHTS_DATASET_DIR",
            "--clip-version {CLIP_MODEL}",
            "kaggle datasets create",
            "--dir-mode zip",
        ):
            with self.subTest(token=token):
                self.assertIn(token, source)
        self.assertEqual(source.count("python infer_flir.py"), 1)
        self.assertNotIn("--limit 5", source)
        self.assertNotIn("--limit 20", source)


if __name__ == "__main__":
    unittest.main()
