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
        self.assertGreaterEqual(self.source.count("--val-txt    {PART_VAL_TXT}"), 4)
        self.assertGreaterEqual(self.source.count("--output     {OUTPUT}"), 4)

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
        upload_section = self.source.split("CELL 11", 1)[-1]
        self.assertNotIn("--public", upload_section)

    def test_notebook_is_valid_and_contains_the_shard_contract(self):
        notebook_path = RUNBOOK.with_name("kaggle_run.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        for token in ("START_INDEX", "END_INDEX", "kaggle datasets create", "--dir-mode zip"):
            with self.subTest(token=token):
                self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
