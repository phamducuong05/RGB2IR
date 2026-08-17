import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kaggle_shards import (
    build_dataset_metadata,
    build_manifest,
    select_key_range,
    validate_predictions,
    write_kaggle_credentials,
)


class SelectKeyRangeTests(unittest.TestCase):
    def test_uses_a_half_open_interval(self):
        keys = ["a", "b", "c", "d"]
        self.assertEqual(select_key_range(keys, 1, 3), ["b", "c"])

    def test_rejects_invalid_bounds(self):
        keys = ["a", "b"]
        for start, end in [(-1, 1), (1, 1), (2, 1), (0, 3)]:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    select_key_range(keys, start, end)


class PredictionValidationTests(unittest.TestCase):
    def test_requires_exact_prediction_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            prediction_dir = Path(tmp)
            (prediction_dir / "a_pred.png").touch()
            (prediction_dir / "unexpected_pred.png").touch()

            missing, extra = validate_predictions(["a", "b"], prediction_dir)

            self.assertEqual(missing, {"b_pred.png"})
            self.assertEqual(extra, {"unexpected_pred.png"})


class MetadataTests(unittest.TestCase):
    def test_manifest_contains_slice_and_inference_configuration(self):
        manifest = build_manifest(
            source_val_txt="/input/align_validation.txt",
            total_source_keys=4,
            start_index=1,
            end_index=3,
            selected_keys=["b", "c"],
            generated_prediction_count=2,
            missing_prediction_keys=[],
            extra_prediction_files=[],
            inference_config={"resolution": 512, "steps": 100},
        )

        self.assertEqual(manifest["range"], {"start": 1, "end": 3})
        self.assertEqual(manifest["expected_key_count"], 2)
        self.assertEqual(manifest["selected_keys"], ["b", "c"])
        self.assertEqual(manifest["inference"]["steps"], 100)

    def test_dataset_metadata_is_private_by_default(self):
        metadata = build_dataset_metadata(
            username="example",
            slug="diffv2ir-generated-00000-00002",
            title="DiffV2IR Generated 00000-00002",
        )

        self.assertEqual(metadata["id"], "example/diffv2ir-generated-00000-00002")
        self.assertEqual(metadata["licenses"], [{"name": "CC0-1.0"}])
        self.assertNotIn("public", metadata)


class CredentialsTests(unittest.TestCase):
    def test_writes_kaggle_json_without_printing_or_extra_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kaggle" / "kaggle.json"
            write_kaggle_credentials("example", "secret", path)

            self.assertEqual(json.loads(path.read_text()), {
                "username": "example",
                "key": "secret",
            })
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
