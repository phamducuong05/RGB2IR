# Kaggle Sharded Inference Datasets Design

## Goal

Allow independent Kaggle notebooks/accounts to generate non-overlapping slices of the DiffV2IR validation list, validate each slice, and publish it as a private Kaggle Dataset that can later be merged into one complete result.

## Slice semantics

Each run accepts `START_INDEX` and `END_INDEX` using Python half-open interval semantics: `START_INDEX` is included and `END_INDEX` is excluded. Validation must enforce:

```text
0 <= START_INDEX < END_INDEX <= len(all_validation_keys)
```

For 5,142 entries, the recommended three-way split is:

```text
[0, 1714)
[1714, 3428)
[3428, 5142)
```

The notebook writes the selected keys to `/kaggle/working/validation_<start>_<end>.txt` and passes that file to the existing `infer_flir.py`. The inference/model implementation, prompt construction, seed, and sampling settings remain unchanged.

## Configuration

The Kaggle configuration cell exposes:

```python
START_INDEX = 0
END_INDEX = 1714
DATASET_SLUG_PREFIX = "diffv2ir-generated"
KAGGLE_USERNAME_SECRET = "phamduccuong05"
KAGGLE_KEY_SECRET = "KEY"
```

The secret named by `KAGGLE_USERNAME_SECRET` contains the Kaggle username. The secret named by `KAGGLE_KEY_SECRET` contains the Kaggle API key. Credentials are written to `/root/.kaggle/kaggle.json` with mode `0600`; the API key is never printed.

## Output and packaging

Each run uses this package directory:

```text
/kaggle/working/diffv2ir-part-<start:05d>-<end:05d>/
├── dataset-metadata.json
├── manifest.json
└── predictions/
    ├── *_pred.png
    ├── metrics.json
    ├── metrics_report.txt
    └── visualization/
```

The dataset slug is `<prefix>-<start:05d>-<end:05d>`. The dataset ID is `<authenticated-username>/<slug>`, and the title is `DiffV2IR Generated <start:05d>-<end:05d>`. Metadata uses the `CC0-1.0` license.

`manifest.json` records the source validation path, total source key count, start/end indices, expected key count, selected keys, generated prediction count, missing prediction keys, and inference parameters (`resolution`, `steps`, CFG scales, seed, and BLIP model).

## Validation gate

Before upload, the notebook compares the selected keys against files named `<key>_pred.png` in `predictions/`.

- If any expected prediction is missing, packaging stops with an error and does not call Kaggle.
- Extra prediction files are recorded and also cause packaging to stop, preventing contamination from a reused output directory.
- If counts and names match exactly, metadata and manifest are written and upload may proceed.

## Kaggle Dataset creation

The notebook creates a private dataset using:

```bash
kaggle datasets create -p <package-directory> --dir-mode zip
```

No `--public` flag is passed. `predictions/` is a directory so Kaggle CLI uploads it as a ZIP archive, avoiding the per-file upload limit for large slices. A non-zero Kaggle CLI exit code fails the notebook cell. After creation, the notebook prints the expected dataset URL and queries dataset status.

Dataset creation is intentionally create-only. Re-running an already published slice with the same owner and slug fails rather than silently making a new version. The operator must choose a new slug prefix or explicitly manage the existing dataset.

## Files and responsibilities

- `DiffV2IR/kaggle_shards.py`: dependency-free slice validation, manifest construction, prediction validation, Kaggle metadata construction, and credential writing.
- `DiffV2IR/tests/test_kaggle_shards.py`: unit tests for half-open slicing, boundary errors, exact prediction validation, manifest fields, metadata, and credential permissions where supported.
- `DiffV2IR/kaggle_run.py`: runnable notebook source with slice preparation, inference invocation, packaging gate, credentials, and private dataset creation.
- `DiffV2IR/kaggle_run.ipynb`: notebook synchronized from the same cells as `kaggle_run.py`.

## Testing

Pure helper behavior is tested locally without Kaggle credentials or network access. The repository test suite must pass, both notebook files must parse, and their configuration/inference/upload cells must contain matching parameters and commands. Actual Dataset publication is verified only on Kaggle because it requires account secrets and network access.
