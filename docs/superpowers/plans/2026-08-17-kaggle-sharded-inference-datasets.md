# Kaggle Sharded Inference Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each Kaggle notebook select a non-overlapping validation range, generate only that range, validate the exact prediction set, and create a private Kaggle Dataset with the predictions archived by Kaggle.

**Architecture:** Keep `infer_flir.py` unchanged for sampling quality. Add a dependency-free `kaggle_shards.py` helper for range validation, prediction validation, manifest/metadata creation, and credential setup. Add notebook cells to prepare a range-specific validation file, run inference into a package directory, validate it, and call `kaggle datasets create --dir-mode zip` without `--public`.

**Tech Stack:** Python standard library, existing DiffV2IR inference CLI, Kaggle CLI, Jupyter notebook JSON.

**Spec:** `docs/superpowers/specs/2026-08-17-kaggle-sharded-inference-datasets-design.md`

## Global Constraints

- Use half-open ranges `[START_INDEX, END_INDEX)`.
- Enforce `0 <= START_INDEX < END_INDEX <= len(all_validation_keys)`.
- Never upload when expected predictions are missing or extra prediction files exist.
- Create private datasets by omitting `--public` from `kaggle datasets create`.
- Do not change DiffV2IR model, prompt, seed, or sampling behavior.
- Never print Kaggle API credentials.
- Preserve the existing inference timing changes.

---

### Task 1: Add pure sharding and packaging helpers

**Files:**
- Create: `DiffV2IR/kaggle_shards.py`
- Create: `DiffV2IR/tests/test_kaggle_shards.py`

**Interfaces:**
- `select_key_range(keys: Sequence[str], start: int, end: int) -> list[str]`
- `prediction_names(keys: Sequence[str]) -> set[str]`
- `validate_predictions(keys: Sequence[str], prediction_dir: Path) -> tuple[set[str], set[str]]`
- `build_manifest(...) -> dict[str, object]`
- `build_dataset_metadata(username: str, slug: str, title: str) -> dict[str, object]`
- `write_kaggle_credentials(username: str, api_key: str, config_path: Path) -> None`

- [x] **Step 1: Write failing tests** for valid half-open selection, invalid bounds, exact prediction matching, missing/extra prediction detection, metadata ID/title/license, manifest fields, and credential file mode/content.
- [x] **Step 2: Run `python -m unittest DiffV2IR.tests.test_kaggle_shards -v`** and confirm it fails because `kaggle_shards.py` does not exist.
- [x] **Step 3: Implement only the standard-library helpers.** Prediction validation must compare exact filenames `<key>_pred.png`; range validation must raise `ValueError` with the bounds and total in the message.
- [x] **Step 4: Run the focused tests** and confirm all pass.
- [x] **Step 5: Run `python -m py_compile DiffV2IR/kaggle_shards.py`**.

### Task 2: Add range configuration and preparation to the Kaggle runbook

**Files:**
- Modify: `DiffV2IR/kaggle_run.py` in Cell 2 and a new preparation cell after Cell 5.
- Test: `DiffV2IR/tests/test_kaggle_run_contract.py`

**Interfaces:**
- Configuration variables: `START_INDEX`, `END_INDEX`, `DATASET_SLUG_PREFIX`, `KAGGLE_USERNAME_SECRET`, `KAGGLE_KEY_SECRET`.
- Runtime variables: `SELECTED_KEYS`, `RANGE_TAG`, `PACKAGE_DIR`, `PART_OUTPUT`, `PART_VAL_TXT`.

- [x] **Step 1: Write failing contract tests** that parse `kaggle_run.py` and assert the range variables, half-open slice expression, range validation call, and range-specific output/validation paths exist.
- [x] **Step 2: Run the contract test** and confirm it fails because the new configuration/cell is absent.
- [x] **Step 3: Add the four configuration variables and comments** with the 5,142-image examples `[0,1714)`, `[1714,3428)`, `[3428,5142)`.
- [x] **Step 4: Add a preparation cell** that reads `INPUT_FLIR/align_validation.txt`, calls `select_key_range`, writes `PART_VAL_TXT`, creates `PART_OUTPUT/predictions`, and prints selected count and first/last key.
- [x] **Step 5: Run the contract test** and a Python syntax compile of `kaggle_run.py`.

### Task 3: Route smoke/test/full inference and metrics to the selected slice

**Files:**
- Modify: `DiffV2IR/kaggle_run.py` Cells 6–9.
- Test: extend `DiffV2IR/tests/test_kaggle_run_contract.py`.

**Interfaces:**
- All inference commands consume `PART_VAL_TXT` and `PART_OUTPUT`.
- Smoke/test use `--limit` relative to the selected slice; full run omits `--limit` and processes exactly the selected keys.

- [x] **Step 1: Add failing assertions** that no production run command uses the original full `align_validation.txt` or shared `OUTPUT` after preparation.
- [x] **Step 2: Run the contract test** and confirm the old paths are detected.
- [x] **Step 3: Replace the Cell 6–9 paths** with `PART_VAL_TXT` and `PART_OUTPUT`; retain the existing model/config/quality arguments and WandB behavior.
- [x] **Step 4: Add a visible print** before each run showing `RANGE_TAG`, selected count, and output directory.
- [x] **Step 5: Run contract tests and compile the `.py` runbook**.

### Task 4: Validate and package exact predictions

**Files:**
- Modify: `DiffV2IR/kaggle_run.py` with a packaging cell after inference/metrics.
- Test: extend `DiffV2IR/tests/test_kaggle_run_contract.py`.

**Interfaces:**
- Packaging consumes `SELECTED_KEYS`, `PART_OUTPUT/predictions`, CLI arguments, and `RANGE_TAG`.
- It produces `manifest.json` and `dataset-metadata.json` under `PACKAGE_DIR` and exits before upload on any mismatch.

- [x] **Step 1: Add failing contract assertions** for `validate_predictions`, `manifest.json`, `dataset-metadata.json`, and the missing/extra hard gate.
- [x] **Step 2: Run the contract test** and confirm it fails.
- [x] **Step 3: Implement the packaging cell**: validate exact expected files, build the manifest with inference parameters, write metadata with authenticated username and `CC0-1.0`, and print a summary.
- [x] **Step 4: Run tests and compile the runbook**.

### Task 5: Authenticate and create the private Kaggle Dataset

**Files:**
- Modify: `DiffV2IR/kaggle_run.py` with an upload cell.
- Modify: `DiffV2IR/kaggle_run.ipynb` to mirror all updated cells.
- Test: extend `DiffV2IR/tests/test_kaggle_run_contract.py` and parse notebook JSON.

**Interfaces:**
- Reads `UserSecretsClient` values using the configured secret names.
- Writes `/root/.kaggle/kaggle.json` through `write_kaggle_credentials`.
- Executes `kaggle datasets create -p PACKAGE_DIR --dir-mode zip` with no `--public`.

- [x] **Step 1: Add failing assertions** that the upload command is private, uses `PACKAGE_DIR`, uses `--dir-mode zip`, and that notebook JSON contains the same configuration and commands as `kaggle_run.py`.
- [x] **Step 2: Run contract tests** and confirm they fail.
- [x] **Step 3: Add the credential/upload cell** with no secret printing, dataset ID `<username>/<slug>`, create-only behavior, and a status check after upload.
- [x] **Step 4: Synchronize the notebook cells** and preserve valid JSON and executable cell order.
- [x] **Step 5: Run the complete unit suite, compile `kaggle_run.py`, parse notebook JSON, and run `git diff --check`**.

### Task 6: Final verification and handoff

**Files:**
- Modify: no additional files.

- [x] **Step 1: Run `python -m unittest discover -s DiffV2IR/tests -v`**.
- [x] **Step 2: Run `python -m py_compile DiffV2IR/kaggle_run.py DiffV2IR/kaggle_shards.py`**.
- [x] **Step 3: Parse `DiffV2IR/kaggle_run.ipynb` with `json.load`** and confirm all cells have valid source arrays.
- [x] **Step 4: Review `git diff --stat`, `git diff --check`, and the final status**.
- [x] **Step 5: Report the exact Kaggle configuration for the three ranges and the private upload command; do not claim Kaggle publication until the user runs the notebook with valid secrets.**
