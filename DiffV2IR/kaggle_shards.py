"""Dependency-free helpers for running and publishing sharded Kaggle jobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_image_keys(input_rgb: str | os.PathLike[str] | Path) -> list[str]:
    """Discover inference keys from every image in an RGB directory.

    The inference script expects keys ending in ``_PreviewData``.  FLIR RGB
    files normally end in ``_RGB`` while some combined datasets use the plain
    stem; both forms are supported.  Ground-truth files ending in
    ``_PreviewData`` are ignored so an RGB/GT pair produces one key.
    """
    input_rgb = Path(input_rgb)
    if not input_rgb.is_dir():
        raise FileNotFoundError(f"RGB image directory not found: {input_rgb}")

    keys: set[str] = set()
    for path in input_rgb.iterdir():
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue

        stem = path.stem
        stem_lower = stem.lower()
        if stem_lower.endswith("_previewdata"):
            continue
        if stem_lower.endswith("_rgb"):
            stem = stem[:-len("_RGB")]
        if stem:
            keys.add(f"{stem}_PreviewData")

    return sorted(keys)


def select_key_range(keys: Sequence[str], start: int, end: int) -> list[str]:
    """Return keys in the validated half-open interval ``[start, end)``."""
    total = len(keys)
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"start/end must be integers: start={start!r}, end={end!r}")
    if start < 0 or end > total or start >= end:
        raise ValueError(
            f"invalid range [{start}, {end}) for {total} validation keys; "
            "require 0 <= start < end <= total"
        )
    return list(keys[start:end])


def prediction_names(keys: Sequence[str]) -> set[str]:
    """Return the exact prediction filenames expected for ``keys``."""
    return {f"{key}_pred.png" for key in keys}


def validate_predictions(
    keys: Sequence[str], prediction_dir: Path
) -> tuple[set[str], set[str]]:
    """Return ``(missing, extra)`` prediction filenames.

    Only files ending in ``_pred.png`` participate in the comparison; metadata,
    reports and visualization files can safely live beside the predictions.
    """
    prediction_dir = Path(prediction_dir)
    actual = {
        path.name
        for path in prediction_dir.iterdir()
        if path.is_file() and path.name.endswith("_pred.png")
    } if prediction_dir.is_dir() else set()
    expected = prediction_names(keys)
    return expected - actual, actual - expected


def build_manifest(
    *,
    source_val_txt: str,
    total_source_keys: int,
    start_index: int,
    end_index: int,
    selected_keys: Sequence[str],
    generated_prediction_count: int,
    missing_prediction_keys: Sequence[str],
    extra_prediction_files: Sequence[str],
    inference_config: dict[str, object],
) -> dict[str, object]:
    """Build a JSON-serializable record describing one generated slice."""
    return {
        "source_validation_file": source_val_txt,
        "total_source_keys": total_source_keys,
        "range": {"start": start_index, "end": end_index},
        "expected_key_count": len(selected_keys),
        "selected_keys": list(selected_keys),
        "generated_prediction_count": generated_prediction_count,
        "missing_prediction_keys": list(missing_prediction_keys),
        "extra_prediction_files": list(extra_prediction_files),
        "inference": dict(inference_config),
    }


def build_dataset_metadata(
    *, username: str, slug: str, title: str
) -> dict[str, object]:
    """Build private Kaggle Dataset metadata (no ``public`` flag)."""
    return {
        "id": f"{username}/{slug}",
        "title": title,
        "licenses": [{"name": "CC0-1.0"}],
    }


def write_kaggle_credentials(
    username: str, api_key: str, config_path: Path
) -> None:
    """Write Kaggle credentials with restrictive permissions."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"username": username, "key": api_key}),
        encoding="utf-8",
    )
    os.chmod(config_path, 0o600)
