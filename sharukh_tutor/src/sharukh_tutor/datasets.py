"""Curated Public Health, Epidemiology & Biostatistics dataset loader."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List


_DATA_DIR = Path(__file__).resolve().parent / "data"

_CATALOG: Dict[str, Dict[str, str]] = {
    "clinic_triage": {
        "filename": "clinic_triage.csv",
        "description": "Upazila community clinic patient intake records (vitals, BMI, hemoglobin, triage).",
    },
    "mch_survey_sample": {
        "filename": "mch_survey_sample.csv",
        "description": "DHS-style Maternal & Child Health household survey with realistic missing codes.",
    },
    "epi_surveillance": {
        "filename": "epi_surveillance.csv",
        "description": "District-level weekly infectious disease surveillance & weather covariates.",
    },
}


def list_datasets() -> Dict[str, str]:
    """Return available spine dataset names and descriptions."""
    return {name: meta["description"] for name, meta in _CATALOG.items()}


def _resolve_path(name: str) -> Path:
    if name not in _CATALOG:
        valid = ", ".join(sorted(_CATALOG.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {valid}")

    filename = _CATALOG[name]["filename"]
    bundled = _DATA_DIR / filename
    if bundled.exists():
        return bundled

    # Fallback when running directly from repository root before packaging
    repo_fallback = Path.cwd() / "colab_drive" / "datasets" / filename
    if repo_fallback.exists():
        return repo_fallback

    raise FileNotFoundError(
        f"Dataset file '{filename}' not found in package data ({bundled}) or {repo_fallback}."
    )


def load_data(name: str, as_pandas: bool = False) -> Any:
    """Load a Public Health / Epidemiology spine dataset.

    Parameters
    ----------
    name:
        Dataset identifier: 'clinic_triage', 'mch_survey_sample', or 'epi_surveillance'.
    as_pandas:
        If False (default for Parts I-II), returns a list of dictionaries using stdlib csv.
        If True (Parts IV-VII), returns a pandas.DataFrame.
    """
    path = _resolve_path(name)
    if as_pandas:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pandas is required when as_pandas=True. Install with 'pip install pandas' or pass as_pandas=False."
            ) from exc
        return pd.read_csv(path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
