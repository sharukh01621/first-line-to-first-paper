#!/usr/bin/env python3
"""One-command Zero-Drift Curriculum Build & QA Pipeline.

Runs:
  1. generate_spine_datasets.py  (Public Health / Epidemiology CSVs)
  2. build_notebooks.py          (master_notebooks/ -> Student + Solution .ipynb + 00_START_HERE)
  3. inject_colab_links.py       (colab_manifest.json + print PDF QR codes)
  4. verify_all.py               (Automated headless QA of all notebooks & sharukh_tutor checks)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS = [
    "scripts/generate_spine_datasets.py",
    "scripts/build_notebooks.py",
    "scripts/inject_colab_links.py",
    "scripts/verify_all.py",
]


def main() -> int:
    for script in STEPS:
        print(f"\n=== Running {script} ===")
        res = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if res.returncode != 0:
            print(f"Pipeline halted: {script} exited with code {res.returncode}.")
            return res.returncode
    print("\nALL PIPELINE STAGES SUCCEEDED (Zero-Drift Sync & QA Verified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
