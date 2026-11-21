#!/usr/bin/env python3
"""Option B Dual-Mirror Sync Helper (GitHub + Google Drive via rclone).

Usage:
  python3 scripts/sync_drive.py --dry-run   # Safe preview (no remote writes)
  python3 scripts/sync_drive.py             # Mirror colab_drive/ -> Google Drive remote
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COLAB_DIR = ROOT / "colab_drive"
METADATA_PATH = ROOT / "config" / "book_metadata.yml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror colab_drive/ to Google Drive via rclone.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without uploading.")
    args = parser.parse_args()

    meta = yaml.safe_load(METADATA_PATH.read_text(encoding="utf-8"))
    remote = meta["distribution"]["rclone_remote"]

    rclone_bin = shutil.which("rclone")
    if rclone_bin is None:
        print("rclone is not installed on PATH yet.")
        print("Install on Ubuntu/Debian: sudo apt install rclone")
        print(f"Then run 'rclone config' to connect your Google Drive remote, and sync to: {remote}")
        return 1

    cmd = [rclone_bin, "sync", str(COLAB_DIR), remote, "--progress"]
    if args.dry_run:
        cmd.append("--dry-run")

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
