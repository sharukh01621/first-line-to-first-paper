#!/usr/bin/env python3
"""Option B Dual-Mirror Sync Helper (GitHub + Google Drive via rclone).

Usage:
  python3 scripts/sync_drive.py --dry-run   # Safe preview (no remote writes)
  python3 scripts/sync_drive.py             # Mirror colab_drive/ + PDF/EPUB -> Google Drive
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COLAB_DIR = ROOT / "colab_drive"
BOOK_OUT_DIR = ROOT / "book" / "_book"
METADATA_PATH = ROOT / "config" / "book_metadata.yml"
BOOK_DOWNLOADS_SUBDIR = "00_Book_Downloads"
BOOK_ARTIFACTS = [
    "From-First-Line-to-First-Paper.pdf",
    "From-First-Line-to-First-Paper.epub",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mirror colab_drive/ and compiled PDF/EPUB builds to Google Drive via rclone."
    )
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

    sync_cmd = [
        rclone_bin,
        "sync",
        str(COLAB_DIR),
        remote,
        "--exclude",
        f"{BOOK_DOWNLOADS_SUBDIR}/**",
        "--progress",
    ]
    if args.dry_run:
        sync_cmd.append("--dry-run")

    print("Running:", " ".join(sync_cmd))
    res = subprocess.run(sync_cmd, check=False)
    if res.returncode != 0:
        return res.returncode

    downloads_remote = f"{remote.rstrip('/')}/{BOOK_DOWNLOADS_SUBDIR}"
    for filename in BOOK_ARTIFACTS:
        artifact = BOOK_OUT_DIR / filename
        if not artifact.exists():
            continue
        copy_cmd = [rclone_bin, "copy", str(artifact), downloads_remote, "--progress"]
        if args.dry_run:
            copy_cmd.append("--dry-run")
        print("Running:", " ".join(copy_cmd))
        res = subprocess.run(copy_cmd, check=False)
        if res.returncode != 0:
            return res.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
