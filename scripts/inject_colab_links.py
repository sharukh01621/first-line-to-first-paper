#!/usr/bin/env python3
"""Generate config/colab_manifest.json and print-ready QR codes for Quarto PDF/HTML.

Maps every generated notebook in colab_drive/ to its deterministic Option B
GitHub-backed Google Colab URL and renders crisp PNG QR codes into book/assets/qr/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    import qrcode
except ImportError:
    qrcode = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
COLAB_DIR = ROOT / "colab_drive"
METADATA_PATH = ROOT / "config" / "book_metadata.yml"
MANIFEST_PATH = ROOT / "config" / "colab_manifest.json"
QR_DIR = ROOT / "book" / "assets" / "qr"


def build_colab_url(rel: Path, meta: Dict[str, Any]) -> str:
    dist = meta["distribution"]
    return (
        f"https://colab.research.google.com/github/"
        f"{dist['github_owner']}/{dist['github_repo']}/blob/"
        f"{dist['github_branch']}/{dist['colab_drive_subdir']}/{rel.as_posix()}"
    )


def slug_for(rel: Path) -> str:
    return rel.with_suffix("").as_posix().replace("/", "__")


def main() -> None:
    meta = yaml.safe_load(METADATA_PATH.read_text(encoding="utf-8"))
    QR_DIR.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "distribution_mode": meta["distribution"]["mode"],
        "google_drive_folder_url": meta["distribution"]["google_drive_folder_url"],
        "notebooks": {},
    }

    notebooks = sorted(COLAB_DIR.rglob("*.ipynb"))
    qr_count = 0

    for nb_path in notebooks:
        rel = nb_path.relative_to(COLAB_DIR)
        is_solution = rel.parts[0] == "solutions"
        url = build_colab_url(rel, meta)
        slug = slug_for(rel)
        qr_rel = f"assets/qr/{slug}.png"
        qr_abs = QR_DIR / f"{slug}.png"

        if is_solution:
            if qr_abs.exists():
                qr_abs.unlink()
        elif qrcode is not None:
            img = qrcode.make(url)
            img.save(qr_abs)
            qr_count += 1

        manifest["notebooks"][rel.as_posix()] = {
            "colab_url": url,
            "qr_path": qr_rel if (not is_solution and qr_abs.exists()) else None,
            "is_solution": is_solution,
        }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {MANIFEST_PATH.relative_to(ROOT)} ({len(notebooks)} notebooks, {qr_count} QR codes generated).")


if __name__ == "__main__":
    main()
