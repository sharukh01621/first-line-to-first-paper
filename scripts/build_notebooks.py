#!/usr/bin/env python3
"""Zero-drift notebook compiler: master_notebooks/ -> colab_drive/ (Student + Solution).

Rules applied to each master notebook:
  - Code blocks between `### BEGIN SOLUTION` and `### END SOLUTION`:
      * In STUDENT notebooks: replaced with `# YOUR CODE HERE` + stub (or `raise NotImplementedError(...)`
        when inside a `def` block, or `None` placeholder specified via `# STUB: ...`).
      * In SOLUTION notebooks: marker comments removed; full solution preserved.
  - Cells tagged `solution-only`: omitted in STUDENT notebooks, preserved in SOLUTION notebooks.
  - Top cell receives an auto-generated "Open in Colab" badge banner synced to config/book_metadata.yml.
  - Automatically generates `colab_drive/00_START_HERE.ipynb` master index notebook.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
MASTER_DIR = ROOT / "master_notebooks"
COLAB_DIR = ROOT / "colab_drive"
SOLUTIONS_DIR = COLAB_DIR / "solutions"
METADATA_PATH = ROOT / "config" / "book_metadata.yml"

BEGIN_RE = re.compile(r"^(\s*)###\s*BEGIN SOLUTION\s*$")
END_RE = re.compile(r"^(\s*)###\s*END SOLUTION\s*$")
STUB_RE = re.compile(r"^(\s*)#\s*STUB:\s*(.*)$")


def load_metadata() -> Dict[str, Any]:
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def colab_url_for(rel_from_colab_drive: Path, meta: Dict[str, Any]) -> str:
    dist = meta["distribution"]
    owner = dist["github_owner"]
    repo = dist["github_repo"]
    branch = dist["github_branch"]
    subdir = dist["colab_drive_subdir"]
    posix_rel = rel_from_colab_drive.as_posix()
    return f"https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/{subdir}/{posix_rel}"


def make_colab_badge_cell(rel_from_colab_drive: Path, meta: Dict[str, Any], is_solution: bool) -> Dict[str, Any]:
    url = colab_url_for(rel_from_colab_drive, meta)
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"
    book = meta["book"]
    label = "ANNOTATED SOLUTION NOTEBOOK" if is_solution else "INTERACTIVE STUDENT NOTEBOOK"
    lines = [
        f"{badge}\n",
        "\n",
        f"> **{book['title']}** — *{book['subtitle']}*  \n",
        f"> **Author:** {book['author']} | **Mode:** `{label}`\n",
    ]
    return {
        "cell_type": "markdown",
        "id": "colab-header-banner",
        "metadata": {"tags": ["colab-banner"]},
        "source": lines,
    }


def transform_code_source(lines: List[str], student: bool) -> List[str]:
    out: List[str] = []
    in_solution = False
    pending_stubs: List[str] = []
    indent = ""

    for raw_line in lines:
        line_no_nl = raw_line.rstrip("\r\n")
        has_nl = raw_line.endswith("\n")

        m_stub = STUB_RE.match(line_no_nl)
        if m_stub and not in_solution:
            if student:
                pending_stubs.append(m_stub.group(1) + m_stub.group(2) + ("\n" if has_nl else ""))
            continue

        m_begin = BEGIN_RE.match(line_no_nl)
        if m_begin:
            in_solution = True
            indent = m_begin.group(1)
            if student:
                out.append(f"{indent}# YOUR CODE HERE\n")
                if pending_stubs:
                    out.extend(pending_stubs)
                    pending_stubs = []
            continue

        m_end = END_RE.match(line_no_nl)
        if m_end:
            in_solution = False
            pending_stubs = []
            continue

        if in_solution and student:
            continue

        out.append(raw_line)

    return out


def strip_outputs(cell: Dict[str, Any]) -> Dict[str, Any]:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def compile_notebook(master_path: Path, meta: Dict[str, Any]) -> Dict[str, Path]:
    rel = master_path.relative_to(MASTER_DIR)
    nb_master = json.loads(master_path.read_text(encoding="utf-8"))
    has_solutions = any(
        "### BEGIN SOLUTION" in "".join(c.get("source", []))
        or "solution-only" in c.get("metadata", {}).get("tags", [])
        for c in nb_master.get("cells", [])
    )

    # 1. Build Student Notebook
    student_nb = copy.deepcopy(nb_master)
    student_cells: List[Dict[str, Any]] = []
    for cell in student_nb.get("cells", []):
        tags = cell.get("metadata", {}).get("tags", [])
        if "solution-only" in tags or "colab-banner" in tags:
            continue
        if cell.get("cell_type") == "code":
            src = cell.get("source", [])
            if isinstance(src, str):
                src = src.splitlines(keepends=True)
            cell["source"] = transform_code_source(src, student=True)
            strip_outputs(cell)
        student_cells.append(cell)

    student_cells.insert(0, make_colab_badge_cell(rel, meta, is_solution=False))
    student_nb["cells"] = student_cells

    student_out = COLAB_DIR / rel
    student_out.parent.mkdir(parents=True, exist_ok=True)
    student_out.write_text(json.dumps(student_nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    result = {"student": student_out}

    # 2. Build Annotated Solution Notebook (if notebook contains solutions/exercises)
    if has_solutions:
        sol_rel = Path("solutions") / rel.with_name(f"{rel.stem}_SOLUTION{rel.suffix}")
        sol_nb = copy.deepcopy(nb_master)
        sol_cells: List[Dict[str, Any]] = []
        for cell in sol_nb.get("cells", []):
            tags = cell.get("metadata", {}).get("tags", [])
            if "colab-banner" in tags:
                continue
            if cell.get("cell_type") == "code":
                src = cell.get("source", [])
                if isinstance(src, str):
                    src = src.splitlines(keepends=True)
                cell["source"] = transform_code_source(src, student=False)
                strip_outputs(cell)
            sol_cells.append(cell)

        sol_cells.insert(0, make_colab_badge_cell(sol_rel, meta, is_solution=True))
        sol_nb["cells"] = sol_cells

        sol_out = COLAB_DIR / sol_rel
        sol_out.parent.mkdir(parents=True, exist_ok=True)
        sol_out.write_text(json.dumps(sol_nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        result["solution"] = sol_out

    return result


def build_start_here_notebook(compiled_entries: List[Dict[str, Path]], meta: Dict[str, Any]) -> Path:
    book = meta["book"]
    gdrive_url = meta["distribution"]["google_drive_folder_url"]

    rows_md = [
        "| Chapter / Path | Student Notebook (Colab) | Annotated Solution (Colab) |\n",
        "| :--- | :--- | :--- |\n",
    ]
    for entry in compiled_entries:
        st_rel = entry["student"].relative_to(COLAB_DIR)
        st_url = colab_url_for(st_rel, meta)
        st_link = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({st_url})"
        if "solution" in entry:
            sol_rel = entry["solution"].relative_to(COLAB_DIR)
            sol_url = colab_url_for(sol_rel, meta)
            sol_link = f"[Open Solution]({sol_url})"
        else:
            sol_link = "*(Walkthrough — runnable as-is)*"
        rows_md.append(f"| `{st_rel.as_posix()}` | {st_link} | {sol_link} |\n")

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "welcome-header",
                "metadata": {},
                "source": [
                    f"# {book['title']}\n",
                    f"### *{book['subtitle']}*\n",
                    f"**Author:** {book['author']} ({book['credentials']})\n",
                    "\n",
                    "Welcome to the interactive **Google Colab & Google Drive Companion Hub**!\n",
                    "Every single walkthrough, guided To-Do practice exercise, and coding exam in the textbook has a **1-to-1 mirrored notebook** below.\n",
                    "\n",
                    f"- **Google Drive Mirror Folder:** [Open Full Curriculum Folder]({gdrive_url})\n",
                    "- **How to save your own progress:** Click any **Open in Colab** badge below, then choose **File -> Save a copy in Drive**.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "chapter-directory",
                "metadata": {},
                "source": ["## Interactive Notebook Directory\n\n"] + rows_md,
            },
            {
                "cell_type": "code",
                "id": "quick-healthcheck",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Optional Quick Test: Click Play (▶) to confirm Python is alive in Colab!\n",
                    'print("Welcome to From First Line to First Paper! Your Python runtime is ready.")\n',
                ],
            },
        ],
    }

    out = COLAB_DIR / "00_START_HERE.ipynb"
    out.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> None:
    meta = load_metadata()
    masters = sorted(MASTER_DIR.rglob("*.ipynb"))
    compiled: List[Dict[str, Path]] = []

    for master in masters:
        res = compile_notebook(master, meta)
        compiled.append(res)
        sol_note = f" + {res['solution'].relative_to(ROOT)}" if "solution" in res else ""
        print(f"Compiled {master.relative_to(ROOT)} -> {res['student'].relative_to(ROOT)}{sol_note}")

    start_here = build_start_here_notebook(compiled, meta)
    print(f"Generated master dashboard: {start_here.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
