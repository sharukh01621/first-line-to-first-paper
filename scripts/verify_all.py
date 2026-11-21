#!/usr/bin/env python3
"""Automated QA verifier for all generated Student and Solution notebooks.

Executes notebooks headlessly in isolated Python namespaces:
  1. SOLUTION & WALKTHROUGH notebooks:
       - Every runnable code cell (except cells tagged `expected-error`) MUST execute with 0 uncaught errors.
       - Every `tutor.check(...)` / `tutor.check_exam(...)` recorded in the session MUST PASS (100%).
  2. STUDENT exercise/exam notebooks:
       - Setup & hint cells MUST execute cleanly with 0 uncaught errors.
       - All `tutor.check(...)` calls MUST fail gracefully (status in {'unattempted', 'failed'}),
         confirming no solution accidentally leaked into the student workbook.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
COLAB_DIR = ROOT / "colab_drive"
TUTOR_SRC = ROOT / "sharukh_tutor" / "src"

if str(TUTOR_SRC) not in sys.path:
    sys.path.insert(0, str(TUTOR_SRC))

import sharukh_tutor as tutor  # noqa: E402


def _strip_colab_magics(code: str) -> str:
    """Ignore shell/pip lines like `!pip install -q sharukh-tutor` during local offline QA."""
    cleaned = []
    for line in code.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("!pip ") or stripped.startswith("%pip "):
            cleaned.append(f"# [local-qa-skipped] {line}")
        else:
            cleaned.append(line)
    return "".join(cleaned)


def execute_notebook(nb_path: Path, expect_all_pass: bool) -> Tuple[bool, List[str]]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    errors: List[str] = []

    # Reset active tutor session before running notebook
    tutor.activate(chapter="qa_reset", notebook="qa", quiet=True)

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        tags = set(cell.get("metadata", {}).get("tags", []))
        src = cell.get("source", [])
        code = "".join(src) if isinstance(src, list) else str(src)
        code = _strip_colab_magics(code)

        if not code.strip():
            continue

        buf = io.StringIO()
        err_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err_buf):
                exec(compile(code, f"{nb_path.name}#cell-{idx}", "exec"), ns)
            if "expected-error" in tags:
                errors.append(f"Cell {idx} tagged 'expected-error' did NOT raise an error as expected.")
        except (Exception, SystemExit) as exc:
            if "expected-error" in tags:
                continue
            errors.append(f"Cell {idx} raised unexpected {type(exc).__name__}: {exc}")

    sess = tutor.get_session()
    if sess.results:
        if expect_all_pass:
            for ex_id, res in sess.results.items():
                if not res.passed:
                    errors.append(f"Solution check FAILED for {ex_id}: {res.message}")
        else:
            for ex_id, res in sess.results.items():
                if res.passed:
                    errors.append(f"SOLUTION LEAK DETECTED: student notebook passed {ex_id} without solving!")

    return (len(errors) == 0), errors


def main() -> int:
    notebooks = sorted(COLAB_DIR.rglob("*.ipynb"))
    if not notebooks:
        print("No generated notebooks found in colab_drive/. Run build_notebooks.py first.")
        return 1

    failures = 0
    for nb_path in notebooks:
        rel = nb_path.relative_to(COLAB_DIR)
        is_solution = rel.parts[0] == "solutions"
        is_walkthrough = "walkthrough" in nb_path.stem or nb_path.stem in {"00_START_HERE", "00_colab_orientation"}
        expect_pass = is_solution or is_walkthrough

        ok, errs = execute_notebook(nb_path, expect_all_pass=expect_pass)
        mode = "SOLUTION/WALKTHROUGH" if expect_pass else "STUDENT-STUB"
        if ok:
            print(f"[PASS] ({mode}) {rel.as_posix()}")
        else:
            failures += 1
            print(f"[FAIL] ({mode}) {rel.as_posix()}")
            for err in errs:
                print(f"       -> {err}")

    print(f"\nQA Verification Complete: {len(notebooks) - failures}/{len(notebooks)} notebooks passed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
