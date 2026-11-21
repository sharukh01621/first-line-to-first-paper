"""Chapter 13 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch13_todo_01..06) and Chapter 13 Coding Exam
(ch13_exam_01..03) for Mastering Jupyter Notebooks & Google Colab Engineering.
Every file-based verifier runs inside tempfile.TemporaryDirectory() so zero
temporary files are ever left behind in the working directory.
"""

from __future__ import annotations

import copy
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _check_callable(
    ex_id: str,
    title: str,
    fn: Any,
    expected_name: str,
    points_possible: int = 10,
) -> Optional[CheckResult]:
    if not callable(fn):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Pass the Function Name Without Parentheses",
            (
                f"Expected a callable function (like {expected_name}), "
                f"but received {type(fn).__name__} ({fn!r})."
            ),
            details=[
                f"Pass the function itself: tutor.check('{ex_id}', {expected_name})",
                f"Do NOT add parentheses like {expected_name}(...) inside tutor.check().",
            ],
            points_earned=0,
            points_possible=points_possible,
        )
    return None


def _none_return_trap(
    ex_id: str,
    title: str,
    call_str: str,
    points_possible: int = 10,
) -> CheckResult:
    return CheckResult(
        ex_id,
        False,
        "failed",
        f"{title} — Returned None (print() vs. return Trap!)",
        f"Calling {call_str} returned None instead of the expected result.",
        details=[
            "Remember: print() only displays text in the cell output; it sends back None.",
            "Make sure your function ends with an explicit `return ...` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Notebook automation helpers must return values so CI checks and pipelines can inspect them.",
    )


def _make_nb(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": cells,
    }


def _md_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def _code_cell(
    source: list[str],
    execution_count: Optional[int],
    outputs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": list(outputs) if outputs is not None else [],
        "source": list(source),
    }


# ---------------------------------------------------------------------------
# GUIDED TO-DO 13.1: Detect Out-of-Order or Unexecuted Code Cells
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch13_todo_01", "To-Do 13.1", fn, "detect_out_of_order_cells", 10
    )
    if bad is not None:
        return bad

    clean_nb = _make_nb(
        [
            _md_cell("# Clean Experiment\n"),
            _code_cell(["x = 10\n"], 1),
            _md_cell("## Step 2\n"),
            _code_cell(["y = x * 2\n"], 2),
            _code_cell(["print(y)\n"], 3),
        ]
    )

    try:
        nominal = fn(clean_nb)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch13_todo_01",
            False,
            "failed",
            "To-Do 13.1 — Runtime Error Inspecting Notebook Dict",
            f"detect_out_of_order_cells(nb_dict) raised {type(exc).__name__}: {exc}",
            details=[
                "Skip non-code cells (`if cell.get('cell_type') != 'code': continue`) — markdown cells have no `execution_count`!"
            ],
            points_earned=0,
            points_possible=10,
        )

    if nominal is None:
        return _none_return_trap(
            "ch13_todo_01", "To-Do 13.1", "detect_out_of_order_cells(nb_dict)", 10
        )

    errors = []
    if type(nominal) is not bool or nominal is not False:
        errors.append(
            f"Clean sequential notebook (counts 1, 2, 3 + markdown): expected False, got {nominal!r}"
        )

    cases = [
        (
            "Strictly increasing with gaps (1, 4, 7)",
            _make_nb(
                [
                    _code_cell(["a = 1\n"], 1),
                    _md_cell("Note"),
                    _code_cell(["b = a + 1\n"], 4),
                    _code_cell(["c = b + 1\n"], 7),
                ]
            ),
            False,
        ),
        (
            "Out-of-order re-run top cell (5, 2, 3)",
            _make_nb(
                [
                    _code_cell(["x = 99\n"], 5),
                    _code_cell(["y = x + 1\n"], 2),
                    _code_cell(["print(y)\n"], 3),
                ]
            ),
            True,
        ),
        (
            "Duplicate execution_count (1, 2, 2)",
            _make_nb(
                [
                    _code_cell(["a = 1\n"], 1),
                    _code_cell(["b = 2\n"], 2),
                    _code_cell(["c = 3\n"], 2),
                ]
            ),
            True,
        ),
        (
            "Unexecuted code cell (1, None, 2)",
            _make_nb(
                [
                    _code_cell(["a = 1\n"], 1),
                    _code_cell(["b = 2\n"], None),
                    _code_cell(["c = 3\n"], 2),
                ]
            ),
            True,
        ),
        (
            "Markdown-only notebook (0 code cells)",
            _make_nb([_md_cell("# Notes only\n")]),
            False,
        ),
    ]

    for label, nb, expected in cases:
        try:
            got = fn(nb)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if type(got) is not bool or got is not expected:
            extra = ""
            if "Duplicate" in label and got is False:
                extra = " (check `<= prev`, not just `< prev`!)"
            elif "Unexecuted" in label and got is False:
                extra = " (a code cell with `execution_count is None` counts as out-of-order/unverified!)"
            errors.append(f"{label}: expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch13_todo_01",
            False,
            "failed",
            "To-Do 13.1 — Check Code-Cell Execution Order Logic",
            "Return True if any code cell has execution_count is None or <= previous count; otherwise return False.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch13_todo_01",
        True,
        "passed",
        "To-Do 13.1 Passed — Out-of-Order Execution Detector Verified!",
        "Flagged hidden-state re-runs, duplicate counts, and unexecuted code cells while accepting strictly increasing runs.",
        points_earned=10,
        points_possible=10,
        tip="Before submitting a notebook for publication, always run 'Restart & Run All' so every code cell counts strictly 1, 2, 3, ...",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_todo_01",
        chapter="ch13",
        title="To-Do 13.1: Detect Out-of-Order Notebook Cells",
        book_section="Section 13.3.1 (Hidden State & Execution Counts)",
        points=10,
        hints={
            1: "Loop over `nb_dict.get('cells', [])`, ignore non-code cells, track `prev = 0`, and return `True` if any code cell has `ec is None or ec <= prev`.",
            2: "Algorithm:\n1. prev = 0\n2. for cell in nb_dict.get('cells', []):\n       if cell.get('cell_type') != 'code': continue\n       ec = cell.get('execution_count')\n       if ec is None or ec <= prev: return True\n       prev = ec\n3. return False",
            3: "def detect_out_of_order_cells(nb_dict):\n    prev = 0\n    for cell in nb_dict.get('cells', []):\n        if cell.get('cell_type') != 'code':\n            continue\n        ec = cell.get('execution_count')\n        if ec is None or ec ___ prev:\n            return True\n        prev = ec\n    return False",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 13.2: Strip Notebook Outputs Without Mutating Input
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch13_todo_02", "To-Do 13.2", fn, "strip_notebook_outputs", 10
    )
    if bad is not None:
        return bad

    dirty_nb = _make_nb(
        [
            _md_cell("# Report\n"),
            _code_cell(
                ["print('ok')\n"],
                1,
                outputs=[{"output_type": "stream", "name": "stdout", "text": ["ok\n"]}],
            ),
            _code_cell(
                ["x = 42\n"],
                2,
                outputs=[
                    {
                        "output_type": "execute_result",
                        "data": {"text/plain": ["42"]},
                        "execution_count": 2,
                        "metadata": {},
                    }
                ],
            ),
        ]
    )
    snapshot = copy.deepcopy(dirty_nb)

    try:
        cleaned = fn(dirty_nb)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch13_todo_02",
            False,
            "failed",
            "To-Do 13.2 — Runtime Error Stripping Outputs",
            f"strip_notebook_outputs(nb_dict) raised {type(exc).__name__}: {exc}",
            details=[
                "Use `copy.deepcopy(nb_dict)` first, then set `outputs = []` and `execution_count = None` on code cells."
            ],
            points_earned=0,
            points_possible=10,
        )

    if cleaned is None:
        return _none_return_trap(
            "ch13_todo_02", "To-Do 13.2", "strip_notebook_outputs(nb_dict)", 10
        )

    errors = []
    if not isinstance(cleaned, dict):
        errors.append(f"Expected a dict return value, got {type(cleaned).__name__}")
    else:
        if dirty_nb != snapshot:
            errors.append(
                "Input nb_dict was mutated in place! Use `copy.deepcopy(nb_dict)` — a shallow `.copy()` still shares cell dicts."
            )
        if cleaned is dirty_nb:
            errors.append("Returned the exact same dict object instead of a deep copy.")
        if cleaned.get("nbformat") != 4 or "metadata" not in cleaned:
            errors.append("Top-level notebook metadata ('nbformat', 'metadata') must be preserved.")

        cells = cleaned.get("cells", [])
        if len(cells) != 3:
            errors.append(f"Expected 3 cells preserved, got {len(cells)}")
        else:
            if "outputs" in cells[0] or "execution_count" in cells[0]:
                errors.append(
                    "Markdown cell should NOT have 'outputs' or 'execution_count' keys added to it!"
                )
            for idx in (1, 2):
                cell = cells[idx]
                if cell.get("outputs") != []:
                    errors.append(
                        f"Code cell {idx} outputs were not cleared to []: got {cell.get('outputs')!r}"
                    )
                if cell.get("execution_count") is not None:
                    errors.append(
                        f"Code cell {idx} execution_count was not reset to None: got {cell.get('execution_count')!r}"
                    )
                if cell.get("source") != snapshot["cells"][idx]["source"]:
                    errors.append(f"Code cell {idx} source code was altered unexpectedly.")

    # Edge case: empty cells list
    empty_nb = _make_nb([])
    try:
        cleaned_empty = fn(empty_nb)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Empty notebook raised {type(exc).__name__}: {exc}")
    else:
        if cleaned_empty != empty_nb or cleaned_empty is empty_nb:
            errors.append("Empty notebook should return an equal deep copy, not the original object.")

    if errors:
        return CheckResult(
            "ch13_todo_02",
            False,
            "failed",
            "To-Do 13.2 — Check Deep Copy & Output Stripping",
            "Deep-copy nb_dict, reset outputs=[] and execution_count=None on every code cell only, and leave input unchanged.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch13_todo_02",
        True,
        "passed",
        "To-Do 13.2 Passed — Pure Notebook Output Stripper Verified!",
        "Cleared all code-cell outputs and execution counts on a deep copy without mutating the caller's notebook.",
        points_earned=10,
        points_possible=10,
        tip="Stripping base64 plot outputs before `git commit` shrinks diffs from megabytes of noise to clean readable Python lines.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_todo_02",
        chapter="ch13",
        title="To-Do 13.2: Strip Notebook Outputs Safely",
        book_section="Section 13.3.1 (Git-Friendly Notebooks & Deep Copy)",
        points=10,
        hints={
            1: "Start with `cleaned = copy.deepcopy(nb_dict)` so you never mutate `nb_dict`, then loop over code cells and reset `outputs` to `[]` and `execution_count` to `None`.",
            2: "Algorithm:\n1. import copy\n2. cleaned = copy.deepcopy(nb_dict)\n3. for cell in cleaned.get('cells', []):\n       if cell.get('cell_type') == 'code':\n           cell['outputs'] = []\n           cell['execution_count'] = None\n4. return cleaned",
            3: "import copy\n\ndef strip_notebook_outputs(nb_dict):\n    cleaned = copy.___(nb_dict)\n    for cell in cleaned.get('cells', []):\n        if cell.get('cell_type') == 'code':\n            cell['outputs'] = []\n            cell['execution_count'] = None\n    return cleaned",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 13.3: Build Canonical Google Colab GitHub Launch URL
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch13_todo_03", "To-Do 13.3", fn, "build_colab_url", 10)
    if bad is not None:
        return bad

    try:
        nominal = fn(
            "sharukh01621",
            "first-line-to-first-paper",
            "main",
            "colab_drive/part_03_engineering_workflow/ch13_jupyter_colab/01_walkthrough.ipynb",
        )
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch13_todo_03",
            False,
            "failed",
            "To-Do 13.3 — Runtime Error Building Colab URL",
            f"build_colab_url(...) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if nominal is None:
        return _none_return_trap(
            "ch13_todo_03",
            "To-Do 13.3",
            "build_colab_url(owner, repo, branch, rel_path)",
            10,
        )

    errors = []
    expected_nom = (
        "https://colab.research.google.com/github/"
        "sharukh01621/first-line-to-first-paper/blob/main/"
        "colab_drive/part_03_engineering_workflow/ch13_jupyter_colab/01_walkthrough.ipynb"
    )
    if nominal != expected_nom:
        errors.append(f"Nominal URL mismatch:\n  expected: {expected_nom!r}\n  got:      {nominal!r}")

    cases = [
        (
            ("lab-org", "mch-study", "main", "/notebooks/01_eda.ipynb"),
            "https://colab.research.google.com/github/lab-org/mch-study/blob/main/notebooks/01_eda.ipynb",
            "Single leading slash on rel_path",
        ),
        (
            ("lab-org", "mch-study", "dev", "///analysis/fig1.ipynb"),
            "https://colab.research.google.com/github/lab-org/mch-study/blob/dev/analysis/fig1.ipynb",
            "Multiple leading slashes + non-main branch",
        ),
    ]
    for args, expected, desc in cases:
        try:
            got = fn(*args)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{desc} raised {type(exc).__name__}: {exc}")
            continue
        if got != expected:
            extra = ""
            if "//notebooks" in str(got) or "//analysis" in str(got):
                extra = " (strip leading '/' from rel_path using `rel_path.lstrip('/')`!)"
            errors.append(f"{desc}: expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch13_todo_03",
            False,
            "failed",
            "To-Do 13.3 — Check Colab GitHub URL Format & Slash Stripping",
            "Return https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/{rel_path} with leading '/' stripped from rel_path.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch13_todo_03",
        True,
        "passed",
        "To-Do 13.3 Passed — Colab GitHub Launcher URL Builder Verified!",
        "Generated exact Colab badge URLs across branches and cleanly stripped leading slashes from relative paths.",
        points_earned=10,
        points_possible=10,
        tip="Embedding 'Open in Colab' links in your GitHub README lets peer reviewers launch your exact figure notebooks in one click.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_todo_03",
        chapter="ch13",
        title="To-Do 13.3: Build Google Colab GitHub URL",
        book_section="Section 13.3.2 (Colab GitHub Integration)",
        points=10,
        hints={
            1: "Strip any leading `/` from `rel_path` with `str(rel_path).lstrip('/')`, then interpolate into `https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/{clean_path}`.",
            2: "Algorithm:\n1. clean_path = str(rel_path).lstrip('/')\n2. return f'https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/{clean_path}'",
            3: "def build_colab_url(owner, repo, branch, rel_path):\n    clean_path = str(rel_path).___('/')\n    return (\n        'https://colab.research.google.com/github/'\n        f'{owner}/{repo}/blob/{branch}/{clean_path}'\n    )",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 13.4: Portable Colab/Local Data Path Resolver
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch13_todo_04", "To-Do 13.4", fn, "resolve_data_path", 10)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        drive_dir = base / "fake_drive"
        local_dir = base / "local_data"
        drive_dir.mkdir()
        local_dir.mkdir()

        both_drive = drive_dir / "cohort.csv"
        both_local = local_dir / "cohort.csv"
        both_drive.write_text("id\n1\n", encoding="utf-8")
        both_local.write_text("id\n2\n", encoding="utf-8")

        local_only = local_dir / "fallback.csv"
        local_only.write_text("id\n3\n", encoding="utf-8")

        # 1. Precedence: drive_dir wins when present in both
        try:
            got_both = fn("cohort.csv", drive_dir=drive_dir, local_dir=local_dir)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch13_todo_04",
                False,
                "failed",
                "To-Do 13.4 — Check Default Parameters",
                f"Calling resolve_data_path('cohort.csv', drive_dir=..., local_dir=...) raised TypeError: {exc}",
                details=[
                    "Define signature as: def resolve_data_path(filename, drive_dir=None, local_dir=None):"
                ],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch13_todo_04",
                False,
                "failed",
                "To-Do 13.4 — Runtime Error Resolving Data Path",
                f"resolve_data_path(...) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got_both is None:
            return _none_return_trap(
                "ch13_todo_04",
                "To-Do 13.4",
                "resolve_data_path('cohort.csv', drive_dir, local_dir)",
                10,
            )

        errors = []
        if not isinstance(got_both, Path):
            errors.append(
                f"Expected a pathlib.Path object, got {type(got_both).__name__} ({got_both!r})"
            )
        elif got_both != both_drive:
            errors.append(
                f"When file exists in both drive_dir and local_dir, expected drive_dir path ({both_drive}), got {got_both}"
            )

        # 2. Fallback to local_dir when missing in drive_dir (and string dir support)
        try:
            got_fallback = fn(
                "fallback.csv", drive_dir=str(drive_dir), local_dir=str(local_dir)
            )
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Fallback to local_dir raised {type(exc).__name__}: {exc}")
        else:
            if not isinstance(got_fallback, Path) or got_fallback != local_only:
                errors.append(
                    f"Fallback to local_dir: expected Path({local_only!r}), got {got_fallback!r}"
                )

        # 3. drive_dir=None -> checks local_dir cleanly
        try:
            got_no_drive = fn("fallback.csv", drive_dir=None, local_dir=local_dir)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"drive_dir=None raised {type(exc).__name__}: {exc}")
        else:
            if not isinstance(got_no_drive, Path) or got_no_drive != local_only:
                errors.append(
                    f"drive_dir=None case: expected {local_only!r}, got {got_no_drive!r}"
                )

        # 4. Missing file raises FileNotFoundError
        try:
            missing_res = fn("does_not_exist.csv", drive_dir=drive_dir, local_dir=local_dir)
        except NotImplementedError:
            raise
        except FileNotFoundError:
            pass
        except Exception as exc:
            errors.append(
                f"Missing file should raise FileNotFoundError, but raised {type(exc).__name__}: {exc}"
            )
        else:
            errors.append(
                f"Missing file should raise FileNotFoundError, but returned {missing_res!r}"
            )

        # 5. Both dirs None raises FileNotFoundError
        try:
            both_none = fn("cohort.csv")
        except NotImplementedError:
            raise
        except FileNotFoundError:
            pass
        except Exception as exc:
            errors.append(
                f"Calling with both dirs None should raise FileNotFoundError, got {type(exc).__name__}: {exc}"
            )
        else:
            errors.append(
                f"Calling with both dirs None should raise FileNotFoundError, but returned {both_none!r}"
            )

    if errors:
        return CheckResult(
            "ch13_todo_04",
            False,
            "failed",
            "To-Do 13.4 — Check Drive/Local Precedence & FileNotFoundError",
            "Check Path(drive_dir)/filename first, then Path(local_dir)/filename, return existing Path or raise FileNotFoundError.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch13_todo_04",
        True,
        "passed",
        "To-Do 13.4 Passed — Portable Colab/Local Path Resolver Verified!",
        "Resolved Drive-first with clean local fallback and raised FileNotFoundError when datasets are missing.",
        points_earned=10,
        points_possible=10,
        tip="Centralizing data lookup in one resolver lets the exact same notebook run unchanged in Colab Drive and local VS Code.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_todo_04",
        chapter="ch13",
        title="To-Do 13.4: Resolve Portable Colab/Local Data Path",
        book_section="Section 13.3.2 (Portable Colab & Local Paths)",
        points=10,
        hints={
            1: "Check `(drive_dir, local_dir)` in order: when a directory is not `None`, form `candidate = Path(directory) / filename` and return it if `candidate.exists()`; if neither exists, `raise FileNotFoundError(...)`.",
            2: "Algorithm:\n1. for directory in (drive_dir, local_dir):\n       if directory is not None:\n           candidate = Path(directory) / filename\n           if candidate.exists():\n               return candidate\n2. raise FileNotFoundError(f'Missing dataset: {filename}')",
            3: "from pathlib import Path\n\ndef resolve_data_path(filename, drive_dir=None, local_dir=None):\n    for directory in (drive_dir, local_dir):\n        if directory is not None:\n            candidate = Path(directory) / filename\n            if candidate.___():\n                return candidate\n    raise FileNotFoundError(f'Dataset not found: {filename}')",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 13.5: Atomic JSON Checkpoint Writer
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch13_todo_05", "To-Do 13.5", fn, "save_checkpoint_atomic", 10
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ckpt_path = base / "train_state.json"
        tmp_path = ckpt_path.with_suffix(".tmp")
        state_v1 = {
            "epoch": 5,
            "best_val_auc": 0.874,
            "site": "Dhaka — 38.4°C",
            "completed_folds": [0, 1],
        }

        try:
            fn(ckpt_path, state_v1)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch13_todo_05",
                False,
                "failed",
                "To-Do 13.5 — Runtime Error Writing Checkpoint",
                f"save_checkpoint_atomic(ckpt_path, state_dict) raised {type(exc).__name__}: {exc}",
                details=[
                    "Convert `target = Path(ckpt_path)`, write UTF-8 JSON to `tmp_path = target.with_suffix('.tmp')`, then call `tmp_path.replace(target)`."
                ],
                points_earned=0,
                points_possible=10,
            )

        errors = []
        if not ckpt_path.exists():
            errors.append("Target checkpoint file was not created at ckpt_path.")
        else:
            try:
                loaded_v1 = json.loads(ckpt_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"Checkpoint file is not valid JSON: {exc}")
            else:
                if loaded_v1 != state_v1:
                    errors.append(f"Saved JSON mismatch: expected {state_v1!r}, got {loaded_v1!r}")

        if tmp_path.exists():
            errors.append(
                f"Temporary file {tmp_path.name} still exists on disk — atomically rename it with `tmp_path.replace(ckpt_path)`!"
            )

        # Overwrite existing checkpoint cleanly
        state_v2 = {"epoch": 10, "best_val_auc": 0.912, "completed_folds": [0, 1, 2]}
        try:
            fn(ckpt_path, state_v2)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Overwriting existing checkpoint raised {type(exc).__name__}: {exc}")
        else:
            try:
                loaded_v2 = json.loads(ckpt_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"Overwritten checkpoint is invalid JSON: {exc}")
            else:
                if loaded_v2 != state_v2:
                    errors.append(f"Overwrite mismatch: expected {state_v2!r}, got {loaded_v2!r}")
            if tmp_path.exists():
                errors.append("Temporary .tmp file was left behind after overwriting checkpoint.")

        # Atomicity proof: failed serialization MUST NOT corrupt existing ckpt_path!
        if ckpt_path.exists() and not errors:
            try:
                fn(ckpt_path, {"epoch": 11, "unserializable": object()})
            except NotImplementedError:
                raise
            except TypeError:
                try:
                    still_valid = json.loads(ckpt_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    errors.append(
                        "Failed serialization corrupted the existing checkpoint! Write to `ckpt_path.with_suffix('.tmp')` FIRST, then rename with `.replace(ckpt_path)`."
                    )
                else:
                    if still_valid != state_v2:
                        errors.append(
                            "Existing checkpoint changed after a failed write — write to `.tmp` first before calling `.replace()`!"
                        )
            except Exception as exc:
                errors.append(
                    f"Unexpected exception during atomicity check: {type(exc).__name__}: {exc}"
                )

    if errors:
        return CheckResult(
            "ch13_todo_05",
            False,
            "failed",
            "To-Do 13.5 — Check Atomic .tmp Write & .replace() Rename",
            "Write UTF-8 JSON to ckpt_path.with_suffix('.tmp') and atomically swap it into place with tmp_path.replace(ckpt_path).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch13_todo_05",
        True,
        "passed",
        "To-Do 13.5 Passed — Atomic JSON Checkpoint Writer Verified!",
        "Wrote UTF-8 JSON via .tmp, replaced existing checkpoints cleanly, and preserved intact state when a write failed mid-stream.",
        points_earned=10,
        points_possible=10,
        tip="Colab free runtimes can disconnect mid-epoch. Atomic `.tmp` -> `.replace()` guarantees your last good checkpoint is never half-written.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_todo_05",
        chapter="ch13",
        title="To-Do 13.5: Save Checkpoint Atomically",
        book_section="Section 13.3.3 (Crash-Proof Atomic Checkpointing)",
        points=10,
        hints={
            1: "Convert `target = Path(ckpt_path)`, create `tmp_path = target.with_suffix('.tmp')`, write UTF-8 JSON to `tmp_path`, then call `tmp_path.replace(target)`.",
            2: "Algorithm:\n1. target = Path(ckpt_path)\n2. tmp_path = target.with_suffix('.tmp')\n3. with open(tmp_path, 'w', encoding='utf-8') as f:\n       json.dump(state_dict, f, indent=2)\n4. tmp_path.replace(target)",
            3: "import json\nfrom pathlib import Path\n\ndef save_checkpoint_atomic(ckpt_path, state_dict):\n    target = Path(ckpt_path)\n    tmp_path = target.with_suffix('.tmp')\n    with open(tmp_path, 'w', encoding='utf-8') as f:\n        json.dump(state_dict, f, indent=2)\n    tmp_path.___(target)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 13.6: Isolated-RNG Deterministic Train/Test Split
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch13_todo_06", "To-Do 13.6", fn, "seeded_train_test_indices", 10
    )
    if bad is not None:
        return bad

    random.seed(202601)
    global_before = random.getstate()

    try:
        split_a = fn(20)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch13_todo_06",
            False,
            "failed",
            "To-Do 13.6 — Check Default Parameters `test_ratio=0.2, seed=42`",
            f"Calling seeded_train_test_indices(20) raised TypeError: {exc}",
            details=[
                "Define signature as: def seeded_train_test_indices(n_samples, test_ratio=0.2, seed=42):"
            ],
            points_earned=0,
            points_possible=10,
        )
    except Exception as exc:
        return CheckResult(
            "ch13_todo_06",
            False,
            "failed",
            "To-Do 13.6 — Runtime Error Splitting Indices",
            f"seeded_train_test_indices(20) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    global_after = random.getstate()

    if split_a is None:
        return _none_return_trap(
            "ch13_todo_06", "To-Do 13.6", "seeded_train_test_indices(20)", 10
        )

    errors = []
    if global_after != global_before:
        errors.append(
            "Global `random` state was modified! Instantiate a local `rng = random.Random(seed)` and call `rng.shuffle(indices)` instead of global `random.seed()` / `random.shuffle()`."
        )

    if not isinstance(split_a, tuple) or len(split_a) != 2:
        errors.append(
            f"Expected a 2-tuple (train_idx, test_idx), got {type(split_a).__name__} ({split_a!r})"
        )
    else:
        train_a, test_a = split_a
        if not isinstance(train_a, list) or not isinstance(test_a, list):
            errors.append(
                f"Both train_idx and test_idx must be lists, got ({type(train_a).__name__}, {type(test_a).__name__})"
            )
        else:
            if len(test_a) != 4 or len(train_a) != 16:
                errors.append(
                    f"For n_samples=20, test_ratio=0.2: expected len(train)=16, len(test)=4; got ({len(train_a)}, {len(test_a)})"
                )
            if set(train_a) & set(test_a):
                errors.append("Train/test data leakage! train_idx and test_idx share overlapping indices.")
            if set(train_a) | set(test_a) != set(range(20)):
                errors.append("Combined train_idx and test_idx must cover every index in range(20) exactly once.")
            if train_a + test_a == list(range(20)) or test_a + train_a == list(range(20)):
                errors.append("Indices were not shuffled — call `rng.shuffle(indices)` before slicing!")

        # Determinism check with identical seed
        try:
            split_a2 = fn(20, test_ratio=0.2, seed=42)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Repeat call raised {type(exc).__name__}: {exc}")
        else:
            if split_a2 != split_a:
                errors.append("Two calls with identical arguments produced different splits — not deterministic!")

        # Different seed produces different split
        try:
            split_b = fn(20, test_ratio=0.2, seed=99)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"seed=99 call raised {type(exc).__name__}: {exc}")
        else:
            if split_b == split_a:
                errors.append("Different seeds (42 vs 99) produced identical splits — make sure `seed` is passed to `random.Random(seed)`!")

        # Zero test_ratio edge case (catches indices[-0:] bug)
        try:
            tr0, te0 = fn(10, test_ratio=0.0, seed=7)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"test_ratio=0.0 raised {type(exc).__name__}: {exc}")
        else:
            if len(te0) != 0 or len(tr0) != 10:
                errors.append(
                    f"test_ratio=0.0: expected len(train)=10, len(test)=0; got ({len(tr0)}, {len(te0)}) "
                    "(watch out for `indices[-0:]` which returns the entire list!)"
                )

    if errors:
        return CheckResult(
            "ch13_todo_06",
            False,
            "failed",
            "To-Do 13.6 — Check Isolated random.Random(seed) & Split Sizes",
            "Use local rng = random.Random(seed) to shuffle list(range(n_samples)) and return (train_idx, test_idx) with len(test_idx) == int(n_samples * test_ratio).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch13_todo_06",
        True,
        "passed",
        "To-Do 13.6 Passed — Isolated Deterministic Index Splitter Verified!",
        "Produced reproducible, leakage-free train/test splits without mutating Python's global RNG state.",
        points_earned=10,
        points_possible=10,
        tip="Using `random.Random(seed)` instead of global `random.seed()` ensures re-running another cell never shifts your held-out test set.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_todo_06",
        chapter="ch13",
        title="To-Do 13.6: Isolated Seeded Train/Test Indices",
        book_section="Section 13.3.4 (Isolated RNG & Reproducible Splits)",
        points=10,
        hints={
            1: "Create a local generator `rng = random.Random(seed)`, shuffle `indices = list(range(n_samples))` with `rng.shuffle(indices)`, compute `n_test = int(n_samples * test_ratio)`, and return `(indices[n_test:], indices[:n_test])`.",
            2: "Algorithm:\n1. rng = random.Random(seed)\n2. indices = list(range(n_samples))\n3. rng.shuffle(indices)\n4. n_test = int(n_samples * test_ratio)\n5. return indices[n_test:], indices[:n_test]",
            3: "import random\n\ndef seeded_train_test_indices(n_samples, test_ratio=0.2, seed=42):\n    rng = random.___(seed)\n    indices = list(range(n_samples))\n    rng.shuffle(indices)\n    n_test = int(n_samples * test_ratio)\n    return indices[n_test:], indices[:n_test]",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 13.1 (30 pts): Audit Notebook Reproducibility
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch13_exam_01", "Exam Task 13.1", fn, "audit_notebook_reproducibility", 30
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        dirty_path = base / "dirty_out_of_order.ipynb"
        dirty_path.write_text(
            json.dumps(
                _make_nb(
                    [
                        _md_cell("# Analysis\n"),
                        _code_cell(["x = 1\n"], 3),
                        _md_cell("## Plot\n"),
                        _code_cell(
                            ["print(x)\n"],
                            1,
                            outputs=[{"output_type": "stream", "name": "stdout", "text": ["1\n"]}],
                        ),
                        _code_cell(["y = x + 2\n"], 2),
                    ]
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            res_dirty = fn(dirty_path)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch13_exam_01",
                False,
                "failed",
                "Exam Task 13.1 — Runtime Error Auditing .ipynb File",
                f"audit_notebook_reproducibility(nb_path) raised {type(exc).__name__}: {exc}",
                details=[
                    "Load the JSON with `with open(nb_path, 'r', encoding='utf-8') as f: nb = json.load(f)` and inspect `nb.get('cells', [])`."
                ],
                points_earned=0,
                points_possible=30,
            )

        if res_dirty is None:
            return _none_return_trap(
                "ch13_exam_01",
                "Exam Task 13.1",
                "audit_notebook_reproducibility(nb_path)",
                30,
            )

        errors = []
        exp_dirty = {
            "code_cells": 3,
            "markdown_cells": 2,
            "out_of_order": True,
            "has_outputs": True,
            "clean_for_git": False,
        }
        if not isinstance(res_dirty, dict):
            errors.append(f"Expected a dict, got {type(res_dirty).__name__} ({res_dirty!r})")
        elif res_dirty != exp_dirty:
            errors.append(
                f"Dirty out-of-order notebook:\n  expected: {exp_dirty!r}\n  got:      {res_dirty!r}"
            )

        cases = [
            (
                "executed_inorder_with_outputs.ipynb",
                _make_nb(
                    [
                        _md_cell("# Clean Run\n"),
                        _code_cell(["a = 10\n"], 1),
                        _code_cell(
                            ["print(a)\n"],
                            2,
                            outputs=[{"output_type": "stream", "name": "stdout", "text": ["10\n"]}],
                        ),
                    ]
                ),
                {
                    "code_cells": 2,
                    "markdown_cells": 1,
                    "out_of_order": False,
                    "has_outputs": True,
                    "clean_for_git": False,
                },
                "In-order notebook with saved outputs",
            ),
            (
                "stripped_git_ready.ipynb",
                _make_nb(
                    [
                        _md_cell("# Git Master\n"),
                        _code_cell(["a = 1\n"], None, outputs=[]),
                        _code_cell(["b = 2\n"], None, outputs=[]),
                    ]
                ),
                {
                    "code_cells": 2,
                    "markdown_cells": 1,
                    "out_of_order": True,
                    "has_outputs": False,
                    "clean_for_git": True,
                },
                "Stripped Git-ready notebook (outputs=[], counts=None)",
            ),
            (
                "inorder_no_outputs.ipynb",
                _make_nb(
                    [
                        _code_cell(["x = 1\n"], 1, outputs=[]),
                        _code_cell(["y = 2\n"], 2, outputs=[]),
                    ]
                ),
                {
                    "code_cells": 2,
                    "markdown_cells": 0,
                    "out_of_order": False,
                    "has_outputs": False,
                    "clean_for_git": True,
                },
                "In-order executed notebook with empty outputs",
            ),
        ]

        for fname, nb_obj, expected, label in cases:
            path = base / fname
            path.write_text(json.dumps(nb_obj, indent=2), encoding="utf-8")
            try:
                got = fn(path)
            except NotImplementedError:
                raise
            except Exception as exc:
                errors.append(f"{label} raised {type(exc).__name__}: {exc}")
                continue
            if got != expected:
                errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch13_exam_01",
            False,
            "failed",
            "Exam Task 13.1 — Check Reproducibility Audit Keys & Flags",
            "Load nb_path as UTF-8 JSON and return code_cells, markdown_cells, out_of_order, has_outputs, and clean_for_git.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch13_exam_01",
        True,
        "passed",
        "Exam Task 13.1 Passed (30/30 pts) — Notebook Reproducibility Auditor Verified!",
        "Accurately audited cell counts, execution order, saved outputs, and Git cleanliness across all notebook states.",
        points_earned=30,
        points_possible=30,
        tip="Running an automated reproducibility audit in pre-commit hooks catches out-of-order experiments before peer review.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_exam_01",
        chapter="ch13",
        title="Exam 13.1: Audit Notebook Reproducibility",
        book_section="Chapter 13 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Load JSON from `nb_path`, count `code` and `markdown` cells, track `out_of_order` (any code cell with `execution_count is None` or `<= prev`) and `has_outputs` (any code cell with non-empty `outputs`), and set `clean_for_git = not has_outputs`.",
            2: "Algorithm:\n1. Load nb via json.load(f)\n2. Initialize code_cells=0, markdown_cells=0, out_of_order=False, has_outputs=False, prev=0\n3. Loop over nb.get('cells', []):\n     - markdown -> markdown_cells += 1\n     - code -> code_cells += 1; check ec is None or ec <= prev; check bool(cell.get('outputs'))\n4. Return dict with all 5 required keys",
            3: "import json\n\ndef audit_notebook_reproducibility(nb_path):\n    with open(nb_path, 'r', encoding='utf-8') as f:\n        nb = json.load(f)\n    code_cells = markdown_cells = 0\n    out_of_order = has_outputs = False\n    prev = 0\n    for cell in nb.get('cells', []):\n        ctype = cell.get('cell_type')\n        if ctype == 'markdown':\n            markdown_cells += 1\n        elif ctype == 'code':\n            code_cells += 1\n            ec = cell.get('execution_count')\n            if ec is None or ec <= prev:\n                out_of_order = True\n            else:\n                prev = ec\n            if cell.get('outputs'):\n                has_outputs = True\n    return {\n        'code_cells': code_cells,\n        'markdown_cells': markdown_cells,\n        'out_of_order': out_of_order,\n        'has_outputs': has_outputs,\n        'clean_for_git': not ___,\n    }",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 13.2 (35 pts): Extract Code Cells to Pure Python Module
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch13_exam_02", "Exam Task 13.2", fn, "extract_code_to_module", 35
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        nb_path = base / "prototype.ipynb"
        py_out = base / "pipeline_module.py"

        nb_path.write_text(
            json.dumps(
                _make_nb(
                    [
                        _md_cell("# Prototype Pipeline\nThis markdown must not appear.\n"),
                        _code_cell(
                            [
                                "!pip install -q sharukh-tutor\n",
                                "%matplotlib inline\n",
                                "import math\n",
                                "\n",
                                "def bmi(weight_kg, height_m):\n",
                                "    return round(weight_kg / (height_m ** 2), 2)\n",
                            ],
                            1,
                        ),
                        _code_cell(
                            [
                                "%%time\n",
                                "SAMPLE_BMI = bmi(68.0, 1.72)\n",
                            ],
                            2,
                        ),
                        _code_cell(
                            [
                                "!ls -la\n",
                                "%pwd\n",
                            ],
                            3,
                        ),
                    ]
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        try:
            line_count = fn(nb_path, py_out)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch13_exam_02",
                False,
                "failed",
                "Exam Task 13.2 — Runtime Error Extracting Python Module",
                f"extract_code_to_module(nb_path, py_out_path) raised {type(exc).__name__}: {exc}",
                details=[
                    "Read code cells from `nb_path`, skip lines starting with '!' or '%', join cells with blank lines, write UTF-8 to `py_out_path`, and return the count of non-empty lines written."
                ],
                points_earned=0,
                points_possible=35,
            )

        if line_count is None:
            return _none_return_trap(
                "ch13_exam_02",
                "Exam Task 13.2",
                "extract_code_to_module(nb_path, py_out_path)",
                35,
            )

        errors = []
        if type(line_count) is not int or line_count != 4:
            errors.append(
                f"Nominal notebook has 4 non-empty Python code lines (import math, def bmi, return ..., SAMPLE_BMI = ...): expected 4, got {line_count!r}"
            )

        if not py_out.exists():
            errors.append("Output Python file was not created at py_out_path.")
        else:
            py_text = py_out.read_text(encoding="utf-8")
            for forbidden in ("!pip", "%matplotlib", "%%time", "!ls", "%pwd", "Prototype Pipeline"):
                if forbidden in py_text:
                    errors.append(
                        f"Output .py file still contains forbidden notebook text {forbidden!r} — filter lines starting with '!' or '%' and skip markdown cells!"
                    )
            if "def bmi(" not in py_text or "SAMPLE_BMI = bmi(68.0, 1.72)" not in py_text:
                errors.append("Expected Python function and constant definitions were missing from py_out_path.")
            if "\n\n" not in py_text.strip():
                errors.append("Code cells should be separated by a blank line (`\\n\\n`) in the exported .py file.")
            try:
                ns: dict[str, Any] = {}
                exec(compile(py_text, str(py_out), "exec"), ns)
                if ns.get("SAMPLE_BMI") != 22.99:
                    errors.append(
                        f"Executing extracted module gave SAMPLE_BMI={ns.get('SAMPLE_BMI')!r}, expected 22.99"
                    )
            except SyntaxError as exc:
                errors.append(f"Extracted .py file has invalid Python syntax: {exc}")
            except Exception as exc:
                errors.append(f"Executing extracted .py file raised {type(exc).__name__}: {exc}")

        # Edge case: notebook with only markdown & magic cells -> 0 lines
        magic_only_nb = base / "magic_only.ipynb"
        magic_only_py = base / "magic_only.py"
        magic_only_nb.write_text(
            json.dumps(
                _make_nb(
                    [
                        _md_cell("# Setup Only\n"),
                        _code_cell(["!pip install -q pytest\n", "%config InlineBackend.figure_format = 'png'\n"], 1),
                    ]
                )
            ),
            encoding="utf-8",
        )
        try:
            zero_lines = fn(magic_only_nb, magic_only_py)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Magic-only notebook raised {type(exc).__name__}: {exc}")
        else:
            if type(zero_lines) is not int or zero_lines != 0:
                errors.append(f"Magic-only notebook: expected 0 non-empty lines, got {zero_lines!r}")
            if not magic_only_py.exists():
                errors.append("Magic-only notebook should still create py_out_path.")
            elif magic_only_py.read_text(encoding="utf-8").strip() != "":
                errors.append("Magic-only notebook should write an empty .py file.")

    if errors:
        return CheckResult(
            "ch13_exam_02",
            False,
            "failed",
            "Exam Task 13.2 — Check Notebook-to-Module Extraction",
            "Extract code cells skipping '!' and '%' magic lines, join cells with blank lines, write UTF-8 .py, and return count of non-empty code lines.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch13_exam_02",
        True,
        "passed",
        "Exam Task 13.2 Passed (35/35 pts) — Notebook-to-Python Extractor Verified!",
        "Exported valid, executable Python modules free of notebook magics and markdown, and returned exact non-empty line counts.",
        points_earned=35,
        points_possible=35,
        tip="Promoting reusable functions from exploratory notebooks into clean `.py` modules is the single biggest leap toward publication-grade code.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_exam_02",
        chapter="ch13",
        title="Exam 13.2: Extract Notebook Code to Python Module",
        book_section="Chapter 13 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Load `nb_path` with `json.load`, iterate over `cell_type == 'code'` cells, drop lines where `line.lstrip().startswith(('!', '%'))`, join non-empty cell blocks with `'\\n\\n'`, write UTF-8 to `py_out_path`, and return the count of non-empty lines.",
            2: "Algorithm:\n1. Load nb JSON\n2. blocks = []\n3. For each code cell:\n     text = ''.join(cell.get('source', []))\n     kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith(('!', '%'))]\n     block = '\\n'.join(kept).strip('\\n')\n     if block: blocks.append(block)\n4. content = ('\\n\\n'.join(blocks) + '\\n') if blocks else ''\n5. Write content UTF-8 to py_out_path\n6. Return sum(1 for ln in content.splitlines() if ln.strip())",
            3: "import json\n\ndef extract_code_to_module(nb_path, py_out_path):\n    with open(nb_path, 'r', encoding='utf-8') as f:\n        nb = json.load(f)\n    blocks = []\n    for cell in nb.get('cells', []):\n        if cell.get('cell_type') != 'code':\n            continue\n        src = cell.get('source', [])\n        text = ''.join(src) if isinstance(src, list) else src\n        kept = [\n            ln for ln in text.splitlines()\n            if not ln.lstrip().___(('!', '%'))\n        ]\n        block = '\\n'.join(kept).strip('\\n')\n        if block:\n            blocks.append(block)\n    content = ('\\n\\n'.join(blocks) + '\\n') if blocks else ''\n    with open(py_out_path, 'w', encoding='utf-8') as out_f:\n        out_f.write(content)\n    return sum(1 for ln in content.splitlines() if ln.strip())",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 13.3 (35 pts): Resume or Atomically Initialize Experiment
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch13_exam_03", "Exam Task 13.3", fn, "resume_or_init_experiment", 35
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        default_state = {"epoch": 0, "best_loss": 999.0, "history": []}
        default_snapshot = copy.deepcopy(default_state)

        # Case 1: Fresh init when checkpoint does not exist
        ckpt_fresh = base / "exp_fresh.json"
        try:
            res_fresh = fn(ckpt_fresh, default_state)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch13_exam_03",
                False,
                "failed",
                "Exam Task 13.3 — Runtime Error Initializing Experiment",
                f"resume_or_init_experiment(ckpt_path, default_state) raised {type(exc).__name__}: {exc}",
                details=[
                    "When `ckpt_path` does not exist (or holds invalid JSON), atomically write `default_state` via `.tmp` + `.replace()` and return `(copy_of_default, False)`."
                ],
                points_earned=0,
                points_possible=35,
            )

        if res_fresh is None:
            return _none_return_trap(
                "ch13_exam_03",
                "Exam Task 13.3",
                "resume_or_init_experiment(ckpt_path, default_state)",
                35,
            )

        errors = []
        if not isinstance(res_fresh, tuple) or len(res_fresh) != 2:
            errors.append(
                f"Expected a 2-tuple (state_dict, resumed_bool), got {type(res_fresh).__name__} ({res_fresh!r})"
            )
        else:
            st1, resumed1 = res_fresh
            if st1 != default_snapshot or resumed1 is not False:
                errors.append(
                    f"Missing checkpoint init: expected ({default_snapshot!r}, False), got ({st1!r}, {resumed1!r})"
                )
            elif st1 is default_state:
                errors.append(
                    "Returned default_state itself instead of a copy! Use `copy.deepcopy(default_state)`."
                )
            else:
                st1["history"].append(0.5)
                if default_state != default_snapshot:
                    errors.append(
                        "Mutating returned nested list modified caller's default_state! Use `copy.deepcopy(default_state)`."
                    )

        if not ckpt_fresh.exists():
            errors.append("Missing checkpoint init did not write ckpt_path to disk.")
        else:
            if json.loads(ckpt_fresh.read_text(encoding="utf-8")) != default_snapshot:
                errors.append("Written fresh checkpoint JSON does not match default_state.")
            if ckpt_fresh.with_suffix(".tmp").exists():
                errors.append("Leftover .tmp file found after fresh checkpoint initialization.")

        # Case 2: Resume from valid existing JSON dict
        ckpt_valid = base / "exp_resume.json"
        saved_state = {"epoch": 12, "best_loss": 0.214, "history": [0.9, 0.5, 0.214]}
        ckpt_valid.write_text(json.dumps(saved_state, indent=2), encoding="utf-8")
        try:
            st2, resumed2 = fn(ckpt_valid, default_snapshot)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Valid resume case raised {type(exc).__name__}: {exc}")
        else:
            if st2 != saved_state or resumed2 is not True:
                errors.append(
                    f"Valid existing checkpoint: expected ({saved_state!r}, True), got ({st2!r}, {resumed2!r})"
                )

        # Case 3: Corrupted/truncated JSON checkpoint -> recover cleanly with default_state
        ckpt_corrupt = base / "exp_corrupt.json"
        ckpt_corrupt.write_text('{"epoch": 4, "best_loss": ', encoding="utf-8")
        try:
            st3, resumed3 = fn(ckpt_corrupt, default_snapshot)
        except NotImplementedError:
            raise
        except json.JSONDecodeError:
            errors.append(
                "Corrupted checkpoint raised JSONDecodeError — catch `json.JSONDecodeError`, re-initialize atomically with `default_state`, and return `(copy, False)`!"
            )
        except Exception as exc:
            errors.append(f"Corrupted checkpoint raised {type(exc).__name__}: {exc}")
        else:
            if st3 != default_snapshot or resumed3 is not False:
                errors.append(
                    f"Corrupted checkpoint recovery: expected ({default_snapshot!r}, False), got ({st3!r}, {resumed3!r})"
                )
            elif json.loads(ckpt_corrupt.read_text(encoding="utf-8")) != default_snapshot:
                errors.append("Corrupted checkpoint file was not overwritten with valid default_state JSON.")
            if ckpt_corrupt.with_suffix(".tmp").exists():
                errors.append("Leftover .tmp file found after corrupted checkpoint recovery.")

        # Case 4: Valid JSON but NOT a dict (e.g. JSON list) -> re-initialize
        ckpt_list = base / "exp_list.json"
        ckpt_list.write_text("[1, 2, 3]", encoding="utf-8")
        try:
            st4, resumed4 = fn(ckpt_list, default_snapshot)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Non-dict JSON checkpoint raised {type(exc).__name__}: {exc}")
        else:
            if st4 != default_snapshot or resumed4 is not False:
                errors.append(
                    f"Non-dict JSON checkpoint ('[1, 2, 3]'): expected ({default_snapshot!r}, False), got ({st4!r}, {resumed4!r})"
                )
            elif json.loads(ckpt_list.read_text(encoding="utf-8")) != default_snapshot:
                errors.append("Non-dict checkpoint file was not replaced with default_state JSON.")

    if errors:
        return CheckResult(
            "ch13_exam_03",
            False,
            "failed",
            "Exam Task 13.3 — Check Resume, Corruption Recovery & Atomic Init",
            "Return (loaded_dict, True) if ckpt_path holds a valid JSON dict; otherwise atomically write default_state and return (copy_of_default, False).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch13_exam_03",
        True,
        "passed",
        "Exam Task 13.3 Passed (35/35 pts) — Crash-Resilient Experiment Resumer Verified!",
        "Resumed valid JSON checkpoints, recovered gracefully from truncated/non-dict files, and initialized fresh state atomically.",
        points_earned=35,
        points_possible=35,
        tip="Pairing `resume_or_init_experiment()` with `save_checkpoint_atomic()` turns fragile Colab training loops into restartable research pipelines.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch13_exam_03",
        chapter="ch13",
        title="Exam 13.3: Resume or Atomically Initialize Experiment",
        book_section="Chapter 13 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "If `target.exists()`, try loading JSON inside `try ... except json.JSONDecodeError:`; if `isinstance(loaded, dict)`, return `(loaded, True)`. Otherwise deep-copy `default_state`, write atomically via `.tmp` + `.replace(target)`, and return `(fresh, False)`.",
            2: "Algorithm:\n1. target = Path(ckpt_path)\n2. if target.exists():\n       try:\n           with open(target, 'r', encoding='utf-8') as f:\n               loaded = json.load(f)\n           if isinstance(loaded, dict):\n               return loaded, True\n       except json.JSONDecodeError:\n           pass\n3. fresh = copy.deepcopy(default_state)\n4. Write fresh to target.with_suffix('.tmp') and .replace(target)\n5. return fresh, False",
            3: "import copy\nimport json\nfrom pathlib import Path\n\ndef resume_or_init_experiment(ckpt_path, default_state):\n    target = Path(ckpt_path)\n    if target.exists():\n        try:\n            with open(target, 'r', encoding='utf-8') as f:\n                loaded = json.load(f)\n            if isinstance(loaded, ___):\n                return loaded, True\n        except json.JSONDecodeError:\n            pass\n    fresh = copy.deepcopy(default_state)\n    tmp_path = target.with_suffix('.tmp')\n    with open(tmp_path, 'w', encoding='utf-8') as f:\n        json.dump(fresh, f, indent=2)\n    tmp_path.replace(target)\n    return fresh, False",
        },
        verifier=_verify_exam_03,
    )
)
