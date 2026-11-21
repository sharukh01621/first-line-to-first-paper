"""Chapter 9 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch09_todo_01..06) and Chapter 9 Coding Exam
(ch09_exam_01..03) for File I/O, CSV/JSON Parsing, and Persistent Data.
Every verifier creates isolated files inside tempfile.TemporaryDirectory()
so zero temporary files are ever left behind in the working directory.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


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
            "Remember: print() only displays text on screen; it sends back None to the caller.",
            "Make sure your function ends with an explicit `return ...` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="In file pipelines, functions should return counts or parsed data structures so downstream code can verify them.",
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 9.1: Count Non-Empty Lines in a Text File
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch09_todo_01", "To-Do 9.1", fn, "count_nonempty_lines", 10)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        nominal_file = base / "clinic_notes.txt"
        nominal_file.write_text(
            "UHC-01: Fever 38.6 C\n"
            "\n"
            "   \t  \n"
            "UHC-02: Normal vitals\n"
            "UHC-03: Referred to lab\n"
            "  \n",
            encoding="utf-8",
        )

        try:
            nominal = fn(nominal_file)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch09_todo_01",
                False,
                "failed",
                "To-Do 9.1 — Runtime Error Reading File",
                f"count_nonempty_lines(filepath) raised {type(exc).__name__}: {exc}",
                details=[
                    "Open the file with `with open(filepath, 'r', encoding='utf-8') as f:` and iterate line by line."
                ],
                points_earned=0,
                points_possible=10,
            )

        if nominal is None:
            return _none_return_trap(
                "ch09_todo_01", "To-Do 9.1", "count_nonempty_lines(filepath)", 10
            )

        errors = []
        if type(nominal) is not int or nominal != 3:
            extra = ""
            if nominal == 6:
                extra = " (looks like you counted blank/whitespace lines too — check `if line.strip():`!)"
            elif nominal == 5:
                extra = " (whitespace-only lines like `'   \\t  \\n'` are empty after `.strip()`!)"
            errors.append(
                f"Nominal file (3 non-empty lines + 3 blank/space lines): expected 3, got {nominal!r}{extra}"
            )

        edge_cases = [
            ("single.txt", "Single entry without newline", 1, "Single-line file"),
            ("spaces_only.txt", "\n   \n\t\t\n", 0, "Whitespace-only file"),
            ("empty.txt", "", 0, "Completely empty file"),
        ]
        for fname, content, expected, label in edge_cases:
            path = base / fname
            path.write_text(content, encoding="utf-8")
            try:
                got = fn(path)
            except NotImplementedError:
                raise
            except Exception as exc:
                errors.append(f"{label} raised {type(exc).__name__}: {exc}")
                continue
            if type(got) is not int or got != expected:
                errors.append(f"{label}: expected {expected}, got {got!r}")

    if errors:
        return CheckResult(
            "ch09_todo_01",
            False,
            "failed",
            "To-Do 9.1 — Check Non-Empty Line Counting Logic",
            "Count only lines where `line.strip()` is non-empty and return an integer.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch09_todo_01",
        True,
        "passed",
        "To-Do 9.1 Passed — Non-Empty Line Counter Verified!",
        "Accurately counted non-empty lines across mixed, single-line, whitespace-only, and empty files.",
        points_earned=10,
        points_possible=10,
        tip="Iterating `for line in f:` streams a file one line at a time using minimal RAM, even for gigabyte logs.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_todo_01",
        chapter="ch09",
        title="To-Do 9.1: Count Non-Empty Lines",
        book_section="Section 9.3.1 (Context Managers & Line Iteration)",
        points=10,
        hints={
            1: "Open `filepath` with `with open(filepath, 'r', encoding='utf-8') as f:` and increment a counter whenever `line.strip()` is non-empty.",
            2: "Algorithm:\n1. count = 0\n2. with open(filepath, 'r', encoding='utf-8') as f:\n       for line in f:\n           if line.strip():\n               count += 1\n3. return count",
            3: "def count_nonempty_lines(filepath):\n    count = 0\n    with open(filepath, 'r', encoding='utf-8') as f:\n        for line in f:\n            if line.___():\n                count += 1\n    return count",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 9.2: Write Clinic Log in UTF-8
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch09_todo_02", "To-Do 9.2", fn, "write_clinic_log", 10)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        log_path = base / "daily_log.txt"
        entries = [
            "08:15 | Dhaka-North | Temp 38.4°C",
            "09:00 | Sylhet-East | Hb 10.6 g/dL",
            "10:30 | Khulna-West | Routine ANC",
        ]

        try:
            written = fn(log_path, list(entries))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch09_todo_02",
                False,
                "failed",
                "To-Do 9.2 — Runtime Error Writing Log File",
                f"write_clinic_log(filepath, entries) raised {type(exc).__name__}: {exc}",
                details=[
                    "Open the file with `with open(filepath, 'w', encoding='utf-8') as f:` and write each entry followed by `'\\n'`."
                ],
                points_earned=0,
                points_possible=10,
            )

        if written is None:
            return _none_return_trap(
                "ch09_todo_02", "To-Do 9.2", "write_clinic_log(filepath, entries)", 10
            )

        errors = []
        if type(written) is not int or written != 3:
            errors.append(f"Expected return count 3 for 3 entries, got {written!r}")

        if not log_path.exists():
            errors.append("File was not created at the given filepath.")
        else:
            raw_text = log_path.read_text(encoding="utf-8")
            lines = raw_text.splitlines()
            if len(lines) == 1 and len(entries) > 1:
                errors.append(
                    "All entries were concatenated onto 1 line — add `'\\n'` after each entry!"
                )
            elif lines != entries:
                errors.append(f"File lines mismatch: expected {entries!r}, got {lines!r}")

        # Verify overwrite mode ('w', not append 'a')
        overwrite_entries = ["11:00 | Rajshahi | Fresh shift"]
        try:
            written_2 = fn(log_path, list(overwrite_entries))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Second write (overwrite check) raised {type(exc).__name__}: {exc}")
        else:
            if type(written_2) is not int or written_2 != 1:
                errors.append(f"Overwrite call: expected count 1, got {written_2!r}")
            lines_2 = log_path.read_text(encoding="utf-8").splitlines()
            if lines_2 != overwrite_entries:
                errors.append(
                    f"Overwrite check failed: expected {overwrite_entries!r}, got {lines_2!r} (use mode 'w', not 'a'!)"
                )

        # Verify empty list edge case
        empty_path = base / "empty_log.txt"
        try:
            written_empty = fn(empty_path, [])
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Empty entries list raised {type(exc).__name__}: {exc}")
        else:
            if type(written_empty) is not int or written_empty != 0:
                errors.append(f"Empty list: expected return count 0, got {written_empty!r}")
            if not empty_path.exists() or empty_path.read_text(encoding="utf-8") != "":
                errors.append("Empty entries list should create an empty file.")

    if errors:
        return CheckResult(
            "ch09_todo_02",
            False,
            "failed",
            "To-Do 9.2 — Check UTF-8 Line Writing & Return Count",
            "Write each entry on its own line in UTF-8 using mode 'w', and return the integer count written.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch09_todo_02",
        True,
        "passed",
        "To-Do 9.2 Passed — UTF-8 Clinic Log Writer Verified!",
        "Wrote multi-line UTF-8 logs, cleanly overwrote existing files, and handled empty entry lists.",
        points_earned=10,
        points_possible=10,
        tip="Always specify encoding='utf-8' explicitly so symbols like °C or local place names never break across Windows and Linux.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_todo_02",
        chapter="ch09",
        title="To-Do 9.2: Write UTF-8 Clinic Log",
        book_section="Section 9.3.1 (Writing Text Files Safely)",
        points=10,
        hints={
            1: "Open `filepath` with `with open(filepath, 'w', encoding='utf-8') as f:`, write `f\"{entry}\\n\"` for each item in `entries`, and return `len(entries)`.",
            2: "Algorithm:\n1. with open(filepath, 'w', encoding='utf-8') as f:\n       for entry in entries:\n           f.write(f'{entry}\\n')\n2. return len(entries)",
            3: "def write_clinic_log(filepath, entries):\n    with open(filepath, 'w', encoding='utf-8') as f:\n        for entry in entries:\n            f.write(f'{entry}___')\n    return len(entries)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 9.3: Read CSV File as List of Dictionaries
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch09_todo_03", "To-Do 9.3", fn, "read_csv_as_dicts", 10)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        csv_path = base / "clinic_patients.csv"
        csv_path.write_text(
            "patient_id,district,hb_g_dl\n"
            "P001,Dhaka,10.8\n"
            "P002,Sylhet,12.1\n"
            "P003,Khulna,9.6\n",
            encoding="utf-8",
        )

        try:
            rows = fn(csv_path)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch09_todo_03",
                False,
                "failed",
                "To-Do 9.3 — Runtime Error Reading CSV",
                f"read_csv_as_dicts(csv_path) raised {type(exc).__name__}: {exc}",
                details=[
                    "Use `with open(csv_path, 'r', encoding='utf-8', newline='') as f:` and `return list(csv.DictReader(f))`."
                ],
                points_earned=0,
                points_possible=10,
            )

        if rows is None:
            return _none_return_trap(
                "ch09_todo_03", "To-Do 9.3", "read_csv_as_dicts(csv_path)", 10
            )

        errors = []
        expected_rows = [
            {"patient_id": "P001", "district": "Dhaka", "hb_g_dl": "10.8"},
            {"patient_id": "P002", "district": "Sylhet", "hb_g_dl": "12.1"},
            {"patient_id": "P003", "district": "Khulna", "hb_g_dl": "9.6"},
        ]
        if isinstance(rows, csv.DictReader):
            errors.append(
                "You returned the open csv.DictReader iterator directly — convert it with `list(csv.DictReader(f))` BEFORE exiting the `with` block!"
            )
        elif not isinstance(rows, list):
            errors.append(f"Expected a list of dicts, got {type(rows).__name__} ({rows!r})")
        elif rows != expected_rows:
            errors.append(f"Nominal CSV rows mismatch: expected {expected_rows!r}, got {rows!r}")

        # Edge case: header-only CSV
        header_only = base / "empty_rows.csv"
        header_only.write_text("patient_id,district,hb_g_dl\n", encoding="utf-8")
        try:
            empty_rows = fn(header_only)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Header-only CSV raised {type(exc).__name__}: {exc}")
        else:
            if empty_rows != []:
                errors.append(f"Header-only CSV: expected [], got {empty_rows!r}")

    if errors:
        return CheckResult(
            "ch09_todo_03",
            False,
            "failed",
            "To-Do 9.3 — Check csv.DictReader Loading",
            "Open the CSV with newline='' and encoding='utf-8', and return list(csv.DictReader(f)).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch09_todo_03",
        True,
        "passed",
        "To-Do 9.3 Passed — CSV Dictionary Loader Verified!",
        "Loaded tabular CSV records as a list of keyed dictionaries and returned [] for header-only files.",
        points_earned=10,
        points_possible=10,
        tip="Remember: csv.DictReader loads every column value as a string ('10.8', not 10.8) — convert explicitly with float() or int() before math.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_todo_03",
        chapter="ch09",
        title="To-Do 9.3: Read CSV as List of Dicts",
        book_section="Section 9.3.2 (Reading CSVs with csv.DictReader)",
        points=10,
        hints={
            1: "Open `csv_path` with `encoding='utf-8'` and `newline=''`, pass the file handle to `csv.DictReader(f)`, and materialize it as a `list` inside the `with` block.",
            2: "Algorithm:\n1. with open(csv_path, 'r', encoding='utf-8', newline='') as f:\n       return list(csv.DictReader(f))",
            3: "def read_csv_as_dicts(csv_path):\n    with open(csv_path, 'r', encoding='utf-8', newline='') as f:\n        return list(csv.___(f))",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 9.4: Mean of a CSV Column (Skipping "", "N/A", "999")
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch09_todo_04", "To-Do 9.4", fn, "mean_csv_column", 10)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        nominal_csv = base / "lab_results.csv"
        nominal_csv.write_text(
            "patient_id,hb_g_dl,anc_visits\n"
            "P001,10.4,4\n"
            "P002,,3\n"
            "P003,11.8,999\n"
            "P004,N/A,5\n"
            "P005,999,N/A\n"
            "P006,10.8,\n",
            encoding="utf-8",
        )

        try:
            nominal = fn(nominal_csv, "hb_g_dl")
        except NotImplementedError:
            raise
        except ValueError as exc:
            return CheckResult(
                "ch09_todo_04",
                False,
                "failed",
                "To-Do 9.4 — ValueError Converting Missing String to float",
                f"mean_csv_column(csv_path, 'hb_g_dl') raised ValueError: {exc}",
                details=[
                    "Check `if raw in ('', 'N/A', '999'): continue` BEFORE calling `float(raw)`!"
                ],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch09_todo_04",
                False,
                "failed",
                "To-Do 9.4 — Runtime Error on Nominal CSV",
                f"mean_csv_column(csv_path, 'hb_g_dl') raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if nominal is None:
            return _none_return_trap(
                "ch09_todo_04", "To-Do 9.4", "mean_csv_column(csv_path, 'hb_g_dl')", 10
            )

        errors = []
        if _is_number(nominal) and nominal > 100:
            errors.append(
                f"mean_csv_column(..., 'hb_g_dl') returned {nominal!r} — sentinel '999' was not skipped!"
            )
        elif not _is_number(nominal) or not math.isclose(nominal, 11.0, abs_tol=1e-2):
            errors.append(
                f"mean_csv_column(..., 'hb_g_dl'): expected 11.0 (from 10.4, 11.8, 10.8), got {nominal!r}"
            )

        # Second column test (verifies col_name parameter isn't hardcoded)
        try:
            anc_mean = fn(nominal_csv, "anc_visits")
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"mean_csv_column(..., 'anc_visits') raised {type(exc).__name__}: {exc}")
        else:
            if not _is_number(anc_mean) or not math.isclose(anc_mean, 4.0, abs_tol=1e-2):
                errors.append(
                    f"mean_csv_column(..., 'anc_visits'): expected 4.0 (from 4, 3, 5), got {anc_mean!r}"
                )

        # Rounding check (non-terminating decimal)
        round_csv = base / "rounding.csv"
        round_csv.write_text(
            "id,score\n1,2\n2,3\n3,5\n",
            encoding="utf-8",
        )
        try:
            rounded = fn(round_csv, "score")
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Rounding CSV raised {type(exc).__name__}: {exc}")
        else:
            if not _is_number(rounded) or not math.isclose(rounded, 3.33, abs_tol=1e-2):
                errors.append(f"Rounding CSV ('score' = 2, 3, 5): expected 3.33, got {rounded!r}")

        # All-invalid values -> None
        invalid_csv = base / "all_invalid.csv"
        invalid_csv.write_text(
            "id,hb_g_dl\n1,\n2,N/A\n3,999\n",
            encoding="utf-8",
        )
        try:
            none_res = fn(invalid_csv, "hb_g_dl")
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                "All-invalid CSV raised ZeroDivisionError — return None when valid count is 0!"
            )
        except Exception as exc:
            errors.append(f"All-invalid CSV raised {type(exc).__name__}: {exc}")
        else:
            if none_res is not None:
                errors.append(f"All-invalid CSV: expected None, got {none_res!r}")

        # Header-only CSV -> None
        header_only = base / "empty_col.csv"
        header_only.write_text("id,hb_g_dl\n", encoding="utf-8")
        try:
            empty_res = fn(header_only, "hb_g_dl")
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                "Header-only CSV raised ZeroDivisionError — return None when no valid values exist!"
            )
        except Exception as exc:
            errors.append(f"Header-only CSV raised {type(exc).__name__}: {exc}")
        else:
            if empty_res is not None:
                errors.append(f"Header-only CSV: expected None, got {empty_res!r}")

    if errors:
        return CheckResult(
            "ch09_todo_04",
            False,
            "failed",
            "To-Do 9.4 — Check Sentinel Filtering, float() Conversion & Empty Guard",
            "Skip '', 'N/A', and '999', convert valid values with float(), round mean to 2 decimals, or return None if none are valid.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch09_todo_04",
        True,
        "passed",
        "To-Do 9.4 Passed — Sentinel-Safe CSV Column Mean Verified!",
        "Filtered '', 'N/A', and '999', averaged valid floats across multiple columns, and returned None on empty batches.",
        points_earned=10,
        points_possible=10,
        tip="Stripping whitespace before checking `val in {'', 'N/A', '999'}` guards against stray spaces in field spreadsheets.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_todo_04",
        chapter="ch09",
        title="To-Do 9.4: Sentinel-Safe CSV Column Mean",
        book_section="Section 9.3.2 (Parsing Numeric Columns from CSV)",
        points=10,
        hints={
            1: "Read rows with `csv.DictReader`, grab `raw = row[col_name].strip()`, skip if `raw in ('', 'N/A', '999')`, collect `float(raw)`, and return `None` if empty else `round(sum(vals) / len(vals), 2)`.",
            2: "Algorithm:\n1. values = []\n2. with open(csv_path, 'r', encoding='utf-8', newline='') as f:\n       for row in csv.DictReader(f):\n           raw = row[col_name].strip()\n           if raw in ('', 'N/A', '999'): continue\n           values.append(float(raw))\n3. if not values: return None\n4. return round(sum(values) / len(values), 2)",
            3: "def mean_csv_column(csv_path, col_name):\n    values = []\n    with open(csv_path, 'r', encoding='utf-8', newline='') as f:\n        for row in csv.DictReader(f):\n            raw = row[col_name].strip()\n            if raw in ('', 'N/A', '999'):\n                continue\n            values.append(float(___))\n    if not values:\n        return None\n    return round(sum(values) / len(values), 2)",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 9.5: Save Study Metadata to Indented UTF-8 JSON
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch09_todo_05", "To-Do 9.5", fn, "save_study_metadata", 10)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        meta_path = base / "study_meta.json"
        sample_meta = {
            "study_id": "MCH-BD-2026",
            "sites": ["Dhaka", "Sylhet", "Khulna"],
            "target_n": 1200,
            "active": True,
            "cutoffs": {"anemia_g_dl": 11.0, "fever_c": 38.0},
        }

        try:
            fn(meta_path, sample_meta)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch09_todo_05",
                False,
                "failed",
                "To-Do 9.5 — Runtime Error Writing JSON",
                f"save_study_metadata(json_path, metadata_dict) raised {type(exc).__name__}: {exc}",
                details=[
                    "Use `with open(json_path, 'w', encoding='utf-8') as f:` and `json.dump(metadata_dict, f, indent=2)`."
                ],
                points_earned=0,
                points_possible=10,
            )

        errors = []
        if not meta_path.exists():
            errors.append("JSON file was not created at json_path.")
        else:
            raw = meta_path.read_text(encoding="utf-8")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"Output file is not valid JSON ({exc}) — use `json.dump(...)`, not `f.write(str(metadata_dict))`!"
                )
            else:
                if parsed != sample_meta:
                    errors.append(f"Parsed JSON mismatch: expected {sample_meta!r}, got {parsed!r}")
                if "\n" not in raw.strip():
                    errors.append(
                        "JSON was written on a single line — pass `indent=2` to `json.dump(...)` so it is human-readable!"
                    )

        # Edge case: empty dict
        empty_path = base / "empty_meta.json"
        try:
            fn(empty_path, {})
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Empty metadata dict raised {type(exc).__name__}: {exc}")
        else:
            if not empty_path.exists():
                errors.append("Empty metadata dict did not create json_path.")
            else:
                try:
                    parsed_empty = json.loads(empty_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    errors.append(f"Empty metadata file is invalid JSON: {exc}")
                else:
                    if parsed_empty != {}:
                        errors.append(f"Empty metadata dict: expected {{}}, got {parsed_empty!r}")

    if errors:
        return CheckResult(
            "ch09_todo_05",
            False,
            "failed",
            "To-Do 9.5 — Check Indented UTF-8 JSON Serialization",
            "Write metadata_dict to json_path in UTF-8 using json.dump(metadata_dict, f, indent=2).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch09_todo_05",
        True,
        "passed",
        "To-Do 9.5 Passed — Indented Study Metadata JSON Writer Verified!",
        "Serialized nested dicts, lists, numbers, and booleans cleanly as indented UTF-8 JSON.",
        points_earned=10,
        points_possible=10,
        tip="Never serialize dicts with str(d) — Python uses single quotes and True/False, whereas JSON requires double quotes and true/false.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_todo_05",
        chapter="ch09",
        title="To-Do 9.5: Save Study Metadata as JSON",
        book_section="Section 9.3.3 (Serializing Data with json.dump)",
        points=10,
        hints={
            1: "Open `json_path` in `'w'` mode with `encoding='utf-8'`, then call `json.dump(metadata_dict, f, indent=2)`.",
            2: "Algorithm:\n1. with open(json_path, 'w', encoding='utf-8') as f:\n       json.dump(metadata_dict, f, indent=2)",
            3: "def save_study_metadata(json_path, metadata_dict):\n    with open(json_path, 'w', encoding='utf-8') as f:\n        json.dump(metadata_dict, f, indent=___)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 9.6: Safe JSON Key Lookup with Default
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch09_todo_06", "To-Do 9.6", fn, "load_json_key", 10)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        json_path = base / "config.json"
        json_path.write_text(
            json.dumps(
                {
                    "study_id": "MCH-2026",
                    "missing_count": 0,
                    "blinded": False,
                    "anemia_cutoff": 11.0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Nominal existing key (tests default=None parameter signature too)
        try:
            nominal = fn(json_path, "study_id")
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch09_todo_06",
                False,
                "failed",
                "To-Do 9.6 — Missing Default Parameter `default=None`",
                f"Calling load_json_key(json_path, 'study_id') raised TypeError: {exc}",
                details=["Define the signature as: def load_json_key(json_path, key, default=None):"],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch09_todo_06",
                False,
                "failed",
                "To-Do 9.6 — Runtime Error Loading JSON",
                f"load_json_key(json_path, 'study_id') raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if nominal is None:
            return _none_return_trap(
                "ch09_todo_06", "To-Do 9.6", "load_json_key(json_path, 'study_id')", 10
            )

        errors = []
        if nominal != "MCH-2026":
            errors.append(f"load_json_key(path, 'study_id'): expected 'MCH-2026', got {nominal!r}")

        tests = [
            (("missing_count",), {"default": 99}, 0, "existing falsy int 0"),
            (("blinded",), {"default": True}, False, "existing falsy bool False"),
            (("nonexistent_key",), {}, None, "missing key with implicit default=None"),
            (("protocol",), {"default": "v1.0"}, "v1.0", "missing key with explicit default"),
        ]
        for args, kwargs, expected, desc in tests:
            try:
                got = fn(json_path, *args, **kwargs)
            except NotImplementedError:
                raise
            except KeyError as exc:
                errors.append(
                    f"Case ({desc}) raised KeyError: {exc} — use `data.get(key, default)` instead of `data[key]`!"
                )
                continue
            except Exception as exc:
                errors.append(f"Case ({desc}) raised {type(exc).__name__}: {exc}")
                continue
            if got != expected or type(got) is not type(expected):
                extra = ""
                if desc.startswith("existing falsy"):
                    extra = " (avoid `data.get(key) or default` — use `data.get(key, default)` directly!)"
                errors.append(f"Case ({desc}): expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch09_todo_06",
            False,
            "failed",
            "To-Do 9.6 — Check Safe JSON Key Retrieval",
            "Load the JSON dict with json.load(f) and return data.get(key, default).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch09_todo_06",
        True,
        "passed",
        "To-Do 9.6 Passed — Safe JSON Key Loader Verified!",
        "Handled existing keys, falsy values (0, False), missing keys, and custom default arguments cleanly.",
        points_earned=10,
        points_possible=10,
        tip="Using `data.get(key, default)` preserves valid falsy values like 0 and False while safely falling back on missing keys.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_todo_06",
        chapter="ch09",
        title="To-Do 9.6: Safe JSON Key Lookup",
        book_section="Section 9.3.3 (Loading JSON Safely with .get)",
        points=10,
        hints={
            1: "Open `json_path` in UTF-8 `'r'` mode, parse with `data = json.load(f)`, and return `data.get(key, default)`.",
            2: "Algorithm:\n1. def load_json_key(json_path, key, default=None):\n2. with open(json_path, 'r', encoding='utf-8') as f:\n       data = json.load(f)\n3. return data.get(key, default)",
            3: "def load_json_key(json_path, key, default=None):\n    with open(json_path, 'r', encoding='utf-8') as f:\n        data = json.load(f)\n    return data.___(key, default)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 9.1 (30 pts): Filter Urgent Triage Rows to CSV
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch09_exam_01", "Exam Task 9.1", fn, "filter_urgent_to_csv", 30)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        in_csv = base / "triage_all.csv"
        out_csv = base / "triage_urgent.csv"
        in_csv.write_text(
            "patient_id,district,temp_c,triage_priority\n"
            "P01,Dhaka,39.8,Urgent\n"
            "P02,Sylhet,37.1,Routine\n"
            "P03,Khulna,38.4,Moderate\n"
            "P04,Rajshahi,40.1,Urgent\n"
            "P05,Barishal,36.8,Routine\n",
            encoding="utf-8",
        )

        try:
            count = fn(in_csv, out_csv)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch09_exam_01",
                False,
                "failed",
                "Exam Task 9.1 — Runtime Error Filtering CSV",
                f"filter_urgent_to_csv(in_csv, out_csv) raised {type(exc).__name__}: {exc}",
                details=[
                    "Read with `csv.DictReader`, write matching rows with `csv.DictWriter(out_f, fieldnames=reader.fieldnames)`."
                ],
                points_earned=0,
                points_possible=30,
            )

        if count is None:
            return _none_return_trap(
                "ch09_exam_01", "Exam Task 9.1", "filter_urgent_to_csv(in_csv, out_csv)", 30
            )

        errors = []
        if type(count) is not int or count != 2:
            errors.append(f"Nominal CSV (2 Urgent of 5 rows): expected return count 2, got {count!r}")

        expected_fields = ["patient_id", "district", "temp_c", "triage_priority"]
        expected_urgent = [
            {"patient_id": "P01", "district": "Dhaka", "temp_c": "39.8", "triage_priority": "Urgent"},
            {"patient_id": "P04", "district": "Rajshahi", "temp_c": "40.1", "triage_priority": "Urgent"},
        ]
        if not out_csv.exists():
            errors.append("Output CSV file was not created at out_csv.")
        else:
            with open(out_csv, "r", encoding="utf-8", newline="") as f:
                out_reader = csv.DictReader(f)
                out_fields = out_reader.fieldnames
                out_rows = list(out_reader)
            if out_fields != expected_fields:
                errors.append(
                    f"Output CSV header mismatch: expected {expected_fields!r}, got {out_fields!r} (did you call `writer.writeheader()`?)"
                )
            elif out_rows != expected_urgent:
                errors.append(f"Output CSV rows mismatch: expected {expected_urgent!r}, got {out_rows!r}")

        # Edge case: zero Urgent rows (must still write header and return 0)
        none_in = base / "no_urgent.csv"
        none_out = base / "no_urgent_out.csv"
        none_in.write_text(
            "patient_id,district,triage_priority\n"
            "P10,Dhaka,Routine\n"
            "P11,Sylhet,Moderate\n",
            encoding="utf-8",
        )
        try:
            zero_count = fn(none_in, none_out)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Zero-urgent CSV raised {type(exc).__name__}: {exc}")
        else:
            if type(zero_count) is not int or zero_count != 0:
                errors.append(f"Zero-urgent CSV: expected return count 0, got {zero_count!r}")
            if not none_out.exists():
                errors.append("Zero-urgent CSV did not create out_csv.")
            else:
                with open(none_out, "r", encoding="utf-8", newline="") as f:
                    r0 = csv.DictReader(f)
                    f0 = r0.fieldnames
                    rows0 = list(r0)
                if f0 != ["patient_id", "district", "triage_priority"] or rows0 != []:
                    errors.append(
                        f"Zero-urgent CSV should preserve header and 0 rows; got header={f0!r}, rows={rows0!r}"
                    )

    if errors:
        return CheckResult(
            "ch09_exam_01",
            False,
            "failed",
            "Exam Task 9.1 — Check Urgent Row Filtering & DictWriter Header",
            "Copy matching header with writer.writeheader(), write only rows where triage_priority == 'Urgent', and return int count.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch09_exam_01",
        True,
        "passed",
        "Exam Task 9.1 Passed (30/30 pts) — Urgent Triage CSV Filter Verified!",
        "Filtered Urgent rows with matching headers via DictReader/DictWriter and handled 0-match files cleanly.",
        points_earned=30,
        points_possible=30,
        tip="Always call writer.writeheader() before the row loop so even 0-match output tables remain valid CSVs for downstream scripts.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_exam_01",
        chapter="ch09",
        title="Exam 9.1: Filter Urgent Triage Rows to CSV",
        book_section="Chapter 9 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Open `in_csv` with `csv.DictReader` and `out_csv` with `csv.DictWriter(out_f, fieldnames=reader.fieldnames)`. Write the header, loop over rows, write rows where `row['triage_priority'] == 'Urgent'`, and return the count.",
            2: "Algorithm:\n1. Open in_csv (r) and out_csv (w) with encoding='utf-8', newline=''\n2. reader = csv.DictReader(in_f)\n3. writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)\n4. writer.writeheader()\n5. Count and write rows where row['triage_priority'] == 'Urgent'\n6. return count",
            3: "def filter_urgent_to_csv(in_csv, out_csv):\n    count = 0\n    with open(in_csv, 'r', encoding='utf-8', newline='') as in_f, \\\n         open(out_csv, 'w', encoding='utf-8', newline='') as out_f:\n        reader = csv.DictReader(in_f)\n        writer = csv.DictWriter(out_f, fieldnames=reader.___)\n        writer.writeheader()\n        for row in reader:\n            if row['triage_priority'] == 'Urgent':\n                writer.writerow(row)\n                count += 1\n    return count",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 9.2 (35 pts): Messy MCH CSV -> Per-Residence JSON Summary
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch09_exam_02", "Exam Task 9.2", fn, "summarize_mch_csv_to_json", 35
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        in_csv = base / "mch_messy.csv"
        out_json = base / "mch_summary.json"
        in_csv.write_text(
            "mother_id,residence,anc_visits\n"
            "M01,rural,4\n"
            "M02,  Rural ,2\n"
            "M03,RURAL,999\n"
            "M04,rural,\n"
            "M05,Rural,5\n"
            "M06,urban,6\n"
            "M07,  URBAN  ,5\n"
            "M08,Urban,999\n"
            "M09,urban,4\n",
            encoding="utf-8",
        )

        try:
            summary = fn(in_csv, out_json)
        except NotImplementedError:
            raise
        except ValueError as exc:
            return CheckResult(
                "ch09_exam_02",
                False,
                "failed",
                "Exam Task 9.2 — ValueError Parsing anc_visits",
                f"summarize_mch_csv_to_json(in_csv, out_json) raised ValueError: {exc}",
                details=[
                    "Skip missing/sentinel values ('', '999', 'N/A') BEFORE calling float() on anc_visits!"
                ],
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch09_exam_02",
                False,
                "failed",
                "Exam Task 9.2 — Runtime Error on Messy MCH CSV",
                f"summarize_mch_csv_to_json(in_csv, out_json) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if summary is None:
            return _none_return_trap(
                "ch09_exam_02",
                "Exam Task 9.2",
                "summarize_mch_csv_to_json(in_csv, out_json)",
                35,
            )

        errors = []
        expected = {"Rural": 3.67, "Urban": 5.0}
        if not isinstance(summary, dict):
            errors.append(f"Expected a dict return value, got {type(summary).__name__} ({summary!r})")
        else:
            if set(summary.keys()) != set(expected.keys()):
                errors.append(
                    f"Keys mismatch: expected {sorted(expected.keys())}, got {sorted(summary.keys())} "
                    "(normalize residence with `row['residence'].strip().title()`!)"
                )
            else:
                for k, exp_val in expected.items():
                    got_val = summary[k]
                    if _is_number(got_val) and got_val > 100:
                        errors.append(
                            f"Group '{k}' mean is {got_val!r} — sentinel '999' was not skipped!"
                        )
                    elif not _is_number(got_val) or not math.isclose(got_val, exp_val, abs_tol=1e-2):
                        errors.append(f"Group '{k}': expected {exp_val}, got {got_val!r}")

        if not out_json.exists():
            errors.append("Output JSON file was not created at out_json.")
        else:
            try:
                loaded = json.loads(out_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"out_json is not valid JSON: {exc}")
            else:
                if not isinstance(loaded, dict) or set(loaded.keys()) != set(expected.keys()):
                    errors.append(f"out_json content mismatch: expected {expected!r}, got {loaded!r}")
                else:
                    for k, exp_val in expected.items():
                        if not _is_number(loaded[k]) or not math.isclose(loaded[k], exp_val, abs_tol=1e-2):
                            errors.append(f"out_json['{k}']: expected {exp_val}, got {loaded[k]!r}")

        # Second dataset test (prevents hardcoding & checks all-invalid CSV)
        in_csv_2 = base / "mch_case2.csv"
        out_json_2 = base / "mch_summary_2.json"
        in_csv_2.write_text(
            "mother_id,residence,anc_visits\n"
            "M20,semi-urban,3\n"
            "M21,Semi-Urban,4\n"
            "M22,rural,999\n"
            "M23,rural,\n",
            encoding="utf-8",
        )
        try:
            summary_2 = fn(in_csv_2, out_json_2)
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                "Case 2 (group with only invalid rows) raised ZeroDivisionError — skip invalid rows before creating/averaging groups!"
            )
        except Exception as exc:
            errors.append(f"Case 2 raised {type(exc).__name__}: {exc}")
        else:
            exp_2 = {"Semi-Urban": 3.5}
            if not isinstance(summary_2, dict) or set(summary_2.keys()) != {"Semi-Urban"} or not math.isclose(summary_2["Semi-Urban"], 3.5, abs_tol=1e-2):
                errors.append(
                    f"Case 2 (with all-invalid 'rural' rows): expected {exp_2!r}, got {summary_2!r}"
                )

    if errors:
        return CheckResult(
            "ch09_exam_02",
            False,
            "failed",
            "Exam Task 9.2 — Check Residence Normalization, Sentinel Skipping & JSON Export",
            "Normalize residence with .strip().title(), skip '' and '999' anc_visits, compute per-group mean rounded to 2 decimals, write to out_json, and return the dict.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch09_exam_02",
        True,
        "passed",
        "Exam Task 9.2 Passed (35/35 pts) — Messy MCH CSV-to-JSON Pipeline Verified!",
        "Normalized messy residence labels, filtered '' and '999', computed exact group means, and persisted valid JSON.",
        points_earned=35,
        points_possible=35,
        tip="Skipping invalid rows BEFORE creating a group key automatically prevents empty groups and ZeroDivisionError.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_exam_02",
        chapter="ch09",
        title="Exam 9.2: Messy MCH CSV to JSON Summary",
        book_section="Chapter 9 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Read rows via `csv.DictReader`, skip rows where `anc_visits.strip()` is `''`, `'999'`, or `'N/A'`, group valid floats under `row['residence'].strip().title()`, compute rounded means, save via `json.dump(..., indent=2)`, and return the summary dict.",
            2: "Algorithm:\n1. grouped = {}\n2. Read in_csv with DictReader:\n       raw_anc = row['anc_visits'].strip()\n       if raw_anc in ('', '999', 'N/A'): continue\n       res = row['residence'].strip().title()\n       grouped.setdefault(res, []).append(float(raw_anc))\n3. summary = {k: round(sum(v) / len(v), 2) for k, v in grouped.items()}\n4. Write summary to out_json with json.dump(summary, out_f, indent=2)\n5. return summary",
            3: "def summarize_mch_csv_to_json(in_csv, out_json):\n    grouped = {}\n    with open(in_csv, 'r', encoding='utf-8', newline='') as f:\n        for row in csv.DictReader(f):\n            raw = row['anc_visits'].strip()\n            if raw in ('', '999', 'N/A'):\n                continue\n            res = row['residence'].strip().___()\n            grouped.setdefault(res, []).append(float(raw))\n    summary = {k: round(sum(v) / len(v), 2) for k, v in grouped.items()}\n    with open(out_json, 'w', encoding='utf-8') as out_f:\n        json.dump(summary, out_f, indent=2)\n    return summary",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 9.3 (35 pts): Merge Multiple Daily JSON Disease Reports
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch09_exam_03", "Exam Task 9.3", fn, "merge_daily_json_reports", 35
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        day1 = base / "day1.json"
        day2 = base / "day2.json"
        day3 = base / "day3.json"
        day_empty = base / "day_empty.json"

        day1.write_text(json.dumps({"Dengue": 14, "Cholera": 3}), encoding="utf-8")
        day2.write_text(json.dumps({"Dengue": 19, "Malaria": 5}), encoding="utf-8")
        day3.write_text(
            json.dumps({"Cholera": 7, "Dengue": 11, "Typhoid": 2}), encoding="utf-8"
        )
        day_empty.write_text(json.dumps({}), encoding="utf-8")

        try:
            merged = fn([day1, day2, day3])
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch09_exam_03",
                False,
                "failed",
                "Exam Task 9.3 — Runtime Error Merging JSON Reports",
                f"merge_daily_json_reports([day1, day2, day3]) raised {type(exc).__name__}: {exc}",
                details=[
                    "Loop over each path in `json_paths`, load its dict with `json.load(f)`, and add each disease count to `totals`."
                ],
                points_earned=0,
                points_possible=35,
            )

        if merged is None:
            return _none_return_trap(
                "ch09_exam_03",
                "Exam Task 9.3",
                "merge_daily_json_reports([day1, day2, day3])",
                35,
            )

        errors = []
        expected = {"Dengue": 44, "Cholera": 10, "Malaria": 5, "Typhoid": 2}
        if not isinstance(merged, dict):
            errors.append(f"Expected a dict, got {type(merged).__name__} ({merged!r})")
        elif merged != expected:
            extra = ""
            if merged.get("Dengue") == 11:
                extra = " (looks like later files overwrote earlier counts — accumulate with `totals[disease] = totals.get(disease, 0) + count`!)"
            errors.append(f"3-file merge mismatch: expected {expected!r}, got {merged!r}{extra}")

        # Single file + empty file mix
        try:
            mixed = fn([day_empty, day1])
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Empty+single JSON mix raised {type(exc).__name__}: {exc}")
        else:
            if mixed != {"Dengue": 14, "Cholera": 3}:
                errors.append(
                    f"Empty+single file mix: expected {{'Dengue': 14, 'Cholera': 3}}, got {mixed!r}"
                )

        # Empty path list
        try:
            empty_merged = fn([])
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Empty path list [] raised {type(exc).__name__}: {exc}")
        else:
            if empty_merged != {}:
                errors.append(f"Empty path list []: expected {{}}, got {empty_merged!r}")

    if errors:
        return CheckResult(
            "ch09_exam_03",
            False,
            "failed",
            "Exam Task 9.3 — Check Multi-File JSON Accumulation",
            "Load every JSON file in json_paths and sum counts per disease into one combined dict.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch09_exam_03",
        True,
        "passed",
        "Exam Task 9.3 Passed (35/35 pts) — Multi-File JSON Aggregator Verified!",
        "Merged overlapping disease counts across multiple daily JSON files and handled empty inputs cleanly.",
        points_earned=35,
        points_possible=35,
        tip="Using `totals[k] = totals.get(k, 0) + count` scales cleanly from 3 clinic files to 365 daily surveillance exports.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch09_exam_03",
        chapter="ch09",
        title="Exam 9.3: Merge Daily JSON Disease Reports",
        book_section="Chapter 9 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Start with `totals = {}`, loop over every `path` in `json_paths`, load `report = json.load(f)`, and add each `count` via `totals[disease] = totals.get(disease, 0) + count`.",
            2: "Algorithm:\n1. totals = {}\n2. for path in json_paths:\n       with open(path, 'r', encoding='utf-8') as f:\n           report = json.load(f)\n       for disease, count in report.items():\n           totals[disease] = totals.get(disease, 0) + count\n3. return totals",
            3: "def merge_daily_json_reports(json_paths):\n    totals = {}\n    for path in json_paths:\n        with open(path, 'r', encoding='utf-8') as f:\n            report = json.load(f)\n        for disease, count in report.items():\n            totals[disease] = totals.get(disease, ___) + count\n    return totals",
        },
        verifier=_verify_exam_03,
    )
)
