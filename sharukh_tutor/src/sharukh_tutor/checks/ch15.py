"""Chapter 15 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch15_todo_01..06) and the Part III Capstone Coding Exam
(ch15_exam_01..03) for Version Control with Git & GitHub, Clean Code, Type
Hints, Logging, and Reproducible Research Project Templates.
Every verifier isolates filesystem and logging side effects inside
tempfile.TemporaryDirectory() so zero stray files remain behind.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
            "Remember: print() only displays text on screen; it returns None to the caller.",
            "Make sure your function ends with an explicit `return ...` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reproducible engineering helpers must return structured values so CI checks and pipelines can inspect them.",
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 15.1: Audit .gitignore Coverage with fnmatch
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_todo_01", "To-Do 15.1", fn, "check_gitignore_coverage", 10
    )
    if bad is not None:
        return bad

    sample_gitignore = (
        "# Ignore raw patient exports and secrets\n"
        "*.csv\n"
        ".env\n"
        "\n"
        "# Logs and cache\n"
        "*.log\n"
        "__pycache__/*\n"
        "# *.py\n"
    )
    sample_files = [
        "utils.py",
        "raw_patients.csv",
        "README.md",
        ".env",
        "pipeline.log",
        "analysis.py",
        "__pycache__/utils.cpython-312.pyc",
    ]
    expected_nominal = ["README.md", "analysis.py", "utils.py"]

    try:
        nominal = fn(sample_gitignore, list(sample_files))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch15_todo_01",
            False,
            "failed",
            "To-Do 15.1 — Runtime Error Checking .gitignore Patterns",
            f"check_gitignore_coverage(...) raised {type(exc).__name__}: {exc}",
            details=[
                "Split gitignore_text with .splitlines(), skip blank and '#' comment lines, and test each filename with fnmatch.fnmatch(fname, pat)."
            ],
            points_earned=0,
            points_possible=10,
        )

    if nominal is None:
        return _none_return_trap(
            "ch15_todo_01",
            "To-Do 15.1",
            "check_gitignore_coverage(gitignore_text, filenames)",
            10,
        )

    errors = []
    if not isinstance(nominal, list):
        errors.append(f"Expected a list of unignored filenames, got {type(nominal).__name__} ({nominal!r})")
    elif nominal != expected_nominal:
        ignored_expected = sorted(set(sample_files) - set(expected_nominal))
        extra = ""
        if sorted(nominal) == ignored_expected:
            extra = " (you returned IGNORED files instead of UNIGNORED files — negate the match check!)"
        elif sorted(nominal) == expected_nominal:
            extra = " (right files, but return the list sorted alphabetically with `sorted(...)`!)"
        elif "analysis.py" not in nominal and "utils.py" not in nominal:
            extra = " (comment line `# *.py` should be skipped, not treated as a pattern!)"
        errors.append(
            f"Nominal .gitignore check: expected {expected_nominal!r}, got {nominal!r}{extra}"
        )

    # Edge cases
    edge_cases = [
        ("", ["b.py", "a.md"], ["a.md", "b.py"], "Empty .gitignore"),
        ("# only comments\n\n# *.json\n", ["meta.json", "app.py"], ["app.py", "meta.json"], "Comments-only .gitignore"),
        ("*.csv\n*.json\n", ["data.csv", "config.json"], [], "All files ignored"),
        ("*.csv\n", [], [], "Empty filenames list"),
    ]
    for gi_text, files, expected, label in edge_cases:
        try:
            got = fn(gi_text, list(files))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch15_todo_01",
            False,
            "failed",
            "To-Do 15.1 — Check .gitignore Pattern Parsing & Unignored Filter",
            "Ignore blank/# comment lines, match patterns via fnmatch.fnmatch, and return the sorted list of UNIGNORED filenames.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch15_todo_01",
        True,
        "passed",
        "To-Do 15.1 Passed — .gitignore Coverage Auditor Verified!",
        "Filtered comment/blank lines, matched glob patterns with fnmatch, and returned sorted unignored filenames.",
        points_earned=10,
        points_possible=10,
        tip="In research repos, auditing which files remain tracked prevents accidental commits of raw clinical CSVs or .env secrets.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_todo_01",
        chapter="ch15",
        title="To-Do 15.1: Audit .gitignore Coverage",
        book_section="Section 15.3.1 (Protecting Sensitive Data with .gitignore)",
        points=10,
        hints={
            1: "Extract non-empty, non-comment patterns from `gitignore_text.splitlines()`, then keep filenames where `not any(fnmatch.fnmatch(fname, pat) for pat in patterns)` and return them sorted.",
            2: "Algorithm:\n1. patterns = [line.strip() for line in gitignore_text.splitlines() if line.strip() and not line.strip().startswith('#')]\n2. unignored = [f for f in filenames if not any(fnmatch.fnmatch(f, p) for p in patterns)]\n3. return sorted(unignored)",
            3: "import fnmatch\n\ndef check_gitignore_coverage(gitignore_text, filenames):\n    patterns = [\n        line.strip()\n        for line in gitignore_text.splitlines()\n        if line.strip() and not line.strip().startswith('#')\n    ]\n    unignored = [\n        fname for fname in filenames\n        if not any(fnmatch.___(fname, pat) for pat in patterns)\n    ]\n    return sorted(unignored)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 15.2: Validate Git Commit Message Subject Line
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_todo_02", "To-Do 15.2", fn, "validate_commit_message", 10
    )
    if bad is not None:
        return bad

    try:
        nominal = fn("Add SHA-256 provenance verifier for cohort CSVs")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch15_todo_02",
            False,
            "failed",
            "To-Do 15.2 — Runtime Error Validating Commit Message",
            f"validate_commit_message(...) raised {type(exc).__name__}: {exc}",
            details=[
                "Extract the first line with `message.splitlines()[0].strip()` (guarding empty input) and return `(is_valid, reason)`."
            ],
            points_earned=0,
            points_possible=10,
        )

    if nominal is None:
        return _none_return_trap(
            "ch15_todo_02", "To-Do 15.2", "validate_commit_message(message)", 10
        )

    errors = []

    def _check_case(msg: str, exp_valid: bool, label: str) -> None:
        try:
            res = fn(msg)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            return
        if not isinstance(res, tuple) or len(res) != 2:
            errors.append(
                f"{label}: expected a 2-tuple (bool, str), got {type(res).__name__} ({res!r})"
            )
            return
        flag, reason = res
        if type(flag) is not bool:
            errors.append(f"{label}: first tuple element must be bool, got {type(flag).__name__}")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label}: second tuple element must be a non-empty str, got {reason!r}")
        if flag is not exp_valid:
            extra = ""
            if "multi-line" in label.lower() and not flag:
                extra = " (check ONLY the first subject line, not the full body length or body period!)"
            errors.append(f"{label}: expected is_valid={exp_valid}, got {res!r}{extra}")

    _check_case("Add SHA-256 provenance verifier for cohort CSVs", True, "Nominal valid subject")
    _check_case(
        "Fix hemoglobin sentinel filter\n\nPreviously rows coded 999 were averaged into district means across all 8 divisions.",
        True,
        "Valid multi-line commit (long body ending in period)",
    )
    _check_case("Fix 1", True, "Boundary valid length (5 chars)")
    _check_case("A" * 72, True, "Boundary valid length (72 chars)")
    _check_case("", False, "Empty string")
    _check_case("   \n\nBody text only", False, "Blank first subject line")
    _check_case("Fix!", False, "Too short (4 chars)")
    _check_case("A" * 73, False, "Too long (73 chars)")
    _check_case("Add unit tests for cleaning pipeline.", False, "Trailing period on subject")

    if errors:
        return CheckResult(
            "ch15_todo_02",
            False,
            "failed",
            "To-Do 15.2 — Check Commit Subject Rules & (bool, str) Return",
            "Validate only the first subject line: non-empty, 5 <= len <= 72, no trailing '.', returning (is_valid, reason).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch15_todo_02",
        True,
        "passed",
        "To-Do 15.2 Passed — Git Commit Message Validator Verified!",
        "Validated subject length boundaries (5–72), trailing periods, empty subjects, and multi-line commit bodies.",
        points_earned=10,
        points_possible=10,
        tip="Keeping commit subjects under 72 characters prevents ugly line wrapping in `git log --oneline` and GitHub PR headers.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_todo_02",
        chapter="ch15",
        title="To-Do 15.2: Validate Commit Message",
        book_section="Section 15.3.1 (Writing Clean Git Commit Messages)",
        points=10,
        hints={
            1: "Split `message` into lines to get the first line (`subject = lines[0].strip() if lines else ''`), then check non-empty, `5 <= len(subject) <= 72`, and `not subject.endswith('.')`.",
            2: "Algorithm:\n1. lines = message.splitlines()\n2. subject = lines[0].strip() if lines else ''\n3. If not subject -> return (False, 'Empty subject line')\n4. If len(subject) < 5 -> return (False, 'Subject too short')\n5. If len(subject) > 72 -> return (False, 'Subject exceeds 72 chars')\n6. If subject.endswith('.') -> return (False, 'Subject ends with period')\n7. return (True, 'Valid commit subject')",
            3: "def validate_commit_message(message):\n    lines = message.splitlines()\n    subject = lines[0].strip() if lines else ''\n    if not subject:\n        return (False, 'Empty subject line')\n    if len(subject) < 5:\n        return (False, 'Subject too short')\n    if len(subject) > 72:\n        return (False, 'Subject too long')\n    if subject.___('.'):\n        return (False, 'No trailing period')\n    return (True, 'OK')",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 15.3: Scan Source Text for Leaked Secrets
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_todo_03", "To-Do 15.3", fn, "scan_for_leaked_secrets", 10
    )
    if bad is not None:
        return bad

    sample_text = (
        "# Safe configuration header\n"
        "STUDY_ID = 'MCH-BD-2026'\n"
        "TOKEN = 'ghp_1234567890abcdef'\n"
        "api_key=lowercase_should_not_match\n"
        "MAPS = 'AIzaSyFakeKey999'\n"
        "OPENAI_KEY = 'sk-proj-demo999'\n"
        "API_KEY=sk-double-pattern-same-line\n"
        "password=lowercase_safe_note\n"
        "PASSWORD=super_secret_db_pass\n"
    )
    expected_nominal = [3, 5, 6, 7, 9]

    try:
        nominal = fn(sample_text)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch15_todo_03",
            False,
            "failed",
            "To-Do 15.3 — Runtime Error Scanning for Secrets",
            f"scan_for_leaked_secrets(text) raised {type(exc).__name__}: {exc}",
            details=[
                "Loop with `for line_no, line in enumerate(text.splitlines(), start=1):` and check the 5 patterns case-sensitively."
            ],
            points_earned=0,
            points_possible=10,
        )

    if nominal is None:
        return _none_return_trap(
            "ch15_todo_03", "To-Do 15.3", "scan_for_leaked_secrets(text)", 10
        )

    errors = []
    if not isinstance(nominal, list):
        errors.append(f"Expected a list[int] of 1-based line numbers, got {type(nominal).__name__} ({nominal!r})")
    elif nominal != expected_nominal:
        extra = ""
        if nominal == [n - 1 for n in expected_nominal]:
            extra = " (off-by-one! Use 1-based line numbers with `enumerate(text.splitlines(), start=1)`!)"
        elif len(nominal) > len(set(nominal)):
            extra = " (duplicate line number when a line matches 2 patterns — use `any(p in line for p in patterns)`!)"
        elif 4 in nominal or 8 in nominal:
            extra = " (matching must be CASE-SENSITIVE — lowercase `api_key=` / `password=` should not match!)"
        errors.append(
            f"Mixed config scan: expected {expected_nominal!r}, got {nominal!r}{extra}"
        )

    edge_cases = [
        ("STUDY_ID = 'BD-2026'\n# Clean script\nprint('ok')\n", [], "Clean file"),
        ("", [], "Empty text"),
        ("ghp_abc\n", [1], "Single-line match at line 1"),
        ("GHP_fake\naiza_fake\nSK-fake\n", [], "Wrong-case patterns only"),
    ]
    for txt, expected, label in edge_cases:
        try:
            got = fn(txt)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch15_todo_03",
            False,
            "failed",
            "To-Do 15.3 — Check Case-Sensitive Secret Patterns & 1-Based Line Numbers",
            "Scan lines 1-indexed for 'ghp_', 'AIza', 'sk-', 'API_KEY=', and 'PASSWORD=' case-sensitively without duplicates.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch15_todo_03",
        True,
        "passed",
        "To-Do 15.3 Passed — Leaked-Secret Scanner Verified!",
        "Detected all five credential prefixes with exact 1-based line numbers, case sensitivity, and zero duplicates.",
        points_earned=10,
        points_possible=10,
        tip="Pre-commit hooks run scanners just like this before `git commit` finishes—stopping API key leaks before they ever reach GitHub.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_todo_03",
        chapter="ch15",
        title="To-Do 15.3: Scan for Leaked Secrets",
        book_section="Section 15.3.1 (Credential Hygiene & Pre-Commit Checks)",
        points=10,
        hints={
            1: "Define `patterns = ('ghp_', 'AIza', 'sk-', 'API_KEY=', 'PASSWORD=')`, iterate over `enumerate(text.splitlines(), start=1)`, and append `line_no` when `any(p in line for p in patterns)`.",
            2: "Algorithm:\n1. patterns = ('ghp_', 'AIza', 'sk-', 'API_KEY=', 'PASSWORD=')\n2. hits = []\n3. for line_no, line in enumerate(text.splitlines(), start=1):\n       if any(pat in line for pat in patterns):\n           hits.append(line_no)\n4. return hits",
            3: "def scan_for_leaked_secrets(text):\n    patterns = ('ghp_', 'AIza', 'sk-', 'API_KEY=', 'PASSWORD=')\n    return [\n        line_no\n        for line_no, line in enumerate(text.splitlines(), start=___)\n        if any(pat in line for pat in patterns)\n    ]",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 15.4: Append Structured Log Line Safely
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_todo_04", "To-Do 15.4", fn, "write_structured_log", 10
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "pipeline.log"

        try:
            fn(log_path, "info", "Loaded 120 maternal records")
            fn(log_path, "WARNING", "Skipped 3 sentinel rows")
            fn(log_path, "error", "Failed checksum on batch_02.csv")
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch15_todo_04",
                False,
                "failed",
                "To-Do 15.4 — Runtime Error Writing Structured Log",
                f"write_structured_log(log_path, level, message) raised {type(exc).__name__}: {exc}",
                details=[
                    "Format the line as `[{level.upper()}] {message}` and append in UTF-8, closing any logging FileHandler cleanly."
                ],
                points_earned=0,
                points_possible=10,
            )

        errors = []
        if not log_path.exists():
            errors.append("Log file was not created at log_path.")
        else:
            lines = [
                ln for ln in log_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            if len(lines) == 1:
                errors.append(
                    "Only 1 log line found after 3 calls — open/append with mode 'a' instead of overwriting with mode 'w'!"
                )
            elif len(lines) > 3:
                errors.append(
                    f"Found {len(lines)} lines after 3 calls — duplicate handlers detected! Always call `logger.removeHandler(handler)` and `handler.close()` in a `finally:` block."
                )
            elif len(lines) != 3:
                errors.append(f"Expected 3 log lines after 3 calls, got {len(lines)}: {lines!r}")
            else:
                checks = [
                    (lines[0], "[INFO]", "Loaded 120 maternal records", "Line 1 (lowercase 'info')"),
                    (lines[1], "[WARNING]", "Skipped 3 sentinel rows", "Line 2 ('WARNING')"),
                    (lines[2], "[ERROR]", "Failed checksum on batch_02.csv", "Line 3 (lowercase 'error')"),
                ]
                for line, tag, msg, label in checks:
                    if tag not in line:
                        errors.append(f"{label}: expected uppercase level tag {tag!r} in {line!r}")
                    if msg not in line:
                        errors.append(f"{label}: expected message {msg!r} in {line!r}")

    if errors:
        return CheckResult(
            "ch15_todo_04",
            False,
            "failed",
            "To-Do 15.4 — Check Structured Log Formatting, Appending & Handler Cleanup",
            "Append one '[LEVEL] message' line per call with uppercase level and properly closed file/logging handler.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch15_todo_04",
        True,
        "passed",
        "To-Do 15.4 Passed — Structured Logger Verified!",
        "Appended cleanly formatted [INFO], [WARNING], and [ERROR] lines with zero duplicate handlers or leaked file descriptors.",
        points_earned=10,
        points_possible=10,
        tip="Always remove and close logging FileHandlers inside a `finally:` block so repeated runs never duplicate log lines or lock temp directories.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_todo_04",
        chapter="ch15",
        title="To-Do 15.4: Write Structured Log Entry",
        book_section="Section 15.3.3 (Structured Logging vs. print())",
        points=10,
        hints={
            1: "Normalize `level_name = level.strip().upper()`, attach a UTF-8 `logging.FileHandler(log_path, mode='a', encoding='utf-8')` formatted as `'[%(levelname)s] %(message)s'` (or append `f'[{level_name}] {message}\\n'`), and close/remove the handler in `finally:`.",
            2: "Algorithm (using logging):\n1. Map level.strip().upper() to logging level int (raise ValueError if invalid)\n2. Create a logger with propagate=False and FileHandler(log_path, mode='a', encoding='utf-8')\n3. Set Formatter('[%(levelname)s] %(message)s')\n4. try: logger.log(level_num, message)\n   finally: logger.removeHandler(handler); handler.close()",
            3: "import logging\n\n_LEVELS = {\n    'DEBUG': logging.DEBUG, 'INFO': logging.INFO,\n    'WARNING': logging.WARNING, 'ERROR': logging.ERROR,\n    'CRITICAL': logging.CRITICAL,\n}\n\ndef write_structured_log(log_path, level, message):\n    level_name = level.strip().upper()\n    if level_name not in _LEVELS:\n        raise ValueError(f'Unknown log level: {level!r}')\n    logger = logging.getLogger('ch15_structured')\n    logger.setLevel(logging.DEBUG)\n    logger.propagate = False\n    handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')\n    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))\n    logger.addHandler(handler)\n    try:\n        logger.log(_LEVELS[level_name], message)\n    finally:\n        logger.removeHandler(handler)\n        handler.___()",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 15.5: Compute SHA-256 Digest of a File
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_todo_05", "To-Do 15.5", fn, "compute_file_sha256", 10
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        csv_file = base / "cohort_v1.csv"
        csv_bytes = b"patient_id,hb_g_dl\nP001,10.8\nP002,11.4\n"
        csv_file.write_bytes(csv_bytes)
        expected_csv_hash = _sha256_bytes(csv_bytes)

        try:
            got_csv = fn(csv_file)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch15_todo_05",
                False,
                "failed",
                "To-Do 15.5 — TypeError Hashing str Instead of bytes",
                f"compute_file_sha256(filepath) raised TypeError: {exc}",
                details=[
                    "Open the file in binary mode `'rb'` (not text mode `'r'`) so `f.read()` yields raw `bytes` for `hashlib.sha256()`."
                ],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch15_todo_05",
                False,
                "failed",
                "To-Do 15.5 — Runtime Error Computing SHA-256",
                f"compute_file_sha256(filepath) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got_csv is None:
            return _none_return_trap(
                "ch15_todo_05", "To-Do 15.5", "compute_file_sha256(filepath)", 10
            )

        errors = []
        if not isinstance(got_csv, str) or len(got_csv) != 64 or got_csv != expected_csv_hash:
            extra = ""
            if isinstance(got_csv, bytes):
                extra = " (use `.hexdigest()`, not `.digest()`, to get a 64-char hex string!)"
            elif isinstance(got_csv, str) and got_csv.lower() == expected_csv_hash:
                extra = " (return lowercase hex via `.hexdigest()`!)"
            errors.append(
                f"Nominal CSV hash: expected {expected_csv_hash!r}, got {got_csv!r}{extra}"
            )

        # Edge cases: empty file + binary non-UTF-8 bytes
        empty_file = base / "empty.bin"
        empty_file.write_bytes(b"")
        bin_file = base / "header.png"
        bin_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe"
        bin_file.write_bytes(bin_bytes)

        for path, raw_bytes, label in [
            (empty_file, b"", "Empty file"),
            (bin_file, bin_bytes, "Binary non-UTF-8 file"),
        ]:
            exp = _sha256_bytes(raw_bytes)
            try:
                got = fn(path)
            except NotImplementedError:
                raise
            except UnicodeDecodeError:
                errors.append(
                    f"{label} raised UnicodeDecodeError — open files in binary mode `'rb'`, not text mode `'r'`!"
                )
            except Exception as exc:
                errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            else:
                if got != exp:
                    errors.append(f"{label}: expected {exp!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch15_todo_05",
            False,
            "failed",
            "To-Do 15.5 — Check Binary Reading & SHA-256 Hex Digest",
            "Read raw bytes in binary mode ('rb') into hashlib.sha256() and return the lowercase 64-char .hexdigest().",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch15_todo_05",
        True,
        "passed",
        "To-Do 15.5 Passed — SHA-256 Cryptographic File Hasher Verified!",
        "Computed exact 64-character SHA-256 digests across CSV, empty, and binary files.",
        points_earned=10,
        points_possible=10,
        tip="Reading in 64 KB chunks (`while chunk := f.read(65536):`) hashes multi-gigabyte genomic datasets using almost zero RAM.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_todo_05",
        chapter="ch15",
        title="To-Do 15.5: Compute File SHA-256",
        book_section="Section 15.3.4 (Dataset Provenance with SHA-256)",
        points=10,
        hints={
            1: "Create `hasher = hashlib.sha256()`, open `filepath` in binary mode `'rb'`, feed chunks with `hasher.update(chunk)`, and return `hasher.hexdigest()`.",
            2: "Algorithm:\n1. hasher = hashlib.sha256()\n2. with open(filepath, 'rb') as f:\n       while chunk := f.read(65536):\n           hasher.update(chunk)\n3. return hasher.hexdigest()",
            3: "import hashlib\n\ndef compute_file_sha256(filepath):\n    hasher = hashlib.sha256()\n    with open(filepath, '___') as f:\n        while chunk := f.read(65536):\n            hasher.update(chunk)\n    return hasher.hexdigest()",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 15.6: Build Dataset Provenance Manifest
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_todo_06", "To-Do 15.6", fn, "build_dataset_manifest", 10
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_dir = base / "raw_data"
        data_dir.mkdir()

        f_z = data_dir / "z_cohort.csv"
        f_a = data_dir / "a_meta.json"
        f_m = data_dir / "m_labs.csv"
        f_ignore_txt = data_dir / "notes.txt"
        f_ignore_md = data_dir / "README.md"
        subdir = data_dir / "archive.csv"
        subdir.mkdir()

        z_bytes = b"id,hb\n1,10.5\n2,11.2\n"
        a_bytes = b'{"study": "MCH-2026"}\n'
        m_bytes = b"id,temp\n1,37.8\n"

        f_z.write_bytes(z_bytes)
        f_a.write_bytes(a_bytes)
        f_m.write_bytes(m_bytes)
        f_ignore_txt.write_text("do not include txt\n", encoding="utf-8")
        f_ignore_md.write_text("# ignored\n", encoding="utf-8")

        expected_keys = ["a_meta.json", "m_labs.csv", "z_cohort.csv"]
        expected_manifest = {
            "a_meta.json": {"bytes": len(a_bytes), "sha256": _sha256_bytes(a_bytes)},
            "m_labs.csv": {"bytes": len(m_bytes), "sha256": _sha256_bytes(m_bytes)},
            "z_cohort.csv": {"bytes": len(z_bytes), "sha256": _sha256_bytes(z_bytes)},
        }

        try:
            got = fn(data_dir)
        except NotImplementedError:
            raise
        except IsADirectoryError:
            return CheckResult(
                "ch15_todo_06",
                False,
                "failed",
                "To-Do 15.6 — Tried to Hash a Subdirectory",
                "build_dataset_manifest(data_dir) raised IsADirectoryError.",
                details=[
                    "Filter with `if p.is_file() and p.suffix.lower() in {'.csv', '.json'}:` before computing size and SHA-256."
                ],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch15_todo_06",
                False,
                "failed",
                "To-Do 15.6 — Runtime Error Building Manifest",
                f"build_dataset_manifest(data_dir) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch15_todo_06", "To-Do 15.6", "build_dataset_manifest(data_dir)", 10
            )

        errors = []
        if not isinstance(got, dict):
            errors.append(f"Expected a dict, got {type(got).__name__} ({got!r})")
        else:
            if list(got.keys()) != expected_keys:
                extra = ""
                if set(got.keys()) == set(expected_keys):
                    extra = " (keys must be inserted in sorted filename order!)"
                elif "notes.txt" in got or "README.md" in got:
                    extra = " (include ONLY `.csv` and `.json` files!)"
                errors.append(
                    f"Manifest keys mismatch: expected {expected_keys!r}, got {list(got.keys())!r}{extra}"
                )
            if got != expected_manifest:
                errors.append(
                    f"Manifest entries mismatch: expected {expected_manifest!r}, got {got!r}"
                )

        # Edge case: empty directory
        empty_dir = base / "empty_dir"
        empty_dir.mkdir()
        try:
            got_empty = fn(empty_dir)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Empty directory raised {type(exc).__name__}: {exc}")
        else:
            if got_empty != {}:
                errors.append(f"Empty directory: expected {{}}, got {got_empty!r}")

    if errors:
        return CheckResult(
            "ch15_todo_06",
            False,
            "failed",
            "To-Do 15.6 — Check Sorted .csv/.json Scanning & Manifest Schema",
            "Scan regular .csv and .json files in sorted name order, mapping filename -> {'bytes': int, 'sha256': str}.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch15_todo_06",
        True,
        "passed",
        "To-Do 15.6 Passed — Dataset Provenance Manifest Builder Verified!",
        "Built sorted byte-size + SHA-256 manifests for .csv/.json files while ignoring non-data files and directories.",
        points_earned=10,
        points_possible=10,
        tip="Committing a tiny `manifest.json` (while gitignoring raw `.csv` files) proves to peer reviewers that every analysis used identical data.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_todo_06",
        chapter="ch15",
        title="To-Do 15.6: Build Dataset Provenance Manifest",
        book_section="Section 15.3.4 (Dataset Manifests & Reproducibility)",
        points=10,
        hints={
            1: "Iterate over `sorted(Path(data_dir).iterdir(), key=lambda p: p.name)`, keep `p.is_file()` with `p.suffix.lower() in {'.csv', '.json'}`, and store `{'bytes': p.stat().st_size, 'sha256': ...}`.",
            2: "Algorithm:\n1. manifest = {}\n2. for p in sorted(Path(data_dir).iterdir(), key=lambda x: x.name):\n       if p.is_file() and p.suffix.lower() in {'.csv', '.json'}:\n           digest = hashlib.sha256(p.read_bytes()).hexdigest()\n           manifest[p.name] = {'bytes': p.stat().st_size, 'sha256': digest}\n3. return manifest",
            3: "import hashlib\nfrom pathlib import Path\n\ndef build_dataset_manifest(data_dir):\n    manifest = {}\n    for p in sorted(Path(data_dir).iterdir(), key=lambda x: x.name):\n        if p.is_file() and p.suffix.lower() in {'.csv', '.json'}:\n            digest = hashlib.sha256(p.___()).hexdigest()\n            manifest[p.name] = {\n                'bytes': p.stat().st_size,\n                'sha256': digest,\n            }\n    return manifest",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# PART III CAPSTONE EXAM 15.1 (30 pts): Audit Repository Hygiene
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_exam_01", "Exam Task 15.1", fn, "audit_repo_hygiene", 30
    )
    if bad is not None:
        return bad

    expected_keys = {
        "has_readme",
        "has_gitignore",
        "has_env_leak",
        "has_checkpoints",
        "ready_to_push",
    }

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Scenario 1: Clean, push-ready research repo
        clean_repo = base / "repo_clean"
        clean_repo.mkdir()
        (clean_repo / "README.md").write_text("# Study Repo\n", encoding="utf-8")
        (clean_repo / ".gitignore").write_text(".env\n*.csv\n", encoding="utf-8")
        (clean_repo / "src").mkdir()

        # Scenario 2: Leaked .env file
        leak_repo = base / "repo_leak"
        leak_repo.mkdir()
        (leak_repo / "README.md").write_text("# Study\n", encoding="utf-8")
        (leak_repo / ".gitignore").write_text("*.csv\n", encoding="utf-8")
        (leak_repo / ".env").write_text("API_KEY=sk-secret\n", encoding="utf-8")

        # Scenario 3: Stray .ipynb_checkpoints directory
        ckpt_repo = base / "repo_ckpt"
        ckpt_repo.mkdir()
        (ckpt_repo / "README.md").write_text("# Study\n", encoding="utf-8")
        (ckpt_repo / ".gitignore").write_text(".env\n", encoding="utf-8")
        (ckpt_repo / ".ipynb_checkpoints").mkdir()

        # Scenario 4: Bare repo with missing docs + both hazards
        messy_repo = base / "repo_messy"
        messy_repo.mkdir()
        (messy_repo / ".env").write_text("PASSWORD=123\n", encoding="utf-8")
        (messy_repo / ".ipynb_checkpoints").mkdir()

        scenarios = [
            (
                clean_repo,
                {
                    "has_readme": True,
                    "has_gitignore": True,
                    "has_env_leak": False,
                    "has_checkpoints": False,
                    "ready_to_push": True,
                },
                "Clean repo",
            ),
            (
                leak_repo,
                {
                    "has_readme": True,
                    "has_gitignore": True,
                    "has_env_leak": True,
                    "has_checkpoints": False,
                    "ready_to_push": False,
                },
                "Repo with .env leak",
            ),
            (
                ckpt_repo,
                {
                    "has_readme": True,
                    "has_gitignore": True,
                    "has_env_leak": False,
                    "has_checkpoints": True,
                    "ready_to_push": False,
                },
                "Repo with .ipynb_checkpoints/",
            ),
            (
                messy_repo,
                {
                    "has_readme": False,
                    "has_gitignore": False,
                    "has_env_leak": True,
                    "has_checkpoints": True,
                    "ready_to_push": False,
                },
                "Incomplete messy repo",
            ),
        ]

        try:
            first_res = fn(clean_repo)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch15_exam_01",
                False,
                "failed",
                "Exam Task 15.1 — Runtime Error Auditing Repository",
                f"audit_repo_hygiene(repo_dir) raised {type(exc).__name__}: {exc}",
                details=[
                    "Convert `repo_dir` via `root = Path(repo_dir)` and inspect `README.md`, `.gitignore`, `.env`, and `.ipynb_checkpoints`."
                ],
                points_earned=0,
                points_possible=30,
            )

        if first_res is None:
            return _none_return_trap(
                "ch15_exam_01", "Exam Task 15.1", "audit_repo_hygiene(repo_dir)", 30
            )

        errors = []
        for repo_path, exp, label in scenarios:
            try:
                got = fn(repo_path)
            except NotImplementedError:
                raise
            except Exception as exc:
                errors.append(f"{label} raised {type(exc).__name__}: {exc}")
                continue

            if not isinstance(got, dict):
                errors.append(f"{label}: expected dict, got {type(got).__name__}")
                continue
            if set(got.keys()) != expected_keys:
                errors.append(
                    f"{label}: expected keys {sorted(expected_keys)}, got {sorted(got.keys())}"
                )
                continue
            for k, exp_v in exp.items():
                if type(got[k]) is not bool or got[k] is not exp_v:
                    errors.append(
                        f"{label} ['{k}']: expected {exp_v!r}, got {got[k]!r}"
                    )

    if errors:
        return CheckResult(
            "ch15_exam_01",
            False,
            "failed",
            "Exam Task 15.1 — Check Repository Hygiene Flags & ready_to_push Logic",
            "Return all 5 booleans (has_readme, has_gitignore, has_env_leak, has_checkpoints, ready_to_push).",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch15_exam_01",
        True,
        "passed",
        "Exam Task 15.1 Passed (30/30 pts) — Repository Hygiene Auditor Verified!",
        "Accurately audited documentation, .gitignore presence, .env leaks, notebook checkpoints, and push readiness.",
        points_earned=30,
        points_possible=30,
        tip="Running an automated repo-hygiene script before `git push` protects both your credentials and your collaborators' clones.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_exam_01",
        chapter="ch15",
        title="Exam 15.1: Audit Repository Hygiene",
        book_section="Chapter 15 Part III Capstone Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Wrap `root = Path(repo_dir)` and compute `has_readme = (root / 'README.md').is_file()`, `has_gitignore = (root / '.gitignore').is_file()`, `has_env_leak = (root / '.env').is_file()`, `has_checkpoints = (root / '.ipynb_checkpoints').is_dir()`, and `ready_to_push`.",
            2: "Algorithm:\n1. root = Path(repo_dir)\n2. has_readme = (root / 'README.md').is_file()\n3. has_gitignore = (root / '.gitignore').is_file()\n4. has_env_leak = (root / '.env').is_file()\n5. has_checkpoints = (root / '.ipynb_checkpoints').is_dir()\n6. ready = has_readme and has_gitignore and not has_env_leak and not has_checkpoints\n7. Return dict with all 5 boolean keys.",
            3: "from pathlib import Path\n\ndef audit_repo_hygiene(repo_dir):\n    root = Path(repo_dir)\n    has_readme = (root / 'README.md').is_file()\n    has_gitignore = (root / '.gitignore').is_file()\n    has_env_leak = (root / '.env').is_file()\n    has_checkpoints = (root / '.ipynb_checkpoints').___()\n    ready = (\n        has_readme and has_gitignore\n        and not has_env_leak and not has_checkpoints\n    )\n    return {\n        'has_readme': has_readme,\n        'has_gitignore': has_gitignore,\n        'has_env_leak': has_env_leak,\n        'has_checkpoints': has_checkpoints,\n        'ready_to_push': ready,\n    }",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# PART III CAPSTONE EXAM 15.2 (35 pts): Verify Data Provenance Against Manifest
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_exam_02", "Exam Task 15.2", fn, "verify_data_provenance", 35
    )
    if bad is not None:
        return bad

    expected_keys = {"matched", "modified", "missing", "all_verified"}

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        data_dir = base / "cohort_data"
        data_dir.mkdir()

        a_bytes = b"id,hb_g_dl\nP1,11.2\n"
        b_bytes = b"id,anc\nP1,4\n"
        c_orig = b'{"version": 1}\n'
        c_tampered = b'{"version": 2}\n'
        d_orig = b"id,site\nP1,Dhaka\n"

        (data_dir / "a_cohort.csv").write_bytes(a_bytes)
        (data_dir / "b_labs.csv").write_bytes(b_bytes)
        (data_dir / "c_meta.json").write_bytes(c_tampered)
        # d_deleted.csv intentionally missing on disk!

        mixed_manifest = {
            "d_deleted.csv": {"bytes": len(d_orig), "sha256": _sha256_bytes(d_orig)},
            "b_labs.csv": {"bytes": len(b_bytes), "sha256": _sha256_bytes(b_bytes)},
            "c_meta.json": {"bytes": len(c_orig), "sha256": _sha256_bytes(c_orig)},
            "a_cohort.csv": {"bytes": len(a_bytes), "sha256": _sha256_bytes(a_bytes)},
        }

        try:
            got_mixed = fn(data_dir, mixed_manifest)
        except NotImplementedError:
            raise
        except FileNotFoundError as exc:
            return CheckResult(
                "ch15_exam_02",
                False,
                "failed",
                "Exam Task 15.2 — Crashed with FileNotFoundError on Missing File",
                f"verify_data_provenance(...) raised FileNotFoundError: {exc}",
                details=[
                    "Check `if not fpath.is_file(): missing.append(fname)` BEFORE trying to open and hash the file!"
                ],
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch15_exam_02",
                False,
                "failed",
                "Exam Task 15.2 — Runtime Error Verifying Provenance",
                f"verify_data_provenance(data_dir, manifest_dict) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got_mixed is None:
            return _none_return_trap(
                "ch15_exam_02",
                "Exam Task 15.2",
                "verify_data_provenance(data_dir, manifest_dict)",
                35,
            )

        errors = []
        exp_mixed = {
            "matched": ["a_cohort.csv", "b_labs.csv"],
            "modified": ["c_meta.json"],
            "missing": ["d_deleted.csv"],
            "all_verified": False,
        }
        if not isinstance(got_mixed, dict) or set(got_mixed.keys()) != expected_keys:
            errors.append(
                f"Mixed test: expected dict with keys {sorted(expected_keys)}, got {got_mixed!r}"
            )
        elif got_mixed != exp_mixed:
            errors.append(f"Mixed test mismatch: expected {exp_mixed!r}, got {got_mixed!r}")

        # Scenario 2: All verified
        intact_manifest = {
            "b_labs.csv": {"bytes": len(b_bytes), "sha256": _sha256_bytes(b_bytes)},
            "a_cohort.csv": {"bytes": len(a_bytes), "sha256": _sha256_bytes(a_bytes)},
        }
        try:
            got_ok = fn(data_dir, intact_manifest)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"All-verified test raised {type(exc).__name__}: {exc}")
        else:
            exp_ok = {
                "matched": ["a_cohort.csv", "b_labs.csv"],
                "modified": [],
                "missing": [],
                "all_verified": True,
            }
            if got_ok != exp_ok:
                errors.append(f"All-verified test: expected {exp_ok!r}, got {got_ok!r}")

        # Scenario 3: Empty manifest
        try:
            got_empty = fn(data_dir, {})
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Empty manifest test raised {type(exc).__name__}: {exc}")
        else:
            exp_empty = {
                "matched": [],
                "modified": [],
                "missing": [],
                "all_verified": True,
            }
            if got_empty != exp_empty:
                errors.append(f"Empty manifest test: expected {exp_empty!r}, got {got_empty!r}")

    if errors:
        return CheckResult(
            "ch15_exam_02",
            False,
            "failed",
            "Exam Task 15.2 — Check SHA-256 Re-Verification & Sorted Categories",
            "Classify every manifest file into sorted 'matched', 'modified', or 'missing' lists and set all_verified = not modified and not missing.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch15_exam_02",
        True,
        "passed",
        "Exam Task 15.2 Passed (35/35 pts) — Data Provenance Verifier Verified!",
        "Detected intact, silently modified, and deleted datasets with sorted outputs and accurate all_verified status.",
        points_earned=35,
        points_possible=35,
        tip="Running verify_data_provenance() at the top of your analysis notebook catches silent Excel re-saves before they skew a single p-value.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_exam_02",
        chapter="ch15",
        title="Exam 15.2: Verify Data Provenance",
        book_section="Chapter 15 Part III Capstone Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Loop over `sorted(manifest_dict)`. For each `fname`, check `(Path(data_dir) / fname).is_file()`; if missing append to `missing`, else recompute its SHA-256 and compare against `manifest_dict[fname]['sha256']`.",
            2: "Algorithm:\n1. matched, modified, missing = [], [], []\n2. for fname in sorted(manifest_dict):\n       fpath = Path(data_dir) / fname\n       if not fpath.is_file():\n           missing.append(fname)\n       else:\n           actual = hashlib.sha256(fpath.read_bytes()).hexdigest()\n           if actual == manifest_dict[fname]['sha256']:\n               matched.append(fname)\n           else:\n               modified.append(fname)\n3. return {'matched': matched, 'modified': modified, 'missing': missing, 'all_verified': not modified and not missing}",
            3: "import hashlib\nfrom pathlib import Path\n\ndef verify_data_provenance(data_dir, manifest_dict):\n    root = Path(data_dir)\n    matched, modified, missing = [], [], []\n    for fname in sorted(manifest_dict):\n        fpath = root / fname\n        if not fpath.is_file():\n            missing.append(fname)\n            continue\n        actual = hashlib.sha256(fpath.read_bytes()).hexdigest()\n        if actual == manifest_dict[fname]['___']:\n            matched.append(fname)\n        else:\n            modified.append(fname)\n    return {\n        'matched': matched,\n        'modified': modified,\n        'missing': missing,\n        'all_verified': not modified and not missing,\n    }",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# PART III CAPSTONE EXAM 15.3 (35 pts): Logged, Provenance-Tracked Pipeline
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch15_exam_03", "Exam Task 15.3", fn, "run_logged_cleaning_pipeline", 35
    )
    if bad is not None:
        return bad

    expected_keys = {"source_sha256", "valid_count", "rejected_count", "mean_hb"}

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Scenario 1: Nominal mixed CSV (4 valid, 3 rejected)
        in_csv = base / "mch_raw.csv"
        out_json = base / "mch_clean_summary.json"
        log_path = base / "mch_pipeline.log"
        csv_text = (
            "patient_id,district,hb_g_dl\n"
            "P01,Dhaka,10.4\n"
            "P02,Sylhet,\n"
            "P03,Khulna,11.8\n"
            "P04,Dhaka,N/A\n"
            "P05,Rajshahi,999\n"
            "P06,Sylhet,12.0\n"
            "P07,Khulna,10.4\n"
        )
        in_csv.write_text(csv_text, encoding="utf-8")
        exp_sha = _sha256_bytes(in_csv.read_bytes())

        try:
            res = fn(in_csv, out_json, log_path)
        except NotImplementedError:
            raise
        except ValueError as exc:
            return CheckResult(
                "ch15_exam_03",
                False,
                "failed",
                "Exam Task 15.3 — ValueError Parsing hb_g_dl",
                f"run_logged_cleaning_pipeline(...) raised ValueError: {exc}",
                details=[
                    "Check `if raw in ('', 'N/A', '999'):` and log a WARNING BEFORE calling `float(raw)`!"
                ],
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch15_exam_03",
                False,
                "failed",
                "Exam Task 15.3 — Runtime Error in Cleaning Pipeline",
                f"run_logged_cleaning_pipeline(...) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if res is None:
            return _none_return_trap(
                "ch15_exam_03",
                "Exam Task 15.3",
                "run_logged_cleaning_pipeline(in_csv, out_json, log_path)",
                35,
            )

        errors = []
        if not isinstance(res, dict) or set(res.keys()) != expected_keys:
            errors.append(
                f"Nominal return dict keys mismatch: expected {sorted(expected_keys)}, got {res!r}"
            )
        else:
            if res["source_sha256"] != exp_sha:
                errors.append(
                    f"source_sha256 mismatch: expected {exp_sha!r}, got {res['source_sha256']!r}"
                )
            if type(res["valid_count"]) is not int or res["valid_count"] != 4:
                errors.append(f"valid_count: expected 4, got {res['valid_count']!r}")
            if type(res["rejected_count"]) is not int or res["rejected_count"] != 3:
                errors.append(f"rejected_count: expected 3, got {res['rejected_count']!r}")
            if not _is_number(res["mean_hb"]) or not math.isclose(res["mean_hb"], 11.15, abs_tol=1e-2):
                errors.append(
                    f"mean_hb: expected 11.15 (mean of 10.4, 11.8, 12.0, 10.4), got {res['mean_hb']!r}"
                )

        # Verify indented JSON output
        if not out_json.exists():
            errors.append("Output JSON file was not created at out_json.")
        else:
            raw_json = out_json.read_text(encoding="utf-8")
            try:
                parsed_json = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                errors.append(f"out_json is not valid JSON: {exc}")
            else:
                if "\n" not in raw_json.strip():
                    errors.append("out_json should be indented (`json.dump(..., indent=2)`).")
                if isinstance(res, dict) and parsed_json != res:
                    errors.append(
                        f"out_json contents {parsed_json!r} do not match returned dict {res!r}."
                    )

        # Verify log file contains INFO and WARNING entries
        if not log_path.exists():
            errors.append("Log file was not created at log_path.")
        else:
            log_text = log_path.read_text(encoding="utf-8")
            if "INFO" not in log_text:
                errors.append("Log file must include at least one INFO log line.")
            if "WARNING" not in log_text:
                errors.append("Log file must include WARNING lines for skipped sentinel rows.")

        # Scenario 2: All-invalid CSV -> mean_hb must be None (no ZeroDivisionError!)
        bad_csv = base / "all_bad.csv"
        bad_json = base / "all_bad.json"
        bad_log = base / "all_bad.log"
        bad_csv.write_text(
            "patient_id,hb_g_dl\nP1,\nP2,N/A\nP3,999\n", encoding="utf-8"
        )
        bad_sha = _sha256_bytes(bad_csv.read_bytes())

        try:
            bad_res = fn(bad_csv, bad_json, bad_log)
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                "All-invalid CSV raised ZeroDivisionError — set mean_hb = None when valid_count == 0!"
            )
        except Exception as exc:
            errors.append(f"All-invalid CSV raised {type(exc).__name__}: {exc}")
        else:
            exp_bad = {
                "source_sha256": bad_sha,
                "valid_count": 0,
                "rejected_count": 3,
                "mean_hb": None,
            }
            if bad_res != exp_bad:
                errors.append(f"All-invalid CSV: expected {exp_bad!r}, got {bad_res!r}")
            if not bad_json.exists() or json.loads(bad_json.read_text(encoding="utf-8")) != exp_bad:
                errors.append("All-invalid CSV did not write matching JSON summary with mean_hb: null.")

    if errors:
        return CheckResult(
            "ch15_exam_03",
            False,
            "failed",
            "Exam Task 15.3 — Check SHA-256 Provenance, Logging, JSON & Empty Guard",
            "Hash in_csv, skip ''/'N/A'/'999' while logging INFO/WARNING to log_path, write indented JSON summary, and return the summary dict.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch15_exam_03",
        True,
        "passed",
        "Exam Task 15.3 Passed (35/35 pts) — Logged Provenance Cleaning Pipeline Verified!",
        "Combined SHA-256 hashing, sentinel-safe CSV cleaning, structured INFO/WARNING logging, and indented JSON export.",
        points_earned=35,
        points_possible=35,
        tip="You just built a publication-grade research pipeline: every derived number traces cryptographically to its raw source file and full audit log!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch15_exam_03",
        chapter="ch15",
        title="Exam 15.3: Logged Provenance Cleaning Pipeline",
        book_section="Chapter 15 Part III Capstone Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Hash `in_csv` with `hashlib.sha256`, open a UTF-8 `logging.FileHandler` (closed in `finally:`), stream `csv.DictReader` skipping `''`/`'N/A'`/`'999'` with `logger.warning(...)`, compute `mean_hb` (rounded to 2 decimals or `None`), write indented JSON to `out_json`, and return the dict.",
            2: "Algorithm:\n1. source_sha256 = hashlib.sha256(Path(in_csv).read_bytes()).hexdigest()\n2. Set up logger + FileHandler(log_path, mode='w', encoding='utf-8') inside try/finally\n3. Log INFO start; read in_csv with DictReader:\n       raw = row['hb_g_dl'].strip()\n       if raw in ('', 'N/A', '999'): rejected += 1; logger.warning(...)\n       else: valid_vals.append(float(raw))\n4. mean_hb = round(sum(valid_vals)/len(valid_vals), 2) if valid_vals else None\n5. Build summary dict, log INFO summary, dump indented JSON to out_json, return summary",
            3: "import csv, hashlib, json, logging\nfrom pathlib import Path\n\ndef run_logged_cleaning_pipeline(in_csv, out_json, log_path):\n    src_hash = hashlib.sha256(Path(in_csv).read_bytes()).hexdigest()\n    logger = logging.getLogger('ch15_pipeline')\n    logger.setLevel(logging.INFO)\n    logger.propagate = False\n    handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')\n    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))\n    logger.addHandler(handler)\n    valid_vals, rejected = [], 0\n    try:\n        logger.info('Starting pipeline: sha256=%s', src_hash)\n        with open(in_csv, 'r', encoding='utf-8', newline='') as f:\n            for idx, row in enumerate(csv.DictReader(f), start=1):\n                raw = row['hb_g_dl'].strip()\n                if raw in ('', 'N/A', '999'):\n                    rejected += 1\n                    logger.warning('Row %d skipped: %r', idx, raw)\n                else:\n                    valid_vals.append(float(raw))\n        mean_hb = round(sum(valid_vals) / len(valid_vals), 2) if valid_vals else None\n        summary = {\n            'source_sha256': src_hash,\n            'valid_count': len(valid_vals),\n            'rejected_count': rejected,\n            'mean_hb': mean_hb,\n        }\n        logger.info('Done: valid=%d rejected=%d', len(valid_vals), rejected)\n    finally:\n        logger.removeHandler(handler)\n        handler.___()\n    with open(out_json, 'w', encoding='utf-8') as out_f:\n        json.dump(summary, out_f, indent=2)\n    return summary",
        },
        verifier=_verify_exam_03,
    )
)
