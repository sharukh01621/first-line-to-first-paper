"""Chapter 14 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch14_todo_01..06) and Chapter 14 Coding Exam
(ch14_exam_01..03) for Professional Engineering with VS Code, Virtual
Environments, Modules, and Packages. Every filesystem verifier operates
inside tempfile.TemporaryDirectory() so zero stray files remain behind.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
            "Remember: print() only displays text; it sends back None to the caller.",
            "Make sure your function ends with an explicit `return ...` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reusable package helpers must return structured values so tests and CLI wrappers can inspect them.",
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 14.1: Parse requirements.txt File
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch14_todo_01", "To-Do 14.1", fn, "parse_requirements_file", 10
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        req_file = base / "requirements.txt"
        req_file.write_text(
            "# Core scientific stack\n"
            "\n"
            "Pandas==2.2.0\n"
            "  numpy>=1.26\n"
            "matplotlib<=3.9.0\n"
            "Seaborn~=0.13.2\n"
            "   \n"
            "# Unpinned statistical helper\n"
            "SciPy\n",
            encoding="utf-8",
        )

        try:
            parsed = fn(req_file)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch14_todo_01",
                False,
                "failed",
                "To-Do 14.1 — Runtime Error Parsing requirements.txt",
                f"parse_requirements_file(req_path) raised {type(exc).__name__}: {exc}",
                details=[
                    "Open req_path with `with open(req_path, 'r', encoding='utf-8') as f:`.",
                    "Skip blank lines and lines starting with `#`, then check for `==`, `>=`, `<=`, or `~=`.",
                ],
                points_earned=0,
                points_possible=10,
            )

        if parsed is None:
            return _none_return_trap(
                "ch14_todo_01", "To-Do 14.1", "parse_requirements_file(req_path)", 10
            )

        errors = []
        expected = {
            "pandas": "==2.2.0",
            "numpy": ">=1.26",
            "matplotlib": "<=3.9.0",
            "seaborn": "~=0.13.2",
            "scipy": "",
        }
        if not isinstance(parsed, dict):
            errors.append(
                f"Expected a dict mapping package -> version spec, got {type(parsed).__name__} ({parsed!r})"
            )
        elif parsed != expected:
            hints = []
            if any(k != k.lower() for k in parsed):
                hints.append("Normalize package names with `.lower()` (e.g., `'Pandas'` -> `'pandas'`).")
            if parsed.get("pandas") == "2.2.0":
                hints.append("Keep the operator prefix in the value (e.g., `'==2.2.0'`, not `'2.2.0'`).")
            if any(k.startswith("#") or k == "" for k in parsed):
                hints.append("Skip blank lines and `#` comment lines before adding keys.")
            suffix = f" ({' '.join(hints)})" if hints else ""
            errors.append(
                f"Nominal requirements.txt mismatch: expected {expected!r}, got {parsed!r}{suffix}"
            )

        # Edge case: comments and blank lines only
        empty_req = base / "empty_req.txt"
        empty_req.write_text("# Only comments\n\n   # Another note\n", encoding="utf-8")
        try:
            empty_parsed = fn(empty_req)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Comments-only file raised {type(exc).__name__}: {exc}")
        else:
            if empty_parsed != {}:
                errors.append(
                    f"Comments-only requirements.txt: expected {{}}, got {empty_parsed!r}"
                )

        # Edge case: str path input with single bare package
        single_req = base / "single.txt"
        single_req.write_text("pytest\n", encoding="utf-8")
        try:
            single_parsed = fn(str(single_req))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"String path input raised {type(exc).__name__}: {exc}")
        else:
            if single_parsed != {"pytest": ""}:
                errors.append(
                    f"Single bare package 'pytest': expected {{'pytest': ''}}, got {single_parsed!r}"
                )

    if errors:
        return CheckResult(
            "ch14_todo_01",
            False,
            "failed",
            "To-Do 14.1 — Check requirements.txt Parsing & Operator Preservation",
            "Ignore blanks/# comments, lowercase package names, keep operators (==, >=, <=, ~=), and map bare names to ''.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch14_todo_01",
        True,
        "passed",
        "To-Do 14.1 Passed — requirements.txt Parser Verified!",
        "Parsed pinned, bounded, compatible-release, and bare dependencies while ignoring comments and blank lines.",
        points_earned=10,
        points_possible=10,
        tip="Lowercasing package names prevents duplicate entries when collaborators mix 'Pandas' and 'pandas'.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_todo_01",
        chapter="ch14",
        title="To-Do 14.1: Parse requirements.txt",
        book_section="Section 14.3.2 (Virtual Environments & Dependency Files)",
        points=10,
        hints={
            1: "Read each line, strip whitespace, skip empty lines and lines starting with `#`, then check whether any operator in `('==', '>=', '<=', '~=')` appears in the line.",
            2: "Algorithm:\n1. reqs = {}\n2. For each stripped line in req_path:\n       if not line or line.startswith('#'): continue\n       check op in ('==', '>=', '<=', '~='):\n           name, ver = line.split(op, 1)\n           reqs[name.strip().lower()] = op + ver.strip()\n       else: reqs[line.lower()] = ''\n3. return reqs",
            3: "def parse_requirements_file(req_path):\n    reqs = {}\n    with open(req_path, 'r', encoding='utf-8') as f:\n        for raw in f:\n            line = raw.strip()\n            if not line or line.startswith('#'):\n                continue\n            for op in ('==', '>=', '<=', '~='):\n                if op in line:\n                    name, ver = line.split(op, 1)\n                    reqs[name.strip().___()] = op + ver.strip()\n                    break\n            else:\n                reqs[line.lower()] = ''\n    return reqs",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 14.2: Find Unpinned Packages
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch14_todo_02", "To-Do 14.2", fn, "find_unpinned_packages", 10
    )
    if bad is not None:
        return bad

    sample = {
        "scipy": "",
        "pandas": "==2.2.0",
        "numpy": ">=1.26",
        "matplotlib": "==3.9.0",
        "seaborn": "~=0.13.0",
        "statsmodels": "<=0.14.2",
    }
    snapshot = dict(sample)

    try:
        unpinned = fn(sample)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch14_todo_02",
            False,
            "failed",
            "To-Do 14.2 — Runtime Error Inspecting Dependency Dict",
            f"find_unpinned_packages(req_dict) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if unpinned is None:
        return _none_return_trap(
            "ch14_todo_02", "To-Do 14.2", "find_unpinned_packages(req_dict)", 10
        )

    errors = []
    expected = ["numpy", "scipy", "seaborn", "statsmodels"]
    if not isinstance(unpinned, list):
        errors.append(f"Expected a sorted list, got {type(unpinned).__name__} ({unpinned!r})")
    elif unpinned != expected:
        extra = ""
        if sorted(unpinned) == expected:
            extra = " (your items match, but remember to return them in sorted alphabetical order!)"
        elif set(unpinned) == {"pandas", "matplotlib"}:
            extra = " (inverted condition — return packages whose spec does NOT start with `'=='`!)"
        errors.append(f"Nominal dict: expected {expected!r}, got {unpinned!r}{extra}")

    if sample != snapshot:
        errors.append("Do not mutate the input `req_dict` in place.")

    edge_cases = [
        ({"pandas": "==2.2.0", "numpy": "==1.26.4"}, [], "All-pinned dict"),
        ({}, [], "Empty dict"),
        ({"zlib": "", "altair": ">=5.0"}, ["altair", "zlib"], "All-unpinned unsorted dict"),
    ]
    for data, exp, label in edge_cases:
        try:
            got = fn(dict(data))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if got != exp:
            errors.append(f"{label}: expected {exp!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch14_todo_02",
            False,
            "failed",
            "To-Do 14.2 — Check Unpinned Package Filtering & Sorting",
            "Return a sorted list of package names whose version spec does NOT start with '=='.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch14_todo_02",
        True,
        "passed",
        "To-Do 14.2 Passed — Unpinned Dependency Auditor Verified!",
        "Flagged bare, >=, <=, and ~= packages in deterministic alphabetical order while accepting exact == pins.",
        points_earned=10,
        points_possible=10,
        tip="Pinning exact versions (`==`) ensures reviewers running your manuscript code next year get identical numerical results.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_todo_02",
        chapter="ch14",
        title="To-Do 14.2: Find Unpinned Packages",
        book_section="Section 14.3.2 (Reproducible Dependency Pinning)",
        points=10,
        hints={
            1: "Filter `req_dict.items()` for entries where `not spec.startswith('==')` and sort the resulting package names.",
            2: "Algorithm:\n1. Collect `pkg` for `pkg, spec in req_dict.items()` whenever `not spec.startswith('==')`\n2. Return `sorted(...)`",
            3: "def find_unpinned_packages(req_dict):\n    return sorted(\n        pkg for pkg, spec in req_dict.items()\n        if not spec.___('==')\n    )",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 14.3: Build CLI Parser with argparse
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch14_todo_03", "To-Do 14.3", fn, "build_cli_parser", 10)
    if bad is not None:
        return bad

    try:
        parser = fn()
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch14_todo_03",
            False,
            "failed",
            "To-Do 14.3 — Runtime Error Building ArgumentParser",
            f"build_cli_parser() raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if parser is None:
        return _none_return_trap("ch14_todo_03", "To-Do 14.3", "build_cli_parser()", 10)

    if not isinstance(parser, argparse.ArgumentParser):
        return CheckResult(
            "ch14_todo_03",
            False,
            "failed",
            "To-Do 14.3 — Return an argparse.ArgumentParser Object",
            f"Expected argparse.ArgumentParser, got {type(parser).__name__} ({parser!r}).",
            details=["Return `parser` itself — do NOT call `parser.parse_args()` inside `build_cli_parser()`."],
            points_earned=0,
            points_possible=10,
        )

    errors = []

    # 1. Defaults check (--input only)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            args_default = parser.parse_args(["--input", "cohort.csv"])
    except SystemExit:
        errors.append(
            "Parsing ['--input', 'cohort.csv'] failed — make sure `--input` is registered and `--threshold`/`--verbose` are optional."
        )
    except Exception as exc:
        errors.append(f"Parsing ['--input', 'cohort.csv'] raised {type(exc).__name__}: {exc}")
    else:
        if getattr(args_default, "input", None) != "cohort.csv":
            errors.append(f"Expected args.input == 'cohort.csv', got {getattr(args_default, 'input', None)!r}")
        thr = getattr(args_default, "threshold", None)
        if not isinstance(thr, float) or not math.isclose(thr, 11.0, abs_tol=1e-6):
            errors.append(f"Expected default args.threshold == 11.0 (float), got {thr!r} ({type(thr).__name__})")
        verb = getattr(args_default, "verbose", None)
        if verb is not False:
            errors.append(f"Expected default args.verbose is False (action='store_true'), got {verb!r}")

    # 2. All flags explicit
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            args_full = parser.parse_args(
                ["--input", "lab_data.csv", "--threshold", "9.5", "--verbose"]
            )
    except SystemExit:
        errors.append("Parsing full flags ['--input', 'lab_data.csv', '--threshold', '9.5', '--verbose'] failed.")
    except Exception as exc:
        errors.append(f"Parsing full flags raised {type(exc).__name__}: {exc}")
    else:
        if getattr(args_full, "input", None) != "lab_data.csv":
            errors.append(f"Full flags: expected args.input == 'lab_data.csv', got {getattr(args_full, 'input', None)!r}")
        thr_f = getattr(args_full, "threshold", None)
        if not isinstance(thr_f, float) or not math.isclose(thr_f, 9.5, abs_tol=1e-6):
            errors.append(
                f"Full flags: expected args.threshold == 9.5 (float), got {thr_f!r} — did you pass `type=float`?"
            )
        if getattr(args_full, "verbose", None) is not True:
            errors.append(f"Full flags: expected args.verbose is True, got {getattr(args_full, 'verbose', None)!r}")

    # 3. Missing required --input must trigger SystemExit
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args([])
    except SystemExit:
        pass
    else:
        errors.append("Missing `--input` did not raise SystemExit — pass `required=True` to `--input`!")

    # 4. Non-numeric --threshold must trigger SystemExit
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["--input", "a.csv", "--threshold", "not_a_number"])
    except SystemExit:
        pass
    else:
        errors.append("Non-numeric `--threshold` was accepted — pass `type=float` to `--threshold`!")

    if errors:
        return CheckResult(
            "ch14_todo_03",
            False,
            "failed",
            "To-Do 14.3 — Check argparse Flag Definitions",
            "Register --input (required str), --threshold (float, default 11.0), and --verbose (action='store_true').",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch14_todo_03",
        True,
        "passed",
        "To-Do 14.3 Passed — argparse CLI Builder Verified!",
        "Validated required --input, float --threshold default 11.0, boolean --verbose flag, and error rejection.",
        points_earned=10,
        points_possible=10,
        tip="Returning an ArgumentParser from a dedicated helper lets unit tests call `parser.parse_args([...])` without touching `sys.argv`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_todo_03",
        chapter="ch14",
        title="To-Do 14.3: Build argparse CLI Parser",
        book_section="Section 14.3.4 (Command-Line Interfaces with argparse)",
        points=10,
        hints={
            1: "Create `parser = argparse.ArgumentParser(...)`, add `--input` (`required=True`), `--threshold` (`type=float, default=11.0`), and `--verbose` (`action='store_true'`), and return `parser`.",
            2: "Algorithm:\n1. parser = argparse.ArgumentParser(description='...')\n2. parser.add_argument('--input', type=str, required=True)\n3. parser.add_argument('--threshold', type=float, default=11.0)\n4. parser.add_argument('--verbose', action='store_true')\n5. return parser",
            3: "def build_cli_parser():\n    parser = argparse.ArgumentParser(description='Clinical CLI')\n    parser.add_argument('--input', type=str, required=True)\n    parser.add_argument('--threshold', type=___, default=11.0)\n    parser.add_argument('--verbose', action='___')\n    return parser",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 14.4: Validate src-Layout Python Package
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch14_todo_04", "To-Do 14.4", fn, "validate_src_package_layout", 10
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Case 1: Complete valid src-layout package
        good_root = base / "valid_pkg"
        (good_root / "src" / "clinic_tools").mkdir(parents=True)
        (good_root / "src" / "clinic_tools" / "__init__.py").write_text(
            '__version__ = "0.1.0"\n', encoding="utf-8"
        )
        (good_root / "pyproject.toml").write_text(
            '[project]\nname = "clinic_tools"\n', encoding="utf-8"
        )
        (good_root / "tests").mkdir()

        try:
            res_good = fn(good_root, "clinic_tools")
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch14_todo_04",
                False,
                "failed",
                "To-Do 14.4 — Runtime Error Checking Package Layout",
                f"validate_src_package_layout(root_dir, 'clinic_tools') raised {type(exc).__name__}: {exc}",
                details=["Convert root_dir with `root = Path(root_dir)` and use `.is_file()` / `.is_dir()`."],
                points_earned=0,
                points_possible=10,
            )

        if res_good is None:
            return _none_return_trap(
                "ch14_todo_04",
                "To-Do 14.4",
                "validate_src_package_layout(root_dir, 'clinic_tools')",
                10,
            )

        errors = []
        exp_good = {
            "has_pyproject": True,
            "has_init": True,
            "has_tests": True,
            "is_valid": True,
        }
        if res_good != exp_good:
            errors.append(f"Valid package layout: expected {exp_good!r}, got {res_good!r}")

        # Case 2: Missing pyproject.toml
        no_pyp = base / "no_pyproject"
        (no_pyp / "src" / "epi_kit").mkdir(parents=True)
        (no_pyp / "src" / "epi_kit" / "__init__.py").write_text("", encoding="utf-8")
        (no_pyp / "tests").mkdir()
        try:
            res_no_pyp = fn(str(no_pyp), "epi_kit")
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Missing pyproject.toml (str path) raised {type(exc).__name__}: {exc}")
        else:
            exp_no_pyp = {
                "has_pyproject": False,
                "has_init": True,
                "has_tests": True,
                "is_valid": False,
            }
            if res_no_pyp != exp_no_pyp:
                errors.append(f"Missing pyproject.toml: expected {exp_no_pyp!r}, got {res_no_pyp!r}")

        # Case 3: Missing __init__.py (wrong pkg_name)
        try:
            res_wrong_pkg = fn(good_root, "wrong_pkg")
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Wrong pkg_name raised {type(exc).__name__}: {exc}")
        else:
            exp_wrong = {
                "has_pyproject": True,
                "has_init": False,
                "has_tests": True,
                "is_valid": False,
            }
            if res_wrong_pkg != exp_wrong:
                errors.append(
                    f"Missing src/wrong_pkg/__init__.py: expected {exp_wrong!r}, got {res_wrong_pkg!r}"
                )

        # Case 4: Missing tests/ directory
        no_tests = base / "no_tests"
        (no_tests / "src" / "mch_pkg").mkdir(parents=True)
        (no_tests / "src" / "mch_pkg" / "__init__.py").write_text("", encoding="utf-8")
        (no_tests / "pyproject.toml").write_text('[project]\nname="mch_pkg"\n', encoding="utf-8")
        try:
            res_no_tests = fn(no_tests, "mch_pkg")
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Missing tests/ raised {type(exc).__name__}: {exc}")
        else:
            exp_no_tests = {
                "has_pyproject": True,
                "has_init": True,
                "has_tests": False,
                "is_valid": False,
            }
            if res_no_tests != exp_no_tests:
                errors.append(f"Missing tests/ folder: expected {exp_no_tests!r}, got {res_no_tests!r}")

    if errors:
        return CheckResult(
            "ch14_todo_04",
            False,
            "failed",
            "To-Do 14.4 — Check src-Layout Path Inspection",
            "Check pyproject.toml (file), src/<pkg_name>/__init__.py (file), tests/ (dir), and is_valid (all three True).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch14_todo_04",
        True,
        "passed",
        "To-Do 14.4 Passed — src-Layout Package Validator Verified!",
        "Accurately inspected pyproject.toml, src/<pkg>/__init__.py, and tests/ across valid and incomplete repositories.",
        points_earned=10,
        points_possible=10,
        tip="The `src/` layout prevents Python from accidentally importing your uninstalled working directory instead of the installed package.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_todo_04",
        chapter="ch14",
        title="To-Do 14.4: Validate src-Layout Package",
        book_section="Section 14.3.3 (Standard src/ Package Layout)",
        points=10,
        hints={
            1: "Wrap `root = Path(root_dir)` and check `(root / 'pyproject.toml').is_file()`, `(root / 'src' / pkg_name / '__init__.py').is_file()`, and `(root / 'tests').is_dir()`.",
            2: "Algorithm:\n1. root = Path(root_dir)\n2. has_pyproject = (root / 'pyproject.toml').is_file()\n3. has_init = (root / 'src' / pkg_name / '__init__.py').is_file()\n4. has_tests = (root / 'tests').is_dir()\n5. return {'has_pyproject': ..., 'has_init': ..., 'has_tests': ..., 'is_valid': has_pyproject and has_init and has_tests}",
            3: "def validate_src_package_layout(root_dir, pkg_name):\n    root = Path(root_dir)\n    has_pyproject = (root / 'pyproject.toml').is_file()\n    has_init = (root / 'src' / pkg_name / '___').is_file()\n    has_tests = (root / 'tests').is_dir()\n    return {\n        'has_pyproject': has_pyproject,\n        'has_init': has_init,\n        'has_tests': has_tests,\n        'is_valid': has_pyproject and has_init and has_tests,\n    }",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 14.5: Normalize Package Name
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch14_todo_05", "To-Do 14.5", fn, "normalize_package_name", 10
    )
    if bad is not None:
        return bad

    try:
        nominal = fn("  MCH-Surveillance  Tool ")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch14_todo_05",
            False,
            "failed",
            "To-Do 14.5 — Runtime Error Normalizing Package Name",
            f"normalize_package_name('  MCH-Surveillance  Tool ') raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if nominal is None:
        return _none_return_trap(
            "ch14_todo_05",
            "To-Do 14.5",
            "normalize_package_name('  MCH-Surveillance  Tool ')",
            10,
        )

    errors = []
    cases = [
        ("  MCH-Surveillance  Tool ", "mch_surveillance_tool", "Mixed case, hyphens & double space"),
        ("--Dhaka__Clinic--Tools--", "dhaka_clinic_tools", "Leading/trailing hyphens & repeated underscores"),
        ("epidemiology-kit", "epidemiology_kit", "Standard hyphenated distribution name"),
        ("  ___clean_pkg___  ", "clean_pkg", "Leading/trailing underscores"),
        ("simple", "simple", "Already valid lowercase identifier"),
        ("---   ___", "", "Separators only"),
    ]
    for raw, expected, label in cases:
        try:
            got = fn(raw)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} ({raw!r}) raised {type(exc).__name__}: {exc}")
            continue
        if got != expected:
            extra = ""
            if isinstance(got, str) and "__" in got:
                extra = " (collapse repeated underscores with `re.sub(r'_+', '_', cleaned)`!)"
            elif isinstance(got, str) and (got.startswith("_") or got.endswith("_")):
                extra = " (strip leading and trailing underscores with `.strip('_')`!)"
            errors.append(f"{label}: input {raw!r} -> expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch14_todo_05",
            False,
            "failed",
            "To-Do 14.5 — Check Package Name Normalization Rules",
            "Strip, lowercase, replace '-' and spaces with '_', collapse repeated '_', and strip leading/trailing '_'.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch14_todo_05",
        True,
        "passed",
        "To-Do 14.5 Passed — Package Name Normalizer Verified!",
        "Converted messy project titles and hyphenated PyPI names into valid snake_case Python import identifiers.",
        points_earned=10,
        points_possible=10,
        tip="Distribution names on PyPI often use hyphens (`sharukh-tutor`), while Python import names must use underscores (`sharukh_tutor`).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_todo_05",
        chapter="ch14",
        title="To-Do 14.5: Normalize Python Package Name",
        book_section="Section 14.3.3 (PyPI vs. Python Import Naming)",
        points=10,
        hints={
            1: "Strip and lowercase `raw_name`, replace `'-'` and `' '` with `'_'`, collapse repeated `'_'` (via `re.sub(r'_+', '_', ...)` or a loop), and call `.strip('_')`.",
            2: "Algorithm:\n1. cleaned = raw_name.strip().lower()\n2. cleaned = cleaned.replace('-', '_').replace(' ', '_')\n3. cleaned = re.sub(r'_+', '_', cleaned)\n4. return cleaned.strip('_')",
            3: "import re\n\ndef normalize_package_name(raw_name):\n    cleaned = raw_name.strip().lower()\n    cleaned = cleaned.replace('-', '_').replace(' ', '_')\n    cleaned = re.sub(r'_+', '_', cleaned)\n    return cleaned.strip('___')",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 14.6: Load .env Configuration File
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch14_todo_06", "To-Do 14.6", fn, "load_env_file", 10)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        env_path = base / ".env"
        env_path.write_text(
            "# Local study environment configuration\n"
            "\n"
            'STUDY_NAME="MCH-BD-2026"\n'
            "API_KEY='secret-token-42'\n"
            "DATA_DIR=/srv/data/raw\n"
            'QUERY_URL="https://api.clinic.org/v1?site=dhaka"\n'
            "   \n"
            "# End of config\n",
            encoding="utf-8",
        )

        try:
            loaded = fn(env_path)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch14_todo_06",
                False,
                "failed",
                "To-Do 14.6 — Runtime Error Parsing .env File",
                f"load_env_file(env_path) raised {type(exc).__name__}: {exc}",
                details=[
                    "Skip blank lines and `#` comments, then use `key, val = line.split('=', 1)` so `=` inside values never crashes!"
                ],
                points_earned=0,
                points_possible=10,
            )

        if loaded is None:
            return _none_return_trap("ch14_todo_06", "To-Do 14.6", "load_env_file(env_path)", 10)

        errors = []
        expected = {
            "STUDY_NAME": "MCH-BD-2026",
            "API_KEY": "secret-token-42",
            "DATA_DIR": "/srv/data/raw",
            "QUERY_URL": "https://api.clinic.org/v1?site=dhaka",
        }
        if not isinstance(loaded, dict):
            errors.append(f"Expected a dict, got {type(loaded).__name__} ({loaded!r})")
        elif loaded != expected:
            extra = ""
            if loaded.get("STUDY_NAME") == '"MCH-BD-2026"' or loaded.get("API_KEY") == "'secret-token-42'":
                extra = " (strip surrounding single or double quotes from parsed values!)"
            errors.append(f"Nominal .env mismatch: expected {expected!r}, got {loaded!r}{extra}")

        # Edge case: comments/blanks only + str path
        empty_env = base / "empty.env"
        empty_env.write_text("# No secrets yet\n\n", encoding="utf-8")
        try:
            empty_loaded = fn(str(empty_env))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Comments-only .env raised {type(exc).__name__}: {exc}")
        else:
            if empty_loaded != {}:
                errors.append(f"Comments-only .env: expected {{}}, got {empty_loaded!r}")

    if errors:
        return CheckResult(
            "ch14_todo_06",
            False,
            "failed",
            "To-Do 14.6 — Check .env Key/Value Parsing & Quote Stripping",
            "Ignore blanks/# comments, split on the first '=' via split('=', 1), and strip surrounding quotes from values.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch14_todo_06",
        True,
        "passed",
        "To-Do 14.6 Passed — .env Configuration Loader Verified!",
        "Parsed unquoted, single-quoted, double-quoted, and URL-with-'=' secrets cleanly while skipping comments.",
        points_earned=10,
        points_possible=10,
        tip="Always add `.env` to `.gitignore` so database passwords and API tokens never leak into a public GitHub repository.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_todo_06",
        chapter="ch14",
        title="To-Do 14.6: Load .env Configuration File",
        book_section="Section 14.3.5 (Separating Secrets with .env Files)",
        points=10,
        hints={
            1: "Open `env_path` in UTF-8, skip blank and `#` lines, split with `key, val = line.split('=', 1)`, strip whitespace, and strip matching surrounding quotes (`'...'` or `\"...\"`) from `val`.",
            2: "Algorithm:\n1. env = {}\n2. For stripped line in env_path:\n       if not line or line.startswith('#'): continue\n       key, val = line.split('=', 1)\n       key, val = key.strip(), val.strip()\n       if len(val) >= 2 and val[0] == val[-1] and val[0] in ('\"', \"'\"):\n           val = val[1:-1]\n       env[key] = val\n3. return env",
            3: "def load_env_file(env_path):\n    env = {}\n    with open(env_path, 'r', encoding='utf-8') as f:\n        for raw in f:\n            line = raw.strip()\n            if not line or line.startswith('#'):\n                continue\n            key, val = line.split('=', ___)\n            key, val = key.strip(), val.strip()\n            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('\"', \"'\"):\n                val = val[1:-1]\n            env[key] = val\n    return env",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 14.1 (30 pts): Dependency Reproducibility Auditor
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch14_exam_01", "Exam Task 14.1", fn, "audit_dependencies", 30)
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        mixed_req = base / "mixed_requirements.txt"
        mixed_req.write_text(
            "# Manuscript analysis environment\n"
            "pandas==2.2.0\n"
            "NumPy>=1.26.0\n"
            "scipy\n"
            "\n"
            "matplotlib==3.9.0\n"
            "seaborn~=0.13.2\n",
            encoding="utf-8",
        )

        try:
            report = fn(mixed_req)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch14_exam_01",
                False,
                "failed",
                "Exam Task 14.1 — Runtime Error Auditing requirements.txt",
                f"audit_dependencies(req_path) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if report is None:
            return _none_return_trap(
                "ch14_exam_01", "Exam Task 14.1", "audit_dependencies(req_path)", 30
            )

        errors = []
        exp_mixed = {
            "total": 5,
            "pinned_count": 2,
            "unpinned": ["numpy", "scipy", "seaborn"],
            "reproducible": False,
        }
        if report != exp_mixed:
            errors.append(f"Mixed requirements.txt: expected {exp_mixed!r}, got {report!r}")

        # Case 2: Fully pinned reproducible file
        pinned_req = base / "pinned_requirements.txt"
        pinned_req.write_text(
            "# Frozen lock export\n"
            "pandas==2.2.0\n"
            "numpy==1.26.4\n"
            "pytest==8.3.2\n",
            encoding="utf-8",
        )
        try:
            rep_pinned = fn(pinned_req)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Fully pinned file raised {type(exc).__name__}: {exc}")
        else:
            exp_pinned = {
                "total": 3,
                "pinned_count": 3,
                "unpinned": [],
                "reproducible": True,
            }
            if rep_pinned != exp_pinned:
                errors.append(f"Fully pinned file: expected {exp_pinned!r}, got {rep_pinned!r}")

        # Case 3: Comments/blanks only
        empty_req = base / "empty_requirements.txt"
        empty_req.write_text("# Nothing installed yet\n\n", encoding="utf-8")
        try:
            rep_empty = fn(empty_req)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Empty requirements file raised {type(exc).__name__}: {exc}")
        else:
            exp_empty = {
                "total": 0,
                "pinned_count": 0,
                "unpinned": [],
                "reproducible": True,
            }
            if rep_empty != exp_empty:
                errors.append(f"Empty requirements file: expected {exp_empty!r}, got {rep_empty!r}")

    if errors:
        return CheckResult(
            "ch14_exam_01",
            False,
            "failed",
            "Exam Task 14.1 — Check Dependency Audit Summary Fields",
            "Return {'total': int, 'pinned_count': int, 'unpinned': sorted_list, 'reproducible': bool}.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch14_exam_01",
        True,
        "passed",
        "Exam Task 14.1 Passed (30/30 pts) — Dependency Reproducibility Auditor Verified!",
        "Audited mixed, fully pinned, and empty requirements files with exact counts, sorted unpinned names, and reproducibility flags.",
        points_earned=30,
        points_possible=30,
        tip="Running an automated dependency audit in CI catches unpinned packages before you submit your supplementary repository.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_exam_01",
        chapter="ch14",
        title="Exam 14.1: Audit Dependency Reproducibility",
        book_section="Chapter 14 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Parse `req_path` (skipping blanks and `#` comments, lowercasing names, keeping operators), find sorted unpinned packages (`not spec.startswith('==')`), and return `total`, `pinned_count`, `unpinned`, and `reproducible` (`len(unpinned) == 0`).",
            2: "Algorithm:\n1. Parse req_path into reqs dict (`pkg_lower -> spec`)\n2. unpinned = sorted(pkg for pkg, spec in reqs.items() if not spec.startswith('=='))\n3. total = len(reqs); pinned_count = total - len(unpinned)\n4. return {'total': total, 'pinned_count': pinned_count, 'unpinned': unpinned, 'reproducible': len(unpinned) == 0}",
            3: "def audit_dependencies(req_path):\n    reqs = {}\n    with open(req_path, 'r', encoding='utf-8') as f:\n        for raw in f:\n            line = raw.strip()\n            if not line or line.startswith('#'):\n                continue\n            for op in ('==', '>=', '<=', '~='):\n                if op in line:\n                    name, ver = line.split(op, 1)\n                    reqs[name.strip().lower()] = op + ver.strip()\n                    break\n            else:\n                reqs[line.lower()] = ''\n    unpinned = sorted(k for k, v in reqs.items() if not v.startswith('=='))\n    return {\n        'total': len(reqs),\n        'pinned_count': len(reqs) - len(unpinned),\n        'unpinned': unpinned,\n        'reproducible': len(unpinned) == ___,\n    }",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 14.2 (35 pts): Scaffold Research Package
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch14_exam_02", "Exam Task 14.2", fn, "scaffold_research_package", 35
    )
    if bad is not None:
        return bad

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        proj_1 = base / "proj_1"
        proj_1.mkdir()

        try:
            created_1 = fn(proj_1, "  MCH-Clinic  Toolkit ")
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch14_exam_02",
                False,
                "failed",
                "Exam Task 14.2 — Runtime Error Scaffolding Package",
                f"scaffold_research_package(base_dir, pkg_name) raised {type(exc).__name__}: {exc}",
                details=[
                    "Create parent folders with `target.parent.mkdir(parents=True, exist_ok=True)` before writing each file."
                ],
                points_earned=0,
                points_possible=35,
            )

        if created_1 is None:
            return _none_return_trap(
                "ch14_exam_02",
                "Exam Task 14.2",
                "scaffold_research_package(base_dir, '  MCH-Clinic  Toolkit ')",
                35,
            )

        errors = []
        exp_1 = [
            "README.md",
            "pyproject.toml",
            "src/mch_clinic_toolkit/__init__.py",
            "tests/test_smoke.py",
        ]
        if not isinstance(created_1, list):
            errors.append(f"Expected a sorted list of relative POSIX paths, got {type(created_1).__name__} ({created_1!r})")
        elif created_1 != exp_1:
            extra = ""
            if sorted(created_1) == exp_1:
                extra = " (sort the returned list!)"
            elif any("\\" in str(p) for p in created_1):
                extra = " (use `.as_posix()` so paths use forward slashes `/`!)"
            errors.append(f"Case 1 returned paths mismatch: expected {exp_1!r}, got {created_1!r}{extra}")

        for rel in exp_1:
            fpath = proj_1 / rel
            if not fpath.is_file():
                errors.append(f"Expected file was not created on disk: {rel}")
            elif not fpath.read_text(encoding="utf-8").strip():
                errors.append(f"Created file {rel} is empty — write non-empty starter text!")

        # Case 2: Second package name to prevent hardcoding
        proj_2 = base / "proj_2"
        proj_2.mkdir()
        try:
            created_2 = fn(str(proj_2), "--Dengue__Alert--")
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"Case 2 ('--Dengue__Alert--') raised {type(exc).__name__}: {exc}")
        else:
            exp_2 = [
                "README.md",
                "pyproject.toml",
                "src/dengue_alert/__init__.py",
                "tests/test_smoke.py",
            ]
            if created_2 != exp_2:
                errors.append(f"Case 2 returned paths mismatch: expected {exp_2!r}, got {created_2!r}")
            for rel in exp_2:
                fpath = proj_2 / rel
                if not fpath.is_file() or not fpath.read_text(encoding="utf-8").strip():
                    errors.append(f"Case 2 missing or empty file: {rel}")

    if errors:
        return CheckResult(
            "ch14_exam_02",
            False,
            "failed",
            "Exam Task 14.2 — Check Package Scaffolding & Returned POSIX Paths",
            "Normalize pkg_name, create pyproject.toml, README.md, src/<pkg>/__init__.py, and tests/test_smoke.py with non-empty starter text, and return sorted relative POSIX paths.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch14_exam_02",
        True,
        "passed",
        "Exam Task 14.2 Passed (35/35 pts) — Research Package Scaffolder Verified!",
        "Normalized package names, generated complete src-layout starter repositories on disk, and returned sorted POSIX paths.",
        points_earned=35,
        points_possible=35,
        tip="Using `Path.relative_to(root).as_posix()` guarantees identical forward-slash paths across Linux, macOS, and Windows.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_exam_02",
        chapter="ch14",
        title="Exam 14.2: Scaffold Research Package",
        book_section="Chapter 14 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Normalize `pkg_name` first, create parent directories with `mkdir(parents=True, exist_ok=True)`, write non-empty starter text into `pyproject.toml`, `README.md`, `src/<pkg>/__init__.py`, and `tests/test_smoke.py`, and return their sorted relative POSIX paths.",
            2: "Algorithm:\n1. Normalize pkg_name -> pkg\n2. Define dict mapping each of the 4 relative paths to non-empty starter text\n3. For rel_path, content in files.items():\n       target = Path(base_dir) / rel_path\n       target.parent.mkdir(parents=True, exist_ok=True)\n       target.write_text(content, encoding='utf-8')\n4. Return sorted(target.relative_to(root).as_posix() ...)",
            3: "from pathlib import Path\nimport re\n\ndef scaffold_research_package(base_dir, pkg_name):\n    cleaned = pkg_name.strip().lower().replace('-', '_').replace(' ', '_')\n    pkg = re.sub(r'_+', '_', cleaned).strip('_')\n    root = Path(base_dir)\n    files = {\n        'pyproject.toml': f'[project]\\nname = \"{pkg}\"\\nversion = \"0.1.0\"\\n',\n        'README.md': f'# {pkg}\\n',\n        f'src/{pkg}/__init__.py': '__version__ = \"0.1.0\"\\n',\n        'tests/test_smoke.py': f'import {pkg}\\n',\n    }\n    created = []\n    for rel, text in files.items():\n        target = root / rel\n        target.parent.mkdir(parents=True, exist_ok=True)\n        target.write_text(text, encoding='utf-8')\n        created.append(target.relative_to(root).___())\n    return sorted(created)",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 14.3 (35 pts): Clinical Vitals Triage CLI Runner
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch14_exam_03", "Exam Task 14.3", fn, "run_triage_cli", 35)
    if bad is not None:
        return bad

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            nominal = fn(["--temp", "39.5", "--hb", "12.4", "--sbp", "122"])
    except NotImplementedError:
        raise
    except SystemExit as exc:
        return CheckResult(
            "ch14_exam_03",
            False,
            "failed",
            "Exam Task 14.3 — ArgumentParser Rejected Valid Flags",
            f"run_triage_cli(['--temp', '39.5', '--hb', '12.4', '--sbp', '122']) exited with code {exc.code}.",
            details=[
                "Register `--temp` (type=float, required=True), `--hb` (type=float, required=True), and `--sbp` (type=int, required=True).",
                "Make sure you call `parser.parse_args(argv)` — NOT `parser.parse_args()` without `argv`!",
            ],
            points_earned=0,
            points_possible=35,
        )
    except Exception as exc:
        return CheckResult(
            "ch14_exam_03",
            False,
            "failed",
            "Exam Task 14.3 — Runtime Error Running Triage CLI",
            f"run_triage_cli(argv) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch14_exam_03",
            "Exam Task 14.3",
            "run_triage_cli(['--temp', '39.5', '--hb', '12.4', '--sbp', '122'])",
            35,
        )

    errors = []
    cases = [
        (
            ["--temp", "39.5", "--hb", "12.4", "--sbp", "122"],
            {"temp_c": 39.5, "hb_g_dl": 12.4, "sbp_mmhg": 122, "priority": "Urgent"},
            "Urgent boundary (temp == 39.5)",
        ),
        (
            ["--temp", "37.0", "--hb", "13.0", "--sbp", "170"],
            {"temp_c": 37.0, "hb_g_dl": 13.0, "sbp_mmhg": 170, "priority": "Urgent"},
            "Urgent boundary (sbp == 170)",
        ),
        (
            ["--temp", "36.9", "--hb", "8.4", "--sbp", "118"],
            {"temp_c": 36.9, "hb_g_dl": 8.4, "sbp_mmhg": 118, "priority": "Urgent"},
            "Urgent boundary (hb < 8.5)",
        ),
        (
            ["--temp", "37.2", "--hb", "8.5", "--sbp", "120"],
            {"temp_c": 37.2, "hb_g_dl": 8.5, "sbp_mmhg": 120, "priority": "Moderate"},
            "Moderate boundary (hb == 8.5 is Moderate, not Urgent)",
        ),
        (
            ["--temp", "38.0", "--hb", "12.0", "--sbp", "120"],
            {"temp_c": 38.0, "hb_g_dl": 12.0, "sbp_mmhg": 120, "priority": "Moderate"},
            "Moderate boundary (temp == 38.0)",
        ),
        (
            ["--temp", "37.1", "--hb", "12.0", "--sbp", "140"],
            {"temp_c": 37.1, "hb_g_dl": 12.0, "sbp_mmhg": 140, "priority": "Moderate"},
            "Moderate boundary (sbp == 140)",
        ),
        (
            ["--temp", "37.1", "--hb", "11.0", "--sbp", "125"],
            {"temp_c": 37.1, "hb_g_dl": 11.0, "sbp_mmhg": 125, "priority": "Routine"},
            "Routine boundary (hb == 11.0 is Routine)",
        ),
    ]

    for argv, exp, label in cases:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                got = fn(argv)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if not isinstance(got, dict):
            errors.append(f"{label}: expected dict, got {type(got).__name__} ({got!r})")
            continue
        if set(got.keys()) != set(exp.keys()):
            errors.append(f"{label}: expected keys {sorted(exp.keys())}, got {sorted(got.keys())}")
            continue
        if (
            not isinstance(got["temp_c"], float)
            or not math.isclose(got["temp_c"], exp["temp_c"], abs_tol=1e-6)
            or not isinstance(got["hb_g_dl"], float)
            or not math.isclose(got["hb_g_dl"], exp["hb_g_dl"], abs_tol=1e-6)
            or type(got["sbp_mmhg"]) is not int
            or got["sbp_mmhg"] != exp["sbp_mmhg"]
            or got["priority"] != exp["priority"]
        ):
            errors.append(f"{label}: expected {exp!r}, got {got!r}")

    # Verify missing required flag raises SystemExit (confirms argparse required=True)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            fn(["--temp", "38.2", "--hb", "11.5"])
    except SystemExit:
        pass
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(
            f"Missing `--sbp` raised {type(exc).__name__} instead of SystemExit — use `argparse` with `required=True`!"
        )
    else:
        errors.append("Missing `--sbp` did not exit — mark `--temp`, `--hb`, and `--sbp` as `required=True`!")

    if errors:
        return CheckResult(
            "ch14_exam_03",
            False,
            "failed",
            "Exam Task 14.3 — Check Triage CLI Parsing & Priority Cutoffs",
            "Parse --temp (float), --hb (float), --sbp (int) via argparse.parse_args(argv) and return {'temp_c', 'hb_g_dl', 'sbp_mmhg', 'priority'}.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch14_exam_03",
        True,
        "passed",
        "Exam Task 14.3 Passed (35/35 pts) — Clinical Triage CLI Runner Verified!",
        "Parsed typed CLI arguments via argparse, enforced exact Urgent/Moderate/Routine cutoffs, and rejected incomplete flags.",
        points_earned=35,
        points_possible=35,
        tip="Passing an explicit `argv` list into `parser.parse_args(argv)` makes every CLI tool 100% testable in pytest and notebooks.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch14_exam_03",
        chapter="ch14",
        title="Exam 14.3: Clinical Vitals Triage CLI",
        book_section="Chapter 14 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Create an `argparse.ArgumentParser`, add `--temp` (`type=float, required=True`), `--hb` (`type=float, required=True`), and `--sbp` (`type=int, required=True`), call `args = parser.parse_args(argv)`, classify Urgent -> Moderate -> Routine, and return the 4-key dict.",
            2: "Algorithm:\n1. Build parser with --temp (float), --hb (float), --sbp (int), all required=True\n2. args = parser.parse_args(argv)\n3. If temp >= 39.5 or sbp >= 170 or hb < 8.5 -> 'Urgent'\n   elif temp >= 38.0 or sbp >= 140 or hb < 11.0 -> 'Moderate'\n   else -> 'Routine'\n4. return {'temp_c': args.temp, 'hb_g_dl': args.hb, 'sbp_mmhg': args.sbp, 'priority': priority}",
            3: "import argparse\n\ndef run_triage_cli(argv):\n    parser = argparse.ArgumentParser(description='Triage CLI')\n    parser.add_argument('--temp', type=float, required=True)\n    parser.add_argument('--hb', type=float, required=True)\n    parser.add_argument('--sbp', type=int, required=True)\n    args = parser.parse_args(___)\n    if args.temp >= 39.5 or args.sbp >= 170 or args.hb < 8.5:\n        priority = 'Urgent'\n    elif args.temp >= 38.0 or args.sbp >= 140 or args.hb < 11.0:\n        priority = 'Moderate'\n    else:\n        priority = 'Routine'\n    return {\n        'temp_c': args.temp,\n        'hb_g_dl': args.hb,\n        'sbp_mmhg': args.sbp,\n        'priority': priority,\n    }",
        },
        verifier=_verify_exam_03,
    )
)
