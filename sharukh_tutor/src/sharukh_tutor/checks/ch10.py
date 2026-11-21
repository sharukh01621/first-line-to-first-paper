"""Chapter 10 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch10_todo_01..06) and Chapter 10 Coding Exam
(ch10_exam_01..03). Every verifier tests a callable function across nominal,
invalid, and boundary inputs, verifies that expected ValueError/AssertionError
exceptions are raised when specified, catches print()-only None returns, and
lets NotImplementedError propagate cleanly.
"""

from __future__ import annotations

import math
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
        f"Calling {call_str} returned None instead of the expected value.",
        details=[
            "Remember: print() only displays text on screen and hands back None.",
            "Make sure every valid branch ends with an explicit `return ...` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Only return sends an object back to the caller or automated test suite.",
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 10.1: Safe Float Parser (try / except (ValueError, TypeError))
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_todo_01", "To-Do 10.1", fn, "parse_float_safely", 10)
    if bad is not None:
        return bad

    # Test nominal case FIRST to distinguish print()-only None from default=None
    try:
        nominal = fn("38.5")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch10_todo_01",
            False,
            "failed",
            "To-Do 10.1 — Runtime Error on Nominal Input",
            f"parse_float_safely('38.5') raised {type(exc).__name__}: {exc}",
            details=["Wrap float(raw_val) in a try/except (ValueError, TypeError) block."],
        )

    if nominal is None:
        return _none_return_trap(
            "ch10_todo_01", "To-Do 10.1", "parse_float_safely('38.5')", 10
        )

    errors = []
    if not isinstance(nominal, float) or not math.isclose(nominal, 38.5, abs_tol=1e-6):
        errors.append(f"parse_float_safely('38.5'): expected 38.5 (float), got {nominal!r}")

    tests = [
        (("  37.2  ",), {}, 37.2, "parse_float_safely('  37.2  ')"),
        ((40,), {}, 40.0, "parse_float_safely(40)"),
        (("N/A",), {}, None, "parse_float_safely('N/A')"),
        ((None,), {}, None, "parse_float_safely(None)"),
        (("",), {}, None, "parse_float_safely('')"),
        (("N/A",), {"default": 0.0}, 0.0, "parse_float_safely('N/A', default=0.0)"),
        ((None, -1.0), {}, -1.0, "parse_float_safely(None, -1.0)"),
    ]
    for args, kwargs, expected, label in tests:
        try:
            got = fn(*args, **kwargs)
        except NotImplementedError:
            raise
        except (ValueError, TypeError) as exc:
            errors.append(
                f"{label} raised uncaught {type(exc).__name__}: {exc} "
                f"— catch both (ValueError, TypeError) and return default!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if expected is None:
            if got is not None:
                errors.append(f"{label}: expected None, got {got!r}")
        else:
            if not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-6):
                errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch10_todo_01",
            False,
            "failed",
            "To-Do 10.1 — Check Exception Tuple & Default Fallback",
            "Return float(raw_val) inside try, and catch (ValueError, TypeError) to return default.",
            details=errors,
        )
    return CheckResult(
        "ch10_todo_01",
        True,
        "passed",
        "To-Do 10.1 Passed — Safe Float Parser Verified!",
        "Cleanly parsed valid strings/ints and returned default on 'N/A', '', and None without crashing.",
        tip="Catching specific exceptions like (ValueError, TypeError) keeps unexpected bugs visible while handling dirty fields gracefully.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_todo_01",
        chapter="ch10",
        title="To-Do 10.1: Safe Float Parser",
        book_section="Section 10.3.1 (try / except Specific Exceptions)",
        points=10,
        hints={
            1: "Define `def parse_float_safely(raw_val, default=None):` and wrap `return float(raw_val)` inside `try / except (ValueError, TypeError):`.",
            2: "Algorithm:\n1. try:\n       return float(raw_val)\n2. except (ValueError, TypeError):\n       return default",
            3: "def parse_float_safely(raw_val, default=None):\n    try:\n        return float(raw_val)\n    except (ValueError, ___):\n        return default",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 10.2: Safe Case Fatality Rate (%) with ValueError & Zero Guard
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_todo_02", "To-Do 10.2", fn, "safe_cfr_percent", 10)
    if bad is not None:
        return bad

    # Nominal test first to catch print()-only None trap
    try:
        nominal = fn(12, 400)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch10_todo_02",
            False,
            "failed",
            "To-Do 10.2 — Runtime Error on Nominal Input",
            f"safe_cfr_percent(12, 400) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch10_todo_02", "To-Do 10.2", "safe_cfr_percent(12, 400)", 10
        )

    errors = []
    if not _is_number(nominal) or not math.isclose(nominal, 3.0, abs_tol=1e-2):
        errors.append(f"safe_cfr_percent(12, 400): expected 3.0, got {nominal!r}")

    value_tests = [
        ((7, 300), 2.33, "safe_cfr_percent(7, 300)"),
        ((0, 250), 0.0, "safe_cfr_percent(0, 250)"),
        ((0, 0), None, "safe_cfr_percent(0, 0)"),
        ((5, 0), None, "safe_cfr_percent(5, 0)"),
    ]
    for (deaths, cases), expected, label in value_tests:
        try:
            got = fn(deaths, cases)
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                f"{label} raised ZeroDivisionError — return None when cases == 0!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised unexpected {type(exc).__name__}: {exc}")
            continue

        if expected is None:
            if got is not None:
                errors.append(f"{label}: expected None (cases == 0), got {got!r}")
        else:
            if not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
                errors.append(f"{label}: expected {expected}, got {got!r}")

    # Negative inputs MUST raise ValueError
    negative_tests = [
        (-1, 100),
        (5, -10),
        (-2, -5),
        (-1, 0),
    ]
    for deaths, cases in negative_tests:
        label = f"safe_cfr_percent({deaths}, {cases})"
        try:
            got = fn(deaths, cases)
        except NotImplementedError:
            raise
        except ValueError:
            continue
        except Exception as exc:
            errors.append(
                f"{label} raised {type(exc).__name__} instead of ValueError: {exc}"
            )
        else:
            errors.append(
                f"{label} returned {got!r} instead of raising ValueError for negative count!"
            )

    if errors:
        return CheckResult(
            "ch10_todo_02",
            False,
            "failed",
            "To-Do 10.2 — Check Negative ValueError Guard & Zero-Case Handling",
            "Raise ValueError first if deaths < 0 or cases < 0; return None if cases == 0; else return round((deaths / cases) * 100, 2).",
            details=errors,
        )
    return CheckResult(
        "ch10_todo_02",
        True,
        "passed",
        "To-Do 10.2 Passed — Defensive CFR Function Verified!",
        "Correctly raised ValueError on negative counts, returned None when cases == 0, and computed exact CFR percentages.",
        tip="Distinguish impossible data (raise ValueError for negative counts) from undefined-but-valid states (return None for zero cases).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_todo_02",
        chapter="ch10",
        title="To-Do 10.2: Defensive Case Fatality Rate (%)",
        book_section="Section 10.3.3 (Raising ValueError vs. Returning None)",
        points=10,
        hints={
            1: "Check `if deaths < 0 or cases < 0: raise ValueError(...)` first, then `if cases == 0: return None`, then compute `round((deaths / cases) * 100, 2)`.",
            2: "Algorithm:\n1. if deaths < 0 or cases < 0:\n       raise ValueError('Counts cannot be negative')\n2. if cases == 0:\n       return None\n3. return round((deaths / cases) * 100, 2)",
            3: "def safe_cfr_percent(deaths, cases):\n    if deaths < 0 or cases < 0:\n        raise ValueError('deaths and cases must be >= 0')\n    if cases == 0:\n        return ___\n    return round((deaths / cases) * 100, 2)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 10.3: Physiological Body Temperature Validator
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_todo_03", "To-Do 10.3", fn, "validate_body_temp", 10)
    if bad is not None:
        return bad

    # Nominal check first
    try:
        nominal = fn(37.2)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch10_todo_03",
            False,
            "failed",
            "To-Do 10.3 — Runtime Error on Valid Temperature",
            f"validate_body_temp(37.2) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch10_todo_03", "To-Do 10.3", "validate_body_temp(37.2)", 10
        )

    errors = []
    valid_tests = [
        (37.2, 37.2),
        (38, 38.0),
        (30.0, 30.0),
        (45.0, 45.0),
    ]
    for raw_t, expected in valid_tests:
        label = f"validate_body_temp({raw_t!r})"
        try:
            got = fn(raw_t)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised unexpected {type(exc).__name__}: {exc}")
            continue
        if not isinstance(got, float) or not math.isclose(got, expected, abs_tol=1e-6):
            errors.append(f"{label}: expected float {expected!r}, got {got!r} ({type(got).__name__})")

    invalid_tests = [
        29.9,
        45.1,
        -5.0,
        999,
        "37.5",
        None,
        True,
    ]
    for bad_t in invalid_tests:
        label = f"validate_body_temp({bad_t!r})"
        try:
            got = fn(bad_t)
        except NotImplementedError:
            raise
        except ValueError:
            continue
        except TypeError as exc:
            errors.append(
                f"{label} raised TypeError instead of ValueError ({exc}) "
                f"— check isinstance(temp_c, (int, float)) BEFORE comparing < 30.0 or > 45.0!"
            )
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__} instead of ValueError: {exc}")
        else:
            errors.append(f"{label} returned {got!r} instead of raising ValueError!")

    if errors:
        return CheckResult(
            "ch10_todo_03",
            False,
            "failed",
            "To-Do 10.3 — Check Type Validation & [30.0, 45.0] Bounds",
            "Raise ValueError if temp_c is not an int/float (or is bool) or outside [30.0, 45.0]; otherwise return float(temp_c).",
            details=errors,
        )
    return CheckResult(
        "ch10_todo_03",
        True,
        "passed",
        "To-Do 10.3 Passed — Body Temperature Validator Verified!",
        "Validated numeric types, inclusive physiological bounds [30.0, 45.0], and rejected strings, None, booleans, and out-of-range values with ValueError.",
        tip="Validating types before range checks turns cryptic TypeErrors into clean, domain-specific ValueErrors.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_todo_03",
        chapter="ch10",
        title="To-Do 10.3: Physiological Body Temperature Validator",
        book_section="Section 10.3.3 (Defensive Input Contracts & raise ValueError)",
        points=10,
        hints={
            1: "First reject non-numeric inputs (and booleans) with `ValueError`, then reject `temp_c < 30.0 or temp_c > 45.0` with `ValueError`, else return `float(temp_c)`.",
            2: "Algorithm:\n1. if not isinstance(temp_c, (int, float)) or isinstance(temp_c, bool):\n       raise ValueError('Temperature must be numeric')\n2. if temp_c < 30.0 or temp_c > 45.0:\n       raise ValueError('Temperature out of range [30.0, 45.0]')\n3. return float(temp_c)",
            3: "def validate_body_temp(temp_c):\n    if isinstance(temp_c, bool) or not isinstance(temp_c, (int, float)):\n        raise ValueError('Non-numeric temperature')\n    if not (30.0 <= temp_c <= ___):\n        raise ValueError('Out of physiological range')\n    return float(temp_c)",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 10.4: Nested Surveillance Dictionary Metric Extractor
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_todo_04", "To-Do 10.4", fn, "extract_nested_metric", 10)
    if bad is not None:
        return bad

    sample = {
        "Dhaka": {"cases": 140, "deaths": 3},
        "Sylhet": {"cases": 25, "deaths": 0},
        "Barishal": None,
    }

    # Nominal test first
    try:
        nominal = fn(sample, "Dhaka", "cases")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch10_todo_04",
            False,
            "failed",
            "To-Do 10.4 — Runtime Error on Valid Lookup",
            f"extract_nested_metric(sample, 'Dhaka', 'cases') raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch10_todo_04", "To-Do 10.4", "extract_nested_metric(sample, 'Dhaka', 'cases')", 10
        )

    errors = []
    if nominal != 140:
        errors.append(f"extract_nested_metric(sample, 'Dhaka', 'cases'): expected 140, got {nominal!r}")

    tests = [
        (sample, "Sylhet", "deaths", 0, "extract_nested_metric(sample, 'Sylhet', 'deaths')"),
        (sample, "Khulna", "cases", None, "extract_nested_metric(sample, 'Khulna', 'cases')"),
        (sample, "Dhaka", "icu_beds", None, "extract_nested_metric(sample, 'Dhaka', 'icu_beds')"),
        (sample, "Barishal", "cases", None, "extract_nested_metric(sample, 'Barishal', 'cases')"),
        (None, "Dhaka", "cases", None, "extract_nested_metric(None, 'Dhaka', 'cases')"),
    ]
    for data, district, metric, expected, label in tests:
        try:
            got = fn(data, district, metric)
        except NotImplementedError:
            raise
        except (KeyError, TypeError) as exc:
            errors.append(
                f"{label} raised uncaught {type(exc).__name__}: {exc} "
                f"— catch (KeyError, TypeError) and return None!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch10_todo_04",
            False,
            "failed",
            "To-Do 10.4 — Check KeyError & TypeError Handling",
            "Return report_dict[district][metric] inside try, and return None on (KeyError, TypeError).",
            details=errors,
        )
    return CheckResult(
        "ch10_todo_04",
        True,
        "passed",
        "To-Do 10.4 Passed — Safe Nested Dictionary Lookup Verified!",
        "Preserved legitimate 0 values while safely returning None on missing districts, missing metrics, and None containers.",
        tip="EAFP (Easier to Ask for Forgiveness than Permission) with `try/except (KeyError, TypeError)` is clean and idiomatic for nested JSON payloads.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_todo_04",
        chapter="ch10",
        title="To-Do 10.4: Safe Nested Metric Extractor",
        book_section="Section 10.3.2 (EAFP & Multi-Exception Handling)",
        points=10,
        hints={
            1: "Access `report_dict[district][metric]` inside a `try` block and return `None` in `except (KeyError, TypeError):`.",
            2: "Algorithm:\n1. try:\n       return report_dict[district][metric]\n2. except (KeyError, TypeError):\n       return None",
            3: "def extract_nested_metric(report_dict, district, metric):\n    try:\n        return report_dict[district][___]\n    except (KeyError, TypeError):\n        return None",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 10.5: Clean Numeric Series with Error Counter
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_todo_05", "To-Do 10.5", fn, "clean_numeric_series", 10)
    if bad is not None:
        return bad

    sample = ["38.5", "N/A", None, 37, "39.1", ""]
    try:
        nominal = fn(list(sample))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch10_todo_05",
            False,
            "failed",
            "To-Do 10.5 — Runtime Error on Mixed Series",
            f"clean_numeric_series({sample}) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch10_todo_05", "To-Do 10.5", f"clean_numeric_series({sample})", 10
        )

    tests = [
        (["38.5", "N/A", None, 37, "39.1", ""], ([38.5, 37.0, 39.1], 3)),
        ([10.5, 11.2, "12.0"], ([10.5, 11.2, 12.0], 0)),
        (["err", None, "??"], ([], 3)),
        ([], ([], 0)),
    ]
    errors = []
    for raw_list, (exp_vals, exp_errs) in tests:
        label = f"clean_numeric_series({raw_list})"
        try:
            got = fn(list(raw_list))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if not isinstance(got, tuple) or len(got) != 2:
            errors.append(
                f"{label}: expected a 2-tuple (clean_floats, error_count), got {got!r}"
            )
            continue

        clean_vals, err_count = got
        if not isinstance(clean_vals, list):
            errors.append(f"{label}: first item must be a list, got {type(clean_vals).__name__}")
            continue
        if not all(isinstance(v, float) for v in clean_vals):
            errors.append(f"{label}: all cleaned values must be float, got {clean_vals!r}")
            continue
        if len(clean_vals) != len(exp_vals) or not all(
            math.isclose(a, b, abs_tol=1e-6) for a, b in zip(clean_vals, exp_vals)
        ):
            errors.append(f"{label}: expected floats {exp_vals}, got {clean_vals}")
        if err_count != exp_errs or isinstance(err_count, bool):
            errors.append(f"{label}: expected error_count={exp_errs}, got {err_count!r}")

    if errors:
        return CheckResult(
            "ch10_todo_05",
            False,
            "failed",
            "To-Do 10.5 — Check Clean List, Float Casting & Error Count",
            "Convert each item with float(val) inside try/except (ValueError, TypeError), incrementing error_count on failure, and return (clean_floats, error_count).",
            details=errors,
        )
    return CheckResult(
        "ch10_todo_05",
        True,
        "passed",
        "To-Do 10.5 Passed — Resilient Series Cleaner Verified!",
        "Parsed valid numeric entries into floats while counting and skipping corrupt/missing values cleanly.",
        tip="Returning (clean_data, error_count) lets your manuscript Methods section report exact missingness/exclusion counts transparently.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_todo_05",
        chapter="ch10",
        title="To-Do 10.5: Clean Numeric Series with Error Audit",
        book_section="Section 10.3.2 (try / except / else in Loops)",
        points=10,
        hints={
            1: "Initialize `clean = []` and `errors = 0`. For each item, try `num = float(item)` — on `(ValueError, TypeError)` increment `errors`, otherwise (`else`) append `num` to `clean`.",
            2: "Algorithm:\n1. clean = []; errors = 0\n2. for val in raw_values:\n       try:\n           num = float(val)\n       except (ValueError, TypeError):\n           errors += 1\n       else:\n           clean.append(num)\n3. return (clean, errors)",
            3: "def clean_numeric_series(raw_values):\n    clean = []\n    errors = 0\n    for val in raw_values:\n        try:\n            num = float(val)\n        except (ValueError, TypeError):\n            errors += ___\n        else:\n            clean.append(num)\n    return (clean, errors)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 10.6: Automated Unit-Test Harness with assert
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_todo_06", "To-Do 10.6", fn, "verify_bmi_function", 10)
    if bad is not None:
        return bad

    def _valid_rounded(w: float, h: float) -> float:
        return round(w / (h ** 2), 2)

    def _valid_raw(w: float, h: float) -> float:
        return w / (h ** 2)

    # 1. Valid implementations MUST return True
    for label, candidate in [
        ("valid rounded BMI", _valid_rounded),
        ("valid unrounded BMI", _valid_raw),
    ]:
        try:
            got = fn(candidate)
        except NotImplementedError:
            raise
        except AssertionError as exc:
            return CheckResult(
                "ch10_todo_06",
                False,
                "failed",
                "To-Do 10.6 — Valid BMI Implementation Rejected!",
                f"Your verify_bmi_function raised AssertionError on a correct {label} function: {exc}",
                details=[
                    "Check your expected BMI arithmetic: BMI = weight_kg / (height_m ** 2).",
                    "Use a tolerance check like `abs(bmi_fn(70, 1.75) - 22.86) < 0.05` instead of strict `==` so both rounded and unrounded floats pass.",
                ],
            )
        except Exception as exc:
            return CheckResult(
                "ch10_todo_06",
                False,
                "failed",
                "To-Do 10.6 — Runtime Error on Valid BMI Function",
                f"verify_bmi_function({label}) raised {type(exc).__name__}: {exc}",
            )

        if got is None:
            return _none_return_trap(
                "ch10_todo_06", "To-Do 10.6", "verify_bmi_function(valid_bmi)", 10
            )
        if got is not True:
            return CheckResult(
                "ch10_todo_06",
                False,
                "failed",
                "To-Do 10.6 — Return True When All Assertions Pass",
                f"Expected True for a valid BMI function, got {got!r}.",
                details=["End verify_bmi_function(bmi_fn) with `return True` after all `assert` lines."],
            )

    # 2. Buggy implementations MUST raise AssertionError
    buggy_candidates = [
        ("forgot to square height (w / h)", lambda w, h: round(w / h, 2)),
        ("multiplied instead of divided (w * h**2)", lambda w, h: round(w * (h ** 2), 2)),
        ("shifted by +2.0", lambda w, h: round((w / (h ** 2)) + 2.0, 2)),
        ("hardcoded constant 20.0", lambda w, h: 20.0),
        ("hardcoded constant 22.86", lambda w, h: 22.86),
    ]
    errors = []
    for desc, buggy_fn in buggy_candidates:
        try:
            got = fn(buggy_fn)
        except NotImplementedError:
            raise
        except AssertionError:
            continue
        except Exception as exc:
            errors.append(
                f"Buggy candidate ({desc}) raised {type(exc).__name__} instead of AssertionError: {exc}"
            )
        else:
            extra = ""
            if "hardcoded" in desc:
                extra = " (include at least two test cases with DIFFERENT expected BMIs!)"
            errors.append(
                f"Buggy candidate ({desc}) returned {got!r} instead of raising AssertionError!{extra}"
            )

    if errors:
        return CheckResult(
            "ch10_todo_06",
            False,
            "failed",
            "To-Do 10.6 — Test Harness Missed a Buggy BMI Function",
            "Add `assert` checks covering at least two distinct (weight_kg, height_m) pairs using tolerance `< 0.05`.",
            details=errors,
        )
    return CheckResult(
        "ch10_todo_06",
        True,
        "passed",
        "To-Do 10.6 Passed — Automated Unit-Test Harness Verified!",
        "Accepted both rounded and unrounded correct BMI functions while catching every broken implementation via AssertionError.",
        tip="Writing tests that fail on intentionally broken functions ('mutation testing') proves your test suite actually has teeth!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_todo_06",
        chapter="ch10",
        title="To-Do 10.6: Automated BMI Unit-Test Harness",
        book_section="Section 10.3.5 (assert Statements & Unit Testing)",
        points=10,
        hints={
            1: "Call `bmi_fn` on at least two distinct `(weight_kg, height_m)` inputs (e.g., `(80, 2.0) -> 20.0` and `(70, 1.75) -> 22.86`), verify each with `assert abs(got - expected) < 0.05`, then `return True`.",
            2: "Algorithm:\n1. assert abs(bmi_fn(80, 2.0) - 20.0) < 0.05\n2. assert abs(bmi_fn(64, 1.6) - 25.0) < 0.05\n3. assert abs(bmi_fn(70, 1.75) - 22.86) < 0.05\n4. return True",
            3: "def verify_bmi_function(bmi_fn):\n    assert abs(bmi_fn(80, 2.0) - 20.0) < 0.05, 'Failed (80, 2.0)'\n    assert abs(bmi_fn(64, 1.6) - ___) < 0.05, 'Failed (64, 1.6)'\n    assert abs(bmi_fn(70, 1.75) - 22.86) < 0.05, 'Failed (70, 1.75)'\n    return True",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 10.1 (30 pts): Defensive Patient Record Parser
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_exam_01", "Exam Task 10.1", fn, "parse_patient_record", 30)
    if bad is not None:
        return bad

    sample = {"patient_id": "UHC-001", "temp_c": "38.4", "hb_g_dl": "10.6"}
    try:
        nominal = fn(dict(sample))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch10_exam_01",
            False,
            "failed",
            "Exam Task 10.1 — Runtime Error on Valid Patient Record",
            f"parse_patient_record({sample}) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if nominal is None:
        return _none_return_trap(
            "ch10_exam_01", "Exam Task 10.1", f"parse_patient_record({sample})", 30
        )

    errors = []
    valid_cases = [
        (
            {"patient_id": "UHC-001", "temp_c": "38.4", "hb_g_dl": "10.6"},
            {"patient_id": "UHC-001", "temp_c": 38.4, "hb_g_dl": 10.6},
        ),
        (
            {"patient_id": "UHC-002", "temp_c": 37, "hb_g_dl": 12},
            {"patient_id": "UHC-002", "temp_c": 37.0, "hb_g_dl": 12.0},
        ),
        (
            {"patient_id": "UHC-003", "temp_c": 30.0, "hb_g_dl": 2.0},
            {"patient_id": "UHC-003", "temp_c": 30.0, "hb_g_dl": 2.0},
        ),
        (
            {"patient_id": "UHC-004", "temp_c": 45.0, "hb_g_dl": 25.0},
            {"patient_id": "UHC-004", "temp_c": 45.0, "hb_g_dl": 25.0},
        ),
    ]
    for raw_rec, exp in valid_cases:
        label = f"parse_patient_record({raw_rec})"
        try:
            got = fn(dict(raw_rec))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised unexpected {type(exc).__name__}: {exc}")
            continue
        if not isinstance(got, dict):
            errors.append(f"{label}: expected dict, got {type(got).__name__}")
            continue
        for k in ("patient_id", "temp_c", "hb_g_dl"):
            if k not in got:
                errors.append(f"{label}: returned dict missing key {k!r}")
        if got.get("patient_id") != exp["patient_id"]:
            errors.append(
                f"{label}: expected patient_id={exp['patient_id']!r}, got {got.get('patient_id')!r}"
            )
        for num_k in ("temp_c", "hb_g_dl"):
            v = got.get(num_k)
            if not isinstance(v, float) or not math.isclose(v, exp[num_k], abs_tol=1e-6):
                errors.append(
                    f"{label}: expected {num_k}={exp[num_k]} (float), got {v!r} ({type(v).__name__})"
                )

    invalid_cases = [
        ("missing hb_g_dl", {"patient_id": "P1", "temp_c": 37.5}),
        ("missing temp_c", {"patient_id": "P1", "hb_g_dl": 11.2}),
        ("missing patient_id", {"temp_c": 37.5, "hb_g_dl": 11.2}),
        ("blank patient_id", {"patient_id": "   ", "temp_c": 37.5, "hb_g_dl": 11.2}),
        ("non-numeric temp_c", {"patient_id": "P2", "temp_c": "fever", "hb_g_dl": 11.2}),
        ("None hb_g_dl", {"patient_id": "P3", "temp_c": 37.5, "hb_g_dl": None}),
        ("temp_c below 30.0", {"patient_id": "P4", "temp_c": 29.9, "hb_g_dl": 11.2}),
        ("temp_c above 45.0", {"patient_id": "P5", "temp_c": 45.1, "hb_g_dl": 11.2}),
        ("hb_g_dl below 2.0", {"patient_id": "P6", "temp_c": 37.5, "hb_g_dl": 1.9}),
        ("hb_g_dl above 25.0", {"patient_id": "P7", "temp_c": 37.5, "hb_g_dl": 25.1}),
    ]
    for desc, bad_rec in invalid_cases:
        try:
            got = fn(dict(bad_rec))
        except NotImplementedError:
            raise
        except ValueError:
            continue
        except KeyError as exc:
            errors.append(
                f"Invalid case ({desc}) raised KeyError ({exc}) instead of ValueError "
                f"— check required keys and raise ValueError!"
            )
        except TypeError as exc:
            errors.append(
                f"Invalid case ({desc}) raised TypeError ({exc}) instead of ValueError "
                f"— catch TypeError during float() casting and re-raise ValueError!"
            )
        except Exception as exc:
            errors.append(
                f"Invalid case ({desc}) raised {type(exc).__name__} instead of ValueError: {exc}"
            )
        else:
            errors.append(f"Invalid case ({desc}) returned {got!r} instead of raising ValueError!")

    if errors:
        return CheckResult(
            "ch10_exam_01",
            False,
            "failed",
            "Exam Task 10.1 — Check Required Keys, Float Casting & Range Guards",
            "Validate patient_id, temp_c in [30.0, 45.0], and hb_g_dl in [2.0, 25.0], raising ValueError on any violation.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch10_exam_01",
        True,
        "passed",
        "Exam Task 10.1 Passed (30/30 pts) — Defensive Record Parser Verified!",
        "Passed all valid, boundary, missing-key, non-numeric, and physiological range tests.",
        points_earned=30,
        points_possible=30,
        tip="Normalizing diverse raw failures (KeyError, TypeError, out-of-range) into a single ValueError contract makes downstream batch code dramatically simpler.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_exam_01",
        chapter="ch10",
        title="Exam 10.1: Defensive Patient Record Parser",
        book_section="Chapter 10 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Verify all 3 required keys exist (raising `ValueError` if not), check non-empty `patient_id`, cast `temp_c` and `hb_g_dl` to `float` (catching `ValueError, TypeError`), check `[30.0, 45.0]` and `[2.0, 25.0]`, and return the cleaned dict.",
            2: "Algorithm:\n1. Check required keys ('patient_id', 'temp_c', 'hb_g_dl') -> raise ValueError if missing\n2. Validate patient_id is non-empty string\n3. Try casting temp_c and hb_g_dl via float() (rejecting bool) -> raise ValueError on failure\n4. Check 30.0 <= temp_c <= 45.0 and 2.0 <= hb_g_dl <= 25.0\n5. Return {'patient_id': pid, 'temp_c': temp_c, 'hb_g_dl': hb_g_dl}",
            3: "def parse_patient_record(raw_dict):\n    required = ('patient_id', 'temp_c', 'hb_g_dl')\n    for key in required:\n        if key not in raw_dict:\n            raise ValueError(f'Missing key: {key}')\n    # validate patient_id, float conversion, and bounds...\n    return {'patient_id': pid, 'temp_c': temp_c, 'hb_g_dl': hb_g_dl}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 10.2 (35 pts): Resilient Batch Record Validator
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_exam_02", "Exam Task 10.2", fn, "batch_validate_records", 35)
    if bad is not None:
        return bad

    mixed_batch = [
        {"patient_id": "UHC-101", "temp_c": "38.2", "hb_g_dl": "10.8"},
        {"patient_id": "UHC-102", "temp_c": "999", "hb_g_dl": "11.5"},
        {"patient_id": "UHC-103", "temp_c": "bad", "hb_g_dl": "11.0"},
        {"patient_id": "UHC-104", "temp_c": 36.9, "hb_g_dl": 12.4},
        {"patient_id": "UHC-105", "temp_c": 37.5},
    ]

    try:
        nominal = fn([dict(r) for r in mixed_batch])
    except NotImplementedError:
        raise
    except ValueError as exc:
        return CheckResult(
            "ch10_exam_02",
            False,
            "failed",
            "Exam Task 10.2 — Batch Crashed on Invalid Record!",
            f"batch_validate_records raised uncaught ValueError: {exc}",
            details=[
                "Wrap the call to parse_patient_record(record) inside `try / except ValueError as exc:`!",
                "Append valid cleaned dicts to valid_list and `str(exc)` to rejected_errors.",
            ],
            points_earned=0,
            points_possible=35,
        )
    except Exception as exc:
        return CheckResult(
            "ch10_exam_02",
            False,
            "failed",
            "Exam Task 10.2 — Runtime Error on Mixed Batch",
            f"batch_validate_records raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch10_exam_02", "Exam Task 10.2", "batch_validate_records(mixed_batch)", 35
        )

    errors = []
    if not isinstance(nominal, tuple) or len(nominal) != 2:
        errors.append(
            f"Mixed batch: expected 2-tuple (valid_list, rejected_errors), got {nominal!r}"
        )
    else:
        valid_list, rejected_errors = nominal
        if not isinstance(valid_list, list) or not isinstance(rejected_errors, list):
            errors.append("Both elements of the returned tuple must be lists.")
        else:
            expected_valid = [
                {"patient_id": "UHC-101", "temp_c": 38.2, "hb_g_dl": 10.8},
                {"patient_id": "UHC-104", "temp_c": 36.9, "hb_g_dl": 12.4},
            ]
            if len(valid_list) != 2:
                errors.append(f"Mixed batch: expected 2 valid records, got {len(valid_list)}")
            elif valid_list != expected_valid:
                errors.append(f"Mixed batch: expected valid={expected_valid}, got {valid_list}")

            if len(rejected_errors) != 3:
                errors.append(
                    f"Mixed batch: expected 3 rejected error messages, got {len(rejected_errors)}"
                )
            elif not all(isinstance(msg, str) and len(msg.strip()) > 0 for msg in rejected_errors):
                errors.append(
                    f"rejected_errors must contain non-empty strings, got {rejected_errors!r}"
                )

    # Extra edge batches: all valid, all invalid, empty
    extra_batches = [
        (
            "all-valid batch",
            [{"patient_id": "P1", "temp_c": 37.0, "hb_g_dl": 12.0}],
            1,
            0,
        ),
        (
            "all-invalid batch",
            [{"patient_id": "P2", "temp_c": 10.0, "hb_g_dl": 12.0}, {"temp_c": 37.0}],
            0,
            2,
        ),
        ("empty batch", [], 0, 0),
    ]
    for desc, batch, exp_v, exp_e in extra_batches:
        try:
            got = fn([dict(r) for r in batch])
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{desc} raised {type(exc).__name__}: {exc}")
            continue
        if not isinstance(got, tuple) or len(got) != 2:
            errors.append(f"{desc}: expected (valid_list, rejected_errors), got {got!r}")
            continue
        v_list, e_list = got
        if len(v_list) != exp_v or len(e_list) != exp_e:
            errors.append(
                f"{desc}: expected ({exp_v} valid, {exp_e} rejected), got ({len(v_list)}, {len(e_list)})"
            )

    if errors:
        return CheckResult(
            "ch10_exam_02",
            False,
            "failed",
            "Exam Task 10.2 — Check Try/Except Batch Isolation & Return Tuple",
            "Process every record through parse_patient_record inside try/except ValueError, collecting (valid_list, rejected_errors).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch10_exam_02",
        True,
        "passed",
        "Exam Task 10.2 Passed (35/35 pts) — Resilient Batch Validator Verified!",
        "Isolated bad records without crashing, preserved all clean records, and logged clear error strings across mixed, empty, and all-invalid batches.",
        points_earned=35,
        points_possible=35,
        tip="Quarantining corrupt rows into a rejection log instead of halting an overnight ETL job is a hallmark of production research pipelines.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_exam_02",
        chapter="ch10",
        title="Exam 10.2: Resilient Batch Record Validator",
        book_section="Chapter 10 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Loop over `raw_records`, call `parse_patient_record(rec)` inside `try`, append valid dicts in `else` (or after the call), and append `str(exc)` inside `except ValueError as exc:`.",
            2: "Algorithm:\n1. valid_list = []; rejected_errors = []\n2. for rec in raw_records:\n       try:\n           cleaned = parse_patient_record(rec)\n       except ValueError as exc:\n           rejected_errors.append(str(exc))\n       else:\n           valid_list.append(cleaned)\n3. return (valid_list, rejected_errors)",
            3: "def batch_validate_records(raw_records):\n    valid_list = []\n    rejected_errors = []\n    for rec in raw_records:\n        try:\n            cleaned = parse_patient_record(rec)\n        except ValueError as exc:\n            rejected_errors.append(str(exc))\n        else:\n            valid_list.append(___)\n    return (valid_list, rejected_errors)",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 10.3 (35 pts): Strictly Validated 2x2 Odds Ratio
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch10_exam_03", "Exam Task 10.3", fn, "compute_safe_odds_ratio", 35)
    if bad is not None:
        return bad

    # Nominal test first
    try:
        nominal = fn(45, 105, 20, 130)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch10_exam_03",
            False,
            "failed",
            "Exam Task 10.3 — Runtime Error on Nominal Input",
            f"compute_safe_odds_ratio(45, 105, 20, 130) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch10_exam_03", "Exam Task 10.3", "compute_safe_odds_ratio(45, 105, 20, 130)", 35
        )

    errors = []
    valid_tests = [
        ((45, 105, 20, 130), 2.79),
        ((30, 70, 30, 70), 1.0),
        ((10, 90, 25, 75), 0.33),
        ((0, 80, 15, 85), 0.0),
        ((20, 0, 15, 85), None),
        ((20, 80, 0, 85), None),
        ((0, 0, 0, 0), None),
    ]
    for (a, b, c, d), expected in valid_tests:
        label = f"compute_safe_odds_ratio({a}, {b}, {c}, {d})"
        try:
            got = fn(a, b, c, d)
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                f"{label} raised ZeroDivisionError — return None when b == 0 or c == 0!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised unexpected {type(exc).__name__}: {exc}")
            continue

        if expected is None:
            if got is not None:
                errors.append(f"{label}: expected None (zero cell), got {got!r}")
        else:
            if _is_number(got) and (a, b, c, d) == (45, 105, 20, 130) and math.isclose(got, 1114.29, abs_tol=0.1):
                errors.append(
                    f"{label} returned {got!r} — missing parentheses around (b * c) in denominator!"
                )
            elif not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
                errors.append(f"{label}: expected {expected}, got {got!r}")

    invalid_tests = [
        ("negative a", (-5, 10, 20, 30)),
        ("negative b", (10, -1, 20, 30)),
        ("negative c", (10, 20, -1, 30)),
        ("negative d", (10, 20, 30, -1)),
        ("float input", (10.5, 20, 30, 40)),
        ("bool True", (True, 10, 20, 30)),
        ("bool False", (10, False, 20, 30)),
        ("string input", ("10", 20, 30, 40)),
        ("None input", (10, None, 20, 30)),
    ]
    for desc, args in invalid_tests:
        label = f"compute_safe_odds_ratio{args}"
        try:
            got = fn(*args)
        except NotImplementedError:
            raise
        except ValueError:
            continue
        except TypeError as exc:
            errors.append(
                f"{label} ({desc}) raised TypeError instead of ValueError ({exc}) "
                f"— validate isinstance(v, int) BEFORE checking v < 0!"
            )
        except Exception as exc:
            errors.append(f"{label} ({desc}) raised {type(exc).__name__} instead of ValueError: {exc}")
        else:
            extra = ""
            if "bool" in desc:
                extra = " (Remember: in Python isinstance(True, int) is True — explicitly reject isinstance(v, bool)!)"
            errors.append(f"{label} ({desc}) returned {got!r} instead of raising ValueError!{extra}")

    if errors:
        return CheckResult(
            "ch10_exam_03",
            False,
            "failed",
            "Exam Task 10.3 — Check Strict Non-Negative Int Validation & Zero Guard",
            "Raise ValueError unless all four inputs are non-negative ints (not bool); return None if b == 0 or c == 0; else return round((a * d) / (b * c), 2).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch10_exam_03",
        True,
        "passed",
        "Exam Task 10.3 Passed (35/35 pts) — Hardened 2x2 Odds Ratio Verified!",
        "Enforced strict non-negative integer contracts (rejecting bools, floats, strings, None, and negatives) while handling zero denominators cleanly.",
        points_earned=35,
        points_possible=35,
        tip="Because `bool` subclasses `int` in Python (`isinstance(True, int) == True`), defensive validators always guard with `isinstance(v, bool)`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch10_exam_03",
        chapter="ch10",
        title="Exam 10.3: Hardened 2x2 Odds Ratio Validator",
        book_section="Chapter 10 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Loop over `(a, b, c, d)` and raise `ValueError` if `isinstance(val, bool) or not isinstance(val, int) or val < 0`. Next return `None` if `b == 0 or c == 0`, else `round((a * d) / (b * c), 2)`.",
            2: "Algorithm:\n1. for val in (a, b, c, d):\n       if isinstance(val, bool) or not isinstance(val, int) or val < 0:\n           raise ValueError('Cell counts must be non-negative integers')\n2. if b == 0 or c == 0:\n       return None\n3. return round((a * d) / (b * c), 2)",
            3: "def compute_safe_odds_ratio(a, b, c, d):\n    for val in (a, b, c, d):\n        if isinstance(val, bool) or not isinstance(val, int) or val < 0:\n            raise ValueError('Counts must be non-negative ints')\n    if b == 0 or c == 0:\n        return ___\n    return round((a * d) / (b * c), 2)",
        },
        verifier=_verify_exam_03,
    )
)
