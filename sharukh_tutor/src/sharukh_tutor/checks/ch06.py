"""Chapter 6 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch06_todo_01..06) and Part I Capstone Coding Exam
(ch06_exam_01..03). Every verifier tests a callable function across both
nominal and edge-case inputs and explicitly diagnoses the print()-vs-return
None trap.
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
        f"Calling {call_str} returned None instead of the computed value.",
        details=[
            "This is the #1 beginner function trap: print() only displays text on screen; it sends back None to the caller.",
            "Make sure your function ends with an explicit `return ...` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Think of print() as a monitor screen, and return as the lab report handed back to downstream code.",
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 6.1: Celsius to Fahrenheit Converter
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_todo_01", "To-Do 6.1", fn, "celsius_to_fahrenheit", 10)
    if bad is not None:
        return bad

    cases = [
        (37.0, 98.6),
        (38.5, 101.3),
        (0.0, 32.0),
        (40.0, 104.0),
    ]
    errors = []
    for temp_c, expected in cases:
        try:
            got = fn(temp_c)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch06_todo_01",
                False,
                "failed",
                "To-Do 6.1 — Runtime Error When Calling Function",
                f"celsius_to_fahrenheit({temp_c}) raised {type(exc).__name__}: {exc}",
                details=["Check that your function accepts one parameter (temp_c) and uses numeric math."],
            )
        if got is None:
            return _none_return_trap(
                "ch06_todo_01", "To-Do 6.1", f"celsius_to_fahrenheit({temp_c})", 10
            )
        if not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
            errors.append(
                f"celsius_to_fahrenheit({temp_c}): expected {expected}, got {got!r}"
            )

    if errors:
        return CheckResult(
            "ch06_todo_01",
            False,
            "failed",
            "To-Do 6.1 — Check Temperature Conversion Formula",
            "Use F = (temp_c * 9 / 5) + 32 and round to 2 decimal places.",
            details=errors,
        )
    return CheckResult(
        "ch06_todo_01",
        True,
        "passed",
        "To-Do 6.1 Passed — Reusable Temperature Converter Verified!",
        "Passed all 4 clinical and cold-chain test cases (0.0 C -> 32.0 F, 37.0 C -> 98.6 F, 38.5 C -> 101.3 F, 40.0 C -> 104.0 F).",
        tip="A pure function takes input parameters and returns a value without relying on global variables.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_todo_01",
        chapter="ch06",
        title="To-Do 6.1: Celsius to Fahrenheit Function",
        book_section="Section 6.3.1 (Anatomy of def & return)",
        points=10,
        hints={
            1: "Multiply temp_c by 9 / 5, add 32, round to 2 decimal places, and use `return` (not `print`).",
            2: "Algorithm:\n1. temp_f = (temp_c * 9 / 5) + 32\n2. return round(temp_f, 2)",
            3: "def celsius_to_fahrenheit(temp_c):\n    return round((temp_c * 9 / 5) + ___, 2)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 6.2: Reusable BMI Function (cm -> m)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_todo_02", "To-Do 6.2", fn, "compute_bmi", 10)
    if bad is not None:
        return bad

    cases = [
        (64.5, 168.0, 22.85),
        (54.6, 156.0, 22.44),
        (70.0, 175.0, 22.86),
        (48.0, 160.0, 18.75),
    ]
    errors = []
    for weight_kg, height_cm, expected in cases:
        try:
            got = fn(weight_kg, height_cm)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch06_todo_02",
                False,
                "failed",
                "To-Do 6.2 — Runtime Error When Calling Function",
                f"compute_bmi({weight_kg}, {height_cm}) raised {type(exc).__name__}: {exc}",
                details=["Make sure compute_bmi accepts two parameters: (weight_kg, height_cm)."],
            )
        if got is None:
            return _none_return_trap(
                "ch06_todo_02", "To-Do 6.2", f"compute_bmi({weight_kg}, {height_cm})", 10
            )
        if _is_number(got) and got < 0.05:
            errors.append(
                f"compute_bmi({weight_kg}, {height_cm}) returned {got!r} — did you forget to convert height_cm to meters (height_cm / 100)?"
            )
        elif not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
            errors.append(
                f"compute_bmi({weight_kg}, {height_cm}): expected {expected}, got {got!r}"
            )

    if errors:
        return CheckResult(
            "ch06_todo_02",
            False,
            "failed",
            "To-Do 6.2 — Check BMI Unit Conversion & Rounding",
            "Convert height_cm to meters first (height_m = height_cm / 100), then return round(weight_kg / (height_m ** 2), 2).",
            details=errors,
        )
    return CheckResult(
        "ch06_todo_02",
        True,
        "passed",
        "To-Do 6.2 Passed — Reusable BMI Calculator Verified!",
        "Passed all 4 anthropometric test cases with exact cm-to-m unit conversion.",
        tip="Encapsulating unit conversions inside a function guarantees every script in your lab applies the exact same formula.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_todo_02",
        chapter="ch06",
        title="To-Do 6.2: Reusable BMI Calculator",
        book_section="Section 6.3.1 (Parameters, Return Values & Docstrings)",
        points=10,
        hints={
            1: "Convert height_cm to meters by dividing by 100, compute weight_kg / (height_m ** 2), and return the value rounded to 2 decimal places.",
            2: "Algorithm:\n1. height_m = height_cm / 100\n2. bmi = weight_kg / (height_m ** 2)\n3. return round(bmi, 2)",
            3: "def compute_bmi(weight_kg, height_cm):\n    height_m = height_cm / ___\n    return round(weight_kg / (height_m ** 2), 2)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 6.3: Anemia Classifier with Default Parameter
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_todo_03", "To-Do 6.3", fn, "classify_anemia", 10)
    if bad is not None:
        return bad

    # Test default parameter (called with 1 arg) first
    try:
        default_res = fn(10.4)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch06_todo_03",
            False,
            "failed",
            "To-Do 6.3 — Missing Default Value for pregnant Parameter",
            f"Calling classify_anemia(10.4) with 1 argument raised TypeError: {exc}",
            details=[
                "Define the signature with a default parameter: def classify_anemia(hb_g_dl, pregnant=True):",
            ],
        )
    except Exception as exc:
        return CheckResult(
            "ch06_todo_03",
            False,
            "failed",
            "To-Do 6.3 — Runtime Error",
            f"classify_anemia(10.4) raised {type(exc).__name__}: {exc}",
        )

    if default_res is None:
        return _none_return_trap("ch06_todo_03", "To-Do 6.3", "classify_anemia(10.4)", 10)

    tests = [
        ((10.4,), {}, "Anemic", "classify_anemia(10.4)"),
        ((11.0,), {}, "Normal", "classify_anemia(11.0)"),
        ((11.5,), {}, "Normal", "classify_anemia(11.5)"),
        ((11.5,), {"pregnant": False}, "Anemic", "classify_anemia(11.5, pregnant=False)"),
        ((12.0,), {"pregnant": False}, "Normal", "classify_anemia(12.0, pregnant=False)"),
        ((12.4, False), {}, "Normal", "classify_anemia(12.4, False)"),
    ]
    errors = []
    for args, kwargs, expected, label in tests:
        try:
            got = fn(*args, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if got is None:
            return _none_return_trap("ch06_todo_03", "To-Do 6.3", label, 10)
        if got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch06_todo_03",
            False,
            "failed",
            "To-Do 6.3 — Check WHO Cutoffs & Default Parameter",
            "Use cutoff 11.0 when pregnant=True (default) and 12.0 when pregnant=False (strictly `< cutoff` -> 'Anemic', else 'Normal').",
            details=errors,
        )
    return CheckResult(
        "ch06_todo_03",
        True,
        "passed",
        "To-Do 6.3 Passed — Default Parameter & WHO Cutoffs Verified!",
        "Passed default pregnant=True (11.0 g/dL), non-pregnant (12.0 g/dL), and exact boundary checks.",
        tip="Default parameters make research functions ergonomic for the common case while remaining flexible for subgroups.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_todo_03",
        chapter="ch06",
        title="To-Do 6.3: WHO Anemia Classifier (Default Parameter)",
        book_section="Section 6.3.3 (Default & Keyword Arguments)",
        points=10,
        hints={
            1: "Set `cutoff = 11.0 if pregnant else 12.0`, then return `'Anemic'` if `hb_g_dl < cutoff` else `'Normal'`.",
            2: "Algorithm:\n1. def classify_anemia(hb_g_dl, pregnant=True):\n2. cutoff = 11.0 if pregnant else 12.0\n3. if hb_g_dl < cutoff: return 'Anemic'\n4. return 'Normal'",
            3: "def classify_anemia(hb_g_dl, pregnant=True):\n    cutoff = 11.0 if pregnant else ___\n    return 'Anemic' if hb_g_dl < cutoff else 'Normal'",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 6.4: Safe Incidence Rate with Guard Clause
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_todo_04", "To-Do 6.4", fn, "safe_incidence_rate", 10)
    if bad is not None:
        return bad

    # Test nominal case FIRST to distinguish print-only None trap from valid None guard
    try:
        nominal = fn(100, 250000)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch06_todo_04",
            False,
            "failed",
            "To-Do 6.4 — Check Function Signature & Default per=100000",
            f"Calling safe_incidence_rate(100, 250000) raised TypeError: {exc}",
            details=["Use signature: def safe_incidence_rate(cases, population, per=100000):"],
        )
    except Exception as exc:
        return CheckResult(
            "ch06_todo_04",
            False,
            "failed",
            "To-Do 6.4 — Runtime Error on Nominal Input",
            f"safe_incidence_rate(100, 250000) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch06_todo_04", "To-Do 6.4", "safe_incidence_rate(100, 250000)", 10
        )

    errors = []
    if not _is_number(nominal) or not math.isclose(nominal, 40.0, abs_tol=1e-2):
        errors.append(f"safe_incidence_rate(100, 250000): expected 40.0, got {nominal!r}")

    extra_tests = [
        ((45, 15000), {"per": 1000}, 3.0, "safe_incidence_rate(45, 15000, per=1000)"),
        ((0, 50000), {}, 0.0, "safe_incidence_rate(0, 50000)"),
        ((12, 0), {}, None, "safe_incidence_rate(12, 0)"),
        ((12, -500), {}, None, "safe_incidence_rate(12, -500)"),
    ]
    for args, kwargs, expected, label in extra_tests:
        try:
            got = fn(*args, **kwargs)
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                f"{label} raised ZeroDivisionError — add a guard clause `if population <= 0: return None` BEFORE dividing!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if expected is None:
            if got is not None:
                errors.append(f"{label}: expected None (invalid population), got {got!r}")
        else:
            if not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
                errors.append(f"{label}: expected {expected}, got {got!r}")

    if errors:
        return CheckResult(
            "ch06_todo_04",
            False,
            "failed",
            "To-Do 6.4 — Check Guard Clause & Rate Calculation",
            "Return None when population <= 0; otherwise return round((cases / population) * per, 2).",
            details=errors,
        )
    return CheckResult(
        "ch06_todo_04",
        True,
        "passed",
        "To-Do 6.4 Passed — Guard-Protected Incidence Rate Verified!",
        "Handled default per=100000, custom per=1000, zero cases, and population <= 0 safely.",
        tip="Early-return guard clauses keep your function flat, readable, and crash-proof on messy district tables.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_todo_04",
        chapter="ch06",
        title="To-Do 6.4: Safe Incidence Rate with Guard Clause",
        book_section="Section 6.3.3 & Section 6.4 (Guard Clauses & Defaults)",
        points=10,
        hints={
            1: "Put `if population <= 0: return None` as the very first line, then return `round((cases / population) * per, 2)`.",
            2: "Algorithm:\n1. def safe_incidence_rate(cases, population, per=100000):\n2. if population <= 0: return None\n3. return round((cases / population) * per, 2)",
            3: "def safe_incidence_rate(cases, population, per=100000):\n    if population <= 0:\n        return ___\n    return round((cases / population) * per, 2)",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 6.5: Clean Mean of Lab Readings (Skip None & 999)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_todo_05", "To-Do 6.5", fn, "clean_mean_readings", 10)
    if bad is not None:
        return bad

    nominal_input = [10.4, None, 11.8, 999, 10.8]
    try:
        nominal = fn(list(nominal_input))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch06_todo_05",
            False,
            "failed",
            "To-Do 6.5 — Runtime Error on Nominal List",
            f"clean_mean_readings({nominal_input}) raised {type(exc).__name__}: {exc}",
            details=["Check that you skip None BEFORE comparing or adding to the running total."],
        )

    if nominal is None:
        return _none_return_trap(
            "ch06_todo_05", "To-Do 6.5", f"clean_mean_readings({nominal_input})", 10
        )

    errors = []
    if _is_number(nominal) and nominal > 100:
        errors.append(
            f"clean_mean_readings({nominal_input}) returned {nominal!r} — sentinel value 999 was not skipped!"
        )
    elif _is_number(nominal) and math.isclose(nominal, 6.6, abs_tol=1e-2):
        errors.append(
            f"clean_mean_readings({nominal_input}) returned {nominal!r} — you divided by len(readings) (5) instead of valid_count (3)!"
        )
    elif not _is_number(nominal) or not math.isclose(nominal, 11.0, abs_tol=1e-2):
        errors.append(f"clean_mean_readings({nominal_input}): expected 11.0, got {nominal!r}")

    extra_tests = [
        ([12.1, 11.5, 10.6], 11.4),
        ([9.5, 999, 10.2], 9.85),
        ([None, 999, None], None),
        ([], None),
    ]
    for data, expected in extra_tests:
        try:
            got = fn(list(data))
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                f"clean_mean_readings({data}) raised ZeroDivisionError — return None when valid_count == 0!"
            )
            continue
        except Exception as exc:
            errors.append(f"clean_mean_readings({data}) raised {type(exc).__name__}: {exc}")
            continue

        if expected is None:
            if got is not None:
                errors.append(f"clean_mean_readings({data}): expected None, got {got!r}")
        else:
            if not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
                errors.append(f"clean_mean_readings({data}): expected {expected}, got {got!r}")

    if errors:
        return CheckResult(
            "ch06_todo_05",
            False,
            "failed",
            "To-Do 6.5 — Check Sentinel Filtering & Empty-Valid Guard",
            "Skip None and 999, divide by the count of valid readings, round to 2 decimals, and return None if no valid readings exist.",
            details=errors,
        )
    return CheckResult(
        "ch06_todo_05",
        True,
        "passed",
        "To-Do 6.5 Passed — Sentinel-Safe Mean Function Verified!",
        "Correctly filtered None and 999, divided by valid count, and returned None on empty/all-invalid batches.",
        tip="Notice how wrapping Chapter 5's sentinel loop inside a function turns a one-off script into a reusable cleaning tool.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_todo_05",
        chapter="ch06",
        title="To-Do 6.5: Sentinel-Safe Mean of Lab Readings",
        book_section="Section 6.3.4 & Section 6.4 (Modular Cleaning Helpers)",
        points=10,
        hints={
            1: "Loop over `readings`, skip any `val is None or val == 999`, track `total` and `count`, return `None` if `count == 0`, else `round(total / count, 2)`.",
            2: "Algorithm:\n1. total = 0.0; count = 0\n2. for val in readings:\n       if val is None or val == 999: continue\n       total += val; count += 1\n3. if count == 0: return None\n4. return round(total / count, 2)",
            3: "def clean_mean_readings(readings):\n    valid = [v for v in readings if v is not None and v != ___]\n    if not valid:\n        return None\n    return round(sum(valid) / len(valid), 2)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 6.6: Multi-Indicator Clinical Triage Priority
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_todo_06", "To-Do 6.6", fn, "triage_priority", 10)
    if bad is not None:
        return bad

    cases = [
        ((39.8, 120, 12.2), "Urgent"),
        ((37.0, 165, 12.0), "Urgent"),
        ((36.8, 118, 7.4), "Urgent"),
        ((39.5, 118, 12.5), "Urgent"),
        ((38.4, 122, 12.0), "Moderate"),
        ((37.1, 146, 12.1), "Moderate"),
        ((36.9, 118, 10.2), "Moderate"),
        ((38.0, 118, 12.5), "Moderate"),
        ((36.8, 118, 12.5), "Routine"),
        ((37.9, 139, 11.0), "Routine"),
    ]
    errors = []
    for (temp_c, sbp_mmhg, hb_g_dl), expected in cases:
        label = f"triage_priority({temp_c}, {sbp_mmhg}, {hb_g_dl})"
        try:
            got = fn(temp_c, sbp_mmhg, hb_g_dl)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch06_todo_06",
                False,
                "failed",
                "To-Do 6.6 — Runtime Error When Calling Function",
                f"{label} raised {type(exc).__name__}: {exc}",
            )
        if got is None:
            return _none_return_trap("ch06_todo_06", "To-Do 6.6", label, 10)
        if got != expected:
            hint_extra = ""
            if expected == "Urgent" and got == "Moderate":
                hint_extra = " (check Urgent conditions BEFORE Moderate conditions!)"
            errors.append(f"{label}: expected {expected!r}, got {got!r}{hint_extra}")

    if errors:
        return CheckResult(
            "ch06_todo_06",
            False,
            "failed",
            "To-Do 6.6 — Check Clinical Triage Thresholds & Branch Order",
            "Urgent: temp >= 39.5 or sbp >= 160 or hb < 8.0. Moderate: temp >= 38.0 or sbp >= 140 or hb < 11.0. Otherwise: Routine.",
            details=errors,
        )
    return CheckResult(
        "ch06_todo_06",
        True,
        "passed",
        "To-Do 6.6 Passed — Multi-Indicator Triage Engine Verified!",
        "Passed all 10 clinical test cases including Urgent, Moderate, Routine, and exact cutoff boundaries.",
        tip="Always test highest-severity triage criteria first so critical patients are never downgraded by broader moderate rules.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_todo_06",
        chapter="ch06",
        title="To-Do 6.6: Multi-Indicator Clinical Triage Function",
        book_section="Section 6.4 (Modular Clinical Decision Functions)",
        points=10,
        hints={
            1: "Check 'Urgent' first (temp_c >= 39.5 or sbp_mmhg >= 160 or hb_g_dl < 8.0), then 'Moderate' (temp_c >= 38.0 or sbp_mmhg >= 140 or hb_g_dl < 11.0), else return 'Routine'.",
            2: "Algorithm:\n1. if temp_c >= 39.5 or sbp_mmhg >= 160 or hb_g_dl < 8.0: return 'Urgent'\n2. if temp_c >= 38.0 or sbp_mmhg >= 140 or hb_g_dl < 11.0: return 'Moderate'\n3. return 'Routine'",
            3: "def triage_priority(temp_c, sbp_mmhg, hb_g_dl):\n    if temp_c >= 39.5 or sbp_mmhg >= 160 or hb_g_dl < 8.0:\n        return 'Urgent'\n    if temp_c >= ___ or sbp_mmhg >= ___ or hb_g_dl < ___:\n        return 'Moderate'\n    return 'Routine'",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 6.1 (30 pts): District Name Normalizer
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_exam_01", "Exam Task 6.1", fn, "normalize_district_name", 30)
    if bad is not None:
        return bad

    # Nominal test first to catch print()-only None return
    try:
        nominal = fn("  dhaka   north  ")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch06_exam_01",
            False,
            "failed",
            "Exam Task 6.1 — Runtime Error on Nominal Input",
            f"normalize_district_name('  dhaka   north  ') raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if nominal is None:
        return _none_return_trap(
            "ch06_exam_01", "Exam Task 6.1", "normalize_district_name('  dhaka   north  ')", 30
        )

    tests = [
        ("  dhaka   north  ", "Dhaka North"),
        ("SYLHET", "Sylhet"),
        ("  chattogram\t  south\n", "Chattogram South"),
        ("rajshahi", "Rajshahi"),
        ("", "Unknown"),
        ("     ", "Unknown"),
        (None, "Unknown"),
    ]
    errors = []
    for raw_val, expected in tests:
        label = f"normalize_district_name({raw_val!r})"
        try:
            got = fn(raw_val)
        except NotImplementedError:
            raise
        except AttributeError:
            errors.append(
                f"{label} raised AttributeError — check `if raw_name is None:` BEFORE calling string methods!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if got != expected:
            extra = ""
            if isinstance(got, str) and "  " in got:
                extra = " (use `' '.join(raw_name.split())` to collapse extra internal spaces!)"
            errors.append(f"{label}: expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch06_exam_01",
            False,
            "failed",
            "Exam Task 6.1 — Check Whitespace Collapsing & Empty/None Guards",
            "Strip/collapse all whitespace, title-case valid names, and return 'Unknown' for None or blank text.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch06_exam_01",
        True,
        "passed",
        "Exam Task 6.1 Passed (30/30 pts) — District Normalizer Verified!",
        "Cleanly normalized messy whitespace, casing, blank strings, and None inputs.",
        points_earned=30,
        points_possible=30,
        tip="' '.join(text.split()) is one of Python's cleanest idioms for collapsing arbitrary spaces, tabs, and newlines.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_exam_01",
        chapter="ch06",
        title="Exam 6.1: District Name Normalizer",
        book_section="Chapter 6 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Guard against `raw_name is None` first, collapse spaces with `' '.join(raw_name.split())`, return `'Unknown'` if empty, else return `.title()`.",
            2: "Algorithm:\n1. if raw_name is None: return 'Unknown'\n2. cleaned = ' '.join(raw_name.split())\n3. if not cleaned: return 'Unknown'\n4. return cleaned.title()",
            3: "def normalize_district_name(raw_name):\n    if raw_name is None:\n        return 'Unknown'\n    cleaned = ' '.join(raw_name.split())\n    return cleaned.title() if cleaned else ___",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 6.2 (35 pts): Safe 2x2 Odds Ratio Function
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_exam_02", "Exam Task 6.2", fn, "compute_odds_ratio", 35)
    if bad is not None:
        return bad

    # Nominal test first to catch print()-only None return
    try:
        nominal = fn(45, 105, 20, 130)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch06_exam_02",
            False,
            "failed",
            "Exam Task 6.2 — Runtime Error on Nominal Input",
            f"compute_odds_ratio(45, 105, 20, 130) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch06_exam_02", "Exam Task 6.2", "compute_odds_ratio(45, 105, 20, 130)", 35
        )

    tests = [
        ((45, 105, 20, 130), 2.79),
        ((30, 70, 30, 70), 1.0),
        ((10, 90, 25, 75), 0.33),
        ((0, 80, 15, 85), 0.0),
        ((20, 0, 15, 85), None),
        ((20, 80, 0, 85), None),
    ]
    errors = []
    for (a, b, c, d), expected in tests:
        label = f"compute_odds_ratio({a}, {b}, {c}, {d})"
        try:
            got = fn(a, b, c, d)
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                f"{label} raised ZeroDivisionError — add `if b == 0 or c == 0: return None` BEFORE dividing!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if expected is None:
            if got is not None:
                errors.append(f"{label}: expected None (zero denominator cell), got {got!r}")
        else:
            if _is_number(got) and (a, b, c, d) == (45, 105, 20, 130) and math.isclose(got, 1114.29, abs_tol=0.1):
                errors.append(
                    f"{label} returned {got!r} — missing parentheses around (b * c) in the denominator!"
                )
            elif not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
                errors.append(f"{label}: expected {expected}, got {got!r}")

    if errors:
        return CheckResult(
            "ch06_exam_02",
            False,
            "failed",
            "Exam Task 6.2 — Check Odds Ratio Formula & Zero-Cell Guard",
            "Return None if b == 0 or c == 0; otherwise return round((a * d) / (b * c), 2).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch06_exam_02",
        True,
        "passed",
        "Exam Task 6.2 Passed (35/35 pts) — Safe 2x2 Odds Ratio Verified!",
        "Passed elevated risk (2.79), null (1.0), protective (0.33), zero-a (0.0), and zero-denominator guards.",
        points_earned=35,
        points_possible=35,
        tip="In epidemiology pipelines, zero cells happen frequently in rare diseases—explicit None guards prevent batch crashes.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_exam_02",
        chapter="ch06",
        title="Exam 6.2: Safe 2x2 Odds Ratio Function",
        book_section="Chapter 6 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Return None immediately if `b == 0 or c == 0`; otherwise return `round((a * d) / (b * c), 2)`.",
            2: "Algorithm:\n1. if b == 0 or c == 0: return None\n2. return round((a * d) / (b * c), 2)",
            3: "def compute_odds_ratio(a, b, c, d):\n    if b == 0 or c == 0:\n        return ___\n    return round((a * d) / (b * c), 2)",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 6.3 (35 pts): Clinic Batch Hemoglobin Summarizer
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch06_exam_03", "Exam Task 6.3", fn, "summarize_clinic_batch", 35)
    if bad is not None:
        return bad

    nominal_batch = [10.4, None, 11.8, 999, 9.6, 12.2]
    try:
        nominal = fn(list(nominal_batch))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch06_exam_03",
            False,
            "failed",
            "Exam Task 6.3 — Runtime Error on Nominal Batch",
            f"summarize_clinic_batch({nominal_batch}) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch06_exam_03", "Exam Task 6.3", f"summarize_clinic_batch({nominal_batch})", 35
        )

    tests = [
        ([10.4, None, 11.8, 999, 9.6, 12.2], (4, 2, 11.0)),
        ([11.0, 12.4, 11.6], (3, 0, 11.67)),
        ([9.5, 10.2, 999], (2, 2, 9.85)),
        ([None, 999, None], (0, 0, None)),
        ([], (0, 0, None)),
    ]
    errors = []
    for batch, (exp_valid, exp_anemic, exp_mean) in tests:
        label = f"summarize_clinic_batch({batch})"
        try:
            got = fn(list(batch))
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                f"{label} raised ZeroDivisionError — return (0, 0, None) when there are no valid readings!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if not isinstance(got, tuple) or len(got) != 3:
            errors.append(
                f"{label}: expected a 3-tuple (valid_count, anemic_count, mean_hb), got {type(got).__name__} ({got!r})"
            )
            continue

        got_valid, got_anemic, got_mean = got
        mean_ok = (
            (got_mean is None)
            if exp_mean is None
            else (_is_number(got_mean) and math.isclose(got_mean, exp_mean, abs_tol=1e-2))
        )
        if type(got_valid) is not int or got_valid != exp_valid or type(got_anemic) is not int or got_anemic != exp_anemic or not mean_ok:
            extra = ""
            if batch == [11.0, 12.4, 11.6] and got_anemic == 1:
                extra = " (anemia is strictly < 11.0, so 11.0 is Normal!)"
            errors.append(
                f"{label}: expected ({exp_valid}, {exp_anemic}, {exp_mean!r}), got {got!r}{extra}"
            )

    if errors:
        return CheckResult(
            "ch06_exam_03",
            False,
            "failed",
            "Exam Task 6.3 — Check Tuple Return, Sentinel Skipping & Empty Guard",
            "Skip None and 999, count valid and anemic (< 11.0) readings, round mean to 2 decimals, and return (0, 0, None) when no valid readings exist.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch06_exam_03",
        True,
        "passed",
        "Exam Task 6.3 Passed (35/35 pts) — Clinic Batch Summarizer Verified!",
        "Returned exact (valid_count, anemic_count, mean_hb) tuples across nominal, boundary, and empty batches.",
        points_earned=35,
        points_possible=35,
        tip="Returning structured tuples from summary functions lets callers unpack results cleanly: valid_n, anemic_n, avg_hb = summarize_clinic_batch(data).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch06_exam_03",
        chapter="ch06",
        title="Exam 6.3: Clinic Batch Hemoglobin Summarizer",
        book_section="Chapter 6 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Filter out None and 999. If no valid readings remain, return (0, 0, None). Otherwise compute valid_count, anemic_count (< 11.0), and round(total / valid_count, 2).",
            2: "Algorithm:\n1. valid_count = 0; anemic_count = 0; total = 0.0\n2. Loop over hb_readings; skip None and 999\n3. Update valid_count, total, and anemic_count (if val < 11.0)\n4. If valid_count == 0: return (0, 0, None)\n5. return (valid_count, anemic_count, round(total / valid_count, 2))",
            3: "def summarize_clinic_batch(hb_readings):\n    valid = [x for x in hb_readings if x is not None and x != 999]\n    if not valid:\n        return (0, 0, None)\n    anemic_count = sum(1 for x in valid if x < ___)\n    return (len(valid), anemic_count, round(sum(valid) / len(valid), 2))",
        },
        verifier=_verify_exam_03,
    )
)
