"""Chapter 8 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch08_todo_01..06) and Chapter 8 Coding Exam
(ch08_exam_01..03) for Tuples, Sets, and Dictionaries. Every verifier tests a
callable function across nominal and edge-case inputs, catches print()-only
None returns, guards against input mutation, and lets NotImplementedError
propagate cleanly for student starter stubs.
"""

from __future__ import annotations

import copy
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
        f"Calling {call_str} returned None instead of the expected structure.",
        details=[
            "Remember: print() only displays output on screen; it sends back None to the caller.",
            "Make sure your function ends with an explicit `return ...` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Always return the constructed dict, set, tuple, or number so downstream analysis can use it.",
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 8.1: Unpack Patient Vitals Tuple -> Structured Dict
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_todo_01", "To-Do 8.1", fn, "unpack_patient_vitals", 10)
    if bad is not None:
        return bad

    cases = [
        (
            ("UHC-0042", 38.4, 10.2),
            {"patient_id": "UHC-0042", "temp_c": 38.4, "hb_g_dl": 10.2},
        ),
        (
            ("UHC-0109", 36.8, 12.6),
            {"patient_id": "UHC-0109", "temp_c": 36.8, "hb_g_dl": 12.6},
        ),
        (
            ("DHK-0007", 39.7, 7.9),
            {"patient_id": "DHK-0007", "temp_c": 39.7, "hb_g_dl": 7.9},
        ),
    ]
    errors = []
    for rec, expected in cases:
        label = f"unpack_patient_vitals({rec!r})"
        try:
            got = fn(rec)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch08_todo_01",
                False,
                "failed",
                "To-Do 8.1 — Runtime Error Unpacking Tuple",
                f"{label} raised {type(exc).__name__}: {exc}",
                details=[
                    "Unpack the 3-tuple cleanly with `pid, temp_c, hb = record_tuple`.",
                ],
                points_earned=0,
                points_possible=10,
            )
        if got is None:
            return _none_return_trap("ch08_todo_01", "To-Do 8.1", label, 10)
        if not isinstance(got, dict):
            errors.append(f"{label}: expected a dict, got {type(got).__name__} ({got!r})")
            continue
        if set(got.keys()) != set(expected.keys()):
            errors.append(
                f"{label}: expected keys {sorted(expected.keys())}, got {sorted(got.keys())}"
            )
            continue
        if got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch08_todo_01",
            False,
            "failed",
            "To-Do 8.1 — Check Tuple Unpacking & Dictionary Keys",
            "Unpack `(pid, temp_c, hb)` and return `{'patient_id': pid, 'temp_c': temp_c, 'hb_g_dl': hb}`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch08_todo_01",
        True,
        "passed",
        "To-Do 8.1 Passed — Tuple Unpacking to Structured Record Verified!",
        "Cleanly unpacked all 3 clinical vital tuples into self-describing dictionaries.",
        points_earned=10,
        points_possible=10,
        tip="Tuple unpacking (`pid, temp_c, hb = record_tuple`) eliminates fragile magic indices like `record[2]`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_todo_01",
        chapter="ch08",
        title="To-Do 8.1: Unpack Patient Vitals Tuple",
        book_section="Section 8.3.1 (Tuples & Unpacking)",
        points=10,
        hints={
            1: "Unpack the 3-element tuple into `pid, temp_c, hb` and return a dictionary with keys `'patient_id'`, `'temp_c'`, and `'hb_g_dl'`.",
            2: "Algorithm:\n1. pid, temp_c, hb = record_tuple\n2. return {'patient_id': pid, 'temp_c': temp_c, 'hb_g_dl': hb}",
            3: "def unpack_patient_vitals(record_tuple):\n    pid, temp_c, hb = record_tuple\n    return {'patient_id': pid, 'temp_c': ___, 'hb_g_dl': ___}",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 8.2: Cohort Retention Metrics (Set Algebra)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_todo_02", "To-Do 8.2", fn, "cohort_retention_metrics", 10)
    if bad is not None:
        return bad

    cases = [
        (
            ["P01", "P02", "P03", "P02"],
            ["P02", "P03", "P04"],
            ({"P02", "P03"}, {"P01"}, {"P04"}),
            "nominal cohort with duplicate baseline ID",
        ),
        (
            ["A1", "A2"],
            ["A2", "A1"],
            ({"A1", "A2"}, set(), set()),
            "100% retention",
        ),
        (
            ["P01"],
            ["P09"],
            (set(), {"P01"}, {"P09"}),
            "complete turnover",
        ),
        (
            ["P01", "P02"],
            [],
            (set(), {"P01", "P02"}, set()),
            "empty follow-up wave",
        ),
    ]
    errors = []
    for base, follow, expected, desc in cases:
        label = f"cohort_retention_metrics({base!r}, {follow!r})"
        try:
            got = fn(list(base), list(follow))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch08_todo_02",
                False,
                "failed",
                "To-Do 8.2 — Runtime Error Computing Cohort Sets",
                f"{label} raised {type(exc).__name__}: {exc}",
                details=["Convert both inputs to sets first: `base = set(baseline_ids)` and `foll = set(followup_ids)`."],
                points_earned=0,
                points_possible=10,
            )
        if got is None:
            return _none_return_trap("ch08_todo_02", "To-Do 8.2", label, 10)
        if not isinstance(got, tuple) or len(got) != 3:
            errors.append(
                f"{desc}: expected a 3-tuple (retained_set, lost_set, new_set), got {type(got).__name__} ({got!r})"
            )
            continue
        if not all(isinstance(part, set) for part in got):
            types = [type(part).__name__ for part in got]
            errors.append(
                f"{desc}: all 3 elements in the returned tuple must be `set` objects, got types {types}"
            )
            continue
        if got != expected:
            extra = ""
            if (got[0], got[2], got[1]) == expected and expected[1] != expected[2]:
                extra = " (lost_set and new_set are swapped! lost = base - follow, new = follow - base)"
            errors.append(f"{desc}: expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch08_todo_02",
            False,
            "failed",
            "To-Do 8.2 — Check Set Intersection & Difference Order",
            "Return `(base & foll, base - foll, foll - base)` as a 3-tuple of sets.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch08_todo_02",
        True,
        "passed",
        "To-Do 8.2 Passed — Cohort Set Algebra Verified!",
        "Passed duplicate deduplication, full retention, complete turnover, and empty follow-up wave.",
        points_earned=10,
        points_possible=10,
        tip="Set algebra (`&`, `-`, `|`) runs in O(n) hash time and replaces slow nested list loops in longitudinal audits.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_todo_02",
        chapter="ch08",
        title="To-Do 8.2: Cohort Retention Set Metrics",
        book_section="Section 8.3.2 (Sets & Cohort Algebra)",
        points=10,
        hints={
            1: "Convert `baseline_ids` and `followup_ids` to sets, then compute retained (`&`), lost (`base - foll`), and new (`foll - base`).",
            2: "Algorithm:\n1. base = set(baseline_ids)\n2. foll = set(followup_ids)\n3. return (base & foll, base - foll, foll - base)",
            3: "def cohort_retention_metrics(baseline_ids, followup_ids):\n    base = set(baseline_ids)\n    foll = set(followup_ids)\n    return (base & foll, base - foll, ___ - ___)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 8.3: Safe Vaccine Stock Lookup via .get()
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_todo_03", "To-Do 8.3", fn, "safe_lookup_vaccine_stock", 10)
    if bad is not None:
        return bad

    stock_sample = {"BCG": 140, "OPV": 85, "Measles": 0}
    try:
        nominal = fn(dict(stock_sample), "BCG")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch08_todo_03",
            False,
            "failed",
            "To-Do 8.3 — Runtime Error on Nominal Lookup",
            f"safe_lookup_vaccine_stock({stock_sample!r}, 'BCG') raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if nominal is None:
        return _none_return_trap(
            "ch08_todo_03",
            "To-Do 8.3",
            f"safe_lookup_vaccine_stock({stock_sample!r}, 'BCG')",
            10,
        )

    cases = [
        ({"BCG": 140, "OPV": 85, "Measles": 0}, "BCG", 140),
        ({"BCG": 140, "OPV": 85, "Measles": 0}, "Measles", 0),
        ({"BCG": 140, "OPV": 85, "Measles": 0}, "Rota", 0),
        ({}, "BCG", 0),
    ]
    errors = []
    for stock, name, expected in cases:
        stock_copy = dict(stock)
        label = f"safe_lookup_vaccine_stock({stock!r}, {name!r})"
        try:
            got = fn(stock_copy, name)
        except NotImplementedError:
            raise
        except KeyError:
            errors.append(
                f"{label} raised KeyError — use `stock_dict.get(vaccine_name, 0)` instead of `stock_dict[vaccine_name]`!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if got is None:
            errors.append(
                f"{label} returned None — remember to pass default `0`: `stock_dict.get(vaccine_name, 0)`!"
            )
        elif got != expected or isinstance(got, bool):
            errors.append(f"{label}: expected {expected!r}, got {got!r}")
        elif stock_copy != stock:
            errors.append(
                f"{label} mutated stock_dict (e.g. via setdefault) — use read-only `.get(vaccine_name, 0)`!"
            )

    if errors:
        return CheckResult(
            "ch08_todo_03",
            False,
            "failed",
            "To-Do 8.3 — Use Safe Read-Only .get(key, 0)",
            "Return `stock_dict.get(vaccine_name, 0)` so missing vaccines return 0 without raising KeyError or mutating the dict.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch08_todo_03",
        True,
        "passed",
        "To-Do 8.3 Passed — Safe Dictionary .get() Lookup Verified!",
        "Handled present stock, zero stock, missing vaccines, and empty inventories without KeyError or mutation.",
        points_earned=10,
        points_possible=10,
        tip="Prefer `d.get(key, default)` whenever looking up optional fields from real-world health registries.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_todo_03",
        chapter="ch08",
        title="To-Do 8.3: Safe Vaccine Stock Lookup",
        book_section="Section 8.3.3 (Safe Lookups with .get())",
        points=10,
        hints={
            1: "Use the dictionary `.get()` method with a fallback value of `0`.",
            2: "Algorithm:\n1. return stock_dict.get(vaccine_name, 0)",
            3: "def safe_lookup_vaccine_stock(stock_dict, vaccine_name):\n    return stock_dict.get(vaccine_name, ___)",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 8.4: Frequency Counting of Triage Priority Levels
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_todo_04", "To-Do 8.4", fn, "count_triage_levels", 10)
    if bad is not None:
        return bad

    cases = [
        (
            ["Routine", "Urgent", "Routine", "Moderate", "Routine", "Urgent"],
            {"Routine": 3, "Urgent": 2, "Moderate": 1},
        ),
        (["Urgent"], {"Urgent": 1}),
        (["Moderate", "Moderate", "Routine"], {"Moderate": 2, "Routine": 1}),
        ([], {}),
    ]
    errors = []
    for priorities, expected in cases:
        label = f"count_triage_levels({priorities!r})"
        try:
            got = fn(list(priorities))
        except NotImplementedError:
            raise
        except KeyError as exc:
            errors.append(
                f"{label} raised KeyError: {exc} — use `counts[lvl] = counts.get(lvl, 0) + 1` to handle first-seen labels!"
            )
            continue
        except Exception as exc:
            return CheckResult(
                "ch08_todo_04",
                False,
                "failed",
                "To-Do 8.4 — Runtime Error Counting Triage Levels",
                f"{label} raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch08_todo_04", "To-Do 8.4", label, 10)
        if not isinstance(got, dict):
            errors.append(f"{label}: expected a dict, got {type(got).__name__} ({got!r})")
        elif got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch08_todo_04",
            False,
            "failed",
            "To-Do 8.4 — Check Frequency Counter Pattern",
            "Start with `counts = {}` and update `counts[level] = counts.get(level, 0) + 1` in a loop.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch08_todo_04",
        True,
        "passed",
        "To-Do 8.4 Passed — Triage Frequency Counter Verified!",
        "Accurately tallied multi-category queues, single-item lists, and empty inputs in O(n) time.",
        points_earned=10,
        points_possible=10,
        tip="`counts[k] = counts.get(k, 0) + 1` is one of the top three idioms in all of Python data science.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_todo_04",
        chapter="ch08",
        title="To-Do 8.4: Frequency Count of Triage Levels",
        book_section="Section 8.3.4 (Frequency Counting Idiom)",
        points=10,
        hints={
            1: "Initialize `counts = {}`, loop over each `level` in `priority_list`, and increment with `counts.get(level, 0) + 1`.",
            2: "Algorithm:\n1. counts = {}\n2. for level in priority_list:\n       counts[level] = counts.get(level, 0) + 1\n3. return counts",
            3: "def count_triage_levels(priority_list):\n    counts = {}\n    for level in priority_list:\n        counts[level] = counts.get(level, ___) + 1\n    return counts",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 8.5: Group Patient IDs by District
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_todo_05", "To-Do 8.5", fn, "group_patients_by_district", 10)
    if bad is not None:
        return bad

    cases = [
        (
            [
                {"patient_id": "UHC-01", "district": "Dhaka"},
                {"patient_id": "UHC-02", "district": "Sylhet"},
                {"patient_id": "UHC-03", "district": "Dhaka"},
                {"patient_id": "UHC-04", "district": "Khulna"},
                {"patient_id": "UHC-05", "district": "Sylhet"},
            ],
            {
                "Dhaka": ["UHC-01", "UHC-03"],
                "Sylhet": ["UHC-02", "UHC-05"],
                "Khulna": ["UHC-04"],
            },
            "multi-district intake list",
        ),
        (
            [
                {"patient_id": "P1", "district": "Rangpur"},
                {"patient_id": "P2", "district": "Rangpur"},
            ],
            {"Rangpur": ["P1", "P2"]},
            "single district",
        ),
        ([], {}, "empty record list"),
    ]
    errors = []
    for records, expected, desc in cases:
        rec_copy = copy.deepcopy(records)
        label = f"group_patients_by_district(<{desc}>)"
        try:
            got = fn(rec_copy)
        except NotImplementedError:
            raise
        except KeyError as exc:
            errors.append(
                f"{label} raised KeyError: {exc} — initialize an empty list with `grouped.setdefault(dist, []).append(pid)` or `if dist not in grouped: grouped[dist] = []`."
            )
            continue
        except Exception as exc:
            return CheckResult(
                "ch08_todo_05",
                False,
                "failed",
                "To-Do 8.5 — Runtime Error Grouping Patients",
                f"{label} raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch08_todo_05", "To-Do 8.5", label, 10)
        if not isinstance(got, dict):
            errors.append(f"{desc}: expected a dict, got {type(got).__name__} ({got!r})")
        elif got != expected:
            sample_vals = list(got.values())
            if sample_vals and not isinstance(sample_vals[0], list):
                errors.append(
                    f"{desc}: values must be lists of patient_ids (e.g. ['UHC-01', 'UHC-03']), not {type(sample_vals[0]).__name__}!"
                )
            else:
                errors.append(f"{desc}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch08_todo_05",
            False,
            "failed",
            "To-Do 8.5 — Check Dictionary-of-Lists Grouping Pattern",
            "Map each record['district'] to a list of record['patient_id'] values in encounter order.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch08_todo_05",
        True,
        "passed",
        "To-Do 8.5 Passed — Patient Grouping by District Verified!",
        "Cleanly grouped patients into district -> [patient_id, ...] lists across multi-district, single-district, and empty datasets.",
        points_earned=10,
        points_possible=10,
        tip="`grouped.setdefault(dist, []).append(pid)` is the foundation of every SQL/pandas `GROUP BY` operation!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_todo_05",
        chapter="ch08",
        title="To-Do 8.5: Group Patients by District",
        book_section="Section 8.3.4 (Grouping Records into Dict-of-Lists)",
        points=10,
        hints={
            1: "Create `grouped = {}`. For each `rec` in `records`, extract `dist = rec['district']` and append `rec['patient_id']` to `grouped[dist]`.",
            2: "Algorithm:\n1. grouped = {}\n2. for rec in records:\n       dist = rec['district']\n       grouped.setdefault(dist, []).append(rec['patient_id'])\n3. return grouped",
            3: "def group_patients_by_district(records):\n    grouped = {}\n    for rec in records:\n        grouped.setdefault(rec['district'], []).append(rec[___])\n    return grouped",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 8.6: District Incidence Map (Dict Comprehension + Guard)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_todo_06", "To-Do 8.6", fn, "district_incidence_map", 10)
    if bad is not None:
        return bad

    cases = [
        (
            {"Dhaka": 120, "Sylhet": 45, "Unknown": 10, "Ghost": 5, "BadPop": 8},
            {"Dhaka": 300000, "Sylhet": 150000, "Unknown": 0, "BadPop": -200},
            {"Dhaka": 40.0, "Sylhet": 30.0},
            "mixed valid, zero, negative, and missing populations",
        ),
        (
            {"Khulna": 0, "Rajshahi": 33},
            {"Khulna": 100000, "Rajshahi": 75000},
            {"Khulna": 0.0, "Rajshahi": 44.0},
            "zero cases with valid population",
        ),
        (
            {"A": 12},
            {"A": 0},
            {},
            "all invalid populations",
        ),
        (
            {},
            {"Dhaka": 300000},
            {},
            "empty cases dictionary",
        ),
    ]
    errors = []
    for cases_map, pop_map, expected, desc in cases:
        label = f"district_incidence_map(<{desc}>)"
        try:
            got = fn(dict(cases_map), dict(pop_map))
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                f"{desc}: raised ZeroDivisionError — filter out districts where `pop_by_district.get(dist, 0) <= 0`!"
            )
            continue
        except KeyError as exc:
            errors.append(
                f"{desc}: raised KeyError ({exc}) — use `pop_by_district.get(dist, 0) > 0` to skip missing districts!"
            )
            continue
        except Exception as exc:
            return CheckResult(
                "ch08_todo_06",
                False,
                "failed",
                "To-Do 8.6 — Runtime Error Computing Incidence Map",
                f"{label} raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch08_todo_06", "To-Do 8.6", label, 10)
        if not isinstance(got, dict):
            errors.append(f"{desc}: expected a dict, got {type(got).__name__} ({got!r})")
            continue
        if set(got.keys()) != set(expected.keys()):
            errors.append(
                f"{desc}: expected district keys {sorted(expected.keys())}, got {sorted(got.keys())}"
            )
            continue
        for dist, exp_rate in expected.items():
            val = got[dist]
            if not _is_number(val) or not math.isclose(val, exp_rate, abs_tol=1e-2):
                errors.append(f"{desc} ({dist}): expected {exp_rate}, got {val!r}")

    if errors:
        return CheckResult(
            "ch08_todo_06",
            False,
            "failed",
            "To-Do 8.6 — Check Filtered Dict Comprehension & Rate Formula",
            "Compute `round((cases / pop) * 100000, 2)` for every district in `cases_by_district` where `pop_by_district.get(dist, 0) > 0`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )
    return CheckResult(
        "ch08_todo_06",
        True,
        "passed",
        "To-Do 8.6 Passed — Guarded District Incidence Map Verified!",
        "Computed exact per-100k incidence rates while safely skipping zero, negative, and missing populations.",
        points_earned=10,
        points_possible=10,
        tip="Filtered dictionary comprehensions combine lookup, guard filtering, and transformation in one readable expression.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_todo_06",
        chapter="ch08",
        title="To-Do 8.6: District Incidence Map (Dict Comprehension)",
        book_section="Section 8.3.5 (Dictionary Comprehensions)",
        points=10,
        hints={
            1: "Iterate over `cases_by_district.items()`, keep only districts where `pop_by_district.get(dist, 0) > 0`, and map each to `round((cases / pop_by_district[dist]) * 100000, 2)`.",
            2: "Algorithm:\n1. return {\n       dist: round((cases / pop_by_district[dist]) * 100000, 2)\n       for dist, cases in cases_by_district.items()\n       if pop_by_district.get(dist, 0) > 0\n   }",
            3: "def district_incidence_map(cases_by_district, pop_by_district):\n    return {\n        dist: round((cases / pop_by_district[dist]) * 100000, 2)\n        for dist, cases in cases_by_district.items()\n        if pop_by_district.get(dist, ___) > 0\n    }",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 8.1 (30 pts): Survey ID Completion Auditor
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_exam_01", "Exam Task 8.1", fn, "audit_survey_ids", 30)
    if bad is not None:
        return bad

    nominal_enrolled = ["P01", "P02", "P03", "P04", "P02"]
    nominal_completed = ["P01", "P03", "P03", "P99"]
    try:
        nominal = fn(list(nominal_enrolled), list(nominal_completed))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch08_exam_01",
            False,
            "failed",
            "Exam Task 8.1 — Runtime Error on Nominal Cohort",
            f"audit_survey_ids(...) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if nominal is None:
        return _none_return_trap(
            "ch08_exam_01",
            "Exam Task 8.1",
            "audit_survey_ids(['P01', ...], ['P01', ...])",
            30,
        )

    cases = [
        (
            ["P01", "P02", "P03", "P04", "P02"],
            ["P01", "P03", "P03", "P99"],
            {
                "enrolled_unique": 4,
                "completed_valid": 2,
                "missing_ids": {"P02", "P04"},
                "completion_pct": 50.0,
            },
            "duplicates + stray unregistered ID P99",
        ),
        (
            ["A", "B", "C"],
            ["C", "A", "B"],
            {
                "enrolled_unique": 3,
                "completed_valid": 3,
                "missing_ids": set(),
                "completion_pct": 100.0,
            },
            "100% completion",
        ),
        (
            ["M1", "M2", "M3"],
            ["M1", "M2"],
            {
                "enrolled_unique": 3,
                "completed_valid": 2,
                "missing_ids": {"M3"},
                "completion_pct": 66.7,
            },
            "rounding to 1 decimal (66.7%)",
        ),
        (
            [],
            ["X1"],
            {
                "enrolled_unique": 0,
                "completed_valid": 0,
                "missing_ids": set(),
                "completion_pct": 0.0,
            },
            "empty enrolled cohort guard",
        ),
    ]

    expected_keys = {"enrolled_unique", "completed_valid", "missing_ids", "completion_pct"}
    errors = []
    for enrolled, completed, expected, desc in cases:
        try:
            got = fn(list(enrolled), list(completed))
        except NotImplementedError:
            raise
        except ZeroDivisionError:
            errors.append(
                f"{desc}: raised ZeroDivisionError — return 0.0 for completion_pct when enrolled_unique == 0!"
            )
            continue
        except Exception as exc:
            errors.append(f"{desc}: raised {type(exc).__name__}: {exc}")
            continue

        if not isinstance(got, dict):
            errors.append(f"{desc}: expected a dict, got {type(got).__name__} ({got!r})")
            continue
        if set(got.keys()) != expected_keys:
            errors.append(
                f"{desc}: expected keys {sorted(expected_keys)}, got {sorted(got.keys())}"
            )
            continue
        if got["enrolled_unique"] != expected["enrolled_unique"]:
            errors.append(
                f"{desc}: enrolled_unique expected {expected['enrolled_unique']}, got {got['enrolled_unique']!r}"
            )
        if got["completed_valid"] != expected["completed_valid"]:
            extra = ""
            if desc.startswith("duplicates") and got["completed_valid"] == 3:
                extra = " (stray ID 'P99' is NOT in enrolled_ids—use set intersection `enrolled_set & completed_set`!)"
            errors.append(
                f"{desc}: completed_valid expected {expected['completed_valid']}, got {got['completed_valid']!r}{extra}"
            )
        if not isinstance(got["missing_ids"], set):
            errors.append(
                f"{desc}: missing_ids must be a `set`, got {type(got['missing_ids']).__name__}"
            )
        elif got["missing_ids"] != expected["missing_ids"]:
            errors.append(
                f"{desc}: missing_ids expected {expected['missing_ids']!r}, got {got['missing_ids']!r}"
            )
        pct = got["completion_pct"]
        if not _is_number(pct) or not math.isclose(pct, expected["completion_pct"], abs_tol=1e-1):
            errors.append(
                f"{desc}: completion_pct expected {expected['completion_pct']}, got {pct!r}"
            )

    if errors:
        return CheckResult(
            "ch08_exam_01",
            False,
            "failed",
            "Exam Task 8.1 — Check Set Intersection, Missing Set & Empty Guard",
            "Deduplicate enrolled_ids, intersect with completed_ids to ignore stray IDs, compute missing_ids as a set, and guard against empty enrolled_ids.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch08_exam_01",
        True,
        "passed",
        "Exam Task 8.1 Passed (30/30 pts) — Survey ID Auditor Verified!",
        "Accurately deduplicated cohorts, filtered stray IDs via intersection, returned missing_ids set, and guarded empty cohorts.",
        points_earned=30,
        points_possible=30,
        tip="In real field surveys, tablets occasionally record mistyped or unenrolled IDs—set intersection prevents completion rates from exceeding 100%.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_exam_01",
        chapter="ch08",
        title="Exam 8.1: Survey ID Completion Auditor",
        book_section="Chapter 8 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Convert both inputs to sets. `valid = enrolled_set & completed_set`, `missing = enrolled_set - valid`, and compute `completion_pct` rounded to 1 decimal (`0.0` if `enrolled_unique == 0`).",
            2: "Algorithm:\n1. enrolled_set = set(enrolled_ids)\n2. valid = enrolled_set & set(completed_ids)\n3. missing = enrolled_set - valid\n4. pct = round((len(valid) / len(enrolled_set)) * 100, 1) if enrolled_set else 0.0\n5. return {'enrolled_unique': len(enrolled_set), 'completed_valid': len(valid), 'missing_ids': missing, 'completion_pct': pct}",
            3: "def audit_survey_ids(enrolled_ids, completed_ids):\n    enrolled_set = set(enrolled_ids)\n    valid = enrolled_set & set(completed_ids)\n    missing = enrolled_set - ___\n    pct = round((len(valid) / len(enrolled_set)) * 100, 1) if enrolled_set else 0.0\n    return {\n        'enrolled_unique': len(enrolled_set),\n        'completed_valid': len(valid),\n        'missing_ids': missing,\n        'completion_pct': pct,\n    }",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 8.2 (35 pts): District-Level Maternal Anemia Summarizer
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_exam_02", "Exam Task 8.2", fn, "summarize_district_anemia", 35)
    if bad is not None:
        return bad

    nominal_records = [
        {"district": "Dhaka", "hb_g_dl": 10.2},
        {"district": "Dhaka", "hb_g_dl": 11.0},
        {"district": "Dhaka", "hb_g_dl": None},
        {"district": "Dhaka", "hb_g_dl": 9.8},
        {"district": "Sylhet", "hb_g_dl": 12.1},
        {"district": "Sylhet", "hb_g_dl": 999},
        {"district": "Sylhet", "hb_g_dl": 10.5},
        {"district": "Khulna", "hb_g_dl": None},
        {"district": "Khulna", "hb_g_dl": 999},
    ]
    try:
        nominal = fn(copy.deepcopy(nominal_records))
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch08_exam_02",
            False,
            "failed",
            "Exam Task 8.2 — TypeError Comparing None with 11.0",
            f"summarize_district_anemia(...) raised TypeError: {exc}",
            details=["Skip `hb is None or hb == 999` BEFORE checking `hb < 11.0`!"],
            points_earned=0,
            points_possible=35,
        )
    except ZeroDivisionError:
        return CheckResult(
            "ch08_exam_02",
            False,
            "failed",
            "Exam Task 8.2 — ZeroDivisionError on All-Invalid District",
            "A district with only None/999 readings has 0 valid patients and should be omitted (not divided by 0).",
            points_earned=0,
            points_possible=35,
        )
    except Exception as exc:
        return CheckResult(
            "ch08_exam_02",
            False,
            "failed",
            "Exam Task 8.2 — Runtime Error on Nominal Records",
            f"summarize_district_anemia(...) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch08_exam_02", "Exam Task 8.2", "summarize_district_anemia(records)", 35
        )

    cases = [
        (
            nominal_records,
            {"Dhaka": 66.7, "Sylhet": 50.0},
            "multi-district with None, 999, boundary 11.0, and all-invalid Khulna",
        ),
        (
            [
                {"district": "Rajshahi", "hb_g_dl": 12.4},
                {"district": "Rajshahi", "hb_g_dl": 11.5},
                {"district": "Rangpur", "hb_g_dl": 9.4},
            ],
            {"Rajshahi": 0.0, "Rangpur": 100.0},
            "0% and 100% prevalence districts",
        ),
        ([], {}, "empty record list"),
    ]
    errors = []
    for records, expected, desc in cases:
        try:
            got = fn(copy.deepcopy(records))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{desc}: raised {type(exc).__name__}: {exc}")
            continue

        if not isinstance(got, dict):
            errors.append(f"{desc}: expected a dict, got {type(got).__name__} ({got!r})")
            continue
        if set(got.keys()) != set(expected.keys()):
            extra = ""
            if "Khulna" in got:
                extra = " (omit districts that have 0 valid readings after skipping None and 999!)"
            errors.append(
                f"{desc}: expected district keys {sorted(expected.keys())}, got {sorted(got.keys())}{extra}"
            )
            continue
        for dist, exp_pct in expected.items():
            val = got[dist]
            if not _is_number(val) or not math.isclose(val, exp_pct, abs_tol=1e-1):
                extra = ""
                if dist == "Dhaka" and _is_number(val) and math.isclose(val, 100.0, abs_tol=1e-1):
                    extra = " (boundary check: 11.0 g/dL is Normal—use strictly `hb < 11.0`, not `<=`!)"
                errors.append(f"{desc} ({dist}): expected {exp_pct}, got {val!r}{extra}")

    if errors:
        return CheckResult(
            "ch08_exam_02",
            False,
            "failed",
            "Exam Task 8.2 — Check Sentinel Skipping, Strict < 11.0 & Grouping",
            "Skip None/999 readings, group valid readings per district, and compute `round((anemic / valid) * 100, 1)` where strictly `< 11.0` is anemic.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch08_exam_02",
        True,
        "passed",
        "Exam Task 8.2 Passed (35/35 pts) — District Anemia Summarizer Verified!",
        "Cleaned None/999 sentinels, respected the strict < 11.0 WHO cutoff, omitted all-invalid districts, and rounded to 1 decimal.",
        points_earned=35,
        points_possible=35,
        tip="Grouping valid records into per-district buckets before summarizing mirrors pandas `groupby('district')` from first principles.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_exam_02",
        chapter="ch08",
        title="Exam 8.2: District Maternal Anemia Summarizer",
        book_section="Chapter 8 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Maintain `valid_counts = {}` and `anemic_counts = {}`. Skip records where `hb is None or hb == 999`. Then compute `round((anemic_counts.get(d, 0) / valid_counts[d]) * 100, 1)` for each district `d` in `valid_counts`.",
            2: "Algorithm:\n1. valid = {}; anemic = {}\n2. for rec in records:\n       hb = rec['hb_g_dl']\n       if hb is None or hb == 999: continue\n       d = rec['district']\n       valid[d] = valid.get(d, 0) + 1\n       if hb < 11.0: anemic[d] = anemic.get(d, 0) + 1\n3. return {d: round((anemic.get(d, 0) / n) * 100, 1) for d, n in valid.items()}",
            3: "def summarize_district_anemia(records):\n    valid = {}\n    anemic = {}\n    for rec in records:\n        hb = rec['hb_g_dl']\n        if hb is None or hb == 999:\n            continue\n        d = rec['district']\n        valid[d] = valid.get(d, 0) + 1\n        if hb < ___:\n            anemic[d] = anemic.get(d, 0) + 1\n    return {d: round((anemic.get(d, 0) / n) * 100, 1) for d, n in valid.items()}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 8.3 (35 pts): Non-Mutating Clinic Inventory Merger
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch08_exam_03", "Exam Task 8.3", fn, "merge_clinic_inventories", 35)
    if bad is not None:
        return bad

    a_nom = {"ORS": 200, "Zinc": 150, "BCG": 80}
    b_nom = {"Zinc": 50, "BCG": 20, "Syringe": 500}
    try:
        nominal = fn(dict(a_nom), dict(b_nom))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch08_exam_03",
            False,
            "failed",
            "Exam Task 8.3 — Runtime Error Merging Inventories",
            f"merge_clinic_inventories(...) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch08_exam_03", "Exam Task 8.3", "merge_clinic_inventories(warehouse_a, warehouse_b)", 35
        )

    cases = [
        (
            {"ORS": 200, "Zinc": 150, "BCG": 80},
            {"Zinc": 50, "BCG": 20, "Syringe": 500},
            {"ORS": 200, "Zinc": 200, "BCG": 100, "Syringe": 500},
            "overlapping + disjoint supplies",
        ),
        (
            {"Paracetamol": 300},
            {"Amoxicillin": 120},
            {"Paracetamol": 300, "Amoxicillin": 120},
            "completely disjoint warehouses",
        ),
        ({}, {"ORS": 75}, {"ORS": 75}, "empty warehouse_a"),
        ({"ORS": 75}, {}, {"ORS": 75}, "empty warehouse_b"),
        ({}, {}, {}, "both warehouses empty"),
    ]
    errors = []
    for raw_a, raw_b, expected, desc in cases:
        a_in = dict(raw_a)
        b_in = dict(raw_b)
        try:
            got = fn(a_in, b_in)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{desc}: raised {type(exc).__name__}: {exc}")
            continue

        if not isinstance(got, dict):
            errors.append(f"{desc}: expected a dict, got {type(got).__name__} ({got!r})")
            continue
        if a_in != raw_a:
            errors.append(
                f"{desc}: input `warehouse_a` was MUTATED (became {a_in!r})! Start with a fresh copy: `combined = dict(warehouse_a)`."
            )
            continue
        if b_in != raw_b:
            errors.append(
                f"{desc}: input `warehouse_b` was MUTATED (became {b_in!r})! Never modify input dicts in place."
            )
            continue
        if got is a_in or got is b_in:
            errors.append(
                f"{desc}: returned an alias to an input dict instead of a brand-new dictionary! Use `dict(warehouse_a)`."
            )
            continue
        if got != expected:
            extra = ""
            if desc == "overlapping + disjoint supplies" and got.get("Zinc") == 50:
                extra = " (shared keys were OVERWRITTEN instead of SUMMED—use `combined[item] = combined.get(item, 0) + qty`!)"
            errors.append(f"{desc}: expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch08_exam_03",
            False,
            "failed",
            "Exam Task 8.3 — Check Non-Mutating Copy & Shared-Key Summation",
            "Copy warehouse_a into a NEW dict (`combined = dict(warehouse_a)`), then add every `(item, qty)` from warehouse_b using `.get(item, 0) + qty`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch08_exam_03",
        True,
        "passed",
        "Exam Task 8.3 Passed (35/35 pts) — Non-Mutating Inventory Merger Verified!",
        "Summed overlapping supplies cleanly across all 5 test cases while leaving both input dictionaries completely untouched.",
        points_earned=35,
        points_possible=35,
        tip="Writing `combined = warehouse_a` only creates an alias! Copying with `dict(warehouse_a)` prevents silent mutation bugs.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch08_exam_03",
        chapter="ch08",
        title="Exam 8.3: Non-Mutating Clinic Inventory Merger",
        book_section="Chapter 8 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Start with `combined = dict(warehouse_a)` (a new copy!), then loop over `warehouse_b.items()` and add each quantity via `combined.get(item, 0) + qty`.",
            2: "Algorithm:\n1. combined = dict(warehouse_a)\n2. for item, qty in warehouse_b.items():\n       combined[item] = combined.get(item, 0) + qty\n3. return combined",
            3: "def merge_clinic_inventories(warehouse_a, warehouse_b):\n    combined = dict(warehouse_a)\n    for item, qty in warehouse_b.items():\n        combined[item] = combined.get(item, ___) + qty\n    return combined",
        },
        verifier=_verify_exam_03,
    )
)
