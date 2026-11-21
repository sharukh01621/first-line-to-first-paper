"""Chapter 7 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch07_todo_01..06) and Chapter 7 Coding Exam
(ch07_exam_01..03). Every verifier tests a callable function across nominal
and edge cases, diagnoses the print()-vs-return None trap, lets
NotImplementedError propagate cleanly, and explicitly asserts that caller
input lists are not silently mutated where required.
"""

from __future__ import annotations

import copy
import math
from typing import Any, List, Optional

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
    extra_hint: str = "",
) -> CheckResult:
    details = [
        "Did you use print() instead of return, or return an in-place list method like .append() or .sort()?",
        "In Python, list.append(), list.sort(), and list.reverse() mutate in place and return None!",
        "Make sure your function returns the new list explicitly with `return ...`.",
    ]
    if extra_hint:
        details.append(extra_hint)
    return CheckResult(
        ex_id,
        False,
        "failed",
        f"{title} — Returned None (print() or In-Place Method Trap!)",
        f"Calling {call_str} returned None instead of the expected result.",
        details=details,
        points_earned=0,
        points_possible=points_possible,
        tip="Never write `return my_list.sort()` or `return my_list.append(x)`—those methods return None!",
    )


def _mutation_trap(
    ex_id: str,
    title: str,
    call_str: str,
    before: Any,
    after: Any,
    points_possible: int = 10,
) -> CheckResult:
    return CheckResult(
        ex_id,
        False,
        "failed",
        f"{title} — Caller Input List Was Mutated (Aliasing Trap!)",
        f"Calling {call_str} modified the caller's original list in place!",
        details=[
            f"Original input before call: {before!r}",
            f"Original input after call:  {after!r}",
            "Work on a copy (e.g., `.copy()`, `sorted(...)`, slicing, or a list comprehension) so the caller's data stays intact.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="In scientific pipelines, silently mutating a caller's raw data list corrupts downstream tables and breaks reproducibility.",
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 7.1: Extract Recent Surveillance Window (Slicing)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_todo_01", "To-Do 7.1", fn, "extract_recent_window", 10)
    if bad is not None:
        return bad

    nominal_in = [12, 18, 25, 31, 40, 28]
    nominal_snap = list(nominal_in)
    try:
        nominal = fn(nominal_in, 3)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch07_todo_01",
            False,
            "failed",
            "To-Do 7.1 — Runtime Error on Nominal Input",
            f"extract_recent_window({nominal_snap}, 3) raised {type(exc).__name__}: {exc}",
            details=["Make sure extract_recent_window accepts two parameters: (weekly_cases, k)."],
        )

    if nominal is None:
        return _none_return_trap(
            "ch07_todo_01", "To-Do 7.1", f"extract_recent_window({nominal_snap}, 3)", 10
        )
    if nominal_in != nominal_snap:
        return _mutation_trap(
            "ch07_todo_01", "To-Do 7.1", f"extract_recent_window({nominal_snap}, 3)",
            nominal_snap, nominal_in, 10,
        )

    cases = [
        ([12, 18, 25, 31, 40, 28], 3, [31, 40, 28]),
        ([12, 18, 25], 1, [25]),
        ([12, 18], 5, [12, 18]),
        ([12, 18, 25], 0, []),
        ([12, 18, 25], -2, []),
        ([], 3, []),
    ]
    errors: List[str] = []
    for data, k, expected in cases:
        buf = list(data)
        snap = list(data)
        label = f"extract_recent_window({snap}, {k})"
        try:
            got = fn(buf, k)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if buf != snap:
            return _mutation_trap("ch07_todo_01", "To-Do 7.1", label, snap, buf, 10)
        if not isinstance(got, list):
            errors.append(f"{label}: expected a list, got {type(got).__name__} ({got!r})")
            continue
        if got != expected:
            extra = ""
            if k == 0 and got == snap:
                extra = " (Python's `lst[-0:]` is `lst[0:]`, which returns the whole list! Guard `if k <= 0: return []`.)"
            errors.append(f"{label}: expected {expected!r}, got {got!r}{extra}")
        elif len(buf) > 0 and got is buf:
            errors.append(f"{label}: returned the exact same list object (alias) instead of a new sliced copy!")

    if errors:
        return CheckResult(
            "ch07_todo_01",
            False,
            "failed",
            "To-Do 7.1 — Check Negative Slicing & k <= 0 Guard",
            "Return the last k elements as a new list, and return [] whenever k <= 0.",
            details=errors,
        )
    return CheckResult(
        "ch07_todo_01",
        True,
        "passed",
        "To-Do 7.1 Passed — Recent Window Slicer Verified!",
        "Passed nominal slicing, k > len(list), empty list, non-aliasing copy, and the sneaky k == 0 trap!",
        tip="Remember the `-0` slicing gotcha: in Python `-0 == 0`, so `data[-0:]` returns the entire list unless guarded with `if k <= 0: return []`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_todo_01",
        chapter="ch07",
        title="To-Do 7.1: Extract Recent Surveillance Window",
        book_section="Section 7.3.1 (Indexing & Slicing)",
        points=10,
        hints={
            1: "Guard against `k <= 0` by returning `[]`; otherwise use negative slicing `weekly_cases[-k:]`.",
            2: "Algorithm:\n1. if k <= 0: return []\n2. return weekly_cases[-k:]",
            3: "def extract_recent_window(weekly_cases, k):\n    if k <= 0:\n        return []\n    return weekly_cases[___:]",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 7.2: Non-Mutating Safe Append
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_todo_02", "To-Do 7.2", fn, "append_reading_safely", 10)
    if bad is not None:
        return bad

    orig = [118, 122, 125]
    snap = list(orig)
    try:
        nominal = fn(orig, 130)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch07_todo_02",
            False,
            "failed",
            "To-Do 7.2 — Runtime Error on Nominal Input",
            f"append_reading_safely({snap}, 130) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch07_todo_02",
            "To-Do 7.2",
            f"append_reading_safely({snap}, 130)",
            10,
            extra_hint="Writing `return updated.append(new_val)` returns None! Call `updated.append(new_val)` on one line, then `return updated` on the next.",
        )
    if orig != snap:
        return _mutation_trap(
            "ch07_todo_02", "To-Do 7.2", f"append_reading_safely({snap}, 130)", snap, orig, 10
        )

    cases = [
        ([118, 122, 125], 130, [118, 122, 125, 130]),
        ([], 11.4, [11.4]),
        ([10.5, 11.2], 0, [10.5, 11.2, 0]),
    ]
    errors: List[str] = []
    for data, new_val, expected in cases:
        buf = list(data)
        buf_snap = list(data)
        label = f"append_reading_safely({buf_snap}, {new_val!r})"
        try:
            got = fn(buf, new_val)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if buf != buf_snap:
            return _mutation_trap("ch07_todo_02", "To-Do 7.2", label, buf_snap, buf, 10)
        if got is buf:
            errors.append(f"{label}: returned the caller's list object instead of a fresh copy!")
        elif got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch07_todo_02",
            False,
            "failed",
            "To-Do 7.2 — Check Non-Mutating Copy & Append",
            "Copy `readings` first (e.g., `updated = readings.copy()`), append `new_val`, and return `updated`.",
            details=errors,
        )
    return CheckResult(
        "ch07_todo_02",
        True,
        "passed",
        "To-Do 7.2 Passed — Non-Mutating Safe Append Verified!",
        "Appended cleanly across nominal, empty, and zero-value cases while leaving the caller's original list completely untouched.",
        tip="Creating a copy before mutating (`readings.copy()` or `readings + [new_val]`) keeps your functions pure and side-effect free.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_todo_02",
        chapter="ch07",
        title="To-Do 7.2: Non-Mutating Safe Append",
        book_section="Section 7.3.2 & Section 7.3.3 (Mutation & Aliasing)",
        points=10,
        hints={
            1: "Never call `readings.append(new_val)` directly on the parameter! Make a shallow copy first with `readings.copy()` (or use `readings + [new_val]`).",
            2: "Algorithm:\n1. updated = readings.copy()\n2. updated.append(new_val)\n3. return updated",
            3: "def append_reading_safely(readings, new_val):\n    updated = readings.___()\n    updated.append(new_val)\n    return updated",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 7.3: Filter Valid Hemoglobin Readings (Skip None & 999)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_todo_03", "To-Do 7.3", fn, "filter_valid_hb", 10)
    if bad is not None:
        return bad

    orig = [10.4, None, 11.8, 999, 12.1, None]
    snap = list(orig)
    try:
        nominal = fn(orig)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch07_todo_03",
            False,
            "failed",
            "To-Do 7.3 — Runtime Error on Nominal List",
            f"filter_valid_hb({snap}) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap("ch07_todo_03", "To-Do 7.3", f"filter_valid_hb({snap})", 10)
    if orig != snap:
        return _mutation_trap("ch07_todo_03", "To-Do 7.3", f"filter_valid_hb({snap})", snap, orig, 10)

    cases = [
        ([10.4, None, 11.8, 999, 12.1, None], [10.4, 11.8, 12.1]),
        ([11.0, 12.5], [11.0, 12.5]),
        ([None, 999, None], []),
        ([], []),
        ([9.8, 999.0, 10.6], [9.8, 10.6]),
    ]
    errors: List[str] = []
    for data, expected in cases:
        buf = list(data)
        buf_snap = list(data)
        label = f"filter_valid_hb({buf_snap})"
        try:
            got = fn(buf)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if buf != buf_snap:
            return _mutation_trap("ch07_todo_03", "To-Do 7.3", label, buf_snap, buf, 10)
        if not isinstance(got, list):
            errors.append(f"{label}: expected a list, got {type(got).__name__} ({got!r})")
            continue
        if got != expected:
            errors.append(f"{label}: expected {expected!r}, got {got!r}")
        elif len(buf) > 0 and got is buf:
            errors.append(f"{label}: returned the caller's list object instead of a new filtered list!")

    if errors:
        return CheckResult(
            "ch07_todo_03",
            False,
            "failed",
            "To-Do 7.3 — Check Sentinel Filtering (None & 999)",
            "Return a new list keeping only elements where `val is not None and val != 999`.",
            details=errors,
        )
    return CheckResult(
        "ch07_todo_03",
        True,
        "passed",
        "To-Do 7.3 Passed — Hemoglobin Sentinel Filter Verified!",
        "Cleanly filtered out None and 999 without mutating the caller's raw list.",
        tip="A filtering list comprehension `[v for v in raw if v is not None and v != 999]` is concise, readable, and always creates a fresh list.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_todo_03",
        chapter="ch07",
        title="To-Do 7.3: Filter Valid Hemoglobin Readings",
        book_section="Section 7.3.4 (Filtering with List Comprehensions)",
        points=10,
        hints={
            1: "Build a new list (with a comprehension or loop) keeping only values where `val is not None and val != 999`.",
            2: "Algorithm:\nreturn [val for val in raw_hb_list if val is not None and val != 999]",
            3: "def filter_valid_hb(raw_hb_list):\n    return [val for val in raw_hb_list if val is not ___ and val != ___]",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 7.4: Convert Heights from cm to m (Comprehension Transform)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_todo_04", "To-Do 7.4", fn, "convert_heights_cm_to_m", 10)
    if bad is not None:
        return bad

    orig = [168, 156, 172.4, 160]
    snap = list(orig)
    try:
        nominal = fn(orig)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch07_todo_04",
            False,
            "failed",
            "To-Do 7.4 — Runtime Error on Nominal Input",
            f"convert_heights_cm_to_m({snap}) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch07_todo_04", "To-Do 7.4", f"convert_heights_cm_to_m({snap})", 10
        )
    if orig != snap:
        return _mutation_trap(
            "ch07_todo_04", "To-Do 7.4", f"convert_heights_cm_to_m({snap})", snap, orig, 10
        )

    cases = [
        ([168, 156, 172.4, 160], [1.68, 1.56, 1.72, 1.6]),
        ([150, 182.6, 164.3], [1.5, 1.83, 1.64]),
        ([170], [1.7]),
        ([], []),
    ]
    errors: List[str] = []
    for data, expected in cases:
        buf = list(data)
        buf_snap = list(data)
        label = f"convert_heights_cm_to_m({buf_snap})"
        try:
            got = fn(buf)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if buf != buf_snap:
            return _mutation_trap("ch07_todo_04", "To-Do 7.4", label, buf_snap, buf, 10)
        if not isinstance(got, list) or len(got) != len(expected):
            errors.append(f"{label}: expected list of length {len(expected)} ({expected!r}), got {got!r}")
            continue
        for idx, (g_val, e_val) in enumerate(zip(got, expected)):
            if not _is_number(g_val) or not math.isclose(g_val, e_val, abs_tol=1e-6):
                errors.append(
                    f"{label}[{idx}]: expected {e_val} (rounded to 2 decimals), got {g_val!r}"
                )
                break

    if errors:
        return CheckResult(
            "ch07_todo_04",
            False,
            "failed",
            "To-Do 7.4 — Check Unit Conversion (cm -> m) & Rounding",
            "Divide each height by 100 and round to 2 decimal places: `[round(h / 100, 2) for h in heights_cm]`.",
            details=errors,
        )
    return CheckResult(
        "ch07_todo_04",
        True,
        "passed",
        "To-Do 7.4 Passed — Vectorized Height Conversion Verified!",
        "Converted all centimeter lists cleanly to meters (2 decimals) without mutating the caller's list.",
        tip="List comprehensions replace 4-line append loops with a single expressive transformation that never mutates the source list.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_todo_04",
        chapter="ch07",
        title="To-Do 7.4: Convert Heights (cm -> m) with Comprehension",
        book_section="Section 7.3.4 (Transforming Lists with Comprehensions)",
        points=10,
        hints={
            1: "Use a list comprehension to compute `round(h / 100, 2)` for every `h` in `heights_cm`.",
            2: "Algorithm:\nreturn [round(h / 100, 2) for h in heights_cm]",
            3: "def convert_heights_cm_to_m(heights_cm):\n    return [round(h / ___, 2) for h in heights_cm]",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 7.5: Top-k Outbreak Weeks (sorted() Non-Mutating)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_todo_05", "To-Do 7.5", fn, "top_k_outbreak_weeks", 10)
    if bad is not None:
        return bad

    orig = [14, 42, 19, 68, 35, 51]
    snap = list(orig)
    try:
        nominal = fn(orig, 3)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch07_todo_05",
            False,
            "failed",
            "To-Do 7.5 — Runtime Error on Nominal Input",
            f"top_k_outbreak_weeks({snap}, 3) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch07_todo_05",
            "To-Do 7.5",
            f"top_k_outbreak_weeks({snap}, 3)",
            10,
            extra_hint="Use `sorted(weekly_cases, reverse=True)[:k]` instead of `weekly_cases.sort()` (which returns None and mutates in place!).",
        )
    if orig != snap:
        return _mutation_trap(
            "ch07_todo_05", "To-Do 7.5", f"top_k_outbreak_weeks({snap}, 3)", snap, orig, 10
        )

    cases = [
        ([14, 42, 19, 68, 35, 51], 3, [68, 51, 42]),
        ([20, 50, 50, 10], 2, [50, 50]),
        ([10, 30, 20], 5, [30, 20, 10]),
        ([10, 30, 20], 0, []),
        ([10, 30, 20], -1, []),
        ([], 3, []),
    ]
    errors: List[str] = []
    for data, k, expected in cases:
        buf = list(data)
        buf_snap = list(data)
        label = f"top_k_outbreak_weeks({buf_snap}, {k})"
        try:
            got = fn(buf, k)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if buf != buf_snap:
            return _mutation_trap("ch07_todo_05", "To-Do 7.5", label, buf_snap, buf, 10)
        if not isinstance(got, list):
            errors.append(f"{label}: expected a list, got {type(got).__name__} ({got!r})")
            continue
        if got != expected:
            extra = ""
            if k > 0 and got == sorted(buf_snap)[:k]:
                extra = " (returned lowest k instead of highest k — add `reverse=True`!)"
            errors.append(f"{label}: expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch07_todo_05",
            False,
            "failed",
            "To-Do 7.5 — Check Descending Sort, Slicing & Non-Mutation",
            "Return `[]` if `k <= 0`; otherwise return `sorted(weekly_cases, reverse=True)[:k]`.",
            details=errors,
        )
    return CheckResult(
        "ch07_todo_05",
        True,
        "passed",
        "To-Do 7.5 Passed — Non-Mutating Top-k Outbreak Finder Verified!",
        "Returned exact descending top-k peaks across nominal, tie, k > len, and k <= 0 cases while preserving original weekly chronological order.",
        tip="Prefer `sorted(data, reverse=True)` over `data.sort()` whenever the original chronological order of a surveillance series matters!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_todo_05",
        chapter="ch07",
        title="To-Do 7.5: Top-k Outbreak Weeks (Non-Mutating)",
        book_section="Section 7.3.2 (list.sort() vs. sorted())",
        points=10,
        hints={
            1: "Use Python's built-in `sorted(weekly_cases, reverse=True)` to produce a descending copy without changing `weekly_cases`, then slice `[:k]` (guarding `k <= 0`).",
            2: "Algorithm:\n1. if k <= 0: return []\n2. return sorted(weekly_cases, reverse=True)[:k]",
            3: "def top_k_outbreak_weeks(weekly_cases, k):\n    if k <= 0:\n        return []\n    return sorted(weekly_cases, reverse=___)[:k]",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 7.6: Deep Clone Ward Matrix (Shallow vs. Deep Copy Trap)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_todo_06", "To-Do 7.6", fn, "clone_ward_matrix", 10)
    if bad is not None:
        return bad

    orig = [[12, 14], [18, 20], [9, 11]]
    snap = copy.deepcopy(orig)
    try:
        cloned = fn(orig)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch07_todo_06",
            False,
            "failed",
            "To-Do 7.6 — Runtime Error on Nested Ward Matrix",
            f"clone_ward_matrix({snap}) raised {type(exc).__name__}: {exc}",
        )

    if cloned is None:
        return _none_return_trap("ch07_todo_06", "To-Do 7.6", f"clone_ward_matrix({snap})", 10)
    if not isinstance(cloned, list) or cloned != snap:
        return CheckResult(
            "ch07_todo_06",
            False,
            "failed",
            "To-Do 7.6 — Cloned Values Do Not Match Original",
            f"Expected {snap!r}, got {cloned!r}.",
            details=["Make sure your clone preserves every ward row and its values."],
        )
    if cloned is orig:
        return CheckResult(
            "ch07_todo_06",
            False,
            "failed",
            "To-Do 7.6 — Direct Alias Returned (No Copy Made!)",
            "clone_ward_matrix(wards) returned the exact same list object (`cloned is orig` is True).",
            details=[
                "Returning `wards` directly creates an alias—any edit to the clone corrupts the original!",
                "Use `copy.deepcopy(wards)` to clone both the outer list and every inner ward list.",
            ],
        )

    # Probe shallow-copy trap by mutating inner row of clone
    cloned[0][0] = 999
    cloned[1].append(777)
    if orig != snap:
        return CheckResult(
            "ch07_todo_06",
            False,
            "failed",
            "To-Do 7.6 — Shallow Copy Trap Caught! Inner Ward List Was Shared",
            "Mutating `cloned[0][0] = 999` silently changed the original `wards` matrix!",
            details=[
                f"Original matrix became: {orig!r} (should remain {snap!r}).",
                "`.copy()` and `wards[:]` only make a SHALLOW copy—the outer list is new, but inner lists still point to the same memory!",
                "Import `copy` and return `copy.deepcopy(wards)` so nested lists are completely independent.",
            ],
            tip="Shallow copy copies the outer container; `copy.deepcopy()` recursively clones nested containers too.",
        )

    # Extra edge cases: ragged rows & empty matrix
    for extra in ([[10], [20, 30, 40], []], []):
        extra_snap = copy.deepcopy(extra)
        try:
            got_extra = fn(extra)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch07_todo_06",
                False,
                "failed",
                "To-Do 7.6 — Failed on Edge Case",
                f"clone_ward_matrix({extra_snap}) raised {type(exc).__name__}: {exc}",
            )
        if got_extra != extra_snap or got_extra is extra:
            return CheckResult(
                "ch07_todo_06",
                False,
                "failed",
                "To-Do 7.6 — Edge Case Copy Failed",
                f"clone_ward_matrix({extra_snap}) must return a distinct equal copy, got {got_extra!r}.",
            )
        if extra:
            got_extra[0].append(-1)
            if extra != extra_snap:
                return CheckResult(
                    "ch07_todo_06",
                    False,
                    "failed",
                    "To-Do 7.6 — Shallow Copy Trap on Ragged Matrix",
                    "Mutating an inner row of the clone modified the original matrix.",
                )

    return CheckResult(
        "ch07_todo_06",
        True,
        "passed",
        "To-Do 7.6 Passed — Deep Copy Isolation Verified!",
        "Passed nested mutation stress test, ragged rows, and empty matrix checks with full memory isolation.",
        tip="Whenever a list contains nested lists (or dicts), use `copy.deepcopy()` before running simulations or sensitivity edits.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_todo_06",
        chapter="ch07",
        title="To-Do 7.6: Deep Clone Ward Matrix",
        book_section="Section 7.3.3 (Aliasing vs. Shallow Copy vs. Deep Copy)",
        points=10,
        hints={
            1: "A shallow `.copy()` still shares the inner ward lists! Use Python's standard-library `copy.deepcopy(wards)`.",
            2: "Algorithm:\n1. import copy\n2. return copy.deepcopy(wards)",
            3: "import copy\n\ndef clone_ward_matrix(wards):\n    return copy.___(wards)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 7.1 (30 pts): Clean & Sort Systolic Blood Pressure
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_exam_01", "Exam Task 7.1", fn, "clean_and_sort_sbp", 30)
    if bad is not None:
        return bad

    orig = [142, None, 118, 999, 0, -15, 130, 118]
    snap = list(orig)
    try:
        nominal = fn(orig)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch07_exam_01",
            False,
            "failed",
            "Exam Task 7.1 — TypeError Filtering or Sorting None",
            f"clean_and_sort_sbp({snap}) raised TypeError: {exc}",
            details=[
                "Check `val is not None` BEFORE comparing `val > 0` or sorting!",
                "Sorting a list that still contains `None` raises TypeError.",
            ],
            points_earned=0,
            points_possible=30,
        )
    except Exception as exc:
        return CheckResult(
            "ch07_exam_01",
            False,
            "failed",
            "Exam Task 7.1 — Runtime Error on Nominal Input",
            f"clean_and_sort_sbp({snap}) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if nominal is None:
        return _none_return_trap(
            "ch07_exam_01",
            "Exam Task 7.1",
            f"clean_and_sort_sbp({snap})",
            30,
            extra_hint="Remember: `valid.sort()` returns None! Use `return sorted(valid)`.",
        )
    if orig != snap:
        return _mutation_trap(
            "ch07_exam_01", "Exam Task 7.1", f"clean_and_sort_sbp({snap})", snap, orig, 30
        )

    tests = [
        ([142, None, 118, 999, 0, -15, 130, 118], [118, 118, 130, 142]),
        ([126.5, 999, 112.0, -5.0, None, 138.0], [112.0, 126.5, 138.0]),
        ([110, 120, 130], [110, 120, 130]),
        ([None, 999, 0, -20], []),
        ([], []),
    ]
    errors: List[str] = []
    for data, expected in tests:
        buf = list(data)
        buf_snap = list(data)
        label = f"clean_and_sort_sbp({buf_snap})"
        try:
            got = fn(buf)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if buf != buf_snap:
            return _mutation_trap("ch07_exam_01", "Exam Task 7.1", label, buf_snap, buf, 30)
        if not isinstance(got, list):
            errors.append(f"{label}: expected a list, got {type(got).__name__} ({got!r})")
            continue
        if got != expected:
            extra = ""
            if 0 in got:
                extra = " (0 is physiologically impossible for SBP — exclude all values <= 0!)"
            elif 999 in got:
                extra = " (sentinel 999 was not excluded!)"
            errors.append(f"{label}: expected {expected!r}, got {got!r}{extra}")
        elif len(buf) > 0 and got is buf:
            errors.append(f"{label}: returned the caller's original list object instead of a new list!")

    if errors:
        return CheckResult(
            "ch07_exam_01",
            False,
            "failed",
            "Exam Task 7.1 — Check Filtering Rules & Ascending Sort",
            "Skip None, 999, and values <= 0; return a new ascending-sorted list without mutating `readings`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )
    return CheckResult(
        "ch07_exam_01",
        True,
        "passed",
        "Exam Task 7.1 Passed (30/30 pts) — Clean & Sort SBP Verified!",
        "Filtered out None, 999, zero, and negative readings, returned an ascending sorted copy, and left caller inputs untouched.",
        points_earned=30,
        points_possible=30,
        tip="Always filter `None` BEFORE numeric comparisons or `sorted()` so Python never hits a `NoneType` comparison error.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_exam_01",
        chapter="ch07",
        title="Exam 7.1: Clean & Sort Systolic Blood Pressure",
        book_section="Chapter 7 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Filter into a new list keeping only `val is not None and val != 999 and val > 0`, then return `sorted(valid)`.",
            2: "Algorithm:\n1. valid = [v for v in readings if v is not None and v != 999 and v > 0]\n2. return sorted(valid)",
            3: "def clean_and_sort_sbp(readings):\n    valid = [v for v in readings if v is not None and v != ___ and v > 0]\n    return sorted(valid)",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 7.2 (35 pts): Rolling 3-Day Outbreak Totals
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_exam_02", "Exam Task 7.2", fn, "rolling_3day_totals", 35)
    if bad is not None:
        return bad

    orig = [10, 15, 20, 25, 30]
    snap = list(orig)
    try:
        nominal = fn(orig)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch07_exam_02",
            False,
            "failed",
            "Exam Task 7.2 — Runtime Error on Nominal Input",
            f"rolling_3day_totals({snap}) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap("ch07_exam_02", "Exam Task 7.2", f"rolling_3day_totals({snap})", 35)
    if orig != snap:
        return _mutation_trap(
            "ch07_exam_02", "Exam Task 7.2", f"rolling_3day_totals({snap})", snap, orig, 35
        )

    tests = [
        ([10, 15, 20, 25, 30], [45, 60, 75]),
        ([4, 7, 9], [20]),
        ([0, 5, 0, 12, 3, 0], [5, 17, 15, 15]),
        ([10, 20], []),
        ([10], []),
        ([], []),
    ]
    errors: List[str] = []
    for data, expected in tests:
        buf = list(data)
        buf_snap = list(data)
        label = f"rolling_3day_totals({buf_snap})"
        try:
            got = fn(buf)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if buf != buf_snap:
            return _mutation_trap("ch07_exam_02", "Exam Task 7.2", label, buf_snap, buf, 35)
        if not isinstance(got, list):
            errors.append(f"{label}: expected a list, got {type(got).__name__} ({got!r})")
            continue
        if got != expected:
            extra = ""
            if len(buf_snap) >= 3 and len(got) == len(expected) - 1:
                extra = " (off-by-one: use `range(len(daily_cases) - 2)` so the final 3-day window is included!)"
            elif len(buf_snap) >= 3 and len(got) > len(expected):
                extra = " (too many windows: stop at `len(daily_cases) - 2` so partial 1-day or 2-day tails are not summed!)"
            errors.append(f"{label}: expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch07_exam_02",
            False,
            "failed",
            "Exam Task 7.2 — Check Sliding Window Bounds & Short-List Guard",
            "Return [] when `len(daily_cases) < 3`; otherwise return `[sum(daily_cases[i:i+3]) for i in range(len(daily_cases) - 2)]`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch07_exam_02",
        True,
        "passed",
        "Exam Task 7.2 Passed (35/35 pts) — Rolling 3-Day Window Sums Verified!",
        "Computed exact sliding 3-day totals across nominal, length-3, zero-count, and short (< 3) series.",
        points_earned=35,
        points_possible=35,
        tip="For any window width W on a sequence of length N, there are exactly `N - W + 1` full windows—here `N - 3 + 1 = N - 2`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_exam_02",
        chapter="ch07",
        title="Exam 7.2: Rolling 3-Day Outbreak Totals",
        book_section="Chapter 7 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "If `len(daily_cases) < 3`, return `[]`. Otherwise loop `i` from `0` up to `len(daily_cases) - 3` (inclusive) and sum `daily_cases[i : i + 3]`.",
            2: "Algorithm:\n1. if len(daily_cases) < 3: return []\n2. return [sum(daily_cases[i : i + 3]) for i in range(len(daily_cases) - 2)]",
            3: "def rolling_3day_totals(daily_cases):\n    if len(daily_cases) < 3:\n        return []\n    return [sum(daily_cases[i : i + 3]) for i in range(len(daily_cases) - ___)]",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 7.3 (35 pts): Flag & Adjust Medication Doses (Non-Mutating)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch07_exam_03", "Exam Task 7.3", fn, "flag_and_adjust_doses", 35)
    if bad is not None:
        return bad

    orig = [250, 500, 650, 400, 800]
    snap = list(orig)
    try:
        nominal = fn(orig, 500)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch07_exam_03",
            False,
            "failed",
            "Exam Task 7.3 — Runtime Error on Nominal Input",
            f"flag_and_adjust_doses({snap}, 500) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch07_exam_03", "Exam Task 7.3", f"flag_and_adjust_doses({snap}, 500)", 35
        )
    if orig != snap:
        return _mutation_trap(
            "ch07_exam_03", "Exam Task 7.3", f"flag_and_adjust_doses({snap}, 500)", snap, orig, 35
        )

    tests = [
        ([250, 500, 650, 400, 800], 500, ([250, 500, 500, 400, 500], 2)),
        ([100, 200, 300], 500, ([100, 200, 300], 0)),
        ([600, 750, 900], 500, ([500, 500, 500], 3)),
        ([12.5, 25.0, 30.5], 25.0, ([12.5, 25.0, 25.0], 1)),
        ([], 500, ([], 0)),
    ]
    errors: List[str] = []
    for data, cap, (exp_list, exp_count) in tests:
        buf = list(data)
        buf_snap = list(data)
        label = f"flag_and_adjust_doses({buf_snap}, {cap})"
        try:
            got = fn(buf, cap)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if buf != buf_snap:
            return _mutation_trap("ch07_exam_03", "Exam Task 7.3", label, buf_snap, buf, 35)
        if not isinstance(got, tuple) or len(got) != 2:
            errors.append(
                f"{label}: expected a 2-tuple (adjusted_list, capped_count), got {type(got).__name__} ({got!r})"
            )
            continue
        got_list, got_count = got
        if not isinstance(got_list, list) or type(got_count) is not int:
            errors.append(
                f"{label}: expected (list, int), got ({type(got_list).__name__}, {type(got_count).__name__})"
            )
            continue
        if len(buf) > 0 and got_list is buf:
            errors.append(f"{label}: returned the caller's list object as adjusted_list (must return a new list!).")
            continue
        if got_list != exp_list or got_count != exp_count:
            extra = ""
            if got_list == exp_list and got_count == exp_count + 1 and cap in buf_snap:
                extra = " (doses equal to `cap_mg` are safe—only count strictly `dose > cap_mg`!)"
            errors.append(
                f"{label}: expected ({exp_list!r}, {exp_count}), got ({got_list!r}, {got_count}){extra}"
            )

    if errors:
        return CheckResult(
            "ch07_exam_03",
            False,
            "failed",
            "Exam Task 7.3 — Check Dose Capping, Tuple Return & Non-Mutation",
            "Cap any dose strictly > cap_mg down to cap_mg in a new list, count how many were capped, and return (adjusted_list, capped_count).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )
    return CheckResult(
        "ch07_exam_03",
        True,
        "passed",
        "Exam Task 7.3 Passed (35/35 pts) — Safe Dose Adjuster Verified!",
        "Returned exact (adjusted_list, capped_count) tuples across nominal, exact-cap boundary, float, and empty batches without mutating inputs.",
        points_earned=35,
        points_possible=35,
        tip="Returning `(adjusted_list, capped_count)` preserves the original prescribed doses for audit trails while handing pharmacy code both the capped doses and the override count.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch07_exam_03",
        chapter="ch07",
        title="Exam 7.3: Flag & Adjust Medication Doses",
        book_section="Chapter 7 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Create an empty `adjusted = []` list and `capped_count = 0`. Loop over `doses_mg`: if `dose > cap_mg`, append `cap_mg` and increment `capped_count`; else append `dose`. Return `(adjusted, capped_count)`.",
            2: "Algorithm:\n1. adjusted = []; capped_count = 0\n2. for dose in doses_mg:\n       if dose > cap_mg:\n           adjusted.append(cap_mg); capped_count += 1\n       else:\n           adjusted.append(dose)\n3. return (adjusted, capped_count)",
            3: "def flag_and_adjust_doses(doses_mg, cap_mg):\n    adjusted = []\n    capped_count = 0\n    for dose in doses_mg:\n        if dose > ___:\n            adjusted.append(cap_mg)\n            capped_count += 1\n        else:\n            adjusted.append(dose)\n    return (adjusted, capped_count)",
        },
        verifier=_verify_exam_03,
    )
)
