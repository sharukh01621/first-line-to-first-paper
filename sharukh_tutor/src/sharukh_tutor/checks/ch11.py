"""Chapter 11 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch11_todo_01..06) and Part II Capstone Coding Exam
(ch11_exam_01..03): Algorithmic Thinking, Complexity Intuition, Searching,
Sorting, and Recursion. Every verifier tests a callable function across
nominal, edge-case, non-mutation, and complexity/performance checks,
diagnoses the print()-vs-return None trap, and lets NotImplementedError
propagate cleanly.
"""

from __future__ import annotations

import copy
import math
import sys
import time
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
    extra_detail: Optional[str] = None,
) -> CheckResult:
    details = [
        "print() only displays text on screen; it sends back None to the caller.",
        "Make sure your function ends with an explicit `return ...` statement.",
    ]
    if extra_detail:
        details.append(extra_detail)
    return CheckResult(
        ex_id,
        False,
        "failed",
        f"{title} — Returned None (print() vs. return Trap!)",
        f"Calling {call_str} returned None instead of the expected result.",
        details=details,
        points_earned=0,
        points_possible=points_possible,
        tip="Remember: return hands a value back to downstream code; print() only writes to the screen.",
    )


class _TrackedId:
    """Comparable wrapper that counts comparisons to verify O(log n) search."""

    def __init__(self, value: int, counter: List[int]) -> None:
        self.value = value
        self._counter = counter

    def __eq__(self, other: Any) -> bool:
        self._counter[0] += 1
        if isinstance(other, _TrackedId):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other: Any) -> bool:
        self._counter[0] += 1
        if isinstance(other, _TrackedId):
            return self.value < other.value
        return self.value < other

    def __le__(self, other: Any) -> bool:
        self._counter[0] += 1
        if isinstance(other, _TrackedId):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other: Any) -> bool:
        self._counter[0] += 1
        if isinstance(other, _TrackedId):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other: Any) -> bool:
        self._counter[0] += 1
        if isinstance(other, _TrackedId):
            return self.value >= other.value
        return self.value >= other

    def __repr__(self) -> str:
        return f"ID({self.value})"


# ---------------------------------------------------------------------------
# GUIDED TO-DO 11.1: Linear Search for a Patient ID
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_todo_01", "To-Do 11.1", fn, "linear_search_patient", 10)
    if bad is not None:
        return bad

    sample = ["P-101", "P-104", "P-109", "P-112"]
    try:
        nominal = fn(list(sample), "P-109")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch11_todo_01",
            False,
            "failed",
            "To-Do 11.1 — Runtime Error on Nominal Input",
            f"linear_search_patient({sample}, 'P-109') raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch11_todo_01", "To-Do 11.1", "linear_search_patient(['P-101', ...], 'P-109')", 10
        )

    tests = [
        (sample, "P-109", 2),
        (sample, "P-101", 0),
        (sample, "P-112", 3),
        (["P-201", "P-205", "P-205", "P-209"], "P-205", 1),
        (sample, "P-999", -1),
        ([], "P-101", -1),
    ]
    errors = []
    for ids, target, expected in tests:
        label = f"linear_search_patient({ids}, {target!r})"
        try:
            got = fn(list(ids), target)
        except NotImplementedError:
            raise
        except ValueError:
            errors.append(
                f"{label} raised ValueError — return -1 instead of letting .index() crash on missing IDs!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue

        if type(got) is not int or got != expected:
            extra = ""
            if expected >= 0 and got == expected + 1:
                extra = " (Python indices are 0-based, not 1-based!)"
            elif expected == -1 and got is None:
                extra = " (return -1 when target_id is not found, not None!)"
            errors.append(f"{label}: expected {expected}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch11_todo_01",
            False,
            "failed",
            "To-Do 11.1 — Check 0-Based Index & Missing-Target (-1) Return",
            "Scan patient_ids left-to-right with enumerate(); return the first matching 0-based index, or -1 if absent.",
            details=errors,
        )
    return CheckResult(
        "ch11_todo_01",
        True,
        "passed",
        "To-Do 11.1 Passed — O(n) Linear Search Verified!",
        "Passed first, middle, last, duplicate-first-match, missing (-1), and empty-list test cases.",
        tip="Linear search works on any unsorted list in O(n) time—simple and reliable for small or one-off searches.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_todo_01",
        chapter="ch11",
        title="To-Do 11.1: Linear Search for a Patient ID",
        book_section="Section 11.3.1 (Linear Search & O(n) Intuition)",
        points=10,
        hints={
            1: "Loop over `enumerate(patient_ids)` and return `idx` as soon as `pid == target_id`. After the loop ends, return `-1`.",
            2: "Algorithm:\n1. for idx, pid in enumerate(patient_ids):\n       if pid == target_id: return idx\n2. return -1",
            3: "def linear_search_patient(patient_ids, target_id):\n    for idx, pid in enumerate(patient_ids):\n        if pid == target_id:\n            return ___\n    return -1",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 11.2: Iterative Binary Search on Sorted IDs
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_todo_02", "To-Do 11.2", fn, "binary_search_sorted", 10)
    if bad is not None:
        return bad

    sorted_sample = ["P-010", "P-020", "P-030", "P-040", "P-050"]
    try:
        nominal = fn(list(sorted_sample), "P-040")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch11_todo_02",
            False,
            "failed",
            "To-Do 11.2 — Runtime Error on Nominal Input",
            f"binary_search_sorted({sorted_sample}, 'P-040') raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch11_todo_02", "To-Do 11.2", "binary_search_sorted(['P-010', ...], 'P-040')", 10
        )

    tests = [
        (sorted_sample, "P-040", 3),
        (sorted_sample, "P-010", 0),
        (sorted_sample, "P-050", 4),
        (sorted_sample, "P-030", 2),
        (sorted_sample, "P-001", -1),
        (sorted_sample, "P-099", -1),
        (sorted_sample, "P-025", -1),
        (["P-010"], "P-010", 0),
        (["P-010"], "P-020", -1),
        ([], "P-010", -1),
    ]
    errors = []
    for ids, target, expected in tests:
        label = f"binary_search_sorted({ids}, {target!r})"
        try:
            got = fn(list(ids), target)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if type(got) is not int or got != expected:
            extra = ""
            if expected in (0, len(ids) - 1) and got == -1:
                extra = " (check your loop condition: use `while low <= high:`, NOT `low < high`!)"
            errors.append(f"{label}: expected {expected}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch11_todo_02",
            False,
            "failed",
            "To-Do 11.2 — Check Binary Search Pointers & Boundary Cases",
            "Maintain low=0, high=len(sorted_ids)-1, loop `while low <= high:`, check mid=(low+high)//2, and shrink by mid+1 / mid-1.",
            details=errors,
        )

    # Complexity check: verify O(log n) comparison count on 1,024 sorted items
    counter = [0]
    tracked = [_TrackedId(i, counter) for i in range(1024)]
    try:
        idx_last = fn(tracked, _TrackedId(1023, counter))
        idx_miss = fn(tracked, _TrackedId(9999, counter))
        if idx_last != 1023 or idx_miss != -1:
            return CheckResult(
                "ch11_todo_02",
                False,
                "failed",
                "To-Do 11.2 — Failed Large Sorted List Verification",
                f"On a 1,024-item sorted list, expected (1023, -1), got ({idx_last!r}, {idx_miss!r}).",
            )
        if counter[0] > 120:
            return CheckResult(
                "ch11_todo_02",
                False,
                "failed",
                "To-Do 11.2 — Too Many Comparisons (Looks Like O(n) Linear Scan!)",
                (
                    f"Searching a 1,024-item sorted list twice took {counter[0]} comparisons. "
                    "True O(log n) binary search needs at most ~20–45 comparisons total!"
                ),
                details=[
                    "Do not loop over every element or call sorted_ids.index(target_id).",
                    "Halve the search interval each iteration with mid = (low + high) // 2.",
                ],
                tip="log2(1024) = 10: binary search eliminates half the remaining registry on every single step.",
            )
    except NotImplementedError:
        raise
    except TypeError:
        # Fallback wall-clock check if learner coerced items specifically as strings
        big_str = [f"P-{i:06d}" for i in range(50000)]
        t0 = time.perf_counter()
        for _ in range(100):
            r1 = fn(big_str, "P-049999")
            r2 = fn(big_str, "P-999999")
        elapsed = time.perf_counter() - t0
        if r1 != 49999 or r2 != -1 or elapsed > 0.15:
            return CheckResult(
                "ch11_todo_02",
                False,
                "failed",
                "To-Do 11.2 — Performance Check Failed (Use O(log n) Binary Search)",
                f"200 searches on 50,000 sorted IDs took {elapsed:.3f}s (expected < 0.15s with binary search).",
            )

    return CheckResult(
        "ch11_todo_02",
        True,
        "passed",
        "To-Do 11.2 Passed — True O(log n) Binary Search Verified!",
        f"Passed all 10 boundary tests and completed 2 searches over 1,024 sorted IDs in just {counter[0]} comparisons.",
        tip="Binary search turns 1,000,000 sorted patient IDs into at most 20 comparisons—as long as the list stays sorted!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_todo_02",
        chapter="ch11",
        title="To-Do 11.2: Iterative Binary Search on Sorted IDs",
        book_section="Section 11.3.2 (Binary Search & Logarithmic Halving)",
        points=10,
        hints={
            1: "Initialize `low = 0` and `high = len(sorted_ids) - 1`. While `low <= high`, check `mid = (low + high) // 2` and adjust `low = mid + 1` or `high = mid - 1`.",
            2: "Algorithm:\n1. low, high = 0, len(sorted_ids) - 1\n2. while low <= high:\n       mid = (low + high) // 2\n       if sorted_ids[mid] == target_id: return mid\n       elif sorted_ids[mid] < target_id: low = mid + 1\n       else: high = mid - 1\n3. return -1",
            3: "def binary_search_sorted(sorted_ids, target_id):\n    low = 0\n    high = len(sorted_ids) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if sorted_ids[mid] == target_id:\n            return mid\n        elif sorted_ids[mid] < target_id:\n            low = mid + ___\n        else:\n            high = mid - ___\n    return -1",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 11.3: Fast O(n) Set-Based Duplicate ID Finder
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_todo_03", "To-Do 11.3", fn, "find_duplicate_ids_fast", 10)
    if bad is not None:
        return bad

    sample = ["P-105", "P-101", "P-105", "P-109", "P-101", "P-105"]
    sample_copy = list(sample)
    try:
        nominal = fn(sample_copy)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch11_todo_03",
            False,
            "failed",
            "To-Do 11.3 — Runtime Error on Nominal Input",
            f"find_duplicate_ids_fast({sample}) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch11_todo_03", "To-Do 11.3", f"find_duplicate_ids_fast({sample})", 10
        )

    if sample_copy != sample:
        return CheckResult(
            "ch11_todo_03",
            False,
            "failed",
            "To-Do 11.3 — Do Not Mutate Input List",
            "Your function modified the caller's patient_ids list in place.",
            details=["Iterate over patient_ids without calling .sort() or .pop() on the input list."],
        )

    if not isinstance(nominal, list):
        return CheckResult(
            "ch11_todo_03",
            False,
            "failed",
            "To-Do 11.3 — Return a Sorted List",
            f"Expected a sorted list of duplicate IDs, but received {type(nominal).__name__} ({nominal!r}).",
            details=["Convert your duplicates set with `return sorted(duplicates)`."],
        )

    tests = [
        (sample, ["P-101", "P-105"]),
        (["P-001", "P-002", "P-003"], []),
        ([], []),
        (["P-007", "P-007", "P-007"], ["P-007"]),
        (["P-300", "P-100", "P-200", "P-300", "P-100"], ["P-100", "P-300"]),
    ]
    errors = []
    for ids, expected in tests:
        label = f"find_duplicate_ids_fast({ids})"
        try:
            got = fn(list(ids))
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if got != expected:
            extra = ""
            if isinstance(got, list) and sorted(got) == expected:
                extra = " (duplicates found, but forgot to sort the returned list!)"
            elif isinstance(got, list) and sorted(set(got)) == expected:
                extra = " (each duplicate ID should appear only ONCE — store duplicates in a set first!)"
            errors.append(f"{label}: expected {expected!r}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch11_todo_03",
            False,
            "failed",
            "To-Do 11.3 — Check Deduplication & Sorting",
            "Track `seen = set()` and `duplicates = set()` in a single pass, then return `sorted(duplicates)`.",
            details=errors,
        )

    # Performance check on 8,000 IDs (O(n) takes ~0.002s; O(n^2) .count()/list-in takes ~0.6s+)
    big_ids = [f"P-{i:05d}" for i in range(7500)] + [f"P-{i:05d}" for i in range(500)]
    expected_big = [f"P-{i:05d}" for i in range(500)]
    t0 = time.perf_counter()
    got_big = fn(big_ids)
    elapsed = time.perf_counter() - t0
    if got_big != expected_big:
        return CheckResult(
            "ch11_todo_03",
            False,
            "failed",
            "To-Do 11.3 — Large-Batch Result Mismatch",
            f"On 8,000 patient IDs, expected 500 sorted duplicates, got {len(got_big) if isinstance(got_big, list) else type(got_big).__name__}.",
        )
    if elapsed > 0.20:
        return CheckResult(
            "ch11_todo_03",
            False,
            "failed",
            "To-Do 11.3 — Too Slow (Looks Like O(n^2) Duplicate Scan!)",
            (
                f"Processing 8,000 IDs took {elapsed:.3f}s (limit: 0.20s). "
                "Using `patient_ids.count(pid)` or `if pid in seen_list` inside a loop is O(n^2)!"
            ),
            details=[
                "Use `seen = set()` and `duplicates = set()` for O(1) hash-table membership checks.",
                "With sets, 8,000 IDs finishes in under 0.005 seconds!",
            ],
        )

    return CheckResult(
        "ch11_todo_03",
        True,
        "passed",
        "To-Do 11.3 Passed — Fast O(n) Set-Based Duplicate Detector Verified!",
        f"Passed all edge cases and scanned 8,000 patient IDs in {elapsed * 1000:.2f} ms.",
        tip="Replacing `list` membership with `set` membership is the single highest-ROI optimization in Python data cleaning.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_todo_03",
        chapter="ch11",
        title="To-Do 11.3: Fast O(n) Duplicate Patient ID Finder",
        book_section="Section 11.3.1 (O(1) Hash Lookup vs. O(n^2) Scans)",
        points=10,
        hints={
            1: "Create two sets: `seen = set()` and `duplicates = set()`. Loop once through `patient_ids`; if `pid in seen`, add to `duplicates`, else add to `seen`. Return `sorted(duplicates)`.",
            2: "Algorithm:\n1. seen = set(); duplicates = set()\n2. for pid in patient_ids:\n       if pid in seen: duplicates.add(pid)\n       else: seen.add(pid)\n3. return sorted(duplicates)",
            3: "def find_duplicate_ids_fast(patient_ids):\n    seen = set()\n    duplicates = set()\n    for pid in patient_ids:\n        if pid in seen:\n            duplicates.add(pid)\n        else:\n            seen.add(pid)\n    return sorted(___)",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 11.4: Multi-Key Triage Queue Sorter (Non-Mutating)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_todo_04", "To-Do 11.4", fn, "sort_triage_queue", 10)
    if bad is not None:
        return bad

    sample = [
        {"patient_id": "P-01", "urgency": "Routine", "temp_c": 37.1},
        {"patient_id": "P-02", "urgency": "Urgent", "temp_c": 39.2},
        {"patient_id": "P-03", "urgency": "Moderate", "temp_c": 38.8},
        {"patient_id": "P-04", "urgency": "Urgent", "temp_c": 40.1},
        {"patient_id": "P-05", "urgency": "Moderate", "temp_c": 38.2},
        {"patient_id": "P-06", "urgency": "Routine", "temp_c": 37.6},
    ]
    snapshot = copy.deepcopy(sample)
    sample_arg = copy.deepcopy(sample)
    try:
        got = fn(sample_arg)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch11_todo_04",
            False,
            "failed",
            "To-Do 11.4 — TypeError Sorting Dicts Without a Key Function",
            f"sort_triage_queue(...) raised TypeError: {exc}",
            details=[
                "Python cannot compare dicts directly with `<`.",
                "Pass a `key=` function returning a tuple: `(rank[p['urgency']], -p['temp_c'])`.",
            ],
        )
    except Exception as exc:
        return CheckResult(
            "ch11_todo_04",
            False,
            "failed",
            "To-Do 11.4 — Runtime Error on Nominal Queue",
            f"sort_triage_queue(...) raised {type(exc).__name__}: {exc}",
        )

    if got is None:
        return _none_return_trap(
            "ch11_todo_04",
            "To-Do 11.4",
            "sort_triage_queue(patients)",
            10,
            extra_detail="Also check: `patients.sort(...)` mutates in place and returns None! Use `return sorted(patients, key=...)`.",
        )

    if sample_arg != snapshot:
        return CheckResult(
            "ch11_todo_04",
            False,
            "failed",
            "To-Do 11.4 — Input List Was Mutated!",
            "Your function reordered the caller's original `patients` list in place.",
            details=[
                "Use `sorted(patients, key=...)` which returns a brand-new list and leaves `patients` untouched.",
                "Do NOT call `patients.sort(...)` on the input argument.",
            ],
        )

    if got is sample_arg:
        return CheckResult(
            "ch11_todo_04",
            False,
            "failed",
            "To-Do 11.4 — Return a New List Copy",
            "Your function returned the exact same list object as `patients` instead of a new sorted list.",
        )

    expected_ids = ["P-04", "P-02", "P-03", "P-05", "P-06", "P-01"]
    got_ids = [p.get("patient_id") for p in got] if isinstance(got, list) and all(isinstance(x, dict) for x in got) else None

    errors = []
    if got_ids != expected_ids:
        if got_ids == ["P-03", "P-05", "P-06", "P-01", "P-04", "P-02"]:
            errors.append(
                "Urgencies were sorted alphabetically (Moderate < Routine < Urgent)! Map them with `{'Urgent': 0, 'Moderate': 1, 'Routine': 2}`."
            )
        elif got_ids == ["P-02", "P-04", "P-05", "P-03", "P-01", "P-06"]:
            errors.append(
                "Temperatures were sorted ascending instead of descending! Negate temp_c (`-p['temp_c']`) in the sort key."
            )
        else:
            errors.append(f"Expected patient order {expected_ids}, got {got_ids!r}.")

    # Edge cases: empty list and single patient
    try:
        empty_res = fn([])
        if empty_res != []:
            errors.append(f"sort_triage_queue([]): expected [], got {empty_res!r}")
        single = [{"patient_id": "P-99", "urgency": "Routine", "temp_c": 36.8}]
        single_res = fn(single)
        if single_res != single or single_res is single:
            errors.append(f"sort_triage_queue(single) must return a new list equal to {single!r}.")
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Edge-case call raised {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch11_todo_04",
            False,
            "failed",
            "To-Do 11.4 — Check Urgency Rank Map & Descending Temperature Key",
            "Sort by clinical urgency (Urgent -> Moderate -> Routine), then descending temp_c, using `sorted()` without mutating input.",
            details=errors,
        )

    return CheckResult(
        "ch11_todo_04",
        True,
        "passed",
        "To-Do 11.4 Passed — Non-Mutating Multi-Key Triage Sorter Verified!",
        "Ordered Urgent -> Moderate -> Routine with highest fever first inside each tier, leaving the input list untouched.",
        tip="Tuple keys like `(rank[p['urgency']], -p['temp_c'])` let you mix ascending and descending sort directions cleanly in one O(n log n) pass.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_todo_04",
        chapter="ch11",
        title="To-Do 11.4: Multi-Key Triage Queue Sorter",
        book_section="Section 11.3.3 (Custom Sorting with Key Functions)",
        points=10,
        hints={
            1: "Create a rank dict `{'Urgent': 0, 'Moderate': 1, 'Routine': 2}` and pass `key=lambda p: (rank[p['urgency']], -p['temp_c'])` to `sorted(patients, ...)`.",
            2: "Algorithm:\n1. urgency_rank = {'Urgent': 0, 'Moderate': 1, 'Routine': 2}\n2. return sorted(patients, key=lambda p: (urgency_rank[p['urgency']], -p['temp_c']))",
            3: "def sort_triage_queue(patients):\n    urgency_rank = {'Urgent': 0, 'Moderate': 1, 'Routine': 2}\n    return sorted(\n        patients,\n        key=lambda p: (urgency_rank[p['urgency']], -p[___]),\n    )",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 11.5: Recursive Outbreak Case Projection
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_todo_05", "To-Do 11.5", fn, "recursive_outbreak_cases", 10)
    if bad is not None:
        return bad

    try:
        nominal = fn(10, 1.5, 3)
    except NotImplementedError:
        raise
    except RecursionError:
        return CheckResult(
            "ch11_todo_05",
            False,
            "failed",
            "To-Do 11.5 — RecursionError (Missing or Unreachable Base Case!)",
            "Calling recursive_outbreak_cases(10, 1.5, 3) exceeded Python's recursion limit.",
            details=[
                "Add a base case at the very top: `if generations == 0: return round(initial_cases, 2)`.",
                "Ensure your recursive call decrements `generations - 1` toward 0!",
            ],
        )
    except Exception as exc:
        return CheckResult(
            "ch11_todo_05",
            False,
            "failed",
            "To-Do 11.5 — Runtime Error on Nominal Input",
            f"recursive_outbreak_cases(10, 1.5, 3) raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch11_todo_05", "To-Do 11.5", "recursive_outbreak_cases(10, 1.5, 3)", 10
        )

    tests = [
        ((10, 1.5, 0), 10.0),
        ((10, 1.5, 3), 33.75),
        ((5, 2.0, 4), 80.0),
        ((100, 0.8, 3), 51.2),
        ((10, 1.2, 3), 17.28),
        ((10, 1.3, 4), 28.56),
    ]
    errors = []
    for (init_c, r_fac, gens), expected in tests:
        label = f"recursive_outbreak_cases({init_c}, {r_fac}, {gens})"
        try:
            got = fn(init_c, r_fac, gens)
        except NotImplementedError:
            raise
        except RecursionError:
            errors.append(f"{label} raised RecursionError — check your base case `generations == 0`!")
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if not _is_number(got) or not math.isclose(got, expected, abs_tol=1e-2):
            errors.append(f"{label}: expected {expected}, got {got!r}")

    if errors:
        return CheckResult(
            "ch11_todo_05",
            False,
            "failed",
            "To-Do 11.5 — Check Base Case & Recursive Growth Formula",
            "Base case: generations == 0 returns round(initial_cases, 2). Recursive step: round(r_factor * recursive_outbreak_cases(initial_cases, r_factor, generations - 1), 2).",
            details=errors,
        )

    # Trace call depth to verify actual recursive invocation
    call_counts: dict[str, int] = {}

    def _tracer(frame: Any, event: str, arg: Any) -> Any:
        if event == "call":
            name = frame.f_code.co_name
            call_counts[name] = call_counts.get(name, 0) + 1
        return _tracer

    sys.settrace(_tracer)
    try:
        fn(10, 1.5, 3)
    finally:
        sys.settrace(None)

    max_calls = max(call_counts.values()) if call_counts else 0
    if max_calls < 4:
        return CheckResult(
            "ch11_todo_05",
            False,
            "failed",
            "To-Do 11.5 — Practice True Recursion (No Loop / Exponent Shortcut)",
            "Your numerical output is right, but your function did not call itself recursively across the 3 generations!",
            details=[
                "Base case: `if generations == 0: return round(initial_cases, 2)`",
                "Recursive step: call `recursive_outbreak_cases(initial_cases, r_factor, generations - 1)`.",
            ],
            tip="Mastering base case + shrinking subproblem here prepares you directly for tree traversal in To-Do 11.6 and Exam Task 11.3!",
        )

    return CheckResult(
        "ch11_todo_05",
        True,
        "passed",
        "To-Do 11.5 Passed — Recursive Outbreak Model Verified!",
        "Verified base case (gen=0), growth (R>1), decay (R<1), 2-decimal rounding, and true recursive call-stack execution.",
        tip="Every valid recursion has two pillars: a stopping base case (`generations == 0`) and a step that shrinks toward it (`generations - 1`).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_todo_05",
        chapter="ch11",
        title="To-Do 11.5: Recursive Outbreak Case Projection",
        book_section="Section 11.3.4 (Recursion & Call-Stack Tracing)",
        points=10,
        hints={
            1: "Handle `if generations == 0: return round(initial_cases, 2)` first. Otherwise multiply `r_factor` by `recursive_outbreak_cases(initial_cases, r_factor, generations - 1)` and round to 2 decimals.",
            2: "Algorithm:\n1. if generations == 0: return round(initial_cases, 2)\n2. prev = recursive_outbreak_cases(initial_cases, r_factor, generations - 1)\n3. return round(prev * r_factor, 2)",
            3: "def recursive_outbreak_cases(initial_cases, r_factor, generations):\n    if generations == 0:\n        return round(initial_cases, 2)\n    prev = recursive_outbreak_cases(initial_cases, r_factor, generations - ___)\n    return round(prev * r_factor, 2)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 11.6: Recursive Contact-Tracing Chain Counter
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_todo_06", "To-Do 11.6", fn, "count_contact_chain", 10)
    if bad is not None:
        return bad

    tree = {
        "P-01": ["P-02", "P-03"],
        "P-02": ["P-04", "P-05"],
        "P-03": ["P-06"],
        "P-04": [],
    }
    try:
        nominal = fn(tree, "P-01")
    except NotImplementedError:
        raise
    except KeyError as exc:
        return CheckResult(
            "ch11_todo_06",
            False,
            "failed",
            "To-Do 11.6 — KeyError on Leaf Person Omitted from Dictionary",
            f"count_contact_chain(tree, 'P-01') raised KeyError: {exc}",
            details=[
                "Leaf contacts like 'P-05' and 'P-06' often have no entry in `contact_tree`.",
                "Use `contact_tree.get(root_id, [])` instead of `contact_tree[root_id]`!",
            ],
        )
    except RecursionError:
        return CheckResult(
            "ch11_todo_06",
            False,
            "failed",
            "To-Do 11.6 — RecursionError in Contact-Tree Traversal",
            "Make sure you recurse on each child `contact` in `contact_tree.get(root_id, [])`, not on `root_id` again!",
        )
    except Exception as exc:
        return CheckResult(
            "ch11_todo_06",
            False,
            "failed",
            "To-Do 11.6 — Runtime Error on Contact Tree",
            f"count_contact_chain(tree, 'P-01') raised {type(exc).__name__}: {exc}",
        )

    if nominal is None:
        return _none_return_trap(
            "ch11_todo_06", "To-Do 11.6", "count_contact_chain(tree, 'P-01')", 10
        )

    tests = [
        (tree, "P-01", 6),
        (tree, "P-02", 3),
        (tree, "P-03", 2),
        (tree, "P-04", 1),
        (tree, "P-06", 1),
        ({}, "P-99", 1),
    ]
    errors = []
    for t, root, expected in tests:
        label = f"count_contact_chain(tree, {root!r})" if t else f"count_contact_chain({{}}, {root!r})"
        try:
            got = fn(t, root)
        except NotImplementedError:
            raise
        except KeyError as exc:
            errors.append(
                f"{label} raised KeyError: {exc} — use `contact_tree.get(root_id, [])` for safe leaf lookup!"
            )
            continue
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if type(got) is not int or got != expected:
            extra = ""
            if got == expected - 1:
                extra = " (remember to include `root_id` itself by starting `total = 1`!)"
            errors.append(f"{label}: expected {expected}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch11_todo_06",
            False,
            "failed",
            "To-Do 11.6 — Check Root Count & Safe `.get()` Leaf Handling",
            "Start with total = 1 (for root_id), then add count_contact_chain(contact_tree, c) for every c in contact_tree.get(root_id, []).",
            details=errors,
        )

    return CheckResult(
        "ch11_todo_06",
        True,
        "passed",
        "To-Do 11.6 Passed — Recursive Contact-Tree Counter Verified!",
        "Correctly counted full clusters, subtrees, explicit empty leaves, and missing-key leaves without KeyError.",
        tip="Trees are naturally recursive: a cluster's size is always 1 (the index person) plus the sizes of all direct contact sub-clusters.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_todo_06",
        chapter="ch11",
        title="To-Do 11.6: Recursive Contact-Tracing Chain Counter",
        book_section="Section 11.3.4 (Recursive Tree Traversal)",
        points=10,
        hints={
            1: "Start `total = 1` to count `root_id`, then loop over `contact_tree.get(root_id, [])` and add `count_contact_chain(contact_tree, contact)` to `total`.",
            2: "Algorithm:\n1. total = 1\n2. for contact in contact_tree.get(root_id, []):\n       total += count_contact_chain(contact_tree, contact)\n3. return total",
            3: "def count_contact_chain(contact_tree, root_id):\n    total = 1\n    for contact in contact_tree.get(root_id, []):\n        total += count_contact_chain(contact_tree, ___)\n    return total",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 11.1 (30 pts): Fast O(n + m) Hash-Join Lab Matcher
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_exam_01", "Exam Task 11.1", fn, "match_lab_results_fast", 30)
    if bad is not None:
        return bad

    patients = [
        {"patient_id": "P-01", "district": "Dhaka"},
        {"patient_id": "P-02", "district": "Sylhet"},
        {"patient_id": "P-03", "district": "Khulna"},
    ]
    labs = [
        {"patient_id": "P-02", "hb_g_dl": 10.4},
        {"patient_id": "P-01", "hb_g_dl": 12.1},
        {"patient_id": "P-99", "hb_g_dl": 11.0},
    ]
    p_snap = copy.deepcopy(patients)
    l_snap = copy.deepcopy(labs)
    p_arg = copy.deepcopy(patients)
    l_arg = copy.deepcopy(labs)

    try:
        nominal = fn(p_arg, l_arg)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch11_exam_01",
            False,
            "failed",
            "Exam Task 11.1 — Runtime Error on Nominal Input",
            f"match_lab_results_fast(patients, lab_results) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if nominal is None:
        return _none_return_trap(
            "ch11_exam_01", "Exam Task 11.1", "match_lab_results_fast(patients, lab_results)", 30
        )

    if p_arg != p_snap or l_arg != l_snap:
        return CheckResult(
            "ch11_exam_01",
            False,
            "failed",
            "Exam Task 11.1 — Do Not Mutate Input Lists or Dicts!",
            "Your function modified one of the input patient or lab dictionaries in place.",
            details=[
                "Build a fresh dict copy for each patient: `merged = dict(p)` or `{**p, 'hb_g_dl': ...}`.",
                "Never assign `p['hb_g_dl'] = ...` directly onto the caller's dict!",
            ],
            points_earned=0,
            points_possible=30,
        )

    expected = [
        {"patient_id": "P-01", "district": "Dhaka", "hb_g_dl": 12.1},
        {"patient_id": "P-02", "district": "Sylhet", "hb_g_dl": 10.4},
        {"patient_id": "P-03", "district": "Khulna", "hb_g_dl": None},
    ]
    errors = []
    if nominal != expected:
        errors.append(f"Nominal join mismatch:\n  expected: {expected}\n  got:      {nominal!r}")
    elif any(out_d is in_d for out_d, in_d in zip(nominal, p_arg)):
        errors.append("Returned dicts must be new dict copies, not references to input patient dicts.")

    # Edge cases: empty patients, empty labs
    try:
        if fn([], labs) != []:
            errors.append("match_lab_results_fast([], labs) should return [].")
        no_lab_res = fn(copy.deepcopy(patients), [])
        expected_no_lab = [
            {"patient_id": "P-01", "district": "Dhaka", "hb_g_dl": None},
            {"patient_id": "P-02", "district": "Sylhet", "hb_g_dl": None},
            {"patient_id": "P-03", "district": "Khulna", "hb_g_dl": None},
        ]
        if no_lab_res != expected_no_lab:
            errors.append(f"With empty lab_results, expected all hb_g_dl=None, got {no_lab_res!r}")
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Edge-case check raised {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch11_exam_01",
            False,
            "failed",
            "Exam Task 11.1 — Check Hash-Join Logic & Non-Mutation",
            "Index lab_results into a dict {patient_id: hb_g_dl}, then build new merged patient dicts with .get(pid, None).",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    # Performance sanity check on 3,000 rows (O(n+m) ~3 ms vs O(n^2) ~350-1200 ms)
    n_rows = 3000
    big_patients = [{"patient_id": f"P-{i:05d}", "ward": i % 6} for i in range(n_rows)]
    big_labs = [
        {"patient_id": f"P-{i:05d}", "hb_g_dl": round(9.5 + (i % 35) * 0.1, 1)}
        for i in range(n_rows - 1, -1, -1)
    ]
    t0 = time.perf_counter()
    big_out = fn(big_patients, big_labs)
    elapsed = time.perf_counter() - t0

    if (
        not isinstance(big_out, list)
        or len(big_out) != n_rows
        or big_out[0].get("hb_g_dl") != 9.5
        or big_out[-1].get("hb_g_dl") != round(9.5 + ((n_rows - 1) % 35) * 0.1, 1)
    ):
        return CheckResult(
            "ch11_exam_01",
            False,
            "failed",
            "Exam Task 11.1 — 3,000-Row Join Verification Failed",
            "Merged output on 3,000 patients did not contain the expected matched hb_g_dl values.",
            points_earned=0,
            points_possible=30,
        )

    if elapsed > 0.20:
        return CheckResult(
            "ch11_exam_01",
            False,
            "failed",
            "Exam Task 11.1 — Too Slow on 3,000 Rows (O(n^2) Nested Loop Detected!)",
            (
                f"Joining 3,000 patients with 3,000 lab records took {elapsed:.3f}s (limit: 0.20s). "
                "Nested `for p in patients: for lab in lab_results:` runs 9,000,000 iterations!"
            ),
            details=[
                "First build a hash lookup dict in one O(m) pass: `lab_map = {r['patient_id']: r['hb_g_dl'] for r in lab_results}`.",
                "Then build the merged patient list in one O(n) pass using `lab_map.get(p['patient_id'])`.",
            ],
            points_earned=0,
            points_possible=30,
            tip="Hash joins are how pandas.merge() and SQL databases join millions of clinical rows in milliseconds.",
        )

    return CheckResult(
        "ch11_exam_01",
        True,
        "passed",
        "Exam Task 11.1 Passed (30/30 pts) — O(n + m) Hash-Join Verified!",
        f"Verified exact merging, missing-lab None handling, immutability, and 3,000-row hash join in {elapsed * 1000:.2f} ms.",
        points_earned=30,
        points_possible=30,
        tip="Pre-indexing secondary tables into a dictionary turns quadratic O(n*m) merges into lightning-fast O(n+m) pipelines.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_exam_01",
        chapter="ch11",
        title="Exam 11.1: Fast O(n + m) Hash-Join Lab Matcher",
        book_section="Chapter 11 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "First build a lookup dictionary `{row['patient_id']: row['hb_g_dl'] for row in lab_results}`, then loop over `patients` creating a new dict `{**p, 'hb_g_dl': lab_map.get(p['patient_id'])}`.",
            2: "Algorithm:\n1. lab_map = {lab['patient_id']: lab['hb_g_dl'] for lab in lab_results}\n2. merged = []\n3. for p in patients:\n       record = dict(p)\n       record['hb_g_dl'] = lab_map.get(p['patient_id'], None)\n       merged.append(record)\n4. return merged",
            3: "def match_lab_results_fast(patients, lab_results):\n    lab_map = {r['patient_id']: r['hb_g_dl'] for r in lab_results}\n    return [\n        {**p, 'hb_g_dl': lab_map.get(p['patient_id'], ___)}\n        for p in patients\n    ]",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 11.2 (35 pts): District Disease-Burden Ranker
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_exam_02", "Exam Task 11.2", fn, "rank_districts_by_burden", 35)
    if bad is not None:
        return bad

    records = [
        {"district": "Sylhet", "cases": 60, "population": 200000},
        {"district": "Khulna", "cases": 80, "population": 200000},
        {"district": "Barishal", "cases": 40, "population": 100000},
        {"district": "Dhaka", "cases": 250, "population": 500000},
        {"district": "GhostTown", "cases": 15, "population": 0},
        {"district": "BadPop", "cases": 5, "population": -1000},
    ]
    snap = copy.deepcopy(records)
    arg = copy.deepcopy(records)
    try:
        nominal = fn(arg)
    except NotImplementedError:
        raise
    except ZeroDivisionError:
        return CheckResult(
            "ch11_exam_02",
            False,
            "failed",
            "Exam Task 11.2 — ZeroDivisionError on population <= 0",
            "Your function divided by a zero population record instead of filtering it out first.",
            details=["Skip any district record where `row['population'] <= 0` BEFORE dividing!"],
            points_earned=0,
            points_possible=35,
        )
    except Exception as exc:
        return CheckResult(
            "ch11_exam_02",
            False,
            "failed",
            "Exam Task 11.2 — Runtime Error on Nominal Records",
            f"rank_districts_by_burden(records) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch11_exam_02", "Exam Task 11.2", "rank_districts_by_burden(records)", 35
        )

    if arg != snap:
        return CheckResult(
            "ch11_exam_02",
            False,
            "failed",
            "Exam Task 11.2 — Do Not Mutate Input List",
            "Keep the caller's `district_records` unchanged.",
            points_earned=0,
            points_possible=35,
        )

    expected = [
        ("Dhaka", 50.0),
        ("Barishal", 40.0),
        ("Khulna", 40.0),
        ("Sylhet", 30.0),
    ]
    errors = []
    if not isinstance(nominal, list) or not all(isinstance(t, tuple) and len(t) == 2 for t in nominal):
        errors.append(
            f"Expected a list of (district, incidence_per_100k) tuples, got {nominal!r}"
        )
    elif nominal != expected:
        if nominal == [("Dhaka", 50.0), ("Khulna", 40.0), ("Barishal", 40.0), ("Sylhet", 30.0)]:
            errors.append(
                "Tie-breaker order is reversed! When incidence ties (40.0), 'Barishal' must precede 'Khulna'. "
                "Do NOT use `reverse=True` on `(rate, district)` — use `key=lambda item: (-item[1], item[0])`!"
            )
        else:
            errors.append(f"Nominal ranking mismatch: expected {expected}, got {nominal!r}")

    # Extra edge cases: rounding, 0 cases, all invalid, empty
    extra_records = [
        {"district": "Rangpur", "cases": 17, "population": 30000},   # 56.67
        {"district": "Rajshahi", "cases": 0, "population": 120000},  # 0.0
    ]
    try:
        got_extra = fn(extra_records)
        if got_extra != [("Rangpur", 56.67), ("Rajshahi", 0.0)]:
            errors.append(
                f"Rounding/zero-cases test: expected [('Rangpur', 56.67), ('Rajshahi', 0.0)], got {got_extra!r}"
            )
        if fn([]) != []:
            errors.append("rank_districts_by_burden([]) should return [].")
        if fn([{"district": "ZeroPop", "cases": 10, "population": 0}]) != []:
            errors.append("When all districts have population <= 0, return [].")
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Edge-case check raised {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch11_exam_02",
            False,
            "failed",
            "Exam Task 11.2 — Check Filtering, Rounding & Tie-Breaker Sort Key",
            "Filter population > 0, compute round((cases / population) * 100000, 2), and sort by (-incidence, district).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch11_exam_02",
        True,
        "passed",
        "Exam Task 11.2 Passed (35/35 pts) — District Burden Ranker Verified!",
        "Filtered invalid populations, rounded rates to 2 decimals, and broke exact incidence ties alphabetically.",
        points_earned=35,
        points_possible=35,
        tip="Negating numeric keys `(-rate, name)` instead of using `reverse=True` guarantees deterministic ascending alphabetical tie-breaking.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_exam_02",
        chapter="ch11",
        title="Exam 11.2: District Disease-Burden Ranker",
        book_section="Chapter 11 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Keep only rows with `row['population'] > 0`, compute `(row['district'], round((row['cases'] / row['population']) * 100000, 2))`, and sort with `key=lambda item: (-item[1], item[0])`.",
            2: "Algorithm:\n1. ranked = []\n2. for row in district_records:\n       if row['population'] <= 0: continue\n       rate = round((row['cases'] / row['population']) * 100000, 2)\n       ranked.append((row['district'], rate))\n3. return sorted(ranked, key=lambda item: (-item[1], item[0]))",
            3: "def rank_districts_by_burden(district_records):\n    ranked = [\n        (r['district'], round((r['cases'] / r['population']) * 100000, 2))\n        for r in district_records\n        if r['population'] > 0\n    ]\n    return sorted(ranked, key=lambda item: (-item[___], item[___]))",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# CODING EXAM 11.3 (35 pts): Recursive Exposure-Chain Depth Tracer
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch11_exam_03", "Exam Task 11.3", fn, "trace_exposure_depth", 35)
    if bad is not None:
        return bad

    tree = {
        "P-01": ["P-02", "P-03"],
        "P-02": ["P-04"],
        "P-03": ["P-05", "P-06"],
        "P-05": ["P-07"],
        "P-07": ["P-08"],
        "P-04": [],
    }
    try:
        nominal = fn(tree, "P-01")
    except NotImplementedError:
        raise
    except KeyError as exc:
        return CheckResult(
            "ch11_exam_03",
            False,
            "failed",
            "Exam Task 11.3 — KeyError on Leaf Contact",
            f"trace_exposure_depth(tree, 'P-01') raised KeyError: {exc}",
            details=["Use `contacts = contact_tree.get(person_id, [])` so missing leaf keys safely return []."],
            points_earned=0,
            points_possible=35,
        )
    except ValueError as exc:
        return CheckResult(
            "ch11_exam_03",
            False,
            "failed",
            "Exam Task 11.3 — ValueError Calling max() on Empty Contact List",
            f"trace_exposure_depth(tree, 'P-01') raised ValueError: {exc}",
            details=[
                "Add the base case BEFORE calling max(): `if not contacts: return 0`.",
            ],
            points_earned=0,
            points_possible=35,
        )
    except RecursionError:
        return CheckResult(
            "ch11_exam_03",
            False,
            "failed",
            "Exam Task 11.3 — RecursionError",
            "Ensure you recurse on each child `c` in `contact_tree.get(person_id, [])` and return 0 when `contacts` is empty.",
            points_earned=0,
            points_possible=35,
        )
    except Exception as exc:
        return CheckResult(
            "ch11_exam_03",
            False,
            "failed",
            "Exam Task 11.3 — Runtime Error on Nominal Tree",
            f"trace_exposure_depth(tree, 'P-01') raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if nominal is None:
        return _none_return_trap(
            "ch11_exam_03", "Exam Task 11.3", "trace_exposure_depth(tree, 'P-01')", 35
        )

    star_tree = {"A": ["B", "C", "D"]}
    chain_tree = {"N0": ["N1"], "N1": ["N2"], "N2": ["N3"], "N3": ["N4"]}
    tests = [
        (tree, "P-01", 4, "trace_exposure_depth(tree, 'P-01')"),
        (tree, "P-02", 1, "trace_exposure_depth(tree, 'P-02')"),
        (tree, "P-03", 3, "trace_exposure_depth(tree, 'P-03')"),
        (tree, "P-04", 0, "trace_exposure_depth(tree, 'P-04')"),
        (tree, "P-08", 0, "trace_exposure_depth(tree, 'P-08')"),
        ({}, "P-99", 0, "trace_exposure_depth({}, 'P-99')"),
        (star_tree, "A", 1, "trace_exposure_depth({'A': ['B','C','D']}, 'A')"),
        (chain_tree, "N0", 4, "trace_exposure_depth(chain_4_hops, 'N0')"),
    ]
    errors = []
    for t, root, expected, label in tests:
        try:
            got = fn(t, root)
        except NotImplementedError:
            raise
        except Exception as exc:
            errors.append(f"{label} raised {type(exc).__name__}: {exc}")
            continue
        if type(got) is not int or got != expected:
            extra = ""
            if got == expected + 1:
                extra = " (a leaf with no onward contacts has depth 0, not 1!)"
            elif label.startswith("trace_exposure_depth({'A'") and got in (3, 4):
                extra = " (use `max(...)` over child depths, NOT `sum(...)`!)"
            errors.append(f"{label}: expected {expected}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch11_exam_03",
            False,
            "failed",
            "Exam Task 11.3 — Check Base Case (0) & Recursive Max-Depth Logic",
            "Return 0 if contact_tree.get(person_id, []) is empty; else return 1 + max(trace_exposure_depth(contact_tree, c) for c in contacts).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch11_exam_03",
        True,
        "passed",
        "Exam Task 11.3 Passed (35/35 pts) — Recursive Exposure-Depth Tracer Verified!",
        "Passed asymmetric trees, multi-branch star clusters, 4-hop chains, empty-list leaves, and missing-key leaves.",
        points_earned=35,
        points_possible=35,
        tip="Notice the structural parallel: `count_contact_chain` sums subtrees (`1 + sum(...)`), while `trace_exposure_depth` takes the deepest branch (`1 + max(...)`).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch11_exam_03",
        chapter="ch11",
        title="Exam 11.3: Recursive Exposure-Chain Depth Tracer",
        book_section="Chapter 11 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Fetch `contacts = contact_tree.get(person_id, [])`. If `not contacts`, return `0`. Otherwise return `1 + max(trace_exposure_depth(contact_tree, c) for c in contacts)`.",
            2: "Algorithm:\n1. contacts = contact_tree.get(person_id, [])\n2. if not contacts: return 0\n3. return 1 + max(trace_exposure_depth(contact_tree, c) for c in contacts)",
            3: "def trace_exposure_depth(contact_tree, person_id):\n    contacts = contact_tree.get(person_id, [])\n    if not contacts:\n        return ___\n    return 1 + max(\n        trace_exposure_depth(contact_tree, c) for c in contacts\n    )",
        },
        verifier=_verify_exam_03,
    )
)
