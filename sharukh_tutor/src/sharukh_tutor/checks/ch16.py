"""Chapter 16 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch16_todo_01..06) and Coding Exam (ch16_exam_01..03)
for NumPy & Vectorized Thinking — Arrays, Broadcasting, and Linear Algebra
in Action. NumPy is imported lazily inside verifiers so `sharukh_tutor`
imports cleanly even in pure-Python environments. Every verifier checks
array types, shapes, dtypes, boundary conditions, and non-mutation while
letting `NotImplementedError` propagate cleanly for unattempted stubs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _get_np() -> Any:
    """Lazy-import NumPy inside verifiers to keep package import pure-Python."""
    import numpy as np

    return np


def _arr_close(np: Any, got: Any, expected: Any, atol: float = 1e-6) -> bool:
    """Return True iff got is an ndarray with matching shape and close values."""
    exp = np.asarray(expected)
    return (
        isinstance(got, np.ndarray)
        and got.shape == exp.shape
        and bool(np.allclose(got, exp, atol=atol))
    )


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
                f"Expected a function (like `{expected_name}`), "
                f"but received {type(fn).__name__}."
            ),
            details=[
                f"Pass the function itself: tutor.check('{ex_id}', {expected_name})",
                f"Do NOT call {expected_name}(...) inside tutor.check().",
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
        f"{title} — Function Returned None (print() vs. return Trap!)",
        f"Calling {call_str} returned None instead of the expected result.",
        details=[
            "Remember: print() only displays output on screen; functions must use `return ...`.",
            f"Verify that {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Vectorized functions should compute and `return` a NumPy array (or scalar/dict), never just print it.",
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 16.1: vectorized_bmi(weights_kg, heights_cm)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_todo_01", "To-Do 16.1", fn, "vectorized_bmi", 10)
    if bad is not None:
        return bad

    np = _get_np()

    w_list = [64.5, 54.6, 70.0, 81.2]
    h_list = [168.0, 156.0, 175.0, 180.0]
    w_snap = list(w_list)
    h_snap = list(h_list)
    expected = np.array([22.85, 22.44, 22.86, 25.06], dtype=float)

    try:
        got_from_lists = fn(w_list, h_list)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch16_todo_01",
            False,
            "failed",
            "To-Do 16.1 — Runtime Error on List Inputs",
            f"vectorized_bmi(weights_list, heights_list) raised {type(exc).__name__}: {exc}",
            details=[
                "Convert both inputs with `np.asarray(..., dtype=float)` first!",
                "Plain Python lists do not support `/` or `**` directly.",
            ],
        )

    if got_from_lists is None:
        return _none_return_trap("ch16_todo_01", "To-Do 16.1", "vectorized_bmi(w, h)", 10)

    errors = []
    if w_list != w_snap or h_list != h_snap:
        errors.append("Do NOT mutate the caller's input lists/arrays!")

    if not isinstance(got_from_lists, np.ndarray):
        errors.append(
            f"Expected a 1D `np.ndarray`, got `{type(got_from_lists).__name__}` ({got_from_lists!r})."
        )
    else:
        if got_from_lists.ndim != 1 or got_from_lists.shape != (4,):
            errors.append(f"Expected shape (4,), got shape {got_from_lists.shape}.")
        elif np.all(got_from_lists < 0.05):
            errors.append(
                f"Returned {got_from_lists.tolist()} — did you forget to convert `heights_cm` to meters (`heights_cm / 100.0`)?"
            )
        elif not np.allclose(got_from_lists, expected, atol=1e-6):
            errors.append(
                f"For weights={w_snap}, heights={h_snap}: expected {expected.tolist()}, "
                f"got {got_from_lists.tolist()} (round to 2 decimals with `np.round(..., 2)`)."
            )

    # Also test NumPy array inputs & non-mutation
    w_arr = np.array([50.0, 90.0], dtype=float)
    h_arr = np.array([160.0, 180.0], dtype=float)
    w_arr_snap = w_arr.copy()
    h_arr_snap = h_arr.copy()
    exp2 = np.array([19.53, 27.78], dtype=float)

    try:
        got_arr = fn(w_arr, h_arr)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"vectorized_bmi(np.ndarray, np.ndarray) raised {type(exc).__name__}: {exc}")
    else:
        if not np.array_equal(w_arr, w_arr_snap) or not np.array_equal(h_arr, h_arr_snap):
            errors.append("Input NumPy arrays were mutated in place!")
        if not _arr_close(np, got_arr, exp2):
            errors.append(f"For array inputs [50, 90] & [160, 180]: expected {exp2.tolist()}, got {got_arr!r}.")

    if errors:
        return CheckResult(
            "ch16_todo_01",
            False,
            "failed",
            "To-Do 16.1 — Check Array Conversion, Unit Conversion & Rounding",
            "Convert inputs with `np.asarray(..., dtype=float)`, divide heights by 100, and return `np.round(bmi, 2)`.",
            details=errors,
        )

    return CheckResult(
        "ch16_todo_01",
        True,
        "passed",
        "To-Do 16.1 Passed — Vectorized BMI Verified!",
        "Handled both Python list and NumPy array inputs, computed elementwise BMI without loops, and rounded to 2 decimals.",
        tip="Calling `np.asarray(x, dtype=float)` at the top of a scientific function makes it seamlessly accept lists, tuples, or ndarrays.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_todo_01",
        chapter="ch16",
        title="To-Do 16.1: Vectorized Cohort BMI (`vectorized_bmi`)",
        book_section="Section 16.3.1 (Contiguous Arrays vs. Python Lists)",
        points=10,
        hints={
            1: "Convert `weights_kg` and `heights_cm` to float NumPy arrays via `np.asarray(..., dtype=float)`, convert cm to meters (`/ 100.0`), compute `w / (h_m ** 2)`, and round with `np.round(..., 2)`.",
            2: "Blueprint:\n1. `w = np.asarray(weights_kg, dtype=float)`\n2. `h_m = np.asarray(heights_cm, dtype=float) / 100.0`\n3. `return np.round(w / (h_m ** 2), 2)`",
            3: "def vectorized_bmi(weights_kg, heights_cm):\n    w = np.asarray(weights_kg, dtype=float)\n    h_m = np.asarray(heights_cm, dtype=float) / ___\n    return np.round(w / (h_m ** 2), ___)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 16.2: clean_and_mean_hb(hb_array)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_todo_02", "To-Do 16.2", fn, "clean_and_mean_hb", 10)
    if bad is not None:
        return bad

    np = _get_np()

    raw = np.array([10.4, 999.0, 11.8, np.nan, 12.6], dtype=float)
    raw_snap = raw.copy()

    try:
        got = fn(raw)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch16_todo_02",
            False,
            "failed",
            "To-Do 16.2 — Runtime Error Cleaning Hemoglobin Array",
            f"clean_and_mean_hb(raw) raised {type(exc).__name__}: {exc}",
            details=[
                "Start with `arr = np.array(hb_array, dtype=float, copy=True)` so integer inputs can hold `np.nan` and caller data stays untouched."
            ],
        )

    if got is None:
        return _none_return_trap("ch16_todo_02", "To-Do 16.2", "clean_and_mean_hb([10.4, 999.0, 11.8, nan, 12.6])", 10)

    errors = []
    if not np.array_equal(raw, raw_snap, equal_nan=True):
        errors.append(
            "Input `hb_array` was mutated in place! Always work on a float copy (`np.array(hb_array, dtype=float, copy=True)`)."
        )

    if isinstance(got, (float, np.floating)) and math.isnan(float(got)):
        errors.append(
            "Returned `nan`! Use `np.nanmean(arr)` instead of `np.mean(arr)` so missing `NaN` values are ignored."
        )
    elif type(got) is not float:
        errors.append(
            f"Expected a native Python `float` (wrap with `float(...)`), got `{type(got).__name__}` ({got!r})."
        )
    else:
        # Valid values: 10.4, 11.8, 12.6 -> mean = 11.6
        if got > 50.0:
            errors.append(
                f"Got {got!r} — sentinel `999` leaked into the mean! Replace `arr[arr == 999] = np.nan` before averaging."
            )
        elif not math.isclose(got, 11.6, abs_tol=1e-6):
            errors.append(f"For [10.4, 999.0, 11.8, nan, 12.6]: expected 11.6, got {got!r}.")

    # Test integer input list with 999 & rounding to 2 decimals
    int_in = np.array([10, 11, 999, 13], dtype=int)
    int_snap = int_in.copy()
    try:
        got_int = fn(int_in)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(
            f"Integer input `np.array([10, 11, 999, 13])` raised {type(exc).__name__}: {exc} "
            "(convert to `dtype=float` before assigning `np.nan`!)."
        )
    else:
        if not np.array_equal(int_in, int_snap):
            errors.append("Integer input array was mutated in place!")
        if type(got_int) is not float or not math.isclose(got_int, 11.33, abs_tol=1e-6):
            errors.append(f"For [10, 11, 999, 13]: expected 11.33 (Python float), got {got_int!r} ({type(got_int).__name__}).")

    # All-missing case (only 999 and NaN) -> must return None
    all_bad = np.array([999.0, np.nan, 999.0], dtype=float)
    try:
        got_all_bad = fn(all_bad)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"All-missing input [999.0, nan, 999.0] raised {type(exc).__name__}: {exc}")
    else:
        if got_all_bad is not None:
            errors.append(
                f"When all entries are 999 or NaN, expected `None`, got {got_all_bad!r}. "
                "Guard with `if np.all(np.isnan(arr)): return None`."
            )

    if errors:
        return CheckResult(
            "ch16_todo_02",
            False,
            "failed",
            "To-Do 16.2 — Check Float Copy, Sentinel Masking & All-NaN Guard",
            "Copy as float, replace `999` with `np.nan`, return `None` if all NaN, else `round(float(np.nanmean(arr)), 2)`.",
            details=errors,
        )

    return CheckResult(
        "ch16_todo_02",
        True,
        "passed",
        "To-Do 16.2 Passed — Sentinel Cleaning & `np.nanmean` Verified!",
        "Protected caller arrays via float copy, converted 999 to NaN, handled all-NaN cleanly with None, and returned native Python float.",
        tip="Integer NumPy arrays cannot hold `np.nan` — always cast to `dtype=float` on a copy before inserting NaNs.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_todo_02",
        chapter="ch16",
        title="To-Do 16.2: Sentinel-Safe Hemoglobin Mean (`clean_and_mean_hb`)",
        book_section="Section 16.3.4 (Boolean Masking & Sentinel-Safe `np.nanmean`)",
        points=10,
        hints={
            1: "Create a float copy with `np.array(hb_array, dtype=float, copy=True)`, replace `arr[arr == 999] = np.nan`, check `np.all(np.isnan(arr))` for `None`, and return `round(float(np.nanmean(arr)), 2)`.",
            2: "Blueprint:\n1. `arr = np.array(hb_array, dtype=float, copy=True)`\n2. `arr[arr == 999] = np.nan`\n3. `if arr.size == 0 or np.all(np.isnan(arr)): return None`\n4. `return round(float(np.nanmean(arr)), 2)`",
            3: "def clean_and_mean_hb(hb_array):\n    arr = np.array(hb_array, dtype=float, copy=True)\n    arr[arr == ___] = np.nan\n    if arr.size == 0 or np.all(np.isnan(arr)):\n        return None\n    return round(float(np.nanmean(arr)), 2)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 16.3: standardize_columns(X)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_todo_03", "To-Do 16.3", fn, "standardize_columns", 10)
    if bad is not None:
        return bad

    np = _get_np()

    # Column 0: [10, 20, 30] -> mean=20, std=sqrt(200/3)
    # Column 1: [5, 5, 5] -> constant! zero std -> must become [0, 0, 0]
    # Column 2: [2, 4, 6] -> mean=4, std=sqrt(8/3)
    X = np.array(
        [
            [10.0, 5.0, 2.0],
            [20.0, 5.0, 4.0],
            [30.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    X_snap = X.copy()

    expected = np.array(
        [
            [-1.22474487, 0.0, -1.22474487],
            [0.0, 0.0, 0.0],
            [1.22474487, 0.0, 1.22474487],
        ],
        dtype=float,
    )

    try:
        got = fn(X)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch16_todo_03",
            False,
            "failed",
            "To-Do 16.3 — Runtime Error Standardizing Columns",
            f"standardize_columns(X) raised {type(exc).__name__}: {exc}",
            details=[
                "Compute `means = np.mean(arr, axis=0)` and `stds = np.std(arr, axis=0)`.",
                "Guard zero-std columns before dividing so constant features become 0.0.",
            ],
        )

    if got is None:
        return _none_return_trap("ch16_todo_03", "To-Do 16.3", "standardize_columns(X)", 10)

    errors = []
    if not np.array_equal(X, X_snap):
        errors.append("Input matrix `X` was mutated in place! Work on a copy (`np.array(X, dtype=float, copy=True)`).")

    if not isinstance(got, np.ndarray):
        errors.append(f"Expected a 2D `np.ndarray`, got `{type(got).__name__}`.")
    else:
        if got.shape != X.shape:
            errors.append(f"Expected shape {X.shape}, got {got.shape}.")
        elif np.any(np.isnan(got)) or np.any(np.isinf(got)):
            errors.append(
                "Output contains NaN or Inf! Constant column `[5, 5, 5]` has `std == 0` — "
                "set zero-std columns to `0.0` instead of dividing by zero."
            )
        else:
            # Check if student accidentally standardized along axis=1 (rows)
            row_means = np.mean(X_snap, axis=1, keepdims=True)
            row_stds = np.std(X_snap, axis=1, keepdims=True)
            row_z = (X_snap - row_means) / row_stds
            if np.allclose(got, row_z, atol=1e-3):
                errors.append(
                    "It looks like you standardized across ROWS (`axis=1`) instead of COLUMNS (`axis=0`)! "
                    "Remember: `axis=0` collapses down rows to compute one mean/std per feature column."
                )
            elif not np.allclose(got, expected, atol=1e-3):
                errors.append(
                    f"Column Z-scores mismatch.\nExpected (approx):\n{np.round(expected, 4)}\nGot:\n{np.round(got, 4)}"
                )

    # Non-square (4, 2) matrix check
    X2 = np.array([[1.0, 100.0], [3.0, 200.0], [5.0, 300.0], [7.0, 400.0]])
    exp2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)
    try:
        got2 = fn(X2)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Non-square (4, 2) matrix raised {type(exc).__name__}: {exc}")
    else:
        if not isinstance(got2, np.ndarray) or got2.shape != (4, 2) or not np.allclose(got2, exp2, atol=1e-3):
            errors.append(f"Non-square (4, 2) matrix failed: expected shape (4, 2) and matching Z-scores, got {got2!r}.")

    if errors:
        return CheckResult(
            "ch16_todo_03",
            False,
            "failed",
            "To-Do 16.3 — Check `axis=0` Broadcasting & Zero-Std Guard",
            "Compute column means/stds with `axis=0`, broadcast `(arr - means) / safe_stds`, and set zero-std columns to 0.0.",
            details=errors,
        )

    return CheckResult(
        "ch16_todo_03",
        True,
        "passed",
        "To-Do 16.3 Passed — Column-Wise Broadcasting Standardization Verified!",
        "Standardized both square and rectangular matrices along `axis=0`, kept zero-std columns safely at 0.0, and avoided input mutation.",
        tip="`safe_stds = np.where(stds == 0, 1.0, stds)` followed by `z[:, stds == 0] = 0.0` eliminates divide-by-zero warnings completely.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_todo_03",
        chapter="ch16",
        title="To-Do 16.3: Column-Wise Z-Score Standardization (`standardize_columns`)",
        book_section="Section 16.3.5 (Broadcasting Rules in Action)",
        points=10,
        hints={
            1: "Work on a float copy of `X`. Compute `means = np.mean(arr, axis=0)` and `stds = np.std(arr, axis=0)`. Avoid dividing by zero by substituting `1.0` where `stds == 0`, then zeroing out those constant columns.",
            2: "Blueprint:\n1. `arr = np.array(X, dtype=float, copy=True)`\n2. `means = np.mean(arr, axis=0)` and `stds = np.std(arr, axis=0)`\n3. `safe_stds = np.where(stds == 0, 1.0, stds)`\n4. `z = (arr - means) / safe_stds`\n5. `z[:, stds == 0] = 0.0; return z`",
            3: "def standardize_columns(X):\n    arr = np.array(X, dtype=float, copy=True)\n    means = np.mean(arr, axis=___)\n    stds = np.std(arr, axis=___)\n    safe_stds = np.where(stds == 0, 1.0, stds)\n    z = (arr - means) / safe_stds\n    z[:, stds == 0] = 0.0\n    return z",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 16.4: flag_fever_mask(temps_c, cutoff=38.0)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_todo_04", "To-Do 16.4", fn, "flag_fever_mask", 10)
    if bad is not None:
        return bad

    np = _get_np()

    temps = np.array([36.6, 37.9, 38.0, 39.2, 37.2], dtype=float)
    temps_snap = temps.copy()

    try:
        res = fn(temps)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch16_todo_04",
            False,
            "failed",
            "To-Do 16.4 — Check Default Argument `cutoff=38.0`",
            f"Calling flag_fever_mask(temps) without a second argument raised TypeError: {exc}",
            details=["Define `def flag_fever_mask(temps_c, cutoff=38.0):`."],
        )
    except Exception as exc:
        return CheckResult(
            "ch16_todo_04",
            False,
            "failed",
            "To-Do 16.4 — Runtime Error Flagging Fever",
            f"flag_fever_mask(temps) raised {type(exc).__name__}: {exc}",
        )

    if res is None:
        return _none_return_trap("ch16_todo_04", "To-Do 16.4", "flag_fever_mask(temps)", 10)

    errors = []
    if not np.array_equal(temps, temps_snap):
        errors.append("Input temperature array was mutated in place!")

    if not isinstance(res, tuple) or len(res) != 2:
        return CheckResult(
            "ch16_todo_04",
            False,
            "failed",
            "To-Do 16.4 — Return a 2-Tuple `(mask, count)`",
            f"Expected a 2-tuple `(bool_ndarray, int_count)`, got {type(res).__name__}: {res!r}.",
        )

    mask, count = res
    expected_mask = np.array([False, False, True, True, False], dtype=bool)

    if not isinstance(mask, np.ndarray):
        errors.append(f"First tuple element must be a `np.ndarray`, got `{type(mask).__name__}`.")
    elif mask.dtype != bool:
        errors.append(
            f"First tuple element must have boolean dtype (`bool`), got dtype `{mask.dtype}` "
            "(return `arr >= cutoff`, not `arr[arr >= cutoff]`!)."
        )
    elif not np.array_equal(mask, expected_mask):
        if np.array_equal(mask, np.array([False, False, False, True, False])):
            errors.append(
                "Boundary bug at `38.0`: expected `True` for `38.0` — use inclusive `>= cutoff`, not `> cutoff`!"
            )
        else:
            errors.append(f"Mask mismatch: expected {expected_mask.tolist()}, got {mask.tolist()}.")

    if type(count) is not int:
        errors.append(
            f"Second tuple element should be a native Python `int` (`int(np.sum(mask))`), "
            f"got `{type(count).__name__}` ({count!r})."
        )
    elif count != 2:
        errors.append(f"Expected fever count 2 for {temps_snap.tolist()} at cutoff=38.0, got {count!r}.")

    # Custom cutoff + Python list input check
    try:
        mask2, count2 = fn([36.8, 37.5, 38.1], cutoff=37.5)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Custom cutoff call `flag_fever_mask([36.8, 37.5, 38.1], cutoff=37.5)` raised {type(exc).__name__}: {exc}")
    else:
        exp_m2 = np.array([False, True, True], dtype=bool)
        if not (isinstance(mask2, np.ndarray) and mask2.dtype == bool and np.array_equal(mask2, exp_m2) and count2 == 2):
            errors.append(
                f"Custom cutoff=37.5 on [36.8, 37.5, 38.1]: expected ({exp_m2.tolist()}, 2), got ({mask2!r}, {count2!r})."
            )

    if errors:
        return CheckResult(
            "ch16_todo_04",
            False,
            "failed",
            "To-Do 16.4 — Check Boolean Mask, Inclusive Cutoff & Int Count",
            "Convert `temps_c` with `np.asarray(..., dtype=float)`, compute `mask = arr >= cutoff`, and return `(mask, int(np.sum(mask)))`.",
            details=errors,
        )

    return CheckResult(
        "ch16_todo_04",
        True,
        "passed",
        "To-Do 16.4 Passed — Boolean Fever Mask & Count Verified!",
        "Verified boolean dtype, exact `>= cutoff` boundary at 38.0, custom cutoff=37.5, and Python `int` count.",
        tip="In NumPy, `True == 1` and `False == 0`, so `int(np.sum(mask))` counts matching rows in C speed without any loop!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_todo_04",
        chapter="ch16",
        title="To-Do 16.4: Boolean Mask Fever Detector (`flag_fever_mask`)",
        book_section="Section 16.3.4 (Boolean Masking & Sentinel-Safe `np.nanmean`)",
        points=10,
        hints={
            1: "Convert `temps_c` with `np.asarray(temps_c, dtype=float)`, create the boolean array `mask = arr >= cutoff`, and count `True` entries with `int(np.sum(mask))`.",
            2: "Blueprint:\n1. `arr = np.asarray(temps_c, dtype=float)`\n2. `mask = arr >= cutoff`\n3. `return mask, int(np.sum(mask))`",
            3: "def flag_fever_mask(temps_c, cutoff=38.0):\n    arr = np.asarray(temps_c, dtype=float)\n    mask = arr ___ cutoff\n    return mask, int(np.sum(mask))",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 16.5: weighted_risk_scores(X, weights, bias)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_todo_05", "To-Do 16.5", fn, "weighted_risk_scores", 10)
    if bad is not None:
        return bad

    np = _get_np()

    X = np.array(
        [
            [1.0, 2.0, 0.5],
            [0.0, 1.5, 2.0],
            [2.0, 0.5, 1.0],
            [1.5, 1.5, 1.5],
        ],
        dtype=float,
    )
    w = np.array([0.4, -0.2, 0.6], dtype=float)
    bias = 0.25
    # Row 0: 0.4 - 0.4 + 0.3 + 0.25 = 0.55
    # Row 1: 0.0 - 0.3 + 1.2 + 0.25 = 1.15
    # Row 2: 0.8 - 0.1 + 0.6 + 0.25 = 1.55
    # Row 3: 0.6 - 0.3 + 0.9 + 0.25 = 1.45
    expected = np.array([0.55, 1.15, 1.55, 1.45], dtype=float)

    try:
        got = fn(X, w, bias)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch16_todo_05",
            False,
            "failed",
            "To-Do 16.5 — Runtime Error Computing Weighted Risk Scores",
            f"weighted_risk_scores(X, w, bias) raised {type(exc).__name__}: {exc}",
            details=["Use the `@` matrix-multiplication operator: `np.round(X_arr @ w_arr + bias, 4)`."],
        )

    if got is None:
        return _none_return_trap("ch16_todo_05", "To-Do 16.5", "weighted_risk_scores(X, w, bias)", 10)

    errors = []
    if not isinstance(got, np.ndarray):
        errors.append(f"Expected a 1D `np.ndarray`, got `{type(got).__name__}`.")
    else:
        if got.ndim == 2:
            errors.append(
                f"Returned a 2D array of shape {got.shape} instead of 1D `(4,)`! "
                "Did you use `*` (elementwise multiplication) instead of `@` (matrix multiply)?"
            )
        elif got.shape != (4,):
            errors.append(f"Expected 1D shape (4,), got {got.shape}.")
        elif np.allclose(got, expected - bias, atol=1e-4):
            errors.append("Scores match `X @ weights` without `bias` — don't forget to add `+ bias`!")
        elif not np.allclose(got, expected, atol=1e-6):
            errors.append(f"Expected {expected.tolist()}, got {got.tolist()}.")

    # Rounding to 4 decimals check with list inputs
    X2 = [[1.0, 2.0], [3.0, 4.0]]
    w2 = [1.0 / 3.0, 1.0 / 7.0]
    exp2 = np.round(np.asarray(X2) @ np.asarray(w2) + 0.1, 4)
    try:
        got2 = fn(X2, w2, 0.1)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"List input test raised {type(exc).__name__}: {exc}")
    else:
        if not _arr_close(np, got2, exp2):
            errors.append(f"Rounding test (4 decimals): expected {exp2.tolist()}, got {got2!r}.")

    if errors:
        return CheckResult(
            "ch16_todo_05",
            False,
            "failed",
            "To-Do 16.5 — Check `@` Matrix Multiplication, Bias & 4-Decimal Rounding",
            "Convert `X` and `weights` with `np.asarray(..., dtype=float)` and return `np.round(X_arr @ w_arr + bias, 4)`.",
            details=errors,
        )

    return CheckResult(
        "ch16_todo_05",
        True,
        "passed",
        "To-Do 16.5 Passed — Matrix-Vector Linear Scoring (`X @ w + b`) Verified!",
        "Computed 1D linear risk scores across arbitrary `(n, p)` batches using `@` and rounded cleanly to 4 decimals.",
        tip="`X @ w + b` is the single most common equation in machine learning — powering linear regression, logistic regression, and every neural network layer.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_todo_05",
        chapter="ch16",
        title="To-Do 16.5: Vectorized Linear Risk Scores (`weighted_risk_scores`)",
        book_section="Section 16.3.6 (Linear Algebra in Action: `@` & Norms)",
        points=10,
        hints={
            1: "Convert `X` and `weights` to float NumPy arrays, compute `X_arr @ w_arr + bias`, and round to 4 decimals with `np.round(..., 4)`.",
            2: "Blueprint:\n1. `X_arr = np.asarray(X, dtype=float)`\n2. `w_arr = np.asarray(weights, dtype=float)`\n3. `return np.round(X_arr @ w_arr + bias, 4)`",
            3: "def weighted_risk_scores(X, weights, bias):\n    X_arr = np.asarray(X, dtype=float)\n    w_arr = np.asarray(weights, dtype=float)\n    return np.round(X_arr ___ w_arr + bias, ___)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 16.6: pairwise_l2_distances(points, reference)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_todo_06", "To-Do 16.6", fn, "pairwise_l2_distances", 10)
    if bad is not None:
        return bad

    np = _get_np()

    pts = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [1.0, 1.0],
            [-3.0, 4.0],
        ],
        dtype=float,
    )
    ref = np.array([0.0, 0.0], dtype=float)
    expected = np.array([0.0, 5.0, 1.4142, 5.0], dtype=float)

    try:
        got = fn(pts, ref)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch16_todo_06",
            False,
            "failed",
            "To-Do 16.6 — Runtime Error Computing L2 Distances",
            f"pairwise_l2_distances(pts, ref) raised {type(exc).__name__}: {exc}",
            details=["Use broadcasting subtraction and row-wise norm: `np.round(np.linalg.norm(pts - ref, axis=1), 4)`."],
        )

    if got is None:
        return _none_return_trap("ch16_todo_06", "To-Do 16.6", "pairwise_l2_distances(pts, ref)", 10)

    errors = []
    if not isinstance(got, np.ndarray):
        if isinstance(got, (float, int, np.floating)):
            errors.append(
                f"Returned a single scalar ({got!r}) instead of a 1D distance array! "
                "Pass `axis=1` to `np.linalg.norm(pts - ref, axis=1)` so each row gets its own distance."
            )
        else:
            errors.append(f"Expected a 1D `np.ndarray`, got `{type(got).__name__}`.")
    else:
        if got.ndim != 1 or got.shape != (4,):
            errors.append(
                f"Expected 1D shape (4,), got shape {got.shape} (check that you used `axis=1`, not `axis=0`)."
            )
        elif np.allclose(got, np.array([0.0, 25.0, 2.0, 25.0]), atol=1e-4):
            errors.append("Returned squared distances — take the square root (or use `np.linalg.norm(..., axis=1)`)!")
        elif not np.allclose(got, expected, atol=1e-6):
            errors.append(f"Expected {expected.tolist()}, got {got.tolist()}.")

    # Non-zero 3D reference test
    pts3 = [[1.0, 2.0, 3.0], [4.0, 6.0, 3.0], [1.0, 2.0, 5.0]]
    ref3 = [1.0, 2.0, 3.0]
    exp3 = np.array([0.0, 5.0, 2.0], dtype=float)
    try:
        got3 = fn(pts3, ref3)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"3D point test raised {type(exc).__name__}: {exc}")
    else:
        if not _arr_close(np, got3, exp3):
            errors.append(f"3D test with non-zero reference: expected {exp3.tolist()}, got {got3!r}.")

    if errors:
        return CheckResult(
            "ch16_todo_06",
            False,
            "failed",
            "To-Do 16.6 — Check Broadcasting Subtraction & `axis=1` L2 Norm",
            "Compute `np.round(np.linalg.norm(pts - ref, axis=1), 4)` on float NumPy arrays.",
            details=errors,
        )

    return CheckResult(
        "ch16_todo_06",
        True,
        "passed",
        "To-Do 16.6 Passed — Broadcasting Euclidean Distances Verified!",
        "Computed exact row-wise L2 norms across 2D and 3D feature spaces with zero loops.",
        tip="Broadcasting `points - reference` subtracts a `(d,)` centroid from all `(n, d)` samples simultaneously — the core step inside k-NN and k-Means!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_todo_06",
        chapter="ch16",
        title="To-Do 16.6: Broadcasting Euclidean Distances (`pairwise_l2_distances`)",
        book_section="Section 16.3.6 (Linear Algebra in Action: `@` & Norms)",
        points=10,
        hints={
            1: "Convert `points` and `reference` with `np.asarray(..., dtype=float)`. Subtract `pts - ref` (broadcasting!), compute row-wise L2 norm with `np.linalg.norm(..., axis=1)`, and round to 4 decimals.",
            2: "Blueprint:\n1. `pts = np.asarray(points, dtype=float)`\n2. `ref = np.asarray(reference, dtype=float)`\n3. `dists = np.linalg.norm(pts - ref, axis=1)`\n4. `return np.round(dists, 4)`",
            3: "def pairwise_l2_distances(points, reference):\n    pts = np.asarray(points, dtype=float)\n    ref = np.asarray(reference, dtype=float)\n    return np.round(np.linalg.norm(pts - ref, axis=___), ___)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 16.1 (30 pts): normalize_vital_matrix(X)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_exam_01", "Exam Task 16.1", fn, "normalize_vital_matrix", 30)
    if bad is not None:
        return bad

    np = _get_np()

    # Col 0: [10.0, 999.0, 14.0, 12.0] -> valid [10, 14, 12], mean=12.0 -> [10, 12, 14, 12] -> [0.0, 0.5, 1.0, 0.5]
    # Col 1: [120.0, 130.0, nan, 140.0] -> valid [120, 130, 140], mean=130.0 -> [120, 130, 130, 140] -> [0.0, 0.5, 0.5, 1.0]
    # Col 2: [37.0, 37.0, 999.0, 37.0] -> valid [37, 37, 37], mean=37.0 -> constant! -> [0.0, 0.0, 0.0, 0.0]
    X = np.array(
        [
            [10.0, 120.0, 37.0],
            [999.0, 130.0, 37.0],
            [14.0, np.nan, 999.0],
            [12.0, 140.0, 37.0],
        ],
        dtype=float,
    )
    X_snap = X.copy()
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
        ],
        dtype=float,
    )

    try:
        got = fn(X)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch16_exam_01",
            False,
            "failed",
            "Exam Task 16.1 — Runtime Error Normalizing Vital Matrix",
            f"normalize_vital_matrix(X) raised {type(exc).__name__}: {exc}",
            details=[
                "Copy `X` as float, replace `999.0` with `np.nan`, impute NaNs with column `np.nanmean(arr, axis=0)`, and min-max scale columns."
            ],
            points_earned=0,
            points_possible=30,
        )

    if got is None:
        return _none_return_trap("ch16_exam_01", "Exam Task 16.1", "normalize_vital_matrix(X)", 30)

    errors = []
    if not np.array_equal(X, X_snap, equal_nan=True):
        errors.append(
            "Input matrix `X` was mutated in place! Always start with `arr = np.array(X, dtype=float, copy=True)`."
        )

    if not isinstance(got, np.ndarray):
        errors.append(f"Expected a 2D `np.ndarray`, got `{type(got).__name__}`.")
    else:
        if got.shape != X.shape:
            errors.append(f"Expected shape {X.shape}, got {got.shape}.")
        elif np.any(np.isnan(got)) or np.any(np.isinf(got)):
            errors.append(
                "Output still contains NaN or Inf! Check two things: (1) impute all NaNs with column `np.nanmean`, "
                "and (2) set constant columns (`col_max == col_min`) to `0.0`."
            )
        elif not np.allclose(got, expected, atol=1e-6):
            errors.append(
                f"Output mismatch on clinical vital matrix.\nExpected:\n{expected}\nGot:\n{got}"
            )

    # Second test verifying 4-decimal rounding
    X2 = np.array(
        [
            [10.0, 1.0],
            [11.0, 4.0],
            [13.0, 999.0],
        ],
        dtype=float,
    )
    # Col 0: min=10, max=13, span=3 -> [0.0, 0.3333, 1.0]
    # Col 1: valid [1, 4], mean=2.5 -> [1.0, 4.0, 2.5], min=1, max=4, span=3 -> [0.0, 1.0, 0.5]
    exp2 = np.array([[0.0, 0.0], [0.3333, 1.0], [1.0, 0.5]], dtype=float)
    try:
        got2 = fn(X2)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Rounding test raised {type(exc).__name__}: {exc}")
    else:
        if not _arr_close(np, got2, exp2):
            errors.append(f"4-decimal rounding test failed:\nExpected:\n{exp2}\nGot:\n{got2!r}")

    if errors:
        return CheckResult(
            "ch16_exam_01",
            False,
            "failed",
            "Exam Task 16.1 — Check Sentinel Cleaning, Imputation & Min-Max Scaling",
            "Clean 999.0 -> NaN on a float copy, impute with column `np.nanmean`, min-max scale to [0, 1] (0.0 for constant cols), round to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch16_exam_01",
        True,
        "passed",
        "Exam Task 16.1 Passed (30/30 pts) — `normalize_vital_matrix` Verified!",
        "Replaced 999.0 sentinels, imputed NaNs with column means via broadcasting, min-max scaled with constant-column guard, and preserved caller input.",
        points_earned=30,
        points_possible=30,
        tip="`arr = np.where(np.isnan(arr), col_means, arr)` imputes every missing cell in a 2D matrix in one single broadcasted expression!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_exam_01",
        chapter="ch16",
        title="Exam 16.1: Sentinel-Cleaned & Min-Max Scaled Vital Matrix",
        book_section="Chapter 16 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Start with `arr = np.array(X, dtype=float, copy=True)`, set `arr[arr == 999.0] = np.nan`, fill NaNs with `np.nanmean(arr, axis=0)` using `np.where(np.isnan(arr), col_means, arr)`, then min-max scale each column with a constant-column guard (`span == 0 -> 0.0`) and `np.round(..., 4)`.",
            2: "Blueprint:\n1. `arr = np.array(X, dtype=float, copy=True)`\n2. `arr[arr == 999.0] = np.nan`\n3. `col_means = np.nanmean(arr, axis=0)`\n4. `arr = np.where(np.isnan(arr), col_means, arr)`\n5. `mins, maxs = np.min(arr, axis=0), np.max(arr, axis=0)`\n6. `span = maxs - mins`; divide by `np.where(span == 0, 1.0, span)`, zero constant cols, `np.round(..., 4)`.",
            3: "def normalize_vital_matrix(X):\n    arr = np.array(X, dtype=float, copy=True)\n    arr[arr == 999.0] = np.nan\n    col_means = np.nanmean(arr, axis=0)\n    arr = np.where(np.isnan(arr), col_means, arr)\n    col_min = np.min(arr, axis=0)\n    span = np.max(arr, axis=0) - col_min\n    safe_span = np.where(span == 0, 1.0, span)\n    scaled = (arr - col_min) / safe_span\n    scaled[:, span == 0] = 0.0\n    return np.round(scaled, ___)",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 16.2 (35 pts): district_weekly_summary(cases_2d, pop_1d)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_exam_02", "Exam Task 16.2", fn, "district_weekly_summary", 35)
    if bad is not None:
        return bad

    np = _get_np()

    # 3 districts x 4 weeks (non-square to catch axis=0 vs axis=1 swaps!)
    cases = np.array(
        [
            [12, 25, 40, 23],  # total=100, peak_idx=2
            [30, 15, 10, 5],   # total=60,  peak_idx=0
            [5, 8, 12, 20],    # total=45,  peak_idx=3
        ],
        dtype=float,
    )
    pop = np.array([250_000, 120_000, 90_000], dtype=float)
    cases_snap = cases.copy()
    pop_snap = pop.copy()

    exp_total = np.array([100.0, 60.0, 45.0])
    # Weekly means: [47/3, 48/3, 62/3, 48/3] -> [15.67, 16.0, 20.67, 16.0]
    exp_weekly = np.array([15.67, 16.0, 20.67, 16.0])
    exp_peak = np.array([2, 0, 3])
    # Incidence per 100k: [100/250k*100k, 60/120k*100k, 45/90k*100k] = [40.0, 50.0, 50.0]
    exp_inc = np.array([40.0, 50.0, 50.0])

    try:
        got = fn(cases, pop)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch16_exam_02",
            False,
            "failed",
            "Exam Task 16.2 — Runtime Error Summarizing Surveillance Grid",
            f"district_weekly_summary(cases, pop) raised {type(exc).__name__}: {exc}",
            details=[
                "Check your axis arguments: `axis=1` aggregates across weeks (one value per district); "
                "`axis=0` aggregates across districts (one value per week)."
            ],
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap("ch16_exam_02", "Exam Task 16.2", "district_weekly_summary(cases, pop)", 35)

    if not isinstance(got, dict):
        return CheckResult(
            "ch16_exam_02",
            False,
            "failed",
            "Exam Task 16.2 — Return a Dictionary",
            f"Expected a `dict` with 4 summary arrays, got `{type(got).__name__}`.",
            points_earned=0,
            points_possible=35,
        )

    errors = []
    if not np.array_equal(cases, cases_snap) or not np.array_equal(pop, pop_snap):
        errors.append("Do NOT mutate input `cases_2d` or `pop_1d` arrays!")

    required_keys = [
        "total_per_district",
        "weekly_mean",
        "peak_week_index",
        "incidence_per_100k",
    ]
    missing = [k for k in required_keys if k not in got]
    if missing:
        return CheckResult(
            "ch16_exam_02",
            False,
            "failed",
            "Exam Task 16.2 — Missing Required Dictionary Keys",
            f"Missing key(s): {missing}. Expected exact keys: {required_keys}.",
            points_earned=0,
            points_possible=35,
        )

    # 1. total_per_district
    tot = got["total_per_district"]
    if not isinstance(tot, np.ndarray) or tot.shape != (3,):
        errors.append(
            f"`total_per_district`: expected 1D np.ndarray of shape (3,), got {type(tot).__name__} "
            f"{getattr(tot, 'shape', '')} (use `np.sum(cases, axis=1)`!)."
        )
    elif not np.allclose(tot, exp_total, atol=1e-6):
        errors.append(f"`total_per_district`: expected {exp_total.tolist()}, got {tot.tolist()}.")

    # 2. weekly_mean
    wmean = got["weekly_mean"]
    if not isinstance(wmean, np.ndarray) or wmean.shape != (4,):
        errors.append(
            f"`weekly_mean`: expected 1D np.ndarray of shape (4,), got {type(wmean).__name__} "
            f"{getattr(wmean, 'shape', '')} (use `np.round(np.mean(cases, axis=0), 2)`!)."
        )
    elif not np.allclose(wmean, exp_weekly, atol=1e-6):
        errors.append(f"`weekly_mean`: expected {exp_weekly.tolist()}, got {wmean.tolist()}.")

    # 3. peak_week_index
    peak = got["peak_week_index"]
    if not isinstance(peak, np.ndarray) or peak.shape != (3,):
        errors.append(
            f"`peak_week_index`: expected 1D int np.ndarray of shape (3,), got {type(peak).__name__} "
            f"{getattr(peak, 'shape', '')} (use `np.argmax(cases, axis=1)`!)."
        )
    elif not np.issubdtype(peak.dtype, np.integer):
        errors.append(f"`peak_week_index`: expected integer dtype, got `{peak.dtype}`.")
    elif not np.array_equal(peak, exp_peak):
        errors.append(f"`peak_week_index`: expected {exp_peak.tolist()}, got {peak.tolist()}.")

    # 4. incidence_per_100k
    inc = got["incidence_per_100k"]
    if not isinstance(inc, np.ndarray) or inc.shape != (3,):
        errors.append(
            f"`incidence_per_100k`: expected 1D np.ndarray of shape (3,), got {type(inc).__name__} "
            f"{getattr(inc, 'shape', '')}."
        )
    elif not np.allclose(inc, exp_inc, atol=1e-6):
        errors.append(f"`incidence_per_100k`: expected {exp_inc.tolist()}, got {inc.tolist()}.")

    # Second shape test (2 districts x 3 weeks) with fractional incidence
    c2 = [[7, 14, 9], [18, 6, 12]]
    p2 = [150_000, 300_000]
    try:
        got2 = fn(c2, p2)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Second test (2x3 lists) raised {type(exc).__name__}: {exc}")
    else:
        if (
            not isinstance(got2, dict)
            or not _arr_close(np, got2.get("total_per_district"), [30, 36])
            or not _arr_close(np, got2.get("weekly_mean"), [12.5, 10.0, 10.5])
            or not isinstance(got2.get("peak_week_index"), np.ndarray)
            or not np.array_equal(got2.get("peak_week_index"), [1, 0])
            or not _arr_close(np, got2.get("incidence_per_100k"), [20.0, 12.0])
        ):
            errors.append(f"Second shape test (2 districts x 3 weeks) returned unexpected values: {got2!r}")

    if errors:
        return CheckResult(
            "ch16_exam_02",
            False,
            "failed",
            "Exam Task 16.2 — Check Axis Directions, Shapes & Rounding",
            "Use `axis=1` for per-district stats (`total`, `argmax`), `axis=0` for `weekly_mean`, and `(total / pop) * 100_000` for incidence.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch16_exam_02",
        True,
        "passed",
        "Exam Task 16.2 Passed (35/35 pts) — `district_weekly_summary` Verified!",
        "Mastered `axis=1` vs. `axis=0` reductions, `np.argmax` peak detection, and vectorized per-100k incidence across multiple grid shapes.",
        points_earned=35,
        points_possible=35,
        tip="Always test 2D axis functions on rectangular `(3, 4)` matrices — shape mismatches immediately expose swapped `axis=0` / `axis=1` bugs!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_exam_02",
        chapter="ch16",
        title="Exam 16.2: District Surveillance Matrix Summary (`district_weekly_summary`)",
        book_section="Chapter 16 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Convert `cases_2d` and `pop_1d` with `np.asarray(..., dtype=float)`. Compute `total = np.sum(cases, axis=1)`, `weekly = np.round(np.mean(cases, axis=0), 2)`, `peak = np.argmax(cases, axis=1)`, and `incidence = np.round((total / pop) * 100_000, 2)`.",
            2: "Blueprint:\n1. `cases = np.asarray(cases_2d, dtype=float)` and `pop = np.asarray(pop_1d, dtype=float)`\n2. `total = np.sum(cases, axis=1)`\n3. `weekly_mean = np.round(np.mean(cases, axis=0), 2)`\n4. `peak_week = np.argmax(cases, axis=1)`\n5. `incidence = np.round((total / pop) * 100_000, 2)`\n6. Return dict with the 4 required keys.",
            3: "def district_weekly_summary(cases_2d, pop_1d):\n    cases = np.asarray(cases_2d, dtype=float)\n    pop = np.asarray(pop_1d, dtype=float)\n    total = np.sum(cases, axis=___)\n    weekly_mean = np.round(np.mean(cases, axis=___), 2)\n    peak_week = np.argmax(cases, axis=1)\n    incidence = np.round((total / pop) * 100_000, 2)\n    return {\n        'total_per_district': total,\n        'weekly_mean': weekly_mean,\n        'peak_week_index': peak_week,\n        'incidence_per_100k': incidence,\n    }",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 16.3 (35 pts): linear_clinical_scorer(X, weights, bias, threshold=0.5)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch16_exam_03", "Exam Task 16.3", fn, "linear_clinical_scorer", 35)
    if bad is not None:
        return bad

    np = _get_np()

    # Row 0: z = 0.0 -> prob = 0.5000 -> pred = 1 (tests inclusive >= 0.5!)
    # Row 1: z = 2.0 -> prob = 0.8808 -> pred = 1
    # Row 2: z = -1.0 -> prob = 0.2689 -> pred = 0
    X = np.array(
        [
            [1.0, -1.0],
            [2.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=float,
    )
    w = np.array([1.0, 1.0], dtype=float)
    bias = 0.0
    X_snap = X.copy()

    exp_probs = np.array([0.5, 0.8808, 0.2689], dtype=float)
    exp_preds = np.array([1, 1, 0], dtype=int)

    try:
        res = fn(X, w, bias)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch16_exam_03",
            False,
            "failed",
            "Exam Task 16.3 — Check Default `threshold=0.5` Signature",
            f"Calling linear_clinical_scorer(X, w, bias) without threshold raised TypeError: {exc}",
            details=["Define `def linear_clinical_scorer(X, weights, bias, threshold=0.5):`."],
            points_earned=0,
            points_possible=35,
        )
    except Exception as exc:
        return CheckResult(
            "ch16_exam_03",
            False,
            "failed",
            "Exam Task 16.3 — Runtime Error in Clinical Scorer",
            f"linear_clinical_scorer(X, w, bias) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if res is None:
        return _none_return_trap("ch16_exam_03", "Exam Task 16.3", "linear_clinical_scorer(X, w, bias)", 35)

    if not isinstance(res, tuple) or len(res) != 2:
        return CheckResult(
            "ch16_exam_03",
            False,
            "failed",
            "Exam Task 16.3 — Return a 2-Tuple `(probs, preds)`",
            f"Expected `(np.round(probs, 4), preds)`, got `{type(res).__name__}`.",
            points_earned=0,
            points_possible=35,
        )

    errors = []
    if not np.array_equal(X, X_snap):
        errors.append("Input matrix `X` was mutated in place!")

    probs, preds = res
    if not isinstance(probs, np.ndarray) or probs.shape != (3,):
        errors.append(f"`probs` should be a 1D np.ndarray of shape (3,), got {type(probs).__name__} {getattr(probs, 'shape', '')}.")
    elif not np.allclose(probs, exp_probs, atol=1e-6):
        errors.append(f"`probs` mismatch: expected {exp_probs.tolist()}, got {probs.tolist()}.")

    if not isinstance(preds, np.ndarray) or preds.shape != (3,):
        errors.append(f"`preds` should be a 1D integer np.ndarray of shape (3,), got {type(preds).__name__} {getattr(preds, 'shape', '')}.")
    elif preds.dtype == bool:
        errors.append(
            "`preds` has boolean dtype (`True`/`False`) — convert to 0/1 integers with `.astype(int)`!"
        )
    elif not np.issubdtype(preds.dtype, np.integer):
        errors.append(f"`preds` should have integer dtype, got `{preds.dtype}`.")
    elif np.array_equal(preds, np.array([0, 1, 0])):
        errors.append("Boundary bug at `prob == 0.5`: use inclusive `probs >= threshold`, not `probs > threshold`!")
    elif not np.array_equal(preds, exp_preds):
        errors.append(f"`preds` mismatch: expected {exp_preds.tolist()}, got {preds.tolist()}.")

    # Custom threshold + non-zero bias test
    try:
        probs2, preds2 = fn(X, w, 0.5, threshold=0.7)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Custom threshold=0.7 test raised {type(exc).__name__}: {exc}")
    else:
        # z = [0.5, 2.5, -0.5] -> probs = [0.6225, 0.9241, 0.3775] -> preds(>=0.7) = [0, 1, 0]
        exp_p2 = np.array([0.6225, 0.9241, 0.3775])
        exp_c2 = np.array([0, 1, 0], dtype=int)
        if not _arr_close(np, probs2, exp_p2):
            errors.append(f"Non-zero bias=0.5 probs: expected {exp_p2.tolist()}, got {probs2!r}.")
        if not (
            isinstance(preds2, np.ndarray)
            and np.issubdtype(preds2.dtype, np.integer)
            and np.array_equal(preds2, exp_c2)
        ):
            errors.append(f"Custom threshold=0.7 preds: expected {exp_c2.tolist()}, got {preds2!r}.")

    if errors:
        return CheckResult(
            "ch16_exam_03",
            False,
            "failed",
            "Exam Task 16.3 — Check Sigmoid Formula, Rounding & `.astype(int)`",
            "Compute `z = X_arr @ w_arr + bias`, `probs = 1.0 / (1.0 + np.exp(-z))`, `preds = (probs >= threshold).astype(int)`, and return `(np.round(probs, 4), preds)`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch16_exam_03",
        True,
        "passed",
        "Exam Task 16.3 Passed (35/35 pts) — Logistic Clinical Scorer Verified!",
        "You just implemented the entire forward pass of Logistic Regression (`X @ w + b` -> sigmoid -> thresholded 0/1 predictions) in 5 lines of pure NumPy!",
        points_earned=35,
        points_possible=35,
        tip="Notice that `z = X @ w + b` followed by `1 / (1 + np.exp(-z))` is literally the mathematical engine inside Scikit-Learn's `LogisticRegression` and PyTorch's `nn.Linear + torch.sigmoid`!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch16_exam_03",
        chapter="ch16",
        title="Exam 16.3: Vectorized Logistic Clinical Scorer (`linear_clinical_scorer`)",
        book_section="Chapter 16 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Convert `X` and `weights` to float arrays, compute `z = X_arr @ w_arr + bias`, apply sigmoid `probs = 1.0 / (1.0 + np.exp(-z))`, threshold with `(probs >= threshold).astype(int)`, and return `(np.round(probs, 4), preds)`.",
            2: "Blueprint:\n1. `X_arr = np.asarray(X, dtype=float)` and `w_arr = np.asarray(weights, dtype=float)`\n2. `z = X_arr @ w_arr + bias`\n3. `probs = 1.0 / (1.0 + np.exp(-z))`\n4. `preds = (probs >= threshold).astype(int)`\n5. `return np.round(probs, 4), preds`",
            3: "def linear_clinical_scorer(X, weights, bias, threshold=0.5):\n    X_arr = np.asarray(X, dtype=float)\n    w_arr = np.asarray(weights, dtype=float)\n    z = X_arr @ w_arr + bias\n    probs = 1.0 / (1.0 + np.exp(___))\n    preds = (probs >= threshold).astype(___)\n    return np.round(probs, 4), preds",
        },
        verifier=_verify_exam_03,
    )
)
