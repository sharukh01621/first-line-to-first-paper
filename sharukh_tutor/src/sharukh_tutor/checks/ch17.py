"""Chapter 17 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch17_todo_01..06) and Coding Exam (ch17_exam_01..03)
for Pandas Foundations — Series, DataFrames, Indexing, Filtering, and
Reshaping. Every verifier lazy-imports `pandas` so `sharukh_tutor` stays
importable in minimal Python environments, tests Pandas 3.x non-mutation
guarantees, boundary conditions, column schemas, and tie-breaking, while
letting `NotImplementedError` propagate cleanly for unattempted stubs.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, numbers.Real) and not isinstance(val, bool)


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
                f"Do NOT call `{expected_name}(...)` inside `tutor.check()`.",
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
        f"{title} — Function Returned None (Did You Forget `return` or Use `inplace=True`?)",
        f"Calling {call_str} returned None instead of the expected result.",
        details=[
            "Make sure your function ends with an explicit `return ...` statement.",
            "Never use `inplace=True` in Pandas 3.x — methods with `inplace=True` return `None`!",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Write pure Pandas transformations: `out = df.copy()` or `df.assign(...)` and `return out`.",
    )


def _sample_triage_df() -> Any:
    import pandas as pd

    return pd.DataFrame(
        {
            "patient_id": [
                "P-01",
                "P-02",
                "P-03",
                "P-04",
                "P-05",
                "P-06",
                "P-07",
                "P-08",
            ],
            "district": [
                "Dhaka",
                "Dhaka",
                "Sylhet",
                "Sylhet",
                "Rajshahi",
                "Rajshahi",
                "Khulna",
                "Khulna",
            ],
            "age_years": [25, 17, 18, 42, 31, 60, 14, 29],
            "weight_kg": [64.5, 48.0, 54.6, 70.0, 58.2, 66.0, 38.5, 52.0],
            "height_cm": [168.0, 155.0, 156.0, 175.0, 160.0, 165.0, 148.0, 162.0],
            "temp_c": [38.9, 39.2, 38.9, 36.7, 37.5, 39.2, 38.4, 37.1],
            "hemoglobin_g_dl": [10.4, 9.8, 11.0, 13.2, 10.9, 12.1, 10.2, 11.5],
            "triage_priority": [
                "Urgent",
                "Urgent",
                "Urgent",
                "Routine",
                "Moderate",
                "Urgent",
                "Moderate",
                "Routine",
            ],
        }
    )


# ---------------------------------------------------------------------------
# GUIDED TO-DO 17.1: add_bmi_column(df)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_todo_01", "To-Do 17.1", fn, "add_bmi_column", 10)
    if bad is not None:
        return bad

    df = pd.DataFrame(
        {
            "patient_id": ["P-101", "P-102", "P-103"],
            "weight_kg": [64.5, 54.6, 70.0],
            "height_cm": [168.0, 156.0, 175.0],
        }
    )
    snapshot = df.copy(deep=True)

    try:
        out = fn(df)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch17_todo_01",
            False,
            "failed",
            "To-Do 17.1 — Runtime Error Inside `add_bmi_column`",
            f"Calling add_bmi_column(df) raised {type(exc).__name__}: {exc}",
            details=[
                "Compute `height_m = df['height_cm'] / 100` and `bmi = (df['weight_kg'] / (height_m ** 2)).round(2)`.",
                "Use `df.assign(bmi=...)` or `out = df.copy(); out['bmi'] = ...`.",
            ],
        )

    if out is None:
        return _none_return_trap("ch17_todo_01", "To-Do 17.1", "add_bmi_column(df)", 10)

    if not isinstance(out, pd.DataFrame):
        return CheckResult(
            "ch17_todo_01",
            False,
            "failed",
            "To-Do 17.1 — Expected a pandas DataFrame",
            f"Expected a pandas DataFrame, got {type(out).__name__}.",
            details=["Return a fresh DataFrame copy containing all original columns plus `'bmi'`."],
        )

    errors = []
    if not df.equals(snapshot) or "bmi" in df.columns:
        errors.append(
            "Input DataFrame `df` was mutated in place! Always start with `out = df.copy()` or use `df.assign(...)`."
        )

    if "bmi" not in out.columns:
        errors.append(f"Missing `'bmi'` column in returned DataFrame. Found columns: {list(out.columns)}.")
    else:
        for col in ["patient_id", "weight_kg", "height_cm"]:
            if col not in out.columns:
                errors.append(f"Original column `{col}` is missing from the returned DataFrame.")
        expected_bmi = [22.85, 22.44, 22.86]
        got_bmi = out["bmi"].tolist()
        if len(got_bmi) != len(expected_bmi):
            errors.append(f"Expected {len(expected_bmi)} rows, got {len(got_bmi)}.")
        else:
            for idx, (g, e) in enumerate(zip(got_bmi, expected_bmi)):
                if _is_number(g) and g < 0.05:
                    errors.append(
                        f"Row {idx} BMI is {g!r} — did you forget to convert `height_cm` to meters (`height_cm / 100`)?"
                    )
                    break
                if not _is_number(g) or not math.isclose(float(g), e, abs_tol=1e-2):
                    errors.append(f"Expected BMI values {expected_bmi}, got {got_bmi}.")
                    break
                # Check 2-decimal rounding
                if round(float(g), 2) != float(g):
                    errors.append(
                        f"BMI value {g!r} has more than 2 decimal places — call `.round(2)`."
                    )
                    break

    if errors:
        return CheckResult(
            "ch17_todo_01",
            False,
            "failed",
            "To-Do 17.1 — Check BMI Formula, Rounding & Non-Mutation",
            "Return a fresh copy of `df` with `'bmi'` computed as `(weight_kg / (height_cm / 100) ** 2).round(2)`.",
            details=errors,
        )

    return CheckResult(
        "ch17_todo_01",
        True,
        "passed",
        "To-Do 17.1 Passed — Non-Mutating Vectorized `add_bmi_column` Verified!",
        "Computed 2-decimal BMI across all rows with vectorized Series arithmetic while keeping the caller's DataFrame untouched.",
        tip="In Pandas 3.x, `df.assign(bmi=...)` or `out = df.copy()` is the cleanest way to avoid mutating upstream data.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_todo_01",
        chapter="ch17",
        title="To-Do 17.1: Non-Mutating Vectorized BMI Column (add_bmi_column)",
        book_section="Section 17.3.4 (Vectorized Columns, .assign() & Copy-on-Write)",
        points=10,
        hints={
            1: "Start from `df.copy()` (or use `df.assign(...)`) so the input DataFrame is never mutated. Convert `height_cm` to meters by dividing by `100`, compute `weight_kg / (height_m ** 2)`, and chain `.round(2)`.",
            2: "Blueprint:\n1. `out = df.copy()`\n2. `height_m = out['height_cm'] / 100`\n3. `out['bmi'] = (out['weight_kg'] / (height_m ** 2)).round(2)`\n4. `return out`",
            3: "def add_bmi_column(df):\n    out = df.copy()\n    height_m = out['height_cm'] / ___\n    out['bmi'] = (out['weight_kg'] / (height_m ** 2)).round(___)\n    return out",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 17.2: filter_urgent_adults(df)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_todo_02", "To-Do 17.2", fn, "filter_urgent_adults", 10)
    if bad is not None:
        return bad

    df = _sample_triage_df()
    snapshot = df.copy(deep=True)

    try:
        out = fn(df)
    except NotImplementedError:
        raise
    except ValueError as exc:
        if "truth value of a Series is ambiguous" in str(exc):
            return CheckResult(
                "ch17_todo_02",
                False,
                "failed",
                "To-Do 17.2 — Use Bitwise `&` With Parentheses, Not Python `and`!",
                f"Raised ValueError: {exc}",
                details=[
                    "In Pandas, combine boolean Series masks with `&` and parentheses around each condition:",
                    "    mask = (df['age_years'] >= 18) & (df['triage_priority'] == 'Urgent')",
                ],
                tip="Python's `and`/`or`/`not` expect a single True/False; Pandas Series use elementwise `&`/`|`/`~`.",
            )
        return CheckResult(
            "ch17_todo_02",
            False,
            "failed",
            "To-Do 17.2 — ValueError in `filter_urgent_adults`",
            f"Calling filter_urgent_adults(df) raised ValueError: {exc}",
        )
    except Exception as exc:
        return CheckResult(
            "ch17_todo_02",
            False,
            "failed",
            "To-Do 17.2 — Runtime Error in `filter_urgent_adults`",
            f"Calling filter_urgent_adults(df) raised {type(exc).__name__}: {exc}",
        )

    if out is None:
        return _none_return_trap("ch17_todo_02", "To-Do 17.2", "filter_urgent_adults(df)", 10)

    if not isinstance(out, pd.DataFrame):
        return CheckResult(
            "ch17_todo_02",
            False,
            "failed",
            "To-Do 17.2 — Expected a pandas DataFrame",
            f"Expected a DataFrame, got {type(out).__name__}.",
        )

    errors = []
    if not df.equals(snapshot):
        errors.append("Input DataFrame `df` was mutated in place! Filter without modifying `df`.")

    expected_ids = ["P-01", "P-03", "P-06"]
    if "index" in out.columns:
        errors.append(
            "Found an unexpected `'index'` column! Pass `drop=True` to `.reset_index(drop=True)`."
        )
    if "patient_id" not in out.columns:
        errors.append("Returned DataFrame is missing the original columns (e.g. `'patient_id'`).")
    else:
        got_ids = out["patient_id"].tolist()
        if got_ids != expected_ids:
            extra = ""
            if "P-03" not in got_ids and "P-01" in got_ids:
                extra = " (check boundary: `age_years >= 18` includes 18-year-old P-03!)"
            elif "P-02" in got_ids:
                extra = " (minor P-02 has age 17 and must be excluded!)"
            errors.append(f"Expected patient_ids {expected_ids}, got {got_ids}{extra}.")

    if list(out.index) != list(range(len(out))):
        errors.append(
            f"Index was not reset to 0..{len(out) - 1} — got index {list(out.index)}. Call `.reset_index(drop=True)`."
        )

    # Edge case: no urgent adults at all
    empty_df = df[df["triage_priority"] == "Routine"].copy()
    try:
        empty_out = fn(empty_df)
    except NotImplementedError:
        raise
    except Exception as exc:
        errors.append(f"Filtering a DataFrame with 0 Urgent rows raised {type(exc).__name__}: {exc}")
    else:
        if not isinstance(empty_out, pd.DataFrame) or len(empty_out) != 0:
            errors.append("When no rows match, return an empty DataFrame (0 rows).")

    if errors:
        return CheckResult(
            "ch17_todo_02",
            False,
            "failed",
            "To-Do 17.2 — Check Compound Mask & Index Reset",
            "Filter rows with `(df['age_years'] >= 18) & (df['triage_priority'] == 'Urgent')`, call `.reset_index(drop=True)`, and return a copy.",
            details=errors,
        )

    return CheckResult(
        "ch17_todo_02",
        True,
        "passed",
        "To-Do 17.2 Passed — Compound Boolean Filtering & Clean Index Reset Verified!",
        "Kept exact adult Urgent cases (including age == 18 boundary), excluded minors and Routine/Moderate rows, and reset the index cleanly.",
        tip="Always chain `.reset_index(drop=True)` after filtering when downstream code relies on clean 0-based row positions.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_todo_02",
        chapter="ch17",
        title="To-Do 17.2: Compound Boolean Filtering (filter_urgent_adults)",
        book_section="Section 17.3.3 (Boolean Masking with &, |, ~, and .isin())",
        points=10,
        hints={
            1: "Build a compound boolean mask with `&` and parentheses around BOTH comparisons: `age_years >= 18` and `triage_priority == 'Urgent'`. Slice with `.loc[mask]`, copy, and reset the index with `drop=True`.",
            2: "Blueprint:\n1. `mask = (df['age_years'] >= 18) & (df['triage_priority'] == 'Urgent')`\n2. `return df.loc[mask].reset_index(drop=True).copy()`",
            3: "def filter_urgent_adults(df):\n    mask = (df['age_years'] >= ___) & (df['triage_priority'] == '___')\n    return df.loc[mask].reset_index(drop=___).copy()",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 17.3: select_top_fever_patients(df, n=5)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_todo_03", "To-Do 17.3", fn, "select_top_fever_patients", 10)
    if bad is not None:
        return bad

    df = _sample_triage_df()
    snapshot = df.copy(deep=True)

    try:
        out_default = fn(df)
        out_3 = fn(df, n=3)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch17_todo_03",
            False,
            "failed",
            "To-Do 17.3 — Check Default Parameter `n=5`",
            f"Calling select_top_fever_patients(df) or select_top_fever_patients(df, n=3) raised TypeError: {exc}",
            details=["Define signature: `def select_top_fever_patients(df, n=5):`"],
        )
    except Exception as exc:
        return CheckResult(
            "ch17_todo_03",
            False,
            "failed",
            "To-Do 17.3 — Runtime Error in `select_top_fever_patients`",
            f"Raised {type(exc).__name__}: {exc}",
        )

    if out_default is None or out_3 is None:
        return _none_return_trap("ch17_todo_03", "To-Do 17.3", "select_top_fever_patients(df)", 10)

    if not isinstance(out_default, pd.DataFrame) or not isinstance(out_3, pd.DataFrame):
        return CheckResult(
            "ch17_todo_03",
            False,
            "failed",
            "To-Do 17.3 — Expected a pandas DataFrame",
            f"Expected DataFrame, got {type(out_default).__name__}.",
        )

    errors = []
    if not df.equals(snapshot):
        errors.append("Input DataFrame `df` was mutated in place! Do not pass `inplace=True` to `.sort_values()`.")

    expected_cols = ["patient_id", "district", "temp_c"]
    if list(out_default.columns) != expected_cols:
        errors.append(
            f"Expected exact columns {expected_cols}, got {list(out_default.columns)}."
        )

    # Sorted by temp_c desc, patient_id asc:
    # 39.2: P-02, P-06 | 38.9: P-01, P-03 | 38.4: P-07
    expected_top5 = ["P-02", "P-06", "P-01", "P-03", "P-07"]
    expected_top3 = ["P-02", "P-06", "P-01"]

    if "patient_id" in out_default.columns:
        got5 = out_default["patient_id"].tolist()
        if got5 != expected_top5:
            errors.append(
                f"Top-5 order mismatch: expected {expected_top5} (temp_c desc, then patient_id asc for ties), got {got5}."
            )
    if "patient_id" in out_3.columns:
        got3 = out_3["patient_id"].tolist()
        if got3 != expected_top3:
            errors.append(f"Top-3 order mismatch (n=3): expected {expected_top3}, got {got3}.")

    if list(out_default.index) != list(range(len(out_default))):
        errors.append(
            f"Index must be reset to 0..{len(out_default) - 1} with `drop=True`; got {list(out_default.index)}."
        )

    if errors:
        return CheckResult(
            "ch17_todo_03",
            False,
            "failed",
            "To-Do 17.3 — Check Multi-Key Sort Order, Columns & Reset Index",
            "Sort by `['temp_c', 'patient_id']` with `ascending=[False, True]`, select `['patient_id', 'district', 'temp_c']`, take top `n`, and reset index.",
            details=errors,
        )

    return CheckResult(
        "ch17_todo_03",
        True,
        "passed",
        "To-Do 17.3 Passed — Deterministic Multi-Key Sorting & Top-N Selection Verified!",
        "Ranked patients by temperature descending with ascending `patient_id` tie-breaking, exact 3-column projection, and clean 0-based index.",
        tip="Always specify a deterministic tie-breaker column (like `patient_id`) so top-N clinical tables stay 100% reproducible.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_todo_03",
        chapter="ch17",
        title="To-Do 17.3: Multi-Key Sorting & Column Projection (select_top_fever_patients)",
        book_section="Section 17.3.5 (Sorting, Ranking & Column Selection)",
        points=10,
        hints={
            1: "Call `.sort_values(by=['temp_c', 'patient_id'], ascending=[False, True])`, slice `.head(n)`, select the list of 3 columns `['patient_id', 'district', 'temp_c']`, and chain `.reset_index(drop=True)`.",
            2: "Blueprint:\n1. `cols = ['patient_id', 'district', 'temp_c']`\n2. `ranked = df.sort_values(by=['temp_c', 'patient_id'], ascending=[False, True])`\n3. `return ranked[cols].head(n).reset_index(drop=True)`",
            3: "def select_top_fever_patients(df, n=5):\n    cols = ['patient_id', 'district', 'temp_c']\n    ranked = df.sort_values(\n        by=['temp_c', 'patient_id'],\n        ascending=[___, ___],\n    )\n    return ranked[cols].head(n).reset_index(drop=True)",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 17.4: summarize_numeric_vitals(df, cols)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_todo_04", "To-Do 17.4", fn, "summarize_numeric_vitals", 10)
    if bad is not None:
        return bad

    df = pd.DataFrame(
        {
            "temp_c": [36.8, 37.2, 38.4, 39.0],
            "systolic_bp": [118, 124, 130, 142],
            "hemoglobin_g_dl": [10.5, 11.2, 12.8, 13.5],
        }
    )
    snapshot = df.copy(deep=True)
    cols = ["temp_c", "systolic_bp"]

    try:
        out = fn(df, cols)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch17_todo_04",
            False,
            "failed",
            "To-Do 17.4 — Runtime Error in `summarize_numeric_vitals`",
            f"Calling summarize_numeric_vitals(df, {cols}) raised {type(exc).__name__}: {exc}",
        )

    if out is None:
        return _none_return_trap("ch17_todo_04", "To-Do 17.4", "summarize_numeric_vitals(df, cols)", 10)

    if not isinstance(out, dict):
        return CheckResult(
            "ch17_todo_04",
            False,
            "failed",
            "To-Do 17.4 — Expected a Dictionary Return Value",
            f"Expected a `dict` mapping column names to stat dicts, got {type(out).__name__}.",
        )

    errors = []
    if not df.equals(snapshot):
        errors.append("Input DataFrame `df` was mutated in place!")

    if set(out.keys()) != set(cols):
        errors.append(f"Expected top-level keys {cols}, got {list(out.keys())}.")
    else:
        for col in cols:
            series = df[col]
            exp_mean = float(round(series.mean(), 2))
            exp_med = float(round(series.median(), 2))
            exp_std = float(round(series.std(), 2))
            exp_std_pop = float(round(series.std(ddof=0), 2))

            entry = out[col]
            if not isinstance(entry, dict):
                errors.append(f"Value for `{col}` must be a dict, got {type(entry).__name__}.")
                continue

            for stat_name, exp_val in [
                ("mean", exp_mean),
                ("median", exp_med),
                ("std", exp_std),
            ]:
                if stat_name not in entry:
                    errors.append(f"Missing key `{stat_name}` inside `result[{col!r}]`.")
                else:
                    got_val = entry[stat_name]
                    if not _is_number(got_val):
                        errors.append(
                            f"`result[{col!r}][{stat_name!r}]` should be a float, got {type(got_val).__name__} ({got_val!r})."
                        )
                    elif not math.isclose(float(got_val), exp_val, abs_tol=1e-2):
                        extra = ""
                        if stat_name == "std" and math.isclose(float(got_val), exp_std_pop, abs_tol=1e-2):
                            extra = " (looks like `ddof=0` population std! Use Pandas' default `series.std()` which uses `ddof=1`.)"
                        errors.append(
                            f"`result[{col!r}][{stat_name!r}]`: expected {exp_val}, got {got_val!r}{extra}"
                        )

    if errors:
        return CheckResult(
            "ch17_todo_04",
            False,
            "failed",
            "To-Do 17.4 — Check Summary Dict Structure & Sample Std (ddof=1)",
            "Return `{col: {'mean': float, 'median': float, 'std': float}}` rounded to 2 decimals using Pandas default `.std()` (`ddof=1`).",
            details=errors,
        )

    return CheckResult(
        "ch17_todo_04",
        True,
        "passed",
        "To-Do 17.4 Passed — Numeric Vitals Summary Dictionary Verified!",
        "Computed clean 2-decimal mean, median, and sample standard deviation (`ddof=1`) across requested columns.",
        tip="Remember the Pandas vs. NumPy default trap: Pandas `.std()` defaults to `ddof=1` (sample std), whereas NumPy `np.std()` defaults to `ddof=0`!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_todo_04",
        chapter="ch17",
        title="To-Do 17.4: Summary Statistics Dict (summarize_numeric_vitals)",
        book_section="Section 17.3.1 (Series & DataFrame Summary Methods)",
        points=10,
        hints={
            1: "Loop over each `col` in `cols`, read `s = df[col]`, and store `{'mean': float(round(s.mean(), 2)), 'median': float(round(s.median(), 2)), 'std': float(round(s.std(), 2))}` in a result dictionary.",
            2: "Blueprint:\n1. `summary = {}`\n2. For `col` in `cols`: compute `s = df[col]` and populate `mean`, `median`, `std` wrapped in `float(round(..., 2))`.\n3. `return summary`",
            3: "def summarize_numeric_vitals(df, cols):\n    summary = {}\n    for col in cols:\n        s = df[col]\n        summary[col] = {\n            'mean': float(round(s.mean(), 2)),\n            'median': float(round(s.___(), 2)),\n            'std': float(round(s.___(), 2)),\n        }\n    return summary",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 17.5: melt_vital_measurements(wide_df)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_todo_05", "To-Do 17.5", fn, "melt_vital_measurements", 10)
    if bad is not None:
        return bad

    wide_df = pd.DataFrame(
        {
            "patient_id": ["P-101", "P-102", "P-103"],
            "sbp_day1": [132, 118, 144],
            "sbp_day2": [126, 120, 138],
        }
    )
    snapshot = wide_df.copy(deep=True)

    try:
        out = fn(wide_df)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch17_todo_05",
            False,
            "failed",
            "To-Do 17.5 — Runtime Error in `melt_vital_measurements`",
            f"Calling melt_vital_measurements(wide_df) raised {type(exc).__name__}: {exc}",
            details=[
                "Use `pd.melt(wide_df, id_vars=['patient_id'], value_vars=['sbp_day1', 'sbp_day2'], var_name='visit', value_name='sbp')`."
            ],
        )

    if out is None:
        return _none_return_trap("ch17_todo_05", "To-Do 17.5", "melt_vital_measurements(wide_df)", 10)

    if not isinstance(out, pd.DataFrame):
        return CheckResult(
            "ch17_todo_05",
            False,
            "failed",
            "To-Do 17.5 — Expected a pandas DataFrame",
            f"Expected a DataFrame, got {type(out).__name__}.",
        )

    errors = []
    if not wide_df.equals(snapshot):
        errors.append("Input `wide_df` was mutated in place! `pd.melt` returns a fresh DataFrame automatically.")

    expected_cols = ["patient_id", "visit", "sbp"]
    if list(out.columns) != expected_cols:
        errors.append(
            f"Expected exact columns {expected_cols}, got {list(out.columns)} "
            "(check `var_name='visit'` and `value_name='sbp'`)."
        )
    elif len(out) != 6:
        errors.append(f"Melting 3 patients x 2 days should produce 6 rows, got {len(out)}.")
    else:
        expected_records = {
            ("P-101", "sbp_day1"): 132,
            ("P-101", "sbp_day2"): 126,
            ("P-102", "sbp_day1"): 118,
            ("P-102", "sbp_day2"): 120,
            ("P-103", "sbp_day1"): 144,
            ("P-103", "sbp_day2"): 138,
        }
        got_records = {
            (row["patient_id"], row["visit"]): int(row["sbp"])
            for _, row in out.iterrows()
        }
        if got_records != expected_records:
            errors.append(
                f"Melted `(patient_id, visit) -> sbp` values did not match expected records.\n"
                f"Expected: {expected_records}\nGot: {got_records}"
            )

    if errors:
        return CheckResult(
            "ch17_todo_05",
            False,
            "failed",
            "To-Do 17.5 — Check `pd.melt` Arguments & Long-Format Schema",
            "Melt `['sbp_day1', 'sbp_day2']` keeping `id_vars=['patient_id']`, with `var_name='visit'` and `value_name='sbp'`.",
            details=errors,
        )

    return CheckResult(
        "ch17_todo_05",
        True,
        "passed",
        "To-Do 17.5 Passed — Wide-to-Long Reshaping with `pd.melt` Verified!",
        "Unpivoted repeated blood-pressure visits into clean tidy long format (`patient_id`, `visit`, `sbp`).",
        tip="Tidy long format (one observation per row) is the exact shape expected by Seaborn, Statsmodels, and longitudinal mixed-effects models!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_todo_05",
        chapter="ch17",
        title="To-Do 17.5: Wide-to-Long Reshaping with pd.melt (melt_vital_measurements)",
        book_section="Section 17.3.6 (Reshaping Data — pd.melt, crosstab & pivot_table)",
        points=10,
        hints={
            1: "Use `pd.melt(...)` with `id_vars=['patient_id']`, `value_vars=['sbp_day1', 'sbp_day2']`, `var_name='visit'`, and `value_name='sbp'`.",
            2: "Blueprint:\n1. Call `pd.melt(wide_df, id_vars=['patient_id'], value_vars=['sbp_day1', 'sbp_day2'], var_name='visit', value_name='sbp')`.\n2. Return the resulting DataFrame.",
            3: "def melt_vital_measurements(wide_df):\n    return pd.melt(\n        wide_df,\n        id_vars=['patient_id'],\n        value_vars=['sbp_day1', 'sbp_day2'],\n        var_name='___',\n        value_name='___',\n    )",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 17.6: pivot_district_priority_counts(df)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_todo_06", "To-Do 17.6", fn, "pivot_district_priority_counts", 10)
    if bad is not None:
        return bad

    df = _sample_triage_df()
    snapshot = df.copy(deep=True)

    try:
        out = fn(df)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch17_todo_06",
            False,
            "failed",
            "To-Do 17.6 — Runtime Error in `pivot_district_priority_counts`",
            f"Calling pivot_district_priority_counts(df) raised {type(exc).__name__}: {exc}",
            details=[
                "Try `pd.crosstab(df['district'], df['triage_priority'])` or `df.pivot_table(..., aggfunc='count', fill_value=0)`."
            ],
        )

    if out is None:
        return _none_return_trap("ch17_todo_06", "To-Do 17.6", "pivot_district_priority_counts(df)", 10)

    if not isinstance(out, pd.DataFrame):
        return CheckResult(
            "ch17_todo_06",
            False,
            "failed",
            "To-Do 17.6 — Expected a pandas DataFrame",
            f"Expected a contingency DataFrame, got {type(out).__name__}.",
        )

    errors = []
    if not df.equals(snapshot):
        errors.append("Input DataFrame `df` was mutated in place!")

    expected_districts = {"Dhaka", "Khulna", "Rajshahi", "Sylhet"}
    expected_priorities = {"Moderate", "Routine", "Urgent"}

    if set(out.index) != expected_districts:
        errors.append(
            f"Expected `district` values {sorted(expected_districts)} as the index, got {list(out.index)}."
        )
    if set(out.columns) != expected_priorities:
        errors.append(
            f"Expected `triage_priority` values {sorted(expected_priorities)} as columns, got {list(out.columns)}."
        )

    if not errors:
        if out.isna().any().any():
            errors.append(
                "Found `NaN` values in the contingency table! Fill missing district-priority counts with integer `0` (e.g., `fill_value=0` or `pd.crosstab`)."
            )
        else:
            expected_counts = {
                ("Dhaka", "Moderate"): 0,
                ("Dhaka", "Routine"): 0,
                ("Dhaka", "Urgent"): 2,
                ("Khulna", "Moderate"): 1,
                ("Khulna", "Routine"): 1,
                ("Khulna", "Urgent"): 0,
                ("Rajshahi", "Moderate"): 1,
                ("Rajshahi", "Routine"): 0,
                ("Rajshahi", "Urgent"): 1,
                ("Sylhet", "Moderate"): 0,
                ("Sylhet", "Routine"): 1,
                ("Sylhet", "Urgent"): 1,
            }
            for (dist, prio), exp_cnt in expected_counts.items():
                got_val = out.loc[dist, prio]
                if not _is_number(int(got_val)) or got_val != exp_cnt:
                    errors.append(f"Cell ({dist!r}, {prio!r}): expected {exp_cnt}, got {got_val!r}.")
                    break
                if isinstance(got_val, float) and not float(got_val).is_integer():
                    errors.append(f"Cell ({dist!r}, {prio!r}) should be an integer count, got {got_val!r}.")
                    break
            # Verify integer dtype (not float64)
            non_int_cols = [c for c in out.columns if not pd.api.types.is_integer_dtype(out[c])]
            if non_int_cols:
                errors.append(
                    f"Columns {non_int_cols} have non-integer dtype — ensure counts are integers (`astype(int)` or `pd.crosstab`)."
                )

    if errors:
        return CheckResult(
            "ch17_todo_06",
            False,
            "failed",
            "To-Do 17.6 — Check Contingency Table Axes, Zero-Fill & Integer Counts",
            "Return a contingency DataFrame with `district` as index, `triage_priority` as columns, and integer counts filled with `0`.",
            details=errors,
        )

    return CheckResult(
        "ch17_todo_06",
        True,
        "passed",
        "To-Do 17.6 Passed — District x Triage Priority Contingency Table Verified!",
        "Built a clean zero-filled integer contingency table across all districts and triage levels.",
        tip="`pd.crosstab(df['district'], df['triage_priority'])` produces integer counts with automatic zero-filling in one readable line!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_todo_06",
        chapter="ch17",
        title="To-Do 17.6: Contingency Table (pivot_district_priority_counts)",
        book_section="Section 17.3.6 (Reshaping Data — pd.melt, crosstab & pivot_table)",
        points=10,
        hints={
            1: "The cleanest solution is `pd.crosstab(df['district'], df['triage_priority'])`, which automatically places `district` on the index, `triage_priority` on the columns, and fills absent combinations with integer `0`.",
            2: "Blueprint:\n1. Call `pd.crosstab(df['district'], df['triage_priority'])` (or `df.pivot_table(index='district', columns='triage_priority', values='patient_id', aggfunc='count', fill_value=0)`).\n2. Return the resulting DataFrame.",
            3: "def pivot_district_priority_counts(df):\n    return pd.crosstab(df['___'], df['___'])",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 17.1 (30 Points): prepare_triage_cohort(df)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_exam_01", "Exam Task 17.1", fn, "prepare_triage_cohort", 30)
    if bad is not None:
        return bad

    df = _sample_triage_df()
    snapshot = df.copy(deep=True)

    try:
        out = fn(df)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch17_exam_01",
            False,
            "failed",
            "Exam Task 17.1 — Runtime Error in `prepare_triage_cohort`",
            f"Calling prepare_triage_cohort(df) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if out is None:
        return _none_return_trap("ch17_exam_01", "Exam Task 17.1", "prepare_triage_cohort(df)", 30)

    if not isinstance(out, pd.DataFrame):
        return CheckResult(
            "ch17_exam_01",
            False,
            "failed",
            "Exam Task 17.1 — Expected a pandas DataFrame",
            f"Expected DataFrame, got {type(out).__name__}.",
            points_earned=0,
            points_possible=30,
        )

    errors = []
    if not df.equals(snapshot) or "bmi" in df.columns or "is_anemic" in df.columns:
        errors.append("Input DataFrame `df` was mutated in place! Always work on a copy.")

    for req_col in ("bmi", "is_anemic"):
        if req_col not in out.columns:
            errors.append(f"Missing required derived column `{req_col}`.")

    if "index" in out.columns:
        errors.append("Unexpected `'index'` column detected — pass `drop=True` to `.reset_index(drop=True)`.")

    # Adults (age >= 18): P-01 (25), P-03 (18), P-04 (42), P-05 (31), P-06 (60), P-08 (29)
    expected_ids = ["P-01", "P-03", "P-04", "P-05", "P-06", "P-08"]
    if "patient_id" in out.columns:
        got_ids = out["patient_id"].tolist()
        if got_ids != expected_ids:
            errors.append(
                f"Adult filter (`age_years >= 18`) mismatch: expected {expected_ids}, got {got_ids}."
            )

    if list(out.index) != list(range(len(out))):
        errors.append(f"Index not reset cleanly to 0..{len(out) - 1}; got {list(out.index)}.")

    if not errors:
        expected_bmis = [22.85, 22.44, 22.86, 22.73, 24.24, 19.81]
        got_bmis = out["bmi"].tolist()
        for idx, (g, e) in enumerate(zip(got_bmis, expected_bmis)):
            if not _is_number(g) or not math.isclose(float(g), e, abs_tol=1e-2):
                errors.append(f"BMI mismatch at row {idx}: expected {e}, got {g!r}.")
                break

        # hemoglobin_g_dl for adults: [10.4, 11.0, 13.2, 10.9, 12.1, 11.5]
        # strict < 11.0 -> [True, False, False, True, False, False]
        expected_anemic = [True, False, False, True, False, False]
        if not pd.api.types.is_bool_dtype(out["is_anemic"]):
            errors.append(
                f"`is_anemic` should have boolean dtype, got {out['is_anemic'].dtype}."
            )
        elif out["is_anemic"].tolist() != expected_anemic:
            errors.append(
                f"`is_anemic` mismatch: expected {expected_anemic} (strict `< 11.0`, so `11.0` is False!), "
                f"got {out['is_anemic'].tolist()}."
            )

    if errors:
        return CheckResult(
            "ch17_exam_01",
            False,
            "failed",
            "Exam Task 17.1 — Check Cohort Preparation Pipeline",
            "Copy `df`, add 2-decimal `bmi` and boolean `is_anemic` (`hemoglobin_g_dl < 11.0`), keep `age_years >= 18`, and reset index with `drop=True`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch17_exam_01",
        True,
        "passed",
        "Exam Task 17.1 Passed (30/30) — `prepare_triage_cohort` Verified!",
        "Non-mutating adult triage cohort constructed with exact 2-decimal BMI, strict `< 11.0` boolean anemia flag, and clean 0-based index.",
        points_earned=30,
        points_possible=30,
        tip="Chaining `.loc[...].copy().assign(...).reset_index(drop=True)` makes clinical cohort prep readable from top to bottom.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_exam_01",
        chapter="ch17",
        title="Exam Task 17.1: Adult Triage Cohort Pipeline (prepare_triage_cohort)",
        book_section="Section 17.7 (Chapter 17 Coding Exam — Task 17.1)",
        points=30,
        is_exam=True,
        hints={
            1: "Start with a fresh copy (or filter `df['age_years'] >= 18` into a copy), compute `bmi = (weight_kg / (height_cm / 100) ** 2).round(2)` and `is_anemic = hemoglobin_g_dl < 11.0`, and call `.reset_index(drop=True)`.",
            2: "Blueprint:\n1. `adults = df.loc[df['age_years'] >= 18].copy()`\n2. `adults['bmi'] = (adults['weight_kg'] / (adults['height_cm'] / 100) ** 2).round(2)`\n3. `adults['is_anemic'] = adults['hemoglobin_g_dl'] < 11.0`\n4. `return adults.reset_index(drop=True)`",
            3: "def prepare_triage_cohort(df):\n    out = df.loc[df['age_years'] >= ___].copy()\n    out['bmi'] = (out['weight_kg'] / (out['height_cm'] / 100) ** 2).round(2)\n    out['is_anemic'] = out['hemoglobin_g_dl'] < ___\n    return out.reset_index(drop=True)",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 17.2 (35 Points): rank_districts_by_urgent_share(df)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_exam_02", "Exam Task 17.2", fn, "rank_districts_by_urgent_share", 35)
    if bad is not None:
        return bad

    # Construct test DataFrame with:
    # - Dhaka: 2/2 Urgent -> 100.0%
    # - Rajshahi: 1/2 Urgent -> 50.0%  (tie with Sylhet -> alphabetical 'Rajshahi' first!)
    # - Sylhet: 1/2 Urgent -> 50.0%
    # - Barishal: 1/3 Urgent -> 33.3% (checks 1-decimal rounding!)
    # - Khulna: 0/2 Urgent -> 0.0%    (checks zero-urgent district!)
    df = pd.DataFrame(
        {
            "patient_id": [f"P-{i:02d}" for i in range(1, 12)],
            "district": [
                "Sylhet",
                "Sylhet",
                "Khulna",
                "Khulna",
                "Barishal",
                "Barishal",
                "Barishal",
                "Dhaka",
                "Dhaka",
                "Rajshahi",
                "Rajshahi",
            ],
            "triage_priority": [
                "Urgent",
                "Routine",
                "Moderate",
                "Routine",
                "Urgent",
                "Routine",
                "Moderate",
                "Urgent",
                "Urgent",
                "Moderate",
                "Urgent",
            ],
        }
    )
    snapshot = df.copy(deep=True)

    try:
        out = fn(df)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch17_exam_02",
            False,
            "failed",
            "Exam Task 17.2 — Runtime Error in `rank_districts_by_urgent_share`",
            f"Calling rank_districts_by_urgent_share(df) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if out is None:
        return _none_return_trap("ch17_exam_02", "Exam Task 17.2", "rank_districts_by_urgent_share(df)", 35)

    if not isinstance(out, pd.DataFrame):
        return CheckResult(
            "ch17_exam_02",
            False,
            "failed",
            "Exam Task 17.2 — Expected a pandas DataFrame",
            f"Expected DataFrame, got {type(out).__name__}.",
            points_earned=0,
            points_possible=35,
        )

    errors = []
    if not df.equals(snapshot):
        errors.append("Input DataFrame `df` was mutated in place!")

    expected_cols = ["district", "total_patients", "urgent_patients", "urgent_pct"]
    if list(out.columns) != expected_cols:
        errors.append(f"Expected exact columns {expected_cols}, got {list(out.columns)}.")

    if list(out.index) != list(range(len(out))):
        errors.append(f"Index must be reset to 0..{len(out) - 1} with `drop=True`; got {list(out.index)}.")

    if not errors:
        expected_rows = [
            ("Dhaka", 2, 2, 100.0),
            ("Rajshahi", 2, 1, 50.0),
            ("Sylhet", 2, 1, 50.0),
            ("Barishal", 3, 1, 33.3),
            ("Khulna", 2, 0, 0.0),
        ]
        if len(out) != len(expected_rows):
            errors.append(f"Expected {len(expected_rows)} districts (including 0-urgent Khulna), got {len(out)}.")
        else:
            for idx, (exp_d, exp_tot, exp_urg, exp_pct) in enumerate(expected_rows):
                row = out.iloc[idx]
                got_d = row["district"]
                got_tot = int(row["total_patients"])
                got_urg = int(row["urgent_patients"])
                got_pct = float(row["urgent_pct"])
                if got_d != exp_d or got_tot != exp_tot or got_urg != exp_urg or not math.isclose(got_pct, exp_pct, abs_tol=1e-2):
                    errors.append(
                        f"Row {idx} mismatch: expected ({exp_d!r}, {exp_tot}, {exp_urg}, {exp_pct}), "
                        f"got ({got_d!r}, {got_tot}, {got_urg}, {got_pct})."
                    )
                    break

    if errors:
        return CheckResult(
            "ch17_exam_02",
            False,
            "failed",
            "Exam Task 17.2 — Check District Aggregation, Rounding & Tie-Breaking",
            "Compute `total_patients`, `urgent_patients`, and 1-decimal `urgent_pct` per district, sorted by `urgent_pct` desc then `district` asc with reset index.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch17_exam_02",
        True,
        "passed",
        "Exam Task 17.2 Passed (35/35) — `rank_districts_by_urgent_share` Verified!",
        "Ranked all 5 districts including 1-decimal rounding (33.3%), zero-urgent districts (0.0%), and alphabetical tie-breaking (Rajshahi before Sylhet).",
        points_earned=35,
        points_possible=35,
        tip="Remember: `pd.crosstab` or `groupby` on a boolean `is_urgent` Series preserves zero-urgent districts automatically!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_exam_02",
        chapter="ch17",
        title="Exam Task 17.2: District Urgent Share Ranking (rank_districts_by_urgent_share)",
        book_section="Section 17.7 (Chapter 17 Coding Exam — Task 17.2)",
        points=35,
        is_exam=True,
        hints={
            1: "Create a temporary copy with `is_urgent = (df['triage_priority'] == 'Urgent')`, or use `pd.crosstab(df['district'], df['triage_priority'])`. Count total rows and urgent rows per district, compute `urgent_pct = (urgent_patients / total_patients * 100).round(1)`, sort by `['urgent_pct', 'district']` with `ascending=[False, True]`, and reset index.",
            2: "Blueprint:\n1. `work = df.assign(is_urgent=df['triage_priority'] == 'Urgent')`\n2. `total = work['district'].value_counts()` and `urgent = work.groupby('district')['is_urgent'].sum().astype(int)` (or `pd.DataFrame` indexed by district)\n3. Compute `urgent_pct = (urgent_patients / total_patients * 100).round(1)`\n4. Sort by `['urgent_pct', 'district']` (`ascending=[False, True]`) and `.reset_index(drop=True)`.",
            3: "def rank_districts_by_urgent_share(df):\n    counts = pd.crosstab(df['district'], df['triage_priority'] == 'Urgent')\n    total = counts.sum(axis=1)\n    urgent = counts[True] if True in counts.columns else 0\n    out = pd.DataFrame({\n        'district': total.index,\n        'total_patients': total.values,\n        'urgent_patients': urgent.values if hasattr(urgent, 'values') else urgent,\n    })\n    out['urgent_pct'] = (out['urgent_patients'] / out['total_patients'] * 100).round(___)\n    return out.sort_values(['urgent_pct', 'district'], ascending=[___, ___]).reset_index(drop=True)",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 17.3 (35 Points): reshape_and_compare_visits(wide_df)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch17_exam_03", "Exam Task 17.3", fn, "reshape_and_compare_visits", 35)
    if bad is not None:
        return bad

    wide_pos = pd.DataFrame(
        {
            "patient_id": ["P-101", "P-102", "P-103"],
            "hb_baseline": [9.8, 10.4, 11.2],
            "hb_followup": [11.0, 11.5, 11.8],
        }
    )
    snap_pos = wide_pos.copy(deep=True)
    # deltas: +1.2, +1.1, +0.6 -> mean = 2.9 / 3 = 0.9666... -> 0.97

    wide_neg = pd.DataFrame(
        {
            "patient_id": ["P-201", "P-202"],
            "hb_baseline": [12.0, 11.4],
            "hb_followup": [11.2, 10.9],
        }
    )
    # deltas: -0.8, -0.5 -> mean = -0.65

    try:
        res_pos = fn(wide_pos)
        res_neg = fn(wide_neg)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch17_exam_03",
            False,
            "failed",
            "Exam Task 17.3 — Runtime Error in `reshape_and_compare_visits`",
            f"Calling reshape_and_compare_visits(wide_df) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if res_pos is None or res_neg is None:
        return _none_return_trap("ch17_exam_03", "Exam Task 17.3", "reshape_and_compare_visits(wide_df)", 35)

    if not isinstance(res_pos, tuple) or len(res_pos) != 2:
        return CheckResult(
            "ch17_exam_03",
            False,
            "failed",
            "Exam Task 17.3 — Return a 2-Tuple `(long_df, mean_change)`",
            f"Expected a 2-element tuple `(long_df, mean_change)`, got {type(res_pos).__name__}.",
            details=["Return `return long_df, mean_change` at the end of your function."],
            points_earned=0,
            points_possible=35,
        )

    long_df, mean_change = res_pos
    errors = []

    if not wide_pos.equals(snap_pos):
        errors.append("Input `wide_df` was mutated in place! Do not modify `wide_df`.")

    if not isinstance(long_df, pd.DataFrame):
        errors.append(f"First tuple element `long_df` must be a DataFrame, got {type(long_df).__name__}.")
    else:
        expected_cols = ["patient_id", "visit", "hb_g_dl"]
        if list(long_df.columns) != expected_cols:
            errors.append(f"Expected `long_df` columns {expected_cols}, got {list(long_df.columns)}.")
        elif len(long_df) != 6:
            errors.append(f"Expected 6 rows in `long_df` (3 patients x 2 visits), got {len(long_df)}.")
        else:
            expected_map = {
                ("P-101", "hb_baseline"): 9.8,
                ("P-101", "hb_followup"): 11.0,
                ("P-102", "hb_baseline"): 10.4,
                ("P-102", "hb_followup"): 11.5,
                ("P-103", "hb_baseline"): 11.2,
                ("P-103", "hb_followup"): 11.8,
            }
            got_map = {
                (row["patient_id"], row["visit"]): float(row["hb_g_dl"])
                for _, row in long_df.iterrows()
            }
            if got_map != expected_map:
                errors.append(f"`long_df` melted values mismatch: expected {expected_map}, got {got_map}.")

    if not _is_number(mean_change):
        errors.append(f"Second element `mean_change` must be a float, got {type(mean_change).__name__} ({mean_change!r}).")
    elif not math.isclose(float(mean_change), 0.97, abs_tol=1e-2):
        extra = ""
        if math.isclose(float(mean_change), -0.97, abs_tol=1e-2):
            extra = " (sign is flipped! Subtract `hb_followup - hb_baseline`, not `hb_baseline - hb_followup`.)"
        errors.append(f"Positive cohort `mean_change`: expected 0.97, got {mean_change!r}{extra}.")

    if isinstance(res_neg, tuple) and len(res_neg) == 2:
        _, neg_change = res_neg
        if not _is_number(neg_change) or not math.isclose(float(neg_change), -0.65, abs_tol=1e-2):
            errors.append(f"Negative cohort `mean_change`: expected -0.65, got {neg_change!r}.")
    else:
        errors.append("Second test case did not return a 2-tuple `(long_df, mean_change)`.")

    if errors:
        return CheckResult(
            "ch17_exam_03",
            False,
            "failed",
            "Exam Task 17.3 — Check Melted DataFrame & Paired Mean Change",
            "Melt `['hb_baseline', 'hb_followup']` into `['patient_id', 'visit', 'hb_g_dl']` and return `(long_df, float(round((hb_followup - hb_baseline).mean(), 2)))`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch17_exam_03",
        True,
        "passed",
        "Exam Task 17.3 Passed (35/35) — `reshape_and_compare_visits` Verified!",
        "Produced tidy long-format hemoglobin data and exact 2-decimal paired mean changes (+0.97 and -0.65 g/dL) without mutating inputs.",
        points_earned=35,
        points_possible=35,
        tip="Compute paired differences (`followup - baseline`) in wide format, then melt to long format for plotting or modeling!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch17_exam_03",
        chapter="ch17",
        title="Exam Task 17.3: Longitudinal Melt & Paired Change (reshape_and_compare_visits)",
        book_section="Section 17.7 (Chapter 17 Coding Exam — Task 17.3)",
        points=35,
        is_exam=True,
        hints={
            1: "Use `pd.melt` with `id_vars=['patient_id']`, `value_vars=['hb_baseline', 'hb_followup']`, `var_name='visit'`, and `value_name='hb_g_dl'`. Compute `mean_change = float(round((wide_df['hb_followup'] - wide_df['hb_baseline']).mean(), 2))` and return `(long_df, mean_change)`.",
            2: "Blueprint:\n1. `long_df = pd.melt(wide_df, id_vars=['patient_id'], value_vars=['hb_baseline', 'hb_followup'], var_name='visit', value_name='hb_g_dl')`\n2. `deltas = wide_df['hb_followup'] - wide_df['hb_baseline']`\n3. `mean_change = float(round(deltas.mean(), 2))`\n4. `return long_df, mean_change`",
            3: "def reshape_and_compare_visits(wide_df):\n    long_df = pd.melt(\n        wide_df,\n        id_vars=['patient_id'],\n        value_vars=['hb_baseline', 'hb_followup'],\n        var_name='visit',\n        value_name='hb_g_dl',\n    )\n    mean_change = float(round((wide_df['___'] - wide_df['___']).mean(), 2))\n    return long_df, mean_change",
        },
        verifier=_verify_exam_03,
    )
)
