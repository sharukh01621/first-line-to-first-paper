"""Chapter 18 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch18_todo_01..06) and Coding Exam (ch18_exam_01..03)
for Real-World Data Wrangling — Missing Data, Dirty Surveys, Joins, GroupBy,
and Time Series. Every verifier lazy-imports pandas/numpy, tests non-mutation
(Pandas 3.x Copy-on-Write clean practices), checks domain edge cases and
anti-leakage boundaries, and lets `NotImplementedError` propagate cleanly for
unattempted starter stubs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


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
        f"{title} — Function Returned None (In-Place vs. Return Trap!)",
        f"Calling {call_str} returned None instead of a Pandas DataFrame.",
        details=[
            "In Pandas 3.x, never rely on `inplace=True` or print() alone — always build a `.copy()`, assign columns directly, and `return out`.",
            f"Verify that {call_str} ends with an explicit `return` statement returning a DataFrame.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Write pure DataFrame transforms: `out = df.copy()` -> transform -> `return out`.",
    )


def _check_dataframe(
    ex_id: str,
    title: str,
    obj: Any,
    call_str: str,
    points_possible: int = 10,
) -> Optional[CheckResult]:
    import pandas as pd

    if obj is None:
        return _none_return_trap(ex_id, title, call_str, points_possible)
    if not isinstance(obj, pd.DataFrame):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Expected a Pandas DataFrame",
            f"{call_str} returned {type(obj).__name__} instead of a pd.DataFrame.",
            details=[
                "Make sure your function returns a Pandas DataFrame (not a Series, dict, or list)."
            ],
            points_earned=0,
            points_possible=points_possible,
        )
    return None


# ---------------------------------------------------------------------------
# GUIDED TO-DO 18.1: clean_mch_survey(df)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch18_todo_01", "To-Do 18.1", fn, "clean_mch_survey", 10)
    if bad is not None:
        return bad

    import numpy as np
    import pandas as pd

    raw = pd.DataFrame(
        {
            "household_id": ["H1", "H2", "H3", "H4", "H5"],
            "residence": [" rural ", "URBAN", "Rural", "urban", "  Rural"],
            "anc_visits": [4, 999, 0, 6, 999],
            "maternal_hb_g_dl": ["11.2", "N/A", "9.8", "", "12.5"],
        }
    )
    snapshot = raw.copy(deep=True)

    try:
        res = fn(raw)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_todo_01",
            False,
            "failed",
            "To-Do 18.1 — Runtime Error in `clean_mch_survey`",
            f"clean_mch_survey(df) raised {type(exc).__name__}: {exc}",
            details=[
                "Use `pd.to_numeric(out['maternal_hb_g_dl'], errors='coerce')` instead of `.astype(float)` so strings like `'N/A'` or `''` become `NaN` safely.",
                "Use `out = df.copy()` and direct column assignment (`out['col'] = ...`).",
            ],
        )

    bad_df = _check_dataframe("ch18_todo_01", "To-Do 18.1", res, "clean_mch_survey(df)", 10)
    if bad_df is not None:
        return bad_df

    errors = []
    if not raw.equals(snapshot):
        errors.append(
            "Your function mutated the caller's input DataFrame! Start with `out = df.copy()`."
        )

    for col in ("residence", "anc_visits", "maternal_hb_g_dl"):
        if col not in res.columns:
            errors.append(f"Missing required column `{col}` in returned DataFrame.")

    if not errors:
        expected_res = ["Rural", "Urban", "Rural", "Urban", "Rural"]
        got_res = list(res["residence"])
        if got_res != expected_res:
            errors.append(
                f"`residence` normalization mismatch: expected {expected_res}, got {got_res}. "
                "Apply `.str.strip().str.title()`."
            )

        anc_vals = res["anc_visits"]
        if not (pd.isna(anc_vals.iloc[1]) and pd.isna(anc_vals.iloc[4])):
            errors.append(
                "Sentinel code `999` in `anc_visits` was not converted to `np.nan`! "
                "Use `out['anc_visits'] = out['anc_visits'].replace(999, np.nan)`."
            )
        valid_anc = [anc_vals.iloc[0], anc_vals.iloc[2], anc_vals.iloc[3]]
        if any(pd.isna(v) for v in valid_anc) or [float(v) for v in valid_anc] != [4.0, 0.0, 6.0]:
            errors.append(
                f"Valid `anc_visits` values [4, 0, 6] were altered unexpectedly: got {valid_anc!r}."
            )

        hb_vals = res["maternal_hb_g_dl"]
        if not pd.api.types.is_numeric_dtype(hb_vals):
            errors.append(
                f"`maternal_hb_g_dl` must be numeric, got dtype `{hb_vals.dtype}`. "
                "Cast with `pd.to_numeric(out['maternal_hb_g_dl'], errors='coerce')`."
            )
        else:
            if not (pd.isna(hb_vals.iloc[1]) and pd.isna(hb_vals.iloc[3])):
                errors.append(
                    "Non-numeric strings `'N/A'` and `''` in `maternal_hb_g_dl` should coerce to `NaN`."
                )
            for idx, exp in [(0, 11.2), (2, 9.8), (4, 12.5)]:
                got_v = hb_vals.iloc[idx]
                if pd.isna(got_v) or not math.isclose(float(got_v), exp, abs_tol=1e-6):
                    errors.append(
                        f"`maternal_hb_g_dl` row {idx}: expected {exp}, got {got_v!r}."
                    )

    if errors:
        return CheckResult(
            "ch18_todo_01",
            False,
            "failed",
            "To-Do 18.1 — Check Survey Cleaning Rules & Non-Mutation",
            "Copy `df`, normalize `residence` with `.str.strip().str.title()`, replace `999` with `np.nan`, and coerce `maternal_hb_g_dl` with `pd.to_numeric(..., errors='coerce')`.",
            details=errors,
        )

    return CheckResult(
        "ch18_todo_01",
        True,
        "passed",
        "To-Do 18.1 Passed — Dirty DHS Survey Cleaned Safely!",
        "Normalized whitespace/casing in `residence`, neutralized sentinel `999` in `anc_visits`, coerced dirty strings in `maternal_hb_g_dl` to numeric/NaN, and preserved the input DataFrame.",
        points_earned=10,
        points_possible=10,
        tip="Always clean categorical casing and sentinel codes (999, -99, 'N/A') BEFORE computing any means or groupby tables!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_todo_01",
        chapter="ch18",
        title="To-Do 18.1: Clean Dirty MCH Survey (`clean_mch_survey`)",
        book_section="Section 18.3.1 (Cleaning Sentinel Codes & Dirty Strings)",
        points=10,
        hints={
            1: "Start with `out = df.copy()`. Normalize `out['residence']` using `.str.strip().str.title()`, replace `999` in `out['anc_visits']` with `np.nan`, and convert `out['maternal_hb_g_dl']` via `pd.to_numeric(..., errors='coerce')`.",
            2: "Blueprint:\n1. `out = df.copy()`\n2. `out['residence'] = out['residence'].str.strip().str.title()`\n3. `out['anc_visits'] = out['anc_visits'].replace(999, np.nan)`\n4. `out['maternal_hb_g_dl'] = pd.to_numeric(out['maternal_hb_g_dl'], errors='coerce')`\n5. `return out`",
            3: "def clean_mch_survey(df):\n    out = df.copy()\n    out['residence'] = out['residence'].str.strip().str.___()\n    out['anc_visits'] = out['anc_visits'].replace(___, np.nan)\n    out['maternal_hb_g_dl'] = pd.to_numeric(\n        out['maternal_hb_g_dl'], errors='___'\n    )\n    return out",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 18.2: audit_missingness_table(df)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch18_todo_02", "To-Do 18.2", fn, "audit_missingness_table", 10
    )
    if bad is not None:
        return bad

    import numpy as np
    import pandas as pd

    sample = pd.DataFrame(
        {
            "z_clean": [1, 2, 3, 4],
            "hb": [11.0, np.nan, 10.5, 12.0],
            "a_clean": ["x", "y", "z", "w"],
            "anc": [4.0, np.nan, 2.0, 5.0],
            "haz": [np.nan, np.nan, -1.2, np.nan],
        }
    )
    snapshot = sample.copy(deep=True)

    try:
        res = fn(sample)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_todo_02",
            False,
            "failed",
            "To-Do 18.2 — Runtime Error in `audit_missingness_table`",
            f"audit_missingness_table(df) raised {type(exc).__name__}: {exc}",
            details=[
                "Compute `counts = df.isna().sum().astype(int)` and `pcts = (counts / len(df) * 100).round(2)`."
            ],
        )

    bad_df = _check_dataframe(
        "ch18_todo_02", "To-Do 18.2", res, "audit_missingness_table(df)", 10
    )
    if bad_df is not None:
        return bad_df

    errors = []
    if not sample.equals(snapshot):
        errors.append("Do not mutate the caller's DataFrame inside `audit_missingness_table`.")

    expected_cols = ["missing_count", "missing_pct"]
    if list(res.columns) != expected_cols:
        errors.append(
            f"Expected columns {expected_cols}, got {list(res.columns)}."
        )

    expected_order = ["haz", "anc", "hb", "a_clean", "z_clean"]
    got_index = list(res.index)
    if got_index != expected_order:
        errors.append(
            f"Expected index order {expected_order} (sorted by `missing_pct` descending, then column name ascending), got {got_index}."
        )

    if not errors:
        if not pd.api.types.is_integer_dtype(res["missing_count"]):
            errors.append(
                f"`missing_count` should have integer dtype, got `{res['missing_count'].dtype}`."
            )
        expected_counts = [3, 1, 1, 0, 0]
        expected_pcts = [75.0, 25.0, 25.0, 0.0, 0.0]
        if list(res["missing_count"]) != expected_counts:
            errors.append(
                f"`missing_count` values: expected {expected_counts}, got {list(res['missing_count'])}."
            )
        for col, exp_p in zip(expected_order, expected_pcts):
            got_p = float(res.loc[col, "missing_pct"])
            if not math.isclose(got_p, exp_p, abs_tol=1e-2):
                errors.append(f"`missing_pct` for `{col}`: expected {exp_p}, got {got_p}.")

    # Also test 2-decimal rounding on 1/3 = 33.33%
    if not errors:
        third_df = pd.DataFrame({"col_b": [1.0, np.nan, 3.0], "col_a": [1, 2, 3]})
        res_third = fn(third_df)
        if not math.isclose(float(res_third.loc["col_b", "missing_pct"]), 33.33, abs_tol=1e-6):
            errors.append(
                f"For 1 missing out of 3 rows, expected `missing_pct` == 33.33 (rounded to 2 decimals), got {res_third.loc['col_b', 'missing_pct']!r}."
            )

    if errors:
        return CheckResult(
            "ch18_todo_02",
            False,
            "failed",
            "To-Do 18.2 — Check Missingness Audit Columns, Rounding & Tie-Breaking Sort",
            "Return a DataFrame indexed by column name with `missing_count` (int) and `missing_pct` (rounded to 2 decimals), sorted by `missing_pct` desc then index asc.",
            details=errors,
        )

    return CheckResult(
        "ch18_todo_02",
        True,
        "passed",
        "To-Do 18.2 Passed — Missingness Audit Table Verified!",
        "Computed exact missing counts, 2-decimal percentages, and deterministic tie-breaking sort (`missing_pct` desc, column name asc).",
        points_earned=10,
        points_possible=10,
        tip="Run a missingness audit table BEFORE and AFTER sentinel replacement (`999` -> `NaN`) to spot hidden missingness!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_todo_02",
        chapter="ch18",
        title="To-Do 18.2: Missingness Audit Table (`audit_missingness_table`)",
        book_section="Section 18.3.1 (MCAR / MAR / MNAR & Missingness Auditing)",
        points=10,
        hints={
            1: "Compute `counts = df.isna().sum().astype(int)` and `pcts = (counts / len(df) * 100).round(2)`, build a DataFrame with columns `missing_count` and `missing_pct`, then sort index ascending before stable-sorting `missing_pct` descending.",
            2: "Blueprint:\n1. `counts = df.isna().sum().astype(int)`\n2. `pcts = (counts / len(df) * 100).round(2)`\n3. `audit = pd.DataFrame({'missing_count': counts, 'missing_pct': pcts})`\n4. `return audit.sort_index().sort_values('missing_pct', ascending=False, kind='mergesort')`",
            3: "def audit_missingness_table(df):\n    counts = df.isna().sum().astype(int)\n    pcts = (counts / len(df) * 100).round(___)\n    audit = pd.DataFrame({\n        'missing_count': counts,\n        'missing_pct': pcts,\n    })\n    return audit.sort_index().sort_values(\n        'missing_pct', ascending=___, kind='mergesort'\n    )",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 18.3: impute_group_median(df, value_col, group_col)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    bad = _check_callable("ch18_todo_03", "To-Do 18.3", fn, "impute_group_median", 10)
    if bad is not None:
        return bad

    import numpy as np
    import pandas as pd

    # Chosen so group medians (10.4 vs 12.6) differ clearly from group means
    # (10.8 vs 12.33) and global median (11.6)!
    sample = pd.DataFrame(
        {
            "clinic_zone": ["Rural", "Rural", "Rural", "Rural", "Urban", "Urban", "Urban", "Urban"],
            "hb_level": [10.0, 10.4, 12.0, np.nan, 11.4, 12.6, np.nan, 13.0],
            "tag": list("abcdefgh"),
        }
    )
    snapshot = sample.copy(deep=True)

    try:
        res = fn(sample, "hb_level", "clinic_zone")
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_todo_03",
            False,
            "failed",
            "To-Do 18.3 — Runtime Error in `impute_group_median`",
            f"impute_group_median(df, 'hb_level', 'clinic_zone') raised {type(exc).__name__}: {exc}",
            details=[
                "Use the parameter names `value_col` and `group_col` dynamically (do NOT hardcode column names).",
                "Pattern: `medians = out.groupby(group_col)[value_col].transform('median')` then `out[value_col] = out[value_col].fillna(medians)`.",
            ],
        )

    bad_df = _check_dataframe(
        "ch18_todo_03", "To-Do 18.3", res, "impute_group_median(df, value_col, group_col)", 10
    )
    if bad_df is not None:
        return bad_df

    errors = []
    if not sample.equals(snapshot):
        errors.append("Input DataFrame was mutated! Start with `out = df.copy()`.")

    if "hb_level" not in res.columns:
        errors.append("Returned DataFrame is missing `value_col` ('hb_level').")
    else:
        rural_fill = res.loc[3, "hb_level"]
        urban_fill = res.loc[6, "hb_level"]
        if pd.isna(rural_fill) or pd.isna(urban_fill):
            errors.append("NaN values remain in `value_col` after imputation.")
        else:
            if math.isclose(float(rural_fill), 11.4, abs_tol=0.15) and math.isclose(float(urban_fill), 11.4, abs_tol=0.15):
                errors.append(
                    f"Both groups received ~{rural_fill:.2f} (the global median!) instead of group medians (Rural=10.4, Urban=12.6). "
                    "Use `.groupby(group_col)[value_col].transform('median')`."
                )
            elif math.isclose(float(rural_fill), 10.8, abs_tol=0.05):
                errors.append(
                    f"Rural NaN got {rural_fill} (the group MEAN) instead of 10.4 (the group MEDIAN). Use `.transform('median')`."
                )
            else:
                if not math.isclose(float(rural_fill), 10.4, abs_tol=1e-6):
                    errors.append(f"Rural imputed value: expected 10.4, got {rural_fill!r}.")
                if not math.isclose(float(urban_fill), 12.6, abs_tol=1e-6):
                    errors.append(f"Urban imputed value: expected 12.6, got {urban_fill!r}.")

        # Verify existing non-null values were not overwritten
        expected_all = [10.0, 10.4, 12.0, 10.4, 11.4, 12.6, 12.6, 13.0]
        got_all = [float(x) for x in res["hb_level"]]
        if len(got_all) != len(expected_all) or any(
            not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got_all, expected_all)
        ):
            errors.append(f"Full `hb_level` column mismatch: expected {expected_all}, got {got_all}.")

    if errors:
        return CheckResult(
            "ch18_todo_03",
            False,
            "failed",
            "To-Do 18.3 — Check Group-Median Transform & Non-Mutation",
            "Copy `df`, compute per-group medians via `.groupby(group_col)[value_col].transform('median')`, and fill only NaNs with `.fillna(...)`.",
            details=errors,
        )

    return CheckResult(
        "ch18_todo_03",
        True,
        "passed",
        "To-Do 18.3 Passed — Group-Median Imputation Verified!",
        "Filled Rural NaN with Rural median (10.4) and Urban NaN with Urban median (12.6) without overwriting observed readings or mutating `df`.",
        points_earned=10,
        points_possible=10,
        tip="`.groupby(...).transform('median')` returns a Series aligned 1-to-1 with the original DataFrame index — ideal for MAR imputation!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_todo_03",
        chapter="ch18",
        title="To-Do 18.3: Group-Median Imputation (`impute_group_median`)",
        book_section="Section 18.3.2 (Handling Missingness with Group-Median Transform)",
        points=10,
        hints={
            1: "Copy `df`, compute group medians aligned to every row via `out.groupby(group_col)[value_col].transform('median')`, and assign `out[value_col] = out[value_col].fillna(medians)`.",
            2: "Blueprint:\n1. `out = df.copy()`\n2. `medians = out.groupby(group_col)[value_col].transform('median')`\n3. `out[value_col] = out[value_col].fillna(medians)`\n4. `return out`",
            3: "def impute_group_median(df, value_col, group_col):\n    out = df.copy()\n    medians = out.groupby(group_col)[value_col].transform('___')\n    out[value_col] = out[value_col].___(medians)\n    return out",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 18.4: safe_merge_patient_labs(patients_df, labs_df)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch18_todo_04", "To-Do 18.4", fn, "safe_merge_patient_labs", 10
    )
    if bad is not None:
        return bad

    import pandas as pd

    patients = pd.DataFrame(
        {
            "patient_id": ["P-01", "P-02", "P-03", "P-04"],
            "district": ["Dhaka", "Sylhet", "Khulna", "Dhaka"],
        }
    )
    labs = pd.DataFrame(
        {
            "patient_id": ["P-01", "P-02", "P-04", "P-99"],
            "hb_g_dl": [11.4, 9.8, 12.1, 10.5],
        }
    )
    p_snap = patients.copy(deep=True)
    l_snap = labs.copy(deep=True)

    try:
        merged = fn(patients, labs)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_todo_04",
            False,
            "failed",
            "To-Do 18.4 — Runtime Error in `safe_merge_patient_labs`",
            f"safe_merge_patient_labs(patients, labs) raised {type(exc).__name__}: {exc}",
            details=[
                "Use `pd.merge(patients_df, labs_df, on='patient_id', how='left', validate='one_to_one')`."
            ],
        )

    bad_df = _check_dataframe(
        "ch18_todo_04", "To-Do 18.4", merged, "safe_merge_patient_labs(patients, labs)", 10
    )
    if bad_df is not None:
        return bad_df

    errors = []
    if not patients.equals(p_snap) or not labs.equals(l_snap):
        errors.append("Do not mutate `patients_df` or `labs_df` inside `safe_merge_patient_labs`.")

    if len(merged) == 3:
        errors.append(
            "Returned only 3 rows (missing unmatched patient `P-03`)! Use `how='left'` instead of `how='inner'`."
        )
    elif len(merged) == 5:
        errors.append(
            "Returned 5 rows (included unmatched lab `P-99`)! Use `how='left'` instead of `how='outer'`."
        )
    elif len(merged) != 4:
        errors.append(f"Expected 4 rows from left join on 4 patients, got {len(merged)}.")

    if not errors:
        ids = list(merged["patient_id"])
        if ids != ["P-01", "P-02", "P-03", "P-04"]:
            errors.append(f"Expected patient_id order ['P-01', 'P-02', 'P-03', 'P-04'], got {ids}.")
        if "hb_g_dl" not in merged.columns or not pd.isna(merged.loc[merged["patient_id"] == "P-03", "hb_g_dl"].iloc[0]):
            errors.append("Unmatched patient `P-03` should have `NaN` for `hb_g_dl`.")

    # Test 1: Duplicate key in labs_df MUST raise pd.errors.MergeError
    dup_labs = pd.DataFrame(
        {
            "patient_id": ["P-01", "P-01", "P-02"],
            "hb_g_dl": [11.4, 11.9, 9.8],
        }
    )
    try:
        _ = fn(patients, dup_labs)
    except pd.errors.MergeError:
        pass
    except Exception as exc:
        errors.append(
            f"Duplicate key in `labs_df` raised {type(exc).__name__} instead of `pd.errors.MergeError`. "
            "Pass `validate='one_to_one'` directly to `pd.merge(...)`."
        )
    else:
        errors.append(
            "Silent row-explosion trap! Duplicate `patient_id` in `labs_df` did NOT raise `pd.errors.MergeError`. "
            "Add `validate='one_to_one'` to `pd.merge(...)`."
        )

    # Test 2: Duplicate key in patients_df MUST also raise MergeError (catches many_to_one)
    dup_patients = pd.DataFrame(
        {
            "patient_id": ["P-01", "P-01"],
            "district": ["Dhaka", "Dhaka"],
        }
    )
    try:
        _ = fn(dup_patients, labs)
    except pd.errors.MergeError:
        pass
    except Exception:
        errors.append("Duplicate key in `patients_df` should raise `pd.errors.MergeError` via `validate='one_to_one'`.")
    else:
        errors.append(
            "Duplicate `patient_id` in `patients_df` was allowed! Use `validate='one_to_one'` (not `many_to_one`)."
        )

    if errors:
        return CheckResult(
            "ch18_todo_04",
            False,
            "failed",
            "To-Do 18.4 — Check Left Join & `validate='one_to_one'` Guard",
            "Join with `pd.merge(patients_df, labs_df, on='patient_id', how='left', validate='one_to_one')`.",
            details=errors,
        )

    return CheckResult(
        "ch18_todo_04",
        True,
        "passed",
        "To-Do 18.4 Passed — Safe 1-to-1 Left Merge Verified!",
        "Preserved all 4 patients (keeping unmatched `P-03` as `NaN`), excluded orphan lab `P-99`, and raised `MergeError` on duplicate keys in either table.",
        points_earned=10,
        points_possible=10,
        tip="Always pass `validate=` to `pd.merge()` in research pipelines — silent row explosions from duplicate IDs have retracted real papers!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_todo_04",
        chapter="ch18",
        title="To-Do 18.4: Guarded 1-to-1 Merge (`safe_merge_patient_labs`)",
        book_section="Section 18.3.3 (Safe Joins & Row-Explosion Defense)",
        points=10,
        hints={
            1: "Return `pd.merge(patients_df, labs_df, on='patient_id', how='left', validate='one_to_one')`.",
            2: "Blueprint:\n1. Call `pd.merge(...)` with `patients_df` on the left and `labs_df` on the right.\n2. Set `on='patient_id'`, `how='left'`, and `validate='one_to_one'`.",
            3: "def safe_merge_patient_labs(patients_df, labs_df):\n    return pd.merge(\n        patients_df,\n        labs_df,\n        on='patient_id',\n        how='___',\n        validate='___',\n    )",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 18.5: groupby_wealth_outcomes(mch_df)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch18_todo_05", "To-Do 18.5", fn, "groupby_wealth_outcomes", 10
    )
    if bad is not None:
        return bad

    import numpy as np
    import pandas as pd

    sample = pd.DataFrame(
        {
            "household_id": ["H1", "H2", "H3", "H4", "H5", "H6"],
            "wealth_quintile": [
                "Poorest",
                "Poorest",
                "Poorest",
                "Richest",
                "Richest",
                "Richest",
            ],
            "child_stunted": [1, 1, 0, 0, 0, 1],
            "maternal_anemia": [1, 0, 1, 0, 0, 0],
            "anc_visits": [2.0, 3.0, np.nan, 5.0, 6.0, 7.0],
        }
    )
    snapshot = sample.copy(deep=True)

    try:
        res = fn(sample)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_todo_05",
            False,
            "failed",
            "To-Do 18.5 — Runtime Error in `groupby_wealth_outcomes`",
            f"groupby_wealth_outcomes(mch_df) raised {type(exc).__name__}: {exc}",
            details=[
                "Use `mch_df.groupby('wealth_quintile').agg(...)` with named aggregations for `n_households`, `stunting_pct`, `anemia_pct`, and `mean_anc`."
            ],
        )

    bad_df = _check_dataframe(
        "ch18_todo_05", "To-Do 18.5", res, "groupby_wealth_outcomes(mch_df)", 10
    )
    if bad_df is not None:
        return bad_df

    errors = []
    if not sample.equals(snapshot):
        errors.append("Do not mutate `mch_df` inside `groupby_wealth_outcomes`.")

    if "wealth_quintile" in res.columns:
        errors.append(
            "Keep `wealth_quintile` as the DataFrame index (use default `groupby('wealth_quintile')`, without `as_index=False`)."
        )
    elif set(res.index) != {"Poorest", "Richest"}:
        errors.append(
            f"Expected index values {{'Poorest', 'Richest'}}, got {set(res.index)}."
        )

    expected_cols = ["n_households", "stunting_pct", "anemia_pct", "mean_anc"]
    missing_cols = [c for c in expected_cols if c not in res.columns]
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}. Found: {list(res.columns)}.")

    if not errors:
        checks = {
            "Poorest": {"n_households": 3, "stunting_pct": 66.7, "anemia_pct": 66.7, "mean_anc": 2.5},
            "Richest": {"n_households": 3, "stunting_pct": 33.3, "anemia_pct": 0.0, "mean_anc": 6.0},
        }
        for q, exp_map in checks.items():
            for col, exp_v in exp_map.items():
                got_v = res.loc[q, col]
                if col == "n_households":
                    if int(got_v) != exp_v:
                        errors.append(f"{q} `{col}`: expected {exp_v}, got {got_v!r}.")
                else:
                    if pd.isna(got_v) or not math.isclose(float(got_v), exp_v, abs_tol=1e-6):
                        extra = ""
                        if col.endswith("_pct") and not pd.isna(got_v) and float(got_v) <= 1.0 and exp_v > 1.0:
                            extra = " (did you forget to multiply mean by 100?)"
                        errors.append(f"{q} `{col}`: expected {exp_v}, got {got_v!r}{extra}.")

        # Check 2-decimal rounding on mean_anc (e.g. (1+2+2)/3 = 1.67)
        round_df = pd.DataFrame(
            {
                "household_id": ["A", "B", "C"],
                "wealth_quintile": ["Middle", "Middle", "Middle"],
                "child_stunted": [0, 0, 1],
                "maternal_anemia": [0, 1, 0],
                "anc_visits": [1.0, 2.0, 2.0],
            }
        )
        r_out = fn(round_df)
        if not math.isclose(float(r_out.loc["Middle", "mean_anc"]), 1.67, abs_tol=1e-6):
            errors.append(
                f"Expected `mean_anc` rounded to 2 decimals (1.67 for [1, 2, 2]), got {r_out.loc['Middle', 'mean_anc']!r}."
            )

    if errors:
        return CheckResult(
            "ch18_todo_05",
            False,
            "failed",
            "To-Do 18.5 — Check Named `.agg()` Metrics & Rounding",
            "Aggregate per `wealth_quintile`: `n_households` (count), `stunting_pct` & `anemia_pct` (`mean * 100`, 1 dec), `mean_anc` (2 dec).",
            details=errors,
        )

    return CheckResult(
        "ch18_todo_05",
        True,
        "passed",
        "To-Do 18.5 Passed — Split-Apply-Combine Wealth Table Verified!",
        "Computed exact household counts, 1-decimal stunting/anemia percentages, and 2-decimal mean ANC visits per wealth quintile.",
        points_earned=10,
        points_possible=10,
        tip="Named `.agg(out_col=('in_col', 'func'))` builds publication-ready Table 1 summaries cleanly in a single readable step!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_todo_05",
        chapter="ch18",
        title="To-Do 18.5: Wealth Outcomes Named Aggregation (`groupby_wealth_outcomes`)",
        book_section="Section 18.3.4 (Split-Apply-Combine with Named .agg())",
        points=10,
        hints={
            1: "Group by `'wealth_quintile'` and use named `.agg(...)` for `n_households`, `stunting_pct`, `anemia_pct`, and `mean_anc`. Multiply prevalence means by `100` and round percentages to 1 decimal, `mean_anc` to 2 decimals.",
            2: "Blueprint:\n1. `out = mch_df.groupby('wealth_quintile').agg(n_households=('child_stunted', 'size'), stunting_pct=('child_stunted', 'mean'), anemia_pct=('maternal_anemia', 'mean'), mean_anc=('anc_visits', 'mean'))`\n2. Scale & round `stunting_pct` and `anemia_pct` (`* 100`, `.round(1)`), and round `mean_anc` to 2 decimals.",
            3: "def groupby_wealth_outcomes(mch_df):\n    out = mch_df.groupby('wealth_quintile').agg(\n        n_households=('child_stunted', 'size'),\n        stunting_pct=('child_stunted', 'mean'),\n        anemia_pct=('maternal_anemia', 'mean'),\n        mean_anc=('anc_visits', 'mean'),\n    )\n    out['stunting_pct'] = (out['stunting_pct'] * 100).round(___)\n    out['anemia_pct'] = (out['anemia_pct'] * 100).round(___)\n    out['mean_anc'] = out['mean_anc'].round(___)\n    return out",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 18.6: add_rolling_dengue_avg(epi_df, window=4)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch18_todo_06", "To-Do 18.6", fn, "add_rolling_dengue_avg", 10
    )
    if bad is not None:
        return bad

    import pandas as pd

    # Intentionally shuffled across two districts & weeks!
    shuffled = pd.DataFrame(
        {
            "district": ["Sylhet", "Dhaka", "Sylhet", "Dhaka", "Dhaka", "Sylhet", "Dhaka", "Dhaka"],
            "year": [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024],
            "epi_week": [2, 3, 1, 1, 5, 3, 2, 4],
            "dengue_cases": [30, 40, 10, 100, 50, 20, 20, 60],
        }
    )
    snapshot = shuffled.copy(deep=True)

    try:
        res4 = fn(shuffled)
        res2 = fn(shuffled, window=2)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_todo_06",
            False,
            "failed",
            "To-Do 18.6 — Runtime Error in `add_rolling_dengue_avg`",
            f"add_rolling_dengue_avg(epi_df) raised {type(exc).__name__}: {exc}",
            details=[
                "Sort first: `out = epi_df.sort_values(['district', 'year', 'epi_week']).copy()`.",
                "Compute per-district rolling mean via `.groupby('district')['dengue_cases'].transform(lambda s: s.rolling(window=window, min_periods=1).mean()).round(2)`.",
            ],
        )

    bad_df = _check_dataframe(
        "ch18_todo_06", "To-Do 18.6", res4, "add_rolling_dengue_avg(epi_df)", 10
    )
    if bad_df is not None:
        return bad_df

    errors = []
    if not shuffled.equals(snapshot):
        errors.append("Input `epi_df` was mutated! Sort into a new `.copy()` without modifying `epi_df`.")

    if "rolling_dengue" not in res4.columns:
        errors.append("Missing required column `rolling_dengue` in returned DataFrame.")
    else:
        expected_keys = [
            ("Dhaka", 2024, 1),
            ("Dhaka", 2024, 2),
            ("Dhaka", 2024, 3),
            ("Dhaka", 2024, 4),
            ("Dhaka", 2024, 5),
            ("Sylhet", 2024, 1),
            ("Sylhet", 2024, 2),
            ("Sylhet", 2024, 3),
        ]
        got_keys = list(zip(res4["district"], res4["year"], res4["epi_week"]))
        if got_keys != expected_keys:
            errors.append(
                f"Rows must be sorted by ['district', 'year', 'epi_week']. Got {got_keys}."
            )
        else:
            # Dhaka cases sorted: [100, 20, 40, 60, 50] -> w=4: [100.0, 60.0, 53.33, 55.0, 42.5]
            # Sylhet cases sorted: [10, 30, 20] -> w=4: [10.0, 20.0, 20.0]
            exp4 = [100.0, 60.0, 53.33, 55.0, 42.5, 10.0, 20.0, 20.0]
            got4 = [float(x) if pd.notna(x) else float("nan") for x in res4["rolling_dengue"]]
            if any(pd.isna(x) for x in res4["rolling_dengue"]):
                errors.append(
                    "Found `NaN` in `rolling_dengue` — make sure you pass `min_periods=1` to `.rolling(...)`!"
                )
            elif not math.isclose(got4[5], 10.0, abs_tol=1e-6):
                errors.append(
                    f"Cross-district leakage detected! Sylhet week 1 got {got4[5]} instead of 10.0. "
                    "Compute `.rolling(...)` INSIDE `.groupby('district')`, never across the whole table!"
                )
            elif any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got4, exp4)):
                errors.append(f"`window=4` rolling averages mismatch: expected {exp4}, got {got4}.")

            # Verify custom window=2 is respected (not hardcoded to 4)
            # Dhaka w=2: [100.0, 60.0, 30.0, 50.0, 55.0]; Sylhet w=2: [10.0, 20.0, 25.0]
            exp2 = [100.0, 60.0, 30.0, 50.0, 55.0, 10.0, 20.0, 25.0]
            got2 = [float(x) if pd.notna(x) else float("nan") for x in res2["rolling_dengue"]]
            if any(pd.isna(x) or not math.isclose(a, b, abs_tol=1e-6) for x, a, b in zip(res2["rolling_dengue"], got2, exp2)):
                errors.append(
                    f"`window=2` test failed (expected {exp2}, got {got2}) — make sure you pass `window=window`, not hardcoded `4`."
                )

    if errors:
        return CheckResult(
            "ch18_todo_06",
            False,
            "failed",
            "To-Do 18.6 — Check Sorting, Per-District Rolling & `min_periods=1`",
            "Sort a copy by `['district', 'year', 'epi_week']` and compute per-district rolling mean with `window=window`, `min_periods=1`, rounded to 2 decimals.",
            details=errors,
        )

    return CheckResult(
        "ch18_todo_06",
        True,
        "passed",
        "To-Do 18.6 Passed — Per-District Rolling Surveillance Average Verified!",
        "Sorted shuffled records chronologically per district, computed `rolling_dengue` for both `window=4` and `window=2` with zero cross-district leakage.",
        points_earned=10,
        points_possible=10,
        tip="Panel time-series rule: ALWAYS sort by `[entity, time]` and run `.rolling()` / `.shift()` inside `.groupby(entity)`!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_todo_06",
        chapter="ch18",
        title="To-Do 18.6: Per-District Rolling Dengue Average (`add_rolling_dengue_avg`)",
        book_section="Section 18.3.5 (Panel Time Series: Sorting, Lags & Rolling Windows)",
        points=10,
        hints={
            1: "Sort into a copy using `epi_df.sort_values(['district', 'year', 'epi_week']).copy()`, then compute `rolling_dengue` using `.groupby('district')['dengue_cases'].transform(lambda s: s.rolling(window=window, min_periods=1).mean()).round(2)`.",
            2: "Blueprint:\n1. `out = epi_df.sort_values(['district', 'year', 'epi_week']).copy()`\n2. `out['rolling_dengue'] = out.groupby('district')['dengue_cases'].transform(lambda s: s.rolling(window=window, min_periods=1).mean()).round(2)`\n3. `return out`",
            3: "def add_rolling_dengue_avg(epi_df, window=4):\n    out = epi_df.sort_values(['district', 'year', 'epi_week']).copy()\n    out['rolling_dengue'] = (\n        out.groupby('district')['dengue_cases']\n        .transform(lambda s: s.rolling(window=___, min_periods=1).mean())\n        .round(2)\n    )\n    return out",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 18.1 (30 pts): build_clean_dhs_analytic_table(mch_df)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch18_exam_01", "Exam Task 18.1", fn, "build_clean_dhs_analytic_table", 30
    )
    if bad is not None:
        return bad

    import numpy as np
    import pandas as pd

    # Carefully engineered fixture:
    # - 'rural' lowercase row has NaN Hb; clean Rural observed Hb = [10.0, 10.6, 11.4] -> median = 10.6
    #   (if learner imputes BEFORE normalizing residence, 'rural' has no observed values or wrong median!)
    # - Urban observed Hb = [12.0, 12.8, 13.0] -> median = 12.8
    # - anc_visits includes 999 -> must become NaN AND anc_4plus MUST be False (catches order bug 999 >= 4!)
    dirty = pd.DataFrame(
        {
            "household_id": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"],
            "residence": ["Rural", " rural ", "Rural", "rural", "Urban", "urban", "URBAN", "Urban"],
            "wealth_quintile": ["Poorest", "Poorest", "Poorer", "Poorer", "Richest", "Richest", "Middle", "Middle"],
            "anc_visits": [4, 999, 3, 5, 6, 999, 0, 4],
            "maternal_hb_g_dl": ["10.0", "10.6", "11.4", "N/A", "12.0", "12.8", np.nan, "13.0"],
            "maternal_anemia": [1, 1, 0, 1, 0, 0, 0, 0],
            "child_stunted": [1, 0, 1, 0, 0, 0, 0, 1],
        }
    )
    snapshot = dirty.copy(deep=True)

    try:
        res = fn(dirty)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_exam_01",
            False,
            "failed",
            "Exam Task 18.1 — Runtime Error in `build_clean_dhs_analytic_table`",
            f"build_clean_dhs_analytic_table(mch_df) raised {type(exc).__name__}: {exc}",
            details=[
                "Follow the exact pipeline order: (1) copy, (2) normalize `residence`, (3) replace `999` -> `np.nan`, (4) coerce `maternal_hb_g_dl` to numeric and impute per-`residence` median, (5) compute `anc_4plus`."
            ],
            points_earned=0,
            points_possible=30,
        )

    bad_df = _check_dataframe(
        "ch18_exam_01", "Exam Task 18.1", res, "build_clean_dhs_analytic_table(mch_df)", 30
    )
    if bad_df is not None:
        return bad_df

    errors = []
    if not dirty.equals(snapshot):
        errors.append("Input `mch_df` was mutated! Always start with `out = mch_df.copy()`.")

    # 1. Residence normalized
    if set(res["residence"]) != {"Rural", "Urban"}:
        errors.append(
            f"`residence` must contain only normalized 'Rural'/'Urban', got {sorted(set(res['residence']))}."
        )

    # 2. anc_visits == 999 -> NaN
    if not (pd.isna(res.loc[1, "anc_visits"]) and pd.isna(res.loc[5, "anc_visits"])):
        errors.append("Sentinel `anc_visits == 999` must be replaced with `np.nan`.")

    # 3. maternal_hb_g_dl numeric & per-residence median imputed (AFTER residence normalization!)
    if not pd.api.types.is_numeric_dtype(res["maternal_hb_g_dl"]):
        errors.append(f"`maternal_hb_g_dl` must be numeric, got `{res['maternal_hb_g_dl'].dtype}`.")
    elif int(res["maternal_hb_g_dl"].isna().sum()) != 0:
        errors.append(
            "`maternal_hb_g_dl` still has NaN values! Normalize `residence` BEFORE imputing per-`residence` median."
        )
    else:
        rural_hb = float(res.loc[3, "maternal_hb_g_dl"])
        urban_hb = float(res.loc[6, "maternal_hb_g_dl"])
        if not math.isclose(rural_hb, 10.6, abs_tol=1e-6):
            errors.append(
                f"Imputed Rural Hb mismatch: expected 10.6 (median of cleaned Rural [10.0, 10.6, 11.4]), got {rural_hb}. "
                "Make sure you normalize `residence` BEFORE computing group medians!"
            )
        if not math.isclose(urban_hb, 12.8, abs_tol=1e-6):
            errors.append(
                f"Imputed Urban Hb mismatch: expected 12.8 (median of cleaned Urban [12.0, 12.8, 13.0]), got {urban_hb}."
            )

    # 4. anc_4plus boolean column (False for NaN / former 999!)
    if "anc_4plus" not in res.columns:
        errors.append("Missing required boolean column `anc_4plus`.")
    else:
        if not pd.api.types.is_bool_dtype(res["anc_4plus"]):
            errors.append(f"`anc_4plus` must have boolean dtype, got `{res['anc_4plus'].dtype}`.")
        expected_anc4 = [True, False, False, True, True, False, False, True]
        got_anc4 = [bool(x) for x in res["anc_4plus"]]
        if got_anc4[1] is True or got_anc4[5] is True:
            errors.append(
                "Order-of-operations bug: `anc_visits == 999` produced `anc_4plus == True`! "
                "Replace `999` with `np.nan` BEFORE computing `anc_4plus = (out['anc_visits'] >= 4).fillna(False).astype(bool)`."
            )
        elif got_anc4 != expected_anc4:
            errors.append(f"`anc_4plus` mismatch: expected {expected_anc4}, got {got_anc4}.")

    if errors:
        return CheckResult(
            "ch18_exam_01",
            False,
            "failed",
            "Exam Task 18.1 — Check DHS Cleaning Pipeline Order & Edge Cases",
            "Clean `residence`, neutralize `999` -> `np.nan`, coerce & impute `maternal_hb_g_dl` by cleaned `residence` median, and derive boolean `anc_4plus`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch18_exam_01",
        True,
        "passed",
        "Exam Task 18.1 Passed (30/30 pts) — Clean DHS Analytic Table Verified!",
        "Executed every cleaning step in the leak-free, bug-free order: normalized `residence`, neutralized `999` before `anc_4plus`, imputed Hb per cleaned residence, and preserved input data.",
        points_earned=30,
        points_possible=30,
        tip="Pipeline order matters: normalizing categories before group imputation and replacing `999` before thresholding prevents two of the most common survey-analysis bugs!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_exam_01",
        chapter="ch18",
        title="Exam 18.1: Build Clean DHS Analytic Table (`build_clean_dhs_analytic_table`)",
        book_section="Chapter 18 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Copy `mch_df`, normalize `residence` first, replace `anc_visits == 999` with `np.nan`, coerce `maternal_hb_g_dl` to numeric and impute missing values with per-`residence` median, then add boolean `anc_4plus` (`False` for NaN).",
            2: "Blueprint:\n1. `out = mch_df.copy()`\n2. `out['residence'] = out['residence'].str.strip().str.title()`\n3. `out['anc_visits'] = out['anc_visits'].replace(999, np.nan)`\n4. `out['maternal_hb_g_dl'] = pd.to_numeric(out['maternal_hb_g_dl'], errors='coerce')`\n5. `med = out.groupby('residence')['maternal_hb_g_dl'].transform('median')`; `out['maternal_hb_g_dl'] = out['maternal_hb_g_dl'].fillna(med)`\n6. `out['anc_4plus'] = (out['anc_visits'] >= 4).fillna(False).astype(bool)`",
            3: "def build_clean_dhs_analytic_table(mch_df):\n    out = mch_df.copy()\n    out['residence'] = out['residence'].str.strip().str.title()\n    out['anc_visits'] = out['anc_visits'].replace(999, np.nan)\n    out['maternal_hb_g_dl'] = pd.to_numeric(\n        out['maternal_hb_g_dl'], errors='coerce'\n    )\n    medians = out.groupby('residence')['maternal_hb_g_dl'].transform('___')\n    out['maternal_hb_g_dl'] = out['maternal_hb_g_dl'].fillna(medians)\n    out['anc_4plus'] = (out['anc_visits'] >= ___).fillna(False).astype(bool)\n    return out",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 18.2 (35 pts): equity_gap_by_residence_wealth(clean_mch_df)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch18_exam_02", "Exam Task 18.2", fn, "equity_gap_by_residence_wealth", 35
    )
    if bad is not None:
        return bad

    import pandas as pd

    clean = pd.DataFrame(
        {
            "household_id": [f"H{i}" for i in range(1, 11)],
            "residence": [
                "Urban", "Urban", "Rural", "Rural", "Rural",
                "Urban", "Urban", "Urban", "Rural", "Rural",
            ],
            "wealth_quintile": [
                "Richest", "Richest", "Poorest", "Poorest", "Poorest",
                "Poorest", "Poorest", "Poorest", "Richest", "Richest",
            ],
            "child_stunted": [0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            "maternal_anemia": [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            "anc_4plus": [True, True, False, False, True, True, False, True, True, True],
        }
    )
    snapshot = clean.copy(deep=True)

    try:
        res = fn(clean)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_exam_02",
            False,
            "failed",
            "Exam Task 18.2 — Runtime Error in `equity_gap_by_residence_wealth`",
            f"equity_gap_by_residence_wealth(clean_mch_df) raised {type(exc).__name__}: {exc}",
            details=[
                "Use `clean_mch_df.groupby(['residence', 'wealth_quintile'], as_index=False).agg(...)` and sort by `['residence', 'wealth_quintile']`."
            ],
            points_earned=0,
            points_possible=35,
        )

    bad_df = _check_dataframe(
        "ch18_exam_02", "Exam Task 18.2", res, "equity_gap_by_residence_wealth(clean_mch_df)", 35
    )
    if bad_df is not None:
        return bad_df

    errors = []
    if not clean.equals(snapshot):
        errors.append("Input `clean_mch_df` was mutated!")

    if "residence" not in res.columns or "wealth_quintile" not in res.columns:
        errors.append(
            "`residence` and `wealth_quintile` must be regular columns (pass `as_index=False` to `.groupby(...)`)."
        )

    expected_cols = [
        "residence",
        "wealth_quintile",
        "n_households",
        "stunting_pct",
        "anemia_pct",
        "anc4_pct",
    ]
    missing = [c for c in expected_cols if c not in res.columns]
    if missing:
        errors.append(f"Missing columns: {missing}. Found: {list(res.columns)}.")

    if not errors:
        expected_rows = [
            ("Rural", "Poorest", 3, 66.7, 100.0, 33.3),
            ("Rural", "Richest", 2, 0.0, 0.0, 100.0),
            ("Urban", "Poorest", 3, 33.3, 66.7, 66.7),
            ("Urban", "Richest", 2, 0.0, 50.0, 100.0),
        ]
        got_pairs = list(zip(res["residence"], res["wealth_quintile"]))
        exp_pairs = [(r[0], r[1]) for r in expected_rows]
        if got_pairs != exp_pairs:
            errors.append(
                f"Rows must be sorted by ['residence', 'wealth_quintile'] ascending. Expected {exp_pairs}, got {got_pairs}."
            )
        else:
            for idx, (r_name, w_name, exp_n, exp_st, exp_an, exp_a4) in enumerate(expected_rows):
                row = res.iloc[idx]
                if int(row["n_households"]) != exp_n:
                    errors.append(f"{r_name}/{w_name} `n_households`: expected {exp_n}, got {row['n_households']!r}.")
                for col, exp_v in [
                    ("stunting_pct", exp_st),
                    ("anemia_pct", exp_an),
                    ("anc4_pct", exp_a4),
                ]:
                    got_v = float(row[col])
                    if not math.isclose(got_v, exp_v, abs_tol=1e-6):
                        errors.append(
                            f"{r_name}/{w_name} `{col}`: expected {exp_v}, got {got_v!r}."
                        )

    if errors:
        return CheckResult(
            "ch18_exam_02",
            False,
            "failed",
            "Exam Task 18.2 — Check Multi-Key GroupBy, Rounding & Sorting",
            "Group with `as_index=False`, compute `n_households`, `stunting_pct`, `anemia_pct`, `anc4_pct` (1 decimal), and sort by `['residence', 'wealth_quintile']`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch18_exam_02",
        True,
        "passed",
        "Exam Task 18.2 Passed (35/35 pts) — Health Equity Table Verified!",
        "Computed tidy flat-column stratified outcomes across residence × wealth quintile with exact 1-decimal percentages and deterministic sorting.",
        points_earned=35,
        points_possible=35,
        tip="Using `as_index=False` keeps stratification keys as tidy columns — immediately ready for Seaborn/Matplotlib faceted plots or manuscript tables!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_exam_02",
        chapter="ch18",
        title="Exam 18.2: Stratified Health Equity Table (`equity_gap_by_residence_wealth`)",
        book_section="Chapter 18 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Use `clean_mch_df.groupby(['residence', 'wealth_quintile'], as_index=False).agg(...)` to compute `n_households`, `stunting_pct`, `anemia_pct`, and `anc4_pct`, multiply the three proportions by `100` rounded to 1 decimal, and sort by `['residence', 'wealth_quintile']`.",
            2: "Blueprint:\n1. `out = clean_mch_df.groupby(['residence', 'wealth_quintile'], as_index=False).agg(n_households=('child_stunted', 'size'), stunting_pct=('child_stunted', 'mean'), anemia_pct=('maternal_anemia', 'mean'), anc4_pct=('anc_4plus', 'mean'))`\n2. For each `_pct` column: `out[col] = (out[col] * 100).round(1)`\n3. `return out.sort_values(['residence', 'wealth_quintile']).reset_index(drop=True)`",
            3: "def equity_gap_by_residence_wealth(clean_mch_df):\n    out = clean_mch_df.groupby(\n        ['residence', 'wealth_quintile'], as_index=___\n    ).agg(\n        n_households=('child_stunted', 'size'),\n        stunting_pct=('child_stunted', 'mean'),\n        anemia_pct=('maternal_anemia', 'mean'),\n        anc4_pct=('anc_4plus', 'mean'),\n    )\n    for col in ('stunting_pct', 'anemia_pct', 'anc4_pct'):\n        out[col] = (out[col] * 100).round(___)\n    return out.sort_values(['residence', 'wealth_quintile']).reset_index(drop=True)",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 18.3 (35 pts): prepare_epi_time_features(epi_df)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch18_exam_03", "Exam Task 18.3", fn, "prepare_epi_time_features", 35
    )
    if bad is not None:
        return bad

    import pandas as pd

    # Shuffled multi-district, cross-year fixture (2024 wk 51-52 -> 2025 wk 1-2)
    shuffled = pd.DataFrame(
        {
            "district": ["Dhaka", "Chattogram", "Dhaka", "Chattogram", "Dhaka", "Chattogram", "Dhaka", "Chattogram"],
            "year": [2025, 2024, 2024, 2025, 2024, 2024, 2025, 2025],
            "epi_week": [1, 52, 51, 2, 52, 51, 2, 1],
            "population": [500_000, 200_000, 500_000, 200_000, 500_000, 200_000, 500_000, 200_000],
            "dengue_cases": [40, 20, 10, 60, 30, 10, 80, 30],
        }
    )
    snapshot = shuffled.copy(deep=True)

    try:
        res = fn(shuffled)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch18_exam_03",
            False,
            "failed",
            "Exam Task 18.3 — Runtime Error in `prepare_epi_time_features`",
            f"prepare_epi_time_features(epi_df) raised {type(exc).__name__}: {exc}",
            details=[
                "Sort into a copy by `['district', 'year', 'epi_week']`, then compute `incidence_per_100k`, `prev_week_cases` via `.groupby('district')['dengue_cases'].shift(1)`, and `rolling_4w_cases` via per-district rolling mean (`window=4, min_periods=1`)."
            ],
            points_earned=0,
            points_possible=35,
        )

    bad_df = _check_dataframe(
        "ch18_exam_03", "Exam Task 18.3", res, "prepare_epi_time_features(epi_df)", 35
    )
    if bad_df is not None:
        return bad_df

    errors = []
    if not shuffled.equals(snapshot):
        errors.append("Input `epi_df` was mutated! Always sort into a new `.copy()`.")

    required = ["incidence_per_100k", "prev_week_cases", "rolling_4w_cases"]
    missing = [c for c in required if c not in res.columns]
    if missing:
        errors.append(f"Missing required feature columns: {missing}.")

    expected_order = [
        ("Chattogram", 2024, 51),
        ("Chattogram", 2024, 52),
        ("Chattogram", 2025, 1),
        ("Chattogram", 2025, 2),
        ("Dhaka", 2024, 51),
        ("Dhaka", 2024, 52),
        ("Dhaka", 2025, 1),
        ("Dhaka", 2025, 2),
    ]
    got_order = list(zip(res["district"], res["year"], res["epi_week"]))
    if got_order != expected_order:
        errors.append(
            f"Output must be sorted by ['district', 'year', 'epi_week']. Got {got_order}."
        )

    if not errors:
        # Chattogram sorted cases: [10, 20, 30, 60], pop=200_000
        #   inc: [5.0, 10.0, 15.0, 30.0]
        #   lag: [NaN, 10.0, 20.0, 30.0]
        #   roll4: [10.0, 15.0, 20.0, 30.0]
        # Dhaka sorted cases: [10, 30, 40, 80], pop=500_000
        #   inc: [2.0, 6.0, 8.0, 16.0]
        #   lag: [NaN, 10.0, 30.0, 40.0]
        #   roll4: [10.0, 20.0, 26.67, 40.0]
        exp_inc = [5.0, 10.0, 15.0, 30.0, 2.0, 6.0, 8.0, 16.0]
        exp_lag = [None, 10.0, 20.0, 30.0, None, 10.0, 30.0, 40.0]
        exp_roll = [10.0, 15.0, 20.0, 30.0, 10.0, 20.0, 26.67, 40.0]

        got_inc = [float(x) for x in res["incidence_per_100k"]]
        if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got_inc, exp_inc)):
            errors.append(f"`incidence_per_100k` mismatch: expected {exp_inc}, got {got_inc}.")

        # Explicit leakage check at Dhaka row 0 (index 4 in sorted order)
        dhaka_first_lag = res["prev_week_cases"].iloc[4]
        if pd.notna(dhaka_first_lag):
            errors.append(
                f"Cross-district lag leakage detected! Dhaka's first week got `prev_week_cases={dhaka_first_lag}` "
                "(leaked from Chattogram's last week!) instead of `NaN`. Use `out.groupby('district')['dengue_cases'].shift(1)`."
            )
        else:
            for idx, exp_v in enumerate(exp_lag):
                got_v = res["prev_week_cases"].iloc[idx]
                if exp_v is None:
                    if pd.notna(got_v):
                        errors.append(f"Row {idx} `prev_week_cases`: expected NaN, got {got_v!r}.")
                elif pd.isna(got_v) or not math.isclose(float(got_v), exp_v, abs_tol=1e-6):
                    errors.append(f"Row {idx} `prev_week_cases`: expected {exp_v}, got {got_v!r}.")

        dhaka_first_roll = res["rolling_4w_cases"].iloc[4]
        if pd.isna(dhaka_first_roll) or not math.isclose(float(dhaka_first_roll), 10.0, abs_tol=1e-6):
            errors.append(
                f"Cross-district rolling leakage detected! Dhaka's first week got `rolling_4w_cases={dhaka_first_roll}` "
                "instead of 10.0. Compute `.rolling(window=4, min_periods=1).mean()` inside `.groupby('district')`."
            )
        else:
            got_roll = [float(x) for x in res["rolling_4w_cases"]]
            if any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(got_roll, exp_roll)):
                errors.append(f"`rolling_4w_cases` mismatch: expected {exp_roll}, got {got_roll}.")

    if errors:
        return CheckResult(
            "ch18_exam_03",
            False,
            "failed",
            "Exam Task 18.3 — Check Panel Sorting, Incidence & Zero-Leakage Lag/Rolling",
            "Sort by `['district', 'year', 'epi_week']`, compute `incidence_per_100k` (2 dec), per-district `prev_week_cases` (`shift(1)`), and per-district `rolling_4w_cases` (`window=4, min_periods=1`, 2 dec).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch18_exam_03",
        True,
        "passed",
        "Exam Task 18.3 Passed (35/35 pts) — Leak-Free Epidemiological Time Features Verified!",
        "Sorted multi-year panel records, normalized incidence per 100k population, and verified zero cross-district leakage on both 1-week lag and 4-week rolling means.",
        points_earned=35,
        points_possible=35,
        tip="Panel-grouped `.shift(1)` and `.rolling()` are the gold standard for epidemiological forecasting features — zero boundary leakage across districts!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch18_exam_03",
        chapter="ch18",
        title="Exam 18.3: Leak-Free Epi Time Features (`prepare_epi_time_features`)",
        book_section="Chapter 18 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Sort into a copy by `['district', 'year', 'epi_week']`. Add `incidence_per_100k = (out['dengue_cases'] / out['population'] * 100_000).round(2)`, `prev_week_cases = out.groupby('district')['dengue_cases'].shift(1)`, and `rolling_4w_cases` using per-district `.rolling(window=4, min_periods=1).mean().round(2)`.",
            2: "Blueprint:\n1. `out = epi_df.sort_values(['district', 'year', 'epi_week']).copy()`\n2. `out['incidence_per_100k'] = (out['dengue_cases'] / out['population'] * 100_000).round(2)`\n3. `out['prev_week_cases'] = out.groupby('district')['dengue_cases'].shift(1)`\n4. `out['rolling_4w_cases'] = out.groupby('district')['dengue_cases'].transform(lambda s: s.rolling(window=4, min_periods=1).mean()).round(2)`\n5. `return out`",
            3: "def prepare_epi_time_features(epi_df):\n    out = epi_df.sort_values(['district', 'year', 'epi_week']).copy()\n    out['incidence_per_100k'] = (\n        out['dengue_cases'] / out['population'] * 100_000\n    ).round(2)\n    out['prev_week_cases'] = out.groupby('district')['dengue_cases'].___(1)\n    out['rolling_4w_cases'] = (\n        out.groupby('district')['dengue_cases']\n        .transform(lambda s: s.rolling(window=4, min_periods=1).mean())\n        .round(2)\n    )\n    return out",
        },
        verifier=_verify_exam_03,
    )
)
