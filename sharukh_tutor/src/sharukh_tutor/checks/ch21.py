"""Chapter 21 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch21_todo_01..06) and the Chapter 21 Coding Exam
(ch21_exam_01..03) for The Machine Learning Paradigm — Framing Problems,
Data Leakage, Splits, and Bias-Variance Tradeoff.
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn inside the check
function, validates return structures, non-mutation, zero-leakage invariants,
and rounding, while letting NotImplementedError propagate cleanly for stubs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _is_bool_like(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (bool, np.bool_))


def _is_int_like(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, np.integer)) and not isinstance(val, bool)


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
        f"Calling {call_str} returned None instead of the expected ML result.",
        details=[
            "Remember: print() only displays text; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reusable ML splitting and auditing helpers always return structured tuples, dicts, or DataFrames.",
    )


def _check_dict_keys(
    ex_id: str,
    title: str,
    got: Any,
    expected_keys: list[str],
    points_possible: int = 10,
) -> Optional[CheckResult]:
    if not isinstance(got, dict):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Expected a Dictionary",
            f"Expected a dict with keys {expected_keys}, got {type(got).__name__} ({got!r}).",
            points_earned=0,
            points_possible=points_possible,
        )
    missing = [k for k in expected_keys if k not in got]
    extra = [k for k in got if k not in expected_keys]
    if missing or extra:
        details = []
        if missing:
            details.append(f"Missing key(s): {missing}")
        if extra:
            details.append(f"Unexpected key(s): {extra}")
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Check Dictionary Keys",
            f"Expected exact keys {expected_keys}, got {list(got.keys())}.",
            details=details,
            points_earned=0,
            points_possible=points_possible,
        )
    return None


# ---------------------------------------------------------------------------
# GUIDED TO-DO 21.1: stratified_three_way_split(X, y, test_size=0.2, val_size=0.2, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable("ch21_todo_01", "To-Do 21.1", fn, "stratified_three_way_split", 10)
    if bad is not None:
        return bad

    mch = load_data("mch_survey_sample", as_pandas=True).copy()
    mch["anc_visits"] = mch["anc_visits"].replace(999, np.nan)
    clean = mch[["maternal_age_years", "anc_visits", "child_haz", "maternal_anemia"]].dropna().reset_index(drop=True)
    X_mch = clean[["maternal_age_years", "anc_visits", "child_haz"]].copy()
    y_mch = clean["maternal_anemia"].copy()

    rng = np.random.default_rng(7)
    X_np = rng.normal(0, 1, size=(200, 4))
    y_np = np.array([0] * 140 + [1] * 60)

    cases = [
        (X_mch, y_mch, 0.2, 0.2, 42, "MCH 60/20/20 DataFrame split"),
        (X_np, y_np, 0.15, 0.15, 7, "NumPy 70/15/15 custom-ratio split"),
    ]
    errors = []

    for X_in, y_in, test_sz, val_sz, seed, desc in cases:
        X_copy = X_in.copy()
        y_copy = y_in.copy()
        try:
            got = fn(X_copy, y_copy, test_size=test_sz, val_size=val_sz, seed=seed)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch21_todo_01",
                False,
                "failed",
                "To-Do 21.1 — Check Function Signature",
                f"Failed on {desc} with TypeError: {exc}",
                details=[
                    "Define `def stratified_three_way_split(X, y, test_size=0.2, val_size=0.2, seed=42):`."
                ],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch21_todo_01",
                False,
                "failed",
                "To-Do 21.1 — Runtime Error in `stratified_three_way_split`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch21_todo_01",
                "To-Do 21.1",
                "stratified_three_way_split(X, y)",
                10,
            )

        if not isinstance(got, tuple) or len(got) != 6:
            return CheckResult(
                "ch21_todo_01",
                False,
                "failed",
                "To-Do 21.1 — Return a 6-Tuple `(X_train, X_val, X_test, y_train, y_val, y_test)`",
                f"Expected a 6-tuple, got {type(got).__name__} of length {len(got) if hasattr(got, '__len__') else 'N/A'}.",
                points_earned=0,
                points_possible=10,
            )

        X_tr, X_va, X_te, y_tr, y_va, y_te = got
        n_total = len(X_in)
        if len(X_tr) != len(y_tr) or len(X_va) != len(y_va) or len(X_te) != len(y_te):
            errors.append(f"[{desc}] Feature/target lengths do not match across train/val/test.")
            continue

        if len(X_tr) + len(X_va) + len(X_te) != n_total:
            errors.append(
                f"[{desc}] Row count mismatch: {len(X_tr)} + {len(X_va)} + {len(X_te)} != {n_total}."
            )
            continue

        exp_test_n = round(test_sz * n_total)
        exp_val_n = round(val_sz * n_total)
        if abs(len(X_te) - exp_test_n) > 1:
            errors.append(f"[{desc}] Test split size expected ~{exp_test_n}, got {len(X_te)}.")
        if abs(len(X_va) - exp_val_n) > 1:
            extra = ""
            unadj = round(val_sz * (1.0 - test_sz) * n_total)
            if abs(len(X_va) - unadj) <= 1:
                extra = " (Scale the second split fraction by `val_size / (1.0 - test_size)`!)"
            errors.append(f"[{desc}] Validation split size expected ~{exp_val_n}, got {len(X_va)}.{extra}")

        if isinstance(X_in, pd.DataFrame) and isinstance(X_tr, pd.DataFrame):
            idx_tr = set(X_tr.index)
            idx_va = set(X_va.index)
            idx_te = set(X_te.index)
            if (idx_tr & idx_va) or (idx_tr & idx_te) or (idx_va & idx_te):
                errors.append(f"[{desc}] Overlapping row indices detected across train/val/test!")
            if not X_copy.equals(X_in):
                errors.append(f"[{desc}] Input DataFrame `X` was mutated!")

        overall_rate = float(np.mean(y_in))
        for split_name, y_part in [("train", y_tr), ("val", y_va), ("test", y_te)]:
            part_rate = float(np.mean(y_part))
            if abs(part_rate - overall_rate) > 0.03:
                errors.append(
                    f"[{desc}] {split_name} positive rate {part_rate:.4f} drifted from "
                    f"overall {overall_rate:.4f} — pass `stratify=...` to BOTH splits!"
                )

    if errors:
        return CheckResult(
            "ch21_todo_01",
            False,
            "failed",
            "To-Do 21.1 — Check Two-Stage Stratified Split Proportions & Stratification",
            "Split off `test_size` first with `stratify=y`, then split the remainder with `test_size=val_size / (1 - test_size)` and `stratify=y_temp`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch21_todo_01",
        True,
        "passed",
        "To-Do 21.1 Passed — Stratified 3-Way Train/Val/Test Split Verified!",
        "Preserved exact row counts, disjoint partitions, custom ratios, and class balance across both DataFrame and NumPy inputs.",
        tip="Always lock the test set away before tuning hyperparameters on the validation split.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_todo_01",
        chapter="ch21",
        title="To-Do 21.1: Stratified 3-Way Split (stratified_three_way_split)",
        book_section="Section 21.3.2 (Stratified 3-Way Train / Validation / Test Split)",
        points=10,
        hints={
            1: "Call `train_test_split` twice: first carve off `test_size` using `stratify=y, random_state=seed`, then split `(X_temp, y_temp)` into train and val.",
            2: "To make validation equal `val_size` of the *original* dataset, set the second split's fraction to `val_rel = val_size / (1.0 - test_size)` and pass `stratify=y_temp`.",
            3: "def stratified_three_way_split(X, y, test_size=0.2, val_size=0.2, seed=42):\n    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)\n    val_rel = val_size / (1.0 - ___)\n    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=seed)\n    return (X_train, X_val, X_test, y_train, y_val, y_test)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 21.2: group_cluster_split(df, group_col, test_size=0.2, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable("ch21_todo_02", "To-Do 21.2", fn, "group_cluster_split", 10)
    if bad is not None:
        return bad

    df_mch = load_data("mch_survey_sample", as_pandas=True)
    df_patients = pd.DataFrame(
        {
            "patient_id": [f"P{i:02d}" for i in range(1, 21) for _ in range(4)],
            "visit_no": [1, 2, 3, 4] * 20,
            "systolic_bp": [118 + (i % 15) for i in range(80)],
        }
    )

    cases = [
        (df_mch, "cluster_id", 0.2, 42, "DHS 50-cluster MCH survey"),
        (df_patients, "patient_id", 0.25, 99, "20-patient longitudinal visits"),
    ]
    errors = []

    for df_in, gcol, test_sz, seed, desc in cases:
        df_copy = df_in.copy(deep=True)
        try:
            got = fn(df_copy, gcol, test_size=test_sz, seed=seed)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch21_todo_02",
                False,
                "failed",
                "To-Do 21.2 — Runtime Error in `group_cluster_split`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch21_todo_02",
                "To-Do 21.2",
                f"group_cluster_split(df, {gcol!r})",
                10,
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch21_todo_02",
                False,
                "failed",
                "To-Do 21.2 — Return a 2-Tuple `(train_df, test_df)`",
                f"Expected `(train_df, test_df)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        tr_df, te_df = got
        if not isinstance(tr_df, pd.DataFrame) or not isinstance(te_df, pd.DataFrame):
            errors.append(f"[{desc}] Both elements must be Pandas DataFrames.")
            continue

        if not df_copy.equals(df_in):
            errors.append(f"[{desc}] Input DataFrame was mutated!")

        if len(tr_df) + len(te_df) != len(df_in):
            errors.append(f"[{desc}] Total row count changed: {len(tr_df)} + {len(te_df)} != {len(df_in)}.")

        tr_groups = set(tr_df[gcol])
        te_groups = set(te_df[gcol])
        overlap = tr_groups & te_groups
        if overlap:
            errors.append(
                f"[{desc}] Group leakage detected! {len(overlap)} `{gcol}` values appear in BOTH train and test "
                "(use `GroupShuffleSplit`, not `train_test_split`!)."
            )

        n_groups = df_in[gcol].nunique()
        exp_te_groups = round(test_sz * n_groups)
        if abs(len(te_groups) - exp_te_groups) > 1:
            errors.append(
                f"[{desc}] Expected ~{exp_te_groups} test groups ({int(test_sz * 100)}% of {n_groups}), "
                f"got {len(te_groups)}."
            )

    if errors:
        return CheckResult(
            "ch21_todo_02",
            False,
            "failed",
            "To-Do 21.2 — Check `GroupShuffleSplit` & Zero Group Overlap",
            "Instantiate `GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)` and split on `groups=df[group_col]`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch21_todo_02",
        True,
        "passed",
        "To-Do 21.2 Passed — Zero-Overlap Group Cluster Split Verified!",
        "Held out entire DHS clusters (10/50 clusters) and longitudinal patients (5/20 patients) with zero group leakage.",
        tip="Whenever observations share a cluster, hospital, or patient ID, group splitting prevents memorizing cluster idiosyncrasies.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_todo_02",
        chapter="ch21",
        title="To-Do 21.2: DHS Cluster Group Split (group_cluster_split)",
        book_section="Section 21.3.3 (Preventing Cluster Leakage with GroupShuffleSplit)",
        points=10,
        hints={
            1: "Create `gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)` and get indices via `train_idx, test_idx = next(gss.split(df, groups=df[group_col]))`.",
            2: "Slice non-mutating copies with `df.iloc[train_idx].reset_index(drop=True).copy()` and `df.iloc[test_idx].reset_index(drop=True).copy()`.",
            3: "def group_cluster_split(df, group_col, test_size=0.2, seed=42):\n    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)\n    tr_idx, te_idx = next(gss.split(df, groups=df[___]))\n    return df.iloc[tr_idx].reset_index(drop=True).copy(), df.iloc[te_idx].reset_index(drop=True).copy()",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 21.3: temporal_outbreak_split(epi_df, cutoff_year=2025)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable("ch21_todo_03", "To-Do 21.3", fn, "temporal_outbreak_split", 10)
    if bad is not None:
        return bad

    df_epi = load_data("epi_surveillance", as_pandas=True)
    df_multi = pd.DataFrame(
        {
            "year": [2025, 2022, 2024, 2021, 2023, 2024, 2022, 2025],
            "epi_week": [10, 5, 12, 3, 20, 44, 18, 30],
            "dengue_cases": [120, 15, 88, 9, 45, 92, 20, 140],
        },
        index=[100, 101, 102, 103, 104, 105, 106, 107],
    )

    cases = [
        (df_epi, 2025, "epi_surveillance (cutoff_year=2025)"),
        (df_multi, 2024, "multi-year shuffled index (cutoff_year=2024)"),
    ]
    errors = []

    for df_in, cutoff, desc in cases:
        df_copy = df_in.copy(deep=True)
        try:
            got = fn(df_copy, cutoff_year=cutoff)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch21_todo_03",
                False,
                "failed",
                "To-Do 21.3 — Runtime Error in `temporal_outbreak_split`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch21_todo_03",
                "To-Do 21.3",
                f"temporal_outbreak_split(epi_df, cutoff_year={cutoff})",
                10,
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch21_todo_03",
                False,
                "failed",
                "To-Do 21.3 — Return a 2-Tuple `(train_df, test_df)`",
                f"Expected `(train_df, test_df)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        tr_df, te_df = got
        if not isinstance(tr_df, pd.DataFrame) or not isinstance(te_df, pd.DataFrame):
            errors.append(f"[{desc}] Both returned items must be Pandas DataFrames.")
            continue

        if not df_copy.equals(df_in):
            errors.append(f"[{desc}] Input DataFrame was mutated!")

        exp_tr = df_in.loc[df_in["year"] < cutoff].reset_index(drop=True)
        exp_te = df_in.loc[df_in["year"] >= cutoff].reset_index(drop=True)

        if len(tr_df) != len(exp_tr) or len(te_df) != len(exp_te):
            extra = ""
            if len(tr_df) > 0 and tr_df["year"].max() == cutoff:
                extra = " (Use strictly `< cutoff_year` for train and `>= cutoff_year` for test!)"
            errors.append(
                f"[{desc}] Split sizes expected ({len(exp_tr)}, {len(exp_te)}), "
                f"got ({len(tr_df)}, {len(te_df)}).{extra}"
            )
            continue

        if list(tr_df.index) != list(range(len(tr_df))) or list(te_df.index) != list(range(len(te_df))):
            errors.append(
                f"[{desc}] Index was not cleanly reset to 0..n-1 — call `.reset_index(drop=True)`!"
            )

        if tr_df["year"].max() >= cutoff or te_df["year"].min() < cutoff:
            errors.append(f"[{desc}] Temporal boundary violated: train must be < {cutoff}, test >= {cutoff}.")

    if errors:
        return CheckResult(
            "ch21_todo_03",
            False,
            "failed",
            "To-Do 21.3 — Check Strict Chronological Boundary & Reset Index",
            "Filter `year < cutoff_year` for training and `year >= cutoff_year` for testing on clean `.copy()` frames with `.reset_index(drop=True)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch21_todo_03",
        True,
        "passed",
        "To-Do 21.3 Passed — Chronological Outbreak Split Verified!",
        "Partitioned epi_surveillance cleanly into 2024 training (208 weeks) and 2025 future testing (208 weeks) with reset indices.",
        tip="Never randomly shuffle time-series surveillance data — future outbreak weeks must never leak into training.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_todo_03",
        chapter="ch21",
        title="To-Do 21.3: Chronological Temporal Split (temporal_outbreak_split)",
        book_section="Section 21.3.4 (Chronological Splitting for Outbreak Surveillance)",
        points=10,
        hints={
            1: "Use boolean masks: `train_mask = epi_df['year'] < cutoff_year` and `test_mask = epi_df['year'] >= cutoff_year`.",
            2: "Return `(epi_df.loc[train_mask].reset_index(drop=True).copy(), epi_df.loc[test_mask].reset_index(drop=True).copy())` without mutating `epi_df`.",
            3: "def temporal_outbreak_split(epi_df, cutoff_year=2025):\n    train_df = epi_df.loc[epi_df['year'] < ___].reset_index(drop=True).copy()\n    test_df = epi_df.loc[epi_df['year'] >= cutoff_year].reset_index(drop=True).copy()\n    return train_df, test_df",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 21.4: leak_free_standardize(X_train, X_test)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable("ch21_todo_04", "To-Do 21.4", fn, "leak_free_standardize", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(123)
    X_tr_shift = rng.normal(loc=10.0, scale=2.0, size=(50, 3))
    X_te_shift = rng.normal(loc=50.0, scale=5.0, size=(20, 3))

    X_tr_df = pd.DataFrame(X_tr_shift, columns=["age", "anc", "haz"])
    X_te_df = pd.DataFrame(X_te_shift, columns=["age", "anc", "haz"])

    cases = [
        (X_tr_shift, X_te_shift, "shifted NumPy arrays (train mean=10, test mean=50)"),
        (X_tr_df, X_te_df, "shifted Pandas DataFrames"),
    ]
    errors = []

    for X_tr_in, X_te_in, desc in cases:
        X_tr_copy = X_tr_in.copy()
        X_te_copy = X_te_in.copy()
        try:
            got = fn(X_tr_copy, X_te_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch21_todo_04",
                False,
                "failed",
                "To-Do 21.4 — Runtime Error in `leak_free_standardize`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch21_todo_04",
                "To-Do 21.4",
                "leak_free_standardize(X_train, X_test)",
                10,
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch21_todo_04",
                False,
                "failed",
                "To-Do 21.4 — Return a 2-Tuple `(X_train_s, X_test_s)`",
                f"Expected `(X_train_s, X_test_s)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        tr_s, te_s = got
        tr_arr = np.asarray(tr_s, dtype=float)
        te_arr = np.asarray(te_s, dtype=float)

        if tr_arr.shape != X_tr_in.shape or te_arr.shape != X_te_in.shape:
            errors.append(f"[{desc}] Output shapes {tr_arr.shape}, {te_arr.shape} do not match inputs.")
            continue

        ref_scaler = StandardScaler().fit(X_tr_in)
        exp_tr = ref_scaler.transform(X_tr_in)
        exp_te = ref_scaler.transform(X_te_in)

        if not np.allclose(tr_arr, exp_tr, atol=1e-4):
            errors.append(
                f"[{desc}] `X_train_s` does not match `StandardScaler().fit_transform(X_train)` "
                "(make sure you fit ONLY on `X_train`, not on combined train+test!)."
            )
        if not np.allclose(te_arr, exp_te, atol=1e-4):
            extra = ""
            if np.allclose(te_arr.mean(axis=0), 0.0, atol=1e-3):
                extra = " (CRITICAL LEAKAGE BUG: you called `.fit_transform(X_test)` instead of `.transform(X_test)`!)"
            errors.append(f"[{desc}] `X_test_s` does not match `scaler.transform(X_test)`.{extra}")

    if errors:
        return CheckResult(
            "ch21_todo_04",
            False,
            "failed",
            "To-Do 21.4 — Check Fit-on-Train-Only `StandardScaler` Boundary",
            "Call `scaler.fit_transform(X_train)` and `scaler.transform(X_test)` — NEVER call `.fit()` or `.fit_transform()` on `X_test`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch21_todo_04",
        True,
        "passed",
        "To-Do 21.4 Passed — Leak-Free Train-Only Standardization Verified!",
        "Confirmed mean/std were learned strictly from `X_train` and applied unchanged to shifted `X_test` inputs.",
        tip="Remember the Golden Rule of Preprocessing: `.fit()` touches ONLY training data; validation and test sets get `.transform()` only.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_todo_04",
        chapter="ch21",
        title="To-Do 21.4: Leak-Free Standardization (leak_free_standardize)",
        book_section="Section 21.3.5 (Preprocessing Leakage vs. Fit-on-Train-Only Scaling)",
        points=10,
        hints={
            1: "Instantiate `scaler = StandardScaler()`, call `X_train_s = scaler.fit_transform(X_train)`, and then transform test using the *already fitted* scaler.",
            2: "Call `X_test_s = scaler.transform(X_test)` (NEVER `.fit_transform(X_test)`) and return `(X_train_s, X_test_s)`.",
            3: "def leak_free_standardize(X_train, X_test):\n    scaler = StandardScaler()\n    X_train_s = scaler.fit_transform(X_train)\n    X_test_s = scaler.___(X_test)\n    return X_train_s, X_test_s",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 21.5: evaluate_dummy_baseline(X_train, y_train, X_test, y_test)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.dummy import DummyClassifier

    bad = _check_callable("ch21_todo_05", "To-Do 21.5", fn, "evaluate_dummy_baseline", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(42)
    cases = [
        (
            rng.normal(size=(80, 3)),
            np.array([0] * 52 + [1] * 28),
            rng.normal(size=(40, 3)),
            np.array([0] * 26 + [1] * 14),
            "imbalanced negative-majority cohort (35% positive)",
        ),
        (
            rng.normal(size=(50, 2)),
            np.array([1] * 40 + [0] * 10),
            rng.normal(size=(20, 2)),
            np.array([1] * 16 + [0] * 4),
            "positive-majority cohort (80% positive)",
        ),
        (
            rng.normal(size=(60, 2)),
            np.array([0] * 42 + [1] * 18),
            rng.normal(size=(20, 2)),
            np.array([0] * 8 + [1] * 12),
            "shifted test prevalence (train majority=0, test majority=1)",
        ),
    ]
    expected_keys = ["majority_class", "dummy_accuracy", "positive_rate"]
    errors = []

    for X_tr, y_tr, X_te, y_te, desc in cases:
        try:
            got = fn(X_tr.copy(), y_tr.copy(), X_te.copy(), y_te.copy())
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch21_todo_05",
                False,
                "failed",
                "To-Do 21.5 — Runtime Error in `evaluate_dummy_baseline`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch21_todo_05",
                "To-Do 21.5",
                "evaluate_dummy_baseline(X_train, y_train, X_test, y_test)",
                10,
            )

        key_err = _check_dict_keys("ch21_todo_05", "To-Do 21.5", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        exp_maj = int(dummy.predict(X_te[:1])[0])
        exp_acc = round(float(dummy.score(X_te, y_te)), 4)
        exp_pos = round(float(np.mean(np.asarray(y_te) == 1)), 4)

        if not _is_int_like(got["majority_class"]) or int(got["majority_class"]) != exp_maj:
            extra = " (Fit `DummyClassifier` on `X_train, y_train`, not `y_test`!)" if "shifted" in desc else ""
            errors.append(f"[{desc}] majority_class: expected {exp_maj}, got {got['majority_class']!r}{extra}")
        if not _is_number(got["dummy_accuracy"]) or not math.isclose(float(got["dummy_accuracy"]), exp_acc, abs_tol=1e-4):
            errors.append(f"[{desc}] dummy_accuracy: expected {exp_acc}, got {got['dummy_accuracy']!r}")
        if not _is_number(got["positive_rate"]) or not math.isclose(float(got["positive_rate"]), exp_pos, abs_tol=1e-4):
            errors.append(f"[{desc}] positive_rate: expected {exp_pos} (share of `y_test == 1`), got {got['positive_rate']!r}")

    if errors:
        return CheckResult(
            "ch21_todo_05",
            False,
            "failed",
            "To-Do 21.5 — Check `DummyClassifier(strategy='most_frequent')` Baseline Metrics",
            "Fit `DummyClassifier(strategy='most_frequent')` on `(X_train, y_train)`, evaluate on `(X_test, y_test)`, and compute `positive_rate = mean(y_test == 1)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch21_todo_05",
        True,
        "passed",
        "To-Do 21.5 Passed — Majority-Class Dummy Baseline Verified!",
        "Correctly fitted `DummyClassifier(strategy='most_frequent')` on training data and evaluated accuracy and test positive rate.",
        tip="Never celebrate 65% accuracy until you beat the zero-skill `DummyClassifier` floor!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_todo_05",
        chapter="ch21",
        title="To-Do 21.5: Majority-Class Dummy Baseline (evaluate_dummy_baseline)",
        book_section="Section 21.3.6 (Zero-Skill Baseline Benchmarking with DummyClassifier)",
        points=10,
        hints={
            1: "Fit `dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)` and predict on `X_test`.",
            2: "Extract `majority_class = int(dummy.predict(X_test[:1])[0])`, `dummy_accuracy = round(float(dummy.score(X_test, y_test)), 4)`, and `positive_rate = round(float(np.mean(np.asarray(y_test) == 1)), 4)`.",
            3: "def evaluate_dummy_baseline(X_train, y_train, X_test, y_test):\n    dummy = DummyClassifier(strategy='___').fit(X_train, y_train)\n    return {\n        'majority_class': int(dummy.predict(X_test[:1])[0]),\n        'dummy_accuracy': round(float(dummy.score(X_test, y_test)), 4),\n        'positive_rate': round(float(np.mean(np.asarray(y_test) == 1)), 4),\n    }",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 21.6: diagnose_bias_variance(train_score, val_score, gap_threshold=0.10, low_threshold=0.70)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable("ch21_todo_06", "To-Do 21.6", fn, "diagnose_bias_variance", 10)
    if bad is not None:
        return bad

    cases = [
        ((0.62, 0.60), {}, "Underfitting (High Bias)", "low train & low val"),
        ((0.65, 0.50), {}, "Underfitting (High Bias)", "low train precedence over large gap"),
        ((0.96, 0.78), {}, "Overfitting (High Variance)", "high train with large generalization gap"),
        ((0.84, 0.80), {}, "Well-Balanced", "high train & small gap"),
        ((0.70, 0.68), {}, "Well-Balanced", "exact low_threshold boundary (not < 0.70)"),
        ((0.875, 0.75), {"gap_threshold": 0.125}, "Overfitting (High Variance)", "exact gap_threshold boundary (>=)"),
        ((0.76, 0.75), {"low_threshold": 0.80}, "Underfitting (High Bias)", "custom low_threshold=0.80"),
        ((0.86, 0.79), {"gap_threshold": 0.05}, "Overfitting (High Variance)", "custom gap_threshold=0.05"),
    ]
    errors = []

    for args, kwargs, exp, desc in cases:
        try:
            got = fn(*args, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch21_todo_06",
                False,
                "failed",
                "To-Do 21.6 — Runtime Error in `diagnose_bias_variance`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch21_todo_06",
                "To-Do 21.6",
                f"diagnose_bias_variance{args}",
                10,
            )

        if got != exp:
            errors.append(f"[{desc}] args={args}, kwargs={kwargs}: expected {exp!r}, got {got!r}")

    if errors:
        return CheckResult(
            "ch21_todo_06",
            False,
            "failed",
            "To-Do 21.6 — Check Bias-Variance Threshold Order & Exact Labels",
            "Check `train_score < low_threshold` first (`'Underfitting (High Bias)'`), then `(train_score - val_score) >= gap_threshold` (`'Overfitting (High Variance)'`), else `'Well-Balanced'`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch21_todo_06",
        True,
        "passed",
        "To-Do 21.6 Passed — Bias-Variance Diagnostic Classifier Verified!",
        "Passed underfitting precedence, overfitting gap boundary, well-balanced region, and custom threshold checks.",
        tip="Underfitting calls for richer features or higher model capacity; overfitting calls for regularization, pruning, or more training data.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_todo_06",
        chapter="ch21",
        title="To-Do 21.6: Bias-Variance Regime Diagnosis (diagnose_bias_variance)",
        book_section="Section 21.3.7 (Diagnosing High Bias vs. High Variance)",
        points=10,
        hints={
            1: "Check underfitting first: if `train_score < low_threshold`, return `'Underfitting (High Bias)'`.",
            2: "Next check overfitting: `elif (train_score - val_score) >= gap_threshold:` return `'Overfitting (High Variance)'`; otherwise return `'Well-Balanced'`.",
            3: "def diagnose_bias_variance(train_score, val_score, gap_threshold=0.10, low_threshold=0.70):\n    if train_score < low_threshold:\n        return 'Underfitting (High Bias)'\n    if (train_score - val_score) >= ___:\n        return 'Overfitting (High Variance)'\n    return 'Well-Balanced'",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 21.1 (30 pts): audit_split_leakage(train_df, test_df, id_col, group_col=None, time_col=None)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch21_exam_01", "Exam Task 21.1", fn, "audit_split_leakage", 30)
    if bad is not None:
        return bad

    clean_tr = pd.DataFrame({"id": [1, 2, 3, 4], "cluster": [10, 10, 20, 20], "year": [2023, 2023, 2024, 2024]})
    clean_te = pd.DataFrame({"id": [5, 6], "cluster": [30, 30], "year": [2025, 2025]})

    id_leak_tr = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    id_leak_te = pd.DataFrame({"id": [3, 4, 2], "val": [30, 40, 20]})

    grp_leak_tr = pd.DataFrame({"id": [1, 2, 3], "cluster": [10, 20, 30]})
    grp_leak_te = pd.DataFrame({"id": [4, 5, 6], "cluster": [20, 30, 40]})

    time_eq_tr = pd.DataFrame({"id": [1, 2], "year": [2023, 2024]})
    time_eq_te = pd.DataFrame({"id": [3, 4], "year": [2024, 2025]})

    time_rev_tr = pd.DataFrame({"id": [1, 2], "cluster": [1, 2], "year": [2023, 2025]})
    time_rev_te = pd.DataFrame({"id": [3, 4], "cluster": [3, 4], "year": [2024, 2025]})

    cases = [
        (clean_tr, clean_te, "id", "cluster", "year", {"id_overlap": 0, "group_overlap": 0, "temporal_leak": False, "is_leak_free": True}, "clean 3-check split"),
        (id_leak_tr, id_leak_te, "id", None, None, {"id_overlap": 2, "group_overlap": 0, "temporal_leak": False, "is_leak_free": False}, "duplicate ID overlap (group/time None)"),
        (grp_leak_tr, grp_leak_te, "id", "cluster", None, {"id_overlap": 0, "group_overlap": 2, "temporal_leak": False, "is_leak_free": False}, "DHS cluster group leakage"),
        (time_eq_tr, time_eq_te, "id", None, "year", {"id_overlap": 0, "group_overlap": 0, "temporal_leak": True, "is_leak_free": False}, "boundary temporal leak (train_max == test_min)"),
        (time_rev_tr, time_rev_te, "id", "cluster", "year", {"id_overlap": 0, "group_overlap": 0, "temporal_leak": True, "is_leak_free": False}, "future year leaked into training"),
    ]
    expected_keys = ["id_overlap", "group_overlap", "temporal_leak", "is_leak_free"]
    errors = []

    for tr_in, te_in, id_c, grp_c, tm_c, exp, desc in cases:
        tr_cp = tr_in.copy(deep=True)
        te_cp = te_in.copy(deep=True)
        try:
            got = fn(tr_cp, te_cp, id_c, group_col=grp_c, time_col=tm_c)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch21_exam_01",
                False,
                "failed",
                "Exam Task 21.1 — Runtime Error in `audit_split_leakage`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap("ch21_exam_01", "Exam Task 21.1", "audit_split_leakage(...)", 30)

        key_err = _check_dict_keys("ch21_exam_01", "Exam Task 21.1", got, expected_keys, 30)
        if key_err is not None:
            return key_err

        if not tr_cp.equals(tr_in) or not te_cp.equals(te_in):
            errors.append(f"[{desc}] Input DataFrame(s) were mutated!")

        if not _is_int_like(got["id_overlap"]) or int(got["id_overlap"]) != exp["id_overlap"]:
            errors.append(f"[{desc}] id_overlap: expected {exp['id_overlap']}, got {got['id_overlap']!r}")
        if not _is_int_like(got["group_overlap"]) or int(got["group_overlap"]) != exp["group_overlap"]:
            errors.append(f"[{desc}] group_overlap: expected {exp['group_overlap']}, got {got['group_overlap']!r}")
        if not _is_bool_like(got["temporal_leak"]) or bool(got["temporal_leak"]) is not exp["temporal_leak"]:
            extra = " (Use `>=` for `train_df[time_col].max() >= test_df[time_col].min()`!)" if "boundary" in desc else ""
            errors.append(f"[{desc}] temporal_leak: expected {exp['temporal_leak']}, got {got['temporal_leak']!r}{extra}")
        if not _is_bool_like(got["is_leak_free"]) or bool(got["is_leak_free"]) is not exp["is_leak_free"]:
            errors.append(f"[{desc}] is_leak_free: expected {exp['is_leak_free']}, got {got['is_leak_free']!r}")

    if errors:
        return CheckResult(
            "ch21_exam_01",
            False,
            "failed",
            "Exam Task 21.1 — Check ID, Group, & Temporal Leakage Audit Logic",
            "Count set intersections for `id_col` and optional `group_col`, flag `temporal_leak` when `train_max >= test_min`, and set `is_leak_free` accordingly.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch21_exam_01",
        True,
        "passed",
        "Exam Task 21.1 Passed (30/30 pts) — Automated Split Leakage Auditor Verified!",
        "Accurately caught duplicate IDs, cluster overlap, and temporal boundary leaks while approving clean splits.",
        points_earned=30,
        points_possible=30,
        tip="Drop an automated leakage auditor like this at the top of every research ML script before training a single model!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_exam_01",
        chapter="ch21",
        title="Exam 21.1: Automated Split Leakage Auditor (audit_split_leakage)",
        book_section="Chapter 21 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Compute `id_overlap = len(set(train_df[id_col]) & set(test_df[id_col]))` and similarly for `group_col` when `group_col is not None` (else `0`).",
            2: "Set `temporal_leak = bool(train_df[time_col].max() >= test_df[time_col].min())` when `time_col is not None` (else `False`), then `is_leak_free = (id_overlap == 0 and group_overlap == 0 and not temporal_leak)`.",
            3: "def audit_split_leakage(train_df, test_df, id_col, group_col=None, time_col=None):\n    id_overlap = len(set(train_df[id_col]) & set(test_df[id_col]))\n    group_overlap = len(set(train_df[group_col]) & set(test_df[group_col])) if group_col is not None else 0\n    temporal_leak = bool(train_df[time_col].max() >= test_df[time_col].min()) if time_col is not None else False\n    return {'id_overlap': int(id_overlap), 'group_overlap': int(group_overlap), 'temporal_leak': temporal_leak, 'is_leak_free': bool(id_overlap == 0 and group_overlap == 0 and not temporal_leak)}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 21.2 (35 pts): quantify_preprocessing_leakage(X, y, seed=42)
# ---------------------------------------------------------------------------
def _ref_quantify_leakage(X: Any, y: Any, seed: int = 42) -> dict[str, float]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)

    # Leaky: scale ALL rows before splitting
    X_all_s = StandardScaler().fit_transform(X_arr)
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(
        X_all_s, y_arr, test_size=0.2, stratify=y_arr, random_state=seed
    )
    leaky_acc = float(
        LogisticRegression(max_iter=500).fit(Xl_tr, yl_tr).score(Xl_te, yl_te)
    )

    # Clean: split first, fit scaler on train ONLY
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_arr, y_arr, test_size=0.2, stratify=y_arr, random_state=seed
    )
    scaler = StandardScaler().fit(Xc_tr)
    clean_acc = float(
        LogisticRegression(max_iter=500)
        .fit(scaler.transform(Xc_tr), yc_tr)
        .score(scaler.transform(Xc_te), yc_te)
    )

    return {
        "leaky_acc": round(leaky_acc, 4),
        "clean_acc": round(clean_acc, 4),
        "optimism_gap": round(leaky_acc - clean_acc, 4),
    }


def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.datasets import make_classification

    from ..datasets import load_data

    bad = _check_callable("ch21_exam_02", "Exam Task 21.2", fn, "quantify_preprocessing_leakage", 35)
    if bad is not None:
        return bad

    X_hd, y_hd = make_classification(
        n_samples=60, n_features=30, n_informative=5, n_redundant=5, random_state=2
    )
    X_hd = X_hd * np.linspace(0.1, 100.0, 30)

    mch = load_data("mch_survey_sample", as_pandas=True).copy()
    mch["anc_visits"] = mch["anc_visits"].replace(999, np.nan)
    mch_clean = mch[["maternal_age_years", "anc_visits", "child_haz", "maternal_anemia"]].dropna()
    X_mch = mch_clean[["maternal_age_years", "anc_visits", "child_haz"]]
    y_mch = mch_clean["maternal_anemia"]

    cases = [
        (X_hd, y_hd, 42, "high-dim scaled cohort (leaky=0.8333 vs clean=0.7500)"),
        (X_mch, y_mch, 42, "MCH survey clinical features (seed=42)"),
        (X_hd, y_hd, 7, "high-dim cohort with custom seed=7"),
    ]
    expected_keys = ["leaky_acc", "clean_acc", "optimism_gap"]
    errors = []

    for X_in, y_in, seed, desc in cases:
        try:
            got = fn(X_in.copy(), y_in.copy(), seed=seed)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch21_exam_02",
                False,
                "failed",
                "Exam Task 21.2 — Runtime Error in `quantify_preprocessing_leakage`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch21_exam_02",
                "Exam Task 21.2",
                f"quantify_preprocessing_leakage(X, y, seed={seed})",
                35,
            )

        key_err = _check_dict_keys("ch21_exam_02", "Exam Task 21.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        exp = _ref_quantify_leakage(X_in, y_in, seed=seed)
        for key in expected_keys:
            if not _is_number(got[key]) or not math.isclose(float(got[key]), exp[key], abs_tol=1e-3):
                extra = ""
                if key == "optimism_gap" and _is_number(got[key]) and math.isclose(float(got[key]), -exp[key], abs_tol=1e-3):
                    extra = " (Sign is flipped — compute `leaky_acc - clean_acc`!)"
                errors.append(f"[{desc}] {key}: expected {exp[key]}, got {got[key]!r}{extra}")

    if errors:
        return CheckResult(
            "ch21_exam_02",
            False,
            "failed",
            "Exam Task 21.2 — Check Leaky vs. Clean Scaling & `LogisticRegression(max_iter=500)`",
            "Use `test_size=0.2, stratify=y, random_state=seed` and `LogisticRegression(max_iter=500)` for both pipelines, returning `leaky_acc`, `clean_acc`, and `optimism_gap = leaky_acc - clean_acc` rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch21_exam_02",
        True,
        "passed",
        "Exam Task 21.2 Passed (35/35 pts) — Preprocessing Leakage Benchmark Verified!",
        "Quantified the +0.0833 optimism inflation from pre-split scaling vs. clean fit-on-train-only standardization.",
        points_earned=35,
        points_possible=35,
        tip="In Chapter 22, Scikit-Learn `Pipeline` will automate this boundary so preprocessing leakage becomes structurally impossible!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_exam_02",
        chapter="ch21",
        title="Exam 21.2: Quantifying Preprocessing Leakage (quantify_preprocessing_leakage)",
        book_section="Chapter 21 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Leaky path: `X_all = StandardScaler().fit_transform(X)`, then `train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=seed)`, then fit & score `LogisticRegression(max_iter=500)`.",
            2: "Clean path: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)` FIRST, fit `StandardScaler` on `X_train` only, transform both splits, fit & score `LogisticRegression(max_iter=500)`.",
            3: "def quantify_preprocessing_leakage(X, y, seed=42):\n    X_all = StandardScaler().fit_transform(X)\n    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=seed)\n    leaky_acc = float(LogisticRegression(max_iter=500).fit(Xl_tr, yl_tr).score(Xl_te, yl_te))\n    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)\n    sc = StandardScaler().fit(Xc_tr)\n    clean_acc = float(LogisticRegression(max_iter=500).fit(sc.transform(Xc_tr), yc_tr).score(sc.transform(Xc_te), yc_te))\n    return {'leaky_acc': round(leaky_acc, 4), 'clean_acc': round(clean_acc, 4), 'optimism_gap': round(leaky_acc - clean_acc, 4)}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 21.3 (35 pts): build_learning_curve_table(X, y, fractions, seed=42)
# ---------------------------------------------------------------------------
def _ref_learning_curve(X: Any, y: Any, fractions: list[float], seed: int = 42) -> Any:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_arr, y_arr, test_size=0.25, stratify=y_arr, random_state=seed
    )
    rows = []
    for frac in fractions:
        n_sub = int(round(frac * len(X_tr)))
        clf = DecisionTreeClassifier(max_depth=4, random_state=seed)
        clf.fit(X_tr[:n_sub], y_tr[:n_sub])
        tr_acc = float(clf.score(X_tr[:n_sub], y_tr[:n_sub]))
        val_acc = float(clf.score(X_val, y_val))
        rows.append(
            {
                "fraction": round(float(frac), 4),
                "n_train": int(n_sub),
                "train_acc": round(tr_acc, 4),
                "val_acc": round(val_acc, 4),
                "gap": round(tr_acc - val_acc, 4),
            }
        )
    return pd.DataFrame(rows, columns=["fraction", "n_train", "train_acc", "val_acc", "gap"])


def _verify_exam_03(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    from ..datasets import load_data

    bad = _check_callable("ch21_exam_03", "Exam Task 21.3", fn, "build_learning_curve_table", 35)
    if bad is not None:
        return bad

    mch = load_data("mch_survey_sample", as_pandas=True).copy()
    mch["residence"] = mch["residence"].astype(str).str.strip().str.title()
    mch["anc_visits"] = mch["anc_visits"].replace(999, np.nan)
    mch["is_urban"] = (mch["residence"] == "Urban").astype(int)
    w_map = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
    mch["wealth_idx"] = mch["wealth_quintile"].map(w_map)
    feats = ["maternal_age_years", "anc_visits", "is_urban", "wealth_idx", "child_haz"]
    clean = mch[feats + ["maternal_anemia"]].dropna().reset_index(drop=True)

    X_syn, y_syn = make_classification(
        n_samples=200, n_features=6, n_informative=4, random_state=11
    )

    cases = [
        (clean[feats], clean["maternal_anemia"], [0.2, 0.4, 0.6, 0.8, 1.0], 42, "MCH survey 5-fraction learning curve"),
        (X_syn, y_syn, [0.25, 0.5, 0.75, 1.0], 17, "synthetic 4-fraction curve (seed=17)"),
    ]
    expected_cols = ["fraction", "n_train", "train_acc", "val_acc", "gap"]
    errors = []

    for X_in, y_in, fracs, seed, desc in cases:
        try:
            got = fn(X_in.copy(), y_in.copy(), list(fracs), seed=seed)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch21_exam_03",
                False,
                "failed",
                "Exam Task 21.3 — Runtime Error in `build_learning_curve_table`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch21_exam_03",
                "Exam Task 21.3",
                f"build_learning_curve_table(X, y, {fracs}, seed={seed})",
                35,
            )

        if not isinstance(got, pd.DataFrame):
            return CheckResult(
                "ch21_exam_03",
                False,
                "failed",
                "Exam Task 21.3 — Return a Pandas DataFrame",
                f"Expected a pd.DataFrame with columns {expected_cols}, got {type(got).__name__}.",
                points_earned=0,
                points_possible=35,
            )

        if list(got.columns) != expected_cols:
            errors.append(f"[{desc}] Expected columns {expected_cols}, got {list(got.columns)}.")
            continue

        if len(got) != len(fracs):
            errors.append(f"[{desc}] Expected {len(fracs)} rows, got {len(got)}.")
            continue

        exp_df = _ref_learning_curve(X_in, y_in, fracs, seed=seed)
        for i in range(len(fracs)):
            if abs(int(got.iloc[i]["n_train"]) - int(exp_df.iloc[i]["n_train"])) > 1:
                errors.append(
                    f"[{desc}] row {i} n_train: expected {int(exp_df.iloc[i]['n_train'])}, "
                    f"got {got.iloc[i]['n_train']!r}"
                )
            for col in ["fraction", "train_acc", "val_acc", "gap"]:
                g_val = float(got.iloc[i][col])
                e_val = float(exp_df.iloc[i][col])
                if not math.isclose(g_val, e_val, abs_tol=1e-3):
                    errors.append(f"[{desc}] row {i} {col}: expected {e_val}, got {g_val}")

    if errors:
        return CheckResult(
            "ch21_exam_03",
            False,
            "failed",
            "Exam Task 21.3 — Check Learning Curve Holdout, Tree Depth=4, & Table Columns",
            "Hold out 25% stratified validation (`random_state=seed`), slice `n_train = int(round(frac * len(X_train)))` rows per fraction, fit `DecisionTreeClassifier(max_depth=4, random_state=seed)`, and round metrics to 4 decimals.",
            details=errors[:8],
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch21_exam_03",
        True,
        "passed",
        "Exam Task 21.3 Passed (35/35 pts) — Bias-Variance Learning Curve Table Verified!",
        "Built complete learning-curve diagnostics showing generalization gap shrinking as training sample size grows.",
        points_earned=35,
        points_possible=35,
        tip="Learning curves tell you directly whether collecting more patients (shrinking variance gap) will actually improve validation accuracy!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch21_exam_03",
        chapter="ch21",
        title="Exam 21.3: Bias-Variance Learning Curve Table (build_learning_curve_table)",
        book_section="Chapter 21 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Hold out 25% via `train_test_split(np.asarray(X, dtype=float), np.asarray(y), test_size=0.25, stratify=y, random_state=seed)`.",
            2: "Loop over `frac` in `fractions`, compute `n_sub = int(round(frac * len(X_tr)))`, fit `DecisionTreeClassifier(max_depth=4, random_state=seed)` on `(X_tr[:n_sub], y_tr[:n_sub])`, and record `train_acc`, `val_acc`, and `gap = train_acc - val_acc` rounded to 4 decimals.",
            3: "def build_learning_curve_table(X, y, fractions, seed=42):\n    X_tr, X_val, y_tr, y_val = train_test_split(np.asarray(X, dtype=float), np.asarray(y), test_size=0.25, stratify=y, random_state=seed)\n    rows = []\n    for frac in fractions:\n        n_sub = int(round(frac * len(X_tr)))\n        clf = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(X_tr[:n_sub], y_tr[:n_sub])\n        tr_acc, val_acc = float(clf.score(X_tr[:n_sub], y_tr[:n_sub])), float(clf.score(X_val, y_val))\n        rows.append({'fraction': round(float(frac), 4), 'n_train': n_sub, 'train_acc': round(tr_acc, 4), 'val_acc': round(val_acc, 4), 'gap': round(tr_acc - val_acc, 4)})\n    return pd.DataFrame(rows, columns=['fraction', 'n_train', 'train_acc', 'val_acc', 'gap'])",
        },
        verifier=_verify_exam_03,
    )
)
