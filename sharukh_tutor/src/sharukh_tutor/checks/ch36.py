"""Chapter 36 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch36_todo_01..06) and the Grand Book Finale Capstone Exam
(ch36_exam_01..03) for Chapter 36: "Flagship Research Capstone — From Raw
Survey CSV to Publication-Ready Paper & Reproducible Artifact".

Every verifier lazy-imports NumPy/Pandas/SciPy/Statsmodels/Scikit-Learn/PyTorch/
Joblib inside the check function, runs all filesystem checks strictly inside
`tempfile.TemporaryDirectory()`, enforces non-mutation and anti-leakage checks,
and lets NotImplementedError propagate cleanly for starter stubs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


_REQUIRED_RELEASE_FILES = sorted(
    [
        "MODEL_CARD.md",
        "fairness_audit.json",
        "leaderboard.csv",
        "manifest.json",
        "model.joblib",
        "table1.csv",
    ]
)

_WEALTH_MAP = {
    "Poorest": 1,
    "Poorer": 2,
    "Middle": 3,
    "Richer": 4,
    "Richest": 5,
}

_CORE_FEATURES = [
    "maternal_age_years",
    "anc_visits",
    "maternal_hb_g_dl",
    "is_urban",
    "wealth_rank",
]


def _is_number(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
        val, (bool, np.bool_)
    )


def _is_bool_like(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (bool, np.bool_))


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
        f"Calling {call_str} returned None instead of the expected artifact.",
        details=[
            "Remember: print() only displays text; functions must end with `return ...`.",
            f"Ensure {call_str} returns the required DataFrame, dict, or string.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Publication-grade research helpers always return structured objects for downstream auditing.",
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
            f"Expected a dict with keys {expected_keys}, got {type(got).__name__}.",
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


def _load_mch_df():
    from ..datasets import load_data

    return load_data("mch_survey_sample", as_pandas=True)


def _ref_clean_mch(mch_df):
    import numpy as np
    import pandas as pd

    df = mch_df.copy()
    df["anc_visits"] = pd.to_numeric(df["anc_visits"], errors="coerce").replace(
        999, np.nan
    )
    df["residence"] = df["residence"].astype(str).str.strip().str.title()
    df["is_urban"] = (df["residence"] == "Urban").astype(int)
    if "wealth_quintile" in df.columns:
        df["wealth_rank"] = df["wealth_quintile"].map(_WEALTH_MAP)
    df = df.dropna(subset=["child_stunted"]).copy()
    df["child_stunted"] = df["child_stunted"].astype(int)
    return df


def _ref_table1(mch_df):
    import pandas as pd
    from scipy import stats

    df = _ref_clean_mch(mch_df)
    g0 = df[df["child_stunted"] == 0]
    g1 = df[df["child_stunted"] == 1]

    rows = []
    for col in ["maternal_age_years", "anc_visits", "maternal_hb_g_dl"]:
        s0 = g0[col].dropna()
        s1 = g1[col].dropna()
        _, p_val = stats.ttest_ind(s0, s1, equal_var=False)
        rows.append(
            {
                "variable": col,
                "not_stunted": f"{s0.mean():.2f} ± {s0.std(ddof=1):.2f}",
                "stunted": f"{s1.mean():.2f} ± {s1.std(ddof=1):.2f}",
                "p_value": round(float(p_val), 4),
            }
        )

    ct = pd.crosstab(df["child_stunted"], df["is_urban"])
    _, p_pearson, _, _ = stats.chi2_contingency(ct, correction=False)
    _, p_yates, _, _ = stats.chi2_contingency(ct, correction=True)
    c0, pct0 = int(g0["is_urban"].sum()), 100.0 * float(g0["is_urban"].mean())
    c1, pct1 = int(g1["is_urban"].sum()), 100.0 * float(g1["is_urban"].mean())
    rows.append(
        {
            "variable": "is_urban",
            "not_stunted": f"{c0} ({pct0:.1f}%)",
            "stunted": f"{c1} ({pct1:.1f}%)",
            "p_value": round(float(p_pearson), 4),
        }
    )
    return pd.DataFrame(rows), round(float(p_yates), 4)


def _parse_mean_sd(text: Any) -> Optional[tuple[float, float]]:
    if not isinstance(text, str) or "±" not in text:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) != 2:
        return None
    return float(nums[0]), float(nums[1])


def _parse_count_pct(text: Any) -> Optional[tuple[int, float]]:
    if not isinstance(text, str) or "(" not in text or "%" not in text:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) != 2:
        return None
    return int(round(float(nums[0]))), float(nums[1])


def _validate_table1_df(
    ex_id: str,
    title: str,
    got: Any,
    mch_df: Any,
    points_possible: int = 10,
) -> Optional[CheckResult]:
    import pandas as pd

    if not isinstance(got, pd.DataFrame):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Expected a Pandas DataFrame",
            f"Expected a DataFrame with columns ['variable', 'not_stunted', 'stunted', 'p_value'], got {type(got).__name__}.",
            points_earned=0,
            points_possible=points_possible,
        )

    expected_cols = ["variable", "not_stunted", "stunted", "p_value"]
    if list(got.columns) != expected_cols:
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Check Table 1 Column Names",
            f"Expected exact columns {expected_cols}, got {list(got.columns)}.",
            points_earned=0,
            points_possible=points_possible,
        )

    ref_df, p_yates = _ref_table1(mch_df)
    expected_vars = list(ref_df["variable"])
    got_vars = list(got["variable"])
    if set(got_vars) != set(expected_vars) or len(got) != len(expected_vars):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Check Table 1 Variable Rows",
            f"Expected 4 rows for {expected_vars}, got {got_vars}.",
            points_earned=0,
            points_possible=points_possible,
        )

    got_by_var = got.set_index("variable")
    ref_by_var = ref_df.set_index("variable")
    errors = []

    for col in ["maternal_age_years", "anc_visits", "maternal_hb_g_dl"]:
        for grp in ["not_stunted", "stunted"]:
            g_parsed = _parse_mean_sd(got_by_var.loc[col, grp])
            r_parsed = _parse_mean_sd(ref_by_var.loc[col, grp])
            if g_parsed is None:
                errors.append(
                    f"{col} ({grp}): expected 'mean ± SD' format, got {got_by_var.loc[col, grp]!r}."
                )
                continue
            if r_parsed is not None and (
                abs(g_parsed[0] - r_parsed[0]) > 0.06
                or abs(g_parsed[1] - r_parsed[1]) > 0.06
            ):
                hint_999 = (
                    " (Did you replace sentinel `anc_visits == 999` with `np.nan`?)"
                    if col == "anc_visits" and g_parsed[0] > 10
                    else ""
                )
                errors.append(
                    f"{col} ({grp}): expected '{ref_by_var.loc[col, grp]}', got {got_by_var.loc[col, grp]!r}{hint_999}."
                )

        got_p = got_by_var.loc[col, "p_value"]
        ref_p = ref_by_var.loc[col, "p_value"]
        if not _is_number(got_p) or not math.isclose(float(got_p), float(ref_p), abs_tol=1e-3):
            errors.append(
                f"{col} p_value: expected {ref_p} (Welch's t-test), got {got_p!r}."
            )

    for grp in ["not_stunted", "stunted"]:
        g_cp = _parse_count_pct(got_by_var.loc["is_urban", grp])
        r_cp = _parse_count_pct(ref_by_var.loc["is_urban", grp])
        if g_cp is None:
            errors.append(
                f"is_urban ({grp}): expected 'count (pct%)' format, got {got_by_var.loc['is_urban', grp]!r}."
            )
            continue
        if r_cp is not None and (
            g_cp[0] != r_cp[0] or abs(g_cp[1] - r_cp[1]) > 0.2
        ):
            errors.append(
                f"is_urban ({grp}): expected '{ref_by_var.loc['is_urban', grp]}', got {got_by_var.loc['is_urban', grp]!r} "
                "(ensure `residence` is title-cased so lowercase 'urban' is counted!)."
            )

    got_chi_p = got_by_var.loc["is_urban", "p_value"]
    ref_chi_p = float(ref_by_var.loc["is_urban", "p_value"])
    if not _is_number(got_chi_p) or (
        not math.isclose(float(got_chi_p), ref_chi_p, abs_tol=1e-3)
        and not math.isclose(float(got_chi_p), p_yates, abs_tol=1e-3)
    ):
        errors.append(
            f"is_urban p_value: expected {ref_chi_p} (Pearson chi-square), got {got_chi_p!r}."
        )

    if errors:
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Check Cleaning, Formatting & Hypothesis Tests",
            "Review sentinel cleaning (`999 -> NaN`), title-casing `residence`, Welch's t-test (`equal_var=False`), and chi-square p-values.",
            details=errors,
            points_earned=0,
            points_possible=points_possible,
        )
    return None


# ---------------------------------------------------------------------------
# GUIDED TO-DO 36.1: build_publication_table1(mch_df)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch36_todo_01", "To-Do 36.1", fn, "build_publication_table1", 10)
    if bad is not None:
        return bad

    mch_orig = _load_mch_df()
    mch_copy = mch_orig.copy(deep=True)
    try:
        got = fn(mch_copy)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch36_todo_01",
            False,
            "failed",
            "To-Do 36.1 — Runtime Error in `build_publication_table1`",
            f"build_publication_table1(mch_df) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got is None:
        return _none_return_trap("ch36_todo_01", "To-Do 36.1", "build_publication_table1(mch_df)", 10)

    if not mch_copy.equals(mch_orig):
        return CheckResult(
            "ch36_todo_01",
            False,
            "failed",
            "To-Do 36.1 — Input DataFrame Was Mutated",
            "Start your function with `df = mch_df.copy()` and avoid `inplace=True`.",
            points_earned=0,
            points_possible=10,
        )

    err = _validate_table1_df("ch36_todo_01", "To-Do 36.1", got, mch_orig, 10)
    if err is not None:
        return err

    # Verify missing child_stunted rows are dropped cleanly
    mch_missing = mch_orig.copy(deep=True)
    mch_missing.loc[0, "child_stunted"] = np.nan
    try:
        got_missing = fn(mch_missing)
    except Exception as exc:
        return CheckResult(
            "ch36_todo_01",
            False,
            "failed",
            "To-Do 36.1 — Failed When `child_stunted` Contains NaN",
            f"Raised {type(exc).__name__}: {exc}. Call `df = df.dropna(subset=['child_stunted']).copy()`.",
            points_earned=0,
            points_possible=10,
        )

    err2 = _validate_table1_df("ch36_todo_01", "To-Do 36.1", got_missing, mch_missing, 10)
    if err2 is not None:
        return err2

    return CheckResult(
        "ch36_todo_01",
        True,
        "passed",
        "To-Do 36.1 Passed — Journal Table 1 Verified!",
        "Cleaned `anc_visits == 999`, title-cased `residence`, dropped missing outcomes, and computed exact Welch t-test & chi-square p-values.",
        points_earned=10,
        points_possible=10,
        tip="In epidemiological manuscripts, Table 1 always goes through a scripted cleaning pipeline so numbers match the abstract and regression tables.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_todo_01",
        chapter="ch36",
        title="To-Do 36.1: Publication-Ready Journal Table 1 (build_publication_table1)",
        book_section="Section 36.3.1 (Epidemiological Cohort Cleaning & Table 1)",
        points=10,
        hints={
            1: "Copy `mch_df`, replace `anc_visits == 999` with `np.nan`, standardize `residence` with `.str.strip().str.title()`, create `is_urban`, and drop missing `child_stunted` rows.",
            2: "For continuous columns (`maternal_age_years`, `anc_visits`, `maternal_hb_g_dl`), format `f'{mean:.2f} ± {sd:.2f}'` and run `stats.ttest_ind(s0, s1, equal_var=False)`. For `is_urban`, format `f'{cnt} ({pct:.1f}%)'` and run `stats.chi2_contingency(pd.crosstab(df['child_stunted'], df['is_urban']), correction=False)`.",
            3: "def build_publication_table1(mch_df):\n    df = mch_df.copy()\n    df['anc_visits'] = pd.to_numeric(df['anc_visits'], errors='coerce').replace(999, np.nan)\n    df['residence'] = df['residence'].astype(str).str.strip().str.title()\n    df['is_urban'] = (df['residence'] == 'Urban').astype(int)\n    df = df.dropna(subset=['child_stunted']).copy()\n    # Compare child_stunted == 0 vs 1 and return DataFrame(['variable', 'not_stunted', 'stunted', 'p_value'])",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 36.2: run_capstone_model_benchmark(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable("ch36_todo_02", "To-Do 36.2", fn, "run_capstone_model_benchmark", 10)
    if bad is not None:
        return bad

    mch_orig = _load_mch_df()
    mch_copy = mch_orig.copy(deep=True)
    try:
        got = fn(mch_copy, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch36_todo_02",
            False,
            "failed",
            "To-Do 36.2 — Runtime Error in `run_capstone_model_benchmark`",
            f"run_capstone_model_benchmark(mch_df, seed=42) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got is None:
        return _none_return_trap(
            "ch36_todo_02", "To-Do 36.2", "run_capstone_model_benchmark(mch_df, seed=42)", 10
        )

    if not mch_copy.equals(mch_orig):
        return CheckResult(
            "ch36_todo_02",
            False,
            "failed",
            "To-Do 36.2 — Input DataFrame Was Mutated",
            "Copy `mch_df` before cleaning so the caller's survey table stays untouched.",
            points_earned=0,
            points_possible=10,
        )

    expected_cols = ["model", "mean_auc", "std_auc", "mean_brier"]
    if not isinstance(got, pd.DataFrame) or list(got.columns) != expected_cols:
        return CheckResult(
            "ch36_todo_02",
            False,
            "failed",
            "To-Do 36.2 — Check Leaderboard DataFrame Columns",
            f"Expected a DataFrame with exact columns {expected_cols}, got {type(got).__name__} ({getattr(got, 'columns', None)}).",
            points_earned=0,
            points_possible=10,
        )

    if len(got) != 4:
        return CheckResult(
            "ch36_todo_02",
            False,
            "failed",
            "To-Do 36.2 — Expected 4 Model Families",
            f"Benchmark must evaluate LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier, and PyTorch MLP (4 rows); got {len(got)} rows.",
            points_earned=0,
            points_possible=10,
        )

    names_low = " ".join(str(m).lower() for m in got["model"])
    missing_fam = []
    if "logistic" not in names_low:
        missing_fam.append("LogisticRegression")
    if "forest" not in names_low:
        missing_fam.append("RandomForestClassifier")
    if "boost" not in names_low and "hgb" not in names_low:
        missing_fam.append("HistGradientBoostingClassifier")
    if "mlp" not in names_low and "torch" not in names_low and "neural" not in names_low:
        missing_fam.append("PyTorch_MLP")
    if missing_fam:
        return CheckResult(
            "ch36_todo_02",
            False,
            "failed",
            "To-Do 36.2 — Missing Required Model Family",
            f"Could not find model family name(s): {missing_fam}. Got: {list(got['model'])}.",
            points_earned=0,
            points_possible=10,
        )

    errors = []
    aucs = [float(x) for x in got["mean_auc"]]
    stds = [float(x) for x in got["std_auc"]]
    briers = [float(x) for x in got["mean_brier"]]

    if aucs != sorted(aucs, reverse=True):
        errors.append(f"Leaderboard must be sorted by `mean_auc` descending, got {aucs}.")

    for name, auc, sd, br in zip(got["model"], aucs, stds, briers):
        if auc > 0.90:
            errors.append(
                f"{name}: mean_auc={auc:.4f} is suspiciously high (>0.90) — "
                "did you accidentally include `child_haz` (target leakage!) as a feature?"
            )
        elif not (0.55 <= auc <= 0.85):
            errors.append(f"{name}: mean_auc={auc:.4f} outside expected leak-free range [0.55, 0.85].")
        if not (0.0 < sd <= 0.30):
            errors.append(f"{name}: std_auc={sd:.4f} outside expected range (0.0, 0.30].")
        if not (0.05 <= br <= 0.25):
            errors.append(f"{name}: mean_brier={br:.4f} outside expected calibrated range [0.05, 0.25].")

    # Determinism check
    got2 = fn(mch_orig.copy(deep=True), seed=42)
    if list(got["mean_auc"]) != list(got2["mean_auc"]):
        errors.append("Benchmark is non-deterministic across repeated `seed=42` runs — seed PyTorch & sklearn!")

    if errors:
        return CheckResult(
            "ch36_todo_02",
            False,
            "failed",
            "To-Do 36.2 — Check GroupKFold Benchmark, Anti-Leakage & Determinism",
            "Ensure 5-fold `GroupKFold` on `cluster_id`, train-only imputation/scaling, no `child_haz` feature, descending sort by `mean_auc`, and deterministic seeding.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch36_todo_02",
        True,
        "passed",
        "To-Do 36.2 Passed — 4-Model Cluster-Grouped Benchmark (ML + PyTorch MLP) Verified!",
        "Evaluated LR, RF, HGB, and leak-free PyTorch MLP across 5 survey-cluster folds with deterministic ROC-AUC and Brier calibration scores.",
        points_earned=10,
        points_possible=10,
        tip="Notice how linear and tree baselines rival or beat neural networks on N=500 tabular survey data — always benchmark simple models honestly!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_todo_02",
        chapter="ch36",
        title="To-Do 36.2: 4-Model Cluster-Grouped ML + PyTorch Benchmark (run_capstone_model_benchmark)",
        book_section="Section 36.3.3 (Leak-Free 5-Fold GroupKFold Benchmarking)",
        points=10,
        hints={
            1: "Clean `mch_df` (`anc_visits == 999 -> NaN`, `residence` -> `is_urban`, `wealth_quintile` -> `wealth_rank`), select the 5 predictors (NEVER `child_haz`!), and split with `GroupKFold(n_splits=5)` on `cluster_id`.",
            2: "Wrap sklearn models in `Pipeline` with `SimpleImputer(strategy='median')` (and `StandardScaler()` for `LogisticRegression`). For `PyTorch_MLP`, fit `SimpleImputer` and `StandardScaler` strictly on `X_train` inside each fold, call `torch.set_num_threads(1)` and `torch.manual_seed(seed + fold_idx)`, and train `Linear(5, 16) -> ReLU -> Linear(16, 1)` for 30 epochs with `BCEWithLogitsLoss`.",
            3: "Return `pd.DataFrame(rows).sort_values('mean_auc', ascending=False).reset_index(drop=True)` with columns `['model', 'mean_auc', 'std_auc', 'mean_brier']` rounded to 4 decimals.",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 36.3: build_publication_results_markdown(leaderboard_df, fairness_dict)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable(
        "ch36_todo_03", "To-Do 36.3", fn, "build_publication_results_markdown", 10
    )
    if bad is not None:
        return bad

    cases = [
        (
            pd.DataFrame(
                [
                    {"model": "RandomForestClassifier", "mean_auc": 0.7181, "std_auc": 0.0825, "mean_brier": 0.1137},
                    {"model": "LogisticRegression", "mean_auc": 0.7132, "std_auc": 0.0823, "mean_brier": 0.1138},
                ]
            ),
            {"demographic_parity_diff": 0.2371, "equal_opportunity_diff": 0.4993},
            "capstone leaderboard",
        ),
        (
            pd.DataFrame(
                [
                    {"model": "PyTorch_MLP", "mean_auc": 0.7420, "std_auc": 0.0510, "mean_brier": 0.1095},
                ]
            ),
            {"demographic_parity_diff": 0.0812, "equal_opportunity_diff": 0.0435},
            "single-model custom test",
        ),
    ]

    errors = []
    for lb_df, fair, desc in cases:
        try:
            got = fn(lb_df.copy(), dict(fair))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch36_todo_03",
                False,
                "failed",
                "To-Do 36.3 — Runtime Error in `build_publication_results_markdown`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch36_todo_03",
                "To-Do 36.3",
                "build_publication_results_markdown(leaderboard_df, fairness_dict)",
                10,
            )

        if not isinstance(got, str) or not got.strip():
            errors.append(f"[{desc}] Expected a non-empty Markdown string, got {type(got).__name__}.")
            continue

        if "### Model Benchmark" not in got:
            errors.append(f"[{desc}] Missing exact heading '### Model Benchmark'.")
        if "### Subgroup Fairness Summary" not in got:
            errors.append(f"[{desc}] Missing exact heading '### Subgroup Fairness Summary'.")
        if "|" not in got or "---" not in got:
            errors.append(f"[{desc}] Expected a Markdown pipe table (`| ... |` and `| --- |`).")

        for _, row in lb_df.iterrows():
            if str(row["model"]) not in got:
                errors.append(f"[{desc}] Missing model row for '{row['model']}'.")
            auc_str = f"{float(row['mean_auc']):.4f}"
            if auc_str not in got and str(row["mean_auc"]) not in got:
                errors.append(f"[{desc}] Missing mean_auc value '{auc_str}' for {row['model']}.")

        for k in ["demographic_parity_diff", "equal_opportunity_diff"]:
            val_str = f"{float(fair[k]):.4f}"
            if val_str not in got and str(fair[k]) not in got:
                errors.append(f"[{desc}] Missing fairness metric `{k}` value ({val_str}).")

    if errors:
        return CheckResult(
            "ch36_todo_03",
            False,
            "failed",
            "To-Do 36.3 — Check Publication Results Markdown Structure",
            "Include '### Model Benchmark' with a Markdown table for every model, plus '### Subgroup Fairness Summary' reporting `demographic_parity_diff` and `equal_opportunity_diff`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch36_todo_03",
        True,
        "passed",
        "To-Do 36.3 Passed — Publication Results Markdown Generator Verified!",
        "Programmatically formatted both the Model Benchmark table and Subgroup Fairness Summary with zero manual copy-paste.",
        points_earned=10,
        points_possible=10,
        tip="Generating manuscript tables directly from code eliminates transcription typos during peer review revisions.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_todo_03",
        chapter="ch36",
        title="To-Do 36.3: Automated Publication Results Markdown (build_publication_results_markdown)",
        book_section="Section 36.3.4 (Programmatic Manuscript Tables & Fairness Reporting)",
        points=10,
        hints={
            1: "Start a list of Markdown lines with `'### Model Benchmark'`, a pipe-table header `'| Model | Mean AUC | SD AUC | Mean Brier |'`, and separator `'| :--- | :---: | :---: | :---: |'`.",
            2: "Loop over `leaderboard_df.itertuples(index=False)` to append each row formatted to 4 decimals, then add `'### Subgroup Fairness Summary'` with bullet lines for `fairness_dict['demographic_parity_diff']` and `fairness_dict['equal_opportunity_diff']`.",
            3: "Return `'\\n'.join(lines)` containing both `'### Model Benchmark'` and `'### Subgroup Fairness Summary'`.",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 36.4: compute_clinical_net_benefit(y_true, y_prob, thresholds)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    bad = _check_callable("ch36_todo_04", "To-Do 36.4", fn, "compute_clinical_net_benefit", 10)
    if bad is not None:
        return bad

    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.85, 0.45, 0.25, 0.12, 0.35, 0.22, 0.18, 0.11, 0.08, 0.05,
                       0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
    thresholds = [0.10, 0.20, 0.30, 0.50]

    try:
        got = fn(y_true.copy(), y_prob.copy(), list(thresholds))
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch36_todo_04",
            False,
            "failed",
            "To-Do 36.4 — Runtime Error in `compute_clinical_net_benefit`",
            f"compute_clinical_net_benefit(...) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got is None:
        return _none_return_trap(
            "ch36_todo_04", "To-Do 36.4", "compute_clinical_net_benefit(y_true, y_prob, thresholds)", 10
        )

    expected_cols = ["threshold", "nb_model", "nb_all", "nb_none"]
    if not isinstance(got, pd.DataFrame) or list(got.columns) != expected_cols:
        return CheckResult(
            "ch36_todo_04",
            False,
            "failed",
            "To-Do 36.4 — Check DCA DataFrame Columns",
            f"Expected DataFrame with columns {expected_cols}, got {type(got).__name__} ({getattr(got, 'columns', None)}).",
            points_earned=0,
            points_possible=10,
        )

    errors = []
    if len(got) != len(thresholds):
        errors.append(f"Expected {len(thresholds)} rows, got {len(got)}.")
    else:
        n = len(y_true)
        prev = float(np.mean(y_true))
        for idx, t in enumerate(thresholds):
            yp = (y_prob >= t).astype(int)
            tp = int(np.sum((yp == 1) & (y_true == 1)))
            fp = int(np.sum((yp == 1) & (y_true == 0)))
            odds = t / (1.0 - t)
            exp_m = round((tp / n) - (fp / n) * odds, 5)
            exp_a = round(prev - (1.0 - prev) * odds, 5)

            row = got.iloc[idx]
            if not math.isclose(float(row["threshold"]), t, abs_tol=1e-5):
                errors.append(f"Row {idx}: expected threshold={t}, got {row['threshold']!r}.")
            if not math.isclose(float(row["nb_model"]), exp_m, abs_tol=1e-5):
                errors.append(f"t={t}: expected nb_model={exp_m}, got {row['nb_model']!r}.")
            if not math.isclose(float(row["nb_all"]), exp_a, abs_tol=1e-5):
                errors.append(f"t={t}: expected nb_all={exp_a}, got {row['nb_all']!r}.")
            if not math.isclose(float(row["nb_none"]), 0.0, abs_tol=1e-5):
                errors.append(f"t={t}: expected nb_none=0.0, got {row['nb_none']!r}.")

    # Check invalid threshold guard (t=1.0 -> ValueError)
    for bad_t in [0.0, 1.0]:
        try:
            fn(y_true, y_prob, [bad_t])
            errors.append(
                f"Expected ValueError for invalid threshold t={bad_t} (must satisfy 0 < t < 1)."
            )
        except ValueError:
            pass
        except Exception as exc:
            errors.append(
                f"Expected ValueError for threshold t={bad_t}, got {type(exc).__name__}: {exc}."
            )

    if errors:
        return CheckResult(
            "ch36_todo_04",
            False,
            "failed",
            "To-Do 36.4 — Check Vickers & Elkin Net Benefit Formula & Threshold Guard",
            "Validate `0 < t < 1` (raising `ValueError`), then compute `nb_model = (TP/N) - (FP/N)*(t/(1-t))`, `nb_all = prev - (1-prev)*(t/(1-t))`, and `nb_none = 0.0` rounded to 5 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch36_todo_04",
        True,
        "passed",
        "To-Do 36.4 Passed — Decision Curve Analysis (Clinical Net Benefit) Verified!",
        "Verified Vickers & Elkin (2006) net benefit across thresholds plus ValueError protection for boundary thresholds `0.0` and `1.0`.",
        points_earned=10,
        points_possible=10,
        tip="A model only earns clinical adoption when its net benefit exceeds BOTH Treat-All and Treat-None across realistic decision thresholds.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_todo_04",
        chapter="ch36",
        title="To-Do 36.4: Decision Curve Analysis / Clinical Net Benefit (compute_clinical_net_benefit)",
        book_section="Section 36.3.5 (Decision Curve Analysis — Vickers & Elkin, 2006)",
        points=10,
        hints={
            1: "Convert `y_true` and `y_prob` to NumPy arrays, validate every `t` satisfies `0 < t < 1` (raising `ValueError` otherwise), and compute `prev = float(np.mean(y_true))`.",
            2: "At each threshold `t`, predict `y_pred = (y_prob >= t).astype(int)`, count `tp` and `fp`, set `odds = t / (1.0 - t)`, and compute `nb_model = (tp / n) - (fp / n) * odds` and `nb_all = prev - (1.0 - prev) * odds`.",
            3: "Round `threshold`, `nb_model`, `nb_all`, and `nb_none` (0.0) to 5 decimals (add `+ 0.0` to avoid `-0.0`) and return `pd.DataFrame(rows)`.",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 36.5: draft_structured_abstract(study_stats)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch36_todo_05", "To-Do 36.5", fn, "draft_structured_abstract", 10)
    if bad is not None:
        return bad

    cases = [
        {
            "n_total": 500,
            "stunting_pct": 14.0,
            "best_model": "LogisticRegression",
            "best_auc": 0.7132,
            "brier_score": 0.1138,
            "eo_gap": 0.6775,
        },
        {
            "n_total": 1240,
            "stunting_pct": 21.5,
            "best_model": "RandomForestClassifier",
            "best_auc": 0.7645,
            "brier_score": 0.1320,
            "eo_gap": 0.0910,
        },
    ]

    expected_keys = ["abstract_text", "word_count", "within_limit"]
    required_sections = [
        "**Background:**",
        "**Methods:**",
        "**Results:**",
        "**Conclusions:**",
    ]
    errors = []

    for idx, stats_dict in enumerate(cases, start=1):
        try:
            got = fn(dict(stats_dict))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch36_todo_05",
                False,
                "failed",
                "To-Do 36.5 — Runtime Error in `draft_structured_abstract`",
                f"Case {idx} raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch36_todo_05", "To-Do 36.5", "draft_structured_abstract(study_stats)", 10)

        key_err = _check_dict_keys("ch36_todo_05", "To-Do 36.5", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        text = got["abstract_text"]
        if not isinstance(text, str) or not text.strip():
            errors.append(f"[Case {idx}] `abstract_text` must be a non-empty string.")
            continue

        for sec in required_sections:
            if sec not in text:
                errors.append(f"[Case {idx}] Missing required structured section '{sec}'.")

        if str(stats_dict["n_total"]) not in text:
            errors.append(f"[Case {idx}] Missing `n_total` ({stats_dict['n_total']}) in abstract.")
        if str(stats_dict["best_model"]) not in text:
            errors.append(f"[Case {idx}] Missing `best_model` ({stats_dict['best_model']}) in abstract.")
        for num_k, fmt in [
            ("stunting_pct", f"{stats_dict['stunting_pct']:.1f}"),
            ("best_auc", f"{stats_dict['best_auc']:.4f}"),
            ("brier_score", f"{stats_dict['brier_score']:.4f}"),
            ("eo_gap", f"{stats_dict['eo_gap']:.4f}"),
        ]:
            if fmt not in text and str(stats_dict[num_k]) not in text:
                errors.append(f"[Case {idx}] Missing `{num_k}` ({fmt}) in abstract.")

        actual_wc = len(text.split())
        if got["word_count"] != actual_wc:
            errors.append(
                f"[Case {idx}] `word_count` mismatch: expected {actual_wc} (`len(text.split())`), got {got['word_count']!r}."
            )
        if not _is_bool_like(got["within_limit"]) or bool(got["within_limit"]) != (actual_wc <= 300):
            errors.append(
                f"[Case {idx}] `within_limit` must equal `word_count <= 300` ({actual_wc <= 300})."
            )
        if actual_wc > 300:
            errors.append(f"[Case {idx}] Abstract is {actual_wc} words; keep it concise (<= 300 words).")

    # Verify missing required key raises KeyError
    incomplete = dict(cases[0])
    del incomplete["eo_gap"]
    try:
        fn(incomplete)
        errors.append("Expected `KeyError` when required key `'eo_gap'` is missing from `study_stats`.")
    except KeyError:
        pass
    except Exception as exc:
        errors.append(f"Expected `KeyError` for missing key, got {type(exc).__name__}: {exc}.")

    if errors:
        return CheckResult(
            "ch36_todo_05",
            False,
            "failed",
            "To-Do 36.5 — Check Structured Abstract Sections, Metrics & Word Limit",
            "Include **Background:**, **Methods:**, **Results:**, and **Conclusions:** using all 6 `study_stats` keys within 300 words.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch36_todo_05",
        True,
        "passed",
        "To-Do 36.5 Passed — Structured 4-Section Medical Abstract Verified!",
        "Generated a concise <=300-word structured journal abstract with dynamic metrics and strict KeyError validation.",
        points_earned=10,
        points_possible=10,
        tip="Most public health & medical journals (Lancet Global Health, BMJ Open, PLOS Medicine) enforce a strict 250–300 word structured abstract limit.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_todo_05",
        chapter="ch36",
        title="To-Do 36.5: Structured 300-Word Medical Research Abstract (draft_structured_abstract)",
        book_section="Section 36.3.6 (Manuscript Writing & Structured Abstracts)",
        points=10,
        hints={
            1: "Access all 6 keys directly (`study_stats['n_total']`, `'stunting_pct'`, `'best_model'`, `'best_auc'`, `'brier_score'`, `'eo_gap'`) so a missing key raises `KeyError`.",
            2: "Build a multi-paragraph string with `**Background:**`, `**Methods:**`, `**Results:**`, and `**Conclusions:**`, keeping total length ~100–200 words.",
            3: "Compute `word_count = len(abstract_text.split())` and return `{'abstract_text': abstract_text, 'word_count': word_count, 'within_limit': word_count <= 300}`.",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 36.6: audit_manuscript_reproducibility(release_dir)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch36_todo_06", "To-Do 36.6", fn, "audit_manuscript_reproducibility", 10
    )
    if bad is not None:
        return bad

    expected_keys = ["present", "missing", "ready_for_submission"]
    errors = []

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scenarios = [
            ("empty release dir", []),
            ("partial release dir", ["table1.csv", "leaderboard.csv", "MODEL_CARD.md"]),
            ("complete release dir", list(_REQUIRED_RELEASE_FILES)),
        ]
        for label, to_create in scenarios:
            sub = root / label.replace(" ", "_")
            sub.mkdir(parents=True, exist_ok=True)
            for fname in to_create:
                (sub / fname).write_text("ok\n", encoding="utf-8")

            try:
                got = fn(sub)
            except NotImplementedError:
                raise
            except Exception as exc:
                return CheckResult(
                    "ch36_todo_06",
                    False,
                    "failed",
                    "To-Do 36.6 — Runtime Error in `audit_manuscript_reproducibility`",
                    f"Failed on {label} with {type(exc).__name__}: {exc}",
                    points_earned=0,
                    points_possible=10,
                )

            if got is None:
                return _none_return_trap(
                    "ch36_todo_06", "To-Do 36.6", "audit_manuscript_reproducibility(release_dir)", 10
                )

            key_err = _check_dict_keys("ch36_todo_06", "To-Do 36.6", got, expected_keys, 10)
            if key_err is not None:
                return key_err

            exp_present = sorted(to_create)
            exp_missing = sorted(set(_REQUIRED_RELEASE_FILES) - set(to_create))
            exp_ready = len(exp_missing) == 0

            if got["present"] != exp_present:
                errors.append(f"[{label}] present: expected {exp_present}, got {got['present']!r}.")
            if got["missing"] != exp_missing:
                errors.append(f"[{label}] missing: expected {exp_missing}, got {got['missing']!r}.")
            if not _is_bool_like(got["ready_for_submission"]) or bool(got["ready_for_submission"]) is not exp_ready:
                errors.append(
                    f"[{label}] ready_for_submission: expected {exp_ready}, got {got['ready_for_submission']!r}."
                )

    if errors:
        return CheckResult(
            "ch36_todo_06",
            False,
            "failed",
            "To-Do 36.6 — Check Required Release Files & Sorting",
            f"Audit `Path(release_dir)` for {_REQUIRED_RELEASE_FILES} and return sorted `present`, sorted `missing`, and `ready_for_submission`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch36_todo_06",
        True,
        "passed",
        "To-Do 36.6 Passed — Reproducibility Artifact Audit Verified!",
        "Correctly audited empty, partial, and complete manuscript release bundles inside an isolated temporary directory.",
        points_earned=10,
        points_possible=10,
        tip="Run this automated pre-submission gate in CI before pushing your paper's GitHub/Zenodo artifact release.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_todo_06",
        chapter="ch36",
        title="To-Do 36.6: Pre-Submission Reproducibility Gate (audit_manuscript_reproducibility)",
        book_section="Section 36.3.7 (Artifact Packaging & Reproducibility Audits)",
        points=10,
        hints={
            1: "Define the 6 required filenames (`MODEL_CARD.md`, `fairness_audit.json`, `leaderboard.csv`, `manifest.json`, `model.joblib`, `table1.csv`) and convert `release_dir` with `Path(release_dir)`.",
            2: "Partition the required list into `present = sorted(f for f in required if (root / f).is_file())` and `missing = sorted(f for f in required if not (root / f).is_file())`.",
            3: "Return `{'present': present, 'missing': missing, 'ready_for_submission': len(missing) == 0}`.",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# GRAND FINALE EXAM TASK 36.1 (30 pts): full_table1_and_adjusted_or(mch_df)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    bad = _check_callable("ch36_exam_01", "Exam Task 36.1", fn, "full_table1_and_adjusted_or", 30)
    if bad is not None:
        return bad

    mch_orig = _load_mch_df()
    mch_copy = mch_orig.copy(deep=True)
    try:
        got = fn(mch_copy)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch36_exam_01",
            False,
            "failed",
            "Exam Task 36.1 — Runtime Error in `full_table1_and_adjusted_or`",
            f"full_table1_and_adjusted_or(mch_df) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=30,
        )

    if got is None:
        return _none_return_trap(
            "ch36_exam_01", "Exam Task 36.1", "full_table1_and_adjusted_or(mch_df)", 30
        )

    if not mch_copy.equals(mch_orig):
        return CheckResult(
            "ch36_exam_01",
            False,
            "failed",
            "Exam Task 36.1 — Input DataFrame Was Mutated",
            "Never mutate `mch_df`; copy it first.",
            points_earned=0,
            points_possible=30,
        )

    key_err = _check_dict_keys(
        "ch36_exam_01", "Exam Task 36.1", got, ["table1", "adjusted_ors", "n_analytic"], 30
    )
    if key_err is not None:
        return key_err

    t1_err = _validate_table1_df("ch36_exam_01", "Exam Task 36.1", got["table1"], mch_orig, 30)
    if t1_err is not None:
        return t1_err

    # Reference adjusted ORs
    df = _ref_clean_mch(mch_orig)
    cc = df[["child_stunted"] + _CORE_FEATURES].dropna()
    exp_n = len(cc)
    X_sm = sm.add_constant(cc[_CORE_FEATURES].astype(float))
    res = sm.Logit(cc["child_stunted"].astype(float), X_sm).fit(disp=False)
    ci = res.conf_int()

    errors = []
    if not isinstance(got["n_analytic"], (int, np.integer)) or isinstance(got["n_analytic"], bool):
        errors.append(f"`n_analytic` must be an int ({exp_n}), got {type(got['n_analytic']).__name__}.")
    elif int(got["n_analytic"]) != exp_n:
        errors.append(f"`n_analytic`: expected {exp_n} complete cases, got {got['n_analytic']}.")

    or_df = got["adjusted_ors"]
    expected_or_cols = ["feature", "aor", "ci_low", "ci_high", "p_value"]
    if not isinstance(or_df, pd.DataFrame) or list(or_df.columns) != expected_or_cols:
        errors.append(
            f"`adjusted_ors` must be a DataFrame with columns {expected_or_cols}, got {type(or_df).__name__} ({getattr(or_df, 'columns', None)})."
        )
    else:
        if "const" in set(or_df["feature"]):
            errors.append("Exclude the intercept (`'const'`) row from `adjusted_ors`.")
        if set(or_df["feature"]) != set(_CORE_FEATURES) or len(or_df) != len(_CORE_FEATURES):
            errors.append(f"Expected 5 rows for {_CORE_FEATURES}, got {list(or_df['feature'])}.")
        else:
            by_feat = or_df.set_index("feature")
            for feat in _CORE_FEATURES:
                exp_aor = round(float(np.exp(res.params[feat])), 4)
                exp_lo = round(float(np.exp(ci.loc[feat, 0])), 4)
                exp_hi = round(float(np.exp(ci.loc[feat, 1])), 4)
                exp_p = round(float(res.pvalues[feat]), 4)
                row = by_feat.loc[feat]
                if not math.isclose(float(row["aor"]), exp_aor, abs_tol=1e-3):
                    errors.append(f"{feat} aor: expected {exp_aor}, got {row['aor']!r}.")
                if not math.isclose(float(row["ci_low"]), exp_lo, abs_tol=1e-3):
                    errors.append(f"{feat} ci_low: expected {exp_lo}, got {row['ci_low']!r}.")
                if not math.isclose(float(row["ci_high"]), exp_hi, abs_tol=1e-3):
                    errors.append(f"{feat} ci_high: expected {exp_hi}, got {row['ci_high']!r}.")
                if not math.isclose(float(row["p_value"]), exp_p, abs_tol=1e-3):
                    errors.append(f"{feat} p_value: expected {exp_p}, got {row['p_value']!r}.")

    if errors:
        return CheckResult(
            "ch36_exam_01",
            False,
            "failed",
            "Exam Task 36.1 — Check Multivariable Logit & Adjusted Odds Ratios",
            "Fit `statsmodels.api.Logit` on complete cases (`n=470`) with `sm.add_constant(X)`, exponentiate parameters & 95% CIs, and exclude `'const'`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch36_exam_01",
        True,
        "passed",
        "Exam Task 36.1 Passed (30/30 pts) — Table 1 & Multivariable Adjusted ORs Verified!",
        "Confirmed n=470 complete cases and verified household wealth (aOR=0.5835, p<0.0001) as the dominant independent protective factor.",
        points_earned=30,
        points_possible=30,
        tip="Pairing Table 1 with multivariable adjusted ORs reveals socioeconomic confounding immediately to epidemiological reviewers.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_exam_01",
        chapter="ch36",
        title="Exam Task 36.1 (30 pts): Full Table 1 & Adjusted Odds Ratios (full_table1_and_adjusted_or)",
        book_section="Section 36.7 (Grand Book Finale Capstone Exam)",
        points=30,
        is_exam=True,
        hints={
            1: "Call `build_publication_table1(mch_df)` for `table1`, then map `wealth_quintile` to `wealth_rank` (1..5) and drop rows with missing values in `['child_stunted'] + features`.",
            2: "Fit `res = sm.Logit(cc['child_stunted'].astype(float), sm.add_constant(cc[features].astype(float))).fit(disp=False)` and extract `ci = res.conf_int()`.",
            3: "Loop over `features` (skipping `'const'`), exponentiate `params` and `ci` bounds rounded to 4 decimals, and return `{'table1': table1, 'adjusted_ors': or_df, 'n_analytic': int(len(cc))}`.",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# GRAND FINALE EXAM TASK 36.2 (35 pts): capstone_ml_dl_fairness_pipeline(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable(
        "ch36_exam_02", "Exam Task 36.2", fn, "capstone_ml_dl_fairness_pipeline", 35
    )
    if bad is not None:
        return bad

    mch_orig = _load_mch_df()
    mch_copy = mch_orig.copy(deep=True)
    try:
        got = fn(mch_copy, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch36_exam_02",
            False,
            "failed",
            "Exam Task 36.2 — Runtime Error in `capstone_ml_dl_fairness_pipeline`",
            f"capstone_ml_dl_fairness_pipeline(mch_df, seed=42) raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap(
            "ch36_exam_02", "Exam Task 36.2", "capstone_ml_dl_fairness_pipeline(mch_df, seed=42)", 35
        )

    if not mch_copy.equals(mch_orig):
        return CheckResult(
            "ch36_exam_02",
            False,
            "failed",
            "Exam Task 36.2 — Input DataFrame Was Mutated",
            "Copy `mch_df` before cleaning so the caller's DataFrame stays untouched.",
            points_earned=0,
            points_possible=35,
        )

    key_err = _check_dict_keys(
        "ch36_exam_02",
        "Exam Task 36.2",
        got,
        ["leaderboard", "winner", "fairness_gaps", "dca_table"],
        35,
    )
    if key_err is not None:
        return key_err

    errors = []
    lb = got["leaderboard"]
    lb_cols = ["model", "mean_auc", "std_auc", "mean_brier"]
    if not isinstance(lb, pd.DataFrame) or list(lb.columns) != lb_cols or len(lb) != 3:
        errors.append(
            f"`leaderboard` must be a 3-row DataFrame with columns {lb_cols} (LR, HGB, PyTorch MLP)."
        )
    else:
        names_low = " ".join(str(m).lower() for m in lb["model"])
        for label, tokens in [
            ("LogisticRegression", ("logistic",)),
            ("HistGradientBoostingClassifier", ("boost", "hgb")),
            ("PyTorch_MLP", ("mlp", "torch", "neural")),
        ]:
            if not any(tok in names_low for tok in tokens):
                errors.append(f"`leaderboard` is missing {label}; got {list(lb['model'])}.")
        aucs = [float(x) for x in lb["mean_auc"]]
        if aucs != sorted(aucs, reverse=True):
            errors.append(f"`leaderboard` must be sorted by `mean_auc` descending; got {aucs}.")
        for m, a in zip(lb["model"], aucs):
            if a > 0.90:
                errors.append(f"{m}: mean_auc={a} suggests `child_haz` target leakage!")
            elif not (0.55 <= a <= 0.85):
                errors.append(f"{m}: mean_auc={a} outside expected range [0.55, 0.85].")
        if str(got["winner"]) != str(lb.iloc[0]["model"]):
            errors.append(
                f"`winner` ({got['winner']!r}) must match top leaderboard model ({lb.iloc[0]['model']!r})."
            )

    fair = got["fairness_gaps"]
    if not isinstance(fair, dict):
        errors.append(f"`fairness_gaps` must be a dict, got {type(fair).__name__}.")
    else:
        for req_k in ["demographic_parity_diff", "equal_opportunity_diff"]:
            if req_k not in fair or not _is_number(fair[req_k]):
                errors.append(f"`fairness_gaps` must include numeric `{req_k}`.")
            elif not (0.0 <= float(fair[req_k]) <= 1.0):
                errors.append(f"`fairness_gaps['{req_k}']`={fair[req_k]} outside [0, 1].")
        if "threshold" in fair and (
            not _is_number(fair["threshold"]) or not (0.0 < float(fair["threshold"]) < 1.0)
        ):
            errors.append(f"`fairness_gaps['threshold']`={fair['threshold']} outside (0, 1).")

    dca = got["dca_table"]
    dca_cols = ["threshold", "nb_model", "nb_all", "nb_none"]
    exp_ts = [0.10, 0.15, 0.20, 0.25, 0.30]
    if not isinstance(dca, pd.DataFrame) or list(dca.columns) != dca_cols or len(dca) != 5:
        errors.append(
            f"`dca_table` must be a 5-row DataFrame with columns {dca_cols} for thresholds {exp_ts}."
        )
    else:
        prev = 0.14  # 70 / 500 on mch_survey_sample
        for idx, t in enumerate(exp_ts):
            row = dca.iloc[idx]
            if not math.isclose(float(row["threshold"]), t, abs_tol=1e-4):
                errors.append(f"DCA row {idx}: expected threshold={t}, got {row['threshold']!r}.")
            exp_all = round(prev - (1.0 - prev) * (t / (1.0 - t)), 5)
            if not math.isclose(float(row["nb_all"]), exp_all, abs_tol=1e-4):
                errors.append(f"DCA t={t}: expected nb_all={exp_all}, got {row['nb_all']!r}.")
            if not math.isclose(float(row["nb_none"]), 0.0, abs_tol=1e-5):
                errors.append(f"DCA t={t}: expected nb_none=0.0, got {row['nb_none']!r}.")
        if float(dca.iloc[1]["nb_model"]) <= float(dca.iloc[1]["nb_all"]):
            errors.append("At t=0.15, winning model `nb_model` should exceed `nb_all`.")

    if errors:
        return CheckResult(
            "ch36_exam_02",
            False,
            "failed",
            "Exam Task 36.2 — Check ML/DL Benchmark, Youden Fairness Audit & DCA Table",
            "Evaluate LR, HGB, and PyTorch MLP under 5-fold GroupKFold, audit Rural vs Urban gaps on OOF predictions at Youden's J threshold, and compute DCA across [0.10..0.30].",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch36_exam_02",
        True,
        "passed",
        "Exam Task 36.2 Passed (35/35 pts) — ML/DL Benchmark, Fairness Audit & DCA Verified!",
        f"Top model: {got['winner']} (AUC={lb.iloc[0]['mean_auc']:.4f}); verified Rural vs Urban fairness gaps and positive clinical net benefit.",
        points_earned=35,
        points_possible=35,
        tip="Always compute subgroup fairness and Decision Curve Analysis strictly on out-of-fold (or held-out) probabilities, never in-sample predictions.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_exam_02",
        chapter="ch36",
        title="Exam Task 36.2 (35 pts): ML/DL Benchmark, Fairness & DCA Pipeline (capstone_ml_dl_fairness_pipeline)",
        book_section="Section 36.7 (Grand Book Finale Capstone Exam)",
        points=35,
        is_exam=True,
        hints={
            1: "Run 5-fold `GroupKFold(n_splits=5)` on `cluster_id` for `LogisticRegression`, `HistGradientBoostingClassifier`, and `PyTorch_MLP`, storing full out-of-fold probability arrays `oof[va] = probs` for each model.",
            2: "Select `winner = leaderboard.iloc[0]['model']`, compute Youden's J threshold via `fpr, tpr, thresh = roc_curve(y, oof_w); yt = float(thresh[np.argmax(tpr - fpr)])`, and compute Rural (`is_urban == 0`) vs Urban (`is_urban == 1`) positive-prediction rate gap and TPR gap.",
            3: "Call `compute_clinical_net_benefit(y, oof_w, [0.10, 0.15, 0.20, 0.25, 0.30])` and return `{'leaderboard': lb, 'winner': winner, 'fairness_gaps': fair, 'dca_table': dca}`.",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# GRAND FINALE EXAM TASK 36.3 (35 pts): execute_first_paper_release(mch_df, release_dir, seed=42)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    import joblib
    import pandas as pd

    bad = _check_callable(
        "ch36_exam_03", "Exam Task 36.3", fn, "execute_first_paper_release", 35
    )
    if bad is not None:
        return bad

    mch_orig = _load_mch_df()
    mch_copy = mch_orig.copy(deep=True)
    expected_keys = ["ready_for_submission", "winner", "winner_auc", "artifact_sha256"]

    with tempfile.TemporaryDirectory() as tmp:
        release_path = Path(tmp) / "first_paper_release"
        try:
            got = fn(mch_copy, release_path, seed=42)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch36_exam_03",
                False,
                "failed",
                "Exam Task 36.3 — Runtime Error in `execute_first_paper_release`",
                f"execute_first_paper_release(mch_df, release_dir, seed=42) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch36_exam_03", "Exam Task 36.3", "execute_first_paper_release(mch_df, release_dir, seed=42)", 35
            )

        if not mch_copy.equals(mch_orig):
            return CheckResult(
                "ch36_exam_03",
                False,
                "failed",
                "Exam Task 36.3 — Input DataFrame Was Mutated",
                "Copy `mch_df` before cleaning so the caller's table stays unmodified.",
                points_earned=0,
                points_possible=35,
            )

        key_err = _check_dict_keys("ch36_exam_03", "Exam Task 36.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        errors = []
        all_expected_files = _REQUIRED_RELEASE_FILES + ["config.json"]
        for fname in all_expected_files:
            fpath = release_path / fname
            if not fpath.is_file() or fpath.stat().st_size == 0:
                errors.append(f"Missing or empty release artifact: `{fname}`.")

        if not _is_bool_like(got["ready_for_submission"]) or not bool(got["ready_for_submission"]):
            errors.append(
                f"Expected `ready_for_submission=True`, got {got['ready_for_submission']!r}."
            )

        if not isinstance(got["winner"], str) or not got["winner"].strip():
            errors.append(f"Expected non-empty model name for `winner`, got {got['winner']!r}.")

        if not _is_number(got["winner_auc"]) or not (0.55 <= float(got["winner_auc"]) <= 0.85):
            errors.append(
                f"`winner_auc` ({got['winner_auc']!r}) outside expected leak-free range [0.55, 0.85]."
            )

        model_file = release_path / "model.joblib"
        if model_file.is_file():
            actual_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
            if got["artifact_sha256"] != actual_sha:
                errors.append(
                    f"`artifact_sha256` mismatch: returned {got['artifact_sha256']!r}, "
                    f"actual SHA-256 of `model.joblib` is {actual_sha!r}."
                )
            try:
                loaded_model = joblib.load(model_file)
                if not hasattr(loaded_model, "predict_proba"):
                    errors.append("Loaded `model.joblib` must expose `.predict_proba(...)` (use a standard sklearn Pipeline).")
                else:
                    clean_sample = _ref_clean_mch(mch_orig)[_CORE_FEATURES].iloc[:3].astype(float)
                    probs = loaded_model.predict_proba(clean_sample)
                    if probs.shape != (3, 2):
                        errors.append(f"Loaded model predict_proba returned shape {probs.shape}, expected (3, 2).")
            except Exception as exc:
                errors.append(f"Failed to reload/predict with `model.joblib`: {type(exc).__name__}: {exc}")

            manifest_file = release_path / "manifest.json"
            if manifest_file.is_file():
                try:
                    manifest_raw = manifest_file.read_text(encoding="utf-8")
                    json.loads(manifest_raw)
                    if actual_sha not in manifest_raw:
                        errors.append("`manifest.json` must record the exact SHA-256 digest of `model.joblib`.")
                except Exception as exc:
                    errors.append(f"`manifest.json` is not valid JSON: {exc}")

        for jf in ["fairness_audit.json", "config.json"]:
            p = release_path / jf
            if p.is_file():
                try:
                    json.loads(p.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(f"`{jf}` is not valid JSON: {exc}")

        for cf in ["table1.csv", "leaderboard.csv"]:
            p = release_path / cf
            if p.is_file():
                try:
                    df_chk = pd.read_csv(p)
                    if len(df_chk) == 0:
                        errors.append(f"`{cf}` has 0 rows.")
                except Exception as exc:
                    errors.append(f"`{cf}` failed to load via pandas.read_csv: {exc}")

    if errors:
        return CheckResult(
            "ch36_exam_03",
            False,
            "failed",
            "Exam Task 36.3 — Check Release Bundle Artifacts, Joblib Pipeline & SHA-256 Manifest",
            "Write all 7 files (`table1.csv`, `leaderboard.csv`, `fairness_audit.json`, `MODEL_CARD.md`, `model.joblib`, `config.json`, `manifest.json`) into `release_dir`, embed the exact SHA-256 of `model.joblib`, and audit with `audit_manuscript_reproducibility`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch36_exam_03",
        True,
        "passed",
        "Exam Task 36.3 Passed (35/35 pts) — Publication Release Bundle & SHA-256 Manifest Verified!",
        f"Verified all 7 artifacts, reloadable sklearn Pipeline (`model.joblib`), and cryptographic SHA-256 integrity ({got['artifact_sha256'][:12]}...).",
        points_earned=35,
        points_possible=35,
        tip="Congratulations — you just shipped a journal-grade, cryptographically verifiable research release from raw CSV to submission-ready artifact!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch36_exam_03",
        chapter="ch36",
        title="Exam Task 36.3 (35 pts): End-to-End First Paper Release Bundle (execute_first_paper_release)",
        book_section="Section 36.7 (Grand Book Finale Capstone Exam)",
        points=35,
        is_exam=True,
        hints={
            1: "Create `release_path = Path(release_dir); release_path.mkdir(parents=True, exist_ok=True)`. Run `build_publication_table1(mch_df)` and `capstone_ml_dl_fairness_pipeline(mch_df, seed=seed)`.",
            2: "Fit a standard sklearn `Pipeline` (`SimpleImputer` + `StandardScaler` + `LogisticRegression`) on the cleaned features, save with `joblib.dump(..., release_path / 'model.joblib')`, and compute `sha256 = hashlib.sha256((release_path / 'model.joblib').read_bytes()).hexdigest()`.",
            3: "Write `table1.csv`, `leaderboard.csv`, `fairness_audit.json`, `MODEL_CARD.md`, `config.json`, and `manifest.json` (including `sha256`), audit with `audit_manuscript_reproducibility(release_path)`, and return `{'ready_for_submission': audit['ready_for_submission'], 'winner': winner, 'winner_auc': round(float(winner_auc), 4), 'artifact_sha256': sha256}`.",
        },
        verifier=_verify_exam_03,
    )
)
