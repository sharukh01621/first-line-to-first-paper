"""Chapter 20 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch20_todo_01..06) and the Part IV Capstone Coding Exam
(ch20_exam_01..03) for Probability, Statistical Inference, Hypothesis Testing,
and Epidemiological Metrics in Python.
Every verifier lazy-imports NumPy/Pandas/SciPy/Statsmodels inside the check
function, validates return structures, rounding, mathematical formulas, and
non-mutation, while letting NotImplementedError propagate cleanly for starter
stubs.
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
        f"Calling {call_str} returned None instead of the computed statistical summary.",
        details=[
            "Remember: print() only displays text on screen; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reusable biostatistical helpers always return structured tuples, dicts, or DataFrames.",
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
# GUIDED TO-DO 20.1: wilson_ci(successes, n, z=1.96)
# ---------------------------------------------------------------------------
def _ref_wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = successes / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + (z ** 2) / (4.0 * n)) / n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (round(p, 4), round(low, 4), round(high, 4))


def _verify_todo_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch20_todo_01", "To-Do 20.1", fn, "wilson_ci", 10)
    if bad is not None:
        return bad

    cases = [
        ((18, 50), {}, "wilson_ci(18, 50)"),
        ((179, 500), {}, "wilson_ci(179, 500)"),
        ((0, 20), {}, "wilson_ci(0, 20)"),
        ((18, 50), {"z": 2.576}, "wilson_ci(18, 50, z=2.576)"),
    ]

    errors = []
    for args, kwargs, label in cases:
        try:
            got = fn(*args, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch20_todo_01",
                False,
                "failed",
                "To-Do 20.1 — Check Function Signature `wilson_ci(successes, n, z=1.96)`",
                f"{label} raised TypeError: {exc}",
                details=[
                    "Define `def wilson_ci(successes, n, z=1.96):` with default `z=1.96`."
                ],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch20_todo_01",
                False,
                "failed",
                "To-Do 20.1 — Runtime Error in `wilson_ci`",
                f"{label} raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch20_todo_01", "To-Do 20.1", label, 10)

        if not isinstance(got, tuple) or len(got) != 3:
            return CheckResult(
                "ch20_todo_01",
                False,
                "failed",
                "To-Do 20.1 — Return a 3-Tuple `(prop, low, high)`",
                f"{label} returned {type(got).__name__} ({got!r}); expected a 3-tuple `(prop, low, high)`.",
                points_earned=0,
                points_possible=10,
            )

        exp = _ref_wilson(*args, **kwargs)
        if not all(_is_number(v) for v in got):
            errors.append(f"{label}: all 3 tuple elements must be floats, got {got!r}.")
            continue

        if args == (0, 20) and math.isclose(got[2], 0.0, abs_tol=1e-4):
            errors.append(
                f"{label} returned high={got[2]} — that looks like the naive Wald CI! "
                "Use the Wilson score formula so zero-event samples still yield a non-zero upper bound (0.1611)."
            )
            continue

        if not all(math.isclose(float(g), e, abs_tol=1e-4) for g, e in zip(got, exp)):
            extra = (
                " (Make sure you use the parameter `z`, not a hardcoded 1.96!)"
                if "z" in kwargs
                else ""
            )
            errors.append(f"{label}: expected {exp}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch20_todo_01",
            False,
            "failed",
            "To-Do 20.1 — Check Wilson Score Formula & 4-Decimal Rounding",
            "Compute proportion `p = successes / n`, denominator `1 + z**2 / n`, center, and Wilson margin rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch20_todo_01",
        True,
        "passed",
        "To-Do 20.1 Passed — Wilson Score Confidence Interval Verified!",
        "Passed standard prevalence (35.8% anemia -> [0.3172, 0.401]), boundary zero-event (0/20 -> [0.0, 0.1611]), and 99% CI (z=2.576) checks.",
        tip="Wilson intervals never overshoot [0, 1] and stay valid even when events are rare or zero.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_todo_01",
        chapter="ch20",
        title="To-Do 20.1: Proportion & 95% Wilson Score CI (wilson_ci)",
        book_section="Section 20.3.2 (Confidence Intervals for Proportions)",
        points=10,
        hints={
            1: "Compute `p = successes / n`, `denom = 1 + (z ** 2) / n`, `center = (p + (z ** 2) / (2 * n)) / denom`, and `margin = (z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n)) / denom`.",
            2: "Return `(round(p, 4), round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))` as a 3-tuple of floats.",
            3: "def wilson_ci(successes, n, z=1.96):\n    p = successes / n\n    denom = 1 + (z ** 2) / n\n    center = (p + (z ** 2) / (2 * n)) / denom\n    margin = (z * math.sqrt((p * (1 - p) + (z ** 2) / (4 * n)) / n)) / ___\n    return (round(p, 4), round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 20.2: compare_two_groups_welch(group_a, group_b)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np
    from scipy import stats

    bad = _check_callable("ch20_todo_02", "To-Do 20.2", fn, "compare_two_groups_welch", 10)
    if bad is not None:
        return bad

    cases = [
        (
            [12.4, 11.9, 12.8, 12.1, 13.0, 12.5, 11.8, 12.6],
            [10.2, 9.5, 11.4, 10.8, 9.1, 12.0, 10.1, 9.8, 10.9, 11.2, 9.4, 10.5],
            "significant unequal-variance cohorts",
        ),
        (
            [11.4, 11.2, 11.8, 10.9, 11.5, 11.1],
            [11.3, 11.0, 11.6, 11.2, 11.4, 10.8],
            "non-significant overlapping cohorts",
        ),
    ]

    expected_keys = ["t_stat", "p_value", "cohens_d", "significant_05"]
    errors = []

    for ga_raw, gb_raw, desc in cases:
        ga_copy, gb_copy = list(ga_raw), list(gb_raw)
        try:
            got = fn(ga_copy, gb_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch20_todo_02",
                False,
                "failed",
                "To-Do 20.2 — Runtime Error in `compare_two_groups_welch`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch20_todo_02", "To-Do 20.2", "compare_two_groups_welch(a, b)", 10)

        key_err = _check_dict_keys("ch20_todo_02", "To-Do 20.2", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        if ga_copy != ga_raw or gb_copy != gb_raw:
            errors.append(f"Mutated input list(s) on {desc}! Never modify caller inputs.")

        a = np.asarray(ga_raw, dtype=float)
        b = np.asarray(gb_raw, dtype=float)
        welch = stats.ttest_ind(a, b, equal_var=False)
        student = stats.ttest_ind(a, b, equal_var=True)
        pooled_sd = math.sqrt(
            ((len(a) - 1) * float(np.var(a, ddof=1)) + (len(b) - 1) * float(np.var(b, ddof=1)))
            / (len(a) + len(b) - 2)
        )
        exp_t = round(float(welch.statistic), 4)
        exp_p = round(float(welch.pvalue), 4)
        exp_d = round((float(np.mean(a)) - float(np.mean(b))) / pooled_sd, 4)
        exp_sig = bool(welch.pvalue < 0.05)

        if not _is_number(got["t_stat"]) or not math.isclose(got["t_stat"], exp_t, abs_tol=1e-4):
            extra = ""
            if _is_number(got["t_stat"]) and math.isclose(got["t_stat"], round(float(student.statistic), 4), abs_tol=1e-3):
                extra = " (Looks like Student's t-test — pass `equal_var=False` for Welch's t-test!)"
            errors.append(f"[{desc}] t_stat: expected {exp_t}, got {got['t_stat']!r}{extra}")

        if not _is_number(got["p_value"]) or not math.isclose(got["p_value"], exp_p, abs_tol=1e-4):
            errors.append(f"[{desc}] p_value: expected {exp_p}, got {got['p_value']!r}")

        if not _is_number(got["cohens_d"]) or not math.isclose(got["cohens_d"], exp_d, abs_tol=1e-3):
            extra = ""
            if _is_number(got["cohens_d"]) and math.isclose(got["cohens_d"], -exp_d, abs_tol=1e-3):
                extra = " (Sign is flipped — compute `(mean_a - mean_b) / pooled_sd`!)"
            errors.append(f"[{desc}] cohens_d: expected {exp_d}, got {got['cohens_d']!r}{extra}")

        if not _is_bool_like(got["significant_05"]) or bool(got["significant_05"]) is not exp_sig:
            errors.append(f"[{desc}] significant_05: expected {exp_sig}, got {got['significant_05']!r}")

    if errors:
        return CheckResult(
            "ch20_todo_02",
            False,
            "failed",
            "To-Do 20.2 — Check Welch's t-Test (`equal_var=False`) & Pooled Cohen's d",
            "Run `scipy.stats.ttest_ind(a, b, equal_var=False)` and compute `(mean_a - mean_b) / pooled_sd` with sample variances (`ddof=1`).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch20_todo_02",
        True,
        "passed",
        "To-Do 20.2 Passed — Welch's t-Test & Pooled Cohen's d Verified!",
        "Verified unequal-variance Welch t-statistic, p-value, pooled Cohen's d sign/magnitude, and alpha=0.05 significance flag.",
        tip="Always pair a p-value with an effect size (Cohen's d) so readers see clinical magnitude, not just sample-size sensitivity.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_todo_02",
        chapter="ch20",
        title="To-Do 20.2: Welch's t-Test & Pooled Cohen's d (compare_two_groups_welch)",
        book_section="Section 20.3.3 (Parametric Comparison & Effect Size)",
        points=10,
        hints={
            1: "Call `stats.ttest_ind(a, b, equal_var=False)` for Welch's test. Compute pooled SD from sample variances (`ddof=1`): `sqrt(((n_a - 1)*var_a + (n_b - 1)*var_b) / (n_a + n_b - 2))`.",
            2: "Compute `cohens_d = (np.mean(a) - np.mean(b)) / pooled_sd`, round `t_stat`, `p_value`, and `cohens_d` to 4 decimals, and set `significant_05 = bool(res.pvalue < 0.05)`.",
            3: "def compare_two_groups_welch(group_a, group_b):\n    a = np.asarray(group_a, dtype=float)\n    b = np.asarray(group_b, dtype=float)\n    res = stats.ttest_ind(a, b, equal_var=___)\n    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)\n    pooled_sd = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))\n    d = (float(np.mean(a)) - float(np.mean(b))) / pooled_sd\n    return {'t_stat': round(float(res.statistic), 4), 'p_value': round(float(res.pvalue), 4), 'cohens_d': round(float(d), 4), 'significant_05': bool(res.pvalue < 0.05)}",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 20.3: nonparametric_mannwhitney(group_a, group_b)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    from scipy import stats

    bad = _check_callable("ch20_todo_03", "To-Do 20.3", fn, "nonparametric_mannwhitney", 10)
    if bad is not None:
        return bad

    cases = [
        ([5, 6, 4, 7, 8, 5, 6, 9, 4, 6], [1, 2, 3, 2, 0, 3, 4, 2, 1, 3], "skewed ANC visit counts"),
        ([3, 4, 2, 5, 4, 3], [3, 4, 3, 5, 2, 4], "tied non-significant groups"),
    ]
    expected_keys = ["u_stat", "p_value", "significant_05"]
    errors = []

    for ga_raw, gb_raw, desc in cases:
        ga_copy, gb_copy = list(ga_raw), list(gb_raw)
        try:
            got = fn(ga_copy, gb_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch20_todo_03",
                False,
                "failed",
                "To-Do 20.3 — Runtime Error in `nonparametric_mannwhitney`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch20_todo_03", "To-Do 20.3", "nonparametric_mannwhitney(a, b)", 10)

        key_err = _check_dict_keys("ch20_todo_03", "To-Do 20.3", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        if ga_copy != ga_raw or gb_copy != gb_raw:
            errors.append(f"Mutated input list(s) on {desc}!")

        res = stats.mannwhitneyu(ga_raw, gb_raw, alternative="two-sided")
        exp_u = round(float(res.statistic), 2)
        exp_p = round(float(res.pvalue), 4)
        exp_sig = bool(res.pvalue < 0.05)

        if not _is_number(got["u_stat"]) or not math.isclose(got["u_stat"], exp_u, abs_tol=1e-2):
            errors.append(f"[{desc}] u_stat: expected {exp_u}, got {got['u_stat']!r}")
        if not _is_number(got["p_value"]) or not math.isclose(got["p_value"], exp_p, abs_tol=1e-4):
            errors.append(f"[{desc}] p_value: expected {exp_p}, got {got['p_value']!r}")
        if not _is_bool_like(got["significant_05"]) or bool(got["significant_05"]) is not exp_sig:
            errors.append(f"[{desc}] significant_05: expected {exp_sig}, got {got['significant_05']!r}")

    if errors:
        return CheckResult(
            "ch20_todo_03",
            False,
            "failed",
            "To-Do 20.3 — Check Mann-Whitney U Call & Rounding",
            "Call `scipy.stats.mannwhitneyu(group_a, group_b, alternative='two-sided')`, rounding `u_stat` to 2 decimals and `p_value` to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch20_todo_03",
        True,
        "passed",
        "To-Do 20.3 Passed — Nonparametric Mann-Whitney U Test Verified!",
        "Correctly evaluated rank-sum U statistics, two-sided p-values, and significance flags on skewed and tied clinical data.",
        tip="Reach for Mann-Whitney U whenever count or biomarker distributions are skewed or contain extreme outliers.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_todo_03",
        chapter="ch20",
        title="To-Do 20.3: Nonparametric Mann-Whitney U Test (nonparametric_mannwhitney)",
        book_section="Section 20.3.4 (Nonparametric Rank Tests)",
        points=10,
        hints={
            1: "Call `stats.mannwhitneyu(group_a, group_b, alternative='two-sided')` and extract `res.statistic` and `res.pvalue`.",
            2: "Round `u_stat` to 2 decimals, `p_value` to 4 decimals, and set `significant_05 = bool(res.pvalue < 0.05)`.",
            3: "def nonparametric_mannwhitney(group_a, group_b):\n    res = stats.mannwhitneyu(group_a, group_b, alternative='___')\n    return {\n        'u_stat': round(float(res.statistic), 2),\n        'p_value': round(float(res.pvalue), 4),\n        'significant_05': bool(res.pvalue < 0.05),\n    }",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 20.4: chi2_contingency_audit(table_2x2)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np
    from scipy import stats

    bad = _check_callable("ch20_todo_04", "To-Do 20.4", fn, "chi2_contingency_audit", 10)
    if bad is not None:
        return bad

    tables = [
        ([[133, 223], [46, 98]], "MCH rural vs urban anemia 2x2"),
        ([[42, 18], [15, 45]], "significant exposure 2x2"),
        ([[1, 14], [7, 8]], "sparse expected-count (<5) 2x2"),
    ]
    expected_keys = ["chi2", "p_value", "dof", "expected_min"]
    errors = []

    for tbl, desc in tables:
        try:
            got = fn([row[:] for row in tbl])
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch20_todo_04",
                False,
                "failed",
                "To-Do 20.4 — Runtime Error in `chi2_contingency_audit`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch20_todo_04", "To-Do 20.4", "chi2_contingency_audit(table_2x2)", 10)

        key_err = _check_dict_keys("ch20_todo_04", "To-Do 20.4", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        chi2, p_val, dof, expected = stats.chi2_contingency(tbl, correction=False)
        chi2_yates, _, _, _ = stats.chi2_contingency(tbl, correction=True)

        exp_chi2 = round(float(chi2), 4)
        exp_p = round(float(p_val), 4)
        exp_dof = int(dof)
        exp_min = round(float(np.min(expected)), 2)

        if not _is_number(got["chi2"]) or not math.isclose(got["chi2"], exp_chi2, abs_tol=1e-4):
            extra = ""
            if _is_number(got["chi2"]) and math.isclose(got["chi2"], round(float(chi2_yates), 4), abs_tol=1e-3):
                extra = " (Pass `correction=False` for standard Pearson chi-square without Yates' correction!)"
            errors.append(f"[{desc}] chi2: expected {exp_chi2}, got {got['chi2']!r}{extra}")
        if not _is_number(got["p_value"]) or not math.isclose(got["p_value"], exp_p, abs_tol=1e-4):
            errors.append(f"[{desc}] p_value: expected {exp_p}, got {got['p_value']!r}")
        if not isinstance(got["dof"], int) or isinstance(got["dof"], bool) or got["dof"] != exp_dof:
            errors.append(f"[{desc}] dof: expected int {exp_dof}, got {got['dof']!r}")
        if not _is_number(got["expected_min"]) or not math.isclose(got["expected_min"], exp_min, abs_tol=1e-2):
            errors.append(f"[{desc}] expected_min: expected {exp_min}, got {got['expected_min']!r}")

    if errors:
        return CheckResult(
            "ch20_todo_04",
            False,
            "failed",
            "To-Do 20.4 — Check Pearson Chi-Square (`correction=False`) & Expected Count Audit",
            "Call `stats.chi2_contingency(table_2x2, correction=False)` and report `chi2`, `p_value`, `dof`, and `round(float(np.min(expected)), 2)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch20_todo_04",
        True,
        "passed",
        "To-Do 20.4 Passed — Chi-Square Contingency & Expected-Cell Audit Verified!",
        "Verified Pearson chi2 (correction=False), p-value, integer dof, and minimum expected cell count across standard and sparse tables.",
        tip="Whenever `expected_min < 5.0`, switch from Pearson chi-square to `scipy.stats.fisher_exact(table_2x2)`!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_todo_04",
        chapter="ch20",
        title="To-Do 20.4: Pearson Chi-Square & Expected-Cell Audit (chi2_contingency_audit)",
        book_section="Section 20.3.5 (Categorical Tests: Chi-Square & Fisher's Exact)",
        points=10,
        hints={
            1: "Unpack `chi2, p_val, dof, expected = stats.chi2_contingency(table_2x2, correction=False)` and inspect `expected.min()`.",
            2: "Round `chi2` and `p_val` to 4 decimals, cast `dof` to `int(dof)`, and round `float(np.min(expected))` to 2 decimals.",
            3: "def chi2_contingency_audit(table_2x2):\n    chi2, p_val, dof, expected = stats.chi2_contingency(table_2x2, correction=___)\n    return {\n        'chi2': round(float(chi2), 4),\n        'p_value': round(float(p_val), 4),\n        'dof': int(dof),\n        'expected_min': round(float(np.min(expected)), 2),\n    }",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 20.5: odds_ratio_with_ci(a, b, c, d)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    bad = _check_callable("ch20_todo_05", "To-Do 20.5", fn, "odds_ratio_with_ci", 10)
    if bad is not None:
        return bad

    cases = [
        ((133, 223, 46, 98), "MCH rural vs urban anemia"),
        ((42, 18, 15, 45), "elevated risk exposure"),
        ((12, 48, 30, 30), "protective exposure (OR < 1)"),
    ]
    expected_keys = ["or", "ci_low", "ci_high"]
    errors = []

    for (a, b, c, d), desc in cases:
        try:
            got = fn(a, b, c, d)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch20_todo_05",
                False,
                "failed",
                "To-Do 20.5 — Runtime Error in `odds_ratio_with_ci`",
                f"odds_ratio_with_ci({a}, {b}, {c}, {d}) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch20_todo_05", "To-Do 20.5", f"odds_ratio_with_ci({a}, {b}, {c}, {d})", 10)

        key_err = _check_dict_keys("ch20_todo_05", "To-Do 20.5", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        or_val = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        log_or = math.log(or_val)
        exp_or = round(or_val, 3)
        exp_low = round(math.exp(log_or - 1.96 * se), 3)
        exp_high = round(math.exp(log_or + 1.96 * se), 3)

        if not _is_number(got["or"]) or not math.isclose(got["or"], exp_or, abs_tol=1e-3):
            extra = ""
            if _is_number(got["or"]) and math.isclose(got["or"], round(1 / or_val, 3), abs_tol=1e-3):
                extra = " (OR is inverted — compute `(a * d) / (b * c)`!)"
            errors.append(f"[{desc}] or: expected {exp_or}, got {got['or']!r}{extra}")
        if not _is_number(got["ci_low"]) or not math.isclose(got["ci_low"], exp_low, abs_tol=1e-3):
            errors.append(f"[{desc}] ci_low: expected {exp_low}, got {got['ci_low']!r}")
        if not _is_number(got["ci_high"]) or not math.isclose(got["ci_high"], exp_high, abs_tol=1e-3):
            errors.append(f"[{desc}] ci_high: expected {exp_high}, got {got['ci_high']!r}")

    if errors:
        return CheckResult(
            "ch20_todo_05",
            False,
            "failed",
            "To-Do 20.5 — Check Woolf Log-OR 95% CI Formula",
            "Compute `or_val = (a * d) / (b * c)`, `se = sqrt(1/a + 1/b + 1/c + 1/d)`, and exponentiate `log(or_val) +/- 1.96 * se` rounded to 3 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch20_todo_05",
        True,
        "passed",
        "To-Do 20.5 Passed — Odds Ratio & Woolf 95% CI Verified!",
        "Passed null-crossing (OR=1.271), risk-increasing (OR=7.000), and protective (OR=0.250) 2x2 tables.",
        tip="Remember: OR confidence intervals are symmetric on the log scale and asymmetric on the original odds scale.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_todo_05",
        chapter="ch20",
        title="To-Do 20.5: Crude Odds Ratio & Woolf 95% CI (odds_ratio_with_ci)",
        book_section="Section 20.3.6 (Epidemiological Risk & Odds Ratios)",
        points=10,
        hints={
            1: "Compute `or_val = (a * d) / (b * c)` and Woolf standard error `se = math.sqrt(1/a + 1/b + 1/c + 1/d)`.",
            2: "Compute bounds on the natural log scale (`math.log(or_val) +/- 1.96 * se`), exponentiate with `math.exp(...)`, and round all 3 values to 3 decimals.",
            3: "def odds_ratio_with_ci(a, b, c, d):\n    or_val = (a * d) / (b * c)\n    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)\n    log_or = math.log(or_val)\n    return {\n        'or': round(or_val, 3),\n        'ci_low': round(math.exp(log_or - ___ * se), 3),\n        'ci_high': round(math.exp(log_or + ___ * se), 3),\n    }",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 20.6: benjamini_hochberg_adjust(p_values)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    from scipy import stats

    bad = _check_callable("ch20_todo_06", "To-Do 20.6", fn, "benjamini_hochberg_adjust", 10)
    if bad is not None:
        return bad

    cases = [
        ([0.004, 0.032, 0.048, 0.210, 0.650], "sorted p-values with monotonicity tie"),
        ([0.210, 0.004, 0.650, 0.048, 0.032], "unsorted p-values preserving original order"),
        ([0.001, 0.02, 0.50], "3-hypothesis screen"),
    ]
    errors = []

    for p_raw, desc in cases:
        p_copy = list(p_raw)
        try:
            got = fn(p_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch20_todo_06",
                False,
                "failed",
                "To-Do 20.6 — Runtime Error in `benjamini_hochberg_adjust`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch20_todo_06", "To-Do 20.6", "benjamini_hochberg_adjust(p_values)", 10)

        if p_copy != p_raw:
            errors.append(f"Mutated input list on {desc}! Never sort the caller's list in place.")

        exp = [round(float(q), 4) for q in stats.false_discovery_control(p_raw, method="bh")]

        if not isinstance(got, list):
            errors.append(
                f"[{desc}] Expected a Python list of floats, got {type(got).__name__} ({got!r}). "
                "Convert NumPy arrays using `[round(float(q), 4) for q in q_vals]`."
            )
            continue

        if len(got) != len(exp):
            errors.append(f"[{desc}] Length mismatch: expected {len(exp)}, got {len(got)}.")
            continue

        if not all(_is_number(v) and math.isclose(float(v), e, abs_tol=1e-4) for v, e in zip(got, exp)):
            errors.append(f"[{desc}] Expected {exp}, got {got!r}")

    if errors:
        return CheckResult(
            "ch20_todo_06",
            False,
            "failed",
            "To-Do 20.6 — Check Benjamini-Hochberg FDR Adjustment & Ordering",
            "Use `scipy.stats.false_discovery_control(p_values, method='bh')` and return a list of floats rounded to 4 decimals in the original input order.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch20_todo_06",
        True,
        "passed",
        "To-Do 20.6 Passed — Benjamini-Hochberg FDR q-Values Verified!",
        "Matched `scipy.stats.false_discovery_control(..., method='bh')` across sorted, unsorted, and monotonicity-constrained p-value lists.",
        tip="Whenever you run 5+ subgroup or biomarker tests, report BH-adjusted q-values so false discoveries stay controlled.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_todo_06",
        chapter="ch20",
        title="To-Do 20.6: Benjamini-Hochberg FDR Adjustment (benjamini_hochberg_adjust)",
        book_section="Section 20.3.7 (Multiple Testing & FDR Control)",
        points=10,
        hints={
            1: "SciPy provides `scipy.stats.false_discovery_control(p_values, method='bh')` which automatically handles sorting, step-up monotonicity, and restoring original order.",
            2: "Convert the returned NumPy array to a standard Python list with `[round(float(q), 4) for q in q_vals]`.",
            3: "def benjamini_hochberg_adjust(p_values):\n    q_vals = stats.false_discovery_control(p_values, method='___')\n    return [round(float(q), 4) for q in q_vals]",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 20.1 (30 pts): epidemiological_2x2_report(a, b, c, d)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    from scipy import stats

    bad = _check_callable("ch20_exam_01", "Exam Task 20.1", fn, "epidemiological_2x2_report", 30)
    if bad is not None:
        return bad

    cases = [
        ((42, 18, 15, 45), "high-risk cohort (RR=2.800, OR=7.000)"),
        ((133, 223, 46, 98), "MCH rural vs urban anemia"),
        ((10, 90, 25, 75), "protective exposure (RR=0.400, OR=0.333)"),
    ]
    expected_keys = [
        "risk_exposed",
        "risk_unexposed",
        "risk_ratio",
        "odds_ratio",
        "or_ci_low",
        "or_ci_high",
        "chi2_p_value",
    ]
    errors = []

    for (a, b, c, d), desc in cases:
        try:
            got = fn(a, b, c, d)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch20_exam_01",
                False,
                "failed",
                "Exam Task 20.1 — Runtime Error in `epidemiological_2x2_report`",
                f"epidemiological_2x2_report({a}, {b}, {c}, {d}) raised {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap("ch20_exam_01", "Exam Task 20.1", f"epidemiological_2x2_report({a}, {b}, {c}, {d})", 30)

        key_err = _check_dict_keys("ch20_exam_01", "Exam Task 20.1", got, expected_keys, 30)
        if key_err is not None:
            return key_err

        r_exp = a / (a + b)
        r_unexp = c / (c + d)
        rr = r_exp / r_unexp
        or_val = (a * d) / (b * c)
        se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        log_or = math.log(or_val)
        _, p_val, _, _ = stats.chi2_contingency([[a, b], [c, d]], correction=False)
        _, p_yates, _, _ = stats.chi2_contingency([[a, b], [c, d]], correction=True)

        expected = {
            "risk_exposed": (round(r_exp, 4), 1e-4),
            "risk_unexposed": (round(r_unexp, 4), 1e-4),
            "risk_ratio": (round(rr, 3), 1e-3),
            "odds_ratio": (round(or_val, 3), 1e-3),
            "or_ci_low": (round(math.exp(log_or - 1.96 * se), 3), 1e-3),
            "or_ci_high": (round(math.exp(log_or + 1.96 * se), 3), 1e-3),
            "chi2_p_value": (round(float(p_val), 4), 1e-4),
        }

        for key, (exp_val, tol) in expected.items():
            if not _is_number(got[key]) or not math.isclose(float(got[key]), exp_val, abs_tol=tol):
                extra = ""
                if key == "chi2_p_value" and _is_number(got[key]) and math.isclose(float(got[key]), round(float(p_yates), 4), abs_tol=1e-3):
                    extra = " (Pass `correction=False` to `chi2_contingency`!)"
                elif key == "risk_ratio" and _is_number(got[key]) and math.isclose(float(got[key]), round(or_val, 3), abs_tol=1e-3):
                    extra = " (Looks like Odds Ratio instead of Risk Ratio `risk_exposed / risk_unexposed`!)"
                errors.append(f"[{desc}] {key}: expected {exp_val}, got {got[key]!r}{extra}")

    if errors:
        return CheckResult(
            "ch20_exam_01",
            False,
            "failed",
            "Exam Task 20.1 — Check 2x2 Risk, OR, Woolf CI & Chi-Square Metrics",
            "Verify exact keys, rounding (4 decimals for risks/p-value, 3 decimals for RR/OR/CI), and `correction=False`.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch20_exam_01",
        True,
        "passed",
        "Exam Task 20.1 Passed (30/30 pts) — Epidemiological 2x2 Report Verified!",
        "Computed exposed/unexposed risks, RR, OR, Woolf 95% CI, and Pearson chi-square p-value cleanly across 3 cohorts.",
        points_earned=30,
        points_possible=30,
        tip="Notice how RR (2.800) and OR (7.000) diverge sharply when the outcome is common — never interpret an OR as a relative risk!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_exam_01",
        chapter="ch20",
        title="Exam 20.1: Complete Epidemiological 2x2 Report (epidemiological_2x2_report)",
        book_section="Chapter 20 Part IV Capstone Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Compute `risk_exp = a / (a + b)`, `risk_unexp = c / (c + d)`, `rr = risk_exp / risk_unexp`, `or_val = (a * d) / (b * c)`, Woolf 95% CI with `z=1.96`, and Pearson chi-square p-value with `correction=False`.",
            2: "Round `risk_exposed`, `risk_unexposed`, and `chi2_p_value` to 4 decimals; round `risk_ratio`, `odds_ratio`, `or_ci_low`, and `or_ci_high` to 3 decimals.",
            3: "def epidemiological_2x2_report(a, b, c, d):\n    risk_exp = a / (a + b)\n    risk_unexp = c / (c + d)\n    rr = risk_exp / risk_unexp\n    or_val = (a * d) / (b * c)\n    se = math.sqrt(1/a + 1/b + 1/c + 1/d)\n    log_or = math.log(or_val)\n    _, p_val, _, _ = stats.chi2_contingency([[a, b], [c, d]], correction=___)\n    return {...}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 20.2 (35 pts): anova_and_kruskal_by_group(df, value_col, group_col)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from scipy import stats

    from ..datasets import load_data

    bad = _check_callable("ch20_exam_02", "Exam Task 20.2", fn, "anova_and_kruskal_by_group", 35)
    if bad is not None:
        return bad

    df_synth = pd.DataFrame(
        {
            "clinic": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C", "C"],
            "hb": [10.1, 10.4, np.nan, 9.8, 10.5, 11.4, 11.8, 11.2, np.nan, 12.6, 12.9, 12.2, 13.0, 12.5],
        }
    )
    df_null = pd.DataFrame(
        {
            "arm": ["X", "X", "X", "X", "Y", "Y", "Y", "Y", "Z", "Z", "Z", "Z"],
            "score": [10.2, 11.1, 10.8, 10.5, 10.4, 11.0, 10.6, 10.9, 10.3, 11.2, 10.7, 10.4],
        }
    )
    df_mch = load_data("mch_survey_sample", as_pandas=True)

    cases = [
        (df_synth, "hb", "clinic", "3-clinic synthetic DataFrame with NaNs"),
        (df_mch, "maternal_hb_g_dl", "wealth_quintile", "MCH hemoglobin across 5 wealth quintiles"),
        (df_null, "score", "arm", "3-arm non-significant trial"),
    ]
    expected_keys = ["n_groups", "f_stat", "anova_p", "h_stat", "kruskal_p"]
    errors = []

    for df_in, vcol, gcol, desc in cases:
        df_copy = df_in.copy(deep=True)
        try:
            got = fn(df_copy, vcol, gcol)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch20_exam_02",
                False,
                "failed",
                "Exam Task 20.2 — Runtime Error in `anova_and_kruskal_by_group`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap("ch20_exam_02", "Exam Task 20.2", f"anova_and_kruskal_by_group(df, {vcol!r}, {gcol!r})", 35)

        key_err = _check_dict_keys("ch20_exam_02", "Exam Task 20.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not df_copy.equals(df_in):
            errors.append(f"[{desc}] Input DataFrame was mutated! Avoid in-place edits on `df`.")

        groups = [
            grp[vcol].dropna().to_numpy()
            for _, grp in df_in.groupby(gcol)
            if grp[vcol].notna().any()
        ]
        f_stat, anova_p = stats.f_oneway(*groups)
        h_stat, kruskal_p = stats.kruskal(*groups)

        if not isinstance(got["n_groups"], int) or isinstance(got["n_groups"], bool) or got["n_groups"] != len(groups):
            errors.append(f"[{desc}] n_groups: expected int {len(groups)}, got {got['n_groups']!r}")

        for key, exp_raw in [
            ("f_stat", f_stat),
            ("anova_p", anova_p),
            ("h_stat", h_stat),
            ("kruskal_p", kruskal_p),
        ]:
            exp_val = round(float(exp_raw), 4)
            if not _is_number(got[key]) or math.isnan(float(got[key])):
                extra = " (Did you forget `.dropna()` on each group's values before calling SciPy?)" if _is_number(got[key]) and math.isnan(float(got[key])) else ""
                errors.append(f"[{desc}] {key}: expected {exp_val}, got {got[key]!r}{extra}")
            elif not math.isclose(float(got[key]), exp_val, abs_tol=1e-4):
                errors.append(f"[{desc}] {key}: expected {exp_val}, got {got[key]!r}")

    if errors:
        return CheckResult(
            "ch20_exam_02",
            False,
            "failed",
            "Exam Task 20.2 — Check Group Extraction, NaN Dropping & ANOVA/Kruskal Stats",
            "Group `df` by `group_col`, drop NaNs in `value_col`, and unpack `stats.f_oneway(*groups)` and `stats.kruskal(*groups)`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch20_exam_02",
        True,
        "passed",
        "Exam Task 20.2 Passed (35/35 pts) — Multi-Group ANOVA & Kruskal-Wallis Verified!",
        "Passed NaN-bearing synthetic cohorts, 5-quintile MCH hemoglobin gradient (F=9.7085, H=33.1492), and null trial checks.",
        points_earned=35,
        points_possible=35,
        tip="Running parametric ANOVA alongside nonparametric Kruskal-Wallis is a gold-standard sensitivity check for Table 1/2 manuscripts.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_exam_02",
        chapter="ch20",
        title="Exam 20.2: Multi-Group One-Way ANOVA & Kruskal-Wallis (anova_and_kruskal_by_group)",
        book_section="Chapter 20 Part IV Capstone Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Build a list of non-null arrays per group: `groups = [grp[value_col].dropna().to_numpy() for _, grp in df.groupby(group_col) if grp[value_col].notna().any()]`.",
            2: "Unpack `f_stat, anova_p = stats.f_oneway(*groups)` and `h_stat, kruskal_p = stats.kruskal(*groups)`, rounding all 4 floats to 4 decimals.",
            3: "def anova_and_kruskal_by_group(df, value_col, group_col):\n    groups = [\n        grp[value_col].dropna().to_numpy()\n        for _, grp in df.groupby(group_col)\n        if grp[value_col].notna().any()\n    ]\n    f_stat, anova_p = stats.f_oneway(*groups)\n    h_stat, kruskal_p = stats.kruskal(*groups)\n    return {'n_groups': len(groups), 'f_stat': round(float(f_stat), 4), 'anova_p': round(float(anova_p), 4), 'h_stat': round(float(h_stat), 4), 'kruskal_p': round(float(kruskal_p), 4)}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 20.3 (35 pts): fit_adjusted_anemia_logit(mch_df)
# ---------------------------------------------------------------------------
def _ref_anemia_logit(mch_df: Any) -> Any:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    work = mch_df.copy()
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    work["anc_visits"] = work["anc_visits"].replace(999, np.nan)
    work["is_urban"] = (work["residence"] == "Urban").astype(int)
    model_df = work[["maternal_anemia", "anc_visits", "is_urban"]].dropna()
    X = sm.add_constant(model_df[["anc_visits", "is_urban"]])
    y = model_df["maternal_anemia"]
    fit = sm.Logit(y, X).fit(disp=False)
    conf = fit.conf_int()
    return pd.DataFrame(
        {
            "aor": np.exp(fit.params),
            "ci_low": np.exp(conf[0]),
            "ci_high": np.exp(conf[1]),
            "p_value": fit.pvalues,
        },
        index=["const", "anc_visits", "is_urban"],
    ).round(4)


def _verify_exam_03(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable("ch20_exam_03", "Exam Task 20.3", fn, "fit_adjusted_anemia_logit", 35)
    if bad is not None:
        return bad

    df_mch = load_data("mch_survey_sample", as_pandas=True)

    # Secondary synthetic cohort with messy residence casing, sentinel 999, and NaNs
    rng = np.random.default_rng(2026)
    n_syn = 120
    res_raw = rng.choice(["Urban", "urban", " URBAN ", "Rural", "rural", " Rural "], size=n_syn)
    anc_raw = rng.integers(0, 9, size=n_syn).astype(float)
    is_urb_true = np.array([s.strip().title() == "Urban" for s in res_raw], dtype=int)
    logits = 0.6 - 0.35 * anc_raw - 0.5 * is_urb_true
    probs = 1.0 / (1.0 + np.exp(-logits))
    anemia_raw = rng.binomial(1, probs).astype(float)
    anc_raw[[5, 19, 44]] = 999.0
    anc_raw[[12, 77]] = np.nan
    anemia_raw[[30]] = np.nan
    df_syn = pd.DataFrame(
        {
            "residence": res_raw,
            "anc_visits": anc_raw,
            "maternal_anemia": anemia_raw,
        }
    )

    cases = [
        (df_mch, "real mch_survey_sample (500 rows -> 492 complete)"),
        (df_syn, "synthetic messy cohort (casing + 999 + NaN)"),
    ]
    expected_index = ["const", "anc_visits", "is_urban"]
    expected_cols = ["aor", "ci_low", "ci_high", "p_value"]
    errors = []

    for df_in, desc in cases:
        df_copy = df_in.copy(deep=True)
        try:
            got = fn(df_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch20_exam_03",
                False,
                "failed",
                "Exam Task 20.3 — Runtime Error in `fit_adjusted_anemia_logit`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=[
                    "Make sure you replace `anc_visits == 999` with `np.nan` and drop missing rows BEFORE calling `sm.Logit(y, X).fit(disp=False)`."
                ],
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap("ch20_exam_03", "Exam Task 20.3", "fit_adjusted_anemia_logit(mch_df)", 35)

        if not isinstance(got, pd.DataFrame):
            return CheckResult(
                "ch20_exam_03",
                False,
                "failed",
                "Exam Task 20.3 — Return a Pandas DataFrame",
                f"Expected a pd.DataFrame, got {type(got).__name__}.",
                points_earned=0,
                points_possible=35,
            )

        if not df_copy.equals(df_in):
            errors.append(f"[{desc}] Input DataFrame was mutated! Start with `work = mch_df.copy()`.")

        if list(got.index) != expected_index:
            extra = (
                " (Did you forget `sm.add_constant(X)`?)"
                if "const" not in list(got.index)
                else ""
            )
            errors.append(f"[{desc}] Index mismatch: expected {expected_index}, got {list(got.index)}{extra}")
            continue

        if list(got.columns) != expected_cols:
            errors.append(f"[{desc}] Column mismatch: expected {expected_cols}, got {list(got.columns)}")
            continue

        exp_df = _ref_anemia_logit(df_in)
        for row_name in expected_index:
            for col_name in expected_cols:
                g_val = float(got.loc[row_name, col_name])
                e_val = float(exp_df.loc[row_name, col_name])
                if not math.isclose(g_val, e_val, abs_tol=1e-3):
                    extra = ""
                    if col_name == "aor" and g_val < 0:
                        extra = " (Negative aOR detected — did you forget `np.exp(fit.params)` to convert log-odds to Adjusted Odds Ratios?)"
                    elif row_name == "anc_visits" and col_name == "aor":
                        extra = " (Check that `anc_visits == 999` was replaced with `np.nan` and `residence` casing was cleaned before fitting!)"
                    errors.append(
                        f"[{desc}] loc[{row_name!r}, {col_name!r}]: expected {e_val}, got {g_val}{extra}"
                    )

    if errors:
        return CheckResult(
            "ch20_exam_03",
            False,
            "failed",
            "Exam Task 20.3 — Check Data Cleaning, `sm.add_constant`, & `np.exp()` aOR Table",
            "Clean `residence` casing, replace `anc_visits == 999` with `np.nan`, drop missing rows, fit `sm.Logit(y, sm.add_constant(X)).fit(disp=False)`, and exponentiate params & conf_int.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch20_exam_03",
        True,
        "passed",
        "Exam Task 20.3 Passed (35/35 pts) — Adjusted Logistic Regression Pipeline Verified!",
        "Cleaned messy MCH survey covariates, fitted multivariable `sm.Logit`, and returned publication-ready aOR (anc_visits aOR=0.9006, p=0.0378), 95% CIs, and p-values.",
        points_earned=35,
        points_possible=35,
        tip="Congratulations — you just completed the Part IV Capstone! Every Table 3 multivariable regression in epidemiology follows this exact workflow.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch20_exam_03",
        chapter="ch20",
        title="Exam 20.3: Adjusted Multivariable Logistic Regression (fit_adjusted_anemia_logit)",
        book_section="Chapter 20 Part IV Capstone Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Work on `mch_df.copy()`. Standardize `residence` via `.astype(str).str.strip().str.title()`, replace `anc_visits == 999` with `np.nan`, build `is_urban = (work['residence'] == 'Urban').astype(int)`, and `.dropna()` on the 3 model columns.",
            2: "Fit `fit = sm.Logit(y, sm.add_constant(X)).fit(disp=False)`. Compute `aor = np.exp(fit.params)`, `conf = fit.conf_int()`, `ci_low = np.exp(conf[0])`, `ci_high = np.exp(conf[1])`, and `p_value = fit.pvalues`.",
            3: "def fit_adjusted_anemia_logit(mch_df):\n    work = mch_df.copy()\n    work['residence'] = work['residence'].astype(str).str.strip().str.title()\n    work['anc_visits'] = work['anc_visits'].replace(999, np.nan)\n    work['is_urban'] = (work['residence'] == 'Urban').astype(int)\n    model_df = work[['maternal_anemia', 'anc_visits', 'is_urban']].dropna()\n    X = sm.add_constant(model_df[['anc_visits', 'is_urban']])\n    fit = sm.Logit(model_df['maternal_anemia'], X).fit(disp=False)\n    conf = fit.conf_int()\n    return pd.DataFrame({'aor': np.exp(fit.params), 'ci_low': np.exp(conf[0]), 'ci_high': np.exp(conf[1]), 'p_value': fit.pvalues}, index=['const', 'anc_visits', 'is_urban']).round(4)",
        },
        verifier=_verify_exam_03,
    )
)
