"""Chapter 34 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch34_todo_01..06) and the Chapter 34 Coding Exam
(ch34_exam_01..03) for Error Analysis, Subgroup Fairness Audits,
Calibration Curves, and Responsible Clinical AI.
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn inside the check
function, validates return structures, rounding, mathematical formulas, and
non-mutation, while letting NotImplementedError propagate cleanly for starter
stubs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_np_or_py_number(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
        val, (bool, np.bool_)
    )


def _is_bool(val: Any) -> bool:
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
        f"Calling {call_str} returned None instead of the computed audit result.",
        details=[
            "Remember: print() only displays text on screen; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Responsible AI audit helpers always return structured dicts or DataFrames for downstream reporting.",
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


def _py_scalar(val: Any) -> Any:
    return val.item() if hasattr(val, "item") else val


# ---------------------------------------------------------------------------
# Reference implementations (lazy-importing inside each helper)
# ---------------------------------------------------------------------------
def _ref_subgroup_performance(
    y_true: Any, y_prob: Any, groups: Any, threshold: float = 0.5
) -> Any:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import brier_score_loss, roc_auc_score

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    gr = np.asarray(groups)
    rows = []
    for g in sorted(pd.unique(gr)):
        mask = gr == g
        yt_g, yp_g = yt[mask], yp[mask]
        pred_g = (yp_g >= threshold).astype(int)
        n_g = int(np.sum(mask))
        prev = float(np.mean(yt_g == 1))
        auc = (
            round(float(roc_auc_score(yt_g, yp_g)), 4)
            if len(np.unique(yt_g)) == 2
            else np.nan
        )
        tp = int(np.sum((yt_g == 1) & (pred_g == 1)))
        fn = int(np.sum((yt_g == 1) & (pred_g == 0)))
        fp = int(np.sum((yt_g == 0) & (pred_g == 1)))
        tn = int(np.sum((yt_g == 0) & (pred_g == 0)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        brier = float(brier_score_loss(yt_g, yp_g))
        rows.append(
            {
                "group": _py_scalar(g),
                "n": n_g,
                "prevalence": round(prev, 4),
                "roc_auc": auc,
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
                "brier": round(brier, 4),
            }
        )
    cols = ["group", "n", "prevalence", "roc_auc", "tpr", "fpr", "brier"]
    return pd.DataFrame(rows, columns=cols)


def _ref_fairness_gaps(y_true: Any, y_pred: Any, sensitive_attr: Any) -> dict[str, float]:
    import numpy as np
    import pandas as pd

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    gr = np.asarray(sensitive_attr)
    sel_rates, tprs, fprs = [], [], []
    for g in sorted(pd.unique(gr)):
        mask = gr == g
        yt_g, yp_g = yt[mask], yp[mask]
        sel_rates.append(float(np.mean(yp_g == 1)))
        tp = int(np.sum((yt_g == 1) & (yp_g == 1)))
        fn = int(np.sum((yt_g == 1) & (yp_g == 0)))
        fp = int(np.sum((yt_g == 0) & (yp_g == 1)))
        tn = int(np.sum((yt_g == 0) & (yp_g == 0)))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    dp_diff = max(sel_rates) - min(sel_rates)
    eo_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)
    eq_odds = max(eo_diff, fpr_diff)
    return {
        "demographic_parity_diff": round(dp_diff, 4),
        "equal_opportunity_diff": round(eo_diff, 4),
        "max_fpr_diff": round(fpr_diff, 4),
        "equalized_odds_diff": round(eq_odds, 4),
    }


def _ref_subgroup_calibration(
    y_true: Any, y_prob: Any, groups: Any, n_bins: int = 5
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import brier_score_loss

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    gr = np.asarray(groups)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    eces: dict[str, float] = {}
    for g in sorted(pd.unique(gr)):
        mask = gr == g
        yt_g, yp_g = yt[mask], yp[mask]
        n_g = int(np.sum(mask))
        ece = 0.0
        for idx in range(n_bins):
            lo, hi = edges[idx], edges[idx + 1]
            if idx < n_bins - 1:
                bm = (yp_g >= lo) & (yp_g < hi)
            else:
                bm = (yp_g >= lo) & (yp_g <= hi)
            cnt = int(np.sum(bm))
            if cnt > 0:
                ece += (cnt / n_g) * abs(
                    float(np.mean(yt_g[bm])) - float(np.mean(yp_g[bm]))
                )
        g_label = _py_scalar(g)
        eces[str(g_label)] = ece
        rows.append(
            {
                "group": g_label,
                "n": n_g,
                "ece": round(ece, 4),
                "brier": round(float(brier_score_loss(yt_g, yp_g)), 4),
            }
        )
    per_group = pd.DataFrame(rows, columns=["group", "n", "ece", "brier"])
    worst_group = max(eces, key=lambda k: eces[k])
    max_gap = round(max(eces.values()) - min(eces.values()), 4)
    return {
        "per_group": per_group,
        "worst_ece_group": worst_group,
        "max_ece_gap": max_gap,
    }


def _ref_psi(
    expected: Any, actual: Any, n_bins: int = 10, eps: float = 1e-6
) -> dict[str, Any]:
    import numpy as np

    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
    exp_cnt, _ = np.histogram(exp, bins=edges)
    act_cnt, _ = np.histogram(act, bins=edges)
    p_exp = np.clip(exp_cnt / exp_cnt.sum(), eps, None)
    p_exp = p_exp / p_exp.sum()
    p_act = np.clip(act_cnt / act_cnt.sum(), eps, None)
    p_act = p_act / p_act.sum()
    psi = round(float(np.sum((p_act - p_exp) * np.log(p_act / p_exp))), 4)
    if psi < 0.10:
        status = "Stable"
    elif psi < 0.25:
        status = "Moderate Drift"
    else:
        status = "Severe Drift"
    return {"psi": psi, "drift_status": status}


def _ref_calibrate_group_thresholds(
    y_true: Any, y_prob: Any, groups: Any, target_tpr: float = 0.75
) -> dict[Any, float]:
    import numpy as np
    import pandas as pd

    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)
    gr = np.asarray(groups)
    out: dict[Any, float] = {}
    for g in sorted(pd.unique(gr)):
        mask = gr == g
        yt_g, yp_g = yt[mask], yp[mask]
        n_pos = int(np.sum(yt_g == 1))
        g_key = _py_scalar(g)
        if n_pos == 0:
            out[g_key] = 0.0
            continue
        best_t = 0.0
        for t in np.sort(np.unique(yp_g))[::-1]:
            tpr = float(np.sum((yt_g == 1) & (yp_g >= t))) / n_pos
            if tpr >= target_tpr:
                best_t = float(t)
                break
        out[g_key] = round(best_t, 4)
    return out


# ---------------------------------------------------------------------------
# GUIDED TO-DO 34.1: subgroup_performance_table(y_true, y_prob, groups, threshold=0.5)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    bad = _check_callable(
        "ch34_todo_01", "To-Do 34.1", fn, "subgroup_performance_table", 10
    )
    if bad is not None:
        return bad

    cases = [
        (
            [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [0.82, 0.41, 0.22, 0.65, 0.74, 0.15, 0.35, 0.28, 0.10, 0.55],
            ["Urban", "Rural", "Rural", "Rural", "Rural", "Urban", "Urban", "Urban", "Rural", "Urban"],
            {},
            "two-group Rural/Urban cohort (default threshold=0.5)",
        ),
        (
            [1, 0, 1, 0, 0, 0, 0],
            [0.85, 0.20, 0.60, 0.45, 0.10, 0.30, 0.55],
            ["Rural", "Rural", "Rural", "Rural", "Urban", "Urban", "Urban"],
            {"threshold": 0.4},
            "single-class Urban subgroup (roc_auc must be NaN, custom threshold=0.4)",
        ),
    ]
    expected_cols = ["group", "n", "prevalence", "roc_auc", "tpr", "fpr", "brier"]
    errors = []

    for yt_raw, yp_raw, gr_raw, kwargs, desc in cases:
        yt_c, yp_c, gr_c = list(yt_raw), list(yp_raw), list(gr_raw)
        try:
            got = fn(yt_c, yp_c, gr_c, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch34_todo_01",
                False,
                "failed",
                "To-Do 34.1 — Check Function Signature `subgroup_performance_table(y_true, y_prob, groups, threshold=0.5)`",
                f"Failed on {desc} with TypeError: {exc}",
                details=["Define `def subgroup_performance_table(y_true, y_prob, groups, threshold=0.5):`."],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch34_todo_01",
                False,
                "failed",
                "To-Do 34.1 — Runtime Error in `subgroup_performance_table`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=[
                    "If a subgroup has only one class in `y_true`, set `roc_auc = np.nan` instead of calling `roc_auc_score`."
                ],
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch34_todo_01",
                "To-Do 34.1",
                "subgroup_performance_table(y_true, y_prob, groups)",
                10,
            )

        if not isinstance(got, pd.DataFrame):
            return CheckResult(
                "ch34_todo_01",
                False,
                "failed",
                "To-Do 34.1 — Return a Pandas DataFrame",
                f"[{desc}] Expected pd.DataFrame, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if list(got.columns) != expected_cols:
            errors.append(
                f"[{desc}] Column mismatch: expected {expected_cols}, got {list(got.columns)}."
            )
            continue

        exp = _ref_subgroup_performance(
            yt_raw, yp_raw, gr_raw, threshold=kwargs.get("threshold", 0.5)
        )
        got_r = got.reset_index(drop=True)
        if len(got_r) != len(exp):
            errors.append(
                f"[{desc}] Expected {len(exp)} group rows, got {len(got_r)}."
            )
            continue

        if list(got_r["group"].astype(str)) != list(exp["group"].astype(str)):
            errors.append(
                f"[{desc}] Groups must be sorted: expected {list(exp['group'])}, got {list(got_r['group'])}."
            )
            continue

        for i in range(len(exp)):
            grp_lbl = exp.loc[i, "group"]
            if not _is_np_or_py_number(got_r.loc[i, "n"]) or int(got_r.loc[i, "n"]) != int(exp.loc[i, "n"]):
                errors.append(
                    f"[{desc}] group={grp_lbl} `n`: expected {int(exp.loc[i, 'n'])}, got {got_r.loc[i, 'n']!r}"
                )
            for col in ("prevalence", "roc_auc", "tpr", "fpr", "brier"):
                ev = exp.loc[i, col]
                gv = got_r.loc[i, col]
                if pd.isna(ev):
                    if not pd.isna(gv):
                        errors.append(
                            f"[{desc}] group={grp_lbl} `{col}`: expected NaN for single-class subgroup, got {gv!r}"
                        )
                else:
                    if not _is_np_or_py_number(gv) or not math.isclose(float(gv), float(ev), abs_tol=1e-4):
                        errors.append(
                            f"[{desc}] group={grp_lbl} `{col}`: expected {ev}, got {gv!r}"
                        )

    if errors:
        return CheckResult(
            "ch34_todo_01",
            False,
            "failed",
            "To-Do 34.1 — Check Subgroup Disaggregation Table & Single-Class AUC Guard",
            "Sort unique groups, compute `n`, `prevalence`, `roc_auc` (`np.nan` if single-class), `tpr`, `fpr`, and `brier`, rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch34_todo_01",
        True,
        "passed",
        "To-Do 34.1 Passed — Subgroup Performance Table Verified!",
        "Accurately disaggregated prevalence, ROC-AUC (with single-class NaN guard), TPR, FPR, and Brier score across sorted subgroups.",
        tip="Aggregate AUC can hide total failure in a minority subgroup — always disaggregate by geography, wealth, sex, or age before clinical deployment.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_todo_01",
        chapter="ch34",
        title="To-Do 34.1: Subgroup Disaggregation Table (subgroup_performance_table)",
        book_section="Section 34.3.1 (Aggregate AUC vs. Subgroup Disaggregation)",
        points=10,
        hints={
            1: "Loop over `sorted(pd.unique(gr))` and slice `yt_g = yt[mask]`, `yp_g = yp[mask]`, and `pred_g = (yp_g >= threshold).astype(int)`.",
            2: "Set `auc = round(float(roc_auc_score(yt_g, yp_g)), 4) if len(np.unique(yt_g)) == 2 else np.nan`, and guard `tpr` and `fpr` denominators against zero.",
            3: "def subgroup_performance_table(y_true, y_prob, groups, threshold=0.5):\n    yt, yp, gr = np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float), np.asarray(groups)\n    rows = []\n    for g in sorted(pd.unique(gr)):\n        m = gr == g\n        yt_g, yp_g = yt[m], yp[m]\n        pred_g = (yp_g >= threshold).astype(int)\n        auc = round(float(roc_auc_score(yt_g, yp_g)), 4) if len(np.unique(yt_g)) == 2 else np.nan\n        # compute tp, fn, fp, tn -> tpr, fpr, brier and append dict\n    return pd.DataFrame(rows, columns=['group', 'n', 'prevalence', 'roc_auc', 'tpr', 'fpr', 'brier'])",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 34.2: compute_fairness_gaps(y_true, y_pred, sensitive_attr)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch34_todo_02", "To-Do 34.2", fn, "compute_fairness_gaps", 10
    )
    if bad is not None:
        return bad

    cases = [
        (
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            ["Rural"] * 8 + ["Urban"] * 8,
            "Rural vs. Urban disparate TPR and FPR",
        ),
        (
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
            ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "3-group fairness disparity (max - min across A, B, C)",
        ),
    ]
    expected_keys = [
        "demographic_parity_diff",
        "equal_opportunity_diff",
        "max_fpr_diff",
        "equalized_odds_diff",
    ]
    errors = []

    for yt_raw, yp_raw, gr_raw, desc in cases:
        try:
            got = fn(list(yt_raw), list(yp_raw), list(gr_raw))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch34_todo_02",
                False,
                "failed",
                "To-Do 34.2 — Runtime Error in `compute_fairness_gaps`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch34_todo_02",
                "To-Do 34.2",
                "compute_fairness_gaps(y_true, y_pred, sensitive_attr)",
                10,
            )

        key_err = _check_dict_keys("ch34_todo_02", "To-Do 34.2", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        exp = _ref_fairness_gaps(yt_raw, yp_raw, gr_raw)
        for k, ev in exp.items():
            gv = got[k]
            if not _is_np_or_py_number(gv) or not math.isclose(float(gv), ev, abs_tol=1e-4):
                errors.append(f"[{desc}] `{k}`: expected {ev}, got {gv!r}")

    if errors:
        return CheckResult(
            "ch34_todo_02",
            False,
            "failed",
            "To-Do 34.2 — Check Fairness Gap Definitions & Equalized Odds",
            "Compute per-group selection rate, TPR, and FPR; take `max - min` across groups for each gap, and set `equalized_odds_diff = max(equal_opportunity_diff, max_fpr_diff)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch34_todo_02",
        True,
        "passed",
        "To-Do 34.2 Passed — Demographic Parity, Equal Opportunity & Equalized Odds Verified!",
        "Computed exact max-min disparity gaps across both binary and multi-group sensitive attributes.",
        tip="In clinical screening, Equal Opportunity (TPR parity) asks whether sick patients in every group have an equal chance of being caught!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_todo_02",
        chapter="ch34",
        title="To-Do 34.2: Algorithmic Fairness Gap Audit (compute_fairness_gaps)",
        book_section="Section 34.3.2 (Fairness Metrics: DP, EO, Equalized Odds)",
        points=10,
        hints={
            1: "For each unique group in `sensitive_attr`, compute `sel_rate = mean(yp_g == 1)`, `tpr = tp / (tp + fn)` (guarding 0), and `fpr = fp / (fp + tn)` (guarding 0).",
            2: "Compute `dp_diff = max(sel_rates) - min(sel_rates)`, `eo_diff = max(tprs) - min(tprs)`, `fpr_diff = max(fprs) - min(fprs)`, and `eq_odds = max(eo_diff, fpr_diff)`.",
            3: "def compute_fairness_gaps(y_true, y_pred, sensitive_attr):\n    yt, yp, gr = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int), np.asarray(sensitive_attr)\n    sel, tprs, fprs = [], [], []\n    for g in sorted(pd.unique(gr)):\n        m = gr == g\n        sel.append(float(np.mean(yp[m] == 1)))\n        # compute tp, fn, fp, tn -> append tpr and fpr\n    eo, fp_d = max(tprs) - min(tprs), max(fprs) - min(fprs)\n    return {'demographic_parity_diff': round(max(sel) - min(sel), 4), 'equal_opportunity_diff': round(eo, 4), 'max_fpr_diff': round(fp_d, 4), 'equalized_odds_diff': round(max(eo, fp_d), 4)}",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 34.3: subgroup_calibration_audit(y_true, y_prob, groups, n_bins=5)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable(
        "ch34_todo_03", "To-Do 34.3", fn, "subgroup_calibration_audit", 10
    )
    if bad is not None:
        return bad

    cases = [
        (
            [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
            [0.05, 0.15, 0.85, 0.95, 0.25, 0.75, 0.70, 0.80, 0.90, 0.85, 0.95, 1.0],
            ["Urban"] * 6 + ["Rural"] * 6,
            {},
            "calibrated Urban vs. overconfident Rural (includes p==1.0)",
        ),
        (
            [0, 1, 0, 1, 0, 0, 1, 1],
            [0.10, 0.90, 0.30, 0.70, 0.45, 0.55, 0.60, 0.80],
            ["A", "A", "A", "A", "B", "B", "B", "B"],
            {"n_bins": 4},
            "custom n_bins=4 calibration audit",
        ),
    ]
    expected_keys = ["per_group", "worst_ece_group", "max_ece_gap"]
    expected_cols = ["group", "n", "ece", "brier"]
    errors = []

    for yt_raw, yp_raw, gr_raw, kwargs, desc in cases:
        try:
            got = fn(list(yt_raw), list(yp_raw), list(gr_raw), **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch34_todo_03",
                False,
                "failed",
                "To-Do 34.3 — Runtime Error in `subgroup_calibration_audit`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch34_todo_03",
                "To-Do 34.3",
                "subgroup_calibration_audit(y_true, y_prob, groups)",
                10,
            )

        key_err = _check_dict_keys("ch34_todo_03", "To-Do 34.3", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        exp = _ref_subgroup_calibration(
            yt_raw, yp_raw, gr_raw, n_bins=kwargs.get("n_bins", 5)
        )
        pg = got["per_group"]
        if not isinstance(pg, pd.DataFrame):
            errors.append(f"[{desc}] `per_group` must be a pd.DataFrame, got {type(pg).__name__}.")
            continue
        if list(pg.columns) != expected_cols:
            errors.append(f"[{desc}] `per_group` columns mismatch: expected {expected_cols}, got {list(pg.columns)}.")
            continue

        pg_r = pg.reset_index(drop=True)
        exp_pg = exp["per_group"]
        if len(pg_r) != len(exp_pg) or list(pg_r["group"].astype(str)) != list(exp_pg["group"].astype(str)):
            errors.append(
                f"[{desc}] `per_group` rows must match sorted groups {list(exp_pg['group'])}, got {list(pg_r['group'])}."
            )
            continue

        for i in range(len(exp_pg)):
            g_lbl = exp_pg.loc[i, "group"]
            if int(pg_r.loc[i, "n"]) != int(exp_pg.loc[i, "n"]):
                errors.append(f"[{desc}] group={g_lbl} `n`: expected {exp_pg.loc[i, 'n']}, got {pg_r.loc[i, 'n']!r}")
            for col in ("ece", "brier"):
                gv = pg_r.loc[i, col]
                ev = exp_pg.loc[i, col]
                if not _is_np_or_py_number(gv) or not math.isclose(float(gv), float(ev), abs_tol=1e-4):
                    errors.append(f"[{desc}] group={g_lbl} `{col}`: expected {ev}, got {gv!r}")

        if str(got["worst_ece_group"]) != exp["worst_ece_group"]:
            errors.append(
                f"[{desc}] `worst_ece_group`: expected {exp['worst_ece_group']!r}, got {got['worst_ece_group']!r}"
            )
        if not _is_np_or_py_number(got["max_ece_gap"]) or not math.isclose(
            float(got["max_ece_gap"]), exp["max_ece_gap"], abs_tol=1e-4
        ):
            errors.append(
                f"[{desc}] `max_ece_gap`: expected {exp['max_ece_gap']}, got {got['max_ece_gap']!r}"
            )

    if errors:
        return CheckResult(
            "ch34_todo_03",
            False,
            "failed",
            "To-Do 34.3 — Check Subgroup Equal-Width ECE & Brier Audit",
            "Compute equal-width ECE on `[0, 1]` (final bin right-closed `<= 1.0`) and Brier score for each sorted group, plus `worst_ece_group` (str) and `max_ece_gap` (round4).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch34_todo_03",
        True,
        "passed",
        "To-Do 34.3 Passed — Subgroup Calibration Audit Verified!",
        "Measured per-group ECE and Brier scores cleanly, including exact `p == 1.0` boundary handling and worst-group gap detection.",
        tip="A model can look well-calibrated overall while severely under- or over-predicting risk in the poorest quintile!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_todo_03",
        chapter="ch34",
        title="To-Do 34.3: Subgroup Calibration Audit (subgroup_calibration_audit)",
        book_section="Section 34.3.3 (Subgroup Calibration & ECE Disparity)",
        points=10,
        hints={
            1: "Create `edges = np.linspace(0.0, 1.0, n_bins + 1)`. For each sorted group `g`, compute equal-width ECE with `[lo, hi)` for bins `0..n_bins-2` and `[lo, hi]` for the last bin.",
            2: "Compute `brier = brier_score_loss(yt_g, yp_g)`, collect rows into `per_group` DataFrame `['group', 'n', 'ece', 'brier']`, find `worst_ece_group` (as `str`), and `max_ece_gap = round(max_ece - min_ece, 4)`.",
            3: "def subgroup_calibration_audit(y_true, y_prob, groups, n_bins=5):\n    yt, yp, gr = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float), np.asarray(groups)\n    edges = np.linspace(0.0, 1.0, n_bins + 1)\n    rows, raw_eces = [], {}\n    for g in sorted(pd.unique(gr)):\n        m = gr == g\n        yt_g, yp_g, n_g = yt[m], yp[m], int(np.sum(m))\n        ece = 0.0\n        for i in range(n_bins):\n            lo, hi = edges[i], edges[i + 1]\n            bm = (yp_g >= lo) & (yp_g < hi) if i < n_bins - 1 else (yp_g >= lo) & (yp_g <= hi)\n            if np.any(bm):\n                ece += (np.sum(bm) / n_g) * abs(float(np.mean(yt_g[bm])) - float(np.mean(yp_g[bm])))\n        raw_eces[str(g)] = ece\n        rows.append({'group': g, 'n': n_g, 'ece': round(ece, 4), 'brier': round(float(brier_score_loss(yt_g, yp_g)), 4)})\n    return {'per_group': pd.DataFrame(rows, columns=['group', 'n', 'ece', 'brier']), 'worst_ece_group': max(raw_eces, key=raw_eces.get), 'max_ece_gap': round(max(raw_eces.values()) - min(raw_eces.values()), 4)}",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 34.4: population_stability_index(expected, actual, n_bins=10, eps=1e-6)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable(
        "ch34_todo_04", "To-Do 34.4", fn, "population_stability_index", 10
    )
    if bad is not None:
        return bad

    rng = np.random.default_rng(123)
    base = rng.normal(11.5, 1.2, size=400)
    stable = rng.normal(11.55, 1.2, size=400)
    moderate = rng.normal(11.0, 1.2, size=400)
    severe = rng.normal(9.5, 1.4, size=400)

    cases = [
        (base, stable, {}, "stable distribution (PSI < 0.10)"),
        (base, moderate, {}, "moderate shift (0.10 <= PSI < 0.25)"),
        (base, severe, {}, "severe shift with out-of-range values (PSI >= 0.25)"),
        (base, severe, {"n_bins": 5, "eps": 1e-4}, "custom n_bins=5 and eps=1e-4"),
    ]
    expected_keys = ["psi", "drift_status"]
    errors = []

    for exp_arr, act_arr, kwargs, desc in cases:
        try:
            got = fn(exp_arr.copy(), act_arr.copy(), **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch34_todo_04",
                False,
                "failed",
                "To-Do 34.4 — Runtime Error in `population_stability_index`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch34_todo_04",
                "To-Do 34.4",
                "population_stability_index(expected, actual)",
                10,
            )

        key_err = _check_dict_keys("ch34_todo_04", "To-Do 34.4", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        ref = _ref_psi(
            exp_arr,
            act_arr,
            n_bins=kwargs.get("n_bins", 10),
            eps=kwargs.get("eps", 1e-6),
        )
        if not _is_np_or_py_number(got["psi"]) or not math.isclose(
            float(got["psi"]), ref["psi"], abs_tol=1e-3
        ):
            errors.append(
                f"[{desc}] `psi`: expected {ref['psi']}, got {got['psi']!r} "
                "(Check `-inf`/`+inf` outer edges, clipping to `>= eps`, and re-normalizing shares!)."
            )
        if got["drift_status"] != ref["drift_status"]:
            errors.append(
                f"[{desc}] `drift_status`: expected {ref['drift_status']!r}, got {got['drift_status']!r}"
            )

    if errors:
        return CheckResult(
            "ch34_todo_04",
            False,
            "failed",
            "To-Do 34.4 — Check Quantile Binning, Outer +/-inf Edges & PSI Formula",
            "Build quantile edges from `expected`, set outer edges to `-np.inf`/`np.inf`, histogram both arrays, clip shares to `>= eps` and re-normalize, sum `(p_act - p_exp) * ln(p_act / p_exp)`, and classify Stable / Moderate Drift / Severe Drift.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch34_todo_04",
        True,
        "passed",
        "To-Do 34.4 Passed — Population Stability Index (PSI) Verified!",
        "Detected Stable, Moderate Drift, and Severe Drift accurately, including out-of-range test values and custom `n_bins`/`eps`.",
        tip="Setting outer edges to `-np.inf` and `+np.inf` ensures newly observed extreme clinical values (e.g., famine-era severe anemia) are captured in the tail bins instead of silently dropped!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_todo_04",
        chapter="ch34",
        title="To-Do 34.4: Covariate Shift Monitoring (population_stability_index)",
        book_section="Section 34.3.6 (Covariate Shift & Population Stability Index)",
        points=10,
        hints={
            1: "Compute `edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, n_bins + 1)))` and replace `edges[0] = -np.inf`, `edges[-1] = np.inf`.",
            2: "Count bins via `np.histogram(..., bins=edges)[0]`, divide by totals, clip to `>= eps`, re-normalize (`p /= p.sum()`), and compute `sum((p_act - p_exp) * np.log(p_act / p_exp))`.",
            3: "def population_stability_index(expected, actual, n_bins=10, eps=1e-6):\n    exp, act = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)\n    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, n_bins + 1)))\n    edges[0], edges[-1] = -np.inf, np.inf\n    p_exp = np.clip(np.histogram(exp, bins=edges)[0] / len(exp), eps, None); p_exp /= p_exp.sum()\n    p_act = np.clip(np.histogram(act, bins=edges)[0] / len(act), eps, None); p_act /= p_act.sum()\n    psi = round(float(np.sum((p_act - p_exp) * np.log(p_act / p_exp))), 4)\n    status = 'Stable' if psi < 0.10 else ('Moderate Drift' if psi < 0.25 else 'Severe Drift')\n    return {'psi': psi, 'drift_status': status}",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 34.5: slice_worst_errors(df, y_true, y_prob, top_k=5)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import pandas as pd

    bad = _check_callable(
        "ch34_todo_05", "To-Do 34.5", fn, "slice_worst_errors", 10
    )
    if bad is not None:
        return bad

    df_orig = pd.DataFrame(
        {
            "patient_id": [f"P{i:02d}" for i in range(1, 11)],
            "residence": ["Urban", "Rural", "Urban", "Rural", "Urban", "Rural", "Rural", "Urban", "Rural", "Urban"],
            "anc_visits": [6, 1, 5, 0, 7, 2, 1, 4, 0, 6],
        },
        index=list(range(100, 110)),
    )
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
    y_prob = [0.08, 0.91, 0.22, 0.85, 0.03, 0.40, 0.78, 0.12, 0.67, 0.51]

    expected_keys = ["worst_false_negatives", "worst_false_positives"]
    errors = []

    for top_k in (3, 5):
        df_copy = df_orig.copy(deep=True)
        desc = f"top_k={top_k}"
        try:
            got = fn(df_copy, list(y_true), list(y_prob), top_k=top_k)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch34_todo_05",
                False,
                "failed",
                "To-Do 34.5 — Runtime Error in `slice_worst_errors`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch34_todo_05", "To-Do 34.5", "slice_worst_errors(df, y_true, y_prob)", 10
            )

        if not df_copy.equals(df_orig):
            errors.append(
                f"[{desc}] Input DataFrame `df` was mutated in place! Always start with `work = df.copy()`."
            )

        key_err = _check_dict_keys("ch34_todo_05", "To-Do 34.5", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        fn_df = got["worst_false_negatives"]
        fp_df = got["worst_false_positives"]
        if not isinstance(fn_df, pd.DataFrame) or not isinstance(fp_df, pd.DataFrame):
            errors.append(f"[{desc}] Both dictionary values must be pd.DataFrame objects.")
            continue

        for name, sub in [("worst_false_negatives", fn_df), ("worst_false_positives", fp_df)]:
            if "y_true" not in sub.columns or "y_prob" not in sub.columns:
                errors.append(f"[{desc}] `{name}` must include attached `y_true` and `y_prob` columns.")
            if list(sub.index) != list(range(len(sub))):
                errors.append(f"[{desc}] `{name}` index must be reset (`0..k-1`), got {list(sub.index)}.")

        exp_fn_ids = ["P05", "P01", "P03", "P10", "P07"][:top_k]
        exp_fp_ids = ["P02", "P04", "P09", "P06", "P08"][:top_k]
        if list(fn_df["patient_id"]) != exp_fn_ids:
            errors.append(
                f"[{desc}] `worst_false_negatives` order mismatch: expected {exp_fn_ids} (y_true==1 sorted by y_prob ascending), got {list(fn_df['patient_id'])}."
            )
        if list(fp_df["patient_id"]) != exp_fp_ids:
            errors.append(
                f"[{desc}] `worst_false_positives` order mismatch: expected {exp_fp_ids} (y_true==0 sorted by y_prob descending), got {list(fp_df['patient_id'])}."
            )

    if errors:
        return CheckResult(
            "ch34_todo_05",
            False,
            "failed",
            "To-Do 34.5 — Check Non-Mutating Error Slicing, Sorting & Index Reset",
            "Work on `work = df.copy()`, attach `y_true` and `y_prob`, slice `y_true == 1` sorted ascending by `y_prob` (FN) and `y_true == 0` sorted descending by `y_prob` (FP), take `.head(top_k).reset_index(drop=True)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch34_todo_05",
        True,
        "passed",
        "To-Do 34.5 Passed — Qualitative Worst-Error Slicer Verified!",
        "Preserved the caller's DataFrame untouched and surfaced the most confident false negatives and false positives with clean reset indices.",
        tip="Inspecting the 5 worst false negatives and false positives reveals missing features, label noise, and proxy bias faster than any scalar metric!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_todo_05",
        chapter="ch34",
        title="To-Do 34.5: Worst FN/FP Error Slicing (slice_worst_errors)",
        book_section="Section 34.3.4 (Qualitative Error Analysis & Failure Slicing)",
        points=10,
        hints={
            1: "Start with `work = df.copy()` so you never mutate the caller's DataFrame, then assign `work['y_true'] = np.asarray(y_true, dtype=int)` and `work['y_prob'] = np.asarray(y_prob, dtype=float)`.",
            2: "Filter `work[work['y_true'] == 1].sort_values('y_prob', ascending=True).head(top_k).reset_index(drop=True)` for `worst_false_negatives` and `work[work['y_true'] == 0].sort_values('y_prob', ascending=False).head(top_k).reset_index(drop=True)` for `worst_false_positives`.",
            3: "def slice_worst_errors(df, y_true, y_prob, top_k=5):\n    work = df.copy()\n    work['y_true'] = np.asarray(y_true, dtype=int)\n    work['y_prob'] = np.asarray(y_prob, dtype=float)\n    fn_df = work[work['y_true'] == 1].sort_values('y_prob', ascending=True).head(top_k).reset_index(drop=True)\n    fp_df = work[work['y_true'] == 0].sort_values('y_prob', ascending=False).head(top_k).reset_index(drop=True)\n    return {'worst_false_negatives': fn_df, 'worst_false_positives': fp_df}",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 34.6: calibrate_group_thresholds_for_tpr(y_true, y_prob, groups, target_tpr=0.75)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    bad = _check_callable(
        "ch34_todo_06", "To-Do 34.6", fn, "calibrate_group_thresholds_for_tpr", 10
    )
    if bad is not None:
        return bad

    cases = [
        (
            [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
            [0.80, 0.65, 0.50, 0.30, 0.25, 0.10, 0.60, 0.42, 0.31, 0.15, 0.20, 0.05],
            ["Urban"] * 6 + ["Rural"] * 6,
            {"target_tpr": 0.75},
            "two-group cohort targeting TPR >= 0.75",
        ),
        (
            [1, 1, 0, 0, 0, 0, 0],
            [0.72, 0.44, 0.30, 0.10, 0.60, 0.35, 0.12],
            ["A", "A", "A", "A", "B", "B", "B"],
            {"target_tpr": 0.80},
            "group B has zero positives -> fallback 0.0",
        ),
    ]
    errors = []

    for yt_raw, yp_raw, gr_raw, kwargs, desc in cases:
        try:
            got = fn(list(yt_raw), list(yp_raw), list(gr_raw), **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch34_todo_06",
                False,
                "failed",
                "To-Do 34.6 — Runtime Error in `calibrate_group_thresholds_for_tpr`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch34_todo_06",
                "To-Do 34.6",
                "calibrate_group_thresholds_for_tpr(y_true, y_prob, groups)",
                10,
            )

        if not isinstance(got, dict):
            return CheckResult(
                "ch34_todo_06",
                False,
                "failed",
                "To-Do 34.6 — Return a Dictionary Mapping Group -> Threshold",
                f"[{desc}] Expected a dict, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        exp = _ref_calibrate_group_thresholds(
            yt_raw, yp_raw, gr_raw, target_tpr=kwargs.get("target_tpr", 0.75)
        )
        got_norm = {str(k): v for k, v in got.items()}
        exp_norm = {str(k): v for k, v in exp.items()}
        if set(got_norm.keys()) != set(exp_norm.keys()):
            errors.append(
                f"[{desc}] Group keys mismatch: expected {sorted(exp_norm)}, got {sorted(got_norm)}."
            )
            continue

        for k, ev in exp_norm.items():
            gv = got_norm[k]
            if not _is_np_or_py_number(gv) or not math.isclose(float(gv), ev, abs_tol=1e-4):
                errors.append(
                    f"[{desc}] group={k!r}: expected highest qualifying threshold {ev}, got {gv!r}"
                )

    if errors:
        return CheckResult(
            "ch34_todo_06",
            False,
            "failed",
            "To-Do 34.6 — Check Highest Qualifying Threshold Per Group & Fallback 0.0",
            "For each group, iterate sorted unique probabilities in descending order and select the first (highest) threshold where `tpr >= target_tpr` (fallback `0.0` if no positives or unreachable), rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch34_todo_06",
        True,
        "passed",
        "To-Do 34.6 Passed — Group-Specific Threshold Calibration Verified!",
        "Selected the highest threshold per subgroup achieving `target_tpr` and cleanly fell back to `0.0` when a group had zero positives.",
        tip="Choosing the highest threshold that still satisfies `TPR >= target_tpr` maximizes specificity (minimizes unnecessary false alarms) while honoring Equal Opportunity!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_todo_06",
        chapter="ch34",
        title="To-Do 34.6: Group Threshold Calibration for Equal Opportunity (calibrate_group_thresholds_for_tpr)",
        book_section="Section 34.3.5 (Post-Processing Fairness Mitigation)",
        points=10,
        hints={
            1: "For each sorted unique group, check if `n_pos = int(np.sum(yt_g == 1)) == 0`; if so, set threshold to `0.0`.",
            2: "Loop over candidate thresholds in descending order: `for t in np.sort(np.unique(yp_g))[::-1]:` and stop at the first `t` with `np.sum((yt_g == 1) & (yp_g >= t)) / n_pos >= target_tpr`.",
            3: "def calibrate_group_thresholds_for_tpr(y_true, y_prob, groups, target_tpr=0.75):\n    if not (0.0 < target_tpr <= 1.0):\n        raise ValueError('target_tpr must lie in (0, 1]')\n    yt, yp, gr = np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float), np.asarray(groups)\n    out = {}\n    for g in sorted(pd.unique(gr)):\n        m = gr == g\n        yt_g, yp_g = yt[m], yp[m]\n        n_pos = int(np.sum(yt_g == 1))\n        best_t = 0.0\n        if n_pos > 0:\n            for t in np.sort(np.unique(yp_g))[::-1]:\n                if np.sum((yt_g == 1) & (yp_g >= t)) / n_pos >= target_tpr:\n                    best_t = float(t)\n                    break\n        out[g.item() if hasattr(g, 'item') else g] = round(best_t, 4)\n    return out",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 34.1 (30 pts): audit_rural_urban_fairness_mch(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _ref_audit_rural_urban_mch(mch_df: Any, seed: int = 42) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    work = mch_df.copy()
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    work["anc_visits"] = work["anc_visits"].replace(999, np.nan)
    work["is_urban"] = (work["residence"] == "Urban").astype(int)
    w_map = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
    work["wealth_rank"] = work["wealth_quintile"].map(w_map)
    feat_cols = [
        "maternal_age_years",
        "anc_visits",
        "maternal_hb_g_dl",
        "wealth_rank",
        "is_urban",
    ]
    clean = work[["child_stunted", "residence"] + feat_cols].dropna()
    X = clean[feat_cols]
    y = clean["child_stunted"].astype(int)
    g = clean["residence"]

    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g, test_size=0.30, stratify=y, random_state=seed
    )
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X_tr, y_tr)
    prob_tr = pipe.predict_proba(X_tr)[:, 1]
    prob_te = pipe.predict_proba(X_te)[:, 1]

    fpr_tr, tpr_tr, th_tr = roc_curve(y_tr, prob_tr)
    best_t = float(th_tr[int(np.argmax(tpr_tr - fpr_tr))])
    pred_te = (prob_te >= best_t).astype(int)

    return {
        "overall_auc": round(float(roc_auc_score(y_te, prob_te)), 4),
        "threshold": round(best_t, 4),
        "subgroup_table": _ref_subgroup_performance(y_te, prob_te, g_te, threshold=best_t),
        "fairness_gaps": _ref_fairness_gaps(y_te, pred_te, g_te),
    }


def _verify_exam_01(fn: Any) -> CheckResult:
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable(
        "ch34_exam_01", "Exam Task 34.1", fn, "audit_rural_urban_fairness_mch", 30
    )
    if bad is not None:
        return bad

    mch_df = load_data("mch_survey_sample", as_pandas=True)
    expected_keys = ["overall_auc", "threshold", "subgroup_table", "fairness_gaps"]
    expected_sub_cols = ["group", "n", "prevalence", "roc_auc", "tpr", "fpr", "brier"]
    expected_gap_keys = [
        "demographic_parity_diff",
        "equal_opportunity_diff",
        "max_fpr_diff",
        "equalized_odds_diff",
    ]
    errors = []

    for seed in (42, 7):
        df_copy = mch_df.copy(deep=True)
        desc = f"mch_survey_sample (seed={seed})"
        try:
            got = fn(df_copy, seed=seed)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch34_exam_01",
                False,
                "failed",
                "Exam Task 34.1 — Check Function Signature `audit_rural_urban_fairness_mch(mch_df, seed=42)`",
                f"Failed on {desc} with TypeError: {exc}",
                points_earned=0,
                points_possible=30,
            )
        except Exception as exc:
            return CheckResult(
                "ch34_exam_01",
                False,
                "failed",
                "Exam Task 34.1 — Runtime Error in `audit_rural_urban_fairness_mch`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch34_exam_01",
                "Exam Task 34.1",
                f"audit_rural_urban_fairness_mch(mch_df, seed={seed})",
                30,
            )

        if not df_copy.equals(mch_df):
            errors.append(
                f"[{desc}] Input DataFrame `mch_df` was mutated! Start with `work = mch_df.copy()`."
            )

        key_err = _check_dict_keys("ch34_exam_01", "Exam Task 34.1", got, expected_keys, 30)
        if key_err is not None:
            return key_err

        ref = _ref_audit_rural_urban_mch(mch_df, seed=seed)
        if not _is_np_or_py_number(got["overall_auc"]) or not math.isclose(
            float(got["overall_auc"]), ref["overall_auc"], abs_tol=1e-3
        ):
            errors.append(
                f"[{desc}] `overall_auc`: expected {ref['overall_auc']}, got {got['overall_auc']!r}"
            )
        if not _is_np_or_py_number(got["threshold"]) or not math.isclose(
            float(got["threshold"]), ref["threshold"], abs_tol=1e-3
        ):
            errors.append(
                f"[{desc}] `threshold`: expected {ref['threshold']} (train Youden's J), got {got['threshold']!r}"
            )

        sub_df = got["subgroup_table"]
        if not isinstance(sub_df, pd.DataFrame) or list(sub_df.columns) != expected_sub_cols:
            errors.append(
                f"[{desc}] `subgroup_table` must be a DataFrame with columns {expected_sub_cols}."
            )
        else:
            sub_r = sub_df.reset_index(drop=True)
            ref_sub = ref["subgroup_table"]
            if list(sub_r["group"].astype(str)) != list(ref_sub["group"].astype(str)):
                errors.append(
                    f"[{desc}] `subgroup_table` groups mismatch: expected {list(ref_sub['group'])}, got {list(sub_r['group'])}."
                )
            else:
                for i in range(len(ref_sub)):
                    gl = ref_sub.loc[i, "group"]
                    if int(sub_r.loc[i, "n"]) != int(ref_sub.loc[i, "n"]):
                        errors.append(f"[{desc}] {gl} `n`: expected {ref_sub.loc[i, 'n']}, got {sub_r.loc[i, 'n']!r}")
                    for col in ("prevalence", "roc_auc", "tpr", "fpr", "brier"):
                        gv, ev = sub_r.loc[i, col], ref_sub.loc[i, col]
                        if not _is_np_or_py_number(gv) or not math.isclose(float(gv), float(ev), abs_tol=1e-3):
                            errors.append(f"[{desc}] {gl} `{col}`: expected {ev}, got {gv!r}")

        fg = got["fairness_gaps"]
        fg_err = _check_dict_keys("ch34_exam_01", "Exam Task 34.1", fg, expected_gap_keys, 30)
        if fg_err is not None:
            errors.append(f"[{desc}] `fairness_gaps` keys mismatch: expected {expected_gap_keys}.")
        else:
            for k, ev in ref["fairness_gaps"].items():
                gv = fg[k]
                if not _is_np_or_py_number(gv) or not math.isclose(float(gv), ev, abs_tol=1e-3):
                    errors.append(f"[{desc}] fairness_gaps[{k!r}]: expected {ev}, got {gv!r}")

    if errors:
        return CheckResult(
            "ch34_exam_01",
            False,
            "failed",
            "Exam Task 34.1 — Check Leak-Free MCH Cleaning, Train Youden Cutoff & Rural/Urban Audit",
            "Clean `mch_df` on a `.copy()`, stratified 70/30 split, fit `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])`, choose Youden's J threshold on `(y_tr, prob_tr)`, and audit Rural vs. Urban on the test set.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch34_exam_01",
        True,
        "passed",
        "Exam Task 34.1 Passed (30/30 pts) — Rural vs. Urban Clinical Fairness Audit Verified!",
        "Verified leak-free training, train-only Youden threshold selection, and Rural/Urban disaggregation across multiple seeds (seed=42 overall AUC=0.6471, EO gap=0.7333).",
        points_earned=30,
        points_possible=30,
        tip="Selecting Youden's J strictly on training probabilities prevents threshold leakage while exposing dramatic subgroup sensitivity disparities on held-out patients!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_exam_01",
        chapter="ch34",
        title="Exam 34.1: End-to-End Rural vs. Urban MCH Fairness Audit (audit_rural_urban_fairness_mch)",
        book_section="Chapter 34 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Copy `mch_df`, standardize `residence` (`str.strip().str.title()`), replace `anc_visits == 999` with `np.nan`, encode `is_urban` and `wealth_rank` (Poorest=1..Richest=5), and drop missing rows across `['child_stunted', 'residence'] + feat_cols`.",
            2: "Run `train_test_split(X, y, g, test_size=0.30, stratify=y, random_state=seed)`, fit scaled `LogisticRegression(max_iter=500)`, select Youden's J threshold on `(y_tr, pipe.predict_proba(X_tr)[:, 1])`, then evaluate on `(y_te, prob_te, g_te)`.",
            3: "def audit_rural_urban_fairness_mch(mch_df, seed=42):\n    # clean copy, build X, y, g, split 70/30 stratified\n    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])\n    pipe.fit(X_tr, y_tr)\n    fpr, tpr, th = roc_curve(y_tr, pipe.predict_proba(X_tr)[:, 1])\n    best_t = float(th[int(np.argmax(tpr - fpr))])\n    prob_te = pipe.predict_proba(X_te)[:, 1]\n    pred_te = (prob_te >= best_t).astype(int)\n    return {'overall_auc': round(float(roc_auc_score(y_te, prob_te)), 4), 'threshold': round(best_t, 4), 'subgroup_table': subgroup_performance_table(y_te, prob_te, g_te, threshold=best_t), 'fairness_gaps': compute_fairness_gaps(y_te, pred_te, g_te)}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 34.2 (35 pts): mitigate_equal_opportunity_gap(...)
# ---------------------------------------------------------------------------
def _ref_mitigate_eo(
    y_true_val: Any,
    y_prob_val: Any,
    g_val: Any,
    y_true_test: Any,
    y_prob_test: Any,
    g_test: Any,
    target_tpr: float = 0.75,
) -> dict[str, Any]:
    import numpy as np

    yt_te = np.asarray(y_true_test, dtype=int)
    yp_te = np.asarray(y_prob_test, dtype=float)
    gr_te = np.asarray(g_test)

    pred_before = (yp_te >= 0.5).astype(int)
    before_eo = _ref_fairness_gaps(yt_te, pred_before, gr_te)["equal_opportunity_diff"]

    grp_th = _ref_calibrate_group_thresholds(
        y_true_val, y_prob_val, g_val, target_tpr=target_tpr
    )
    pred_after = np.array(
        [int(p >= grp_th.get(_py_scalar(g), 0.5)) for p, g in zip(yp_te, gr_te)],
        dtype=int,
    )
    after_eo = _ref_fairness_gaps(yt_te, pred_after, gr_te)["equal_opportunity_diff"]
    return {
        "before_eo_diff": before_eo,
        "after_eo_diff": after_eo,
        "group_thresholds": grp_th,
        "gap_reduced": bool(after_eo < before_eo),
    }


def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable(
        "ch34_exam_02", "Exam Task 34.2", fn, "mitigate_equal_opportunity_gap", 35
    )
    if bad is not None:
        return bad

    def _make_cohort(seed: int) -> tuple[Any, Any, Any]:
        rng = np.random.default_rng(seed)
        n_r, n_u = 160, 160
        y_r = rng.binomial(1, 0.35, size=n_r)
        p_r = np.clip(
            np.where(y_r == 1, rng.normal(0.54, 0.13, n_r), rng.normal(0.28, 0.12, n_r)),
            0.01,
            0.99,
        )
        y_u = rng.binomial(1, 0.35, size=n_u)
        p_u = np.clip(
            np.where(y_u == 1, rng.normal(0.76, 0.11, n_u), rng.normal(0.40, 0.13, n_u)),
            0.01,
            0.99,
        )
        return (
            np.concatenate([y_r, y_u]),
            np.concatenate([p_r, p_u]),
            np.array(["Rural"] * n_r + ["Urban"] * n_u),
        )

    yv1, pv1, gv1 = _make_cohort(42)
    yt1, pt1, gt1 = _make_cohort(99)

    yv2, pv2, gv2 = _make_cohort(11)
    yt2, pt2, gt2 = _make_cohort(27)

    cases = [
        (yv1, pv1, gv1, yt1, pt1, gt1, {"target_tpr": 0.75}, "hospital triage cohort (target_tpr=0.75)"),
        (yv2, pv2, gv2, yt2, pt2, gt2, {"target_tpr": 0.80}, "second triage cohort (target_tpr=0.80)"),
    ]
    expected_keys = [
        "before_eo_diff",
        "after_eo_diff",
        "group_thresholds",
        "gap_reduced",
    ]
    errors = []

    for yv, pv, gv, yt, pt, gt, kwargs, desc in cases:
        try:
            got = fn(yv.copy(), pv.copy(), gv.copy(), yt.copy(), pt.copy(), gt.copy(), **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch34_exam_02",
                False,
                "failed",
                "Exam Task 34.2 — Runtime Error in `mitigate_equal_opportunity_gap`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch34_exam_02",
                "Exam Task 34.2",
                "mitigate_equal_opportunity_gap(...)",
                35,
            )

        key_err = _check_dict_keys("ch34_exam_02", "Exam Task 34.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        ref = _ref_mitigate_eo(
            yv, pv, gv, yt, pt, gt, target_tpr=kwargs.get("target_tpr", 0.75)
        )
        for k in ("before_eo_diff", "after_eo_diff"):
            gv_val = got[k]
            if not _is_np_or_py_number(gv_val) or not math.isclose(
                float(gv_val), ref[k], abs_tol=1e-4
            ):
                errors.append(f"[{desc}] `{k}`: expected {ref[k]}, got {gv_val!r}")

        if not _is_bool(got["gap_reduced"]) or bool(got["gap_reduced"]) != ref["gap_reduced"]:
            errors.append(
                f"[{desc}] `gap_reduced`: expected {ref['gap_reduced']}, got {got['gap_reduced']!r}"
            )

        if not isinstance(got["group_thresholds"], dict):
            errors.append(f"[{desc}] `group_thresholds` must be a dict.")
        else:
            gt_norm = {str(k): v for k, v in got["group_thresholds"].items()}
            ref_norm = {str(k): v for k, v in ref["group_thresholds"].items()}
            if set(gt_norm.keys()) != set(ref_norm.keys()):
                errors.append(
                    f"[{desc}] `group_thresholds` keys mismatch: expected {sorted(ref_norm)}, got {sorted(gt_norm)}."
                )
            else:
                for gk, ev in ref_norm.items():
                    if not _is_np_or_py_number(gt_norm[gk]) or not math.isclose(
                        float(gt_norm[gk]), ev, abs_tol=1e-4
                    ):
                        errors.append(
                            f"[{desc}] `group_thresholds[{gk!r}]`: expected {ev}, got {gt_norm[gk]!r}"
                        )

    if errors:
        return CheckResult(
            "ch34_exam_02",
            False,
            "failed",
            "Exam Task 34.2 — Check Validation-Only Threshold Fitting & Test Evaluation",
            "Evaluate uniform `0.5` threshold on test for `before_eo_diff`, learn group thresholds strictly on validation via `calibrate_group_thresholds_for_tpr`, apply them to test for `after_eo_diff`, and set `gap_reduced = bool(after_eo_diff < before_eo_diff)`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch34_exam_02",
        True,
        "passed",
        "Exam Task 34.2 Passed (35/35 pts) — Post-Processing Fairness Mitigation Verified!",
        "Learned group-specific thresholds strictly on validation data and slashed the held-out test Equal Opportunity gap without retraining!",
        points_earned=35,
        points_possible=35,
        tip="Always fit post-processing thresholds on a separate validation set — never on the final test set — so your post-mitigation audit remains genuinely unbiased.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_exam_02",
        chapter="ch34",
        title="Exam 34.2: Post-Processing Equal Opportunity Mitigation (mitigate_equal_opportunity_gap)",
        book_section="Chapter 34 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Compute `before_eo_diff` on `(y_true_test, (y_prob_test >= 0.5).astype(int), g_test)` using `compute_fairness_gaps`.",
            2: "Learn `group_thresholds` strictly on `(y_true_val, y_prob_val, g_val)` via `calibrate_group_thresholds_for_tpr(..., target_tpr=target_tpr)`, then threshold each test sample with `group_thresholds[g]`.",
            3: "def mitigate_equal_opportunity_gap(y_true_val, y_prob_val, g_val, y_true_test, y_prob_test, g_test, target_tpr=0.75):\n    yp_te = np.asarray(y_prob_test, dtype=float)\n    before = compute_fairness_gaps(y_true_test, (yp_te >= 0.5).astype(int), g_test)['equal_opportunity_diff']\n    th = calibrate_group_thresholds_for_tpr(y_true_val, y_prob_val, g_val, target_tpr=target_tpr)\n    pred_after = np.array([int(p >= th[g]) for p, g in zip(yp_te, g_test)], dtype=int)\n    after = compute_fairness_gaps(y_true_test, pred_after, g_test)['equal_opportunity_diff']\n    return {'before_eo_diff': before, 'after_eo_diff': after, 'group_thresholds': th, 'gap_reduced': bool(after < before)}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 34.3 (35 pts): clinical_shift_monitoring_report(baseline_df, current_df, feature_cols)
# ---------------------------------------------------------------------------
def _ref_shift_report(
    baseline_df: Any, current_df: Any, feature_cols: list[str]
) -> dict[str, Any]:
    import pandas as pd

    rows = []
    for col in feature_cols:
        res = _ref_psi(baseline_df[col].dropna(), current_df[col].dropna())
        rows.append(
            {
                "feature": col,
                "psi": res["psi"],
                "drift_status": res["drift_status"],
            }
        )
    psi_df = pd.DataFrame(rows, columns=["feature", "psi", "drift_status"])
    max_psi = round(float(psi_df["psi"].max()), 4) if len(psi_df) > 0 else 0.0
    return {
        "feature_psi": psi_df,
        "max_psi": max_psi,
        "retrain_recommended": bool(max_psi >= 0.25),
    }


def _verify_exam_03(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    bad = _check_callable(
        "ch34_exam_03",
        "Exam Task 34.3",
        fn,
        "clinical_shift_monitoring_report",
        35,
    )
    if bad is not None:
        return bad

    rng = np.random.default_rng(77)
    base_df = pd.DataFrame(
        {
            "maternal_age_years": rng.normal(26.0, 5.0, 350),
            "anc_visits": rng.normal(4.2, 1.5, 350),
            "maternal_hb_g_dl": rng.normal(11.4, 1.2, 350),
        }
    )
    stable_df = pd.DataFrame(
        {
            "maternal_age_years": rng.normal(26.1, 5.0, 350),
            "anc_visits": rng.normal(4.1, 1.5, 350),
            "maternal_hb_g_dl": rng.normal(11.35, 1.2, 350),
        }
    )
    drifted_df = pd.DataFrame(
        {
            "maternal_age_years": rng.normal(26.2, 5.0, 350),
            "anc_visits": rng.normal(2.2, 1.5, 350),
            "maternal_hb_g_dl": rng.normal(10.0, 1.3, 350),
        }
    )
    cols = ["maternal_age_years", "anc_visits", "maternal_hb_g_dl"]
    expected_keys = ["feature_psi", "max_psi", "retrain_recommended"]
    expected_df_cols = ["feature", "psi", "drift_status"]
    errors = []

    for cur_df, desc in [
        (stable_df, "stable clinical stream (retrain_recommended=False)"),
        (drifted_df, "pandemic/famine shifted stream (retrain_recommended=True)"),
    ]:
        b_copy = base_df.copy(deep=True)
        c_copy = cur_df.copy(deep=True)
        try:
            got = fn(b_copy, c_copy, list(cols))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch34_exam_03",
                False,
                "failed",
                "Exam Task 34.3 — Runtime Error in `clinical_shift_monitoring_report`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch34_exam_03",
                "Exam Task 34.3",
                "clinical_shift_monitoring_report(baseline_df, current_df, feature_cols)",
                35,
            )

        if not b_copy.equals(base_df) or not c_copy.equals(cur_df):
            errors.append(f"[{desc}] Mutated input DataFrame(s)!")

        key_err = _check_dict_keys("ch34_exam_03", "Exam Task 34.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        ref = _ref_shift_report(base_df, cur_df, cols)
        f_df = got["feature_psi"]
        if not isinstance(f_df, pd.DataFrame) or list(f_df.columns) != expected_df_cols:
            errors.append(
                f"[{desc}] `feature_psi` must be a DataFrame with columns {expected_df_cols}."
            )
        else:
            f_r = f_df.reset_index(drop=True)
            ref_r = ref["feature_psi"]
            if list(f_r["feature"]) != list(ref_r["feature"]):
                errors.append(
                    f"[{desc}] `feature` column mismatch: expected {list(ref_r['feature'])}, got {list(f_r['feature'])}."
                )
            else:
                for i in range(len(ref_r)):
                    feat = ref_r.loc[i, "feature"]
                    if not _is_np_or_py_number(f_r.loc[i, "psi"]) or not math.isclose(
                        float(f_r.loc[i, "psi"]), float(ref_r.loc[i, "psi"]), abs_tol=1e-3
                    ):
                        errors.append(
                            f"[{desc}] {feat} `psi`: expected {ref_r.loc[i, 'psi']}, got {f_r.loc[i, 'psi']!r}"
                        )
                    if f_r.loc[i, "drift_status"] != ref_r.loc[i, "drift_status"]:
                        errors.append(
                            f"[{desc}] {feat} `drift_status`: expected {ref_r.loc[i, 'drift_status']!r}, got {f_r.loc[i, 'drift_status']!r}"
                        )

        if not _is_np_or_py_number(got["max_psi"]) or not math.isclose(
            float(got["max_psi"]), ref["max_psi"], abs_tol=1e-3
        ):
            errors.append(
                f"[{desc}] `max_psi`: expected {ref['max_psi']}, got {got['max_psi']!r}"
            )
        if not _is_bool(got["retrain_recommended"]) or bool(got["retrain_recommended"]) != ref["retrain_recommended"]:
            errors.append(
                f"[{desc}] `retrain_recommended`: expected {ref['retrain_recommended']}, got {got['retrain_recommended']!r}"
            )

    if errors:
        return CheckResult(
            "ch34_exam_03",
            False,
            "failed",
            "Exam Task 34.3 — Check Multi-Feature PSI Table & Retraining Trigger",
            "Run `population_stability_index` on every column in `feature_cols`, assemble `feature_psi` with columns `['feature', 'psi', 'drift_status']`, compute `max_psi`, and trigger `retrain_recommended = bool(max_psi >= 0.25)`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch34_exam_03",
        True,
        "passed",
        "Exam Task 34.3 Passed (35/35 pts) — Clinical Covariate Shift Monitor Verified!",
        "Passed both stable hospital streams (`retrain_recommended=False`) and severe covariate shift streams (`retrain_recommended=True`).",
        points_earned=35,
        points_possible=35,
        tip="Monitoring feature-level PSI catches covariate shift weeks before delayed ground-truth clinical outcomes ever arrive!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch34_exam_03",
        chapter="ch34",
        title="Exam 34.3: Clinical Covariate Shift Monitoring Report (clinical_shift_monitoring_report)",
        book_section="Chapter 34 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Iterate through `col` in `feature_cols`, call `population_stability_index(baseline_df[col].dropna(), current_df[col].dropna())`, and append `{'feature': col, 'psi': res['psi'], 'drift_status': res['drift_status']}`.",
            2: "Create `feature_psi = pd.DataFrame(rows, columns=['feature', 'psi', 'drift_status'])`, compute `max_psi = round(float(feature_psi['psi'].max()), 4)`, and set `retrain_recommended = bool(max_psi >= 0.25)`.",
            3: "def clinical_shift_monitoring_report(baseline_df, current_df, feature_cols):\n    rows = []\n    for col in feature_cols:\n        res = population_stability_index(baseline_df[col].dropna(), current_df[col].dropna())\n        rows.append({'feature': col, 'psi': res['psi'], 'drift_status': res['drift_status']})\n    df_psi = pd.DataFrame(rows, columns=['feature', 'psi', 'drift_status'])\n    max_psi = round(float(df_psi['psi'].max()), 4) if len(df_psi) > 0 else 0.0\n    return {'feature_psi': df_psi, 'max_psi': max_psi, 'retrain_recommended': bool(max_psi >= 0.25)}",
        },
        verifier=_verify_exam_03,
    )
)
