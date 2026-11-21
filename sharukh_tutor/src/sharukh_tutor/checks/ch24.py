"""Chapter 24 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch24_todo_01..06) and the Chapter 24 Coding Exam
(ch24_exam_01..03) for Tree-Based Models, Ensembles, and Interpretability
(Decision Trees, Random Forests, Histogram Gradient Boosting, Permutation
Importance, and Exact Shapley Attribution).
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn inside the check
function, validates return structures, rounding, mathematical identities,
and non-mutation, while letting NotImplementedError propagate cleanly for
starter stubs.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, float, np.number)) and not isinstance(
        val, (bool, np.bool_)
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
        f"Calling {call_str} returned None instead of the expected result.",
        details=[
            "Remember: print() only displays text on screen; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reusable ML utilities always return structured values, dicts, or DataFrames.",
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


def _ref_gini_raw(y: Any) -> float:
    import numpy as np

    arr = np.asarray(y)
    if arr.size == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / arr.size
    return float(1.0 - np.sum(probs ** 2))


def _ref_entropy_raw(y: Any) -> float:
    import numpy as np

    arr = np.asarray(y)
    if arr.size == 0:
        return 0.0
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / arr.size
    probs = probs[probs > 0]
    val = float(-np.sum(probs * np.log2(probs)))
    return 0.0 if val == 0.0 else val


# ---------------------------------------------------------------------------
# GUIDED TO-DO 24.1: gini_impurity(y)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch24_todo_01", "To-Do 24.1", fn, "gini_impurity", 10)
    if bad is not None:
        return bad

    cases = [
        ([], 0.0, "empty array"),
        ([1, 1, 1, 1], 0.0, "pure node"),
        ([0, 0, 1, 1], 0.5, "50/50 binary split"),
        ([0, 0, 0, 0, 1], 0.32, "80/20 imbalanced binary"),
        ([0, 1, 2, 0, 1, 2], 0.6667, "balanced 3-class node"),
        (np.array([0, 0, 0, 1, 1, 1, 1]), 0.4898, "NumPy array input"),
    ]
    errors = []

    for y_in, exp, desc in cases:
        y_copy = y_in.copy() if isinstance(y_in, np.ndarray) else list(y_in)
        try:
            got = fn(y_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch24_todo_01",
                False,
                "failed",
                "To-Do 24.1 — Runtime Error in `gini_impurity`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=[
                    "Handle empty arrays explicitly: `if arr.size == 0: return 0.0`."
                ],
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch24_todo_01", "To-Do 24.1", "gini_impurity(y)", 10)

        if not _is_number(got) or math.isnan(float(got)):
            errors.append(f"[{desc}] Expected float {exp}, got {got!r}")
            continue

        if not math.isclose(float(got), exp, abs_tol=1e-4):
            extra = ""
            if math.isclose(float(got), round(1.0 - exp, 4), abs_tol=1e-4) and exp > 0:
                extra = " (Looks like sum(p^2) instead of 1 - sum(p^2)!)"
            errors.append(f"[{desc}] Expected {exp}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch24_todo_01",
            False,
            "failed",
            "To-Do 24.1 — Check Gini Impurity Formula & Empty-Node Handling",
            "Compute `1 - sum(p_k ** 2)` using `np.unique(arr, return_counts=True)` rounded to 4 decimals, returning `0.0` if empty.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch24_todo_01",
        True,
        "passed",
        "To-Do 24.1 Passed — Gini Impurity Verified!",
        "Passed empty-node (0.0), pure-node (0.0), 50/50 binary (0.5), imbalanced (0.32), and 3-class (0.6667) impurity tests.",
        tip="CART decision trees evaluate Gini impurity millions of times because squaring probabilities avoids costly logarithms.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_todo_01",
        chapter="ch24",
        title="To-Do 24.1: Gini Impurity From Scratch (gini_impurity)",
        book_section="Section 24.3.1 (Impurity Criteria: Gini & Entropy)",
        points=10,
        hints={
            1: "Convert `y` with `arr = np.asarray(y)`. If `arr.size == 0`, return `0.0`. Otherwise use `_, counts = np.unique(arr, return_counts=True)` to compute class probabilities `probs = counts / arr.size`.",
            2: "Compute `1.0 - np.sum(probs ** 2)` and return `round(float(...), 4)`.",
            3: "def gini_impurity(y):\n    arr = np.asarray(y)\n    if arr.size == 0:\n        return 0.0\n    _, counts = np.unique(arr, return_counts=True)\n    probs = counts / arr.size\n    return round(float(1.0 - np.sum(probs ** ___)), 4)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 24.2: shannon_entropy(y)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch24_todo_02", "To-Do 24.2", fn, "shannon_entropy", 10)
    if bad is not None:
        return bad

    cases = [
        ([], 0.0, "empty array"),
        ([0, 0, 0, 0], 0.0, "pure node"),
        ([0, 1, 0, 1], 1.0, "50/50 binary node (1 bit)"),
        ([0, 0, 0, 1], 0.8113, "75/25 imbalanced binary"),
        ([0, 1, 2, 3], 2.0, "4-class uniform node (2 bits)"),
    ]
    errors = []

    for y_in, exp, desc in cases:
        try:
            got = fn(list(y_in))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch24_todo_02",
                False,
                "failed",
                "To-Do 24.2 — Runtime Error in `shannon_entropy`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=[
                    "Handle empty arrays before dividing and guard against `log2(0)` by filtering `probs = probs[probs > 0]`."
                ],
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch24_todo_02", "To-Do 24.2", "shannon_entropy(y)", 10)

        if not _is_number(got) or math.isnan(float(got)):
            errors.append(f"[{desc}] Expected float {exp}, got {got!r}")
            continue

        if not math.isclose(float(got), exp, abs_tol=1e-4):
            extra = ""
            if exp > 0 and math.isclose(float(got), round(exp * math.log(2), 4), abs_tol=1e-3):
                extra = " (Looks like natural log `np.log` instead of base-2 `np.log2`!)"
            elif exp > 0 and math.isclose(float(got), -exp, abs_tol=1e-4):
                extra = " (Missing leading minus sign in `-np.sum(probs * np.log2(probs))`!)"
            errors.append(f"[{desc}] Expected {exp}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch24_todo_02",
            False,
            "failed",
            "To-Do 24.2 — Check Base-2 Shannon Entropy Formula",
            "Compute `-sum(p_k * log2(p_k))` rounded to 4 decimals, returning `0.0` for pure or empty arrays.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch24_todo_02",
        True,
        "passed",
        "To-Do 24.2 Passed — Base-2 Shannon Entropy Verified!",
        "Verified 0.0 bits (pure/empty), 1.0 bit (50/50 binary), 0.8113 bits (75/25), and 2.0 bits (4-class uniform).",
        tip="Information Gain = H(parent) - weighted H(children); it measures how many bits of uncertainty a split removes.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_todo_02",
        chapter="ch24",
        title="To-Do 24.2: Base-2 Shannon Entropy (shannon_entropy)",
        book_section="Section 24.3.1 (Impurity Criteria: Gini & Entropy)",
        points=10,
        hints={
            1: "Compute class probabilities `probs = counts / arr.size` via `np.unique(arr, return_counts=True)` and return `0.0` when `arr.size == 0`.",
            2: "Use base-2 logarithm `np.log2(probs)` (filtering `probs > 0`) and apply `-np.sum(probs * np.log2(probs))`.",
            3: "def shannon_entropy(y):\n    arr = np.asarray(y)\n    if arr.size == 0:\n        return 0.0\n    _, counts = np.unique(arr, return_counts=True)\n    probs = counts / arr.size\n    probs = probs[probs > 0]\n    ent = float(-np.sum(probs * np.log2(___)))\n    return round(0.0 if ent == 0.0 else ent, 4)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 24.3: best_stump_split(x_feature, y)
# ---------------------------------------------------------------------------
def _ref_best_stump(x_feature: Any, y: Any) -> tuple[float, float]:
    import numpy as np

    x = np.asarray(x_feature, dtype=float)
    y_arr = np.asarray(y)
    uniq = np.unique(x)
    midpoints = (uniq[:-1] + uniq[1:]) / 2.0
    best_t = float(midpoints[0])
    best_g = float("inf")
    n = y_arr.size
    for t in midpoints:
        left = y_arr[x <= t]
        right = y_arr[x > t]
        wg = (left.size / n) * _ref_gini_raw(left) + (right.size / n) * _ref_gini_raw(right)
        if wg < best_g:
            best_g = wg
            best_t = float(t)
    return (round(best_t, 4), round(best_g, 4))


def _verify_todo_03(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch24_todo_03", "To-Do 24.3", fn, "best_stump_split", 10)
    if bad is not None:
        return bad

    cases = [
        (
            [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
            [1, 1, 1, 0, 0, 0],
            "cleanly separable feature (midpoint 4.5, gini 0.0)",
        ),
        (
            [10.0, 9.2, 11.8, 12.5, 9.8, 11.4, 10.6, 12.1],
            [1, 1, 0, 0, 1, 0, 1, 0],
            "unsorted hemoglobin readings",
        ),
        (
            [1, 1, 2, 2, 3, 4, 5, 5],
            [1, 1, 1, 0, 0, 0, 0, 0],
            "duplicate discrete values with imperfect split",
        ),
    ]
    errors = []

    for x_in, y_in, desc in cases:
        x_copy, y_copy = list(x_in), list(y_in)
        try:
            got = fn(x_copy, y_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch24_todo_03",
                False,
                "failed",
                "To-Do 24.3 — Runtime Error in `best_stump_split`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch24_todo_03", "To-Do 24.3", "best_stump_split(x_feature, y)", 10)

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch24_todo_03",
                False,
                "failed",
                "To-Do 24.3 — Return a 2-Tuple `(best_thresh, best_weighted_gini)`",
                f"Expected a 2-tuple of floats, got {type(got).__name__} ({got!r}).",
                points_earned=0,
                points_possible=10,
            )

        if x_copy != x_in or y_copy != y_in:
            errors.append(f"[{desc}] Mutated caller input lists! Avoid in-place sorting.")

        exp_t, exp_g = _ref_best_stump(x_in, y_in)
        if not all(_is_number(v) for v in got):
            errors.append(f"[{desc}] Tuple elements must be numbers, got {got!r}")
            continue

        got_t, got_g = float(got[0]), float(got[1])
        if not math.isclose(got_t, exp_t, abs_tol=1e-4) or not math.isclose(got_g, exp_g, abs_tol=2e-4):
            extra = ""
            if math.isclose(got_g, exp_g, abs_tol=2e-4) and not math.isclose(got_t, exp_t, abs_tol=1e-4):
                extra = " (Weighted Gini matches, but test MIDPOINTS `(u[:-1] + u[1:]) / 2.0` between sorted unique values!)"
            errors.append(f"[{desc}] Expected {(exp_t, exp_g)}, got {(got_t, got_g)}{extra}")

    if errors:
        return CheckResult(
            "ch24_todo_03",
            False,
            "failed",
            "To-Do 24.3 — Check Midpoint Candidates & Weighted Gini Formula",
            "Test midpoints between sorted unique `x_feature` values using `x <= t` vs. `x > t`, minimizing `(n_L/n)*G(y_L) + (n_R/n)*G(y_R)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch24_todo_03",
        True,
        "passed",
        "To-Do 24.3 Passed — Greedy Decision Stump Split Finder Verified!",
        "Passed cleanly separable, unsorted continuous, and tied discrete feature split tests.",
        tip="A full CART Decision Tree is simply this greedy midpoint search applied recursively to every feature and child node!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_todo_03",
        chapter="ch24",
        title="To-Do 24.3: Greedy CART Decision Stump Split (best_stump_split)",
        book_section="Section 24.3.2 (Greedy Midpoint Splitting)",
        points=10,
        hints={
            1: "Sort unique values with `uniq = np.unique(x)` and form candidate thresholds `midpoints = (uniq[:-1] + uniq[1:]) / 2.0`.",
            2: "For each threshold `t`, partition `left = y_arr[x <= t]` and `right = y_arr[x > t]` and compute `w_gini = (len(left) / n) * g_left + (len(right) / n) * g_right`.",
            3: "def best_stump_split(x_feature, y):\n    x = np.asarray(x_feature, dtype=float)\n    y_arr = np.asarray(y)\n    uniq = np.unique(x)\n    midpoints = (uniq[:-1] + uniq[1:]) / 2.0\n    best_t, best_g = float(midpoints[0]), float('inf')\n    for t in midpoints:\n        left, right = y_arr[x <= t], y_arr[x > t]\n        wg = (len(left) / len(y_arr)) * gini_impurity(left) + (len(right) / len(y_arr)) * gini_impurity(right)\n        if wg < best_g:\n            best_g, best_t = wg, float(t)\n    return (round(best_t, 4), round(best_g, 4))",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 24.4: compare_tree_depths(X_train, y_train, X_val, y_val, depths, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    bad = _check_callable("ch24_todo_04", "To-Do 24.4", fn, "compare_tree_depths", 10)
    if bad is not None:
        return bad

    X, y = make_classification(
        n_samples=320,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=19,
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=19
    )
    depths = [1, 2, 4, 8, 12]
    expected_cols = ["max_depth", "train_auc", "val_auc", "gap"]

    try:
        got = fn(X_tr.copy(), y_tr.copy(), X_va.copy(), y_va.copy(), list(depths), seed=42)
    except NotImplementedError:
        raise
    except TypeError as exc:
        return CheckResult(
            "ch24_todo_04",
            False,
            "failed",
            "To-Do 24.4 — Check Signature `compare_tree_depths(X_train, y_train, X_val, y_val, depths, seed=42)`",
            f"Raised TypeError: {exc}",
            points_earned=0,
            points_possible=10,
        )
    except Exception as exc:
        return CheckResult(
            "ch24_todo_04",
            False,
            "failed",
            "To-Do 24.4 — Runtime Error in `compare_tree_depths`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got is None:
        return _none_return_trap(
            "ch24_todo_04",
            "To-Do 24.4",
            "compare_tree_depths(X_train, y_train, X_val, y_val, depths, seed=42)",
            10,
        )

    if not isinstance(got, pd.DataFrame):
        return CheckResult(
            "ch24_todo_04",
            False,
            "failed",
            "To-Do 24.4 — Return a Pandas DataFrame",
            f"Expected pd.DataFrame, got {type(got).__name__}.",
            points_earned=0,
            points_possible=10,
        )

    if list(got.columns) != expected_cols:
        return CheckResult(
            "ch24_todo_04",
            False,
            "failed",
            "To-Do 24.4 — Check DataFrame Columns",
            f"Expected columns {expected_cols}, got {list(got.columns)}.",
            points_earned=0,
            points_possible=10,
        )

    if len(got) != len(depths):
        return CheckResult(
            "ch24_todo_04",
            False,
            "failed",
            "To-Do 24.4 — Row Count Mismatch",
            f"Expected {len(depths)} rows (one per depth), got {len(got)}.",
            points_earned=0,
            points_possible=10,
        )

    errors = []
    for idx, d in enumerate(depths):
        clf = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr, y_tr)
        tr_auc = float(roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1]))
        va_auc = float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))
        gap = tr_auc - va_auc

        row = got.iloc[idx]
        if int(row["max_depth"]) != d:
            errors.append(f"Row {idx}: expected max_depth={d}, got {row['max_depth']!r}")
        if not math.isclose(float(row["train_auc"]), round(tr_auc, 4), abs_tol=2e-4):
            errors.append(
                f"depth={d} train_auc: expected {round(tr_auc, 4)}, got {row['train_auc']!r} "
                "(Use `predict_proba(X)[:, 1]` for ROC-AUC, not hard `.predict(X)`!)"
            )
        if not math.isclose(float(row["val_auc"]), round(va_auc, 4), abs_tol=2e-4):
            errors.append(
                f"depth={d} val_auc: expected {round(va_auc, 4)}, got {row['val_auc']!r}"
            )
        if not math.isclose(float(row["gap"]), round(gap, 4), abs_tol=3e-4):
            errors.append(
                f"depth={d} gap: expected {round(gap, 4)} (`train_auc - val_auc`), got {row['gap']!r}"
            )

    if errors:
        return CheckResult(
            "ch24_todo_04",
            False,
            "failed",
            "To-Do 24.4 — Check Depth Sweep AUCs & Generalization Gap",
            "Fit `DecisionTreeClassifier(max_depth=d, random_state=seed)`, score with `roc_auc_score(y, clf.predict_proba(X)[:, 1])`, and compute `gap = train_auc - val_auc` rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch24_todo_04",
        True,
        "passed",
        "To-Do 24.4 Passed — Tree Depth Overfitting Curve Verified!",
        "Accurately tracked train_auc, val_auc, and overfitting gap across max_depth = [1, 2, 4, 8, 12].",
        tip="Notice how train_auc climbs toward 1.0000 at depth=12 while val_auc stalls — classic high-variance tree memorization!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_todo_04",
        chapter="ch24",
        title="To-Do 24.4: Decision Tree Depth Overfitting Audit (compare_tree_depths)",
        book_section="Section 24.3.3 (Controlling Tree Complexity & Overfitting)",
        points=10,
        hints={
            1: "Loop over `d in depths`, fit `DecisionTreeClassifier(max_depth=d, random_state=seed).fit(X_train, y_train)`, and score probabilities with `clf.predict_proba(...)[:, 1]`.",
            2: "Compute `train_auc`, `val_auc`, and `gap = train_auc - val_auc`, rounding each to 4 decimals in a list of dicts.",
            3: "def compare_tree_depths(X_train, y_train, X_val, y_val, depths, seed=42):\n    rows = []\n    for d in depths:\n        clf = DecisionTreeClassifier(max_depth=d, random_state=seed).fit(X_train, y_train)\n        tr = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])\n        va = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])\n        rows.append({'max_depth': d, 'train_auc': round(float(tr), 4), 'val_auc': round(float(va), 4), 'gap': round(float(tr - va), 4)})\n    return pd.DataFrame(rows)",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 24.5: fit_ensemble_comparison(X_train, y_train, X_test, y_test, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    bad = _check_callable("ch24_todo_05", "To-Do 24.5", fn, "fit_ensemble_comparison", 10)
    if bad is not None:
        return bad

    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        weights=[0.75, 0.25],
        random_state=77,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=77
    )
    expected_cols = ["model", "roc_auc", "pr_auc"]
    expected_specs = [
        ("DecisionTree", DecisionTreeClassifier(max_depth=5, random_state=42)),
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(max_depth=4, random_state=42),
        ),
    ]

    try:
        got = fn(X_tr.copy(), y_tr.copy(), X_te.copy(), y_te.copy(), seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch24_todo_05",
            False,
            "failed",
            "To-Do 24.5 — Runtime Error in `fit_ensemble_comparison`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got is None:
        return _none_return_trap(
            "ch24_todo_05",
            "To-Do 24.5",
            "fit_ensemble_comparison(X_train, y_train, X_test, y_test, seed=42)",
            10,
        )

    if not isinstance(got, pd.DataFrame):
        return CheckResult(
            "ch24_todo_05",
            False,
            "failed",
            "To-Do 24.5 — Return a Pandas DataFrame",
            f"Expected pd.DataFrame, got {type(got).__name__}.",
            points_earned=0,
            points_possible=10,
        )

    if list(got.columns) != expected_cols:
        return CheckResult(
            "ch24_todo_05",
            False,
            "failed",
            "To-Do 24.5 — Check DataFrame Columns",
            f"Expected columns {expected_cols}, got {list(got.columns)}.",
            points_earned=0,
            points_possible=10,
        )

    if len(got) != 3:
        return CheckResult(
            "ch24_todo_05",
            False,
            "failed",
            "To-Do 24.5 — Expected 3 Model Rows",
            f"Expected 3 rows (`DecisionTree`, `RandomForest`, `HistGradientBoosting`), got {len(got)}.",
            points_earned=0,
            points_possible=10,
        )

    exp_map = {}
    for name, clf in expected_specs:
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        exp_map[name] = (
            round(float(roc_auc_score(y_te, proba)), 4),
            round(float(average_precision_score(y_te, proba)), 4),
        )

    errors = []
    got_models = list(got["model"])
    if set(got_models) != set(exp_map.keys()):
        errors.append(
            f"Model names mismatch: expected {list(exp_map.keys())}, got {got_models}."
        )
    else:
        for _, row in got.iterrows():
            m = str(row["model"])
            exp_roc, exp_pr = exp_map[m]
            if not math.isclose(float(row["roc_auc"]), exp_roc, abs_tol=2e-4):
                errors.append(f"[{m}] roc_auc: expected {exp_roc}, got {row['roc_auc']!r}")
            if not math.isclose(float(row["pr_auc"]), exp_pr, abs_tol=2e-4):
                errors.append(
                    f"[{m}] pr_auc: expected {exp_pr} (via `average_precision_score`), got {row['pr_auc']!r}"
                )

    if errors:
        return CheckResult(
            "ch24_todo_05",
            False,
            "failed",
            "To-Do 24.5 — Check Ensemble Hyperparameters & ROC/PR-AUC Scoring",
            "Fit DecisionTree(max_depth=5), RandomForest(n_estimators=100, max_depth=6), and HistGradientBoosting(max_depth=4) with `random_state=seed`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch24_todo_05",
        True,
        "passed",
        "To-Do 24.5 Passed — Three-Family Tree & Ensemble Comparison Verified!",
        "Matched DecisionTree, RandomForest, and HistGradientBoosting test ROC-AUC and PR-AUC scores to 4 decimal places.",
        tip="Always report PR-AUC alongside ROC-AUC on imbalanced clinical datasets where positive cases are < 25%.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_todo_05",
        chapter="ch24",
        title="To-Do 24.5: Tree vs. Random Forest vs. Boosting Benchmark (fit_ensemble_comparison)",
        book_section="Section 24.3.5 (Bagging vs. Gradient Boosting)",
        points=10,
        hints={
            1: "Instantiate the 3 models with exact hyperparameters: `DecisionTreeClassifier(max_depth=5, random_state=seed)`, `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)`, and `HistGradientBoostingClassifier(max_depth=4, random_state=seed)`.",
            2: "For each `(name, clf)`, fit on `(X_train, y_train)`, get `proba = clf.predict_proba(X_test)[:, 1]`, and compute `roc_auc_score` and `average_precision_score` rounded to 4 decimals.",
            3: "def fit_ensemble_comparison(X_train, y_train, X_test, y_test, seed=42):\n    models = [\n        ('DecisionTree', DecisionTreeClassifier(max_depth=5, random_state=seed)),\n        ('RandomForest', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed)),\n        ('HistGradientBoosting', HistGradientBoostingClassifier(max_depth=4, random_state=seed)),\n    ]\n    rows = []\n    for name, clf in models:\n        clf.fit(X_train, y_train)\n        proba = clf.predict_proba(X_test)[:, 1]\n        rows.append({'model': name, 'roc_auc': round(float(roc_auc_score(y_test, proba)), 4), 'pr_auc': round(float(average_precision_score(y_test, proba)), 4)})\n    return pd.DataFrame(rows)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 24.6: compute_permutation_ranking(model, X_test, y_test, feature_names, seed=42)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    bad = _check_callable("ch24_todo_06", "To-Do 24.6", fn, "compute_permutation_ranking", 10)
    if bad is not None:
        return bad

    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=11,
    )
    feat_names = ["wealth_rank", "anc_visits", "maternal_hb", "random_noise"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=11
    )
    rf = RandomForestClassifier(n_estimators=80, max_depth=5, random_state=42).fit(X_tr, y_tr)

    try:
        got = fn(rf, X_te.copy(), y_te.copy(), list(feat_names), seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch24_todo_06",
            False,
            "failed",
            "To-Do 24.6 — Runtime Error in `compute_permutation_ranking`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got is None:
        return _none_return_trap(
            "ch24_todo_06",
            "To-Do 24.6",
            "compute_permutation_ranking(model, X_test, y_test, feature_names, seed=42)",
            10,
        )

    expected_cols = ["feature", "importance_mean", "importance_std"]
    if not isinstance(got, pd.DataFrame) or list(got.columns) != expected_cols:
        return CheckResult(
            "ch24_todo_06",
            False,
            "failed",
            "To-Do 24.6 — Return DataFrame With Exact Columns",
            f"Expected pd.DataFrame with columns {expected_cols}, got {type(got).__name__} ({list(got.columns) if isinstance(got, pd.DataFrame) else got!r}).",
            points_earned=0,
            points_possible=10,
        )

    pi = permutation_importance(
        rf, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=42
    )
    exp_df = (
        pd.DataFrame(
            {
                "feature": list(feat_names),
                "importance_mean": [round(float(v), 4) for v in pi.importances_mean],
                "importance_std": [round(float(v), 4) for v in pi.importances_std],
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    errors = []
    if len(got) != len(exp_df):
        errors.append(f"Expected {len(exp_df)} rows, got {len(got)}.")
    else:
        got_reset = got.reset_index(drop=True)
        for i in range(len(exp_df)):
            if str(got_reset.loc[i, "feature"]) != str(exp_df.loc[i, "feature"]):
                errors.append(
                    f"Row {i} feature order mismatch: expected {exp_df.loc[i, 'feature']!r}, got {got_reset.loc[i, 'feature']!r} "
                    "(Make sure to sort by `importance_mean` descending!)"
                )
            if not math.isclose(
                float(got_reset.loc[i, "importance_mean"]),
                float(exp_df.loc[i, "importance_mean"]),
                abs_tol=2e-4,
            ):
                errors.append(
                    f"Row {i} importance_mean: expected {exp_df.loc[i, 'importance_mean']}, got {got_reset.loc[i, 'importance_mean']!r}"
                )
            if not math.isclose(
                float(got_reset.loc[i, "importance_std"]),
                float(exp_df.loc[i, "importance_std"]),
                abs_tol=2e-4,
            ):
                errors.append(
                    f"Row {i} importance_std: expected {exp_df.loc[i, 'importance_std']}, got {got_reset.loc[i, 'importance_std']!r}"
                )

    if errors:
        return CheckResult(
            "ch24_todo_06",
            False,
            "failed",
            "To-Do 24.6 — Check Permutation Importance Scoring, Sorting & Rounding",
            "Call `permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=seed)` and sort descending by `importance_mean`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch24_todo_06",
        True,
        "passed",
        "To-Do 24.6 Passed — Held-Out Permutation Importance Ranking Verified!",
        "Computed 10-repeat ROC-AUC permutation means/stds and sorted features descending cleanly.",
        tip="Held-out permutation importance exposes high-cardinality noise features that fool default MDI `.feature_importances_`!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_todo_06",
        chapter="ch24",
        title="To-Do 24.6: Held-Out Permutation Importance Ranking (compute_permutation_ranking)",
        book_section="Section 24.3.6 (Permutation Importance vs. MDI Bias)",
        points=10,
        hints={
            1: "Run `res = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=seed)`.",
            2: "Build a DataFrame with columns `feature`, `importance_mean` (`res.importances_mean`), and `importance_std` (`res.importances_std`), round numeric columns to 4 decimals, and sort by `importance_mean` descending.",
            3: "def compute_permutation_ranking(model, X_test, y_test, feature_names, seed=42):\n    res = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=seed)\n    df = pd.DataFrame({'feature': list(feature_names), 'importance_mean': np.round(res.importances_mean, 4), 'importance_std': np.round(res.importances_std, 4)})\n    return df.sort_values('importance_mean', ascending=___).reset_index(drop=True)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 24.1 (30 pts): exact_shapley_3features(predict_fn, x_row, baseline_row)
# ---------------------------------------------------------------------------
def _ref_exact_shapley_3(predict_fn: Any, x_row: Any, baseline_row: Any) -> dict[str, Any]:
    import numpy as np

    x = np.asarray(x_row, dtype=float)
    base = np.asarray(baseline_row, dtype=float)
    base_val = float(predict_fn(base))
    pred_val = float(predict_fn(x))
    contribs = np.zeros(3, dtype=float)
    perms = list(itertools.permutations([0, 1, 2]))
    for perm in perms:
        state = base.copy()
        prev_val = base_val
        for idx in perm:
            state[idx] = x[idx]
            curr_val = float(predict_fn(state))
            contribs[idx] += curr_val - prev_val
            prev_val = curr_val
    shap_vals = contribs / len(perms)
    return {
        "base_value": round(base_val, 4),
        "shap_values": [round(float(v), 4) for v in shap_vals],
        "prediction": round(pred_val, 4),
    }


def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    bad = _check_callable("ch24_exam_01", "Exam Task 24.1", fn, "exact_shapley_3features", 30)
    if bad is not None:
        return bad

    # Case 1: Purely additive linear model
    def linear_fn(v: Any) -> float:
        return float(0.25 + 0.40 * v[0] - 0.30 * v[1] + 0.15 * v[2])

    # Case 2: Nonlinear interaction model (tests symmetric credit splitting!)
    def interaction_fn(v: Any) -> float:
        return float(1.0 + 2.0 * v[0] - 1.0 * v[1] + 4.0 * v[0] * v[2])

    # Case 3: Fitted 3-feature RandomForest probability model
    rng = np.random.default_rng(24)
    X_toy = rng.normal(size=(120, 3))
    y_toy = ((X_toy[:, 0] - 0.8 * X_toy[:, 1] + 0.6 * X_toy[:, 0] * X_toy[:, 2]) > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=42).fit(X_toy, y_toy)

    def rf_proba(v: Any) -> float:
        return float(rf.predict_proba(np.asarray(v, dtype=float).reshape(1, -1))[0, 1])

    cases = [
        (linear_fn, [2.0, 1.0, 4.0], [0.0, 0.0, 0.0], "additive linear function"),
        (interaction_fn, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], "nonlinear interaction (4*x0*x2 splits 2.0 each)"),
        (rf_proba, [1.4, -1.1, 0.9], [0.0, 0.0, 0.0], "fitted 3-feature RandomForest probability"),
    ]
    expected_keys = ["base_value", "shap_values", "prediction"]
    errors = []

    for pred_fn, x_in, base_in, desc in cases:
        x_copy, base_copy = list(x_in), list(base_in)
        try:
            got = fn(pred_fn, x_copy, base_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch24_exam_01",
                False,
                "failed",
                "Exam Task 24.1 — Runtime Error in `exact_shapley_3features`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch24_exam_01",
                "Exam Task 24.1",
                "exact_shapley_3features(predict_fn, x_row, baseline_row)",
                30,
            )

        key_err = _check_dict_keys("ch24_exam_01", "Exam Task 24.1", got, expected_keys, 30)
        if key_err is not None:
            return key_err

        if x_copy != x_in or base_copy != base_in:
            errors.append(f"[{desc}] Mutated caller `x_row` or `baseline_row`! Copy `base` before modifying.")

        exp = _ref_exact_shapley_3(pred_fn, x_in, base_in)

        if not _is_number(got["base_value"]) or not math.isclose(
            float(got["base_value"]), exp["base_value"], abs_tol=1e-4
        ):
            errors.append(f"[{desc}] base_value: expected {exp['base_value']}, got {got['base_value']!r}")

        if not _is_number(got["prediction"]) or not math.isclose(
            float(got["prediction"]), exp["prediction"], abs_tol=1e-4
        ):
            errors.append(f"[{desc}] prediction: expected {exp['prediction']}, got {got['prediction']!r}")

        sv = got["shap_values"]
        if not isinstance(sv, list) or len(sv) != 3 or not all(_is_number(v) for v in sv):
            errors.append(f"[{desc}] shap_values must be a Python list of 3 floats, got {sv!r}")
            continue

        for i in range(3):
            if not math.isclose(float(sv[i]), exp["shap_values"][i], abs_tol=1e-4):
                errors.append(
                    f"[{desc}] shap_values[{i}]: expected {exp['shap_values'][i]}, got {sv[i]!r} "
                    "(Average marginal contributions across ALL 6 permutations in `itertools.permutations([0, 1, 2])`!)"
                )

        # Efficiency Axiom check: base_value + sum(shap_values) == prediction
        eff_sum = float(got["base_value"]) + sum(float(v) for v in sv)
        if not math.isclose(eff_sum, float(got["prediction"]), abs_tol=5e-4):
            errors.append(
                f"[{desc}] Efficiency Axiom violated! base_value + sum(shap_values) = {eff_sum:.4f} != prediction {got['prediction']}"
            )

    if errors:
        return CheckResult(
            "ch24_exam_01",
            False,
            "failed",
            "Exam Task 24.1 — Check 6-Permutation Shapley Averaging & Efficiency Axiom",
            "Average marginal feature contributions across all 3! = 6 orders of [0, 1, 2], returning `base_value`, `shap_values`, and `prediction` rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch24_exam_01",
        True,
        "passed",
        "Exam Task 24.1 Passed (30/30 pts) — Exact 3-Feature Shapley Attribution Verified!",
        "Passed linear, nonlinear interaction (equal 50/50 split of 4*x0*x2), and RandomForest probability tests with the Efficiency Axiom verified on every case.",
        points_earned=30,
        points_possible=30,
        tip="TreeSHAP (in the `shap` library) computes these exact same Shapley values in polynomial time using tree path structure!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_exam_01",
        chapter="ch24",
        title="Exam 24.1: Exact 3-Feature Shapley Attribution (exact_shapley_3features)",
        book_section="Chapter 24 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Iterate over `perms = list(itertools.permutations([0, 1, 2]))`. For each permutation, start from `state = base.copy()` with `prev_val = predict_fn(base)`, switch `state[idx] = x[idx]` one feature at a time, and accumulate `curr_val - prev_val` into `contribs[idx]`.",
            2: "Divide `contribs` by `6` (`len(perms)`), then return `{'base_value': round(base_val, 4), 'shap_values': [round(float(v), 4) for v in shap_vals], 'prediction': round(pred_val, 4)}`.",
            3: "def exact_shapley_3features(predict_fn, x_row, baseline_row):\n    x = np.asarray(x_row, dtype=float)\n    base = np.asarray(baseline_row, dtype=float)\n    base_val, pred_val = float(predict_fn(base)), float(predict_fn(x))\n    contribs = np.zeros(3, dtype=float)\n    perms = list(itertools.permutations([0, 1, 2]))\n    for perm in perms:\n        state, prev = base.copy(), base_val\n        for idx in perm:\n            state[idx] = x[idx]\n            curr = float(predict_fn(state))\n            contribs[idx] += curr - ___\n            prev = curr\n    shap_vals = contribs / len(perms)\n    return {'base_value': round(base_val, 4), 'shap_values': [round(float(v), 4) for v in shap_vals], 'prediction': round(pred_val, 4)}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 24.2 (35 pts): tune_random_forest_oob(X_train, y_train, max_depths, seed=42)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    bad = _check_callable("ch24_exam_02", "Exam Task 24.2", fn, "tune_random_forest_oob", 35)
    if bad is not None:
        return bad

    X, y = make_classification(
        n_samples=300,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=2026,
    )
    depths = [2, 4, 6, 10]
    expected_keys = ["best_depth", "best_oob_score", "oob_table"]

    try:
        got = fn(X.copy(), y.copy(), list(depths), seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch24_exam_02",
            False,
            "failed",
            "Exam Task 24.2 — Runtime Error in `tune_random_forest_oob`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap(
            "ch24_exam_02",
            "Exam Task 24.2",
            "tune_random_forest_oob(X_train, y_train, max_depths, seed=42)",
            35,
        )

    key_err = _check_dict_keys("ch24_exam_02", "Exam Task 24.2", got, expected_keys, 35)
    if key_err is not None:
        return key_err

    rows = []
    best_d = depths[0]
    best_oob = -1.0
    for d in depths:
        rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=d,
            oob_score=True,
            random_state=42,
        ).fit(X, y)
        score = float(rf.oob_score_)
        rows.append({"max_depth": int(d), "oob_score": round(score, 4)})
        if score > best_oob:
            best_oob = score
            best_d = int(d)

    exp_table = pd.DataFrame(rows)
    errors = []

    if not isinstance(got["best_depth"], int) or isinstance(got["best_depth"], bool):
        errors.append(f"best_depth must be a Python int, got {type(got['best_depth']).__name__} ({got['best_depth']!r}).")
    elif got["best_depth"] != best_d:
        errors.append(f"best_depth: expected {best_d}, got {got['best_depth']!r}")

    if not _is_number(got["best_oob_score"]) or not math.isclose(
        float(got["best_oob_score"]), round(best_oob, 4), abs_tol=2e-4
    ):
        errors.append(
            f"best_oob_score: expected {round(best_oob, 4)}, got {got['best_oob_score']!r}"
        )

    tbl = got["oob_table"]
    if not isinstance(tbl, pd.DataFrame):
        errors.append(f"oob_table must be a pd.DataFrame, got {type(tbl).__name__}.")
    elif list(tbl.columns) != ["max_depth", "oob_score"]:
        errors.append(f"oob_table columns: expected ['max_depth', 'oob_score'], got {list(tbl.columns)}.")
    elif len(tbl) != len(exp_table):
        errors.append(f"oob_table length: expected {len(exp_table)}, got {len(tbl)}.")
    else:
        tbl_reset = tbl.reset_index(drop=True)
        for i in range(len(exp_table)):
            if int(tbl_reset.loc[i, "max_depth"]) != int(exp_table.loc[i, "max_depth"]):
                errors.append(
                    f"oob_table row {i} max_depth: expected {exp_table.loc[i, 'max_depth']}, got {tbl_reset.loc[i, 'max_depth']!r}"
                )
            if not math.isclose(
                float(tbl_reset.loc[i, "oob_score"]),
                float(exp_table.loc[i, "oob_score"]),
                abs_tol=2e-4,
            ):
                errors.append(
                    f"oob_table row {i} oob_score: expected {exp_table.loc[i, 'oob_score']}, got {tbl_reset.loc[i, 'oob_score']!r}"
                )

    if errors:
        return CheckResult(
            "ch24_exam_02",
            False,
            "failed",
            "Exam Task 24.2 — Check Random Forest OOB Tuning Parameters & Return Structure",
            "Evaluate `RandomForestClassifier(n_estimators=120, max_depth=d, oob_score=True, random_state=seed)` across `max_depths` and return `best_depth`, `best_oob_score`, and `oob_table`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch24_exam_02",
        True,
        "passed",
        "Exam Task 24.2 Passed (35/35 pts) — Random Forest OOB Hyperparameter Tuner Verified!",
        f"Selected optimal depth={best_d} (OOB accuracy={round(best_oob, 4)}) with complete OOB audit table.",
        points_earned=35,
        points_possible=35,
        tip="Out-of-Bag (OOB) scoring evaluates each training row on the ~36.8% of bootstrap trees that never saw it — giving near-free internal validation!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_exam_02",
        chapter="ch24",
        title="Exam 24.2: Out-of-Bag Random Forest Depth Tuning (tune_random_forest_oob)",
        book_section="Chapter 24 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Loop over `d in max_depths`, fit `RandomForestClassifier(n_estimators=120, max_depth=d, oob_score=True, random_state=seed).fit(X_train, y_train)`, and read `rf.oob_score_`.",
            2: "Store rows `[{'max_depth': int(d), 'oob_score': round(float(rf.oob_score_), 4)}, ...]`, build `oob_table = pd.DataFrame(rows)`, and pick the row with maximum `oob_score`.",
            3: "def tune_random_forest_oob(X_train, y_train, max_depths, seed=42):\n    rows = []\n    best_d, best_oob = int(max_depths[0]), -1.0\n    for d in max_depths:\n        rf = RandomForestClassifier(n_estimators=120, max_depth=d, oob_score=___, random_state=seed).fit(X_train, y_train)\n        score = float(rf.oob_score_)\n        rows.append({'max_depth': int(d), 'oob_score': round(score, 4)})\n        if score > best_oob:\n            best_oob, best_d = score, int(d)\n    return {'best_depth': best_d, 'best_oob_score': round(best_oob, 4), 'oob_table': pd.DataFrame(rows)}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 24.3 (35 pts): public_health_tree_benchmark(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _ref_public_health_benchmark(mch_df: Any, seed: int = 42) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    work = mch_df.copy()
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    work["anc_visits"] = work["anc_visits"].replace(999, np.nan)
    work["is_urban"] = (work["residence"] == "Urban").astype(int)
    wealth_map = {
        "Poorest": 1,
        "Poorer": 2,
        "Middle": 3,
        "Richer": 4,
        "Richest": 5,
    }
    work["wealth_rank"] = (
        work["wealth_quintile"].astype(str).str.strip().str.title().map(wealth_map)
    )
    features = [
        "maternal_age_years",
        "anc_visits",
        "maternal_hb_g_dl",
        "is_urban",
        "wealth_rank",
    ]
    clean = work[features + ["child_stunted"]].dropna()
    X = clean[features]
    y = clean["child_stunted"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    specs = [
        ("DecisionTree", DecisionTreeClassifier(max_depth=5, random_state=seed)),
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed),
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingClassifier(max_depth=4, random_state=seed),
        ),
    ]
    rows = []
    fitted = {}
    for name, clf in specs:
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        rows.append(
            {
                "model": name,
                "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
                "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
            }
        )
        fitted[name] = clf

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    best_model = str(leaderboard.iloc[0]["model"])
    pi = permutation_importance(
        fitted[best_model],
        X_te,
        y_te,
        scoring="roc_auc",
        n_repeats=10,
        random_state=seed,
    )
    top_feature = features[int(np.argmax(pi.importances_mean))]
    return {
        "leaderboard": leaderboard,
        "best_model": best_model,
        "top_feature": top_feature,
    }


def _verify_exam_03(fn: Any) -> CheckResult:
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable("ch24_exam_03", "Exam Task 24.3", fn, "public_health_tree_benchmark", 35)
    if bad is not None:
        return bad

    df_mch = load_data("mch_survey_sample", as_pandas=True)
    df_copy = df_mch.copy(deep=True)
    expected_keys = ["leaderboard", "best_model", "top_feature"]

    try:
        got = fn(df_copy, seed=42)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch24_exam_03",
            False,
            "failed",
            "Exam Task 24.3 — Runtime Error in `public_health_tree_benchmark`",
            f"Raised {type(exc).__name__}: {exc}",
            details=[
                "Clean `residence`, replace `anc_visits == 999` with `np.nan`, map `wealth_quintile` to 1..5, and drop NaNs before splitting."
            ],
            points_earned=0,
            points_possible=35,
        )

    if got is None:
        return _none_return_trap(
            "ch24_exam_03",
            "Exam Task 24.3",
            "public_health_tree_benchmark(mch_df, seed=42)",
            35,
        )

    key_err = _check_dict_keys("ch24_exam_03", "Exam Task 24.3", got, expected_keys, 35)
    if key_err is not None:
        return key_err

    errors = []
    if not df_copy.equals(df_mch):
        errors.append("Input `mch_df` was mutated! Always start with `work = mch_df.copy()`.")

    exp = _ref_public_health_benchmark(df_mch, seed=42)
    exp_lb = exp["leaderboard"]
    got_lb = got["leaderboard"]

    if not isinstance(got_lb, pd.DataFrame):
        errors.append(f"leaderboard must be a pd.DataFrame, got {type(got_lb).__name__}.")
    elif list(got_lb.columns) != ["model", "roc_auc", "pr_auc"]:
        errors.append(
            f"leaderboard columns: expected ['model', 'roc_auc', 'pr_auc'], got {list(got_lb.columns)}."
        )
    elif len(got_lb) != 3:
        errors.append(f"leaderboard must have 3 rows, got {len(got_lb)}.")
    else:
        lb_reset = got_lb.reset_index(drop=True)
        for i in range(3):
            if str(lb_reset.loc[i, "model"]) != str(exp_lb.loc[i, "model"]):
                errors.append(
                    f"leaderboard row {i} model: expected {exp_lb.loc[i, 'model']!r}, got {lb_reset.loc[i, 'model']!r} "
                    "(Sort leaderboard descending by `roc_auc`!)"
                )
            if not math.isclose(
                float(lb_reset.loc[i, "roc_auc"]),
                float(exp_lb.loc[i, "roc_auc"]),
                abs_tol=2e-4,
            ):
                errors.append(
                    f"leaderboard row {i} roc_auc: expected {exp_lb.loc[i, 'roc_auc']}, got {lb_reset.loc[i, 'roc_auc']!r}"
                )
            if not math.isclose(
                float(lb_reset.loc[i, "pr_auc"]),
                float(exp_lb.loc[i, "pr_auc"]),
                abs_tol=2e-4,
            ):
                errors.append(
                    f"leaderboard row {i} pr_auc: expected {exp_lb.loc[i, 'pr_auc']}, got {lb_reset.loc[i, 'pr_auc']!r}"
                )

    if str(got["best_model"]) != exp["best_model"]:
        errors.append(f"best_model: expected {exp['best_model']!r}, got {got['best_model']!r}")

    if str(got["top_feature"]) != exp["top_feature"]:
        errors.append(
            f"top_feature: expected {exp['top_feature']!r} (from test-set ROC-AUC permutation importance on the winning model), got {got['top_feature']!r}"
        )

    if errors:
        return CheckResult(
            "ch24_exam_03",
            False,
            "failed",
            "Exam Task 24.3 — Check MCH Cleaning, Stratified 75/25 Split & Winning-Model Permutation Ranking",
            "Clean `mch_survey_sample`, split 75/25 stratified on `child_stunted`, benchmark DecisionTree/RandomForest/HistGradientBoosting, sort by `roc_auc` desc, and extract `top_feature` via test-set permutation importance.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch24_exam_03",
        True,
        "passed",
        "Exam Task 24.3 Passed (35/35 pts) — End-to-End Public Health Tree Benchmark Verified!",
        f"Cleaned mch_survey_sample (n=470 complete), ranked all 3 tree families (winner: {exp['best_model']}), and identified top risk driver `{exp['top_feature']}` via held-out permutation importance.",
        points_earned=35,
        points_possible=35,
        tip="Notice how empirical benchmarking prevented blind algorithm hype — on n=470 survey data, always report both simpler and boosted tree baselines!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch24_exam_03",
        chapter="ch24",
        title="Exam 24.3: Public Health Child Stunting Tree Benchmark (public_health_tree_benchmark)",
        book_section="Chapter 24 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Copy `mch_df`, standardize `residence`, replace `anc_visits == 999` with `np.nan`, compute `is_urban = (residence == 'Urban').astype(int)`, map `wealth_quintile` (`Poorest: 1 .. Richest: 5`) to `wealth_rank`, and `.dropna()` on the 5 features + `child_stunted`.",
            2: "Split with `train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)`. Fit `DecisionTreeClassifier(max_depth=5)`, `RandomForestClassifier(n_estimators=100, max_depth=6)`, and `HistGradientBoostingClassifier(max_depth=4)` (all `random_state=seed`), sort leaderboard by `roc_auc` descending.",
            3: "Pick `best_model = str(leaderboard.iloc[0]['model'])`, then run `permutation_importance(fitted[best_model], X_te, y_te, scoring='roc_auc', n_repeats=10, random_state=seed)` and set `top_feature = features[int(np.argmax(pi.importances_mean))]`.",
        },
        verifier=_verify_exam_03,
    )
)
