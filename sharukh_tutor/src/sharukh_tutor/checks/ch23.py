"""Chapter 23 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch23_todo_01..06) and the Chapter 23 Coding Exam
(ch23_exam_01..03) for Classification Foundations — Logistic Regression,
Calibration, ROC/PR Curves, and Class Imbalance.
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn inside the check
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


def _is_np_or_py_number(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(
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
        f"Calling {call_str} returned None instead of the computed classification result.",
        details=[
            "Remember: print() only displays text on screen; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reusable ML evaluation helpers always return structured numbers, tuples, dicts, or DataFrames.",
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
# GUIDED TO-DO 23.1: binary_cross_entropy(y_true, y_prob, eps=1e-15)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch23_todo_01", "To-Do 23.1", fn, "binary_cross_entropy", 10)
    if bad is not None:
        return bad

    cases = [
        ([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], {}, "well-calibrated probabilities"),
        ([1, 0, 1, 0], [1.0, 0.0, 0.0, 1.0], {}, "boundary 0.0 and 1.0 clipping"),
        ([1, 0], [0.0, 1.0], {"eps": 1e-6}, "custom eps=1e-6 clipping"),
        ([1, 1, 0, 0, 1], [0.7, 0.4, 0.3, 0.1, 0.85], {}, "mixed clinical probabilities"),
    ]
    errors = []

    for y_raw, p_raw, kwargs, desc in cases:
        y_copy, p_copy = list(y_raw), list(p_raw)
        try:
            got = fn(y_copy, p_copy, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch23_todo_01",
                False,
                "failed",
                "To-Do 23.1 — Check Function Signature `binary_cross_entropy(y_true, y_prob, eps=1e-15)`",
                f"Failed on {desc} with TypeError: {exc}",
                details=["Define `def binary_cross_entropy(y_true, y_prob, eps=1e-15):`."],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch23_todo_01",
                False,
                "failed",
                "To-Do 23.1 — Runtime Error in `binary_cross_entropy`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch23_todo_01", "To-Do 23.1", "binary_cross_entropy(y_true, y_prob)", 10)

        if y_copy != y_raw or p_copy != p_raw:
            errors.append(f"[{desc}] Mutated caller input list(s)! Use `np.asarray(...)` without in-place mutation.")

        eps = kwargs.get("eps", 1e-15)
        y_arr = np.asarray(y_raw, dtype=float)
        p_arr = np.clip(np.asarray(p_raw, dtype=float), eps, 1.0 - eps)
        exp = round(float(-np.mean(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr))), 5)

        if not _is_np_or_py_number(got) or not math.isfinite(float(got)):
            errors.append(
                f"[{desc}] Expected a finite float rounded to 5 decimals ({exp}), got {got!r} "
                "(Did you clip `y_prob` to `[eps, 1 - eps]` before taking `np.log`?)."
            )
            continue

        if not math.isclose(float(got), exp, abs_tol=1e-5):
            extra = ""
            if "eps" in kwargs:
                extra = " (Make sure you clip with the parameter `eps`, not a hardcoded `1e-15`!)"
            elif float(got) < 0:
                extra = " (Did you forget the leading minus sign in `-np.mean(...)`?)"
            errors.append(f"[{desc}] Expected {exp}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch23_todo_01",
            False,
            "failed",
            "To-Do 23.1 — Check Clipped Binary Cross-Entropy Formula & Rounding",
            "Clip `y_prob` to `[eps, 1 - eps]`, compute `-np.mean(y*log(p) + (1-y)*log(1-p))`, and round to 5 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch23_todo_01",
        True,
        "passed",
        "To-Do 23.1 Passed — Clipped Binary Cross-Entropy Verified!",
        "Passed standard probabilities, boundary 0.0/1.0 stability checks, and custom `eps` clipping.",
        tip="Clipping predicted probabilities to `[eps, 1 - eps]` prevents `log(0) -> -inf` and `0 * -inf -> nan` in every ML loss function.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_todo_01",
        chapter="ch23",
        title="To-Do 23.1: Clipped Binary Cross-Entropy (binary_cross_entropy)",
        book_section="Section 23.3.1 (Sigmoid & Log-Loss from Scratch)",
        points=10,
        hints={
            1: "Convert inputs to float NumPy arrays and clip `p = np.clip(y_prob, eps, 1.0 - eps)` so `np.log(0)` never occurs.",
            2: "Compute `loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))` and return `round(float(loss), 5)`.",
            3: "def binary_cross_entropy(y_true, y_prob, eps=1e-15):\n    y = np.asarray(y_true, dtype=float)\n    p = np.clip(np.asarray(y_prob, dtype=float), ___, 1.0 - ___)\n    loss = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))\n    return round(float(loss), 5)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 23.2: logistic_gd_from_scratch(X, y, lr=0.1, epochs=400)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch23_todo_02", "To-Do 23.2", fn, "logistic_gd_from_scratch", 10)
    if bad is not None:
        return bad

    rng1 = np.random.default_rng(42)
    X1_pos = rng1.normal(loc=[1.6, -1.4], scale=0.7, size=(60, 2))
    X1_neg = rng1.normal(loc=[-1.6, 1.4], scale=0.7, size=(60, 2))
    X1 = np.vstack([X1_neg, X1_pos])
    y1 = np.array([0] * 60 + [1] * 60, dtype=int)

    rng2 = np.random.default_rng(7)
    X2_pos = rng2.normal(loc=[1.2, 1.5, -1.2], scale=0.8, size=(50, 3))
    X2_neg = rng2.normal(loc=[-1.2, -1.5, 1.2], scale=0.8, size=(50, 3))
    X2 = np.vstack([X2_neg, X2_pos])
    y2 = np.array([0] * 50 + [1] * 50, dtype=int)

    cases = [
        (X1, y1, {}, "2D separable cohort (default lr=0.1, epochs=400)"),
        (X2, y2, {"lr": 0.2, "epochs": 250}, "3D separable cohort (lr=0.2, epochs=250)"),
    ]
    errors = []

    for X_raw, y_raw, kwargs, desc in cases:
        X_copy = X_raw.copy()
        y_copy = y_raw.copy()
        epochs = kwargs.get("epochs", 400)
        try:
            got = fn(X_copy, y_copy, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch23_todo_02",
                False,
                "failed",
                "To-Do 23.2 — Runtime Error in `logistic_gd_from_scratch`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch23_todo_02", "To-Do 23.2", "logistic_gd_from_scratch(X, y)", 10)

        if not isinstance(got, tuple) or len(got) != 3:
            return CheckResult(
                "ch23_todo_02",
                False,
                "failed",
                "To-Do 23.2 — Return a 3-Tuple `(w, b, loss_history)`",
                f"[{desc}] Expected a 3-tuple `(w_1d, b_float, loss_history)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(X_copy, X_raw) or not np.array_equal(y_copy, y_raw):
            errors.append(f"[{desc}] Mutated caller `X` or `y` array in place!")

        w, b, loss_history = got
        w_arr = np.asarray(w, dtype=float)
        if w_arr.ndim != 1 or w_arr.shape[0] != X_raw.shape[1] or not np.all(np.isfinite(w_arr)):
            errors.append(
                f"[{desc}] `w` must be a finite 1D array of shape ({X_raw.shape[1]},), got shape {w_arr.shape}."
            )
            continue

        if not _is_np_or_py_number(b) or not math.isfinite(float(b)):
            errors.append(f"[{desc}] `b` must be a finite scalar float, got {b!r}.")
            continue

        if not isinstance(loss_history, (list, tuple, np.ndarray)) or len(loss_history) not in (epochs, epochs + 1):
            errors.append(
                f"[{desc}] `loss_history` should have length {epochs} (one entry per epoch), got {len(loss_history) if hasattr(loss_history, '__len__') else type(loss_history).__name__}."
            )
            continue

        hist = [float(v) for v in loss_history]
        if not all(math.isfinite(v) for v in hist):
            errors.append(f"[{desc}] `loss_history` contains non-finite values (NaN/Inf).")
            continue

        if not (hist[-1] < hist[0]):
            errors.append(
                f"[{desc}] Final loss ({hist[-1]:.4f}) must be strictly lower than initial loss ({hist[0]:.4f}). "
                "Check gradient descent update sign: `w -= lr * dw`!"
            )

        z = np.clip(X_raw @ w_arr + float(b), -500.0, 500.0)
        preds = (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)
        acc = float(np.mean(preds == y_raw))
        if acc < 0.85:
            errors.append(
                f"[{desc}] Training accuracy was {acc:.4f} (expected >= 0.85). "
                "Check sigmoid `1 / (1 + exp(-z))` and gradient `(X.T @ (p - y)) / n`."
            )

    if errors:
        return CheckResult(
            "ch23_todo_02",
            False,
            "failed",
            "To-Do 23.2 — Check Pure NumPy Logistic Gradient Descent",
            "Initialize `w = np.zeros(n_features)`, `b = 0.0`, iterate `epochs` times computing sigmoid `p`, log-loss, and gradients `dw = (X.T @ (p - y)) / n`, `db = np.mean(p - y)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch23_todo_02",
        True,
        "passed",
        "To-Do 23.2 Passed — Pure NumPy Logistic Gradient Descent Verified!",
        "Loss decreased monotonically across both 2D and 3D cohorts and achieved >= 85% training accuracy with finite 1D weights.",
        tip="Logistic regression's binary cross-entropy loss is convex, so gradient descent converges reliably to the global optimum.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_todo_02",
        chapter="ch23",
        title="To-Do 23.2: Logistic Gradient Descent from Scratch (logistic_gd_from_scratch)",
        book_section="Section 23.3.1 (Sigmoid & Log-Loss from Scratch)",
        points=10,
        hints={
            1: "Start with `w = np.zeros(X.shape[1])` and `b = 0.0`. Each epoch, compute logits `z = X @ w + b` and sigmoid `p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))`.",
            2: "Record clipped binary cross-entropy in `loss_history`, compute `err = p - y`, `dw = (X.T @ err) / len(y)`, `db = float(np.mean(err))`, and update `w -= lr * dw`, `b -= lr * db`.",
            3: "def logistic_gd_from_scratch(X, y, lr=0.1, epochs=400):\n    X_arr = np.asarray(X, dtype=float)\n    y_arr = np.asarray(y, dtype=float)\n    w = np.zeros(X_arr.shape[1], dtype=float)\n    b, loss_history = 0.0, []\n    for _ in range(epochs):\n        p = 1.0 / (1.0 + np.exp(-np.clip(X_arr @ w + b, -500.0, 500.0)))\n        pc = np.clip(p, 1e-15, 1.0 - 1e-15)\n        loss_history.append(float(-np.mean(y_arr * np.log(pc) + (1.0 - y_arr) * np.log(1.0 - pc))))\n        err = p - y_arr\n        w -= lr * (X_arr.T @ err) / len(y_arr)\n        b -= lr * float(np.mean(err))\n    return (w, float(b), loss_history)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 23.3: clinical_confusion_metrics(y_true, y_pred)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch23_todo_03", "To-Do 23.3", fn, "clinical_confusion_metrics", 10)
    if bad is not None:
        return bad

    cases = [
        (
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "standard clinical screening cohort",
        ),
        (
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            "all-zero predictions (zero-denominator guard for precision/f1)",
        ),
        (
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            "all-one predictions (zero-denominator guard for npv)",
        ),
    ]
    expected_keys = ["sensitivity", "specificity", "precision", "npv", "f1"]
    errors = []

    for yt_raw, yp_raw, desc in cases:
        yt_copy, yp_copy = list(yt_raw), list(yp_raw)
        try:
            got = fn(yt_copy, yp_copy)
        except NotImplementedError:
            raise
        except ZeroDivisionError as exc:
            return CheckResult(
                "ch23_todo_03",
                False,
                "failed",
                "To-Do 23.3 — ZeroDivisionError on Degenerate Predictions!",
                f"Failed on {desc} with ZeroDivisionError: {exc}",
                details=[
                    "Guard every denominator (`tp + fn`, `tn + fp`, `tp + fp`, `tn + fn`, `prec + sens`) and return `0.0` when a denominator is 0."
                ],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch23_todo_03",
                False,
                "failed",
                "To-Do 23.3 — Runtime Error in `clinical_confusion_metrics`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch23_todo_03", "To-Do 23.3", "clinical_confusion_metrics(y_true, y_pred)", 10)

        key_err = _check_dict_keys("ch23_todo_03", "To-Do 23.3", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        yt = np.asarray(yt_raw, dtype=int)
        yp = np.asarray(yp_raw, dtype=int)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn_cnt = int(np.sum((yt == 1) & (yp == 0)))

        sens = tp / (tp + fn_cnt) if (tp + fn_cnt) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn_cnt) if (tn + fn_cnt) > 0 else 0.0
        f1 = (2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

        expected = {
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "npv": round(npv, 4),
            "f1": round(f1, 4),
        }

        for k, exp_v in expected.items():
            if not _is_np_or_py_number(got[k]) or not math.isclose(float(got[k]), exp_v, abs_tol=1e-4):
                errors.append(f"[{desc}] {k}: expected {exp_v}, got {got[k]!r}")

    if errors:
        return CheckResult(
            "ch23_todo_03",
            False,
            "failed",
            "To-Do 23.3 — Check Clinical Confusion Matrix Formulas & Zero Guards",
            "Compute sensitivity, specificity, precision (PPV), NPV, and F1 with zero-denominator guards returning `0.0`, rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch23_todo_03",
        True,
        "passed",
        "To-Do 23.3 Passed — Clinical Confusion Metrics Verified!",
        "Passed standard screening cohort plus degenerate all-zero and all-one prediction edge cases without ZeroDivisionError.",
        tip="In diagnostic medicine, Sensitivity = Recall, Precision = PPV (Positive Predictive Value), and NPV = TN / (TN + FN).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_todo_03",
        chapter="ch23",
        title="To-Do 23.3: Clinical Confusion Matrix Metrics (clinical_confusion_metrics)",
        book_section="Section 23.3.2 (Clinical Confusion Matrix & Accuracy Paradox)",
        points=10,
        hints={
            1: "Count `tp = sum((yt == 1) & (yp == 1))`, `fp = sum((yt == 0) & (yp == 1))`, `tn = sum((yt == 0) & (yp == 0))`, and `fn = sum((yt == 1) & (yp == 0))`.",
            2: "Guard every division: e.g., `prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0`, and compute `f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0`.",
            3: "def clinical_confusion_metrics(y_true, y_pred):\n    yt, yp = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)\n    tp = int(np.sum((yt == 1) & (yp == 1)))\n    fp = int(np.sum((yt == 0) & (yp == 1)))\n    tn = int(np.sum((yt == 0) & (yp == 0)))\n    fn = int(np.sum((yt == 1) & (yp == 0)))\n    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0\n    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0\n    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0\n    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0\n    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0\n    return {'sensitivity': round(sens, 4), 'specificity': round(spec, 4), 'precision': round(prec, 4), 'npv': round(npv, 4), 'f1': round(f1, 4)}",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 23.4: roc_and_pr_auc(y_true, y_prob)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    bad = _check_callable("ch23_todo_04", "To-Do 23.4", fn, "roc_and_pr_auc", 10)
    if bad is not None:
        return bad

    cases = [
        (
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0.10, 0.25, 0.40, 0.55, 0.45, 0.70, 0.85, 0.92],
            "balanced clinical predictions",
        ),
        (
            [0] * 18 + [1, 1],
            [0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.15, 0.18,
             0.20, 0.22, 0.25, 0.30, 0.35, 0.42, 0.55, 0.65, 0.50, 0.80],
            "rare-disease cohort (10% prevalence)",
        ),
    ]
    expected_keys = ["roc_auc", "pr_auc", "brier_score"]
    errors = []

    for yt_raw, yp_raw, desc in cases:
        yt_copy, yp_copy = list(yt_raw), list(yp_raw)
        try:
            got = fn(yt_copy, yp_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch23_todo_04",
                False,
                "failed",
                "To-Do 23.4 — Runtime Error in `roc_and_pr_auc`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch23_todo_04", "To-Do 23.4", "roc_and_pr_auc(y_true, y_prob)", 10)

        key_err = _check_dict_keys("ch23_todo_04", "To-Do 23.4", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        expected = {
            "roc_auc": round(float(roc_auc_score(yt_raw, yp_raw)), 4),
            "pr_auc": round(float(average_precision_score(yt_raw, yp_raw)), 4),
            "brier_score": round(float(brier_score_loss(yt_raw, yp_raw)), 4),
        }
        for k, exp_v in expected.items():
            if not _is_np_or_py_number(got[k]) or not math.isclose(float(got[k]), exp_v, abs_tol=1e-4):
                errors.append(f"[{desc}] {k}: expected {exp_v}, got {got[k]!r}")

    if errors:
        return CheckResult(
            "ch23_todo_04",
            False,
            "failed",
            "To-Do 23.4 — Check ROC-AUC, PR-AUC & Brier Score Calls",
            "Use `roc_auc_score`, `average_precision_score`, and `brier_score_loss`, rounding all 3 metrics to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch23_todo_04",
        True,
        "passed",
        "To-Do 23.4 Passed — ROC-AUC, PR-AUC & Brier Score Verified!",
        "Evaluated discrimination (ROC-AUC), imbalanced precision-recall (PR-AUC), and probability calibration (Brier score) accurately.",
        tip="Always report ROC-AUC, PR-AUC, and Brier score together: ROC-AUC ignores prevalence and calibration, while PR-AUC and Brier expose both.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_todo_04",
        chapter="ch23",
        title="To-Do 23.4: ROC-AUC, PR-AUC & Brier Score (roc_and_pr_auc)",
        book_section="Section 23.3.3 (ROC-AUC, PR-AUC & Brier Score)",
        points=10,
        hints={
            1: "Import `roc_auc_score`, `average_precision_score`, and `brier_score_loss` from `sklearn.metrics`.",
            2: "Compute each metric on `(y_true, y_prob)` and return a dict with keys `'roc_auc'`, `'pr_auc'`, and `'brier_score'` rounded to 4 decimals.",
            3: "def roc_and_pr_auc(y_true, y_prob):\n    return {\n        'roc_auc': round(float(roc_auc_score(y_true, y_prob)), 4),\n        'pr_auc': round(float(average_precision_score(y_true, y_prob)), 4),\n        'brier_score': round(float(brier_score_loss(y_true, y_prob)), 4),\n    }",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 23.5: find_youden_threshold(y_true, y_prob)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.metrics import roc_curve

    bad = _check_callable("ch23_todo_05", "To-Do 23.5", fn, "find_youden_threshold", 10)
    if bad is not None:
        return bad

    cases = [
        (
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0.05, 0.12, 0.21, 0.35, 0.62, 0.38, 0.44, 0.68, 0.79, 0.91],
            "balanced screening cohort",
        ),
        (
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0.03, 0.05, 0.08, 0.11, 0.14, 0.17, 0.28, 0.35, 0.22, 0.26, 0.41, 0.58],
            "low-prevalence cohort (optimal threshold < 0.50)",
        ),
    ]
    errors = []

    for yt_raw, yp_raw, desc in cases:
        try:
            got = fn(list(yt_raw), list(yp_raw))
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch23_todo_05",
                False,
                "failed",
                "To-Do 23.5 — Runtime Error in `find_youden_threshold`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch23_todo_05", "To-Do 23.5", "find_youden_threshold(y_true, y_prob)", 10)

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch23_todo_05",
                False,
                "failed",
                "To-Do 23.5 — Return a 2-Tuple `(best_thresh, best_j)`",
                f"[{desc}] Expected a 2-tuple `(best_thresh, best_j)`, got {type(got).__name__} ({got!r}).",
                points_earned=0,
                points_possible=10,
            )

        fpr, tpr, thresholds = roc_curve(yt_raw, yp_raw)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        exp_t = round(float(thresholds[best_idx]), 4)
        exp_j = round(float(j_scores[best_idx]), 4)

        if not (_is_np_or_py_number(got[0]) and _is_np_or_py_number(got[1])):
            errors.append(f"[{desc}] Both elements must be floats, got {got!r}.")
            continue

        if not math.isclose(float(got[0]), exp_t, abs_tol=1e-4) or not math.isclose(float(got[1]), exp_j, abs_tol=1e-4):
            errors.append(f"[{desc}] Expected ({exp_t}, {exp_j}), got {got!r}")

    if errors:
        return CheckResult(
            "ch23_todo_05",
            False,
            "failed",
            "To-Do 23.5 — Check Youden's J Threshold Selection",
            "Unpack `fpr, tpr, thresholds = roc_curve(y_true, y_prob)`, locate `idx = np.argmax(tpr - fpr)`, and return `(round(thresholds[idx], 4), round(j[idx], 4))`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch23_todo_05",
        True,
        "passed",
        "To-Do 23.5 Passed — Youden's J Optimal Threshold Verified!",
        "Maximized J = Sensitivity + Specificity - 1 = TPR - FPR cleanly on both balanced and low-prevalence cohorts.",
        tip="When disease prevalence is 14%, the optimal decision threshold sits far below 0.50 — tuning the threshold preserves calibrated probabilities!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_todo_05",
        chapter="ch23",
        title="To-Do 23.5: Youden's J Optimal Threshold (find_youden_threshold)",
        book_section="Section 23.3.5 (Decision Threshold Tuning & Youden's J)",
        points=10,
        hints={
            1: "Call `fpr, tpr, thresholds = roc_curve(y_true, y_prob)` and compute `j_scores = tpr - fpr`.",
            2: "Find `best_idx = int(np.argmax(j_scores))` and return `(round(float(thresholds[best_idx]), 4), round(float(j_scores[best_idx]), 4))`.",
            3: "def find_youden_threshold(y_true, y_prob):\n    fpr, tpr, thresholds = roc_curve(y_true, y_prob)\n    j_scores = tpr - ___\n    best_idx = int(np.argmax(j_scores))\n    return (round(float(thresholds[best_idx]), 4), round(float(j_scores[best_idx]), 4))",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 23.6: compare_balanced_logistic(X_train, y_train, X_test, y_test)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from ..datasets import load_data

    bad = _check_callable("ch23_todo_06", "To-Do 23.6", fn, "compare_balanced_logistic", 10)
    if bad is not None:
        return bad

    # Case 1: Imbalanced synthetic cohort with differing feature scales
    rng = np.random.default_rng(101)
    n_neg, n_pos = 270, 30
    X_neg = np.column_stack([
        rng.normal(50.0, 12.0, size=n_neg),
        rng.normal(2.0, 0.8, size=n_neg),
    ])
    X_pos = np.column_stack([
        rng.normal(66.0, 12.0, size=n_pos),
        rng.normal(3.1, 0.8, size=n_pos),
    ])
    X_syn = np.vstack([X_neg, X_pos])
    y_syn = np.array([0] * n_neg + [1] * n_pos, dtype=int)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_syn, y_syn, test_size=0.3, stratify=y_syn, random_state=42
    )

    # Case 2: Real MCH child_stunted (14% prevalence)
    mch = load_data("mch_survey_sample", as_pandas=True).copy()
    mch["residence"] = mch["residence"].astype(str).str.strip().str.title()
    mch["anc_visits"] = mch["anc_visits"].replace(999, np.nan)
    mch["is_urban"] = (mch["residence"] == "Urban").astype(int)
    w_map = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
    mch["wealth_num"] = mch["wealth_quintile"].map(w_map)
    clean = mch[["child_stunted", "maternal_age_years", "anc_visits", "wealth_num", "is_urban"]].dropna()
    Xm = clean[["maternal_age_years", "anc_visits", "wealth_num", "is_urban"]].to_numpy()
    ym = clean["child_stunted"].astype(int).to_numpy()
    Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(
        Xm, ym, test_size=0.25, stratify=ym, random_state=42
    )

    cases = [
        (Xs_tr, ys_tr, Xs_te, ys_te, "imbalanced synthetic cohort (10% positive)"),
        (Xm_tr, ym_tr, Xm_te, ym_te, "MCH child_stunted cohort (13.8% positive)"),
    ]
    expected_keys = ["default_recall", "balanced_recall", "default_pr_auc", "balanced_pr_auc"]
    errors = []

    for X_tr, y_tr, X_te, y_te, desc in cases:
        try:
            got = fn(X_tr.copy(), y_tr.copy(), X_te.copy(), y_te.copy())
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch23_todo_06",
                False,
                "failed",
                "To-Do 23.6 — Runtime Error in `compare_balanced_logistic`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch23_todo_06", "To-Do 23.6", "compare_balanced_logistic(X_train, y_train, X_test, y_test)", 10
            )

        key_err = _check_dict_keys("ch23_todo_06", "To-Do 23.6", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        def _ref(cw: Any) -> tuple[float, float]:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight=cw)),
            ])
            pipe.fit(X_tr, y_tr)
            yp = pipe.predict(X_te)
            yprob = pipe.predict_proba(X_te)[:, 1]
            return (
                round(float(recall_score(y_te, yp)), 4),
                round(float(average_precision_score(y_te, yprob)), 4),
            )

        def_rec, def_pr = _ref(None)
        bal_rec, bal_pr = _ref("balanced")
        expected = {
            "default_recall": def_rec,
            "balanced_recall": bal_rec,
            "default_pr_auc": def_pr,
            "balanced_pr_auc": bal_pr,
        }
        for k, exp_v in expected.items():
            if not _is_np_or_py_number(got[k]) or not math.isclose(float(got[k]), exp_v, abs_tol=1e-3):
                errors.append(f"[{desc}] {k}: expected {exp_v}, got {got[k]!r}")

    if errors:
        return CheckResult(
            "ch23_todo_06",
            False,
            "failed",
            "To-Do 23.6 — Check Scaled Pipeline & `class_weight='balanced'` Comparison",
            "Fit two `Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500, class_weight=...))])` models (None vs `'balanced'`) and evaluate test recall and PR-AUC.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch23_todo_06",
        True,
        "passed",
        "To-Do 23.6 Passed — Balanced vs. Default Logistic Regression Verified!",
        "Demonstrated how `class_weight='balanced'` lifts minority-class recall dramatically (0.0000 -> 0.8824 on MCH stunting) while PR-AUC stays nearly unchanged.",
        tip="Class weighting shifts the effective decision boundary, NOT the underlying ranking — always recalibrate probabilities if you deploy a reweighted classifier!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_todo_06",
        chapter="ch23",
        title="To-Do 23.6: Default vs. Balanced Logistic Regression (compare_balanced_logistic)",
        book_section="Section 23.3.6 (Handling Class Imbalance Safely)",
        points=10,
        hints={
            1: "Wrap `StandardScaler()` and `LogisticRegression(max_iter=500, class_weight=cw)` in a `Pipeline` so scaling is fitted strictly on `X_train`.",
            2: "Run once with `class_weight=None` and once with `class_weight='balanced'`. Evaluate `recall_score(y_test, pipe.predict(X_test))` and `average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1])`.",
            3: "def compare_balanced_logistic(X_train, y_train, X_test, y_test):\n    out = {}\n    for prefix, cw in [('default', None), ('balanced', '___')]:\n        pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500, class_weight=cw))])\n        pipe.fit(X_train, y_train)\n        out[f'{prefix}_recall'] = round(float(recall_score(y_test, pipe.predict(X_test))), 4)\n        out[f'{prefix}_pr_auc'] = round(float(average_precision_score(y_test, pipe.predict_proba(X_test)[:, 1])), 4)\n    return out",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 23.1 (30 pts): evaluate_screening_thresholds(y_true, y_prob, thresholds)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    bad = _check_callable("ch23_exam_01", "Exam Task 23.1", fn, "evaluate_screening_thresholds", 30)
    if bad is not None:
        return bad

    cases = [
        (
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0.05, 0.15, 0.25, 0.40, 0.55, 0.72, 0.28, 0.48, 0.65, 0.88],
            [0.1, 0.3, 0.5, 0.7, 0.95],
            "5-threshold sweep with zero-flagged upper bound",
        ),
        (
            [0, 0, 0, 0, 1, 1],
            [0.10, 0.30, 0.60, 0.80, 0.35, 0.90],
            [0.2, 0.5, 0.85],
            "3-threshold clinical sweep",
        ),
    ]
    expected_cols = ["threshold", "sensitivity", "specificity", "precision", "f1", "flagged_pct"]
    errors = []

    for yt_raw, yp_raw, thresh_list, desc in cases:
        try:
            got = fn(list(yt_raw), list(yp_raw), list(thresh_list))
        except NotImplementedError:
            raise
        except ZeroDivisionError as exc:
            return CheckResult(
                "ch23_exam_01",
                False,
                "failed",
                "Exam Task 23.1 — ZeroDivisionError at Extreme Threshold",
                f"[{desc}] Raised ZeroDivisionError: {exc}",
                details=["Guard `precision` and `f1` with `0.0` when no samples are flagged positive at high thresholds."],
                points_earned=0,
                points_possible=30,
            )
        except Exception as exc:
            return CheckResult(
                "ch23_exam_01",
                False,
                "failed",
                "Exam Task 23.1 — Runtime Error in `evaluate_screening_thresholds`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch23_exam_01", "Exam Task 23.1", "evaluate_screening_thresholds(y_true, y_prob, thresholds)", 30
            )

        if not isinstance(got, pd.DataFrame):
            return CheckResult(
                "ch23_exam_01",
                False,
                "failed",
                "Exam Task 23.1 — Return a Pandas DataFrame",
                f"[{desc}] Expected a pd.DataFrame, got {type(got).__name__}.",
                points_earned=0,
                points_possible=30,
            )

        if list(got.columns) != expected_cols:
            errors.append(f"[{desc}] Column mismatch: expected {expected_cols}, got {list(got.columns)}.")
            continue

        if len(got) != len(thresh_list):
            errors.append(f"[{desc}] Row count mismatch: expected {len(thresh_list)}, got {len(got)}.")
            continue

        yt = np.asarray(yt_raw, dtype=int)
        yprob = np.asarray(yp_raw, dtype=float)
        got_reset = got.reset_index(drop=True)

        for i, t in enumerate(thresh_list):
            yp = (yprob >= t).astype(int)
            tp = int(np.sum((yt == 1) & (yp == 1)))
            fp = int(np.sum((yt == 0) & (yp == 1)))
            tn = int(np.sum((yt == 0) & (yp == 0)))
            fn_cnt = int(np.sum((yt == 1) & (yp == 0)))
            sens = tp / (tp + fn_cnt) if (tp + fn_cnt) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
            flagged_pct = 100.0 * float(np.mean(yp))

            exp_row = {
                "threshold": round(float(t), 4),
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
                "flagged_pct": round(flagged_pct, 4),
            }
            for col, exp_v in exp_row.items():
                g_val = got_reset.loc[i, col]
                if not _is_np_or_py_number(g_val) or not math.isclose(float(g_val), exp_v, abs_tol=1e-4):
                    extra = ""
                    if col == "flagged_pct" and _is_np_or_py_number(g_val) and math.isclose(float(g_val) * 100.0, exp_v, abs_tol=1e-2):
                        extra = " (Looks like a proportion in [0, 1] — multiply by 100.0 for `flagged_pct`!)"
                    errors.append(f"[{desc}] row {i} (t={t}) `{col}`: expected {exp_v}, got {g_val!r}{extra}")

    if errors:
        return CheckResult(
            "ch23_exam_01",
            False,
            "failed",
            "Exam Task 23.1 — Check Threshold Sweep DataFrame & Percentage Scaling",
            "Classify `y_pred = (y_prob >= t).astype(int)`, guard zero denominators with `0.0`, scale `flagged_pct = 100.0 * mean(y_pred)`, and round to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch23_exam_01",
        True,
        "passed",
        "Exam Task 23.1 Passed (30/30 pts) — Clinical Threshold Operating Table Verified!",
        "Generated exact sensitivity, specificity, precision, F1, and flagged-patient percentages across all thresholds including zero-flag upper bounds.",
        points_earned=30,
        points_possible=30,
        tip="Clinicians love a threshold operating table: it turns an abstract probability model into concrete clinic workload (`flagged_pct`) vs. case-capture (`sensitivity`).",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_exam_01",
        chapter="ch23",
        title="Exam 23.1: Clinical Threshold Operating Table (evaluate_screening_thresholds)",
        book_section="Chapter 23 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Loop over `t` in `thresholds`, threshold with `y_pred = (y_prob >= t).astype(int)`, compute sensitivity, specificity, precision, F1 (guarding zero denominators with `0.0`), and `flagged_pct = 100.0 * float(np.mean(y_pred))`.",
            2: "Collect one dict per threshold and build `pd.DataFrame(rows, columns=['threshold', 'sensitivity', 'specificity', 'precision', 'f1', 'flagged_pct'])` with all values rounded to 4 decimals.",
            3: "def evaluate_screening_thresholds(y_true, y_prob, thresholds):\n    yt, yprob = np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)\n    rows = []\n    for t in thresholds:\n        yp = (yprob >= t).astype(int)\n        # compute tp, fp, tn, fn and guarded sens, spec, prec, f1\n        rows.append({'threshold': round(float(t), 4), 'sensitivity': ..., 'specificity': ..., 'precision': ..., 'f1': ..., 'flagged_pct': round(100.0 * float(np.mean(yp)), 4)})\n    return pd.DataFrame(rows, columns=['threshold', 'sensitivity', 'specificity', 'precision', 'f1', 'flagged_pct'])",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 23.2 (35 pts): calibration_audit_bins(y_true, y_prob, n_bins=5)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import brier_score_loss

    bad = _check_callable("ch23_exam_02", "Exam Task 23.2", fn, "calibration_audit_bins", 35)
    if bad is not None:
        return bad

    cases = [
        (
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
            [0.0, 0.08, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 1.0],
            {},
            "5-bin calibration audit including 0.0 and 1.0 boundaries",
        ),
        (
            [0, 0, 0, 1, 1, 1, 0, 1],
            [0.05, 0.10, 0.12, 0.18, 0.82, 0.88, 0.91, 0.95],
            {},
            "overconfident cohort with empty middle bins",
        ),
        (
            [0, 0, 1, 0, 1, 1, 0, 1],
            [0.10, 0.20, 0.30, 0.45, 0.60, 0.70, 0.80, 0.95],
            {"n_bins": 4},
            "custom n_bins=4 calibration audit",
        ),
    ]
    expected_keys = ["ece", "brier_score", "bins"]
    expected_cols = ["mean_pred", "obs_rate", "count"]
    errors = []

    for yt_raw, yp_raw, kwargs, desc in cases:
        n_bins = kwargs.get("n_bins", 5)
        try:
            got = fn(list(yt_raw), list(yp_raw), **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch23_exam_02",
                False,
                "failed",
                "Exam Task 23.2 — Runtime Error in `calibration_audit_bins`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap("ch23_exam_02", "Exam Task 23.2", "calibration_audit_bins(y_true, y_prob)", 35)

        key_err = _check_dict_keys("ch23_exam_02", "Exam Task 23.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        yt = np.asarray(yt_raw, dtype=float)
        yp = np.asarray(yp_raw, dtype=float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        exp_rows = []
        exp_ece = 0.0
        for idx in range(n_bins):
            low, high = edges[idx], edges[idx + 1]
            mask = (yp >= low) & (yp < high) if idx < n_bins - 1 else (yp >= low) & (yp <= high)
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            mp = float(np.mean(yp[mask]))
            obs = float(np.mean(yt[mask]))
            exp_ece += (cnt / len(yt)) * abs(obs - mp)
            exp_rows.append({"mean_pred": round(mp, 4), "obs_rate": round(obs, 4), "count": cnt})

        exp_ece = round(exp_ece, 4)
        exp_brier = round(float(brier_score_loss(yt, yp)), 4)

        if not _is_np_or_py_number(got["ece"]) or not math.isclose(float(got["ece"]), exp_ece, abs_tol=1e-4):
            errors.append(f"[{desc}] ece: expected {exp_ece}, got {got['ece']!r}")
        if not _is_np_or_py_number(got["brier_score"]) or not math.isclose(float(got["brier_score"]), exp_brier, abs_tol=1e-4):
            errors.append(f"[{desc}] brier_score: expected {exp_brier}, got {got['brier_score']!r}")

        bins_df = got["bins"]
        if not isinstance(bins_df, pd.DataFrame):
            errors.append(f"[{desc}] `bins` must be a pd.DataFrame, got {type(bins_df).__name__}.")
            continue

        if list(bins_df.columns) != expected_cols:
            errors.append(f"[{desc}] `bins` columns mismatch: expected {expected_cols}, got {list(bins_df.columns)}.")
            continue

        if len(bins_df) != len(exp_rows):
            errors.append(
                f"[{desc}] Expected {len(exp_rows)} non-empty bins, got {len(bins_df)} "
                "(Skip empty bins and include `p == 1.0` in the final bin `[low, 1.0]`!)."
            )
            continue

        bins_reset = bins_df.reset_index(drop=True)
        total_cnt = int(bins_reset["count"].sum())
        if total_cnt != len(yt):
            errors.append(
                f"[{desc}] Total bin counts sum to {total_cnt} instead of {len(yt)} "
                "(Make sure the final bin uses `<= 1.0` so `p == 1.0` is not dropped!)."
            )

        for i, erow in enumerate(exp_rows):
            for col in ("mean_pred", "obs_rate"):
                gv = bins_reset.loc[i, col]
                if not _is_np_or_py_number(gv) or not math.isclose(float(gv), erow[col], abs_tol=1e-4):
                    errors.append(f"[{desc}] bin {i} `{col}`: expected {erow[col]}, got {gv!r}")
            gcnt = bins_reset.loc[i, "count"]
            if not _is_np_or_py_number(gcnt) or int(gcnt) != erow["count"]:
                errors.append(f"[{desc}] bin {i} `count`: expected {erow['count']}, got {gcnt!r}")

    if errors:
        return CheckResult(
            "ch23_exam_02",
            False,
            "failed",
            "Exam Task 23.2 — Check Equal-Width Calibration Bins, ECE & Brier Score",
            "Partition `[0, 1]` with `np.linspace(0, 1, n_bins + 1)`, use `[low, high)` for earlier bins and `[low, high]` for the last bin, skip empty bins, and compute weighted `ECE = sum((n_b / N) * |obs_b - pred_b|)`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch23_exam_02",
        True,
        "passed",
        "Exam Task 23.2 Passed (35/35 pts) — Calibration Bin Audit & ECE Verified!",
        "Handled boundary probabilities (0.0 and 1.0), skipped empty bins cleanly, and matched exact ECE and Brier score across custom `n_bins`.",
        points_earned=35,
        points_possible=35,
        tip="Expected Calibration Error (ECE) summarizes the reliability curve in one intuitive percentage-point gap between predicted risk and observed outcomes.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_exam_02",
        chapter="ch23",
        title="Exam 23.2: Probability Calibration Audit & ECE (calibration_audit_bins)",
        book_section="Chapter 23 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Create edges `edges = np.linspace(0.0, 1.0, n_bins + 1)`. For bin `i < n_bins - 1`, mask `(yp >= low) & (yp < high)`; for the final bin, use `(yp >= low) & (yp <= high)` so `p == 1.0` is included.",
            2: "Skip bins with `count == 0`. For each non-empty bin, accumulate `ece += (count / len(yt)) * abs(obs_rate - mean_pred)` and store `{'mean_pred': round(mean_pred, 4), 'obs_rate': round(obs_rate, 4), 'count': count}`.",
            3: "def calibration_audit_bins(y_true, y_prob, n_bins=5):\n    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_prob, dtype=float)\n    edges = np.linspace(0.0, 1.0, n_bins + 1)\n    rows, ece = [], 0.0\n    for idx in range(n_bins):\n        low, high = edges[idx], edges[idx + 1]\n        mask = (yp >= low) & (yp < high) if idx < n_bins - 1 else (yp >= low) & (yp <= high)\n        cnt = int(np.sum(mask))\n        if cnt == 0:\n            continue\n        mp, obs = float(np.mean(yp[mask])), float(np.mean(yt[mask]))\n        ece += (cnt / len(yt)) * abs(obs - mp)\n        rows.append({'mean_pred': round(mp, 4), 'obs_rate': round(obs, 4), 'count': cnt})\n    return {'ece': round(ece, 4), 'brier_score': round(float(brier_score_loss(yt, yp)), 4), 'bins': pd.DataFrame(rows, columns=['mean_pred', 'obs_rate', 'count'])}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 23.3 (35 pts): train_calibrated_anemia_classifier(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _ref_calibrated_anemia(mch_df: Any, seed: int = 42) -> dict[str, float]:
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    work = mch_df.copy()
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    work["anc_visits"] = work["anc_visits"].replace(999, np.nan)
    work["is_urban"] = (work["residence"] == "Urban").astype(int)
    wealth_map = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
    work["wealth_num"] = work["wealth_quintile"].map(wealth_map)
    feat_cols = ["maternal_age_years", "anc_visits", "wealth_num", "is_urban"]
    clean = work[["maternal_anemia"] + feat_cols].dropna()
    X = clean[feat_cols]
    y = clean["maternal_anemia"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500)),
    ])
    pipe.fit(X_tr, y_tr)
    prob_raw = pipe.predict_proba(X_te)[:, 1]

    cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
    cal.fit(X_tr, y_tr)
    prob_cal = cal.predict_proba(X_te)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_te, prob_cal)
    best_idx = int(np.argmax(tpr - fpr))
    return {
        "roc_auc": round(float(roc_auc_score(y_te, prob_cal)), 4),
        "pr_auc": round(float(average_precision_score(y_te, prob_cal)), 4),
        "brier_raw": round(float(brier_score_loss(y_te, prob_raw)), 4),
        "brier_calibrated": round(float(brier_score_loss(y_te, prob_cal)), 4),
        "youden_threshold": round(float(thresholds[best_idx]), 4),
    }


def _verify_exam_03(fn: Any) -> CheckResult:
    from ..datasets import load_data

    bad = _check_callable("ch23_exam_03", "Exam Task 23.3", fn, "train_calibrated_anemia_classifier", 35)
    if bad is not None:
        return bad

    df_mch = load_data("mch_survey_sample", as_pandas=True)
    expected_keys = ["roc_auc", "pr_auc", "brier_raw", "brier_calibrated", "youden_threshold"]
    errors = []

    for seed in (42, 7):
        df_copy = df_mch.copy(deep=True)
        desc = f"mch_survey_sample (seed={seed})"
        try:
            got = fn(df_copy, seed=seed)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch23_exam_03",
                False,
                "failed",
                "Exam Task 23.3 — Check Function Signature `train_calibrated_anemia_classifier(mch_df, seed=42)`",
                f"Failed on {desc} with TypeError: {exc}",
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch23_exam_03",
                False,
                "failed",
                "Exam Task 23.3 — Runtime Error in `train_calibrated_anemia_classifier`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch23_exam_03", "Exam Task 23.3", f"train_calibrated_anemia_classifier(mch_df, seed={seed})", 35
            )

        key_err = _check_dict_keys("ch23_exam_03", "Exam Task 23.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not df_copy.equals(df_mch):
            errors.append(f"[{desc}] Input DataFrame was mutated! Start with `work = mch_df.copy()`.")

        exp = _ref_calibrated_anemia(df_mch, seed=seed)
        for k, exp_v in exp.items():
            gv = got[k]
            if not _is_np_or_py_number(gv) or not math.isclose(float(gv), exp_v, abs_tol=1e-3):
                extra = ""
                if k == "roc_auc" and _is_np_or_py_number(gv) and float(gv) > 0.90:
                    extra = " (ROC-AUC > 0.90 suggests target leakage — do NOT include `maternal_hb_g_dl` as a feature!)"
                errors.append(f"[{desc}] {k}: expected {exp_v}, got {gv!r}{extra}")

    if errors:
        return CheckResult(
            "ch23_exam_03",
            False,
            "failed",
            "Exam Task 23.3 — Check Cleaning, Stratified 75/25 Split & Sigmoid Calibration",
            "Clean `residence`, replace `anc_visits == 999` with `np.nan`, encode `wealth_num` (1..5) and `is_urban`, fit `Pipeline([StandardScaler, LogisticRegression(max_iter=500)])` and `CalibratedClassifierCV(pipe, method='sigmoid', cv=5)` on stratified 75/25 split.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch23_exam_03",
        True,
        "passed",
        "Exam Task 23.3 Passed (35/35 pts) — Calibrated Clinical Anemia Classifier Verified!",
        "Built a leak-free scaled + Platt-calibrated logistic classifier on `mch_survey_sample`, verified across multiple random seeds (seed=42 ROC-AUC=0.6275, Youden threshold=0.2968).",
        points_earned=35,
        points_possible=35,
        tip="Excluding `maternal_hb_g_dl` prevents target leakage while `CalibratedClassifierCV(..., cv=5)` yields calibrated screening risk probabilities for threshold selection.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch23_exam_03",
        chapter="ch23",
        title="Exam 23.3: Calibrated Clinical Anemia Classifier (train_calibrated_anemia_classifier)",
        book_section="Chapter 23 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Copy `mch_df`, clean `residence` casing, replace `anc_visits == 999` with `np.nan`, encode `is_urban` (0/1) and `wealth_num` (`Poorest`->1 .. `Richest`->5), and `.dropna()` on `['maternal_anemia', 'maternal_age_years', 'anc_visits', 'wealth_num', 'is_urban']`.",
            2: "Split stratified 75/25 (`test_size=0.25, stratify=y, random_state=seed`). Fit `pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])` for `prob_raw` and `CalibratedClassifierCV(pipe, method='sigmoid', cv=5)` for `prob_cal`.",
            3: "def train_calibrated_anemia_classifier(mch_df, seed=42):\n    # clean work = mch_df.copy(), build X (4 features) and y\n    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)\n    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=500))])\n    pipe.fit(X_tr, y_tr)\n    prob_raw = pipe.predict_proba(X_te)[:, 1]\n    cal = CalibratedClassifierCV(pipe, method='sigmoid', cv=___)\n    cal.fit(X_tr, y_tr)\n    prob_cal = cal.predict_proba(X_te)[:, 1]\n    fpr, tpr, thresholds = roc_curve(y_te, prob_cal)\n    best_idx = int(np.argmax(tpr - fpr))\n    return {'roc_auc': round(float(roc_auc_score(y_te, prob_cal)), 4), 'pr_auc': round(float(average_precision_score(y_te, prob_cal)), 4), 'brier_raw': round(float(brier_score_loss(y_te, prob_raw)), 4), 'brier_calibrated': round(float(brier_score_loss(y_te, prob_cal)), 4), 'youden_threshold': round(float(thresholds[best_idx]), 4)}",
        },
        verifier=_verify_exam_03,
    )
)
