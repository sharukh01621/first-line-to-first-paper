"""Chapter 22 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch22_todo_01..06) and the Coding Exam (ch22_exam_01..03)
for Linear & Regularized Regression (OLS, Ridge, Lasso, ElasticNet) — From
Scratch and Scikit-Learn.
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn inside the check
function, validates return structures, rounding, leak-free Pipeline usage,
and non-mutation, while letting NotImplementedError propagate cleanly for
starter stubs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_number(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, float, np.number)) and not isinstance(
        val, (bool, np.bool_)
    )


def _is_int_like(val: Any) -> bool:
    import numpy as np

    return isinstance(val, (int, np.integer)) and not isinstance(
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
        tip="Reusable ML evaluation and modeling helpers always return structured dicts, tuples, or DataFrames.",
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
# GUIDED TO-DO 22.1: regression_metrics(y_true, y_pred)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch22_todo_01", "To-Do 22.1", fn, "regression_metrics", 10)
    if bad is not None:
        return bad

    cases = [
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], "perfect predictions"),
        (
            [-1.5, -0.8, 0.2, -2.1, -0.4, 0.9],
            [-1.2, -0.9, 0.0, -1.7, -0.5, 0.5],
            "realistic child HAZ predictions",
        ),
        ([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], "worse-than-mean (negative R2)"),
    ]
    expected_keys = ["mae", "rmse", "r2"]
    errors = []

    for yt_raw, yp_raw, desc in cases:
        yt_copy, yp_copy = list(yt_raw), list(yp_raw)
        try:
            got = fn(yt_copy, yp_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch22_todo_01",
                False,
                "failed",
                "To-Do 22.1 — Runtime Error in `regression_metrics`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch22_todo_01", "To-Do 22.1", "regression_metrics(y_true, y_pred)", 10
            )

        key_err = _check_dict_keys("ch22_todo_01", "To-Do 22.1", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        if yt_copy != yt_raw or yp_copy != yp_raw:
            errors.append(f"[{desc}] Mutated input sequence(s)!")

        yt = np.asarray(yt_raw, dtype=float)
        yp = np.asarray(yp_raw, dtype=float)
        err = yt - yp
        mse = float(np.mean(err ** 2))
        exp_mae = round(float(np.mean(np.abs(err))), 4)
        exp_rmse = round(math.sqrt(mse), 4)
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        exp_r2 = round(1.0 - ss_res / ss_tot, 4)

        if not _is_number(got["mae"]) or not math.isclose(float(got["mae"]), exp_mae, abs_tol=1e-4):
            errors.append(f"[{desc}] mae: expected {exp_mae}, got {got['mae']!r}")

        if not _is_number(got["rmse"]) or not math.isclose(float(got["rmse"]), exp_rmse, abs_tol=1e-4):
            extra = ""
            if _is_number(got["rmse"]) and math.isclose(float(got["rmse"]), round(mse, 4), abs_tol=1e-3):
                extra = " (Looks like MSE — take the square root for RMSE!)"
            errors.append(f"[{desc}] rmse: expected {exp_rmse}, got {got['rmse']!r}{extra}")

        if not _is_number(got["r2"]) or not math.isclose(float(got["r2"]), exp_r2, abs_tol=1e-4):
            extra = ""
            if _is_number(got["r2"]) and math.isclose(float(got["r2"]), round(ss_res / ss_tot, 4), abs_tol=1e-3):
                extra = " (Forgot `1 - ss_res / ss_tot`!)"
            errors.append(f"[{desc}] r2: expected {exp_r2}, got {got['r2']!r}{extra}")

    if errors:
        return CheckResult(
            "ch22_todo_01",
            False,
            "failed",
            "To-Do 22.1 — Check MAE, RMSE, and R^2 Formulas & 4-Decimal Rounding",
            "Compute MAE = mean(|y - y_hat|), RMSE = sqrt(mean((y - y_hat)^2)), and R^2 = 1 - SS_res / SS_tot rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch22_todo_01",
        True,
        "passed",
        "To-Do 22.1 Passed — Regression Metrics (MAE, RMSE, R^2) Verified!",
        "Passed perfect-fit (R^2=1.0), realistic child HAZ error, and worse-than-mean (negative R^2) checks.",
        tip="Remember: out-of-sample R^2 can be negative whenever predictions are noisier than simply predicting the sample mean!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_todo_01",
        chapter="ch22",
        title="To-Do 22.1: Core Regression Metrics (regression_metrics)",
        book_section="Section 22.3.1 (Regression Loss & Evaluation Metrics)",
        points=10,
        hints={
            1: "Convert both inputs via `np.asarray(..., dtype=float)` and compute `err = y_true - y_pred` (or use `mean_absolute_error`, `root_mean_squared_error`, `r2_score` from `sklearn.metrics`).",
            2: "Compute `mae = np.mean(np.abs(err))`, `rmse = np.sqrt(np.mean(err ** 2))`, and `r2 = 1.0 - np.sum(err ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)`, rounding each float to 4 decimals.",
            3: "def regression_metrics(y_true, y_pred):\n    yt = np.asarray(y_true, dtype=float)\n    yp = np.asarray(y_pred, dtype=float)\n    err = yt - yp\n    mae = float(np.mean(np.abs(err)))\n    rmse = float(np.sqrt(np.mean(err ** 2)))\n    r2 = float(1.0 - np.sum(err ** 2) / np.sum((yt - np.mean(yt)) ** 2))\n    return {'mae': round(mae, 4), 'rmse': round(rmse, 4), 'r2': round(___, 4)}",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 22.2: ols_normal_equation(X, y)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch22_todo_02", "To-Do 22.2", fn, "ols_normal_equation", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(42)
    X_1d_feat = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_1d_feat = 2.5 + 3.0 * X_1d_feat[:, 0]

    X_multi = rng.normal(size=(60, 3))
    y_multi = -1.2 + X_multi @ np.array([1.8, -0.75, 0.4]) + rng.normal(0, 0.15, size=60)

    x_base = rng.normal(size=(40, 1))
    X_col = np.hstack([x_base, 2.0 * x_base])
    y_col = 0.5 + 1.5 * x_base[:, 0] + rng.normal(0, 0.05, size=40)

    cases = [
        (X_1d_feat, y_1d_feat, "exact 1-feature line (y = 2.5 + 3x)"),
        (X_multi, y_multi, "3-feature noisy linear model"),
        (X_col, y_col, "rank-deficient perfectly collinear X"),
    ]
    errors = []

    for X_in, y_in, desc in cases:
        X_copy = X_in.copy()
        y_copy = y_in.copy()
        try:
            got = fn(X_copy, y_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch22_todo_02",
                False,
                "failed",
                "To-Do 22.2 — Runtime Error in `ols_normal_equation`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=[
                    "Use `np.linalg.lstsq(Xb, y, rcond=None)[0]` or `np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y` instead of `np.linalg.inv(...)` so collinear matrices never crash with LinAlgError!"
                ],
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch22_todo_02", "To-Do 22.2", "ols_normal_equation(X, y)", 10
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch22_todo_02",
                False,
                "failed",
                "To-Do 22.2 — Return a 2-Tuple `(intercept, coef)`",
                f"[{desc}] Expected a 2-tuple `(float_intercept, 1D_coef_array)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(X_copy, X_in) or not np.array_equal(y_copy, y_in):
            errors.append(f"[{desc}] Mutated input X or y arrays!")

        got_b, got_w = got
        if not _is_number(got_b):
            errors.append(f"[{desc}] Intercept must be a float, got {type(got_b).__name__}.")
            continue

        got_w_arr = np.asarray(got_w, dtype=float)
        if got_w_arr.ndim != 1 or got_w_arr.shape != (X_in.shape[1],):
            errors.append(
                f"[{desc}] coef must be a 1D array of shape ({X_in.shape[1]},), got shape {got_w_arr.shape}."
            )
            continue

        Xb = np.column_stack([np.ones(len(X_in)), X_in])
        theta, _, _, _ = np.linalg.lstsq(Xb, y_in, rcond=None)
        exp_b, exp_w = float(theta[0]), theta[1:]

        # On rank-deficient inputs, verify both intercept and exact fitted predictions
        y_hat_got = float(got_b) + X_in @ got_w_arr
        y_hat_exp = exp_b + X_in @ exp_w
        if "collinear" in desc:
            if not np.allclose(y_hat_got, y_hat_exp, atol=1e-4) or not math.isclose(
                float(got_b), exp_b, abs_tol=1e-4
            ):
                errors.append(
                    f"[{desc}] Fitted predictions/intercept did not match minimum-norm OLS solution."
                )
        else:
            if not math.isclose(float(got_b), exp_b, abs_tol=1e-4):
                errors.append(f"[{desc}] intercept: expected {exp_b:.4f}, got {float(got_b):.4f}")
            if not np.allclose(got_w_arr, exp_w, atol=1e-4):
                errors.append(f"[{desc}] coef: expected {np.round(exp_w, 4)}, got {np.round(got_w_arr, 4)}")

    if errors:
        return CheckResult(
            "ch22_todo_02",
            False,
            "failed",
            "To-Do 22.2 — Check Intercept Column & Normal Equation Solver",
            "Prepend a column of 1s (`np.column_stack([np.ones(len(X)), X])`), solve with `np.linalg.lstsq(Xb, y, rcond=None)[0]` or `np.linalg.pinv`, and return `(float(theta[0]), theta[1:])`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch22_todo_02",
        True,
        "passed",
        "To-Do 22.2 Passed — Pure-NumPy OLS Normal Equation Verified!",
        "Matched exact OLS intercept and slope vector across 1D, 3D noisy, and perfectly collinear rank-deficient matrices without LinAlgError.",
        tip="Using `np.linalg.lstsq` (SVD-based) or `np.linalg.pinv` handles multicollinearity gracefully where naive `np.linalg.inv(X.T @ X)` crashes.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_todo_02",
        chapter="ch22",
        title="To-Do 22.2: Pure-NumPy OLS Normal Equation (ols_normal_equation)",
        book_section="Section 22.3.2 (Closed-Form Normal Equation)",
        points=10,
        hints={
            1: "Convert `X` and `y` to float NumPy arrays and prepend an intercept column of ones: `Xb = np.column_stack([np.ones(len(X_arr)), X_arr])`.",
            2: "Solve for `theta` safely via `theta = np.linalg.lstsq(Xb, y_arr, rcond=None)[0]` (or `np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y_arr`), then return `(float(theta[0]), theta[1:])`.",
            3: "def ols_normal_equation(X, y):\n    X_arr = np.asarray(X, dtype=float)\n    y_arr = np.asarray(y, dtype=float)\n    Xb = np.column_stack([np.ones(len(X_arr)), X_arr])\n    theta = np.linalg.lstsq(Xb, y_arr, rcond=None)[0]\n    return (float(theta[0]), theta[1:])",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 22.3: gradient_descent_linear(X, y, lr=0.05, epochs=300)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch22_todo_03", "To-Do 22.3", fn, "gradient_descent_linear", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(123)
    X_syn = rng.normal(size=(80, 2))
    X_syn = (X_syn - X_syn.mean(axis=0)) / X_syn.std(axis=0)
    y_syn = 1.75 + 2.4 * X_syn[:, 0] - 1.1 * X_syn[:, 1] + rng.normal(0, 0.2, size=80)

    Xb = np.column_stack([np.ones(len(X_syn)), X_syn])
    theta_ols = np.linalg.lstsq(Xb, y_syn, rcond=None)[0]
    exp_b, exp_w = float(theta_ols[0]), theta_ols[1:]
    exp_mse = float(np.mean((X_syn @ exp_w + exp_b - y_syn) ** 2))

    cases = [
        ((X_syn, y_syn), {}, 300, "default lr=0.05, epochs=300"),
        ((X_syn, y_syn), {"lr": 0.03, "epochs": 400}, 400, "custom lr=0.03, epochs=400"),
    ]
    errors = []

    for (X_in, y_in), kwargs, exp_epochs, desc in cases:
        X_copy, y_copy = X_in.copy(), y_in.copy()
        try:
            got = fn(X_copy, y_copy, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch22_todo_03",
                False,
                "failed",
                "To-Do 22.3 — Check Function Signature `gradient_descent_linear(X, y, lr=0.05, epochs=300)`",
                f"Call with {desc} raised TypeError: {exc}",
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch22_todo_03",
                False,
                "failed",
                "To-Do 22.3 — Runtime Error in `gradient_descent_linear`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch22_todo_03", "To-Do 22.3", "gradient_descent_linear(X, y)", 10
            )

        if not isinstance(got, tuple) or len(got) != 3:
            return CheckResult(
                "ch22_todo_03",
                False,
                "failed",
                "To-Do 22.3 — Return a 3-Tuple `(w, b, loss_history)`",
                f"[{desc}] Expected `(w_1d, b_float, loss_history_list)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(X_copy, X_in) or not np.array_equal(y_copy, y_in):
            errors.append(f"[{desc}] Mutated input X or y arrays!")

        w_got, b_got, hist = got
        w_arr = np.asarray(w_got, dtype=float)
        if w_arr.ndim != 1 or w_arr.shape != (X_in.shape[1],):
            errors.append(f"[{desc}] w must be a 1D array of shape ({X_in.shape[1]},), got {w_arr.shape}.")
            continue
        if not _is_number(b_got):
            errors.append(f"[{desc}] b must be a float, got {type(b_got).__name__}.")
            continue
        if not isinstance(hist, list):
            errors.append(f"[{desc}] loss_history must be a Python list, got {type(hist).__name__}.")
            continue
        if len(hist) != exp_epochs:
            errors.append(f"[{desc}] len(loss_history) expected {exp_epochs}, got {len(hist)}.")
            continue
        if not all(_is_number(v) and math.isfinite(float(v)) for v in hist):
            errors.append(f"[{desc}] loss_history contains non-finite values (check learning rate / gradient sign!).")
            continue
        if float(hist[-1]) >= float(hist[0]):
            errors.append(
                f"[{desc}] Loss did not decrease: loss_history[0]={hist[0]:.4f}, loss_history[-1]={hist[-1]:.4f}. "
                "Check gradient sign (`w -= lr * dw`, where `err = y_hat - y`)."
            )
            continue
        if not np.allclose(w_arr, exp_w, atol=1e-2) or not math.isclose(float(b_got), exp_b, abs_tol=1e-2):
            errors.append(
                f"[{desc}] Converged parameters (w={np.round(w_arr, 3)}, b={float(b_got):.3f}) "
                f"did not match OLS (w={np.round(exp_w, 3)}, b={exp_b:.3f})."
            )
        if not math.isclose(float(hist[-1]), exp_mse, abs_tol=0.05):
            errors.append(
                f"[{desc}] Final MSE loss_history[-1]={float(hist[-1]):.4f} does not match expected MSE ~{exp_mse:.4f}."
            )

    if errors:
        return CheckResult(
            "ch22_todo_03",
            False,
            "failed",
            "To-Do 22.3 — Check MSE Gradients, Parameter Updates & Loss History",
            "Initialize `w = np.zeros(p)`, `b = 0.0`, and in each epoch record `mse = mean((y_hat - y)^2)`, compute `dw = (2/n) * X.T @ err`, `db = (2/n) * sum(err)`, and subtract `lr * grad`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch22_todo_03",
        True,
        "passed",
        "To-Do 22.3 Passed — Pure-NumPy Batch Gradient Descent Verified!",
        "Confirmed monotonic MSE decrease, exact epoch-length history, and convergence to the closed-form OLS weights and intercept.",
        tip="Batch gradient descent scales linearly with features when n and p grow too large for direct Normal Equation inversion.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_todo_03",
        chapter="ch22",
        title="To-Do 22.3: Batch Gradient Descent for Linear Regression (gradient_descent_linear)",
        book_section="Section 22.3.3 (Iterative Optimization via Gradient Descent)",
        points=10,
        hints={
            1: "Initialize `w = np.zeros(X_arr.shape[1])`, `b = 0.0`, and `loss_history = []`. Loop `for _ in range(epochs):` computing `err = X_arr @ w + b - y_arr`.",
            2: "Append `float(np.mean(err ** 2))` to `loss_history`, compute `dw = (2.0 / n) * (X_arr.T @ err)` and `db = (2.0 / n) * float(np.sum(err))`, then update `w = w - lr * dw` and `b = b - lr * db`.",
            3: "def gradient_descent_linear(X, y, lr=0.05, epochs=300):\n    X_arr = np.asarray(X, dtype=float)\n    y_arr = np.asarray(y, dtype=float)\n    n, p = X_arr.shape\n    w = np.zeros(p, dtype=float)\n    b = 0.0\n    loss_history = []\n    for _ in range(epochs):\n        err = (X_arr @ w + b) - y_arr\n        loss_history.append(float(np.mean(err ** 2)))\n        w = w - lr * ((2.0 / n) * (X_arr.T @ err))\n        b = b - lr * ((2.0 / n) * float(np.sum(err)))\n    return (w, float(b), loss_history)",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 22.4: fit_ridge_standardized(X_train, y_train, X_test, alpha=1.0)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable("ch22_todo_04", "To-Do 22.4", fn, "fit_ridge_standardized", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(7)
    X_tr = np.column_stack(
        [
            rng.normal(0.0, 1.0, size=80),
            rng.normal(500.0, 200.0, size=80),
            rng.normal(-20.0, 5.0, size=80),
        ]
    )
    y_tr = 3.0 + 1.5 * X_tr[:, 0] + 0.01 * X_tr[:, 1] - 0.3 * X_tr[:, 2] + rng.normal(0, 0.5, size=80)

    # Shifted X_test catches fit_transform(X_test) data leakage!
    X_te = np.column_stack(
        [
            rng.normal(1.2, 1.5, size=25),
            rng.normal(650.0, 120.0, size=25),
            rng.normal(-15.0, 8.0, size=25),
        ]
    )

    cases = [
        (1.0, {}, "default alpha=1.0 on multi-scale features"),
        (25.0, {"alpha": 25.0}, "strong regularization alpha=25.0"),
    ]
    errors = []

    for alpha_val, kwargs, desc in cases:
        X_tr_c, y_tr_c, X_te_c = X_tr.copy(), y_tr.copy(), X_te.copy()
        try:
            got = fn(X_tr_c, y_tr_c, X_te_c, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch22_todo_04",
                False,
                "failed",
                "To-Do 22.4 — Runtime Error in `fit_ridge_standardized`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch22_todo_04",
                "To-Do 22.4",
                "fit_ridge_standardized(X_train, y_train, X_test)",
                10,
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch22_todo_04",
                False,
                "failed",
                "To-Do 22.4 — Return a 2-Tuple `(y_pred_test, coef_array)`",
                f"[{desc}] Expected `(y_pred_test, coef_array)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(X_tr_c, X_tr) or not np.array_equal(X_te_c, X_te):
            errors.append(f"[{desc}] Mutated input arrays!")

        y_pred_got = np.asarray(got[0], dtype=float)
        coef_got = np.asarray(got[1], dtype=float)

        pipe_ref = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha_val))])
        pipe_ref.fit(X_tr, y_tr)
        y_pred_exp = pipe_ref.predict(X_te)
        coef_exp = pipe_ref.named_steps["ridge"].coef_

        raw_ridge = Ridge(alpha=alpha_val).fit(X_tr, y_tr)

        if y_pred_got.shape != y_pred_exp.shape:
            errors.append(f"[{desc}] y_pred_test shape expected {y_pred_exp.shape}, got {y_pred_got.shape}.")
            continue
        if coef_got.shape != coef_exp.shape:
            errors.append(f"[{desc}] coef_array shape expected {coef_exp.shape}, got {coef_got.shape}.")
            continue

        if not np.allclose(coef_got, coef_exp, atol=1e-4):
            extra = ""
            if np.allclose(coef_got, raw_ridge.coef_, atol=1e-3):
                extra = " (Missing StandardScaler! Ridge must be wrapped in a Pipeline with StandardScaler.)"
            errors.append(
                f"[{desc}] coef_array mismatch: expected {np.round(coef_exp, 4)}, got {np.round(coef_got, 4)}{extra}"
            )
        if not np.allclose(y_pred_got, y_pred_exp, atol=1e-4):
            errors.append(
                f"[{desc}] y_pred_test mismatch — ensure the scaler is fit ONLY on X_train and applied via Pipeline.predict(X_test)!"
            )

    if errors:
        return CheckResult(
            "ch22_todo_04",
            False,
            "failed",
            "To-Do 22.4 — Check Leak-Free Standardized Ridge Pipeline",
            "Build `Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])`, fit on `(X_train, y_train)`, and return `(pipe.predict(X_test), pipe.named_steps['ridge'].coef_)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch22_todo_04",
        True,
        "passed",
        "To-Do 22.4 Passed — Leak-Free Standardized Ridge Pipeline Verified!",
        "Confirmed leak-free scaling on shifted test features and exact L2 shrinkage across alpha=1.0 and alpha=25.0.",
        tip="Never apply Ridge or Lasso without StandardScaler inside a Pipeline: unscaled penalties unfairly crush small-unit features!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_todo_04",
        chapter="ch22",
        title="To-Do 22.4: Leak-Free Standardized Ridge Pipeline (fit_ridge_standardized)",
        book_section="Section 22.3.5 (Ridge Regression & Feature Scaling)",
        points=10,
        hints={
            1: "Create a two-step Scikit-Learn `Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])`.",
            2: "Call `pipe.fit(X_train, y_train)`, compute `y_pred = pipe.predict(X_test)`, and extract `coef = pipe.named_steps['ridge'].coef_`.",
            3: "def fit_ridge_standardized(X_train, y_train, X_test, alpha=1.0):\n    pipe = Pipeline([\n        ('scaler', StandardScaler()),\n        ('ridge', Ridge(alpha=alpha)),\n    ])\n    pipe.fit(X_train, y_train)\n    return (pipe.predict(X_test), pipe.named_steps['___'].coef_)",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 22.5: count_lasso_selected_features(X_train, y_train, alpha=0.1, tol=1e-6)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable(
        "ch22_todo_05", "To-Do 22.5", fn, "count_lasso_selected_features", 10
    )
    if bad is not None:
        return bad

    rng = np.random.default_rng(99)
    X_sparse = rng.normal(size=(120, 10))
    # Give features wildly different scales to verify StandardScaler is used
    scales = np.array([1.0, 50.0, 0.05, 10.0, 100.0, 0.2, 5.0, 25.0, 0.1, 2.0])
    X_sparse = X_sparse * scales
    # Only features 0, 1, 2 carry real signal
    Z_true = (X_sparse - X_sparse.mean(axis=0)) / X_sparse.std(axis=0)
    y_sparse = 2.0 * Z_true[:, 0] - 1.5 * Z_true[:, 1] + 0.8 * Z_true[:, 2] + rng.normal(0, 0.3, size=120)

    cases = [
        ({}, 0.1, 1e-6, "default alpha=0.1, tol=1e-6"),
        ({"alpha": 0.9}, 0.9, 1e-6, "stronger sparsity alpha=0.9"),
        ({"alpha": 50.0}, 50.0, 1e-6, "extreme penalty alpha=50.0 (0 features)"),
        ({"alpha": 0.1, "tol": 1.0}, 0.1, 1.0, "high threshold tol=1.0"),
    ]
    errors = []

    for kwargs, alpha_v, tol_v, desc in cases:
        X_c, y_c = X_sparse.copy(), y_sparse.copy()
        try:
            got = fn(X_c, y_c, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch22_todo_05",
                False,
                "failed",
                "To-Do 22.5 — Runtime Error in `count_lasso_selected_features`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch22_todo_05", "To-Do 22.5", "count_lasso_selected_features(X_train, y_train)", 10
            )

        if not _is_int_like(got):
            errors.append(f"[{desc}] Expected an int count, got {type(got).__name__} ({got!r}).")
            continue

        pipe_ref = Pipeline(
            [("scaler", StandardScaler()), ("lasso", Lasso(alpha=alpha_v, max_iter=5000))]
        )
        pipe_ref.fit(X_sparse, y_sparse)
        exp_count = int(np.sum(np.abs(pipe_ref.named_steps["lasso"].coef_) > tol_v))

        if int(got) != exp_count:
            errors.append(
                f"[{desc}] Expected {exp_count} selected features, got {int(got)}. "
                "Verify StandardScaler + Lasso(alpha=alpha, max_iter=5000) and `abs(coef) > tol`."
            )

    if errors:
        return CheckResult(
            "ch22_todo_05",
            False,
            "failed",
            "To-Do 22.5 — Check Standardized Lasso Sparsity Count",
            "Fit `Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(alpha=alpha, max_iter=5000))])` and return `int(np.sum(np.abs(coef) > tol))`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch22_todo_05",
        True,
        "passed",
        "To-Do 22.5 Passed — Lasso L1 Feature Selection Count Verified!",
        "Accurately counted surviving nonzero standardized Lasso coefficients across moderate, strong, extreme, and custom-tolerance regimes.",
        tip="Lasso's diamond L1 constraint drives irrelevant noise features to exact zero — turning regularization into automatic feature selection!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_todo_05",
        chapter="ch22",
        title="To-Do 22.5: Lasso Sparsity & Feature Selection (count_lasso_selected_features)",
        book_section="Section 22.3.6 (Lasso L1 & ElasticNet Sparsity)",
        points=10,
        hints={
            1: "Wrap `StandardScaler()` and `Lasso(alpha=alpha, max_iter=5000)` in a `Pipeline` and fit on `(X_train, y_train)`.",
            2: "Extract `coefs = pipe.named_steps['lasso'].coef_` and return `int(np.sum(np.abs(coefs) > tol))`.",
            3: "def count_lasso_selected_features(X_train, y_train, alpha=0.1, tol=1e-6):\n    pipe = Pipeline([\n        ('scaler', StandardScaler()),\n        ('lasso', Lasso(alpha=alpha, max_iter=5000)),\n    ])\n    pipe.fit(X_train, y_train)\n    coefs = pipe.named_steps['lasso'].coef_\n    return int(np.sum(np.abs(coefs) > ___))",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 22.6: residual_summary(y_true, y_pred)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch22_todo_06", "To-Do 22.6", fn, "residual_summary", 10)
    if bad is not None:
        return bad

    cases = [
        (
            [10.0, 12.0, 14.0, 16.0],
            [9.5, 12.5, 13.5, 16.5],
            "unbiased symmetric residuals (mean=0)",
        ),
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.5, 1.5, 2.5, 3.0, 3.5],
            "positively biased under-predictions",
        ),
        (
            [-1.2, -0.5, 0.3, -2.8, 0.1],
            [-1.0, -0.4, 0.2, -0.3, 0.0],
            "single large clinical outlier residual",
        ),
    ]
    expected_keys = ["mean_residual", "std_residual", "max_abs_error"]
    errors = []

    for yt_raw, yp_raw, desc in cases:
        yt_c, yp_c = list(yt_raw), list(yp_raw)
        try:
            got = fn(yt_c, yp_c)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch22_todo_06",
                False,
                "failed",
                "To-Do 22.6 — Runtime Error in `residual_summary`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch22_todo_06", "To-Do 22.6", "residual_summary(y_true, y_pred)", 10
            )

        key_err = _check_dict_keys("ch22_todo_06", "To-Do 22.6", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        if yt_c != yt_raw or yp_c != yp_raw:
            errors.append(f"[{desc}] Mutated input sequence(s)!")

        resid = np.asarray(yt_raw, dtype=float) - np.asarray(yp_raw, dtype=float)
        exp_mean = round(float(np.mean(resid)), 4)
        exp_std = round(float(np.std(resid)), 4)
        exp_std_ddof1 = round(float(np.std(resid, ddof=1)), 4)
        exp_max = round(float(np.max(np.abs(resid))), 4)

        if not _is_number(got["mean_residual"]) or not math.isclose(
            float(got["mean_residual"]), exp_mean, abs_tol=1e-4
        ):
            extra = ""
            if _is_number(got["mean_residual"]) and exp_mean != 0.0 and math.isclose(
                float(got["mean_residual"]), -exp_mean, abs_tol=1e-4
            ):
                extra = " (Sign flipped! Define residual as `y_true - y_pred`, not `y_pred - y_true`.)"
            errors.append(f"[{desc}] mean_residual: expected {exp_mean}, got {got['mean_residual']!r}{extra}")

        if not _is_number(got["std_residual"]) or not math.isclose(
            float(got["std_residual"]), exp_std, abs_tol=1e-4
        ):
            extra = ""
            if _is_number(got["std_residual"]) and math.isclose(
                float(got["std_residual"]), exp_std_ddof1, abs_tol=1e-3
            ):
                extra = " (Use NumPy default `np.std(resid)` with `ddof=0` so unbiased std matches RMSE!)"
            errors.append(f"[{desc}] std_residual: expected {exp_std}, got {got['std_residual']!r}{extra}")

        if not _is_number(got["max_abs_error"]) or not math.isclose(
            float(got["max_abs_error"]), exp_max, abs_tol=1e-4
        ):
            errors.append(f"[{desc}] max_abs_error: expected {exp_max}, got {got['max_abs_error']!r}")

    if errors:
        return CheckResult(
            "ch22_todo_06",
            False,
            "failed",
            "To-Do 22.6 — Check Residual Definition (`y_true - y_pred`) & Summary Stats",
            "Compute `resid = y_true - y_pred` and return mean, `np.std(resid)` (`ddof=0`), and `np.max(np.abs(resid))` rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch22_todo_06",
        True,
        "passed",
        "To-Do 22.6 Passed — Residual Diagnostics Summary Verified!",
        "Verified unbiased residuals, systematic prediction bias detection, and worst-case clinical error tracking.",
        tip="A non-zero mean residual signals systematic bias, while a huge max_abs_error warns of influential outliers distorting MSE!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_todo_06",
        chapter="ch22",
        title="To-Do 22.6: Residual Diagnostics Summary (residual_summary)",
        book_section="Section 22.3.7 (Residual Diagnostics & Error Inspection)",
        points=10,
        hints={
            1: "Define `resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)`.",
            2: "Compute `np.mean(resid)`, `np.std(resid)` (default `ddof=0`), and `np.max(np.abs(resid))`, rounding all three to 4 decimals.",
            3: "def residual_summary(y_true, y_pred):\n    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)\n    return {\n        'mean_residual': round(float(np.mean(resid)), 4),\n        'std_residual': round(float(np.std(resid)), 4),\n        'max_abs_error': round(float(np.max(np.abs(resid))), 4),\n    }",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 22.1 (30 pts): compare_ols_ridge_lasso(...)
# ---------------------------------------------------------------------------
def _ref_compare_models(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    ridge_alpha: float = 1.0,
    lasso_alpha: float = 0.1,
) -> Any:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    specs = [
        ("OLS", LinearRegression()),
        ("Ridge", Ridge(alpha=ridge_alpha)),
        ("Lasso", Lasso(alpha=lasso_alpha, max_iter=5000)),
    ]
    rows = []
    for name, est in specs:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        coefs = pipe.named_steps["model"].coef_
        rows.append(
            {
                "model": name,
                "rmse": round(float(root_mean_squared_error(y_test, pred)), 4),
                "mae": round(float(mean_absolute_error(y_test, pred)), 4),
                "r2": round(float(r2_score(y_test, pred)), 4),
                "nonzero_coefs": int(np.sum(np.abs(coefs) > 1e-6)),
            }
        )
    return pd.DataFrame(rows)


def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    bad = _check_callable("ch22_exam_01", "Exam Task 22.1", fn, "compare_ols_ridge_lasso", 30)
    if bad is not None:
        return bad

    rng = np.random.default_rng(2026)
    X_all = rng.normal(size=(160, 8))
    # Inject collinearity + multi-scale features + 5 noise features
    X_all[:, 1] = X_all[:, 0] * 0.95 + rng.normal(0, 0.1, size=160)
    X_all[:, 2] = X_all[:, 2] * 50.0
    y_all = 1.5 * X_all[:, 0] - 0.04 * X_all[:, 2] + rng.normal(0, 0.6, size=160)
    X_tr, X_te = X_all[:120], X_all[120:]
    y_tr, y_te = y_all[:120], y_all[120:]

    cases = [
        ({}, 1.0, 0.1, "default ridge_alpha=1.0, lasso_alpha=0.1"),
        (
            {"ridge_alpha": 15.0, "lasso_alpha": 0.35},
            15.0,
            0.35,
            "custom ridge_alpha=15.0, lasso_alpha=0.35",
        ),
    ]
    expected_cols = ["model", "rmse", "mae", "r2", "nonzero_coefs"]
    expected_models = ["OLS", "Ridge", "Lasso"]
    errors = []

    for kwargs, r_a, l_a, desc in cases:
        X_tr_c, y_tr_c, X_te_c, y_te_c = X_tr.copy(), y_tr.copy(), X_te.copy(), y_te.copy()
        try:
            got = fn(X_tr_c, y_tr_c, X_te_c, y_te_c, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch22_exam_01",
                False,
                "failed",
                "Exam Task 22.1 — Runtime Error in `compare_ols_ridge_lasso`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch22_exam_01", "Exam Task 22.1", "compare_ols_ridge_lasso(...)", 30
            )

        if not isinstance(got, pd.DataFrame):
            return CheckResult(
                "ch22_exam_01",
                False,
                "failed",
                "Exam Task 22.1 — Return a Pandas DataFrame",
                f"[{desc}] Expected a pd.DataFrame, got {type(got).__name__}.",
                points_earned=0,
                points_possible=30,
            )

        if not np.array_equal(X_tr_c, X_tr) or not np.array_equal(X_te_c, X_te):
            errors.append(f"[{desc}] Mutated input train/test arrays!")

        if list(got.columns) != expected_cols:
            errors.append(f"[{desc}] Columns expected {expected_cols}, got {list(got.columns)}.")
            continue

        got_reset = got.reset_index(drop=True)
        if list(got_reset["model"]) != expected_models:
            errors.append(
                f"[{desc}] 'model' column expected {expected_models}, got {list(got_reset['model'])}."
            )
            continue

        exp_df = _ref_compare_models(X_tr, y_tr, X_te, y_te, ridge_alpha=r_a, lasso_alpha=l_a)
        for idx, m_name in enumerate(expected_models):
            for col in ["rmse", "mae", "r2"]:
                g_val = float(got_reset.loc[idx, col])
                e_val = float(exp_df.loc[idx, col])
                if not math.isclose(g_val, e_val, abs_tol=1e-3):
                    errors.append(
                        f"[{desc}] {m_name} {col}: expected {e_val}, got {g_val} "
                        "(check StandardScaler Pipeline & alpha parameters!)."
                    )
            g_nz = int(got_reset.loc[idx, "nonzero_coefs"])
            e_nz = int(exp_df.loc[idx, "nonzero_coefs"])
            if g_nz != e_nz:
                errors.append(f"[{desc}] {m_name} nonzero_coefs: expected {e_nz}, got {g_nz}.")

    if errors:
        return CheckResult(
            "ch22_exam_01",
            False,
            "failed",
            "Exam Task 22.1 — Check OLS/Ridge/Lasso Benchmark DataFrame",
            "Fit leak-free standardized Pipelines for OLS, Ridge(alpha=ridge_alpha), and Lasso(alpha=lasso_alpha, max_iter=5000), returning ['model', 'rmse', 'mae', 'r2', 'nonzero_coefs'] rounded to 4 decimals.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch22_exam_01",
        True,
        "passed",
        "Exam Task 22.1 Passed (30/30 pts) — OLS vs. Ridge vs. Lasso Benchmark Verified!",
        "Built a leak-free standardized comparison table verifying test RMSE, MAE, R^2, and Lasso sparsity counts.",
        points_earned=30,
        points_possible=30,
        tip="Reporting accuracy metrics alongside nonzero feature counts shows both predictive performance and parsimony in one Table!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_exam_01",
        chapter="ch22",
        title="Exam 22.1: OLS vs. Ridge vs. Lasso Benchmark (compare_ols_ridge_lasso)",
        book_section="Chapter 22 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Loop over `[('OLS', LinearRegression()), ('Ridge', Ridge(alpha=ridge_alpha)), ('Lasso', Lasso(alpha=lasso_alpha, max_iter=5000))]`, wrapping each in `Pipeline([('scaler', StandardScaler()), ('model', est)])`.",
            2: "For each model, fit on `(X_train, y_train)`, predict on `X_test`, compute `rmse`, `mae`, `r2` rounded to 4 decimals, and count `int(np.sum(np.abs(coefs) > 1e-6))`.",
            3: "def compare_ols_ridge_lasso(X_train, y_train, X_test, y_test, ridge_alpha=1.0, lasso_alpha=0.1):\n    rows = []\n    for name, est in [('OLS', LinearRegression()), ('Ridge', Ridge(alpha=ridge_alpha)), ('Lasso', Lasso(alpha=lasso_alpha, max_iter=5000))]:\n        pipe = Pipeline([('scaler', StandardScaler()), ('model', est)])\n        pipe.fit(X_train, y_train)\n        pred = pipe.predict(X_test)\n        coefs = pipe.named_steps['model'].coef_\n        rows.append({'model': name, 'rmse': round(float(root_mean_squared_error(y_test, pred)), 4), 'mae': round(float(mean_absolute_error(y_test, pred)), 4), 'r2': round(float(r2_score(y_test, pred)), 4), 'nonzero_coefs': int(np.sum(np.abs(coefs) > 1e-6))})\n    return pd.DataFrame(rows)",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 22.2 (35 pts): tune_ridge_alpha_cv(X, y, alphas, cv=5)
# ---------------------------------------------------------------------------
def _ref_tune_ridge(X: Any, y: Any, alphas: Any, cv: int = 5) -> dict[str, Any]:
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []
    for a in alphas:
        pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=float(a)))])
        scores = cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        rmses = -scores
        rows.append(
            {
                "alpha": float(a),
                "mean_rmse": round(float(rmses.mean()), 4),
                "std_rmse": round(float(rmses.std()), 4),
            }
        )
    cv_df = pd.DataFrame(rows)
    best_idx = int(cv_df["mean_rmse"].idxmin())
    return {
        "best_alpha": float(cv_df.loc[best_idx, "alpha"]),
        "best_rmse": float(cv_df.loc[best_idx, "mean_rmse"]),
        "cv_table": cv_df,
    }


def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    bad = _check_callable("ch22_exam_02", "Exam Task 22.2", fn, "tune_ridge_alpha_cv", 35)
    if bad is not None:
        return bad

    rng = np.random.default_rng(314)
    n_obs = 90
    base = rng.normal(size=(n_obs, 3))
    # High collinearity + 12 noisy copies so intermediate Ridge alpha shines
    noise_cols = rng.normal(0, 1.0, size=(n_obs, 12))
    collinear_cols = np.column_stack(
        [
            base[:, 0] + rng.normal(0, 0.05, size=n_obs),
            base[:, 1] * 20.0 + rng.normal(0, 0.5, size=n_obs),
        ]
    )
    X_syn = np.hstack([base, collinear_cols, noise_cols])
    y_syn = 2.0 * base[:, 0] - 1.5 * base[:, 1] + 0.8 * base[:, 2] + rng.normal(0, 1.5, size=n_obs)

    alphas_1 = [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]
    alphas_2 = [0.01, 1.0, 50.0]

    cases = [
        (X_syn, y_syn, alphas_1, {}, 5, "5-fold CV over 6 alphas on collinear data"),
        (X_syn, y_syn, alphas_2, {"cv": 3}, 3, "3-fold CV with custom cv=3"),
    ]
    expected_keys = ["best_alpha", "best_rmse", "cv_table"]
    expected_cols = ["alpha", "mean_rmse", "std_rmse"]
    errors = []

    for X_in, y_in, alphas_in, kwargs, cv_val, desc in cases:
        X_c, y_c, a_c = X_in.copy(), y_in.copy(), list(alphas_in)
        try:
            got = fn(X_c, y_c, a_c, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch22_exam_02",
                False,
                "failed",
                "Exam Task 22.2 — Runtime Error in `tune_ridge_alpha_cv`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch22_exam_02", "Exam Task 22.2", "tune_ridge_alpha_cv(X, y, alphas)", 35
            )

        key_err = _check_dict_keys("ch22_exam_02", "Exam Task 22.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not np.array_equal(X_c, X_in) or not np.array_equal(y_c, y_in):
            errors.append(f"[{desc}] Mutated input arrays!")

        exp = _ref_tune_ridge(X_in, y_in, alphas_in, cv=cv_val)

        if not _is_number(got["best_rmse"]) or not math.isclose(
            float(got["best_rmse"]), exp["best_rmse"], abs_tol=1e-3
        ):
            extra = ""
            if _is_number(got["best_rmse"]) and float(got["best_rmse"]) < 0:
                extra = " (Negative RMSE detected — negate `neg_root_mean_squared_error` scores with `-scores`!)"
            errors.append(
                f"[{desc}] best_rmse: expected {exp['best_rmse']}, got {got['best_rmse']!r}{extra}"
            )

        if not _is_number(got["best_alpha"]) or not math.isclose(
            float(got["best_alpha"]), exp["best_alpha"], rel_tol=1e-6, abs_tol=1e-9
        ):
            errors.append(
                f"[{desc}] best_alpha: expected {exp['best_alpha']}, got {got['best_alpha']!r}"
            )

        tbl = got["cv_table"]
        if not isinstance(tbl, pd.DataFrame):
            errors.append(f"[{desc}] cv_table must be a pd.DataFrame, got {type(tbl).__name__}.")
            continue
        if list(tbl.columns) != expected_cols:
            errors.append(f"[{desc}] cv_table columns expected {expected_cols}, got {list(tbl.columns)}.")
            continue
        if len(tbl) != len(alphas_in):
            errors.append(f"[{desc}] cv_table expected {len(alphas_in)} rows, got {len(tbl)}.")
            continue

        tbl_reset = tbl.reset_index(drop=True)
        exp_tbl = exp["cv_table"]
        for i in range(len(exp_tbl)):
            for col in expected_cols:
                g_v = float(tbl_reset.loc[i, col])
                e_v = float(exp_tbl.loc[i, col])
                if not math.isclose(g_v, e_v, abs_tol=1e-3):
                    errors.append(
                        f"[{desc}] cv_table row {i} {col}: expected {e_v}, got {g_v}."
                    )

    if errors:
        return CheckResult(
            "ch22_exam_02",
            False,
            "failed",
            "Exam Task 22.2 — Check Leak-Free KFold CV & Alpha Selection",
            "Use `KFold(n_splits=cv, shuffle=True, random_state=42)` + `Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=a))])` with `scoring='neg_root_mean_squared_error'` and negate scores (`-scores`).",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch22_exam_02",
        True,
        "passed",
        "Exam Task 22.2 Passed (35/35 pts) — Leak-Free Cross-Validated Ridge Tuning Verified!",
        "Correctly evaluated every alpha inside a leak-free KFold Pipeline, negated CV RMSE scores, and selected the optimal penalty.",
        points_earned=35,
        points_possible=35,
        tip="Putting StandardScaler INSIDE cross_val_score's Pipeline guarantees every fold's validation statistics remain 100% unseen!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_exam_02",
        chapter="ch22",
        title="Exam 22.2: Cross-Validated Ridge Alpha Tuning (tune_ridge_alpha_cv)",
        book_section="Chapter 22 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Create `kf = KFold(n_splits=cv, shuffle=True, random_state=42)`. For each `a` in `alphas`, run `cross_val_score(pipe, X, y, cv=kf, scoring='neg_root_mean_squared_error')` on a `Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=float(a)))])`.",
            2: "Negate the scores (`rmses = -scores`) to get positive RMSEs, build `cv_table` with columns `['alpha', 'mean_rmse', 'std_rmse']` (rounded to 4 decimals for RMSE columns), and select the row with minimum `mean_rmse`.",
            3: "def tune_ridge_alpha_cv(X, y, alphas, cv=5):\n    kf = KFold(n_splits=cv, shuffle=True, random_state=42)\n    rows = []\n    for a in alphas:\n        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=float(a)))])\n        rmses = -cross_val_score(pipe, X, y, cv=kf, scoring='neg_root_mean_squared_error')\n        rows.append({'alpha': float(a), 'mean_rmse': round(float(rmses.mean()), 4), 'std_rmse': round(float(rmses.std()), 4)})\n    cv_table = pd.DataFrame(rows)\n    best_row = cv_table.loc[cv_table['mean_rmse'].idxmin()]\n    return {'best_alpha': float(best_row['alpha']), 'best_rmse': float(best_row['mean_rmse']), 'cv_table': cv_table}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 22.3 (35 pts): predict_child_haz_pipeline(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _ref_predict_haz(mch_df: Any, seed: int = 42) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    work = mch_df.copy()
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    work["anc_visits"] = work["anc_visits"].replace(999, np.nan)
    work["is_urban"] = (work["residence"] == "Urban").astype(int)

    feature_cols = [
        "anc_visits",
        "maternal_hb_g_dl",
        "maternal_age_years",
        "is_urban",
    ]
    clean = work[feature_cols + ["child_haz"]].dropna()

    X = clean[feature_cols]
    y = clean["child_haz"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    coefs = {
        feat: round(float(val), 4)
        for feat, val in zip(feature_cols, pipe.named_steps["ridge"].coef_)
    }
    return {
        "n_clean": int(len(clean)),
        "test_rmse": round(float(root_mean_squared_error(y_te, y_pred)), 4),
        "test_r2": round(float(r2_score(y_te, y_pred)), 4),
        "coefs": coefs,
    }


def _verify_exam_03(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable(
        "ch22_exam_03", "Exam Task 22.3", fn, "predict_child_haz_pipeline", 35
    )
    if bad is not None:
        return bad

    df_mch = load_data("mch_survey_sample", as_pandas=True)

    # Secondary synthetic messy survey DataFrame
    rng = np.random.default_rng(2203)
    n_syn = 150
    res_raw = rng.choice(["Urban", "urban", " URBAN ", "Rural", "rural", " Rural "], size=n_syn)
    anc_raw = rng.integers(0, 10, size=n_syn).astype(float)
    hb_raw = rng.normal(11.4, 1.3, size=n_syn)
    age_raw = rng.integers(18, 42, size=n_syn).astype(float)
    is_u = np.array([s.strip().title() == "Urban" for s in res_raw], dtype=int)
    haz_raw = -2.2 + 0.12 * anc_raw + 0.08 * hb_raw - 0.01 * age_raw + 0.25 * is_u + rng.normal(0, 0.7, size=n_syn)
    anc_raw[[3, 17, 51]] = 999.0
    hb_raw[[10, 42, 88]] = np.nan
    haz_raw[[25]] = np.nan
    df_syn = pd.DataFrame(
        {
            "residence": res_raw,
            "anc_visits": anc_raw,
            "maternal_hb_g_dl": hb_raw,
            "maternal_age_years": age_raw,
            "child_haz": haz_raw,
        }
    )

    cases = [
        (df_mch, {}, 42, "real mch_survey_sample (seed=42)"),
        (df_mch, {"seed": 7}, 7, "real mch_survey_sample (custom seed=7)"),
        (df_syn, {"seed": 42}, 42, "synthetic messy cohort (casing + 999 + NaN)"),
    ]
    expected_keys = ["n_clean", "test_rmse", "test_r2", "coefs"]
    feature_cols = [
        "anc_visits",
        "maternal_hb_g_dl",
        "maternal_age_years",
        "is_urban",
    ]
    errors = []

    for df_in, kwargs, seed_val, desc in cases:
        df_copy = df_in.copy(deep=True)
        try:
            got = fn(df_copy, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch22_exam_03",
                False,
                "failed",
                "Exam Task 22.3 — Runtime Error in `predict_child_haz_pipeline`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=[
                    "Make sure you replace `anc_visits == 999` with `np.nan` and drop NaNs on all 5 required columns before train_test_split and fitting!"
                ],
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch22_exam_03", "Exam Task 22.3", "predict_child_haz_pipeline(mch_df)", 35
            )

        key_err = _check_dict_keys("ch22_exam_03", "Exam Task 22.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not df_copy.equals(df_in):
            errors.append(f"[{desc}] Input DataFrame was mutated! Start with `work = mch_df.copy()`.")

        exp = _ref_predict_haz(df_in, seed=seed_val)

        if not _is_int_like(got["n_clean"]) or int(got["n_clean"]) != exp["n_clean"]:
            errors.append(
                f"[{desc}] n_clean: expected {exp['n_clean']}, got {got['n_clean']!r} "
                "(did you replace `anc_visits == 999` with NaN and drop NaNs on all 5 columns?)"
            )
            continue

        for k in ["test_rmse", "test_r2"]:
            if not _is_number(got[k]) or not math.isclose(float(got[k]), exp[k], abs_tol=1e-3):
                errors.append(f"[{desc}] {k}: expected {exp[k]}, got {got[k]!r}")

        if not isinstance(got["coefs"], dict) or set(got["coefs"].keys()) != set(feature_cols):
            errors.append(
                f"[{desc}] coefs must be a dict with keys {feature_cols}, got {got['coefs']!r}."
            )
            continue

        for feat in feature_cols:
            g_c = got["coefs"][feat]
            e_c = exp["coefs"][feat]
            if not _is_number(g_c) or not math.isclose(float(g_c), e_c, abs_tol=1e-3):
                errors.append(f"[{desc}] coefs[{feat!r}]: expected {e_c}, got {g_c!r}")

    if errors:
        return CheckResult(
            "ch22_exam_03",
            False,
            "failed",
            "Exam Task 22.3 — Check MCH Survey Cleaning, 80/20 Split & Ridge Pipeline",
            "Copy `mch_df`, standardize `residence`, replace `anc_visits == 999` with NaN, add `is_urban`, drop NaNs on the 5 model columns, split 80/20 (`random_state=seed`), and fit `Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch22_exam_03",
        True,
        "passed",
        "Exam Task 22.3 Passed (35/35 pts) — End-to-End Child HAZ Ridge Pipeline Verified!",
        "Cleaned messy MCH survey data (500 -> 470 complete cases), prevented leakage with an 80/20 split + standardized Ridge Pipeline, and extracted standardized feature weights.",
        points_earned=35,
        points_possible=35,
        tip="Standardized Ridge coefficients let clinicians compare feature strength per 1-SD shift across visits, g/dL, years, and urbanicity!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch22_exam_03",
        chapter="ch22",
        title="Exam 22.3: End-to-End Child HAZ Prediction Pipeline (predict_child_haz_pipeline)",
        book_section="Chapter 22 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Start with `work = mch_df.copy()`. Standardize `residence` via `.astype(str).str.strip().str.title()`, replace `anc_visits == 999` with `np.nan`, create `is_urban = (work['residence'] == 'Urban').astype(int)`, and drop NaNs on `['anc_visits', 'maternal_hb_g_dl', 'maternal_age_years', 'is_urban', 'child_haz']`.",
            2: "Split `X` (the 4 features) and `y` (`child_haz`) with `train_test_split(X, y, test_size=0.2, random_state=seed)`, fit `Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])`, and evaluate `root_mean_squared_error` and `r2_score` on `(y_test, y_pred)`.",
            3: "def predict_child_haz_pipeline(mch_df, seed=42):\n    work = mch_df.copy()\n    work['residence'] = work['residence'].astype(str).str.strip().str.title()\n    work['anc_visits'] = work['anc_visits'].replace(999, np.nan)\n    work['is_urban'] = (work['residence'] == 'Urban').astype(int)\n    feats = ['anc_visits', 'maternal_hb_g_dl', 'maternal_age_years', 'is_urban']\n    clean = work[feats + ['child_haz']].dropna()\n    X_tr, X_te, y_tr, y_te = train_test_split(clean[feats], clean['child_haz'], test_size=0.2, random_state=seed)\n    pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0))])\n    pipe.fit(X_tr, y_tr)\n    pred = pipe.predict(X_te)\n    return {'n_clean': int(len(clean)), 'test_rmse': round(float(root_mean_squared_error(y_te, pred)), 4), 'test_r2': round(float(r2_score(y_te, pred)), 4), 'coefs': {f: round(float(c), 4) for f, c in zip(feats, pipe.named_steps['ridge'].coef_)}}",
        },
        verifier=_verify_exam_03,
    )
)
