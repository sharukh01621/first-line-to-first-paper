"""Chapter 27 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch27_todo_01..06) and the Coding Exam (ch27_exam_01..03)
for Neural Networks from Scratch in NumPy — Perceptrons, Activations,
Backpropagation, and MLPs.
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn inside the check
function, validates return structures, numerical stability, gradient accuracy,
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
        tip="Neural network layers and loss helpers always return tensors, caches, or metric dicts.",
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
# GUIDED TO-DO 27.1: relu_and_grad(z)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch27_todo_01", "To-Do 27.1", fn, "relu_and_grad", 10)
    if bad is not None:
        return bad

    cases = [
        (np.array([-2.5, 0.0, 3.0, -0.1, 4.2]), "1D mixed signs including exact 0.0"),
        (np.array([[-1.0, 2.0, 0.0], [5.5, -3.2, 0.7]]), "2D hidden pre-activation matrix"),
        ([-4, 0, 2], "Python list of ints"),
    ]
    errors = []

    for z_raw, desc in cases:
        z_copy = z_raw.copy() if isinstance(z_raw, np.ndarray) else list(z_raw)
        try:
            got = fn(z_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_todo_01",
                False,
                "failed",
                "To-Do 27.1 — Runtime Error in `relu_and_grad`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch27_todo_01", "To-Do 27.1", "relu_and_grad(z)", 10)

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch27_todo_01",
                False,
                "failed",
                "To-Do 27.1 — Return a 2-Tuple `(relu_out, relu_grad)`",
                f"[{desc}] Expected a 2-tuple of NumPy arrays, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if isinstance(z_raw, np.ndarray):
            if not np.array_equal(z_copy, z_raw):
                errors.append(f"[{desc}] Mutated caller's input array! Use `np.maximum(0.0, z)` instead of in-place masking.")
        elif z_copy != z_raw:
            errors.append(f"[{desc}] Mutated caller's input list!")

        out_got, grad_got = got
        if not isinstance(out_got, np.ndarray) or not isinstance(grad_got, np.ndarray):
            errors.append(f"[{desc}] Both elements must be NumPy ndarrays.")
            continue

        z_arr = np.asarray(z_raw, dtype=float)
        exp_out = np.maximum(0.0, z_arr)
        exp_grad = (z_arr > 0).astype(float)

        if out_got.shape != exp_out.shape or grad_got.shape != exp_grad.shape:
            errors.append(
                f"[{desc}] Shape mismatch: expected {exp_out.shape}, got out={out_got.shape}, grad={grad_got.shape}."
            )
            continue
        if not np.issubdtype(out_got.dtype, np.floating) or not np.issubdtype(grad_got.dtype, np.floating):
            errors.append(
                f"[{desc}] Expected float ndarrays, got dtypes ({out_got.dtype}, {grad_got.dtype})."
            )
        if not np.allclose(out_got, exp_out, atol=1e-8):
            errors.append(f"[{desc}] ReLU values mismatch: expected {exp_out}, got {out_got}.")
        if not np.allclose(grad_got, exp_grad, atol=1e-8):
            extra = ""
            if np.allclose(grad_got, (z_arr >= 0).astype(float), atol=1e-8):
                extra = " (Use strict `z > 0` so the subgradient at exact 0.0 is 0.0!)"
            errors.append(f"[{desc}] Gradient mismatch: expected {exp_grad}, got {grad_got}.{extra}")

    if errors:
        return CheckResult(
            "ch27_todo_01",
            False,
            "failed",
            "To-Do 27.1 — Check ReLU Value, Subgradient & Non-Mutation",
            "Convert `z` with `np.asarray(z, dtype=float)` and return `(np.maximum(0.0, z_arr), (z_arr > 0).astype(float))`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch27_todo_01",
        True,
        "passed",
        "To-Do 27.1 Passed — ReLU Activation & Subgradient Verified!",
        "Verified 1D, 2D, and list inputs with exact zero handling, float dtypes, and non-mutation.",
        tip="ReLU's derivative is a simple binary mask `(z > 0).astype(float)`—zeroing gradients for inactive neurons during backprop!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_todo_01",
        chapter="ch27",
        title="To-Do 27.1: ReLU Activation & Derivative (relu_and_grad)",
        book_section="Section 27.3.2 (Activation Functions & Numerical Stability)",
        points=10,
        hints={
            1: "Convert the input without mutating it via `z_arr = np.asarray(z, dtype=float)`.",
            2: "Compute `out = np.maximum(0.0, z_arr)` and the derivative mask `grad = (z_arr > 0).astype(float)`.",
            3: "def relu_and_grad(z):\n    z_arr = np.asarray(z, dtype=float)\n    out = np.maximum(0.0, z_arr)\n    grad = (z_arr > 0).astype(float)\n    return (out, ___)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 27.2: stable_softmax(logits)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch27_todo_02", "To-Do 27.2", fn, "stable_softmax", 10)
    if bad is not None:
        return bad

    cases = [
        (np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]), "standard 2D logits & uniform row"),
        (
            np.array([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]]),
            "extreme logits [1000, 1001, 999] overflow/underflow stress test",
        ),
        (np.array([[2.5, -1.5, 0.5, 4.0]]), "single-row 4-class logits"),
    ]
    errors = []

    for logits_raw, desc in cases:
        logits_copy = logits_raw.copy()
        try:
            got = fn(logits_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_todo_02",
                False,
                "failed",
                "To-Do 27.2 — Runtime Error in `stable_softmax`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch27_todo_02", "To-Do 27.2", "stable_softmax(logits)", 10)

        if not np.array_equal(logits_copy, logits_raw):
            errors.append(f"[{desc}] Mutated caller's `logits` array!")

        if not isinstance(got, np.ndarray):
            errors.append(f"[{desc}] Expected a NumPy ndarray, got {type(got).__name__}.")
            continue

        shifted = logits_raw - np.max(logits_raw, axis=1, keepdims=True)
        exp_e = np.exp(shifted)
        exp_p = exp_e / np.sum(exp_e, axis=1, keepdims=True)

        if got.shape != exp_p.shape:
            errors.append(f"[{desc}] Shape mismatch: expected {exp_p.shape}, got {got.shape}.")
            continue
        if not np.all(np.isfinite(got)):
            errors.append(
                f"[{desc}] Output contains NaN/Inf! Subtract row-wise max `z - np.max(z, axis=1, keepdims=True)` BEFORE calling `np.exp`."
            )
            continue
        if not np.allclose(np.sum(got, axis=1), np.ones(len(logits_raw)), atol=1e-6):
            errors.append(
                f"[{desc}] Rows do not sum to 1.0 (row sums: {np.round(np.sum(got, axis=1), 6)}). Check `axis=1, keepdims=True`!"
            )
            continue
        if not np.allclose(got, exp_p, atol=1e-6):
            errors.append(f"[{desc}] Softmax probabilities mismatch: expected {np.round(exp_p, 5)}, got {np.round(got, 5)}.")

    if errors:
        return CheckResult(
            "ch27_todo_02",
            False,
            "failed",
            "To-Do 27.2 — Check Log-Sum-Exp Max-Shift Trick & Row Normalization",
            "Shift each row by its maximum (`z - np.max(z, axis=1, keepdims=True)`), exponentiate, and divide by row sums (`axis=1, keepdims=True`).",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch27_todo_02",
        True,
        "passed",
        "To-Do 27.2 Passed — Overflow-Safe Stable Softmax Verified!",
        "Produced exact row-normalized probabilities even on extreme logits [1000.0, 1001.0, 999.0] with zero NaN/Inf overflow.",
        tip="Subtracting row-wise max is mathematically invariant in softmax (`exp(z - c) / sum(exp(z - c))`) and guarantees `exp(0) = 1` is the largest exponent!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_todo_02",
        chapter="ch27",
        title="To-Do 27.2: Numerically Stable Softmax (stable_softmax)",
        book_section="Section 27.3.2 (Activation Functions & Numerical Stability)",
        points=10,
        hints={
            1: "Convert `logits` to float array `z = np.asarray(logits, dtype=float)` and compute row-wise maxima with `np.max(z, axis=1, keepdims=True)`.",
            2: "Compute `shifted = z - np.max(z, axis=1, keepdims=True)`, `exp_z = np.exp(shifted)`, and divide by `np.sum(exp_z, axis=1, keepdims=True)`.",
            3: "def stable_softmax(logits):\n    z = np.asarray(logits, dtype=float)\n    shifted = z - np.max(z, axis=1, keepdims=True)\n    exp_z = np.exp(shifted)\n    return exp_z / np.sum(exp_z, axis=___, keepdims=True)",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 27.3: binary_bce_loss_and_grad(y_true, logits)
# ---------------------------------------------------------------------------
def _verify_todo_03(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch27_todo_03", "To-Do 27.3", fn, "binary_bce_loss_and_grad", 10)
    if bad is not None:
        return bad

    cases = [
        (np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0]), "neutral zero logits (loss = ln 2)"),
        (np.array([1.0, 0.0, 1.0, 0.0, 1.0]), np.array([1.5, -2.0, 0.4, 0.8, -0.5]), "mixed clinical logits"),
        (np.array([1.0, 0.0]), np.array([40.0, -40.0]), "extreme logits requiring 1e-12 probability clipping"),
    ]
    errors = []

    for y_raw, z_raw, desc in cases:
        y_c, z_c = y_raw.copy(), z_raw.copy()
        try:
            got = fn(y_c, z_c)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_todo_03",
                False,
                "failed",
                "To-Do 27.3 — Runtime Error in `binary_bce_loss_and_grad`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch27_todo_03", "To-Do 27.3", "binary_bce_loss_and_grad(y_true, logits)", 10
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch27_todo_03",
                False,
                "failed",
                "To-Do 27.3 — Return a 2-Tuple `(loss_round5, dlogits_array)`",
                f"[{desc}] Expected `(float, ndarray)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(y_c, y_raw) or not np.array_equal(z_c, z_raw):
            errors.append(f"[{desc}] Mutated caller's input array(s)!")

        loss_got, grad_got = got
        p_raw = 1.0 / (1.0 + np.exp(-np.clip(z_raw, -500.0, 500.0)))
        p_exp = np.clip(p_raw, 1e-12, 1.0 - 1e-12)
        loss_exp = round(-float(np.mean(y_raw * np.log(p_exp) + (1.0 - y_raw) * np.log(1.0 - p_exp))), 5)
        grad_exp = (p_exp - y_raw) / len(y_raw)

        if not _is_number(loss_got) or not math.isfinite(float(loss_got)):
            errors.append(f"[{desc}] Loss must be a finite float, got {loss_got!r}.")
            continue
        if not math.isclose(float(loss_got), loss_exp, abs_tol=1e-5):
            extra = ""
            if math.isclose(float(loss_got), -loss_exp, abs_tol=1e-4):
                extra = " (Missing leading minus sign on BCE loss!)"
            errors.append(f"[{desc}] Loss mismatch: expected {loss_exp}, got {loss_got!r}{extra}")

        grad_arr = np.asarray(grad_got, dtype=float)
        if grad_arr.shape != grad_exp.shape:
            errors.append(f"[{desc}] dlogits shape expected {grad_exp.shape}, got {grad_arr.shape}.")
            continue
        if not np.allclose(grad_arr, grad_exp, atol=1e-6):
            extra = ""
            if np.allclose(grad_arr, p_exp - y_raw, atol=1e-5):
                extra = " (Forgot to divide gradient by `len(y_true)` for mean BCE!)"
            elif np.allclose(grad_arr, -grad_exp, atol=1e-5):
                extra = " (Sign flipped! Gradient of BCE wrt logits is `(p - y_true) / n`, not `(y_true - p) / n`.)"
            errors.append(f"[{desc}] dlogits mismatch: expected {np.round(grad_exp, 5)}, got {np.round(grad_arr, 5)}{extra}")

    if errors:
        return CheckResult(
            "ch27_todo_03",
            False,
            "failed",
            "To-Do 27.3 — Check Clipped Sigmoid, Mean BCE Loss & Analytical Logit Gradient",
            "Compute `p = np.clip(1 / (1 + np.exp(-np.clip(z, -500, 500))), 1e-12, 1 - 1e-12)`, `loss = round(-mean(y*log(p) + (1-y)*log(1-p)), 5)`, and `dlogits = (p - y) / len(y)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch27_todo_03",
        True,
        "passed",
        "To-Do 27.3 Passed — Binary Cross-Entropy Loss & Logit Gradient Verified!",
        "Confirmed neutral ln(2) baseline, mixed clinical logits, and extreme-logit stability with exact analytical `(p - y) / n` gradient.",
        tip="Combining sigmoid and BCE cancels out `p(1-p)` cleanly, leaving `(p - y) / n`—avoiding vanishing gradients on confident wrong predictions!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_todo_03",
        chapter="ch27",
        title="To-Do 27.3: Binary Cross-Entropy Loss & Logit Gradient (binary_bce_loss_and_grad)",
        book_section="Section 27.3.3 (Binary Cross-Entropy & Analytical Gradient)",
        points=10,
        hints={
            1: "Convert inputs to float arrays, compute clipped sigmoid `p = np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))), 1e-12, 1.0 - 1e-12)`.",
            2: "Compute `loss = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))` rounded to 5 decimals, and `dlogits = (p - y) / len(y)`.",
            3: "def binary_bce_loss_and_grad(y_true, logits):\n    y = np.asarray(y_true, dtype=float)\n    z = np.asarray(logits, dtype=float)\n    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0))), 1e-12, 1.0 - 1e-12)\n    loss = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))\n    dlogits = (p - y) / len(y)\n    return (round(loss, 5), ___)",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 27.4: mlp_forward_two_layer(X, W1, b1, W2, b2)
# ---------------------------------------------------------------------------
def _verify_todo_04(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch27_todo_04", "To-Do 27.4", fn, "mlp_forward_two_layer", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(2704)
    cases = [
        (
            np.array([[1.0, -2.0], [0.5, 1.5], [-1.0, 0.0]]),
            np.array([[0.5, -0.4, 0.3], [-0.2, 0.8, -0.5]]),
            np.array([[0.1, -0.1, 0.2]]),
            np.array([[0.6], [-0.7], [0.4]]),
            np.array([[0.05]]),
            "small 3x2 -> 3 -> 1 batch with negative pre-activations",
        ),
        (
            rng.normal(size=(12, 4)),
            rng.normal(0, 0.7, size=(4, 6)),
            np.zeros((1, 6)),
            rng.normal(0, 0.7, size=(6, 1)),
            np.array([[-0.2]]),
            "12x4 -> 6 -> 1 random clinical feature batch",
        ),
    ]
    expected_keys = ["X", "Z1", "A1", "Z2", "probs"]
    errors = []

    for X, W1, b1, W2, b2, desc in cases:
        X_c, W1_c, b1_c, W2_c, b2_c = X.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy()
        try:
            got = fn(X_c, W1_c, b1_c, W2_c, b2_c)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_todo_04",
                False,
                "failed",
                "To-Do 27.4 — Runtime Error in `mlp_forward_two_layer`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch27_todo_04", "To-Do 27.4", "mlp_forward_two_layer(X, W1, b1, W2, b2)", 10
            )

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch27_todo_04",
                False,
                "failed",
                "To-Do 27.4 — Return a 2-Tuple `(probs, cache)`",
                f"[{desc}] Expected `(probs_ndarray, cache_dict)`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if (
            not np.array_equal(X_c, X)
            or not np.array_equal(W1_c, W1)
            or not np.array_equal(b1_c, b1)
            or not np.array_equal(W2_c, W2)
            or not np.array_equal(b2_c, b2)
        ):
            errors.append(f"[{desc}] Mutated one or more input arrays!")

        probs_got, cache_got = got
        key_err = _check_dict_keys("ch27_todo_04", "To-Do 27.4", cache_got, expected_keys, 10)
        if key_err is not None:
            return key_err

        Z1_exp = X @ W1 + b1
        A1_exp = np.maximum(0.0, Z1_exp)
        Z2_exp = A1_exp @ W2 + b2
        probs_exp = 1.0 / (1.0 + np.exp(-np.clip(Z2_exp, -500.0, 500.0)))

        exp_map = {"X": X, "Z1": Z1_exp, "A1": A1_exp, "Z2": Z2_exp, "probs": probs_exp}

        probs_arr = np.asarray(probs_got, dtype=float)
        if probs_arr.shape != probs_exp.shape or not np.allclose(probs_arr, probs_exp, atol=1e-7):
            errors.append(
                f"[{desc}] probs mismatch: expected shape {probs_exp.shape}, got {probs_arr.shape}."
            )

        for k, exp_v in exp_map.items():
            g_v = np.asarray(cache_got[k], dtype=float)
            if g_v.shape != exp_v.shape or not np.allclose(g_v, exp_v, atol=1e-7):
                errors.append(f"[{desc}] cache['{k}'] mismatch (expected shape {exp_v.shape}, got {g_v.shape}).")

    if errors:
        return CheckResult(
            "ch27_todo_04",
            False,
            "failed",
            "To-Do 27.4 — Check Forward Matrix Multiplications, ReLU & Cache Keys",
            "Compute `Z1 = X @ W1 + b1`, `A1 = np.maximum(0.0, Z1)`, `Z2 = A1 @ W2 + b2`, `probs = 1 / (1 + np.exp(-np.clip(Z2, -500, 500)))`, and return `(probs, {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'probs': probs})`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch27_todo_04",
        True,
        "passed",
        "To-Do 27.4 Passed — Two-Layer MLP Forward Pass & Cache Verified!",
        "All matrix shapes, ReLU zeroing, clipped sigmoid outputs, and cache tensors (`X, Z1, A1, Z2, probs`) matched exactly.",
        tip="Caching `X`, `Z1`, `A1`, `Z2`, and `probs` during the forward pass is what lets reverse-mode autodiff (and PyTorch!) compute backprop without recomputing layers.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_todo_04",
        chapter="ch27",
        title="To-Do 27.4: Two-Layer MLP Forward Pass (mlp_forward_two_layer)",
        book_section="Section 27.3.4 (Two-Layer MLP Forward Pass & Caching)",
        points=10,
        hints={
            1: "Compute hidden pre-activations `Z1 = X @ W1 + b1` and ReLU activations `A1 = np.maximum(0.0, Z1)`.",
            2: "Compute output logits `Z2 = A1 @ W2 + b2` and clipped sigmoid `probs = 1.0 / (1.0 + np.exp(-np.clip(Z2, -500.0, 500.0)))`.",
            3: "def mlp_forward_two_layer(X, W1, b1, W2, b2):\n    X_a = np.asarray(X, dtype=float)\n    Z1 = X_a @ W1 + b1\n    A1 = np.maximum(0.0, Z1)\n    Z2 = A1 @ W2 + b2\n    probs = 1.0 / (1.0 + np.exp(-np.clip(Z2, -500.0, 500.0)))\n    cache = {'X': X_a, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'probs': probs}\n    return (probs, ___)",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 27.5: mlp_backward_two_layer(y_true, cache, W2)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch27_todo_05", "To-Do 27.5", fn, "mlp_backward_two_layer", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(2705)
    X = rng.normal(size=(10, 3))
    W1 = rng.normal(0.0, 0.8, size=(3, 5))
    b1 = rng.normal(0.0, 0.2, size=(1, 5))
    W2 = rng.normal(0.0, 0.8, size=(5, 1))
    b2 = np.array([[0.15]])
    y_1d = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=float)
    y_2d = y_1d.reshape(-1, 1)

    # Ensure no Z1 entry sits right on the non-differentiable ReLU kink 0.0
    Z1 = X @ W1 + b1
    if np.any(np.abs(Z1) < 1e-3):
        b1 = b1 + 0.05
        Z1 = X @ W1 + b1
    A1 = np.maximum(0.0, Z1)
    Z2 = A1 @ W2 + b2
    probs = 1.0 / (1.0 + np.exp(-np.clip(Z2, -500.0, 500.0)))
    base_cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "probs": probs}

    def _bce_loss(w1_: Any, b1_: Any, w2_: Any, b2_: Any) -> float:
        a1_ = np.maximum(0.0, X @ w1_ + b1_)
        z2_ = a1_ @ w2_ + b2_
        p_ = np.clip(1.0 / (1.0 + np.exp(-np.clip(z2_, -500.0, 500.0))), 1e-12, 1.0 - 1e-12)
        return -float(np.mean(y_2d * np.log(p_) + (1.0 - y_2d) * np.log(1.0 - p_)))

    def _num_grad(param: Any, fn_p: Any, eps: float = 1e-6) -> Any:
        arr = param.copy()
        g = np.zeros_like(arr)
        for idx in np.ndindex(arr.shape):
            orig = arr[idx]
            arr[idx] = orig + eps
            lp = fn_p(arr)
            arr[idx] = orig - eps
            lm = fn_p(arr)
            arr[idx] = orig
            g[idx] = (lp - lm) / (2.0 * eps)
        return g

    num_grads = {
        "dW1": _num_grad(W1, lambda w: _bce_loss(w, b1, W2, b2)),
        "db1": _num_grad(b1, lambda b: _bce_loss(W1, b, W2, b2)),
        "dW2": _num_grad(W2, lambda w: _bce_loss(W1, b1, w, b2)),
        "db2": _num_grad(b2, lambda b: _bce_loss(W1, b1, W2, b)),
    }

    expected_keys = ["dW1", "db1", "dW2", "db2"]
    cases = [
        (y_2d, "2D column target y_true shape (n, 1)"),
        (y_1d, "1D vector target y_true shape (n,) — catches (n,1)-(n,) broadcasting bug"),
    ]
    errors = []

    for y_in, desc in cases:
        y_c = y_in.copy()
        W2_c = W2.copy()
        cache_c = {k: v.copy() for k, v in base_cache.items()}
        try:
            got = fn(y_c, cache_c, W2_c)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_todo_05",
                False,
                "failed",
                "To-Do 27.5 — Runtime Error in `mlp_backward_two_layer`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=[
                    "Always reshape `y_col = np.asarray(y_true, dtype=float).reshape(-1, 1)` before computing `dZ2 = (probs - y_col) / n`!"
                ],
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch27_todo_05", "To-Do 27.5", "mlp_backward_two_layer(y_true, cache, W2)", 10
            )

        key_err = _check_dict_keys("ch27_todo_05", "To-Do 27.5", got, expected_keys, 10)
        if key_err is not None:
            return key_err

        if not np.array_equal(y_c, y_in) or not np.array_equal(W2_c, W2):
            errors.append(f"[{desc}] Mutated caller's `y_true` or `W2`!")

        for k in expected_keys:
            g_got = np.asarray(got[k], dtype=float)
            g_num = num_grads[k]
            if g_got.shape != g_num.shape:
                extra = ""
                if k in ("db1", "db2") and g_got.ndim == 1:
                    extra = " (Pass `keepdims=True` inside `np.sum(..., axis=0, keepdims=True)`!)"
                elif k == "dW2" and g_got.shape == (5, 10):
                    extra = " (Looks like `(n, 1) - (n,)` broadcast into `(n, n)`! Reshape `y_true.reshape(-1, 1)`.)"
                errors.append(f"[{desc}] {k} shape expected {g_num.shape}, got {g_got.shape}.{extra}")
                continue
            max_diff = float(np.max(np.abs(g_got - g_num)))
            if max_diff >= 1e-5:
                errors.append(
                    f"[{desc}] {k} disagrees with numerical finite-difference gradient (max |diff| = {max_diff:.2e} >= 1e-5)."
                )

    if errors:
        return CheckResult(
            "ch27_todo_05",
            False,
            "failed",
            "To-Do 27.5 — Check Chain-Rule Backpropagation Shapes & Finite-Difference Agreement",
            "Reshape `y_col = y_true.reshape(-1, 1)`, compute `dZ2 = (probs - y_col) / n`, `dW2 = A1.T @ dZ2`, `db2 = sum(dZ2, axis=0, keepdims=True)`, `dZ1 = (dZ2 @ W2.T) * (Z1 > 0)`, `dW1 = X.T @ dZ1`, `db1 = sum(dZ1, axis=0, keepdims=True)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch27_todo_05",
        True,
        "passed",
        "To-Do 27.5 Passed — Analytical Backpropagation Matched Finite Differences (< 1e-5)!",
        "Verified `dW1`, `db1`, `dW2`, `db2` shapes and numerical gradient agreement on both 1D and 2D targets.",
        tip="Matching analytical backprop against centered finite differences (`max |diff| < 1e-5`) is the gold-standard sanity check in deep learning research!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_todo_05",
        chapter="ch27",
        title="To-Do 27.5: Analytical Two-Layer Backpropagation (mlp_backward_two_layer)",
        book_section="Section 27.3.5 (Chain-Rule Backpropagation From Scratch)",
        points=10,
        hints={
            1: "Start with `y_col = np.asarray(y_true, dtype=float).reshape(-1, 1)` and `n = len(y_col)`. Compute `dZ2 = (cache['probs'] - y_col) / n`.",
            2: "Output layer: `dW2 = cache['A1'].T @ dZ2`, `db2 = np.sum(dZ2, axis=0, keepdims=True)`. Hidden layer: `dZ1 = (dZ2 @ W2.T) * (cache['Z1'] > 0).astype(float)`, `dW1 = cache['X'].T @ dZ1`, `db1 = np.sum(dZ1, axis=0, keepdims=True)`.",
            3: "def mlp_backward_two_layer(y_true, cache, W2):\n    y_col = np.asarray(y_true, dtype=float).reshape(-1, 1)\n    n = len(y_col)\n    dZ2 = (cache['probs'] - y_col) / n\n    dW2 = cache['A1'].T @ dZ2\n    db2 = np.sum(dZ2, axis=0, keepdims=True)\n    dZ1 = (dZ2 @ np.asarray(W2, dtype=float).T) * (cache['Z1'] > 0).astype(float)\n    dW1 = cache['X'].T @ dZ1\n    db1 = np.sum(dZ1, axis=0, keepdims=True)\n    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': ___}",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 27.6: numerical_gradient_check(loss_fn, W, eps=1e-5)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch27_todo_06", "To-Do 27.6", fn, "numerical_gradient_check", 10)
    if bad is not None:
        return bad

    coeffs = np.array([[1.5, -2.0, 0.5], [3.0, 0.25, -1.2]])
    W_quad = np.array([[0.4, -1.1, 2.0], [-0.5, 0.8, 1.3]])
    w_trig = np.array([0.2, -0.7, 1.1, 2.3])

    cases = [
        (
            lambda w: float(np.sum(coeffs * (w ** 2))),
            W_quad,
            {},
            2.0 * coeffs * W_quad,
            "2D weighted quadratic loss (exact grad = 2 * C * W)",
        ),
        (
            lambda w: float(np.sum(np.sin(w))),
            w_trig,
            {"eps": 1e-6},
            np.cos(w_trig),
            "1D trigonometric loss with custom eps=1e-6",
        ),
    ]
    errors = []

    for loss_fn, W_in, kwargs, exp_grad, desc in cases:
        W_copy = W_in.copy()
        try:
            got = fn(loss_fn, W_copy, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_todo_06",
                False,
                "failed",
                "To-Do 27.6 — Runtime Error in `numerical_gradient_check`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch27_todo_06", "To-Do 27.6", "numerical_gradient_check(loss_fn, W)", 10
            )

        if not np.array_equal(W_copy, W_in):
            errors.append(f"[{desc}] Mutated caller's parameter array `W`! Perturb a `.copy()` of `W`.")

        g_arr = np.asarray(got, dtype=float)
        if g_arr.shape != exp_grad.shape:
            errors.append(f"[{desc}] Shape mismatch: expected {exp_grad.shape}, got {g_arr.shape}.")
            continue
        max_diff = float(np.max(np.abs(g_arr - exp_grad)))
        if max_diff >= 1e-6:
            extra = ""
            if np.allclose(g_arr, 2.0 * exp_grad, atol=1e-4):
                extra = " (Looks off by 2x — divide by `2 * eps`, not `eps`!)"
            errors.append(
                f"[{desc}] Gradient mismatch (max |diff| = {max_diff:.2e} >= 1e-6).{extra}"
            )

    if errors:
        return CheckResult(
            "ch27_todo_06",
            False,
            "failed",
            "To-Do 27.6 — Check Centered Finite-Difference Perturbation & Non-Mutation",
            "Copy `W`, loop over `np.ndindex(W_copy.shape)`, evaluate `(loss_fn(W_plus) - loss_fn(W_minus)) / (2 * eps)`, restore `W_copy[idx]`, and return `grad`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch27_todo_06",
        True,
        "passed",
        "To-Do 27.6 Passed — Centered Numerical Gradient Checker Verified!",
        "Matched exact analytical derivatives to < 1e-6 on 1D and 2D tensors without mutating `W`.",
        tip="Centered differences `(f(x+e) - f(x-e)) / (2e)` have O(eps^2) error—orders of magnitude tighter than one-sided differences!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_todo_06",
        chapter="ch27",
        title="To-Do 27.6: Centered Finite-Difference Gradient Check (numerical_gradient_check)",
        book_section="Section 27.3.5 (Numerical Gradient Verification)",
        points=10,
        hints={
            1: "Create `W_work = np.asarray(W, dtype=float).copy()` and `grad = np.zeros_like(W_work)` so you never mutate the caller's `W`.",
            2: "Loop `for idx in np.ndindex(W_work.shape):`, save `orig = W_work[idx]`, evaluate `loss_fn` at `orig + eps` and `orig - eps`, restore `W_work[idx] = orig`, and store `(lp - lm) / (2.0 * eps)`.",
            3: "def numerical_gradient_check(loss_fn, W, eps=1e-5):\n    W_work = np.asarray(W, dtype=float).copy()\n    grad = np.zeros_like(W_work)\n    for idx in np.ndindex(W_work.shape):\n        orig = W_work[idx]\n        W_work[idx] = orig + eps\n        lp = float(loss_fn(W_work))\n        W_work[idx] = orig - eps\n        lm = float(loss_fn(W_work))\n        W_work[idx] = orig\n        grad[idx] = (lp - lm) / (___ * eps)\n    return grad",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 27.1 (30 pts): solve_xor_mlp_numpy(seed=42, epochs=2500, lr=0.5)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    bad = _check_callable("ch27_exam_01", "Exam Task 27.1", fn, "solve_xor_mlp_numpy", 30)
    if bad is not None:
        return bad

    cases = [
        ({}, "default seed=42, epochs=2500, lr=0.5"),
        ({"seed": 7, "epochs": 2500, "lr": 0.5}, "alternate seed=7"),
    ]
    expected_keys = ["accuracy", "final_loss", "preds"]
    errors = []

    for kwargs, desc in cases:
        try:
            got = fn(**kwargs)
            got_repeat = fn(**kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_exam_01",
                False,
                "failed",
                "Exam Task 27.1 — Runtime Error in `solve_xor_mlp_numpy`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch27_exam_01", "Exam Task 27.1", "solve_xor_mlp_numpy()", 30
            )

        key_err = _check_dict_keys("ch27_exam_01", "Exam Task 27.1", got, expected_keys, 30)
        if key_err is not None:
            return key_err

        if not _is_number(got["accuracy"]) or not math.isclose(float(got["accuracy"]), 1.0, abs_tol=1e-9):
            errors.append(f"[{desc}] accuracy expected 1.0, got {got['accuracy']!r}.")

        if not _is_number(got["final_loss"]) or not (0.0 <= float(got["final_loss"]) < 0.1):
            errors.append(
                f"[{desc}] final_loss expected in [0.0, 0.1), got {got['final_loss']!r}."
            )
        else:
            if not math.isclose(float(got["final_loss"]), round(float(got["final_loss"]), 5), abs_tol=1e-9):
                errors.append(f"[{desc}] final_loss should be rounded to 5 decimals, got {got['final_loss']!r}.")

        preds = got["preds"]
        if not isinstance(preds, list) or len(preds) != 4 or not all(_is_int_like(v) for v in preds):
            errors.append(f"[{desc}] preds must be a list of 4 ints, got {preds!r}.")
        elif [int(v) for v in preds] != [0, 1, 1, 0]:
            errors.append(f"[{desc}] preds expected [0, 1, 1, 0], got {preds!r}.")

        if got_repeat != got:
            errors.append(
                f"[{desc}] Function is not deterministic for the same seed! Use `np.random.default_rng(seed)`."
            )

    if errors:
        return CheckResult(
            "ch27_exam_01",
            False,
            "failed",
            "Exam Task 27.1 — Check Pure-NumPy 2->8->1 XOR MLP Training",
            "Train a seeded 2->8->1 MLP (`np.random.default_rng(seed)`) on XOR `[[0,0],[0,1],[1,0],[1,1]] -> [0,1,1,0]` with tanh or ReLU hidden units until accuracy == 1.0, preds == [0, 1, 1, 0], and final_loss < 0.1.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch27_exam_01",
        True,
        "passed",
        "Exam Task 27.1 Passed (30/30 pts) — Pure-NumPy MLP Solved XOR Cleanly!",
        "Achieved 100% XOR accuracy ([0, 1, 1, 0]) and BCE loss < 0.1 deterministically across multiple seeds.",
        points_earned=30,
        points_possible=30,
        tip="A single linear hyperplane can never separate XOR (max 75% accuracy), whereas adding one nonlinear hidden layer bends the feature space to solve it effortlessly!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_exam_01",
        chapter="ch27",
        title="Exam 27.1: Solving XOR with a Pure-NumPy 2-Layer MLP (solve_xor_mlp_numpy)",
        book_section="Chapter 27 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Define `X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])`, `y = np.array([[0.],[1.],[1.],[0.]])`, and seed `rng = np.random.default_rng(seed)` with `W1 (2, 8)`, `b1 (1, 8)`, `W2 (8, 1)`, `b2 (1, 1)`.",
            2: "Using `A1 = np.tanh(X @ W1 + b1)` avoids dying ReLUs on 4 points: backprop uses `dZ2 = (probs - y) / 4`, `dW2 = A1.T @ dZ2`, `db2 = sum(dZ2, axis=0, keepdims=True)`, `dZ1 = (dZ2 @ W2.T) * (1 - A1**2)`, `dW1 = X.T @ dZ1`, `db1 = sum(dZ1, axis=0, keepdims=True)`.",
            3: "After `epochs` gradient steps, run a final forward pass to compute clipped BCE `final_loss = round(loss, 5)`, `preds = [int(v >= 0.5) for v in probs.ravel()]`, and `accuracy = float(np.mean(np.array(preds) == y.ravel()))`.",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 27.2 (35 pts): train_numpy_mlp_classifier(...)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.datasets import make_moons
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    bad = _check_callable(
        "ch27_exam_02", "Exam Task 27.2", fn, "train_numpy_mlp_classifier", 35
    )
    if bad is not None:
        return bad

    X_m, y_m = make_moons(n_samples=320, noise=0.18, random_state=42)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_m, y_m, test_size=0.25, stratify=y_m, random_state=42
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    cases = [
        (X_tr_s, y_tr, X_va_s, y_va, {}, 16, 300, "nonlinear make_moons default (hidden_dim=16, epochs=300)"),
        (
            X_tr_s,
            y_tr,
            X_va_s,
            y_va,
            {"hidden_dim": 24, "lr": 0.12, "epochs": 250, "seed": 7},
            24,
            250,
            "custom hidden_dim=24, epochs=250, seed=7",
        ),
    ]
    expected_keys = ["train_losses", "val_losses", "val_roc_auc", "params"]
    param_keys = ["W1", "b1", "W2", "b2"]
    errors = []

    for Xtr, ytr, Xva, yva, kwargs, h_dim, n_ep, desc in cases:
        Xtr_c, ytr_c, Xva_c, yva_c = Xtr.copy(), ytr.copy(), Xva.copy(), yva.copy()
        try:
            got = fn(Xtr_c, ytr_c, Xva_c, yva_c, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_exam_02",
                False,
                "failed",
                "Exam Task 27.2 — Runtime Error in `train_numpy_mlp_classifier`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch27_exam_02", "Exam Task 27.2", "train_numpy_mlp_classifier(...)", 35
            )

        key_err = _check_dict_keys("ch27_exam_02", "Exam Task 27.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not np.array_equal(Xtr_c, Xtr) or not np.array_equal(Xva_c, Xva):
            errors.append(f"[{desc}] Mutated caller's input feature arrays!")

        tr_l, va_l = got["train_losses"], got["val_losses"]
        if not isinstance(tr_l, list) or len(tr_l) != n_ep:
            errors.append(f"[{desc}] train_losses must be a list of length {n_ep}, got {type(tr_l).__name__} len={len(tr_l) if isinstance(tr_l, list) else '?'}.")
            continue
        if not isinstance(va_l, list) or len(va_l) != n_ep:
            errors.append(f"[{desc}] val_losses must be a list of length {n_ep}.")
            continue
        if not all(_is_number(v) and math.isfinite(float(v)) for v in tr_l + va_l):
            errors.append(f"[{desc}] Loss histories contain non-finite values.")
            continue
        if float(tr_l[-1]) >= float(tr_l[0]):
            errors.append(
                f"[{desc}] Training loss did not decrease: start={tr_l[0]:.4f}, end={tr_l[-1]:.4f}."
            )

        params = got["params"]
        p_err = _check_dict_keys("ch27_exam_02", "Exam Task 27.2 (params)", params, param_keys, 35)
        if p_err is not None:
            return p_err

        d_in = Xtr.shape[1]
        W1 = np.asarray(params["W1"], dtype=float)
        b1 = np.asarray(params["b1"], dtype=float)
        W2 = np.asarray(params["W2"], dtype=float)
        b2 = np.asarray(params["b2"], dtype=float)

        if W1.shape != (d_in, h_dim) or W2.shape != (h_dim, 1):
            errors.append(
                f"[{desc}] Weight shapes expected W1={(d_in, h_dim)}, W2={(h_dim, 1)}, got W1={W1.shape}, W2={W2.shape}."
            )
            continue
        if b1.shape not in ((1, h_dim), (h_dim,)) or b2.shape not in ((1, 1), (1,)):
            errors.append(f"[{desc}] Bias shapes expected b1={(1, h_dim)}, b2={(1, 1)}, got b1={b1.shape}, b2={b2.shape}.")
            continue

        # Verify reported val_roc_auc matches actual forward pass of returned params
        A1_v = np.maximum(0.0, Xva @ W1 + b1)
        Z2_v = A1_v @ W2 + b2
        p_v = 1.0 / (1.0 + np.exp(-np.clip(Z2_v, -500.0, 500.0)))
        actual_auc = round(float(roc_auc_score(yva, p_v.ravel())), 4)

        if not _is_number(got["val_roc_auc"]):
            errors.append(f"[{desc}] val_roc_auc must be a float, got {got['val_roc_auc']!r}.")
            continue
        if float(got["val_roc_auc"]) < 0.85:
            errors.append(
                f"[{desc}] val_roc_auc = {float(got['val_roc_auc']):.4f} < 0.85 on nonlinear moons! Check He initialization `sqrt(2 / d_in)` and backprop updates."
            )
        if not math.isclose(float(got["val_roc_auc"]), actual_auc, abs_tol=1e-3):
            errors.append(
                f"[{desc}] Reported val_roc_auc ({got['val_roc_auc']}) does not match forward pass of returned `params` ({actual_auc})."
            )

    if errors:
        return CheckResult(
            "ch27_exam_02",
            False,
            "failed",
            "Exam Task 27.2 — Check He-Initialized ReLU MLP Training Loop & Validation AUC",
            "Initialize W1 with He scaling `rng.normal(0, np.sqrt(2/d_in), (d_in, hidden_dim))`, train with ReLU + BCE backprop for `epochs`, track train/val losses, and return `val_roc_auc` (round4) and `params`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch27_exam_02",
        True,
        "passed",
        "Exam Task 27.2 Passed (35/35 pts) — Pure-NumPy ReLU MLP Classifier Verified!",
        "Trained a He-initialized 2-layer ReLU MLP from scratch, decreased BCE loss cleanly, and achieved > 0.85 ROC-AUC on nonlinear moons.",
        points_earned=35,
        points_possible=35,
        tip="He initialization `std = sqrt(2 / d_in)` keeps ReLU hidden activations well-scaled across layers—preventing both saturation and vanishing gradients!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_exam_02",
        chapter="ch27",
        title="Exam 27.2: Pure-NumPy 2-Layer ReLU MLP Classifier (train_numpy_mlp_classifier)",
        book_section="Chapter 27 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Use `rng = np.random.default_rng(seed)` with He init: `W1 = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, hidden_dim))`, `b1 = np.zeros((1, hidden_dim))`, `W2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1))`, `b2 = np.zeros((1, 1))`.",
            2: "Inside `for _ in range(epochs):`, compute forward pass and clipped BCE on both `X_train` and `X_val`, append to `train_losses` / `val_losses`, compute backprop gradients on training data only, and subtract `lr * grad`.",
            3: "After training, run forward on `X_val` to get `p_val`, compute `val_roc_auc = round(float(roc_auc_score(y_val, p_val.ravel())), 4)`, and return `{'train_losses': train_losses, 'val_losses': val_losses, 'val_roc_auc': val_roc_auc, 'params': {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}}`.",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 27.3 (35 pts): compare_numpy_mlp_vs_logistic_mch(mch_df, seed=42)
# ---------------------------------------------------------------------------
def _ref_compare_mch(mch_df: Any, seed: int = 42) -> dict[str, float]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    work = mch_df.copy()
    work["anc_visits"] = work["anc_visits"].replace(999, np.nan)
    work["residence"] = work["residence"].astype(str).str.strip().str.title()
    work["is_urban"] = (work["residence"] == "Urban").astype(int)
    wmap = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
    work["wealth_num"] = (
        work["wealth_quintile"].astype(str).str.strip().str.title().map(wmap)
    )
    feats = [
        "anc_visits",
        "maternal_hb_g_dl",
        "maternal_age_years",
        "is_urban",
        "wealth_num",
    ]
    clean = work[feats + ["child_stunted"]].dropna()
    X = clean[feats].to_numpy(dtype=float)
    y = clean["child_stunted"].to_numpy(dtype=int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    log_reg = LogisticRegression(max_iter=500)
    log_reg.fit(X_tr_s, y_tr)
    log_auc = round(float(roc_auc_score(y_te, log_reg.predict_proba(X_te_s)[:, 1])), 4)

    # Pure-NumPy 2-layer ReLU MLP
    rng = np.random.default_rng(seed)
    n, d_in = X_tr_s.shape
    h = 16
    W1 = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, h))
    b1 = np.zeros((1, h), dtype=float)
    W2 = rng.normal(0.0, np.sqrt(2.0 / h), size=(h, 1))
    b2 = np.zeros((1, 1), dtype=float)
    y_col = y_tr.astype(float).reshape(-1, 1)
    lr = 0.1
    final_loss = 0.0
    for _ in range(400):
        Z1 = X_tr_s @ W1 + b1
        A1 = np.maximum(0.0, Z1)
        Z2 = A1 @ W2 + b2
        probs = 1.0 / (1.0 + np.exp(-np.clip(Z2, -500.0, 500.0)))
        p_clip = np.clip(probs, 1e-12, 1.0 - 1e-12)
        final_loss = -float(np.mean(y_col * np.log(p_clip) + (1.0 - y_col) * np.log(1.0 - p_clip)))
        dZ2 = (probs - y_col) / n
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dZ1 = (dZ2 @ W2.T) * (Z1 > 0).astype(float)
        dW1 = X_tr_s.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2

    A1_te = np.maximum(0.0, X_te_s @ W1 + b1)
    p_te = 1.0 / (1.0 + np.exp(-np.clip(A1_te @ W2 + b2, -500.0, 500.0)))
    mlp_auc = round(float(roc_auc_score(y_te, p_te.ravel())), 4)
    return {
        "mlp_auc": mlp_auc,
        "logistic_auc": log_auc,
        "mlp_final_loss": round(final_loss, 5),
    }


def _verify_exam_03(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd

    from ..datasets import load_data

    bad = _check_callable(
        "ch27_exam_03", "Exam Task 27.3", fn, "compare_numpy_mlp_vs_logistic_mch", 35
    )
    if bad is not None:
        return bad

    df_mch = load_data("mch_survey_sample", as_pandas=True)

    # Synthetic messy survey with nonlinear stunting interaction
    rng = np.random.default_rng(2703)
    n_syn = 240
    res_raw = rng.choice(["Urban", "urban", "Rural", " rural "], size=n_syn)
    w_raw = rng.choice(["Poorest", "Poorer", "Middle", "Richer", "Richest"], size=n_syn)
    w_map = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}
    w_num = np.array([w_map[w] for w in w_raw], dtype=float)
    anc_raw = rng.integers(0, 9, size=n_syn).astype(float)
    hb_raw = rng.normal(11.3, 1.4, size=n_syn)
    age_raw = rng.integers(18, 42, size=n_syn).astype(float)
    is_u = np.array([s.strip().title() == "Urban" for s in res_raw], dtype=float)
    logit_syn = -0.8 - 0.55 * (anc_raw - 4.0) - 0.45 * (hb_raw - 11.3) - 0.35 * (w_num - 3.0) + 0.35 * ((anc_raw - 4.0) * (w_num - 3.0))
    prob_syn = 1.0 / (1.0 + np.exp(-logit_syn))
    stunted = (rng.random(n_syn) < prob_syn).astype(int)
    anc_raw[[2, 19, 64]] = 999.0
    hb_raw[[5, 31, 90]] = np.nan
    df_syn = pd.DataFrame(
        {
            "residence": res_raw,
            "wealth_quintile": w_raw,
            "anc_visits": anc_raw,
            "maternal_hb_g_dl": hb_raw,
            "maternal_age_years": age_raw,
            "child_stunted": stunted,
        }
    )

    cases = [
        (df_mch, {}, 42, "real mch_survey_sample (seed=42)"),
        (df_mch, {"seed": 7}, 7, "real mch_survey_sample (seed=7)"),
        (df_syn, {"seed": 42}, 42, "synthetic messy cohort (casing + 999 + NaN)"),
    ]
    expected_keys = ["mlp_auc", "logistic_auc", "mlp_final_loss"]
    errors = []

    for df_in, kwargs, seed_val, desc in cases:
        df_copy = df_in.copy(deep=True)
        try:
            got = fn(df_copy, **kwargs)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch27_exam_03",
                False,
                "failed",
                "Exam Task 27.3 — Runtime Error in `compare_numpy_mlp_vs_logistic_mch`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=[
                    "Clean `anc_visits == 999 -> np.nan`, title-case `residence` -> `is_urban`, map `wealth_quintile` to 1..5, and drop NaNs on features + `child_stunted` before stratified 75/25 split!"
                ],
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch27_exam_03", "Exam Task 27.3", "compare_numpy_mlp_vs_logistic_mch(mch_df)", 35
            )

        key_err = _check_dict_keys("ch27_exam_03", "Exam Task 27.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not df_copy.equals(df_in):
            errors.append(f"[{desc}] Input DataFrame was mutated! Start with `work = mch_df.copy()`.")

        exp = _ref_compare_mch(df_in, seed=seed_val)

        if not _is_number(got["logistic_auc"]) or not math.isclose(
            float(got["logistic_auc"]), exp["logistic_auc"], abs_tol=1e-3
        ):
            errors.append(
                f"[{desc}] logistic_auc expected {exp['logistic_auc']}, got {got['logistic_auc']!r}. "
                "Verify exact 5 features, NaN cleanup, stratified 75/25 split, train-only StandardScaler, and LogisticRegression(max_iter=500)."
            )

        if not _is_number(got["mlp_auc"]) or not (0.45 <= float(got["mlp_auc"]) <= 1.0):
            errors.append(f"[{desc}] mlp_auc out of valid range: {got['mlp_auc']!r}.")
        elif "synthetic" in desc and float(got["mlp_auc"]) < 0.72:
            errors.append(
                f"[{desc}] mlp_auc ({float(got['mlp_auc']):.4f}) too low on synthetic nonlinear cohort (expected >= 0.72). Verify MLP training loop!"
            )

        if not _is_number(got["mlp_final_loss"]) or not (0.0 < float(got["mlp_final_loss"]) < 0.65):
            errors.append(
                f"[{desc}] mlp_final_loss ({got['mlp_final_loss']!r}) should be strictly between 0 and 0.65 after training."
            )

    if errors:
        return CheckResult(
            "ch27_exam_03",
            False,
            "failed",
            "Exam Task 27.3 — Check MCH Cleaning, Stratified 75/25 Split, Scaling & MLP vs. Logistic Benchmark",
            "Clean `anc_visits == 999`, title-case `residence` -> `is_urban`, map `wealth_quintile` -> 1..5, drop NaNs, split 75/25 stratified (`random_state=seed`), fit train-only `StandardScaler`, and compare pure-NumPy 2-layer MLP vs `LogisticRegression(max_iter=500)`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch27_exam_03",
        True,
        "passed",
        "Exam Task 27.3 Passed (35/35 pts) — MCH Stunting Pure-NumPy MLP vs. Logistic Benchmark Verified!",
        "Cleaned messy MCH survey data without mutation, applied leak-free scaling on a stratified 75/25 split, and benchmarked your from-scratch NumPy MLP against LogisticRegression.",
        points_earned=35,
        points_possible=35,
        tip="Always benchmark a neural network against a strong linear baseline (`LogisticRegression`) on the exact same leak-free split before adding complexity!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch27_exam_03",
        chapter="ch27",
        title="Exam 27.3: MCH Stunting Benchmark — Pure-NumPy MLP vs. LogisticRegression (compare_numpy_mlp_vs_logistic_mch)",
        book_section="Chapter 27 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Start with `work = mch_df.copy()`. Replace `anc_visits == 999` with `np.nan`, title-case `residence` to create `is_urban`, map `wealth_quintile` (`Poorest`..`Richest`) to `1..5`, and drop NaNs on the 5 features + `child_stunted`.",
            2: "Run `train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)`, fit `StandardScaler()` strictly on `X_train` (`fit_transform(X_train)`, `transform(X_test)`), and fit `LogisticRegression(max_iter=500)` to score `logistic_auc`.",
            3: "Train a He-initialized pure-NumPy 2-layer ReLU MLP (e.g. `hidden_dim=16, lr=0.1, epochs=400, seed=seed`) on `(X_tr_s, y_tr)`, score `mlp_auc` on `(X_te_s, y_te)`, and return `{'mlp_auc': round(mlp_auc, 4), 'logistic_auc': round(log_auc, 4), 'mlp_final_loss': round(final_loss, 5)}`.",
        },
        verifier=_verify_exam_03,
    )
)
