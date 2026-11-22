"""Chapter 28 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch28_todo_01..06) and the Chapter 28 Coding Exam
(ch28_exam_01..03) for PyTorch Core — Tensors, Autograd, nn.Module,
Datasets, and Training Loops.
Every verifier lazy-imports PyTorch/NumPy/Scikit-Learn inside the check
function, enforces deterministic CPU seeds, validates tensor dtypes,
shapes, autograd behavior, non-aliasing, and safe weights_only=True
checkpointing inside TemporaryDirectory(), while letting
NotImplementedError propagate cleanly for starter stubs.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


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


def _check_class(
    ex_id: str,
    title: str,
    cls: Any,
    expected_name: str,
    base_cls: type,
    base_label: str,
    points_possible: int = 10,
) -> Optional[CheckResult]:
    if not isinstance(cls, type):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Pass the Class Itself (Not an Instance)",
            (
                f"Expected the class `{expected_name}`, "
                f"but received {type(cls).__name__} ({cls!r})."
            ),
            details=[
                f"Pass the class name directly: tutor.check('{ex_id}', {expected_name})",
                f"Do NOT instantiate `{expected_name}(...)` inside tutor.check().",
            ],
            points_earned=0,
            points_possible=points_possible,
        )
    if not issubclass(cls, base_cls):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — `{expected_name}` Must Subclass `{base_label}`",
            f"`{cls.__name__}` does not inherit from `{base_label}`.",
            details=[
                f"Define `class {expected_name}({base_label}):` so PyTorch recognizes it.",
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
        f"Calling {call_str} returned None instead of the expected PyTorch result.",
        details=[
            "Remember: print() only displays text on screen; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reusable PyTorch helpers always return tensors, metrics, or models rather than printing them.",
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
# GUIDED TO-DO 28.1: numpy_to_float_tensor(arr)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np
    import torch

    bad = _check_callable("ch28_todo_01", "To-Do 28.1", fn, "numpy_to_float_tensor", 10)
    if bad is not None:
        return bad

    cases = [
        (np.array([1.5, -2.25, 3.0], dtype=np.float64), "1D float64 NumPy array"),
        (np.array([[1, 2], [3, 4]], dtype=np.int64), "2D int64 NumPy array"),
        (np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32), "2D float32 NumPy array (aliasing check)"),
    ]
    errors = []

    for arr_orig, desc in cases:
        arr_work = arr_orig.copy()
        try:
            got = fn(arr_work)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch28_todo_01",
                False,
                "failed",
                "To-Do 28.1 — Runtime Error in `numpy_to_float_tensor`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch28_todo_01", "To-Do 28.1", "numpy_to_float_tensor(arr)", 10)

        if not isinstance(got, torch.Tensor):
            return CheckResult(
                "ch28_todo_01",
                False,
                "failed",
                "To-Do 28.1 — Expected a `torch.Tensor`",
                f"[{desc}] Expected a torch.Tensor, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(arr_work, arr_orig):
            errors.append(f"[{desc}] Mutated caller NumPy array during conversion!")

        if got.dtype != torch.float32:
            errors.append(
                f"[{desc}] Expected dtype `torch.float32`, got `{got.dtype}` "
                "(Pass `dtype=torch.float32` so `nn.Linear` layers never hit a float64 mismatch!)."
            )
            continue

        if got.requires_grad:
            errors.append(
                f"[{desc}] Returned tensor must be detached (`requires_grad=False`), got `requires_grad=True`."
            )

        if tuple(got.shape) != tuple(arr_orig.shape):
            errors.append(f"[{desc}] Shape mismatch: expected {arr_orig.shape}, got {tuple(got.shape)}.")
            continue

        expected_vals = arr_orig.astype(np.float32)
        if not np.allclose(got.detach().cpu().numpy(), expected_vals, atol=1e-6):
            errors.append(f"[{desc}] Values do not match input NumPy array.")
            continue

        # Verify no shared memory / aliasing by mutating arr_work AFTER conversion
        arr_work.flat[0] = 9999.0
        if not np.allclose(got.detach().cpu().numpy(), expected_vals, atol=1e-6):
            errors.append(
                f"[{desc}] Memory aliasing detected! Mutating the input NumPy array changed the tensor. "
                "Use `torch.tensor(arr, dtype=torch.float32)` (which always copies) instead of `torch.from_numpy` or `torch.as_tensor`."
            )

    if errors:
        return CheckResult(
            "ch28_todo_01",
            False,
            "failed",
            "To-Do 28.1 — Check `torch.float32` Copy & Non-Aliasing",
            "Return a detached `torch.float32` copy via `torch.tensor(arr, dtype=torch.float32)` that does not share memory with `arr`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch28_todo_01",
        True,
        "passed",
        "To-Do 28.1 Passed — Safe `torch.float32` Tensor Copy Verified!",
        "Converted float64, int64, and float32 NumPy arrays into detached `torch.float32` tensors with zero memory aliasing.",
        tip="`torch.tensor(arr, dtype=torch.float32)` always copies data safely, whereas `torch.from_numpy(arr)` shares memory with `float32` NumPy buffers.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_todo_01",
        chapter="ch28",
        title="To-Do 28.1: Safe NumPy-to-Float32 Tensor Copy (numpy_to_float_tensor)",
        book_section="Section 28.3.1 (Tensors vs. NumPy, float32 & Aliasing)",
        points=10,
        hints={
            1: "PyTorch layers default to `torch.float32`, while NumPy defaults to `float64`. You need a constructor that explicitly casts to `torch.float32` AND copies memory.",
            2: "Avoid `torch.from_numpy(arr)` and `torch.as_tensor(arr)` because both share memory when `arr` is already `np.float32`. Use `torch.tensor(..., dtype=torch.float32)`.",
            3: "def numpy_to_float_tensor(arr):\n    return torch.tensor(arr, dtype=torch.___)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 28.2: autograd_quadratic_grad(x_val)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import torch

    bad = _check_callable("ch28_todo_02", "To-Do 28.2", fn, "autograd_quadratic_grad", 10)
    if bad is not None:
        return bad

    cases = [
        (2.0, 9.0, 8.0, "x = 2.0"),
        (-1.5, 17.75, -13.0, "x = -1.5"),
        (0.0, 5.0, -4.0, "x = 0.0"),
        (0.75, 3.6875, 0.5, "x = 0.75"),
    ]
    errors = []

    for x_val, exp_y, exp_grad, desc in cases:
        try:
            got = fn(x_val)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch28_todo_02",
                False,
                "failed",
                "To-Do 28.2 — Runtime Error in `autograd_quadratic_grad`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap("ch28_todo_02", "To-Do 28.2", "autograd_quadratic_grad(x_val)", 10)

        if not isinstance(got, tuple) or len(got) != 2:
            return CheckResult(
                "ch28_todo_02",
                False,
                "failed",
                "To-Do 28.2 — Return a 2-Tuple `(y_val, grad_val)`",
                f"[{desc}] Expected a 2-tuple `(round(y.item(), 4), round(x.grad.item(), 4))`, got {type(got).__name__}.",
                points_earned=0,
                points_possible=10,
            )

        g_y, g_grad = got
        if isinstance(g_y, torch.Tensor) or isinstance(g_grad, torch.Tensor):
            errors.append(
                f"[{desc}] Returned a torch.Tensor inside the tuple! Call `.item()` to extract plain Python floats before rounding."
            )
            continue

        if not (_is_np_or_py_number(g_y) and _is_np_or_py_number(g_grad)):
            errors.append(f"[{desc}] Expected two floats, got {got!r}.")
            continue

        if not math.isclose(float(g_y), exp_y, abs_tol=1e-4) or not math.isclose(
            float(g_grad), exp_grad, abs_tol=1e-4
        ):
            errors.append(
                f"[{desc}] Expected ({exp_y}, {exp_grad}) for y = 3x^2 - 4x + 5 and dy/dx = 6x - 4, got {got!r}."
            )

    if errors:
        return CheckResult(
            "ch28_todo_02",
            False,
            "failed",
            "To-Do 28.2 — Check Autograd Forward Formula & `.backward()` Extraction",
            "Create `x = torch.tensor(float(x_val), requires_grad=True)`, compute `y = 3.0 * x**2 - 4.0 * x + 5.0`, call `y.backward()`, and return `(round(y.item(), 4), round(x.grad.item(), 4))`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch28_todo_02",
        True,
        "passed",
        "To-Do 28.2 Passed — PyTorch Autograd Derivative Verified!",
        "Exact forward values and reverse-mode gradients matched `y = 3x^2 - 4x + 5` and `dy/dx = 6x - 4` across all test points.",
        tip="Calling `.item()` detaches scalar metrics from the computation graph so Python logs don't accidentally keep GPU/CPU graph memory alive.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_todo_02",
        chapter="ch28",
        title="To-Do 28.2: Scalar Autograd Forward & Backward (autograd_quadratic_grad)",
        book_section="Section 28.3.2 (Autograd Computation Graph & .backward())",
        points=10,
        hints={
            1: "Wrap `float(x_val)` in `torch.tensor(..., requires_grad=True)` so PyTorch records operations on `x`.",
            2: "Evaluate `y = 3.0 * x**2 - 4.0 * x + 5.0`, invoke `y.backward()`, and extract plain Python numbers with `y.item()` and `x.grad.item()`.",
            3: "def autograd_quadratic_grad(x_val):\n    x = torch.tensor(float(x_val), requires_grad=True)\n    y = 3.0 * x**2 - 4.0 * x + 5.0\n    y.___()\n    return (round(y.item(), 4), round(x.grad.item(), 4))",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 28.3: ClinicalTabularDataset(Dataset)
# ---------------------------------------------------------------------------
def _verify_todo_03(cls: Any) -> CheckResult:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    bad = _check_class(
        "ch28_todo_03",
        "To-Do 28.3",
        cls,
        "ClinicalTabularDataset",
        Dataset,
        "torch.utils.data.Dataset",
        10,
    )
    if bad is not None:
        return bad

    rng = np.random.default_rng(42)
    X1 = rng.normal(size=(20, 4)).astype(np.float64)
    y1 = rng.integers(0, 2, size=(20,)).astype(np.int64)
    X2 = rng.normal(size=(12, 3)).astype(np.float32)
    y2 = rng.integers(0, 2, size=(12, 1)).astype(np.float64)

    cases = [
        (X1, y1, "20x4 float64 features with 1D int64 labels"),
        (X2, y2, "12x3 float32 features with 2D (n, 1) float64 labels"),
    ]
    errors = []

    for X_raw, y_raw, desc in cases:
        X_copy, y_copy = X_raw.copy(), y_raw.copy()
        try:
            ds = cls(X_copy, y_copy)
            n_len = len(ds)
            sample0 = ds[0]
            sample_last = ds[len(X_raw) - 1]
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch28_todo_03",
                False,
                "failed",
                "To-Do 28.3 — Runtime Error in `ClinicalTabularDataset`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if not np.array_equal(X_copy, X_raw) or not np.array_equal(y_copy, y_raw):
            errors.append(f"[{desc}] Mutated caller NumPy arrays!")

        if n_len != len(X_raw):
            errors.append(f"[{desc}] `len(ds)` returned {n_len}, expected {len(X_raw)}.")

        if not isinstance(sample0, tuple) or len(sample0) != 2:
            errors.append(f"[{desc}] `__getitem__` must return a 2-tuple `(x_row, y_row)`, got {type(sample0).__name__}.")
            continue

        x0, y0 = sample0
        xl, yl = sample_last
        if not (isinstance(x0, torch.Tensor) and isinstance(y0, torch.Tensor)):
            errors.append(f"[{desc}] `__getitem__` elements must both be `torch.Tensor`s.")
            continue

        if x0.dtype != torch.float32 or y0.dtype != torch.float32:
            errors.append(
                f"[{desc}] Expected `torch.float32` for both `x` and `y`, got ({x0.dtype}, {y0.dtype})."
            )
            continue

        p = X_raw.shape[1]
        if tuple(x0.shape) != (p,):
            errors.append(f"[{desc}] Expected `x` sample shape ({p},), got {tuple(x0.shape)}.")
        if tuple(y0.shape) != (1,):
            errors.append(
                f"[{desc}] Expected `y` sample shape (1,) (stored as (n, 1) for BCEWithLogitsLoss!), got {tuple(y0.shape)}."
            )

        if not np.allclose(x0.numpy(), X_raw[0].astype(np.float32), atol=1e-6):
            errors.append(f"[{desc}] First feature row `ds[0][0]` does not match `X[0]`.")
        if not math.isclose(float(yl.item()), float(y_raw.reshape(-1)[-1]), abs_tol=1e-6):
            errors.append(f"[{desc}] Last label `ds[-1][1]` does not match `y[-1]`.")

        try:
            loader = DataLoader(ds, batch_size=6, shuffle=False)
            xb, yb = next(iter(loader))
            if tuple(xb.shape) != (6, p) or tuple(yb.shape) != (6, 1):
                errors.append(
                    f"[{desc}] DataLoader batch shapes expected ((6, {p}), (6, 1)), got ({tuple(xb.shape)}, {tuple(yb.shape)})."
                )
        except Exception as exc:
            errors.append(f"[{desc}] DataLoader iteration failed: {type(exc).__name__}: {exc}")

    if errors:
        return CheckResult(
            "ch28_todo_03",
            False,
            "failed",
            "To-Do 28.3 — Check `ClinicalTabularDataset` Tensor Shapes & Dtypes",
            "Store `self.X` as `torch.float32` shape `(n, p)` and `self.y` as `torch.float32` shape `(n, 1)` (via `.reshape(-1, 1)`), implement `__len__` and `__getitem__`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch28_todo_03",
        True,
        "passed",
        "To-Do 28.3 Passed — Custom `ClinicalTabularDataset` & `DataLoader` Verified!",
        "Passed 1D and 2D target reshaping, `torch.float32` storage, `__len__`, `__getitem__`, and mini-batch `DataLoader` batching.",
        tip="Reshaping binary labels to `(n, 1)` inside your `Dataset` guarantees every mini-batch `(batch, 1)` matches `ClinicalRiskMLP` logits without silent broadcasting bugs!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_todo_03",
        chapter="ch28",
        title="To-Do 28.3: Custom Tabular Dataset (ClinicalTabularDataset)",
        book_section="Section 28.3.3 (Custom Dataset & Mini-Batch DataLoader)",
        points=10,
        hints={
            1: "Subclass `torch.utils.data.Dataset` and implement `__init__(self, X, y)`, `__len__(self)`, and `__getitem__(self, idx)`.",
            2: "Convert `X` with `torch.tensor(X, dtype=torch.float32)` and reshape `y` to 2D `(n, 1)` with `torch.tensor(y, dtype=torch.float32).reshape(-1, 1)`.",
            3: "class ClinicalTabularDataset(Dataset):\n    def __init__(self, X, y):\n        self.X = torch.tensor(X, dtype=torch.float32)\n        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)\n    def __len__(self):\n        return len(self.X)\n    def __getitem__(self, idx):\n        return self.X[___], self.y[___]",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 28.4: ClinicalRiskMLP(nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_04(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_class(
        "ch28_todo_04",
        "To-Do 28.4",
        cls,
        "ClinicalRiskMLP",
        nn.Module,
        "torch.nn.Module",
        10,
    )
    if bad is not None:
        return bad

    errors = []
    torch.manual_seed(42)

    try:
        m_default = cls(4)
        m_custom = cls(6, hidden_dim=10)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch28_todo_04",
            False,
            "failed",
            "To-Do 28.4 — Could Not Instantiate `ClinicalRiskMLP`",
            f"Raised {type(exc).__name__}: {exc}",
            details=[
                "Define `def __init__(self, in_dim, hidden_dim=16):` and call `super().__init__()` first!"
            ],
            points_earned=0,
            points_possible=10,
        )

    for model, in_d, hid_d, desc in [
        (m_default, 4, 16, "default hidden_dim=16 (in_dim=4)"),
        (m_custom, 6, 10, "custom hidden_dim=10 (in_dim=6)"),
    ]:
        expected_params = in_d * hid_d + hid_d + hid_d * 1 + 1
        actual_params = sum(p.numel() for p in model.parameters())
        if actual_params != expected_params:
            errors.append(
                f"[{desc}] Expected {expected_params} total parameters for Linear({in_d},{hid_d})->ReLU->Linear({hid_d},1), got {actual_params}."
            )

        has_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        has_relu = any(isinstance(m, nn.ReLU) for m in model.modules())
        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in model.modules())
        if has_linear != 2:
            errors.append(f"[{desc}] Expected exactly 2 `nn.Linear` layers, found {has_linear}.")
        if has_sigmoid:
            errors.append(
                f"[{desc}] Do NOT include `nn.Sigmoid` inside `ClinicalRiskMLP`! Return raw logits for `BCEWithLogitsLoss`."
            )

        x_small = torch.randn(8, in_d)
        try:
            out = model(x_small)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch28_todo_04",
                False,
                "failed",
                "To-Do 28.4 — Runtime Error in `ClinicalRiskMLP.forward`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if out is None:
            return _none_return_trap("ch28_todo_04", "To-Do 28.4", "model(x)", 10)

        if not isinstance(out, torch.Tensor) or tuple(out.shape) != (8, 1):
            errors.append(
                f"[{desc}] Expected output tensor of shape (8, 1), got {type(out).__name__} with shape {getattr(out, 'shape', None)}."
            )
            continue

        # Check functional sigmoid wasn't applied either (large-scale inputs should exceed (0, 1))
        x_big = torch.randn(100, in_d) * 50.0
        with torch.no_grad():
            big_out = model(x_big)
        if float(big_out.min()) >= 0.0 and float(big_out.max()) <= 1.0:
            errors.append(
                f"[{desc}] Outputs on large inputs stayed strictly inside [0, 1]! "
                "Remove `torch.sigmoid` from `forward()` and return raw logits for `BCEWithLogitsLoss`."
            )

        # Verify ReLU nonlinearity (even if F.relu was used instead of nn.ReLU)
        if not has_relu:
            # Check nonlinearity: f(a + b) != f(a) + f(b) - f(0)
            a = torch.randn(16, in_d)
            b = torch.randn(16, in_d)
            z0 = torch.zeros(16, in_d)
            with torch.no_grad():
                lin_diff = (model(a + b) - (model(a) + model(b) - model(z0))).abs().max().item()
            if lin_diff < 1e-5:
                errors.append(f"[{desc}] Model appears linear — make sure you apply `nn.ReLU()` between the two `nn.Linear` layers!")

        # Check backward pass works cleanly with BCEWithLogitsLoss
        y_dummy = torch.randint(0, 2, (8, 1)).float()
        loss = nn.BCEWithLogitsLoss()(out, y_dummy)
        loss.backward()
        if any(p.grad is None for p in model.parameters()):
            errors.append(f"[{desc}] Some parameters received `None` gradient after `.backward()`.")

    if errors:
        return CheckResult(
            "ch28_todo_04",
            False,
            "failed",
            "To-Do 28.4 — Check `ClinicalRiskMLP` Architecture & Raw Logit Output",
            "Build `Linear(in_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)` inside `nn.Module`, returning raw logits of shape `(batch, 1)` without sigmoid.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch28_todo_04",
        True,
        "passed",
        "To-Do 28.4 Passed — `ClinicalRiskMLP` Raw-Logit Architecture Verified!",
        "Verified parameter counts (97 for 4->16->1, 81 for 6->10->1), ReLU nonlinearity, `(batch, 1)` raw logits, and clean `BCEWithLogitsLoss` gradients.",
        tip="Returning raw logits from `forward()` lets `BCEWithLogitsLoss` use the numerically stable log-sum-exp trick — apply `torch.sigmoid` only when inspecting probabilities at inference!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_todo_04",
        chapter="ch28",
        title="To-Do 28.4: Two-Layer Binary Risk MLP (ClinicalRiskMLP)",
        book_section="Section 28.3.4 (Defining nn.Module & Raw Logits)",
        points=10,
        hints={
            1: "Subclass `nn.Module`, call `super().__init__()` inside `__init__(self, in_dim, hidden_dim=16)`, and stack `nn.Linear(in_dim, hidden_dim)`, `nn.ReLU()`, and `nn.Linear(hidden_dim, 1)`.",
            2: "You can store the stack in `self.net = nn.Sequential(...)` and simply `return self.net(x)` in `forward(self, x)`. Do NOT add `nn.Sigmoid()`!",
            3: "class ClinicalRiskMLP(nn.Module):\n    def __init__(self, in_dim, hidden_dim=16):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(in_dim, hidden_dim),\n            nn.ReLU(),\n            nn.Linear(___, 1),\n        )\n    def forward(self, x):\n        return self.net(x)",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 28.5: train_one_epoch(model, loader, optimizer, criterion)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import copy
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    bad = _check_callable("ch28_todo_05", "To-Do 28.5", fn, "train_one_epoch", 10)
    if bad is not None:
        return bad

    torch.manual_seed(42)
    X = torch.randn(50, 4)
    logits_true = 1.5 * X[:, 0] - 1.2 * X[:, 1] + 0.6 * X[:, 2]
    y = (logits_true > 0).float().unsqueeze(1)
    ds = TensorDataset(X, y)
    # Uneven batch size 16 -> batches of 16, 16, 16, 2
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    def _make_model() -> nn.Module:
        torch.manual_seed(123)
        return nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 1))

    # Reference run (both sample-weighted and batch-averaged losses)
    ref_model = _make_model()
    ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()
    ref_model.train()
    ref_weighted_sum = 0.0
    ref_batch_losses = []
    ref_n = 0
    for xb, yb in loader:
        ref_opt.zero_grad()
        out = ref_model(xb)
        loss = criterion(out, yb)
        loss.backward()
        ref_opt.step()
        bs = len(xb)
        ref_weighted_sum += loss.item() * bs
        ref_batch_losses.append(loss.item())
        ref_n += bs
    exp_weighted = round(ref_weighted_sum / ref_n, 5)
    exp_batch_avg = round(sum(ref_batch_losses) / len(ref_batch_losses), 5)

    # Learner run: start in eval() mode AND inject huge stale grads to test zero_grad()!
    stu_model = _make_model()
    init_state = copy.deepcopy(stu_model.state_dict())
    stu_opt = torch.optim.SGD(stu_model.parameters(), lr=0.1)
    for p in stu_model.parameters():
        p.grad = torch.full_like(p, 50.0)
    stu_model.eval()

    errors = []
    try:
        got_loss = fn(stu_model, loader, stu_opt, criterion)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch28_todo_05",
            False,
            "failed",
            "To-Do 28.5 — Runtime Error in `train_one_epoch`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got_loss is None:
        return _none_return_trap(
            "ch28_todo_05", "To-Do 28.5", "train_one_epoch(model, loader, optimizer, criterion)", 10
        )

    if isinstance(got_loss, torch.Tensor):
        return CheckResult(
            "ch28_todo_05",
            False,
            "failed",
            "To-Do 28.5 — Return a Python `float` (Not a `torch.Tensor`)",
            "Convert the epoch loss with `.item()` / `float(...)` and `round(..., 5)`.",
            points_earned=0,
            points_possible=10,
        )

    if not _is_np_or_py_number(got_loss) or not math.isfinite(float(got_loss)):
        return CheckResult(
            "ch28_todo_05",
            False,
            "failed",
            "To-Do 28.5 — Non-Finite or Non-Numeric Loss Returned",
            f"Expected a finite float rounded to 5 decimals, got {got_loss!r}.",
            points_earned=0,
            points_possible=10,
        )

    if not stu_model.training:
        errors.append("Model was left in `eval()` mode! Call `model.train()` at the top of `train_one_epoch`.")

    # Verify parameters actually updated and match reference (catches missing step() or missing zero_grad())
    unchanged = all(
        torch.allclose(stu_model.state_dict()[k], init_state[k], atol=1e-7) for k in init_state
    )
    if unchanged:
        errors.append("Model weights did not change at all — did you call `loss.backward()` and `optimizer.step()`?")
    else:
        weights_match = all(
            torch.allclose(stu_model.state_dict()[k], ref_model.state_dict()[k], atol=1e-5)
            for k in ref_model.state_dict()
        )
        if not weights_match:
            errors.append(
                "Updated weights diverged from the canonical 5-step training loop! "
                "Make sure you call `optimizer.zero_grad()` BEFORE `loss.backward()` inside every batch so stale gradients never accumulate."
            )

    if not (
        math.isclose(float(got_loss), exp_weighted, abs_tol=1e-4)
        or math.isclose(float(got_loss), exp_batch_avg, abs_tol=1e-4)
    ):
        errors.append(
            f"Epoch mean loss mismatch: expected {exp_weighted} (sample-weighted) or {exp_batch_avg} (batch-averaged), got {got_loss!r}."
        )

    # Multi-epoch decreasing loss check
    loss_e1 = float(got_loss)
    loss_last = loss_e1
    for _ in range(5):
        loss_last = float(fn(stu_model, loader, stu_opt, criterion))
    if not (loss_last < loss_e1):
        errors.append(
            f"Loss failed to decrease over 6 epochs ({loss_e1:.5f} -> {loss_last:.5f}). Check update order."
        )

    if errors:
        return CheckResult(
            "ch28_todo_05",
            False,
            "failed",
            "To-Do 28.5 — Check the Canonical 5-Step PyTorch Training Loop",
            "Call `model.train()`, then for each `(xb, yb)` run: `optimizer.zero_grad()`, `logits = model(xb)`, `loss = criterion(logits, yb)`, `loss.backward()`, `optimizer.step()`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch28_todo_05",
        True,
        "passed",
        "To-Do 28.5 Passed — Canonical 5-Step Training Loop Verified!",
        "Switched to `model.train()`, cleared stale gradients with `optimizer.zero_grad()`, matched exact SGD weight updates, and reduced loss smoothly across epochs.",
        tip="Sample-weighted epoch averaging (`total_loss += loss.item() * len(xb)` divided by `n_samples`) stays exact even when the final mini-batch is smaller than `batch_size`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_todo_05",
        chapter="ch28",
        title="To-Do 28.5: Canonical 5-Step Epoch Trainer (train_one_epoch)",
        book_section="Section 28.3.5 (The Canonical 5-Step Training Loop)",
        points=10,
        hints={
            1: "Start with `model.train()`. Inside `for xb, yb in loader:`, execute all 5 canonical steps in exact order: zero_grad -> forward -> loss -> backward -> step.",
            2: "Accumulate `total_loss += loss.item() * len(xb)` and `total_n += len(xb)`, then return `round(total_loss / total_n, 5)`.",
            3: "def train_one_epoch(model, loader, optimizer, criterion):\n    model.train()\n    total_loss, total_n = 0.0, 0\n    for xb, yb in loader:\n        optimizer.___()\n        logits = model(xb)\n        loss = criterion(logits, yb)\n        loss.___()\n        optimizer.___()\n        total_loss += loss.item() * len(xb)\n        total_n += len(xb)\n    return round(total_loss / total_n, 5)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 28.6: evaluate_binary_logits(model, loader)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset

    bad = _check_callable("ch28_todo_06", "To-Do 28.6", fn, "evaluate_binary_logits", 10)
    if bad is not None:
        return bad

    torch.manual_seed(7)
    X = torch.randn(45, 4)
    y = ((1.4 * X[:, 0] - 0.9 * X[:, 1] + 0.4 * torch.randn(45)) > 0).float().unsqueeze(1)
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

    # Deterministic model with Dropout to prove eval() is actually called!
    torch.manual_seed(99)
    model = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Dropout(0.5), nn.Linear(12, 1))
    # Quick deterministic training so predictions are informative
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(25):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    # Compute exact reference under eval() + no_grad()
    model.eval()
    with torch.no_grad():
        all_logits = model(X)
        ref_loss = round(float(crit(all_logits, y).item()), 5)
        ref_probs = torch.sigmoid(all_logits).squeeze(1).numpy()
        y_np = y.squeeze(1).numpy().astype(int)
        ref_auc = round(float(roc_auc_score(y_np, ref_probs)), 4)
        ref_acc = round(float(np.mean((ref_probs >= 0.5).astype(int) == y_np)), 4)

    # Put model back in train() mode and clear grads before testing learner
    model.train()
    for p in model.parameters():
        p.grad = None

    try:
        got = fn(model, loader)
    except NotImplementedError:
        raise
    except Exception as exc:
        return CheckResult(
            "ch28_todo_06",
            False,
            "failed",
            "To-Do 28.6 — Runtime Error in `evaluate_binary_logits`",
            f"Raised {type(exc).__name__}: {exc}",
            points_earned=0,
            points_possible=10,
        )

    if got is None:
        return _none_return_trap("ch28_todo_06", "To-Do 28.6", "evaluate_binary_logits(model, loader)", 10)

    expected_keys = ["loss", "roc_auc", "accuracy"]
    key_err = _check_dict_keys("ch28_todo_06", "To-Do 28.6", got, expected_keys, 10)
    if key_err is not None:
        return key_err

    errors = []
    if model.training:
        errors.append("Model is still in `train()` mode! Call `model.eval()` before evaluating.")
    if any(p.grad is not None for p in model.parameters()):
        errors.append("Found non-None gradients on model parameters — wrap inference inside `with torch.no_grad():`!")

    for k, exp_v, tol in [
        ("loss", ref_loss, 1e-4),
        ("roc_auc", ref_auc, 1e-4),
        ("accuracy", ref_acc, 1e-4),
    ]:
        gv = got[k]
        if not _is_np_or_py_number(gv) or not math.isclose(float(gv), exp_v, abs_tol=tol):
            extra = ""
            if k == "accuracy":
                extra = " (Apply `torch.sigmoid(logits)` and threshold `probs >= 0.5`!)"
            elif k == "loss":
                extra = " (Use sample-weighted `BCEWithLogitsLoss()` across all batches in `eval()` mode!)"
            errors.append(f"`{k}`: expected {exp_v}, got {gv!r}{extra}")

    if errors:
        return CheckResult(
            "ch28_todo_06",
            False,
            "failed",
            "To-Do 28.6 — Check `model.eval()`, `torch.no_grad()` & Sigmoid Metrics",
            "Set `model.eval()`, run under `with torch.no_grad():` with `BCEWithLogitsLoss()`, collect `torch.sigmoid(logits)` probabilities and labels, and return `{'loss': round5, 'roc_auc': round4, 'accuracy': round4}`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch28_todo_06",
        True,
        "passed",
        "To-Do 28.6 Passed — Inference Mode (`eval()` + `no_grad()`) Verified!",
        f"Disabled Dropout cleanly via `model.eval()`, ran gradient-free under `torch.no_grad()`, and matched exact BCE loss ({ref_loss}), ROC-AUC ({ref_auc}), and accuracy ({ref_acc}).",
        tip="`model.eval()` changes layer behavior (Dropout/BatchNorm), while `with torch.no_grad():` disables autograd memory tracking — always use BOTH for validation and test evaluation!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_todo_06",
        chapter="ch28",
        title="To-Do 28.6: Gradient-Free Binary Logit Evaluation (evaluate_binary_logits)",
        book_section="Section 28.3.6 (Inference Mode: eval() & torch.no_grad())",
        points=10,
        hints={
            1: "Call `model.eval()` and wrap the evaluation loop inside `with torch.no_grad():` using `criterion = nn.BCEWithLogitsLoss()`.",
            2: "For each `(xb, yb)`, accumulate `total_loss += criterion(logits, yb).item() * len(xb)` and store `torch.sigmoid(logits).cpu().numpy().ravel()` alongside `yb.cpu().numpy().ravel()`.",
            3: "def evaluate_binary_logits(model, loader):\n    model.eval()\n    criterion = nn.BCEWithLogitsLoss()\n    total_loss, total_n, probs_list, y_list = 0.0, 0, [], []\n    with torch.no_grad():\n        for xb, yb in loader:\n            logits = model(xb)\n            total_loss += criterion(logits, yb).item() * len(xb)\n            total_n += len(xb)\n            probs_list.append(torch.sigmoid(logits).cpu().numpy().ravel())\n            y_list.append(yb.cpu().numpy().ravel())\n    probs = np.concatenate(probs_list)\n    y_true = np.concatenate(y_list).astype(int)\n    return {'loss': round(total_loss / total_n, 5), 'roc_auc': round(float(roc_auc_score(y_true, probs)), 4), 'accuracy': round(float(np.mean((probs >= 0.5).astype(int) == y_true)), 4)}",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 28.1 (30 pts): count_model_parameters(model)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import torch.nn as nn

    bad = _check_callable("ch28_exam_01", "Exam Task 28.1", fn, "count_model_parameters", 30)
    if bad is not None:
        return bad

    m1 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    m2 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
    for p in m2[0].parameters():
        p.requires_grad = False
    m3 = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 6), nn.ReLU(), nn.Linear(6, 1))
    for p in m3.parameters():
        p.requires_grad = False

    cases = [
        (m1, {"total_params": 97, "trainable_params": 97, "frozen_params": 0}, "fully trainable 4->16->1 MLP"),
        (m2, {"total_params": 97, "trainable_params": 17, "frozen_params": 80}, "partially frozen MLP (first layer frozen)"),
        (m3, {"total_params": 193, "trainable_params": 0, "frozen_params": 193}, "fully frozen 3-layer MLP"),
    ]
    expected_keys = ["total_params", "trainable_params", "frozen_params"]
    errors = []

    for model, exp, desc in cases:
        grad_flags_before = [p.requires_grad for p in model.parameters()]
        try:
            got = fn(model)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch28_exam_01",
                False,
                "failed",
                "Exam Task 28.1 — Runtime Error in `count_model_parameters`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap("ch28_exam_01", "Exam Task 28.1", "count_model_parameters(model)", 30)

        key_err = _check_dict_keys("ch28_exam_01", "Exam Task 28.1", got, expected_keys, 30)
        if key_err is not None:
            return key_err

        if [p.requires_grad for p in model.parameters()] != grad_flags_before:
            errors.append(f"[{desc}] Mutated `requires_grad` flags on `model`!")

        for k, exp_v in exp.items():
            gv = got[k]
            if not isinstance(gv, int) or isinstance(gv, bool) or gv != exp_v:
                errors.append(f"[{desc}] `{k}`: expected int {exp_v}, got {gv!r} ({type(gv).__name__}).")

        if (
            isinstance(got["total_params"], int)
            and isinstance(got["trainable_params"], int)
            and isinstance(got["frozen_params"], int)
            and got["total_params"] != got["trainable_params"] + got["frozen_params"]
        ):
            errors.append(f"[{desc}] Invariant violated: total_params != trainable_params + frozen_params.")

    if errors:
        return CheckResult(
            "ch28_exam_01",
            False,
            "failed",
            "Exam Task 28.1 — Check Parameter Counting & `requires_grad` Split",
            "Sum `p.numel()` over `model.parameters()`, splitting into `trainable_params` (`p.requires_grad`) and `frozen_params` (`not p.requires_grad`), returning plain Python `int`s.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch28_exam_01",
        True,
        "passed",
        "Exam Task 28.1 Passed (30/30 pts) — Model Parameter Audit Verified!",
        "Accurately counted total, trainable, and frozen scalar parameters across trainable, partially frozen, and fully frozen architectures.",
        points_earned=30,
        points_possible=30,
        tip="Auditing trainable vs. frozen parameters is your first sanity check whenever you fine-tune pretrained backbones or freeze embedding layers!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_exam_01",
        chapter="ch28",
        title="Exam 28.1: Trainable vs. Frozen Parameter Audit (count_model_parameters)",
        book_section="Chapter 28 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Iterate through `model.parameters()` and use `p.numel()` to count how many scalar weights/biases live in each tensor.",
            2: "Check `p.requires_grad` on every parameter: sum into `trainable` when True and `frozen` when False, with `total = trainable + frozen`.",
            3: "def count_model_parameters(model):\n    total = sum(p.numel() for p in model.parameters())\n    trainable = sum(p.numel() for p in model.parameters() if p.___)\n    frozen = total - trainable\n    return {'total_params': int(total), 'trainable_params': int(trainable), 'frozen_params': int(frozen)}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 28.2 (35 pts):
# train_pytorch_tabular_model(X_train, y_train, X_val, y_val,
#     hidden_dim=16, epochs=40, lr=0.02, batch_size=32, seed=42)
# ---------------------------------------------------------------------------
def _make_synthetic_clinical_cohort(seed: int) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    n_train, n_val = 240, 100
    X_all = rng.normal(0.0, 1.0, size=(n_train + n_val, 4))
    # Clear nonlinear + linear clinical risk score (easily > 0.92 ROC-AUC for MLP)
    score = (
        1.6 * X_all[:, 0]
        - 1.3 * X_all[:, 1]
        + 0.9 * (X_all[:, 2] ** 2 - 1.0)
        - 0.7 * X_all[:, 3]
        + rng.normal(0.0, 0.45, size=n_train + n_val)
    )
    y_all = (score > 0.0).astype(int)
    return X_all[:n_train], y_all[:n_train], X_all[n_train:], y_all[n_train:]


def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score

    bad = _check_callable("ch28_exam_02", "Exam Task 28.2", fn, "train_pytorch_tabular_model", 35)
    if bad is not None:
        return bad

    X_tr1, y_tr1, X_va1, y_va1 = _make_synthetic_clinical_cohort(seed=101)
    X_tr2, y_tr2, X_va2, y_va2 = _make_synthetic_clinical_cohort(seed=202)

    cases = [
        (X_tr1, y_tr1, X_va1, y_va1, {}, "default hyperparameters (hidden_dim=16, epochs=40, lr=0.02, seed=42)"),
        (
            X_tr2,
            y_tr2,
            X_va2,
            y_va2,
            {"hidden_dim": 12, "epochs": 30, "lr": 0.02, "batch_size": 32, "seed": 7},
            "custom hyperparameters (hidden_dim=12, epochs=30, seed=7)",
        ),
    ]
    expected_keys = ["train_losses", "val_losses", "val_auc", "model"]
    errors = []

    for X_tr, y_tr, X_va, y_va, kwargs, desc in cases:
        epochs = kwargs.get("epochs", 40)
        X_tr_c, y_tr_c = X_tr.copy(), y_tr.copy()
        X_va_c, y_va_c = X_va.copy(), y_va.copy()
        try:
            got = fn(X_tr_c, y_tr_c, X_va_c, y_va_c, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch28_exam_02",
                False,
                "failed",
                "Exam Task 28.2 — Check Function Signature",
                f"Failed on {desc} with TypeError: {exc}",
                details=[
                    "Use signature `train_pytorch_tabular_model(X_train, y_train, X_val, y_val, hidden_dim=16, epochs=40, lr=0.02, batch_size=32, seed=42)`."
                ],
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch28_exam_02",
                False,
                "failed",
                "Exam Task 28.2 — Runtime Error in `train_pytorch_tabular_model`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch28_exam_02",
                "Exam Task 28.2",
                "train_pytorch_tabular_model(X_train, y_train, X_val, y_val)",
                35,
            )

        key_err = _check_dict_keys("ch28_exam_02", "Exam Task 28.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not np.array_equal(X_tr_c, X_tr) or not np.array_equal(X_va_c, X_va):
            errors.append(f"[{desc}] Mutated caller NumPy feature arrays!")

        tr_losses = got["train_losses"]
        va_losses = got["val_losses"]
        val_auc = got["val_auc"]
        model = got["model"]

        if not isinstance(model, nn.Module):
            errors.append(f"[{desc}] `model` must be an `nn.Module` instance, got {type(model).__name__}.")
            continue

        if not isinstance(tr_losses, list) or len(tr_losses) != epochs:
            errors.append(
                f"[{desc}] `train_losses` must be a list of length {epochs}, got {type(tr_losses).__name__} len={len(tr_losses) if hasattr(tr_losses, '__len__') else 'N/A'}."
            )
            continue

        if not isinstance(va_losses, list) or len(va_losses) != epochs:
            errors.append(
                f"[{desc}] `val_losses` must be a list of length {epochs}, got {type(va_losses).__name__} len={len(va_losses) if hasattr(va_losses, '__len__') else 'N/A'}."
            )
            continue

        if not all(_is_np_or_py_number(v) and math.isfinite(float(v)) for v in tr_losses + va_losses):
            errors.append(f"[{desc}] Loss histories contain non-numeric or non-finite values.")
            continue

        if not (float(tr_losses[-1]) < float(tr_losses[0])):
            errors.append(
                f"[{desc}] Training loss did not decrease ({tr_losses[0]:.4f} -> {tr_losses[-1]:.4f})."
            )
        if not (float(va_losses[-1]) < float(va_losses[0])):
            errors.append(
                f"[{desc}] Validation loss did not decrease ({va_losses[0]:.4f} -> {va_losses[-1]:.4f})."
            )

        if not _is_np_or_py_number(val_auc) or float(val_auc) < 0.85:
            errors.append(
                f"[{desc}] Expected `val_auc >= 0.85`, got {val_auc!r}. Check optimizer, loss (`BCEWithLogitsLoss`), and learning loop."
            )
            continue

        # Independently verify the returned model on X_val
        model.eval()
        with torch.no_grad():
            X_va_t = torch.tensor(X_va, dtype=torch.float32)
            logits_va = model(X_va_t)
            probs_va = torch.sigmoid(logits_va).cpu().numpy().ravel()
        recomputed_auc = round(float(roc_auc_score(y_va, probs_va)), 4)
        if recomputed_auc < 0.85:
            errors.append(
                f"[{desc}] Independently evaluated `model` on `X_val` got ROC-AUC={recomputed_auc} (< 0.85)."
            )
        elif not math.isclose(float(val_auc), recomputed_auc, abs_tol=1e-3):
            errors.append(
                f"[{desc}] Returned `val_auc` ({val_auc}) does not match independent model evaluation ({recomputed_auc})."
            )

    if errors:
        return CheckResult(
            "ch28_exam_02",
            False,
            "failed",
            "Exam Task 28.2 — Check End-to-End PyTorch Tabular Training Loop",
            "Seed with `torch.manual_seed(seed)`, build `ClinicalRiskMLP` + `DataLoader` + `BCEWithLogitsLoss` + `Adam`, record `train_losses` and `val_losses` every epoch, and return `{'train_losses', 'val_losses', 'val_auc', 'model'}` with `val_auc >= 0.85`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch28_exam_02",
        True,
        "passed",
        "Exam Task 28.2 Passed (35/35 pts) — End-to-End PyTorch Tabular Trainer Verified!",
        "Trained deterministic `ClinicalRiskMLP` models across default and custom hyperparameters, reducing both train and validation BCE loss and achieving val ROC-AUC >= 0.85.",
        points_earned=35,
        points_possible=35,
        tip="Tracking both `train_losses` and `val_losses` every epoch lets you diagnose underfitting vs. overfitting before you ever touch the held-out test set.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_exam_02",
        chapter="ch28",
        title="Exam 28.2: End-to-End PyTorch Tabular Trainer (train_pytorch_tabular_model)",
        book_section="Chapter 28 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Call `torch.manual_seed(seed)` first, wrap `(X_train, y_train)` and `(X_val, y_val)` into `TensorDataset` or `ClinicalTabularDataset` + `DataLoader`, and instantiate a 2-layer `Linear(in_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, 1)` model.",
            2: "Use `criterion = nn.BCEWithLogitsLoss()` and `optimizer = torch.optim.Adam(model.parameters(), lr=lr)`. Each epoch, run a training pass (`model.train()`) and a validation pass (`model.eval()` + `torch.no_grad()`), appending rounded float losses.",
            3: "def train_pytorch_tabular_model(X_train, y_train, X_val, y_val, hidden_dim=16, epochs=40, lr=0.02, batch_size=32, seed=42):\n    torch.manual_seed(seed)\n    # build train_loader, val_loader, model, criterion, optimizer\n    train_losses, val_losses = [], []\n    for _ in range(epochs):\n        # train epoch & append mean train loss\n        # eval epoch under torch.no_grad() & append mean val loss\n    # compute val_auc = round(float(roc_auc_score(y_val, probs)), 4)\n    return {'train_losses': train_losses, 'val_losses': val_losses, 'val_auc': val_auc, 'model': model}",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 28.3 (35 pts):
# save_and_verify_state_dict(model, fresh_model, X_sample, ckpt_path)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch28_exam_03", "Exam Task 28.3", fn, "save_and_verify_state_dict", 35)
    if bad is not None:
        return bad

    expected_keys = ["file_bytes", "predictions_match"]
    errors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Case 1: Standard MLP
        torch.manual_seed(11)
        m1 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        torch.manual_seed(22)
        m1_fresh = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
        X1 = torch.randn(10, 4)

        # Case 2: MLP with Dropout (verifies both models are switched to .eval()!)
        torch.manual_seed(33)
        m2 = nn.Sequential(nn.Linear(5, 12), nn.ReLU(), nn.Dropout(0.5), nn.Linear(12, 1))
        torch.manual_seed(44)
        m2_fresh = nn.Sequential(nn.Linear(5, 12), nn.ReLU(), nn.Dropout(0.5), nn.Linear(12, 1))
        m2.train()
        m2_fresh.train()
        X2 = torch.randn(16, 5)

        cases = [
            (m1, m1_fresh, X1, os.path.join(tmpdir, "mlp_ckpt.pt"), "standard 4->16->1 MLP"),
            (m2, m2_fresh, X2, os.path.join(tmpdir, "dropout_mlp_ckpt.pt"), "MLP with Dropout(0.5) (eval() check)"),
        ]

        for model, fresh, X_sample, ckpt_path, desc in cases:
            try:
                got = fn(model, fresh, X_sample, ckpt_path)
            except NotImplementedError:
                raise
            except Exception as exc:
                return CheckResult(
                    "ch28_exam_03",
                    False,
                    "failed",
                    "Exam Task 28.3 — Runtime Error in `save_and_verify_state_dict`",
                    f"Failed on {desc} with {type(exc).__name__}: {exc}",
                    points_earned=0,
                    points_possible=35,
                )

            if got is None:
                return _none_return_trap(
                    "ch28_exam_03",
                    "Exam Task 28.3",
                    "save_and_verify_state_dict(model, fresh_model, X_sample, ckpt_path)",
                    35,
                )

            key_err = _check_dict_keys("ch28_exam_03", "Exam Task 28.3", got, expected_keys, 35)
            if key_err is not None:
                return key_err

            if not os.path.exists(ckpt_path):
                errors.append(f"[{desc}] Checkpoint file was not written to `ckpt_path`!")
                continue

            actual_bytes = int(os.path.getsize(ckpt_path))
            if not isinstance(got["file_bytes"], int) or isinstance(got["file_bytes"], bool) or got["file_bytes"] != actual_bytes or actual_bytes <= 0:
                errors.append(
                    f"[{desc}] `file_bytes` mismatch: expected exact file size {actual_bytes}, got {got['file_bytes']!r}."
                )

            # Confirm saved artifact loads cleanly with weights_only=True (proves state_dict, not whole model pickle!)
            try:
                loaded_obj = torch.load(ckpt_path, weights_only=True)
                if not isinstance(loaded_obj, dict):
                    errors.append(
                        f"[{desc}] Saved file is not a `state_dict` dict — save `model.state_dict()`, NOT `model`!"
                    )
            except Exception as exc:
                errors.append(
                    f"[{desc}] `torch.load(ckpt_path, weights_only=True)` failed ({type(exc).__name__}: {exc}). "
                    "Save `model.state_dict()` rather than pickling the full model object!"
                )

            if model.training or fresh.training:
                errors.append(
                    f"[{desc}] Both `model` and `fresh_model` must be placed in `.eval()` mode before comparing predictions!"
                )

            if got["predictions_match"] is not True:
                errors.append(
                    f"[{desc}] Expected `predictions_match` to be `True`, got {got['predictions_match']!r} "
                    "(Did you call `.eval()` on both models before running inference on Dropout layers?)."
                )

            # Verify fresh_model weights now match model weights
            for k, orig_t in model.state_dict().items():
                if not torch.allclose(orig_t, fresh.state_dict()[k], atol=1e-6):
                    errors.append(f"[{desc}] `fresh_model` parameter `{k}` does not match `model` after loading!")
                    break

    if errors:
        return CheckResult(
            "ch28_exam_03",
            False,
            "failed",
            "Exam Task 28.3 — Check Safe `state_dict` Save/Load & `.eval()` Verification",
            "Save `model.state_dict()` via `torch.save`, load into `fresh_model` with `torch.load(ckpt_path, weights_only=True)`, switch both to `.eval()`, compare under `with torch.no_grad():`, and return `{'file_bytes': int, 'predictions_match': bool}`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch28_exam_03",
        True,
        "passed",
        "Exam Task 28.3 Passed (35/35 pts) — Safe `state_dict` Checkpointing Verified!",
        "Saved portable `state_dict` weights, reloaded safely with `weights_only=True`, enforced `.eval()` on Dropout layers, and confirmed bit-accurate predictions (`atol=1e-6`).",
        points_earned=35,
        points_possible=35,
        tip="Always checkpoint `model.state_dict()` and load with `weights_only=True` — pickling full `nn.Module` objects breaks across refactors and creates arbitrary-code-execution risks.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch28_exam_03",
        chapter="ch28",
        title="Exam 28.3: Safe state_dict Checkpointing & Verification (save_and_verify_state_dict)",
        book_section="Chapter 28 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Save weights (not the full object) with `torch.save(model.state_dict(), ckpt_path)` and load into `fresh_model` with `fresh_model.load_state_dict(torch.load(ckpt_path, weights_only=True))`.",
            2: "Call BOTH `model.eval()` and `fresh_model.eval()` (critical when Dropout/BatchNorm is present!), run both on `X_sample` inside `with torch.no_grad():`, and check `bool(torch.allclose(p1, p2, atol=1e-6))`.",
            3: "def save_and_verify_state_dict(model, fresh_model, X_sample, ckpt_path):\n    torch.save(model.state_dict(), ckpt_path)\n    state = torch.load(ckpt_path, weights_only=True)\n    fresh_model.load_state_dict(state)\n    model.eval()\n    fresh_model.eval()\n    with torch.no_grad():\n        match = bool(torch.allclose(model(X_sample), fresh_model(X_sample), atol=1e-6))\n    return {'file_bytes': int(os.path.getsize(ckpt_path)), 'predictions_match': match}",
        },
        verifier=_verify_exam_03,
    )
)
