"""Chapter 31 verification checks & 3-tier progressive hints.

Covers Guided To-Dos (ch31_todo_01..06) and the Chapter 31 Coding Exam
(ch31_exam_01..03) for Sequence Modeling — RNNs, LSTMs, GRUs, and
Epidemiological Time-Series Forecasting.
Every verifier lazy-imports NumPy/Pandas/Scikit-Learn/PyTorch inside the
check function, validates return structures, tensor shapes, leak-free
temporal splits, and non-mutation, while letting NotImplementedError
propagate cleanly for starter stubs.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..engine import CheckResult, ExerciseSpec, register_exercise


def _is_np_or_py_number(val: Any) -> bool:
    import numpy as np

    return isinstance(
        val, (int, float, np.integer, np.floating)
    ) and not isinstance(val, (bool, np.bool_))


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


def _check_module_class(
    ex_id: str,
    title: str,
    cls: Any,
    expected_name: str,
    points_possible: int = 10,
) -> Optional[CheckResult]:
    import torch.nn as nn

    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        return CheckResult(
            ex_id,
            False,
            "failed",
            f"{title} — Pass the `nn.Module` Class `{expected_name}`",
            (
                f"Expected the class `{expected_name}` (subclass of `nn.Module`), "
                f"but received {type(cls).__name__} ({cls!r})."
            ),
            details=[
                f"Define `class {expected_name}(nn.Module): ...`.",
                f"Pass the class itself without parentheses: tutor.check('{ex_id}', {expected_name}).",
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
        f"Calling {call_str} returned None instead of the expected sequence modeling result.",
        details=[
            "Remember: print() only displays text on screen; functions must use `return ...`.",
            f"Ensure {call_str} ends with an explicit `return` statement.",
        ],
        points_earned=0,
        points_possible=points_possible,
        tip="Reusable PyTorch and time-series utilities always return arrays, tensors, tuples, dicts, or DataFrames.",
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
# GUIDED TO-DO 31.1: build_sliding_windows(series_2d, seq_len=8, target_col=0)
# ---------------------------------------------------------------------------
def _verify_todo_01(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch31_todo_01", "To-Do 31.1", fn, "build_sliding_windows", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(42)
    s1 = rng.normal(size=(20, 3))
    s2 = np.arange(60, dtype=float).reshape(15, 4)

    cases = [
        (s1, {}, "20-week 3-feature series (default seq_len=8, target_col=0)"),
        (s2, {"seq_len": 5, "target_col": 2}, "15-step 4-feature series (seq_len=5, target_col=2)"),
    ]
    errors = []

    for arr_raw, kwargs, desc in cases:
        arr_copy = arr_raw.copy()
        seq_len = kwargs.get("seq_len", 8)
        target_col = kwargs.get("target_col", 0)
        try:
            got = fn(arr_copy, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch31_todo_01",
                False,
                "failed",
                "To-Do 31.1 — Check Signature `build_sliding_windows(series_2d, seq_len=8, target_col=0)`",
                f"Failed on {desc} with TypeError: {exc}",
                details=["Define `def build_sliding_windows(series_2d, seq_len=8, target_col=0):`."],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch31_todo_01",
                False,
                "failed",
                "To-Do 31.1 — Runtime Error in `build_sliding_windows`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch31_todo_01", "To-Do 31.1", "build_sliding_windows(series_2d)", 10
            )

        if not np.array_equal(arr_copy, arr_raw):
            errors.append(f"[{desc}] Mutated caller `series_2d` array in place!")

        if not isinstance(got, tuple) or len(got) != 2:
            errors.append(f"[{desc}] Expected a 2-tuple `(X_seq, y_next)`, got {type(got).__name__}.")
            continue

        X_got, y_got = got
        X_arr = np.asarray(X_got, dtype=float)
        y_arr = np.asarray(y_got, dtype=float)
        n_win = len(arr_raw) - seq_len
        exp_X = np.stack([arr_raw[i : i + seq_len] for i in range(n_win)])
        exp_y = np.array([arr_raw[i + seq_len, target_col] for i in range(n_win)], dtype=float)

        if X_arr.shape != exp_X.shape:
            errors.append(
                f"[{desc}] `X_seq` shape mismatch: expected {exp_X.shape}, got {X_arr.shape}."
            )
        elif not np.allclose(X_arr, exp_X, atol=1e-6):
            errors.append(
                f"[{desc}] `X_seq` values do not match `series_2d[i : i + seq_len]`."
            )

        if y_arr.shape != exp_y.shape:
            errors.append(
                f"[{desc}] `y_next` shape mismatch: expected {exp_y.shape}, got {y_arr.shape}."
            )
        elif not np.allclose(y_arr, exp_y, atol=1e-6):
            errors.append(
                f"[{desc}] `y_next` values do not match `series_2d[i + seq_len, target_col]` "
                "(Watch for off-by-one target alignment!)."
            )

    # Check ValueError when len(series_2d) <= seq_len
    for short_len in (8, 5):
        short_arr = np.zeros((short_len, 3), dtype=float)
        try:
            fn(short_arr, seq_len=8, target_col=0)
            errors.append(
                f"[short series len={short_len}, seq_len=8] Expected `ValueError` when "
                "`len(series_2d) <= seq_len`, but function returned normally."
            )
        except NotImplementedError:
            raise
        except ValueError:
            pass
        except Exception as exc:
            errors.append(
                f"[short series len={short_len}, seq_len=8] Expected `ValueError`, "
                f"got {type(exc).__name__}: {exc}."
            )

    if errors:
        return CheckResult(
            "ch31_todo_01",
            False,
            "failed",
            "To-Do 31.1 — Check Sliding-Window Slicing & Length Guard",
            "Validate `len(arr) > seq_len` (raising `ValueError` otherwise) and slice `X_seq[i] = arr[i:i+seq_len]`, `y_next[i] = arr[i+seq_len, target_col]`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch31_todo_01",
        True,
        "passed",
        "To-Do 31.1 Passed — Sliding-Window Tensor Builder Verified!",
        "Produced exact `(N, seq_len, F)` input windows and `(N,)` next-step targets across custom columns and enforced the `ValueError` short-series guard.",
        tip="Always verify `X_seq[i]` stops strictly at index `i + seq_len - 1` so the target `i + seq_len` never leaks into the input window.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_todo_01",
        chapter="ch31",
        title="To-Do 31.1: Sliding-Window Sequence Builder (build_sliding_windows)",
        book_section="Section 31.3.2 (Leak-Free Temporal Sliding Windows)",
        points=10,
        hints={
            1: "Convert `series_2d` with `arr = np.asarray(series_2d, dtype=float)`. Raise `ValueError` if `arr.ndim != 2` or `len(arr) <= seq_len`.",
            2: "Loop `i` over `range(len(arr) - seq_len)`: append `arr[i : i + seq_len]` to `X_list` and `arr[i + seq_len, target_col]` to `y_list`, then convert with `np.array(..., dtype=float)`.",
            3: "def build_sliding_windows(series_2d, seq_len=8, target_col=0):\n    arr = np.asarray(series_2d, dtype=float)\n    if arr.ndim != 2 or len(arr) <= ___:\n        raise ValueError('Series length must exceed seq_len')\n    n_win = len(arr) - seq_len\n    X_seq = np.stack([arr[i : i + seq_len] for i in range(n_win)])\n    y_next = np.array([arr[i + seq_len, target_col] for i in range(n_win)], dtype=float)\n    return (X_seq, y_next)",
        },
        verifier=_verify_todo_01,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 31.2: rnn_cell_step_numpy(x_t, h_prev, W_xh, W_hh, b_h)
# ---------------------------------------------------------------------------
def _verify_todo_02(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch31_todo_02", "To-Do 31.2", fn, "rnn_cell_step_numpy", 10)
    if bad is not None:
        return bad

    rng = np.random.default_rng(123)
    cases = [
        (
            rng.normal(size=(4, 3)),
            rng.normal(size=(4, 5)),
            rng.normal(size=(3, 5)),
            rng.normal(size=(5, 5)),
            rng.normal(size=(5,)),
            "batched (B=4, F=3, H=5) recurrent step",
        ),
        (
            np.zeros((2, 3)),
            np.zeros((2, 4)),
            np.ones((3, 4)),
            np.ones((4, 4)),
            np.array([0.5, -0.5, 1.0, -1.0]),
            "zero inputs with nonzero bias (checks `+ b_h` and `np.tanh`)",
        ),
    ]
    errors = []

    for x_t, h_prev, W_xh, W_hh, b_h, desc in cases:
        xc, hc, wxc, whc, bc = (
            x_t.copy(),
            h_prev.copy(),
            W_xh.copy(),
            W_hh.copy(),
            b_h.copy(),
        )
        try:
            got = fn(xc, hc, wxc, whc, bc)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch31_todo_02",
                False,
                "failed",
                "To-Do 31.2 — Check Signature `rnn_cell_step_numpy(x_t, h_prev, W_xh, W_hh, b_h)`",
                f"Failed on {desc} with TypeError: {exc}",
                details=["Define `def rnn_cell_step_numpy(x_t, h_prev, W_xh, W_hh, b_h):`."],
                points_earned=0,
                points_possible=10,
            )
        except Exception as exc:
            return CheckResult(
                "ch31_todo_02",
                False,
                "failed",
                "To-Do 31.2 — Runtime Error in `rnn_cell_step_numpy`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch31_todo_02", "To-Do 31.2", "rnn_cell_step_numpy(x_t, h_prev, W_xh, W_hh, b_h)", 10
            )

        if (
            not np.array_equal(xc, x_t)
            or not np.array_equal(hc, h_prev)
            or not np.array_equal(wxc, W_xh)
            or not np.array_equal(whc, W_hh)
            or not np.array_equal(bc, b_h)
        ):
            errors.append(f"[{desc}] Mutated one or more caller input arrays in place!")

        exp = np.tanh(x_t @ W_xh + h_prev @ W_hh + b_h)
        if not isinstance(got, np.ndarray):
            errors.append(f"[{desc}] Expected a NumPy ndarray, got {type(got).__name__}.")
            continue
        if got.shape != exp.shape:
            errors.append(f"[{desc}] Expected shape {exp.shape}, got {got.shape}.")
            continue
        if not np.allclose(got, exp, atol=1e-6):
            raw_lin = x_t @ W_xh + h_prev @ W_hh + b_h
            hint_extra = ""
            if np.allclose(got, raw_lin, atol=1e-6):
                hint_extra = " (Looks like you forgot to apply `np.tanh(...)`!)"
            elif np.allclose(got, np.tanh(x_t @ W_xh + h_prev @ W_hh), atol=1e-6):
                hint_extra = " (Looks like you forgot to add the bias `+ b_h`!)"
            errors.append(f"[{desc}] Output values did not match `np.tanh(x_t @ W_xh + h_prev @ W_hh + b_h)`{hint_extra}.")

    if errors:
        return CheckResult(
            "ch31_todo_02",
            False,
            "failed",
            "To-Do 31.2 — Check Pure-NumPy RNN Cell Formula",
            "Compute `np.tanh(x_t @ W_xh + h_prev @ W_hh + b_h)` and return the `(B, H)` float ndarray.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch31_todo_02",
        True,
        "passed",
        "To-Do 31.2 Passed — Pure-NumPy Recurrent Cell Step Verified!",
        "Exact `(B, H)` hidden-state update verified for batched inputs, bias broadcasting, tanh nonlinearity, and non-mutation.",
        tip="Notice that `W_hh` is reused at every single timestep — multiplying by `W_hh` repeatedly during backpropagation is exactly why vanilla RNN gradients vanish or explode!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_todo_02",
        chapter="ch31",
        title="To-Do 31.2: Pure-NumPy RNN Cell Step (rnn_cell_step_numpy)",
        book_section="Section 31.3.3 (Elman RNN Recurrence & Vanishing Gradients)",
        points=10,
        hints={
            1: "Convert inputs to float NumPy arrays via `np.asarray(..., dtype=float)` so caller arrays are never mutated.",
            2: "Compute the pre-activation `z = x @ W_xh + h @ W_hh + b_h` using matrix multiplication `@`, then squash with `np.tanh(z)`.",
            3: "def rnn_cell_step_numpy(x_t, h_prev, W_xh, W_hh, b_h):\n    x = np.asarray(x_t, dtype=float)\n    h = np.asarray(h_prev, dtype=float)\n    return np.tanh(x @ np.asarray(W_xh, dtype=float) + h @ np.asarray(W_hh, dtype=float) + np.asarray(b_h, dtype=float))",
        },
        verifier=_verify_todo_02,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 31.3: EpiGRUForecaster (nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_03(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_module_class("ch31_todo_03", "To-Do 31.3", cls, "EpiGRUForecaster", 10)
    if bad is not None:
        return bad

    errors = []
    for input_size, hidden_size, use_default in [(3, 16, True), (5, 12, False)]:
        desc = (
            f"default hidden_size=16 (input_size={input_size})"
            if use_default
            else f"custom input_size={input_size}, hidden_size={hidden_size}"
        )
        try:
            torch.manual_seed(42)
            model = cls(input_size) if use_default else cls(input_size, hidden_size=hidden_size)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch31_todo_03",
                False,
                "failed",
                "To-Do 31.3 — Error Instantiating `EpiGRUForecaster`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=["Define `def __init__(self, input_size, hidden_size=16):` and call `super().__init__()`."],
                points_earned=0,
                points_possible=10,
            )

        grus = [m for m in model.modules() if isinstance(m, nn.GRU)]
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not grus:
            errors.append(f"[{desc}] Missing `nn.GRU` layer inside `EpiGRUForecaster`.")
            continue
        gru_layer = grus[0]
        if not gru_layer.batch_first:
            errors.append(f"[{desc}] `nn.GRU` must set `batch_first=True` to accept `(B, T, F)` inputs.")
        if gru_layer.input_size != input_size or gru_layer.hidden_size != hidden_size:
            errors.append(
                f"[{desc}] `nn.GRU` expected `(input_size={input_size}, hidden_size={hidden_size})`, "
                f"got `({gru_layer.input_size}, {gru_layer.hidden_size})`."
            )
        if not linears:
            errors.append(f"[{desc}] Missing `nn.Linear(hidden_size, 1)` output layer.")
            continue
        lin_layer = linears[-1]
        if lin_layer.in_features != hidden_size or lin_layer.out_features != 1:
            errors.append(
                f"[{desc}] `nn.Linear` expected `({hidden_size}, 1)`, "
                f"got `({lin_layer.in_features}, {lin_layer.out_features})`."
            )

        x = torch.randn(7, 8, input_size)
        try:
            y_hat = model(x)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch31_todo_03",
                False,
                "failed",
                "To-Do 31.3 — Runtime Error in `EpiGRUForecaster.forward`",
                f"Forward pass failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if y_hat is None:
            return _none_return_trap("ch31_todo_03", "To-Do 31.3", "EpiGRUForecaster(3)(x)", 10)

        if not isinstance(y_hat, torch.Tensor) or tuple(y_hat.shape) != (7, 1):
            errors.append(
                f"[{desc}] Expected output tensor of shape `(7, 1)`, "
                f"got {tuple(y_hat.shape) if isinstance(y_hat, torch.Tensor) else type(y_hat).__name__}."
            )
            continue

        with torch.no_grad():
            ref_out, _ = gru_layer(x)
            ref_y = lin_layer(ref_out[:, -1, :])
        if not torch.allclose(y_hat.detach(), ref_y, atol=1e-5):
            errors.append(
                f"[{desc}] Output did not match `linear(out[:, -1, :])` — ensure `forward` takes the final timestep `out[:, -1, :]`."
            )

        loss = y_hat.pow(2).mean()
        loss.backward()
        if gru_layer.weight_hh_l0.grad is None or lin_layer.weight.grad is None:
            errors.append(f"[{desc}] Gradients did not flow through both `nn.GRU` and `nn.Linear`.")

    if errors:
        return CheckResult(
            "ch31_todo_03",
            False,
            "failed",
            "To-Do 31.3 — Check `EpiGRUForecaster` Architecture & Forward Pass",
            "Combine `nn.GRU(input_size, hidden_size, batch_first=True)` with `nn.Linear(hidden_size, 1)` applied to the final timestep `out[:, -1, :]`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch31_todo_03",
        True,
        "passed",
        "To-Do 31.3 Passed — `EpiGRUForecaster` Verified!",
        "Verified `nn.GRU(..., batch_first=True)`, `nn.Linear(hidden_size, 1)`, `(B, 1)` output from final timestep `out[:, -1, :]`, and clean autograd flow.",
        tip="With `batch_first=True`, `out` has shape `(B, T, H)`, so `out[:, -1, :]` cleanly extracts the summary hidden state after reading the entire lookback window.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_todo_03",
        chapter="ch31",
        title="To-Do 31.3: PyTorch GRU Forecaster (EpiGRUForecaster)",
        book_section="Section 31.3.4 (PyTorch nn.GRU & nn.LSTM Tensor Shapes)",
        points=10,
        hints={
            1: "Subclass `nn.Module`, call `super().__init__()`, and define `self.gru = nn.GRU(input_size, hidden_size, batch_first=True)` and `self.head = nn.Linear(hidden_size, 1)`.",
            2: "In `forward(self, x)`, unpack `out, _ = self.gru(x)`, slice the final timestep `last_hidden = out[:, -1, :]`, and return `self.head(last_hidden)`.",
            3: "class EpiGRUForecaster(nn.Module):\n    def __init__(self, input_size, hidden_size=16):\n        super().__init__()\n        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)\n        self.head = nn.Linear(hidden_size, ___)\n\n    def forward(self, x):\n        out, _ = self.gru(x)\n        return self.head(out[:, -1, :])",
        },
        verifier=_verify_todo_03,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 31.4: EpiLSTMForecaster (nn.Module)
# ---------------------------------------------------------------------------
def _verify_todo_04(cls: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_module_class("ch31_todo_04", "To-Do 31.4", cls, "EpiLSTMForecaster", 10)
    if bad is not None:
        return bad

    errors = []
    for input_size, hidden_size, use_default in [(3, 16, True), (4, 10, False)]:
        desc = (
            f"default hidden_size=16 (input_size={input_size})"
            if use_default
            else f"custom input_size={input_size}, hidden_size={hidden_size}"
        )
        try:
            torch.manual_seed(42)
            model = cls(input_size) if use_default else cls(input_size, hidden_size=hidden_size)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch31_todo_04",
                False,
                "failed",
                "To-Do 31.4 — Error Instantiating `EpiLSTMForecaster`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                details=["Define `def __init__(self, input_size, hidden_size=16):` and call `super().__init__()`."],
                points_earned=0,
                points_possible=10,
            )

        lstms = [m for m in model.modules() if isinstance(m, nn.LSTM)]
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not lstms:
            errors.append(f"[{desc}] Missing `nn.LSTM` layer inside `EpiLSTMForecaster`.")
            continue
        lstm_layer = lstms[0]
        if not lstm_layer.batch_first:
            errors.append(f"[{desc}] `nn.LSTM` must set `batch_first=True` to accept `(B, T, F)` inputs.")
        if lstm_layer.input_size != input_size or lstm_layer.hidden_size != hidden_size:
            errors.append(
                f"[{desc}] `nn.LSTM` expected `(input_size={input_size}, hidden_size={hidden_size})`, "
                f"got `({lstm_layer.input_size}, {lstm_layer.hidden_size})`."
            )
        if not linears:
            errors.append(f"[{desc}] Missing `nn.Linear(hidden_size, 1)` output layer.")
            continue
        lin_layer = linears[-1]
        if lin_layer.in_features != hidden_size or lin_layer.out_features != 1:
            errors.append(
                f"[{desc}] `nn.Linear` expected `({hidden_size}, 1)`, "
                f"got `({lin_layer.in_features}, {lin_layer.out_features})`."
            )

        x = torch.randn(6, 8, input_size)
        try:
            y_hat = model(x)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch31_todo_04",
                False,
                "failed",
                "To-Do 31.4 — Runtime Error in `EpiLSTMForecaster.forward`",
                f"Forward pass failed on {desc} with {type(exc).__name__}: {exc}",
                details=["Remember: `nn.LSTM` returns `(out, (h_n, c_n))` — unpack as `out, _ = self.lstm(x)`."],
                points_earned=0,
                points_possible=10,
            )

        if y_hat is None:
            return _none_return_trap("ch31_todo_04", "To-Do 31.4", "EpiLSTMForecaster(3)(x)", 10)

        if not isinstance(y_hat, torch.Tensor) or tuple(y_hat.shape) != (6, 1):
            errors.append(
                f"[{desc}] Expected output tensor of shape `(6, 1)`, "
                f"got {tuple(y_hat.shape) if isinstance(y_hat, torch.Tensor) else type(y_hat).__name__}."
            )
            continue

        with torch.no_grad():
            ref_out, _ = lstm_layer(x)
            ref_y = lin_layer(ref_out[:, -1, :])
        if not torch.allclose(y_hat.detach(), ref_y, atol=1e-5):
            errors.append(
                f"[{desc}] Output did not match `linear(out[:, -1, :])` — ensure `forward` projects the final timestep `out[:, -1, :]`."
            )

        loss = y_hat.pow(2).mean()
        loss.backward()
        if lstm_layer.weight_hh_l0.grad is None or lin_layer.weight.grad is None:
            errors.append(f"[{desc}] Gradients did not flow through both `nn.LSTM` and `nn.Linear`.")

    if errors:
        return CheckResult(
            "ch31_todo_04",
            False,
            "failed",
            "To-Do 31.4 — Check `EpiLSTMForecaster` Architecture & Forward Pass",
            "Combine `nn.LSTM(input_size, hidden_size, batch_first=True)` with `nn.Linear(hidden_size, 1)` applied to the final timestep `out[:, -1, :]`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch31_todo_04",
        True,
        "passed",
        "To-Do 31.4 Passed — `EpiLSTMForecaster` Verified!",
        "Verified `nn.LSTM(..., batch_first=True)`, tuple unpacking `(out, (h_n, c_n))`, `(B, 1)` output from `out[:, -1, :]`, and clean autograd flow.",
        tip="Unlike `nn.GRU`, `nn.LSTM` maintains both a hidden state `h_n` and an additive cell-state highway `c_n` returned as a nested tuple `(out, (h_n, c_n))`.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_todo_04",
        chapter="ch31",
        title="To-Do 31.4: PyTorch LSTM Forecaster (EpiLSTMForecaster)",
        book_section="Section 31.3.4 (PyTorch nn.GRU & nn.LSTM Tensor Shapes)",
        points=10,
        hints={
            1: "Subclass `nn.Module`, call `super().__init__()`, and attach `self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)` plus `self.head = nn.Linear(hidden_size, 1)`.",
            2: "In `forward(self, x)`, unpack `out, _ = self.lstm(x)` (or `out, (h_n, c_n) = self.lstm(x)`), then return `self.head(out[:, -1, :])`.",
            3: "class EpiLSTMForecaster(nn.Module):\n    def __init__(self, input_size, hidden_size=16):\n        super().__init__()\n        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)\n        self.head = nn.Linear(hidden_size, ___)\n\n    def forward(self, x):\n        out, _ = self.lstm(x)\n        return self.head(out[:, -1, :])",
        },
        verifier=_verify_todo_04,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 31.5: pad_variable_vital_sequences(seq_list)
# ---------------------------------------------------------------------------
def _verify_todo_05(fn: Any) -> CheckResult:
    import torch
    import torch.nn as nn

    bad = _check_callable("ch31_todo_05", "To-Do 31.5", fn, "pad_variable_vital_sequences", 10)
    if bad is not None:
        return bad

    cases = [
        (
            [
                torch.tensor([[98.0, 120.0], [97.5, 118.0], [96.0, 115.0]]),
                torch.tensor([[99.0, 110.0], [98.5, 112.0], [98.0, 114.0], [97.0, 116.0], [96.5, 119.0]]),
                torch.tensor([[95.0, 130.0], [94.0, 135.0]]),
            ],
            "variable-length ICU vitals (lengths 3, 5, 2)",
        ),
        (
            [
                torch.ones(4, 3) * 2.0,
                torch.ones(4, 3) * 5.0,
            ],
            "equal-length sequences (lengths 4, 4)",
        ),
    ]
    errors = []

    for seqs_raw, desc in cases:
        seqs_copy = [s.clone() for s in seqs_raw]
        try:
            got = fn(seqs_copy)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch31_todo_05",
                False,
                "failed",
                "To-Do 31.5 — Runtime Error in `pad_variable_vital_sequences`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch31_todo_05", "To-Do 31.5", "pad_variable_vital_sequences(seq_list)", 10
            )

        if not isinstance(got, tuple) or len(got) != 3:
            errors.append(
                f"[{desc}] Expected a 3-tuple `(padded, lengths, valid_mask)`, got {type(got).__name__}."
            )
            continue

        if any(not torch.equal(a, b) for a, b in zip(seqs_copy, seqs_raw)):
            errors.append(f"[{desc}] Mutated caller tensors inside `seq_list`!")

        padded, lengths, valid_mask = got
        exp_lengths = torch.tensor([s.shape[0] for s in seqs_raw], dtype=torch.long)
        exp_padded = nn.utils.rnn.pad_sequence(seqs_raw, batch_first=True, padding_value=0.0)
        exp_mask = torch.arange(exp_padded.shape[1]).unsqueeze(0) < exp_lengths.unsqueeze(1)

        if not isinstance(padded, torch.Tensor) or tuple(padded.shape) != tuple(exp_padded.shape):
            errors.append(
                f"[{desc}] `padded` shape mismatch: expected {tuple(exp_padded.shape)}, "
                f"got {tuple(padded.shape) if isinstance(padded, torch.Tensor) else type(padded).__name__}."
            )
        elif not torch.allclose(padded.float(), exp_padded.float(), atol=1e-6):
            errors.append(f"[{desc}] `padded` values do not match zero-padded `pad_sequence(..., batch_first=True, padding_value=0.0)`.")

        if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long or not torch.equal(lengths, exp_lengths):
            errors.append(
                f"[{desc}] `lengths` expected `torch.long` tensor {exp_lengths.tolist()}, "
                f"got {lengths!r}."
            )

        if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
            errors.append(
                f"[{desc}] `valid_mask` must be a boolean tensor (`dtype=torch.bool`), "
                f"got {valid_mask.dtype if isinstance(valid_mask, torch.Tensor) else type(valid_mask).__name__}."
            )
        elif tuple(valid_mask.shape) != tuple(exp_mask.shape) or not torch.equal(valid_mask, exp_mask):
            errors.append(
                f"[{desc}] `valid_mask` mismatch: expected shape {tuple(exp_mask.shape)} with True for real steps and False for padding."
            )

    if errors:
        return CheckResult(
            "ch31_todo_05",
            False,
            "failed",
            "To-Do 31.5 — Check Padding, Lengths & Boolean Valid Mask",
            "Use `nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=0.0)`, `torch.long` lengths, and boolean `valid_mask` of shape `(B, max_len)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch31_todo_05",
        True,
        "passed",
        "To-Do 31.5 Passed — Variable-Length Padding & Masking Verified!",
        "Constructed exact `(B, max_len, F)` zero-padded batches, `torch.long` lengths, and `(B, max_len)` boolean masks without mutating patient sequences.",
        tip="Always pass `valid_mask` when pooling or computing per-timestep loss so zero-padded clinical tail steps never distort predictions or gradients.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_todo_05",
        chapter="ch31",
        title="To-Do 31.5: Variable-Length Padding & Masking (pad_variable_vital_sequences)",
        book_section="Section 31.3.5 (Variable-Length Patient Sequences, Padding & Masking)",
        points=10,
        hints={
            1: "Compute `lengths = torch.tensor([s.shape[0] for s in seq_list], dtype=torch.long)` and `padded = nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=0.0)`.",
            2: "Create the boolean `(B, max_len)` mask via broadcasting: `max_len = padded.shape[1]` and `valid_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)`.",
            3: "def pad_variable_vital_sequences(seq_list):\n    lengths = torch.tensor([s.shape[0] for s in seq_list], dtype=torch.long)\n    padded = nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=0.0)\n    valid_mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(___)\n    return (padded, lengths, valid_mask)",
        },
        verifier=_verify_todo_05,
    )
)


# ---------------------------------------------------------------------------
# GUIDED TO-DO 31.6: naive_persistence_forecast_rmse(y_seq_last, y_true)
# ---------------------------------------------------------------------------
def _verify_todo_06(fn: Any) -> CheckResult:
    import numpy as np

    bad = _check_callable("ch31_todo_06", "To-Do 31.6", fn, "naive_persistence_forecast_rmse", 10)
    if bad is not None:
        return bad

    cases = [
        ([10.0, 20.0, 30.0, 40.0], [12.0, 18.0, 33.0, 37.0], "weekly case counts (exact RMSE = sqrt(6.5))"),
        ([5.0, 5.0, 5.0], [5.0, 5.0, 5.0], "constant flat series (RMSE = 0.0)"),
        (np.array([-0.5, 0.2, 1.4, 0.8]), np.array([-0.2, 0.5, 1.1, 1.3]), "standardized z-score dengue series"),
    ]
    errors = []

    for last_raw, true_raw, desc in cases:
        lc = np.array(last_raw, dtype=float)
        tc = np.array(true_raw, dtype=float)
        try:
            got = fn(lc, tc)
        except NotImplementedError:
            raise
        except Exception as exc:
            return CheckResult(
                "ch31_todo_06",
                False,
                "failed",
                "To-Do 31.6 — Runtime Error in `naive_persistence_forecast_rmse`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=10,
            )

        if got is None:
            return _none_return_trap(
                "ch31_todo_06", "To-Do 31.6", "naive_persistence_forecast_rmse(y_seq_last, y_true)", 10
            )

        if not np.array_equal(lc, np.array(last_raw, dtype=float)) or not np.array_equal(
            tc, np.array(true_raw, dtype=float)
        ):
            errors.append(f"[{desc}] Mutated caller input array in place!")

        mse = float(np.mean((tc - lc) ** 2))
        mae = float(np.mean(np.abs(tc - lc)))
        exp = round(math.sqrt(mse), 4)

        if not _is_np_or_py_number(got) or not math.isfinite(float(got)):
            errors.append(f"[{desc}] Expected finite float {exp}, got {got!r}.")
            continue
        if not math.isclose(float(got), exp, abs_tol=1e-4):
            extra = ""
            if math.isclose(float(got), round(mse, 4), abs_tol=1e-3):
                extra = " (Looks like MSE — did you forget `np.sqrt(...)`?)"
            elif math.isclose(float(got), round(mae, 4), abs_tol=1e-3):
                extra = " (Looks like MAE instead of RMSE!)"
            errors.append(f"[{desc}] Expected {exp}, got {got!r}{extra}")

    if errors:
        return CheckResult(
            "ch31_todo_06",
            False,
            "failed",
            "To-Do 31.6 — Check Persistence Baseline RMSE Formula",
            "Compute `round(float(np.sqrt(np.mean((y_true - y_seq_last) ** 2))), 4)`.",
            details=errors,
            points_earned=0,
            points_possible=10,
        )

    return CheckResult(
        "ch31_todo_06",
        True,
        "passed",
        "To-Do 31.6 Passed — Naive Persistence Baseline RMSE Verified!",
        "Verified exact RMSE computation and 4-decimal rounding across raw counts, flat series, and standardized sequences.",
        tip="In time-series forecasting, ALWAYS compare your neural network against persistence (`y_hat_t = y_{t-1}`) first — if a GRU/LSTM cannot beat persistence, do not publish it!",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_todo_06",
        chapter="ch31",
        title="To-Do 31.6: Naive Persistence Forecast Baseline (naive_persistence_forecast_rmse)",
        book_section="Section 31.3.6 (Persistence vs. GRU vs. LSTM on Dhaka Dengue)",
        points=10,
        hints={
            1: "Convert both `y_seq_last` and `y_true` to float NumPy arrays via `np.asarray(..., dtype=float)`.",
            2: "Compute `rmse = np.sqrt(np.mean((yt - yl) ** 2))` and return `round(float(rmse), 4)`.",
            3: "def naive_persistence_forecast_rmse(y_seq_last, y_true):\n    yl = np.asarray(y_seq_last, dtype=float)\n    yt = np.asarray(y_true, dtype=float)\n    return round(float(np.sqrt(np.mean((yt - yl) ** ___))), 4)",
        },
        verifier=_verify_todo_06,
    )
)


# ---------------------------------------------------------------------------
# Reference helpers for Chapter 31 Coding Exam
# ---------------------------------------------------------------------------
def _ref_temporal_split(
    epi_df: Any,
    district: str,
    seq_len: int = 8,
    cutoff_year: int = 2025,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    sub = (
        epi_df[epi_df["district"] == district]
        .sort_values(["year", "epi_week"])
        .reset_index(drop=True)
    )
    feat_cols = ["dengue_cases", "rainfall_mm", "mean_temp_c"]
    raw = sub[feat_cols].to_numpy(dtype=np.float32)
    n_train = int((sub["year"] < cutoff_year).sum())
    scaler = StandardScaler()
    scaler.fit(raw[:n_train])
    scaled = scaler.transform(raw).astype(np.float32)
    X_train = np.stack([scaled[t - seq_len : t] for t in range(seq_len, n_train)])
    y_train = scaled[seq_len:n_train, 0].astype(np.float32)
    X_test = np.stack([scaled[t - seq_len : t] for t in range(n_train, len(sub))])
    y_test = scaled[n_train:, 0].astype(np.float32)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }


def _ref_train_rnn(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    cell_type: str = "gru",
    hidden_size: int = 16,
    epochs: int = 40,
    lr: float = 0.02,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)

    class _Forecaster(nn.Module):
        def __init__(self, in_dim: int, hid: int) -> None:
            super().__init__()
            if cell_type == "gru":
                self.rnn = nn.GRU(in_dim, hid, batch_first=True)
            else:
                self.rnn = nn.LSTM(in_dim, hid, batch_first=True)
            self.head = nn.Linear(hid, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    x_tr = torch.tensor(np.asarray(X_train, dtype=np.float32))
    y_tr = torch.tensor(np.asarray(y_train, dtype=np.float32)).unsqueeze(-1)
    x_te = torch.tensor(np.asarray(X_test, dtype=np.float32))
    y_te_np = np.asarray(y_test, dtype=float)

    model = _Forecaster(x_tr.shape[-1], hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_tr), y_tr)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(x_te).squeeze(-1).numpy().astype(float)

    test_rmse = round(float(np.sqrt(np.mean((y_te_np - preds) ** 2))), 4)
    test_mae = round(float(np.mean(np.abs(y_te_np - preds))), 4)
    return {
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "final_train_loss": round(final_loss, 5),
        "preds": preds,
    }


# ---------------------------------------------------------------------------
# EXAM TASK 31.1 (30 pts): temporal_district_window_split(...)
# ---------------------------------------------------------------------------
def _verify_exam_01(fn: Any) -> CheckResult:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from ..datasets import load_data

    bad = _check_callable(
        "ch31_exam_01", "Exam Task 31.1", fn, "temporal_district_window_split", 30
    )
    if bad is not None:
        return bad

    df_epi = load_data("epi_surveillance", as_pandas=True)
    # Also test on a shuffled copy to verify chronological sorting by ["year", "epi_week"]
    df_shuffled = df_epi.sample(frac=1.0, random_state=99).reset_index(drop=True)
    expected_keys = ["X_train", "y_train", "X_test", "y_test", "scaler"]
    cases = [
        (df_epi, "Dhaka", {"seq_len": 8, "cutoff_year": 2025}, "Dhaka (seq_len=8, cutoff=2025)"),
        (df_shuffled, "Chattogram", {"seq_len": 6, "cutoff_year": 2025}, "shuffled Chattogram (seq_len=6)"),
    ]
    errors = []

    for src_df, district, kwargs, desc in cases:
        df_copy = src_df.copy(deep=True)
        try:
            got = fn(df_copy, district, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch31_exam_01",
                False,
                "failed",
                "Exam Task 31.1 — Check Signature `temporal_district_window_split(epi_df, district, seq_len=8, cutoff_year=2025)`",
                f"Failed on {desc} with TypeError: {exc}",
                points_earned=0,
                points_possible=30,
            )
        except Exception as exc:
            return CheckResult(
                "ch31_exam_01",
                False,
                "failed",
                "Exam Task 31.1 — Runtime Error in `temporal_district_window_split`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=30,
            )

        if got is None:
            return _none_return_trap(
                "ch31_exam_01", "Exam Task 31.1", f"temporal_district_window_split(epi_df, '{district}')", 30
            )

        key_err = _check_dict_keys("ch31_exam_01", "Exam Task 31.1", got, expected_keys, 30)
        if key_err is not None:
            return key_err

        if not df_copy.equals(src_df):
            errors.append(f"[{desc}] Mutated caller `epi_df` in place!")

        exp = _ref_temporal_split(src_df, district, **kwargs)
        scaler = got["scaler"]
        if not isinstance(scaler, StandardScaler) or not hasattr(scaler, "mean_"):
            errors.append(f"[{desc}] `scaler` must be a fitted `StandardScaler` instance.")
        else:
            if not np.allclose(scaler.mean_, exp["scaler"].mean_, atol=1e-3):
                # Check if student accidentally fit on the entire district series
                sub_all = src_df[src_df["district"] == district][
                    ["dengue_cases", "rainfall_mm", "mean_temp_c"]
                ].to_numpy(dtype=float)
                leaky_mean = sub_all.mean(axis=0)
                if np.allclose(scaler.mean_, leaky_mean, atol=1e-2):
                    errors.append(
                        f"[{desc}] Temporal leakage detected! `scaler` was fit on ALL years instead of `year < cutoff_year` ONLY."
                    )
                else:
                    errors.append(
                        f"[{desc}] `scaler.mean_` mismatch: expected {np.round(exp['scaler'].mean_, 3)}, "
                        f"got {np.round(scaler.mean_, 3)}. Check feature column order "
                        "`['dengue_cases', 'rainfall_mm', 'mean_temp_c']`."
                    )

        for k in ("X_train", "y_train", "X_test", "y_test"):
            arr = np.asarray(got[k], dtype=float)
            earr = np.asarray(exp[k], dtype=float)
            if arr.shape != earr.shape:
                errors.append(f"[{desc}] `{k}` shape mismatch: expected {earr.shape}, got {arr.shape}.")
            elif not np.allclose(arr, earr, atol=1e-4):
                errors.append(
                    f"[{desc}] `{k}` values mismatch — ensure sorting by `['year', 'epi_week']`, "
                    "fitting scaler on `year < cutoff_year` only, and building windows on the scaled series."
                )

    if errors:
        return CheckResult(
            "ch31_exam_01",
            False,
            "failed",
            "Exam Task 31.1 — Check Temporal Split, Scaler Fit & Window Alignment",
            "Sort by `['year', 'epi_week']`, fit `StandardScaler` ONLY on `year < cutoff_year` rows for `['dengue_cases', 'rainfall_mm', 'mean_temp_c']`, and build standardized train (`year < cutoff_year`) and test (`year >= cutoff_year`) sliding windows.",
            details=errors,
            points_earned=0,
            points_possible=30,
        )

    return CheckResult(
        "ch31_exam_01",
        True,
        "passed",
        "Exam Task 31.1 Passed (30/30 pts) — Leak-Free Temporal Window Split Verified!",
        "Passed chronological sorting (including shuffled inputs), training-only `StandardScaler` fit, exact `(N, seq_len, 3)` window shapes, and standardized target alignment.",
        points_earned=30,
        points_possible=30,
        tip="Fitting `StandardScaler` strictly on `year < cutoff_year` prevents future monsoon peaks from leaking into 2024 training statistics.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_exam_01",
        chapter="ch31",
        title="Exam 31.1: Leak-Free Temporal District Window Split (temporal_district_window_split)",
        book_section="Chapter 31 Coding Exam — Task 1",
        points=30,
        is_exam=True,
        hints={
            1: "Filter `epi_df[epi_df['district'] == district].sort_values(['year', 'epi_week']).reset_index(drop=True)` and extract `['dengue_cases', 'rainfall_mm', 'mean_temp_c']`.",
            2: "Find `n_train = int((sub['year'] < cutoff_year).sum())`, fit `scaler = StandardScaler().fit(raw[:n_train])`, and transform the full sorted array `scaled = scaler.transform(raw)`.",
            3: "X_train = np.stack([scaled[t - seq_len : t] for t in range(seq_len, n_train)])\ny_train = scaled[seq_len:n_train, 0]\nX_test = np.stack([scaled[t - seq_len : t] for t in range(n_train, len(sub))])\ny_test = scaled[n_train:, 0]\nreturn {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'scaler': scaler}",
        },
        verifier=_verify_exam_01,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 31.2 (35 pts): train_epi_gru_forecaster(...)
# ---------------------------------------------------------------------------
def _verify_exam_02(fn: Any) -> CheckResult:
    import numpy as np
    import torch
    from ..datasets import load_data

    bad = _check_callable(
        "ch31_exam_02", "Exam Task 31.2", fn, "train_epi_gru_forecaster", 35
    )
    if bad is not None:
        return bad

    torch.set_num_threads(1)
    df_epi = load_data("epi_surveillance", as_pandas=True)
    dhaka_split = _ref_temporal_split(df_epi, "Dhaka", seq_len=8, cutoff_year=2025)
    ctg_split = _ref_temporal_split(df_epi, "Chattogram", seq_len=8, cutoff_year=2025)
    expected_keys = ["test_rmse", "test_mae", "final_train_loss", "preds"]
    errors = []

    cases = [
        (dhaka_split, {"hidden_size": 16, "epochs": 40, "lr": 0.02, "seed": 42}, "Dhaka (epochs=40, seed=42)"),
        (ctg_split, {"hidden_size": 12, "epochs": 30, "lr": 0.02, "seed": 7}, "Chattogram (hidden=12, epochs=30, seed=7)"),
    ]

    for split, kwargs, desc in cases:
        X_tr, y_tr, X_te, y_te = (
            split["X_train"].copy(),
            split["y_train"].copy(),
            split["X_test"].copy(),
            split["y_test"].copy(),
        )
        try:
            got = fn(X_tr, y_tr, X_te, y_te, **kwargs)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch31_exam_02",
                False,
                "failed",
                "Exam Task 31.2 — Check Signature `train_epi_gru_forecaster(X_train, y_train, X_test, y_test, hidden_size=16, epochs=40, lr=0.02, seed=42)`",
                f"Failed on {desc} with TypeError: {exc}",
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch31_exam_02",
                False,
                "failed",
                "Exam Task 31.2 — Runtime Error in `train_epi_gru_forecaster`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch31_exam_02", "Exam Task 31.2", "train_epi_gru_forecaster(X_train, y_train, X_test, y_test)", 35
            )

        key_err = _check_dict_keys("ch31_exam_02", "Exam Task 31.2", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        preds = got["preds"]
        if not isinstance(preds, np.ndarray) or preds.ndim != 1 or preds.shape[0] != len(y_te):
            errors.append(
                f"[{desc}] `preds` must be a 1D NumPy array of shape ({len(y_te)},), "
                f"got {type(preds).__name__} with shape {getattr(preds, 'shape', None)}."
            )
            continue
        if not np.all(np.isfinite(preds)):
            errors.append(f"[{desc}] `preds` contains NaN or Inf values.")
            continue

        calc_rmse = round(float(np.sqrt(np.mean((y_te - preds) ** 2))), 4)
        calc_mae = round(float(np.mean(np.abs(y_te - preds))), 4)

        if not _is_np_or_py_number(got["test_rmse"]) or not math.isclose(
            float(got["test_rmse"]), calc_rmse, abs_tol=1e-4
        ):
            errors.append(
                f"[{desc}] `test_rmse` ({got['test_rmse']!r}) does not match RMSE computed from `preds` vs `y_test` ({calc_rmse})."
            )
        if not _is_np_or_py_number(got["test_mae"]) or not math.isclose(
            float(got["test_mae"]), calc_mae, abs_tol=1e-4
        ):
            errors.append(
                f"[{desc}] `test_mae` ({got['test_mae']!r}) does not match MAE computed from `preds` vs `y_test` ({calc_mae})."
            )
        if not _is_np_or_py_number(got["final_train_loss"]) or not math.isfinite(
            float(got["final_train_loss"])
        ):
            errors.append(f"[{desc}] `final_train_loss` must be a finite float, got {got['final_train_loss']!r}.")

        ref = _ref_train_rnn(X_tr, y_tr, X_te, y_te, cell_type="gru", **kwargs)
        if float(got["test_rmse"]) > ref["test_rmse"] + 0.08:
            errors.append(
                f"[{desc}] `test_rmse` ({got['test_rmse']}) is far worse than reference GRU ({ref['test_rmse']}). "
                "Check `MSELoss`, `Adam(lr=lr)`, `clip_grad_norm_(..., 1.0)`, `optimizer.zero_grad()`, and `model.eval()`."
            )
        if float(got["final_train_loss"]) > 0.25:
            errors.append(
                f"[{desc}] `final_train_loss` ({got['final_train_loss']}) did not converge (reference: {ref['final_train_loss']})."
            )

    if errors:
        return CheckResult(
            "ch31_exam_02",
            False,
            "failed",
            "Exam Task 31.2 — Check GRU Training Loop, Gradient Clipping & Metrics",
            "Seed with `torch.manual_seed(seed)`, train `nn.GRU(..., batch_first=True)` + `nn.Linear(hidden_size, 1)` with `MSELoss`, `Adam`, and `clip_grad_norm_(..., 1.0)`, and evaluate in `eval()`/`no_grad()`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch31_exam_02",
        True,
        "passed",
        "Exam Task 31.2 Passed (35/35 pts) — Epidemiological GRU Forecaster Verified!",
        "Trained stably across Dhaka and Chattogram with gradient clipping, produced 1D test predictions, and beat the naive persistence baseline cleanly.",
        points_earned=35,
        points_possible=35,
        tip="Gradient clipping (`clip_grad_norm_(model.parameters(), 1.0)`) is a 1-line insurance policy against exploding recurrent gradients.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_exam_02",
        chapter="ch31",
        title="Exam 31.2: Train Epidemiological GRU Forecaster (train_epi_gru_forecaster)",
        book_section="Chapter 31 Coding Exam — Task 2",
        points=35,
        is_exam=True,
        hints={
            1: "Call `torch.manual_seed(seed)`, build a GRU forecaster (`nn.GRU(X_train.shape[-1], hidden_size, batch_first=True)` + `nn.Linear(hidden_size, 1)`), and convert `X_train`, `y_train` (`unsqueeze(-1)`), and `X_test` to `torch.float32` tensors.",
            2: "Loop `epochs` times: `optimizer.zero_grad()`, `loss = loss_fn(model(xt), yt)`, `loss.backward()`, `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, `optimizer.step()`.",
            3: "Switch to `model.eval()` and `with torch.no_grad(): preds = model(xe).squeeze(-1).numpy()`. Return `{'test_rmse': round(..., 4), 'test_mae': round(..., 4), 'final_train_loss': round(float(loss.item()), 5), 'preds': preds}`.",
        },
        verifier=_verify_exam_02,
    )
)


# ---------------------------------------------------------------------------
# EXAM TASK 31.3 (35 pts): compare_gru_lstm_persistence_epi(...)
# ---------------------------------------------------------------------------
def _verify_exam_03(fn: Any) -> CheckResult:
    import numpy as np
    import pandas as pd
    import torch
    from ..datasets import load_data

    bad = _check_callable(
        "ch31_exam_03", "Exam Task 31.3", fn, "compare_gru_lstm_persistence_epi", 35
    )
    if bad is not None:
        return bad

    torch.set_num_threads(1)
    df_epi = load_data("epi_surveillance", as_pandas=True)
    expected_keys = ["leaderboard", "best_model"]
    expected_cols = ["model", "test_rmse", "test_mae"]
    errors = []

    for district in ("Dhaka", "Chattogram"):
        desc = f"{district} (seq_len=8, seed=42)"
        df_copy = df_epi.copy(deep=True)
        try:
            got = fn(df_copy, district=district, seq_len=8, seed=42)
        except NotImplementedError:
            raise
        except TypeError as exc:
            return CheckResult(
                "ch31_exam_03",
                False,
                "failed",
                "Exam Task 31.3 — Check Signature `compare_gru_lstm_persistence_epi(epi_df, district='Dhaka', seq_len=8, seed=42)`",
                f"Failed on {desc} with TypeError: {exc}",
                points_earned=0,
                points_possible=35,
            )
        except Exception as exc:
            return CheckResult(
                "ch31_exam_03",
                False,
                "failed",
                "Exam Task 31.3 — Runtime Error in `compare_gru_lstm_persistence_epi`",
                f"Failed on {desc} with {type(exc).__name__}: {exc}",
                points_earned=0,
                points_possible=35,
            )

        if got is None:
            return _none_return_trap(
                "ch31_exam_03", "Exam Task 31.3", f"compare_gru_lstm_persistence_epi(epi_df, district='{district}')", 35
            )

        key_err = _check_dict_keys("ch31_exam_03", "Exam Task 31.3", got, expected_keys, 35)
        if key_err is not None:
            return key_err

        if not df_copy.equals(df_epi):
            errors.append(f"[{desc}] Mutated caller `epi_df` in place!")

        lb = got["leaderboard"]
        if not isinstance(lb, pd.DataFrame):
            errors.append(f"[{desc}] `leaderboard` must be a `pd.DataFrame`, got {type(lb).__name__}.")
            continue
        if list(lb.columns) != expected_cols:
            errors.append(f"[{desc}] `leaderboard` columns mismatch: expected {expected_cols}, got {list(lb.columns)}.")
            continue
        if len(lb) != 3:
            errors.append(f"[{desc}] `leaderboard` must have 3 rows (Persistence, GRU, LSTM), got {len(lb)}.")
            continue

        lb_reset = lb.reset_index(drop=True)
        rmses = [float(v) for v in lb_reset["test_rmse"].tolist()]
        if rmses != sorted(rmses):
            errors.append(f"[{desc}] `leaderboard` must be sorted ascending by `test_rmse`, got {rmses}.")

        best_model = got["best_model"]
        if not isinstance(best_model, str) or best_model != str(lb_reset.loc[0, "model"]):
            errors.append(
                f"[{desc}] `best_model` ({best_model!r}) must equal the top-ranked row `leaderboard.iloc[0]['model']` ({lb_reset.loc[0, 'model']!r})."
            )

        models_lower = {str(m).strip().lower(): i for i, m in enumerate(lb_reset["model"])}
        required_tags = ["persistence", "gru", "lstm"]
        matched = {}
        for tag in required_tags:
            found = [idx for name, idx in models_lower.items() if tag in name]
            if not found:
                errors.append(f"[{desc}] Missing model containing '{tag}' in `leaderboard['model']`: {lb_reset['model'].tolist()}.")
            else:
                matched[tag] = found[0]

        if len(matched) == 3:
            split = _ref_temporal_split(df_epi, district, seq_len=8, cutoff_year=2025)
            y_te = split["y_test"].astype(float)
            pers_pred = split["X_test"][:, -1, 0].astype(float)
            exp_pers_rmse = round(float(np.sqrt(np.mean((y_te - pers_pred) ** 2))), 4)
            exp_pers_mae = round(float(np.mean(np.abs(y_te - pers_pred))), 4)

            p_row = lb_reset.iloc[matched["persistence"]]
            if not math.isclose(float(p_row["test_rmse"]), exp_pers_rmse, abs_tol=1e-3):
                errors.append(
                    f"[{desc}] Persistence `test_rmse` mismatch: expected {exp_pers_rmse}, got {p_row['test_rmse']!r}."
                )
            if not math.isclose(float(p_row["test_mae"]), exp_pers_mae, abs_tol=1e-3):
                errors.append(
                    f"[{desc}] Persistence `test_mae` mismatch: expected {exp_pers_mae}, got {p_row['test_mae']!r}."
                )

            ref_gru = _ref_train_rnn(split["X_train"], split["y_train"], split["X_test"], split["y_test"], cell_type="gru", seed=42)
            ref_lstm = _ref_train_rnn(split["X_train"], split["y_train"], split["X_test"], split["y_test"], cell_type="lstm", seed=42)

            g_rmse = float(lb_reset.iloc[matched["gru"]]["test_rmse"])
            l_rmse = float(lb_reset.iloc[matched["lstm"]]["test_rmse"])
            if g_rmse > ref_gru["test_rmse"] + 0.08:
                errors.append(
                    f"[{desc}] GRU `test_rmse` ({g_rmse}) is unexpectedly high (reference: {ref_gru['test_rmse']})."
                )
            if l_rmse > ref_lstm["test_rmse"] + 0.08:
                errors.append(
                    f"[{desc}] LSTM `test_rmse` ({l_rmse}) is unexpectedly high (reference: {ref_lstm['test_rmse']})."
                )

    if errors:
        return CheckResult(
            "ch31_exam_03",
            False,
            "failed",
            "Exam Task 31.3 — Check Persistence vs. GRU vs. LSTM Benchmark Leaderboard",
            "Run `temporal_district_window_split`, evaluate Persistence (`X_test[:, -1, 0]`), GRU, and LSTM on the exact same held-out test weeks, sort by `test_rmse` ascending, and return `{'leaderboard': df, 'best_model': str}`.",
            details=errors,
            points_earned=0,
            points_possible=35,
        )

    return CheckResult(
        "ch31_exam_03",
        True,
        "passed",
        "Exam Task 31.3 Passed (35/35 pts) — Epidemiological Sequence Benchmark Verified!",
        "Compared Persistence, GRU, and LSTM on identical leak-free held-out test weeks across Dhaka and Chattogram with clean leaderboard sorting.",
        points_earned=35,
        points_possible=35,
        tip="Evaluating Persistence, GRU, and LSTM on the exact same held-out chronological windows guarantees an apples-to-apples benchmark ready for a publication table.",
    )


register_exercise(
    ExerciseSpec(
        exercise_id="ch31_exam_03",
        chapter="ch31",
        title="Exam 31.3: Persistence vs. GRU vs. LSTM Benchmark (compare_gru_lstm_persistence_epi)",
        book_section="Chapter 31 Coding Exam — Task 3",
        points=35,
        is_exam=True,
        hints={
            1: "Call `temporal_district_window_split(epi_df, district=district, seq_len=seq_len, cutoff_year=2025)` to obtain `(X_train, y_train, X_test, y_test)`.",
            2: "Compute Persistence predictions as `X_test[:, -1, 0]`, then train GRU and LSTM forecasters (`hidden_size=16, epochs=40, lr=0.02, seed=seed`) on `(X_train, y_train)` and score all three on `y_test`.",
            3: "Build `leaderboard = pd.DataFrame(rows, columns=['model', 'test_rmse', 'test_mae']).sort_values('test_rmse').reset_index(drop=True)` and return `{'leaderboard': leaderboard, 'best_model': str(leaderboard.loc[0, 'model'])}`.",
        },
        verifier=_verify_exam_03,
    )
)
